"""Training entrypoint for ZEBRA-Prop with Hydra + PyTorch Lightning."""

import logging
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/zebra_prop_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/zebra_prop_cache")

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .data import EmbeddingDataset, MultiDescriptionDataset, collate_fn, emb_collate
from .model import EmbeddingExtractor, ZEBRAProp
from .utils import (
    get_desc_df,
    get_llm,
    get_prop_data,
    load_description_csvs,
    parity_plot,
    precompute_embeddings,
    split_dataset_kfold,
    standardize_tensor,
)

_DEFAULT_CONFIG_DIR = Path.cwd() / "config"
if not (_DEFAULT_CONFIG_DIR / "config.yaml").is_file():
    # Fallback for package execution contexts where CWD is not repository root.
    _DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
HYDRA_CONFIG_PATH = str(_DEFAULT_CONFIG_DIR.resolve())

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="The '.*_dataloader' does not have many workers.*",
)

DATA_FORMAT_DOC_PATH = "docs/data_format.md"
EMBEDDING_NUM_WORKERS = 2


@dataclass(frozen=True)
class TrainingConfig:
    """Typed runtime config values used by the training pipeline."""

    data_dir: Path
    output_dir: Path
    property_name: str
    task_name: str
    hf_model_name: str
    max_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    k_fold: int
    fold: int
    seed: int
    save_prediction_values: bool
    save_parity_plot: bool


def _normalize_ids(id_list):
    """Convert numpy scalar ids to native Python scalars."""
    if id_list is None:
        return None

    normalized = []
    for value in id_list:
        if isinstance(value, np.generic):
            normalized.append(value.item())
        else:
            normalized.append(value)
    return normalized


def _progress(step, total, message):
    """Log a concise CUI progress line."""
    logger.info("[%d/%d] %s", step, total, message)


def _as_bool(value, key_name: str) -> bool:
    """Convert config values into bool with strict string handling."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"`{key_name}` must be a boolean value, got: {value!r}")


def _resolve_embedding_device() -> torch.device:
    """Pick the fastest available backend for embedding inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_embedding_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    """Build a dataloader for embedding precomputation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _dataset_validation_error_message(summary: str, hint: str) -> str:
    """Build a user-friendly dataset validation error message."""
    return (
        f"{summary}\n"
        f"Hint: {hint}\n"
        f"Documentation: {DATA_FORMAT_DOC_PATH}"
    )


def validate_training_files(
    data_dir: Path,
    property_name: str,
    task_name: str,
):
    """Validate all required input files before starting training."""
    id_prop_dir = data_dir / "id_prop"
    description_base_dir = data_dir / "description"
    embedding_cache_dir = data_dir / "embedding"

    id_prop_path = id_prop_dir / f"id_prop_{property_name}.csv"
    if not id_prop_path.is_file():
        raise FileNotFoundError(
            _dataset_validation_error_message(
                summary=f"Missing target CSV: {id_prop_path}",
                hint=(
                    f"Create `{id_prop_path}` with two columns (id,target), "
                    "no header."
                ),
            )
        )

    try:
        id_prop_preview = pd.read_csv(id_prop_path, header=None, nrows=5)
    except Exception as exc:
        raise ValueError(
            _dataset_validation_error_message(
                summary=f"Failed to parse target CSV: {id_prop_path}",
                hint=(
                    "Ensure this file is valid CSV text with no header and at "
                    "least two columns."
                ),
            )
        ) from exc

    if id_prop_preview.empty:
        raise ValueError(
            _dataset_validation_error_message(
                summary=f"Target CSV is empty: {id_prop_path}",
                hint="Add at least one row with `id,target` values.",
            )
        )

    if id_prop_preview.shape[1] < 2:
        raise ValueError(
            _dataset_validation_error_message(
                summary=f"Target CSV must have at least 2 columns: {id_prop_path}",
                hint=(
                    "Column 1 is material id, column 2 is target value. "
                    "Remove headers if present."
                ),
            )
        )

    task_dir = description_base_dir / task_name
    if not task_dir.is_dir():
        raise FileNotFoundError(
            _dataset_validation_error_message(
                summary=f"Missing description directory: {task_dir}",
                hint=(
                    f"Set `task_name={task_name}` correctly and place description "
                    "CSV files under this directory."
                ),
            )
        )

    csv_paths = sorted(p for p in task_dir.glob("*.csv") if p.name != "all.csv")
    if not csv_paths:
        raise FileNotFoundError(
            _dataset_validation_error_message(
                summary=f"No description CSV files found in {task_dir}",
                hint=(
                    "Add one or more `*.csv` files containing `id,formula,"
                    "description` columns."
                ),
            )
        )

    required_cols = {"id", "formula", "description"}
    for csv_path in csv_paths:
        try:
            # Read only the header to validate required columns with minimal overhead.
            header = pd.read_csv(csv_path, nrows=0)
        except Exception as exc:
            raise ValueError(
                _dataset_validation_error_message(
                    summary=f"Failed to parse description CSV: {csv_path}",
                    hint=(
                        "Ensure the file is UTF-8 compatible CSV and contains "
                        "`id,formula,description` columns."
                    ),
                )
            ) from exc
        missing = required_cols - set(header.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            found_cols = ", ".join(map(str, header.columns.tolist()))
            raise ValueError(
                _dataset_validation_error_message(
                    summary=f"{csv_path} is missing columns: {missing_list}",
                    hint=(
                        f"Found columns: {found_cols or '(none)'}. "
                        "Add the missing required columns."
                    ),
                )
            )

    embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    return {
        "id_prop_dir": id_prop_dir,
        "description_base_dir": description_base_dir,
        "embedding_cache_dir": embedding_cache_dir,
    }


class ZEBRAPropLightningModule(pl.LightningModule):
    """Lightning module for regression from precomputed text embeddings."""

    def __init__(
        self,
        num_descriptions: int,
        hidden_size: int = 768,
        learning_rate: float = 1e-3,
        scaler: StandardScaler | None = None,
        desc_cols: list[str] | None = None,
    ):
        """Initialize the regression module and metric buffers."""
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])

        self.learning_rate = learning_rate
        self.scaler = scaler
        self.desc_cols = desc_cols
        self.hidden_size = hidden_size

        self.model = ZEBRAProp(
            hidden_size=self.hidden_size,
            num_descriptions=num_descriptions,
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, inputs):
        """Return model predictions for one embedding batch."""
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        return self.model(inputs)

    def training_step(self, batch, _batch_idx):
        """Run one training step and log MAE loss."""
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = F.l1_loss(predictions.squeeze(), targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, _batch_idx):
        """Run one validation step and store outputs for epoch metrics."""
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = F.l1_loss(predictions.squeeze(), targets)

        self.validation_step_outputs.append(
            {"preds": predictions, "targets": targets}
        )
        return loss

    def test_step(self, batch, _batch_idx):
        """Run one test step and store outputs for epoch metrics."""
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = F.l1_loss(predictions.squeeze(), targets)

        self.test_step_outputs.append(
            {"preds": predictions, "targets": targets}
        )
        return loss

    def on_validation_epoch_end(self):
        """Aggregate validation predictions and log MAE/RMSE/R2."""
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        if self.scaler is not None:
            # Convert metrics back to original target scale for interpretability.
            all_targets_np = self.scaler.inverse_transform(
                all_targets.cpu().numpy().reshape(-1, 1)
            ).flatten()
            all_preds_np = self.scaler.inverse_transform(
                all_preds.squeeze().cpu().numpy().reshape(-1, 1)
            ).flatten()
        else:
            all_targets_np = all_targets.cpu().numpy()
            all_preds_np = all_preds.squeeze().cpu().numpy()

        val_mae = np.mean(np.abs(all_targets_np - all_preds_np))
        val_rmse = np.sqrt(np.mean((all_targets_np - all_preds_np) ** 2))
        val_r2 = r2_score(all_targets_np, all_preds_np)

        self.log("val_mae", val_mae, prog_bar=False)
        self.log("val_rmse", val_rmse)
        self.log("val_r2", val_r2)

        if hasattr(self.model, "W_desc") and self.desc_cols is not None:
            weights_dict = {
                f"weight_{col}": float(w)
                for col, w in zip(
                    self.desc_cols,
                    self.model.W_desc.detach().cpu().numpy().flatten(),
                )
            }
            self.log_dict(weights_dict)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Aggregate test predictions and log MAE/RMSE/R2."""
        if not self.test_step_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])

        if self.scaler is not None:
            all_targets_np = self.scaler.inverse_transform(
                all_targets.cpu().numpy().reshape(-1, 1)
            ).flatten()
            all_preds_np = self.scaler.inverse_transform(
                all_preds.cpu().numpy().reshape(-1, 1)
            ).flatten()
        else:
            all_targets_np = all_targets.cpu().numpy()
            all_preds_np = all_preds.cpu().numpy()

        test_mae = np.mean(np.abs(all_targets_np - all_preds_np))
        test_rmse = np.sqrt(np.mean((all_targets_np - all_preds_np) ** 2))
        test_r2 = r2_score(all_targets_np, all_preds_np)

        self.log("test_mae", test_mae)
        self.log("test_rmse", test_rmse)
        self.log("test_r2", test_r2)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Use a fixed AdamW + StepLR scheduler setup."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class ZEBRAPropDataModule(pl.LightningDataModule):
    """Lightning data module for fixed k-fold train/valid/test splitting."""

    def __init__(
        self,
        desc_df: pd.DataFrame,
        label_col: str,
        batch_size: int = 32,
        k_fold: int = 5,
        test_fold: int = 0,
        random_state: int = 42,
        embeddings: torch.Tensor | None = None,
    ):
        """Initialize split parameters and embedding tensors."""
        super().__init__()
        self.desc_df = desc_df
        self.label_col = label_col
        self.batch_size = batch_size
        self.k_fold = k_fold
        self.test_fold = test_fold
        self.random_state = random_state
        self.embeddings = embeddings

        self.scaler = StandardScaler()
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def setup(self, _stage: str | None = None):
        """Create standardized train/valid/test embedding datasets."""
        idx = list(range(len(self.desc_df)))
        train_idx, val_idx, test_idx = split_dataset_kfold(
            idx,
            k_fold=self.k_fold,
            test_fold=self.test_fold,
            random_seed=self.random_state,
        )
        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx

        train_y = self.desc_df.iloc[train_idx][self.label_col].values
        val_y = self.desc_df.iloc[val_idx][self.label_col].values
        test_y = self.desc_df.iloc[test_idx][self.label_col].values

        train_y = torch.tensor(self.scaler.fit_transform(train_y.reshape(-1, 1))).squeeze()
        val_y = torch.tensor(self.scaler.transform(val_y.reshape(-1, 1))).squeeze()
        test_y = torch.tensor(self.scaler.transform(test_y.reshape(-1, 1))).squeeze()

        train_emb = self.embeddings[train_idx]
        val_emb = self.embeddings[val_idx]
        test_emb = self.embeddings[test_idx]

        # Standardize each description stream independently using train statistics.
        for i in range(len(train_emb[0])):
            train_emb[:, i], val_emb[:, i], test_emb[:, i] = standardize_tensor(
                train_emb[:, i], val_emb[:, i], test_emb[:, i]
            )

        self.train_dataset = EmbeddingDataset(train_emb, train_y)
        self.val_dataset = EmbeddingDataset(val_emb, val_y)
        self.test_dataset = EmbeddingDataset(test_emb, test_y)

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=emb_collate,
            num_workers=0,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=emb_collate,
            num_workers=0,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=emb_collate,
            num_workers=0,
        )


def load_cached_embeddings(cache_path: str, desc_cols: list):
    """Load cached embedding tensor and id order metadata."""
    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    emb_dict = data.get("embeddings", data)
    missing = [c for c in desc_cols if c not in emb_dict]
    if missing:
        raise KeyError(f"Missing embeddings for columns: {', '.join(missing)}")

    cached_ids = _normalize_ids(data.get("ids"))
    emb_list = [emb_dict[c] for c in desc_cols]
    return torch.stack(emb_list, dim=1), cached_ids


def save_embeddings_cache(
    cache_path: str,
    embeddings: torch.Tensor,
    desc_cols: list,
    ids: list,
):
    """Save embeddings split by description column with id metadata."""
    emb_dict = {col: embeddings[:, i, :].cpu() for i, col in enumerate(desc_cols)}
    payload = {
        "embeddings": emb_dict,
        "desc_cols": desc_cols,
        "ids": _normalize_ids(ids),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


def align_embeddings_with_cache(
    embeddings: torch.Tensor,
    cached_ids: list,
    target_ids: list,
    logger_instance: logging.Logger | None = None,
):
    """Reorder cached embeddings to match target id ordering."""
    cached_ids = _normalize_ids(cached_ids)
    target_ids = _normalize_ids(target_ids)

    if not cached_ids:
        return embeddings, target_ids, []

    cached_id_to_pos = {cid: idx for idx, cid in enumerate(cached_ids)}
    target_id_set = set(target_ids)
    ordered_ids = [cid for cid in cached_ids if cid in target_id_set]
    missing_ids = sorted(target_id_set - set(ordered_ids))

    if not ordered_ids:
        raise ValueError("No overlapping ids between cached embeddings and property labels.")

    index_tensor = torch.tensor(
        [cached_id_to_pos[cid] for cid in ordered_ids],
        dtype=torch.long,
        device=embeddings.device,
    )
    aligned_embeddings = embeddings.index_select(0, index_tensor)

    if missing_ids and logger_instance is not None:
        logger_instance.warning(
            "Dropped %d samples absent from cached embeddings: %s",
            len(missing_ids),
            missing_ids[:10],
        )

    return aligned_embeddings, ordered_ids, missing_ids


def export_predictions(
    output_path: Path,
    preds_array: np.ndarray,
    targets_array: np.ndarray,
    source_df: pd.DataFrame,
    indices,
):
    """Export split-level predictions with optional id/formula columns."""
    idx_list = list(indices) if indices is not None else list(range(len(preds_array)))

    df_dict = {"actual": targets_array, "prediction": preds_array}
    if "id" in source_df.columns:
        df_dict["id"] = source_df.iloc[idx_list]["id"].values
    if "formula" in source_df.columns:
        df_dict["formula"] = source_df.iloc[idx_list]["formula"].values

    df = pd.DataFrame(df_dict)
    ordered_columns = [c for c in ("id", "formula", "actual", "prediction") if c in df]
    df = df[ordered_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _predict_from_loader(model: ZEBRAPropLightningModule, loader: DataLoader):
    """Return concatenated predictions and targets for one dataloader."""
    preds, targets = [], []
    for inputs, batch_targets in loader:
        preds.append(model(inputs).cpu())
        targets.append(batch_targets.cpu())

    all_preds = torch.cat(preds, dim=0).squeeze().numpy()
    all_targets = torch.cat(targets, dim=0).numpy()
    return all_preds, all_targets


def _parse_training_config(cfg: DictConfig) -> TrainingConfig:
    """Read and normalize config values used by `main_lightning`."""
    return TrainingConfig(
        data_dir=Path(str(cfg.get("data_dir", "./data"))),
        output_dir=Path(str(cfg.get("output_dir", "./output"))),
        property_name=str(cfg.get("property_name", "band_gap")),
        task_name=str(cfg.get("task_name", "human-made")),
        hf_model_name=str(cfg.get("hf_model_name")),
        max_length=int(cfg.get("max_length", 512)),
        epochs=int(cfg.get("epochs", 200)),
        batch_size=int(cfg.get("bs", 64)),
        learning_rate=float(cfg.get("lr", 1e-3)),
        k_fold=int(cfg.get("num_of_fold", 5)),
        fold=int(cfg.get("test_fold", 0)),
        seed=int(cfg.get("seed", 42)),
        save_prediction_values=_as_bool(
            cfg.get("save_prediction_values", True),
            "save_prediction_values",
        ),
        save_parity_plot=_as_bool(
            cfg.get("save_parity_plot", True),
            "save_parity_plot",
        ),
    )


def _prepare_model_and_tokenizer(hf_model_name: str, max_length: int):
    """Load tokenizer settings and resolve model path/usable sequence length."""
    if not hf_model_name:
        raise ValueError("`hf_model_name` is required in config/config.yaml")

    model_name = hf_model_name.split("/")[-1]
    model_path, resolved_max_length = get_llm(hf_model_name, max_length=max_length)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    # Avoid exceeding model capacity even when config max_length is larger.
    if isinstance(tokenizer_max_length, int) and tokenizer_max_length <= 100000:
        resolved_max_length = min(resolved_max_length, int(tokenizer_max_length))

    return model_name, model_path, tokenizer, resolved_max_length


def _load_property_and_description_data(
    property_name: str,
    task_name: str,
    id_prop_dir: Path,
    description_base_dir: Path,
):
    """Load target values and aligned text descriptions."""
    index, y_values = get_prop_data(property_name, id_prop_dir=str(id_prop_dir))
    prop_series = pd.Series(y_values, index=index)

    desc_df = load_description_csvs(
        description_name=task_name,
        base_dir=str(description_base_dir),
    )

    # Align description rows to the exact id ordering from target data.
    desc_df = desc_df[desc_df["id"].isin(index)]
    desc_df = desc_df.set_index("id").reindex(index).reset_index()

    desc_df, desc_cols = get_desc_df(desc_df)
    desc_df[property_name] = prop_series.reindex(desc_df["id"]).values
    return prop_series, desc_df, desc_cols


def _load_or_compute_embeddings(
    emb_cache_path: Path,
    desc_df: pd.DataFrame,
    desc_cols: list[str],
    tokenizer,
    model_path: str,
    property_name: str,
    max_length: int,
    batch_size: int,
):
    """Load embeddings from cache, or compute/cache them when unavailable."""
    embedding_source_ids = desc_df["id"].tolist()
    embeddings = None
    cached_ids = None

    if emb_cache_path.exists():
        try:
            embeddings, cached_ids = load_cached_embeddings(str(emb_cache_path), desc_cols)
            if cached_ids is None:
                raise ValueError("Cached embeddings have no id order metadata")
            logger.info("Loaded embedding cache: %s", emb_cache_path)
            logger.info("Skipping embedding computation because cache is available.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load cache (%s). Recomputing embeddings.", exc)
            embeddings = None
            cached_ids = None

    if embeddings is not None:
        return embeddings, cached_ids

    # Compute embeddings once and cache them for repeatable/faster reruns.
    base_model = AutoModel.from_pretrained(model_path)
    base_model.resize_token_embeddings(len(tokenizer))

    dataset = MultiDescriptionDataset(
        desc_df,
        tokenizer,
        desc_cols,
        property_name,
        max_length,
    )
    loader = _build_embedding_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=EMBEDDING_NUM_WORKERS,
    )
    extractor = EmbeddingExtractor(base_model=base_model)
    device = _resolve_embedding_device()
    logger.info(
        "Computing embeddings from scratch | samples=%d, desc_cols=%d, batches=%d, batch_size=%d, workers=%d, device=%s",
        len(dataset),
        len(desc_cols),
        len(loader),
        batch_size,
        EMBEDDING_NUM_WORKERS,
        device,
    )
    logger.info("Description columns: %s", ", ".join(desc_cols))

    embeddings = precompute_embeddings(
        loader,
        extractor,
        device,
        out_path=None,
        dataset_size=len(desc_df),
        progress_desc="Embeddings",
        log_interval=5,
        logger_instance=logger,
    )
    save_embeddings_cache(
        str(emb_cache_path),
        embeddings,
        desc_cols,
        embedding_source_ids,
    )
    logger.info("Saved embedding cache: %s", emb_cache_path)
    return embeddings, cached_ids


def _align_data_with_embeddings(
    desc_df: pd.DataFrame,
    embeddings: torch.Tensor,
    cached_ids: list | None,
    prop_series: pd.Series,
    property_name: str,
):
    """Align dataframe row order with embedding tensor row order."""
    prop_ids = prop_series.index.tolist()
    if cached_ids is not None:
        cache_id_set = set(cached_ids)
        prop_missing_in_cache = sorted(set(prop_ids) - cache_id_set)
        if prop_missing_in_cache:
            raise ValueError(
                "Cached embeddings do not cover all id_prop samples. "
                f"Missing ids: {prop_missing_in_cache[:10]}"
            )
        embeddings, aligned_ids, _ = align_embeddings_with_cache(
            embeddings,
            cached_ids,
            prop_ids,
            logger_instance=logger,
        )
    else:
        aligned_ids = prop_ids

    desc_available_ids = set(desc_df["id"].tolist())
    missing_desc = [i for i in aligned_ids if i not in desc_available_ids]
    if missing_desc:
        raise ValueError(
            "Description CSVs are missing ids present in id_prop data: "
            f"{missing_desc[:10]}"
        )

    aligned_desc_df = desc_df.set_index("id").reindex(aligned_ids).reset_index()
    aligned_desc_df[property_name] = prop_series.reindex(aligned_ids).values
    return aligned_desc_df, embeddings


def _maybe_inverse_transform_arrays(
    scaler: StandardScaler | None,
    *arrays: np.ndarray,
):
    """Inverse-transform 1D arrays if scaler is available."""
    if scaler is None:
        return arrays
    return tuple(
        scaler.inverse_transform(array.reshape(-1, 1)).flatten() for array in arrays
    )


def _calculate_metrics(targets: np.ndarray, preds: np.ndarray):
    """Compute MAE/RMSE/R2 metrics for one split."""
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    return mae, rmse, r2


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name="config")
def main_lightning(cfg: DictConfig):
    """Run one training job with fixed default behavior for public usage."""
    total_steps = 8
    train_cfg = _parse_training_config(cfg)

    _progress(1, total_steps, "Validating required input files")
    required_paths = validate_training_files(
        train_cfg.data_dir,
        train_cfg.property_name,
        train_cfg.task_name,
    )

    _progress(2, total_steps, "Preparing model/tokenizer configuration")
    model_name, model_path, tokenizer, max_length = _prepare_model_and_tokenizer(
        train_cfg.hf_model_name,
        train_cfg.max_length,
    )

    id_prop_dir = required_paths["id_prop_dir"]
    description_base_dir = required_paths["description_base_dir"]
    embedding_cache_dir = required_paths["embedding_cache_dir"]

    model_embedding_cache_dir = embedding_cache_dir / model_name
    model_embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    _progress(3, total_steps, "Loading property and description data")
    prop_series, desc_df, desc_cols = _load_property_and_description_data(
        property_name=train_cfg.property_name,
        task_name=train_cfg.task_name,
        id_prop_dir=id_prop_dir,
        description_base_dir=description_base_dir,
    )

    _progress(4, total_steps, "Loading or computing text embeddings")
    emb_cache_path = model_embedding_cache_dir / train_cfg.task_name / "embeddings.pkl"
    embedding_start_time = time.time()

    embeddings, cached_ids = _load_or_compute_embeddings(
        emb_cache_path=emb_cache_path,
        desc_df=desc_df,
        desc_cols=desc_cols,
        tokenizer=tokenizer,
        model_path=model_path,
        property_name=train_cfg.property_name,
        max_length=max_length,
        batch_size=train_cfg.batch_size,
    )
    desc_df, embeddings = _align_data_with_embeddings(
        desc_df=desc_df,
        embeddings=embeddings,
        cached_ids=cached_ids,
        prop_series=prop_series,
        property_name=train_cfg.property_name,
    )

    embedding_time = time.time() - embedding_start_time
    hidden_size_value = embeddings.shape[-1]
    logger.info("Embedding step completed in %.2f sec", embedding_time)

    _progress(5, total_steps, "Preparing data module and trainer")
    random_state = train_cfg.seed + train_cfg.fold
    pl.seed_everything(random_state)

    data_module = ZEBRAPropDataModule(
        desc_df=desc_df,
        label_col=train_cfg.property_name,
        batch_size=train_cfg.batch_size,
        k_fold=train_cfg.k_fold,
        test_fold=train_cfg.fold,
        random_state=random_state,
        embeddings=embeddings,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    model = ZEBRAPropLightningModule(
        num_descriptions=len(desc_cols),
        hidden_size=hidden_size_value,
        learning_rate=train_cfg.learning_rate,
        scaler=data_module.scaler,
        desc_cols=desc_cols,
    )

    checkpoints_dir = train_cfg.output_dir / "checkpoints"
    prediction_dir = (
        train_cfg.output_dir
        / "predictions"
        / train_cfg.task_name
        / model_name
        / train_cfg.property_name
        / f"fold{train_cfg.fold}"
    )
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        filename=(
            f"{model_name}/{train_cfg.task_name}/"
            f"{train_cfg.property_name}/fold{train_cfg.fold}_best"
        ),
        dirpath=str(checkpoints_dir),
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg.epochs,
        logger=False,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=(1 if torch.cuda.device_count() == 1 else "auto"),
        strategy="auto",
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(train_cfg.output_dir),
    )

    _progress(6, total_steps, "Training and evaluating the model")
    training_start_time = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    training_time = time.time() - training_start_time

    best_model = ZEBRAPropLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        num_descriptions=len(desc_cols),
        hidden_size=hidden_size_value,
        learning_rate=train_cfg.learning_rate,
        scaler=data_module.scaler,
        desc_cols=desc_cols,
    )
    trainer.test(best_model, dataloaders=test_loader)

    best_model.eval()
    with torch.no_grad():
        # Recompute split predictions for CSV exports and parity plotting.
        train_preds, train_targets = _predict_from_loader(best_model, train_loader)
        val_preds, val_targets = _predict_from_loader(best_model, val_loader)
        test_preds, test_targets = _predict_from_loader(best_model, test_loader)

    # Save metrics and CSV outputs in original physical units when scaler exists.
    (
        train_targets,
        train_preds,
        val_targets,
        val_preds,
        test_targets,
        test_preds,
    ) = _maybe_inverse_transform_arrays(
        data_module.scaler,
        train_targets,
        train_preds,
        val_targets,
        val_preds,
        test_targets,
        test_preds,
    )

    train_mae, train_rmse, train_r2 = _calculate_metrics(train_targets, train_preds)
    val_mae, val_rmse, val_r2 = _calculate_metrics(val_targets, val_preds)
    test_mae, test_rmse, test_r2 = _calculate_metrics(test_targets, test_preds)

    _progress(7, total_steps, "Saving optional prediction artifacts")
    if train_cfg.save_prediction_values:
        split_exports = [
            (
                prediction_dir / "train_predictions.csv",
                train_preds,
                train_targets,
                data_module.train_indices,
            ),
            (
                prediction_dir / "valid_predictions.csv",
                val_preds,
                val_targets,
                data_module.val_indices,
            ),
            (
                prediction_dir / "test_predictions.csv",
                test_preds,
                test_targets,
                data_module.test_indices,
            ),
        ]

        for output_path, preds, targets, indices in split_exports:
            export_predictions(
                output_path,
                preds,
                targets,
                data_module.desc_df,
                indices,
            )
            logger.info("Saved prediction values: %s", output_path)
    else:
        logger.info("Skipped prediction values export (save_prediction_values=false)")

    if train_cfg.save_parity_plot:
        parity_plot_path = prediction_dir / "parity_plot.png"
        figure = parity_plot(
            test_data=(test_targets, test_preds),
            valid_data=(val_targets, val_preds),
            train_data=(train_targets, train_preds),
            test_score=(test_mae, test_rmse, test_r2),
            property=train_cfg.property_name,
        )
        figure.savefig(parity_plot_path, dpi=600, bbox_inches="tight")
        figure.clf()
        logger.info("Saved parity plot: %s", parity_plot_path)
    else:
        logger.info("Skipped parity plot export (save_parity_plot=false)")

    _progress(8, total_steps, "Done")
    logger.info("Training time: %.2f sec", training_time)
    logger.info("Best checkpoint: %s", checkpoint_callback.best_model_path)
    logger.info(
        "Metrics | train(MAE=%.4f, RMSE=%.4f, R2=%.4f) | "
        "valid(MAE=%.4f, RMSE=%.4f, R2=%.4f) | "
        "test(MAE=%.4f, RMSE=%.4f, R2=%.4f)",
        train_mae,
        train_rmse,
        train_r2,
        val_mae,
        val_rmse,
        val_r2,
        test_mae,
        test_rmse,
        test_r2,
    )


if __name__ == "__main__":
    main_lightning()


# Backward-compatible aliases for external imports.
MaterialPropertyLightningModule = ZEBRAPropLightningModule
MaterialPropertyDataModule = ZEBRAPropDataModule
