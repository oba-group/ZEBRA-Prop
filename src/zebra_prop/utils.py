"""Utility functions for data loading, preprocessing, and plotting."""

import numpy as np
import pandas as pd
import torch
import os
import logging
import time
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Split-specific plotting style for parity scatter points.
SPLIT_STYLES = {
    "train": {"color": "#1f77b4", "alpha": 0.25, "size": 16},
    "valid": {"color": "#ff9f1c", "alpha": 0.4, "size": 28},
    "test": {"color": "#2b8a3e", "alpha": 0.55, "size": 34},
}


def get_prop_data(property_name, id_prop_dir=None):
    """Load ids and target values from `id_prop_{property_name}.csv`."""
    base_dir = id_prop_dir or "./data/id_prop"
    file_name = f"id_prop_{property_name}.csv"
    file_path = os.path.join(base_dir, file_name)
    df_feat = pd.read_csv(
        file_path,
        header=None,
        names=["id", property_name],
    )
    df_feat = df_feat.set_index("id")
    y = df_feat[property_name].values
    index = df_feat.index.values
    logger.info("Loaded target file: %s (rows=%d)", file_path, len(index))
    return index, y


def parity_plot(
    test_data,
    valid_data,
    train_data,
    test_score,
    property,
):
    """Build a publication-style parity plot figure for train/valid/test splits."""
    # Configure matplotlib cache in a writable location for CLI environments.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/zebra_prop_mplconfig")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 1.2,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 10,
        }
    )

    test_y, test_pred = test_data
    valid_y, valid_pred = valid_data
    train_y, train_pred = train_data

    all_true = np.concatenate([test_y, valid_y, train_y])
    all_pred = np.concatenate([test_pred, valid_pred, train_pred])
    data_min = float(np.min([all_true.min(), all_pred.min()]))
    data_max = float(np.max([all_true.max(), all_pred.max()]))
    padding = max(1e-6, 0.05 * (data_max - data_min if data_max > data_min else 1.0))
    lim_low, lim_high = data_min - padding, data_max + padding

    fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
    # Draw the ideal y=x parity reference line.
    ax.plot(
        [lim_low, lim_high],
        [lim_low, lim_high],
        "--",
        color="#4b4b4b",
        linewidth=1.5,
    )

    ax.scatter(
        train_y,
        train_pred,
        s=SPLIT_STYLES["train"]["size"],
        alpha=SPLIT_STYLES["train"]["alpha"],
        color=SPLIT_STYLES["train"]["color"],
        edgecolor="none",
        label="Train",
    )
    ax.scatter(
        valid_y,
        valid_pred,
        s=SPLIT_STYLES["valid"]["size"],
        alpha=SPLIT_STYLES["valid"]["alpha"],
        color=SPLIT_STYLES["valid"]["color"],
        edgecolor="none",
        label="Valid",
    )
    ax.scatter(
        test_y,
        test_pred,
        s=SPLIT_STYLES["test"]["size"],
        alpha=SPLIT_STYLES["test"]["alpha"],
        color=SPLIT_STYLES["test"]["color"],
        edgecolor="none",
        label="Test",
    )

    property_label = str(property)
    ax.set_xlabel(f"{property_label} (dataset)")
    ax.set_ylabel(f"{property_label} (prediction)")
    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)
    ax.set_aspect("equal", adjustable="box")

    legend = ax.legend(loc="lower right", frameon=True)
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#000000")
    legend.get_frame().set_alpha(0.9)

    test_mae, test_rmse, test_r2 = test_score
    metrics_text = (
        f"RMSE: {test_rmse:.3f}\n"
        f"MAE: {test_mae:.3f}\n"
        f"R$^2$: {test_r2:.3f}"
    )
    # The metric box shows the test split, which is the primary report target.
    ax.text(
        0.02,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "facecolor": "#ffffff",
            "alpha": 0.85,
            "edgecolor": "#000000",
            "boxstyle": "round,pad=0.3",
        },
    )

    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.2)

    return fig


def get_llm(hf_model_name, max_length=None):
    """Resolve Hugging Face model id and effective maximum sequence length."""
    if not hf_model_name:
        raise ValueError("hf_model_name must be specified in config")

    resolved_max_length = 512 if max_length is None else int(max_length)

    logger.info(
        "Using HF model `%s` with max_length=%d",
        hf_model_name,
        resolved_max_length,
    )
    return hf_model_name, resolved_max_length


def load_description_csvs(
    description_name, fold=None, base_dir="./data/description"
):
    """Load and merge all description CSV files under one task directory."""
    description_dir = os.path.join(base_dir, description_name)
    if fold is not None:
        description_dir = os.path.join(description_dir, f"fold_{fold}")
    if not os.path.isdir(description_dir):
        available = (
            sorted(
                [
                    d
                    for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))
                ]
            )
            if os.path.isdir(base_dir)
            else []
        )
        raise FileNotFoundError(
            f"description directory not found: {description_dir}. "
            f"Available under base_dir: {', '.join(available)}"
        )

    csv_files = sorted([f for f in os.listdir(description_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {description_dir}")

    merged_df = None
    for csv_file in csv_files:
        if csv_file == "all.csv":
            continue
        csv_path = os.path.join(description_dir, csv_file)
        df = pd.read_csv(csv_path)

        if "description" not in df.columns:
            raise ValueError(f"'description' column is missing in {csv_path}")
        if "id" not in df.columns or "formula" not in df.columns:
            raise ValueError(f"'id' or 'formula' column is missing in {csv_path}")

        col_name = os.path.splitext(csv_file)[0]
        df_subset = df[["id", "formula", "description"]].copy()
        df_subset.rename(columns={"description": col_name}, inplace=True)

        if merged_df is None:
            merged_df = df_subset
        else:
            # Keep an outer merge on id so rows are not dropped when one source is sparse.
            merged_df = pd.merge(
                merged_df,
                df_subset,
                on="id",
                how="outer",
                suffixes=("", "__new"),
            )
            if "formula__new" in merged_df.columns:
                # Keep the first available formula string for each id.
                merged_df["formula"] = merged_df["formula"].combine_first(
                    merged_df["formula__new"]
                )
                merged_df = merged_df.drop(columns=["formula__new"])

    if merged_df is None:
        raise ValueError(
            f"No usable description CSV files found in {description_dir} (all files were skipped)."
        )

    return merged_df


def get_desc_df(desc_df):
    """Return a dataframe of usable description columns and their names."""
    desc_cols = [col for col in desc_df.columns if col not in ["id", "formula"]]
    if not desc_cols:
        raise ValueError("No description columns found after excluding `id` and `formula`.")

    selected_columns = [*desc_cols, "formula"]
    if "id" in desc_df.columns:
        selected_columns.insert(0, "id")
    desc_df = desc_df[selected_columns]
    logger.info("Using description columns: %s", desc_cols)
    return desc_df, desc_cols


def add_cls_to_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Prefix `[CLS]` to selected text columns."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: "[CLS] " + x)
    return df


def precompute_embeddings(
    loader,
    extractor,
    device,
    out_path=None,
    dataset_size=None,
    progress_desc="Computing embeddings",
    log_interval=10,
    logger_instance=None,
):
    """Precompute and optionally cache embeddings as a float16 tensor."""
    active_logger = logger_instance or logger
    save_outputs = out_path is not None

    if save_outputs and dataset_size is not None:
        base_path, ext = os.path.splitext(out_path)
        out_path = f"{base_path}_{dataset_size}{ext}"

    if save_outputs and os.path.exists(out_path):
        active_logger.info("Loading existing embedding tensor from %s", out_path)
        data = torch.load(out_path, map_location=device)
        return data

    extractor.model.eval()
    extractor.eval()
    extractor.to(device)
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    start_time = time.perf_counter()
    with torch.inference_mode():
        chunks = []
        processed_samples = 0
        non_blocking = device.type == "cuda"
        iterator = tqdm(
            enumerate(loader, start=1),
            total=total_batches,
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_idx, desc_inputs in iterator:
            desc_inputs = desc_inputs["desc_inputs"]
            input_ids = torch.stack([d["input_ids"] for d in desc_inputs], dim=1)
            attention_mask = torch.stack([d["attention_mask"] for d in desc_inputs], dim=1)
            batch_size, num_desc, seq_len = input_ids.shape

            flat_input_ids = input_ids.reshape(batch_size * num_desc, seq_len).to(
                device, non_blocking=non_blocking
            )
            flat_attention_mask = attention_mask.reshape(
                batch_size * num_desc, seq_len
            ).to(device, non_blocking=non_blocking)

            flat_embs = extractor(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
            )
            chunks.append(flat_embs.reshape(batch_size, num_desc, -1).cpu().half())
            processed_samples += batch_size
            iterator.set_postfix(samples=processed_samples)
            if total_batches is not None and (
                batch_idx == 1
                or batch_idx == total_batches
                or (log_interval > 0 and batch_idx % log_interval == 0)
            ):
                progress_pct = 100.0 * batch_idx / total_batches
                active_logger.info(
                    "Embedding progress: batch %d/%d (%.1f%%), samples=%d",
                    batch_idx,
                    total_batches,
                    progress_pct,
                    processed_samples,
                )

        # Concatenate per-batch tensors into one (N, num_desc, hidden) tensor.
        all_embs = torch.cat(chunks, dim=0)
        total_seconds = time.perf_counter() - start_time
        active_logger.info(
            "Embedding computation finished in %.2f sec. Output shape=%s",
            total_seconds,
            tuple(all_embs.shape),
        )
        if save_outputs:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(all_embs, out_path)
        return all_embs


def split_dataset_kfold(X, k_fold=5, test_fold=0, random_seed=42):
    """Split indices into train/valid/test folds with random shuffling."""
    X = np.asarray(X)
    indices = np.arange(len(X))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)
    X_shuffled = X[indices]

    fold_sizes = np.full(k_fold, len(X) // k_fold, dtype=int)
    fold_sizes[: len(X) % k_fold] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(X_shuffled[start:stop])
        current = stop

    test_idx = test_fold
    valid_idx = (test_fold + 1) % k_fold
    train_idx = [i for i in range(k_fold) if i != test_idx and i != valid_idx]

    X_test = folds[test_idx]
    X_valid = folds[valid_idx]
    X_train = np.concatenate([folds[i] for i in train_idx])

    test_preview = X_test[:10]
    logger.info(
        "First 10 entries of the test split (k_fold=%s, test_fold=%s):\n%s",
        k_fold,
        test_fold,
        np.array2string(test_preview, threshold=100),
    )

    return X_train, X_valid, X_test


def standardize_tensor(
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
    test_tensor: torch.Tensor,
):
    """Standardize train/val/test tensors using train split statistics."""
    mean = train_tensor.mean(dim=0, keepdim=True)
    std = train_tensor.std(dim=0, keepdim=True)

    std[std == 0] = 1.0

    train_std = (train_tensor - mean) / std
    val_std = (val_tensor - mean) / std
    test_std = (test_tensor - mean) / std

    return train_std, val_std, test_std
