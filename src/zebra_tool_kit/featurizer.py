"""CIF to matminer descriptor generation utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.core import Element, Structure

AVAILABLE_FEATURIZERS = (
    "AtomicOrbitals",
    "AtomicPackingEfficiency",
    "ElementFraction",
    "ElementProperty",
    "IonProperty",
    "ValenceOrbital",
    "ElectronegativityDiff",
    "OxidationStates",
    "DensityFeatures",
    "GlobalSymmetryFeatures",
)

_STRUCTURE_FEATURIZERS = {
    "DensityFeatures",
    "GlobalSymmetryFeatures",
}
_OXI_COMPOSITION_FEATURIZERS = {
    "ElectronegativityDiff",
    "OxidationStates",
}


def _load_featurizer_map() -> dict[str, Any]:
    """Build matminer featurizer instances lazily."""
    try:
        from matminer.featurizers.composition import (
            AtomicOrbitals,
            AtomicPackingEfficiency,
            ElementFraction,
            ElementProperty,
            ElectronegativityDiff,
            IonProperty,
            OxidationStates,
            ValenceOrbital,
        )
        from matminer.featurizers.structure import (
            DensityFeatures,
            GlobalSymmetryFeatures,
        )
    except ImportError as exc:
        raise RuntimeError(
            "matminer is required for featurization. Install it with: "
            "`uv add matminer` or `pip install matminer`."
        ) from exc

    element_property = ElementProperty.from_preset("matminer")
    element_property.features = [
        "row",
        "group",
        "block",
        "atomic_mass",
        "atomic_radius",
    ]
    element_property.stats = ["minimum", "maximum", "mean", "std_dev"]

    return {
        "AtomicOrbitals": AtomicOrbitals(),
        "AtomicPackingEfficiency": AtomicPackingEfficiency(),
        "ElementFraction": ElementFraction(),
        "ElementProperty": element_property,
        "IonProperty": IonProperty(),
        "ValenceOrbital": ValenceOrbital(),
        "ElectronegativityDiff": ElectronegativityDiff(),
        "OxidationStates": OxidationStates(),
        "DensityFeatures": DensityFeatures(),
        "GlobalSymmetryFeatures": GlobalSymmetryFeatures(),
    }


@lru_cache(maxsize=1)
def _common_oxidation_state_map() -> dict[str, int]:
    """Return a fallback oxidation-state mapping for all elements."""
    oxidation_state_map: dict[str, int] = {}
    for atomic_number in range(1, 119):
        element = Element.from_Z(atomic_number)
        oxidation_state_map[element.symbol] = (
            int(element.common_oxidation_states[0])
            if element.common_oxidation_states
            else 0
        )
    return oxidation_state_map


def _assign_oxidation_states(structure: Structure) -> Structure:
    """Assign oxidation states with guess-first, fallback-second strategy."""
    copied_structure = structure.copy()
    try:
        copied_structure.add_oxidation_state_by_guess()
        return copied_structure
    except Exception:
        try:
            fallback = structure.copy()
            fallback.add_oxidation_state_by_element(_common_oxidation_state_map())
            return fallback
        except Exception:
            return structure


def _feature_source_column(featurizer_name: str) -> str:
    if featurizer_name in _OXI_COMPOSITION_FEATURIZERS:
        return "composition_with_oxi"
    if featurizer_name in _STRUCTURE_FEATURIZERS:
        return "structure"
    return "composition"


def load_structures_from_cif(cif_dir: str | Path) -> pd.DataFrame:
    """Load all CIF files under `cif_dir` into id/formula/structure rows."""
    cif_path = Path(cif_dir)
    if not cif_path.is_dir():
        raise FileNotFoundError(f"CIF directory not found: {cif_path}")

    records: list[dict[str, Any]] = []
    skipped_files: list[Path] = []

    for cif_file in sorted(cif_path.glob("*.cif")):
        try:
            structure = Structure.from_file(str(cif_file))
            records.append(
                {
                    "id": cif_file.stem,
                    "formula": structure.composition.reduced_formula,
                    "structure": structure,
                }
            )
        except Exception:
            skipped_files.append(cif_file)

    if not records:
        raise ValueError(
            f"No CIF structures could be loaded from {cif_path}. "
            f"Skipped files: {len(skipped_files)}"
        )

    if skipped_files:
        print(
            f"[zebra-tool-kit] warning: skipped {len(skipped_files)} invalid CIF files."
        )

    return pd.DataFrame.from_records(records)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["composition"] = prepared["structure"].map(lambda s: s.composition)
    prepared["structure_with_oxi"] = prepared["structure"].map(_assign_oxidation_states)
    prepared["composition_with_oxi"] = prepared["structure_with_oxi"].map(
        lambda s: s.composition
    )
    return prepared


def compute_features(df: pd.DataFrame, featurizer_name: str) -> pd.DataFrame:
    """Compute one matminer descriptor dataframe from a structure dataframe."""
    if featurizer_name not in AVAILABLE_FEATURIZERS:
        raise ValueError(
            f"Unknown featurizer: {featurizer_name}. "
            f"Supported: {', '.join(AVAILABLE_FEATURIZERS)}"
        )

    featurizer_map = _load_featurizer_map()
    featurizer = featurizer_map[featurizer_name]
    prepared = _prepare_dataframe(df)
    source_col = _feature_source_column(featurizer_name)

    featured = featurizer.featurize_dataframe(
        prepared,
        col_id=source_col,
        ignore_errors=True,
    )
    return featured.drop(
        columns=["structure", "structure_with_oxi", "composition", "composition_with_oxi"],
        errors="ignore",
    )


def generate_feature_csvs(
    cif_dir: str | Path,
    output_dir: str | Path,
    featurizer: str = "all",
) -> dict[str, Path]:
    """Generate descriptor CSV files from CIF files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_df = load_structures_from_cif(cif_dir)

    targets = (
        list(AVAILABLE_FEATURIZERS)
        if featurizer == "all"
        else [featurizer]
    )
    saved_files: dict[str, Path] = {}

    for featurizer_name in targets:
        print(f"[zebra-tool-kit] generating {featurizer_name} ...")
        featured_df = compute_features(base_df, featurizer_name)
        output_file = output_path / f"{featurizer_name}.csv"
        featured_df.to_csv(output_file, index=False)
        saved_files[featurizer_name] = output_file
        print(f"[zebra-tool-kit] saved: {output_file}")

    return saved_files
