"""Descriptor-to-description conversion utilities."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import re
from typing import Any

import numpy as np
import pandas as pd

MASK_TOKEN = "[MASK]"
# Allow spaces and symbols used by matminer labels.
_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
_RESERVED_COLUMNS = {"id", "formula", "description"}


def extract_feature_labels(df: pd.DataFrame) -> list[str]:
    """Return descriptor column labels available for sentence templates."""
    return [col for col in df.columns if col not in _RESERVED_COLUMNS]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_value(value: Any) -> str:
    if _is_missing(value):
        return MASK_TOKEN
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.8g}"
        return MASK_TOKEN
    if isinstance(value, str):
        return value
    return str(value)


def extract_template_placeholders(template: str) -> list[str]:
    """Extract unique placeholder labels from `{{label}}` expressions."""
    return sorted(
        {
            token.strip()
            for token in _PLACEHOLDER_PATTERN.findall(template)
            if token.strip()
        }
    )


def render_template(row: pd.Series, template: str) -> str:
    """Render one sentence template against one descriptor row."""

    def _replace(match: re.Match[str]) -> str:
        label = match.group(1)
        if label not in row.index:
            return MASK_TOKEN
        return _normalize_value(row[label])

    rendered = _PLACEHOLDER_PATTERN.sub(_replace, template)
    return rendered.strip()


def _parse_recursive_formula(subformula: str, multiplier: int = 1) -> dict[str, int]:
    elements: dict[str, int] = defaultdict(int)
    pattern = r"(\([A-Za-z0-9]*\)\d*|[A-Z][a-z]?\d*)"
    matches = re.findall(pattern, subformula)

    for match in matches:
        if match.startswith("("):
            inner_formula, count = re.match(r"\((.*?)\)(\d*)", match).groups()
            count = int(count) if count else 1
            inner_elements = _parse_recursive_formula(inner_formula, multiplier * count)
            for element, num in inner_elements.items():
                elements[element] += num
        else:
            element, count = re.match(r"([A-Z][a-z]?)(\d*)", match).groups()
            count = int(count) if count else 1
            elements[element] += count * multiplier

    return dict(elements)


def simplify_formula(formula: Any) -> str:
    """Convert `SiO2` style formula to `Si 1 O 2` style text."""
    if _is_missing(formula):
        return MASK_TOKEN

    formula_text = str(formula).strip()
    if not formula_text:
        return MASK_TOKEN

    try:
        parsed = _parse_recursive_formula(formula_text)
        if not parsed:
            return formula_text
        return " ".join(f"{el} {cnt}" for el, cnt in parsed.items())
    except Exception:
        return formula_text


def _scale_numeric_series(values: pd.Series) -> pd.Series:
    """Scale one numeric column and convert to integer representation."""
    valid = values.dropna()
    if valid.empty:
        return values

    abs_mean = float(np.mean(np.abs(valid.to_numpy(dtype=float))))
    if abs_mean == 0:
        return values.fillna(0).round().astype(int)

    if abs_mean > 100:
        factor = 10 ** np.floor(np.log10(100 / abs_mean))
    elif abs_mean <= 10:
        factor = 10 ** np.ceil(np.log10(10 / abs_mean))
    else:
        factor = 1

    scaled = (values * factor).round()
    return scaled.fillna(0).astype(int)


def preprocess_dataframe(
    df: pd.DataFrame,
    formula_simplified: bool = False,
    integerize: bool = False,
) -> pd.DataFrame:
    """Apply optional global preprocessing to formula and descriptor labels."""
    prepared = df.copy()

    if formula_simplified and "formula" in prepared.columns:
        prepared["formula"] = prepared["formula"].map(simplify_formula)

    if integerize:
        numeric_cols = [
            col
            for col in prepared.columns
            if col not in _RESERVED_COLUMNS and pd.api.types.is_numeric_dtype(prepared[col])
        ]
        for col in numeric_cols:
            prepared[col] = _scale_numeric_series(prepared[col])

    return prepared


def _default_description(
    row: pd.Series,
    labels: list[str],
    include_zero: bool = False,
) -> str:
    formula = _normalize_value(row.get("formula", ""))
    material_name = formula if formula != MASK_TOKEN else "This material"

    parts: list[str] = []
    for label in labels:
        value = row.get(label)
        if _is_missing(value):
            continue

        if not include_zero and isinstance(value, (int, float, np.integer, np.floating)):
            if float(value) == 0.0:
                continue

        parts.append(f"{label.replace('_', ' ')} of {_normalize_value(value)}")

    if not parts:
        return f"{material_name} has no available descriptor values."
    return f"{material_name} has {parts[0]}" + (
        f", {', '.join(parts[1:])}." if len(parts) > 1 else "."
    )


def generate_description_from_dataframe(
    df: pd.DataFrame,
    template: str | None = None,
    include_zero: bool = False,
    formula_simplified: bool = False,
    integerize: bool = False,
) -> pd.DataFrame:
    """Generate `id,formula,description` rows from descriptor dataframe."""
    working_df = preprocess_dataframe(
        df,
        formula_simplified=formula_simplified,
        integerize=integerize,
    )
    if "id" not in working_df.columns and "material_id" in working_df.columns:
        # Accept matminer-style id column names without forcing row-number ids.
        working_df = working_df.rename(columns={"material_id": "id"})
    if "id" not in working_df.columns:
        working_df.insert(0, "id", range(len(working_df)))
    if "formula" not in working_df.columns:
        working_df.insert(1, "formula", "")

    labels = extract_feature_labels(working_df)
    output_rows: list[dict[str, Any]] = []

    for _, row in working_df.iterrows():
        if template:
            description = render_template(row, template)
        else:
            description = _default_description(row, labels, include_zero=include_zero)

        output_rows.append(
            {
                "id": row["id"],
                "formula": row["formula"],
                "description": description,
            }
        )

    return pd.DataFrame(output_rows)


def generate_description_from_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    template: str | None = None,
    template_file: str | Path | None = None,
    include_zero: bool = False,
    formula_simplified: bool = False,
    integerize: bool = False,
) -> Path:
    """Generate description CSV from one descriptor CSV file."""
    if template and template_file:
        raise ValueError("Use either `template` or `template_file`, not both.")

    if template_file:
        template = Path(template_file).read_text(encoding="utf-8")

    input_df = pd.read_csv(input_csv)
    output_df = generate_description_from_dataframe(
        input_df,
        template=template,
        include_zero=include_zero,
        formula_simplified=formula_simplified,
        integerize=integerize,
    )
    if "id" in input_df.columns and len(output_df) == len(input_df):
        # Keep id values exactly as read from the descriptor CSV.
        output_df["id"] = input_df["id"].values

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_path
