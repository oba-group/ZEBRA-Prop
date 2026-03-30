import pandas as pd

from zebra_tool_kit.description import (
    extract_feature_labels,
    extract_template_placeholders,
    generate_description_from_dataframe,
    preprocess_dataframe,
    render_template,
    simplify_formula,
)


def test_extract_feature_labels_excludes_reserved_columns():
    df = pd.DataFrame(
        {
            "id": [1],
            "formula": ["SiO2"],
            "description": ["old"],
            "a": [10],
            "b": [20],
        }
    )
    assert extract_feature_labels(df) == ["a", "b"]


def test_render_template_replaces_missing_values_with_mask():
    row = pd.Series({"formula": "Al2O3", "mean_mass": 123.456, "missing_col": None})
    template = "{{formula}} / {{mean_mass}} / {{missing_col}} / {{unknown}}"
    rendered = render_template(row, template)
    assert rendered == "Al2O3 / 123.456 / [MASK] / [MASK]"


def test_extract_template_placeholders_supports_spaces():
    template = "{{formula}} has {{PymatgenData mean row}} and {{PymatgenData std_dev row}}."
    labels = extract_template_placeholders(template)
    assert labels == ["PymatgenData mean row", "PymatgenData std_dev row", "formula"]


def test_generate_descriptions_from_dataframe_default_mode():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "formula": ["SiO2", "Al2O3"],
            "feat_a": [1.0, 0.0],
            "feat_b": [None, 3.0],
        }
    )
    output = generate_description_from_dataframe(df)

    assert list(output.columns) == ["id", "formula", "description"]
    assert "feat a of 1" in output.iloc[0]["description"]
    assert "feat b of 3" in output.iloc[1]["description"]
    assert "feat a of 0" not in output.iloc[1]["description"]


def test_simplify_formula():
    assert simplify_formula("SiO2") == "Si 1 O 2"


def test_preprocess_dataframe_integerize_and_formula():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "formula": ["SiO2", "Al2O3"],
            "feat_x": [0.12, 0.24],
            "feat_y": [50.5, 100.2],
        }
    )
    out = preprocess_dataframe(df, formula_simplified=True, integerize=True)
    assert out["formula"].iloc[0] == "Si 1 O 2"
    assert pd.api.types.is_integer_dtype(out["feat_x"])
    assert pd.api.types.is_integer_dtype(out["feat_y"])
