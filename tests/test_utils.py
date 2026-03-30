import pandas as pd
import pytest

from zebra_prop.utils import (
    get_desc_df,
    get_llm,
    load_description_csvs,
)


def test_get_llm_returns_hf_name_and_max_length():
    model_name, max_length = get_llm("m3rg-iitd/matscibert", max_length=256)
    assert model_name == "m3rg-iitd/matscibert"
    assert max_length == 256


def test_get_desc_df_uses_non_id_formula_columns():
    df = pd.DataFrame(
        {
            "id": [1],
            "formula": ["SiO2"],
            "a": ["x"],
            "b": ["y"],
        }
    )
    out_df, cols = get_desc_df(df)
    assert cols == ["a", "b"]
    assert list(out_df.columns) == ["id", "a", "b", "formula"]


def test_get_desc_df_raises_when_no_description_columns():
    df = pd.DataFrame({"id": [1], "formula": ["SiO2"]})
    with pytest.raises(ValueError, match="No description columns found"):
        get_desc_df(df)


def test_load_description_csvs_rejects_all_csv_only(tmp_path):
    desc_root = tmp_path / "description"
    task_dir = desc_root / "dummy"
    task_dir.mkdir(parents=True)
    (task_dir / "all.csv").write_text("id,formula,description\n1,SiO2,test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No usable description CSV files found"):
        load_description_csvs("dummy", base_dir=str(desc_root))


def test_load_description_csvs_merges_on_id_only(tmp_path):
    desc_root = tmp_path / "description"
    task_dir = desc_root / "dummy"
    task_dir.mkdir(parents=True)

    (task_dir / "A.csv").write_text(
        "id,formula,description\n1,SiO2,desc_a\n",
        encoding="utf-8",
    )
    (task_dir / "B.csv").write_text(
        "id,formula,description\n1,SiO2_alt,desc_b\n",
        encoding="utf-8",
    )

    merged = load_description_csvs("dummy", base_dir=str(desc_root))
    assert len(merged) == 1
    assert list(merged.columns) == ["id", "formula", "A", "B"]
    assert merged.iloc[0]["A"] == "desc_a"
    assert merged.iloc[0]["B"] == "desc_b"
