import pandas as pd

from zebra_tool_kit.description import generate_description_from_csv


def test_generate_description_from_csv_preserves_input_id_values(tmp_path):
    input_path = tmp_path / "descriptor.csv"
    output_path = tmp_path / "description.csv"
    pd.DataFrame(
        {
            "id": ["0_Ag2BiO3", "1_Ag2CO3"],
            "formula": ["Ag2BiO3", "Ag2CO3"],
            "feat": [1.0, 2.0],
        }
    ).to_csv(input_path, index=False)

    generate_description_from_csv(input_path, output_path)
    out_df = pd.read_csv(output_path)

    assert out_df["id"].tolist() == ["0_Ag2BiO3", "1_Ag2CO3"]


def test_generate_description_from_csv_uses_material_id_column(tmp_path):
    input_path = tmp_path / "descriptor.csv"
    output_path = tmp_path / "description.csv"
    pd.DataFrame(
        {
            "material_id": ["m-1", "m-2"],
            "formula": ["SiO2", "Al2O3"],
            "feat": [1.0, 2.0],
        }
    ).to_csv(input_path, index=False)

    generate_description_from_csv(input_path, output_path)
    out_df = pd.read_csv(output_path)

    assert out_df["id"].tolist() == ["m-1", "m-2"]
