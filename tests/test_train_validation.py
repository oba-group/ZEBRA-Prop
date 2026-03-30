import pytest

from zebra_prop.train import DATA_FORMAT_DOC_PATH, validate_training_files


def test_validate_training_files_missing_target_has_hint(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_training_files(data_dir, property_name="band_gap", task_name="human-made")

    message = str(excinfo.value)
    assert "Hint:" in message
    assert f"Documentation: {DATA_FORMAT_DOC_PATH}" in message


def test_validate_training_files_missing_description_columns_has_hint(tmp_path):
    data_dir = tmp_path / "data"
    id_prop_dir = data_dir / "id_prop"
    desc_dir = data_dir / "description" / "human-made"
    id_prop_dir.mkdir(parents=True)
    desc_dir.mkdir(parents=True)

    (id_prop_dir / "id_prop_band_gap.csv").write_text("1,1.23\n2,2.34\n", encoding="utf-8")
    (desc_dir / "AtomicOrbitals.csv").write_text(
        "id,formula,text\n1,SiO2,desc\n2,Al2O3,desc\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        validate_training_files(data_dir, property_name="band_gap", task_name="human-made")

    message = str(excinfo.value)
    assert "missing columns" in message
    assert "Hint:" in message
    assert f"Documentation: {DATA_FORMAT_DOC_PATH}" in message
