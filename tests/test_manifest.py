from pathlib import Path

from csfseg.batch.manifest import load_input_paths


def test_manifest_uses_filename_as_output_id(tmp_path: Path):
    nii = tmp_path / "sub-001_ses-01_task-rest_run-01_bold.nii.gz"
    nii.write_bytes(b"placeholder")
    csv_path = tmp_path / "input_paths.csv"
    csv_path.write_text(f"input_path\n{nii}\n")

    records = load_input_paths(csv_path)

    assert len(records) == 1
    assert records[0].output_id == "sub-001_ses-01_task-rest_run-01_bold"


def test_text_input_paths(tmp_path: Path):
    nii = tmp_path / "sub-002_ses-01_task-rest_run-01_bold.nii.gz"
    nii.write_bytes(b"placeholder")
    txt_path = tmp_path / "input_paths.txt"
    txt_path.write_text(f"# comment\n\n{nii}\n")

    records = load_input_paths(txt_path)

    assert len(records) == 1
    assert records[0].input_path == nii.resolve()
    assert records[0].output_id == "sub-002_ses-01_task-rest_run-01_bold"


def test_tsv_input_paths_with_output_id(tmp_path: Path):
    nii = tmp_path / "rest_bold.nii.gz"
    nii.write_bytes(b"placeholder")
    tsv_path = tmp_path / "input_paths.tsv"
    tsv_path.write_text(f"input_path\toutput_id\n{nii}\tsiteA_sub-001_rest\n")

    records = load_input_paths(tsv_path)

    assert len(records) == 1
    assert records[0].output_id == "siteA_sub-001_rest"
