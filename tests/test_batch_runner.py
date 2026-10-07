import csv
from pathlib import Path

import pytest

from csfseg.batch import runner
from csfseg.batch.runner import (
    BatchRunConfig,
    expected_core_outputs,
    resolve_subject_workers,
    run_batch,
)
from csfseg.pipeline import SingleRunResult


def _write_manifest(tmp_path: Path, rows: list[tuple[str, Path]]) -> Path:
    manifest_path = tmp_path / "input_paths.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_path", "output_id"])
        writer.writeheader()
        for output_id, input_path in rows:
            writer.writerow({"input_path": str(input_path), "output_id": output_id})
    return manifest_path


def _fake_single_result(config, *, logger=None, log_path=None, run_id=None):
    output_id = config.resolved_output_id()
    out_dir = config.resolved_out_dir
    probability_path = out_dir / "probabilities" / f"{output_id}_csf_prob.nii.gz"
    mask_path = out_dir / "masks" / f"{output_id}_csf_mask_bottom10.nii.gz"
    prediction_qc_path = out_dir / "qc" / f"{output_id}_prediction_qc.png"
    voxel_table_path = out_dir / "timeseries" / f"{output_id}_voxel_table.csv"
    selection_template_path = out_dir / "reports" / "selection_template.csv"
    selected_voxels_path = out_dir / "selected" / "auto" / f"{output_id}_L1_auto_selected_voxels.csv"
    timeseries_qc_path = out_dir / "selected" / "auto" / f"{output_id}_L1_auto_timeseries_qc.png"
    auto_summary_path = out_dir / "reports" / "auto_selection_summary.csv"

    for path in (
        probability_path,
        mask_path,
        prediction_qc_path,
        voxel_table_path,
        selection_template_path,
        selected_voxels_path,
        timeseries_qc_path,
        auto_summary_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder")

    return SingleRunResult(
        output_id=output_id,
        status="ok",
        input_path=config.resolved_input_path,
        out_dir=out_dir,
        checkpoint_path=config.resolved_checkpoint,
        device="cpu",
        log_path=Path(log_path).resolve() if log_path else None,
        probability_path=probability_path,
        mask_path=mask_path,
        prediction_qc_path=prediction_qc_path,
        voxel_table_path=voxel_table_path,
        selection_template_path=selection_template_path,
        selected_voxels_path=selected_voxels_path,
        timeseries_qc_path=timeseries_qc_path,
        auto_summary_path=auto_summary_path,
        auto_layer="L1",
        mask_voxels=20,
        selected_voxels=12,
        patch_shape=(3, 10, 128, 128),
        probability_patch_shape=(10, 128, 128),
        probability_full_shape=(8, 8, 10),
        checkpoint_requires_finetuning=False,
        elapsed_seconds=0.25,
    )


def test_run_batch_writes_reports_and_logs(tmp_path, monkeypatch):
    inputs = []
    for output_id in ("sub001", "sub002"):
        input_path = tmp_path / f"{output_id}_bold.nii.gz"
        input_path.write_bytes(b"placeholder")
        inputs.append((output_id, input_path))
    manifest_path = _write_manifest(tmp_path, inputs)
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(runner, "run_single", _fake_single_result)

    progress = []
    result = run_batch(
        BatchRunConfig(
            input_paths=manifest_path,
            out_dir=tmp_path / "outputs",
            checkpoint=checkpoint_path,
            device="cpu",
        ),
        progress_callback=lambda index, total, row: progress.append((index, total, row.status)),
    )

    assert result.status == "ok"
    assert result.total == 2
    assert result.ok == 2
    assert result.failed == 0
    assert result.skipped == 0
    assert progress == [(1, 2, "ok"), (2, 2, "ok")]
    assert result.input_paths_used_path.exists()
    assert result.batch_report_path.exists()
    assert result.failed_inputs_path.exists()
    assert result.run_config_path.exists()
    assert result.run_log_path.exists()

    report_rows = list(csv.DictReader(result.batch_report_path.open()))
    assert [row["status"] for row in report_rows] == ["ok", "ok"]
    assert report_rows[0]["auto_layer"] == "L1"
    assert report_rows[0]["selected_voxels"] == "12"

    failed_rows = list(csv.DictReader(result.failed_inputs_path.open()))
    assert failed_rows == []
    assert "subject_workers: 1" in result.run_config_path.read_text()
    assert "batch_summary" in result.run_log_path.read_text()


def test_run_batch_skip_existing_does_not_call_pipeline(tmp_path, monkeypatch):
    input_path = tmp_path / "sub001_bold.nii.gz"
    input_path.write_bytes(b"placeholder")
    manifest_path = _write_manifest(tmp_path, [("sub001", input_path)])
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    out_dir = tmp_path / "outputs"
    for path in expected_core_outputs(out_dir, "sub001").values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("already exists")

    def should_not_run(*args, **kwargs):
        raise AssertionError("run_single should not be called when skip_existing is true")

    monkeypatch.setattr(runner, "run_single", should_not_run)

    result = run_batch(
        BatchRunConfig(
            input_paths=manifest_path,
            out_dir=out_dir,
            checkpoint=checkpoint_path,
            skip_existing=True,
        )
    )

    assert result.status == "ok"
    assert result.ok == 0
    assert result.skipped == 1
    assert result.rows[0].status == "skipped_existing"
    assert result.rows[0].stage == "skip_existing"


def test_run_batch_continues_after_subject_failure(tmp_path, monkeypatch):
    input_path = tmp_path / "sub001_bold.nii.gz"
    input_path.write_bytes(b"placeholder")
    manifest_path = _write_manifest(tmp_path, [("sub001", input_path)])
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    def fail_single(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(runner, "run_single", fail_single)

    result = run_batch(
        BatchRunConfig(
            input_paths=manifest_path,
            out_dir=tmp_path / "outputs",
            checkpoint=checkpoint_path,
        )
    )

    assert result.status == "completed_with_failures"
    assert result.failed == 1
    assert result.rows[0].status == "failed"
    assert "RuntimeError: model exploded" in result.rows[0].error
    failed_rows = list(csv.DictReader(result.failed_inputs_path.open()))
    assert len(failed_rows) == 1
    assert failed_rows[0]["output_id"] == "sub001"
    assert "model exploded" in failed_rows[0]["error"]
    assert "Batch item failed." in result.rows[0].log_path.read_text()


def test_resolve_subject_workers_first_version():
    assert resolve_subject_workers("auto") == 1
    assert resolve_subject_workers("1") == 1
    assert resolve_subject_workers(1) == 1
    assert resolve_subject_workers(2) == 2
    with pytest.raises(ValueError, match="positive"):
        resolve_subject_workers(0)
    with pytest.raises(ValueError, match="positive integer"):
        resolve_subject_workers("many")
