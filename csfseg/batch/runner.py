"""Batch runner for manifest-driven CSFSeg prediction."""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from csfseg.batch.manifest import InputRecord, load_input_paths, write_input_paths_used
from csfseg.config import ensure_out_dir
from csfseg.pipeline import SingleRunConfig, SingleRunResult, run_single
from csfseg.utils.logging import (
    close_logger,
    log_run_header,
    log_values,
    setup_run_logger,
    setup_subject_logger,
)


ProgressCallback = Callable[[int, int, "BatchItemResult"], None]


@dataclass(frozen=True)
class BatchRunConfig:
    """Configuration for one manifest-driven batch command."""

    input_paths: str | Path
    out_dir: str | Path
    checkpoint: str | Path
    device: str = "auto"
    threshold: float = 0.5
    min_valid_voxels: int = 5
    dropout_warning_fraction: float = 0.20
    subject_workers: int | str = "auto"
    skip_existing: bool = False
    model_in_channels: int = 3
    model_base: int = 16
    model_out_channels: int = 1

    @property
    def resolved_input_paths(self) -> Path:
        return Path(self.input_paths).expanduser().resolve()

    @property
    def resolved_out_dir(self) -> Path:
        return Path(self.out_dir).expanduser().resolve()

    @property
    def resolved_checkpoint(self) -> Path:
        return Path(self.checkpoint).expanduser().resolve()


@dataclass(frozen=True)
class BatchItemResult:
    """One row of batch status."""

    output_id: str
    input_path: Path
    status: str
    stage: str
    auto_layer: str
    mask_voxels: int | None
    selected_voxels: int | None
    elapsed_seconds: float
    log_path: Path
    error: str
    probability_path: Path | None = None
    mask_path: Path | None = None
    prediction_qc_path: Path | None = None
    voxel_table_path: Path | None = None
    selected_voxels_path: Path | None = None
    timeseries_qc_path: Path | None = None


@dataclass(frozen=True)
class BatchRunResult:
    """Summary of one batch command."""

    run_id: str
    status: str
    total: int
    ok: int
    failed: int
    skipped: int
    elapsed_seconds: float
    input_paths_used_path: Path
    batch_report_path: Path
    failed_inputs_path: Path
    run_config_path: Path
    run_log_path: Path
    rows: tuple[BatchItemResult, ...]


def run_batch(
    config: BatchRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> BatchRunResult:
    """Run a manifest one input at a time and write batch reports."""

    started = time.perf_counter()
    out_dir = ensure_out_dir(config.out_dir)
    subject_workers = resolve_subject_workers(config.subject_workers)
    if subject_workers != 1:
        raise ValueError(
            "This first batch implementation supports only subject_workers=1. "
            "Use --subject-workers auto or --subject-workers 1."
        )

    records = load_input_paths(config.input_paths)
    input_paths_used_path = write_input_paths_used(records, out_dir / "input_paths_used.csv")
    run_logger, run_log_path, run_id = setup_run_logger(out_dir)
    run_config_path = out_dir / "run_config.yaml"
    _write_run_config(config, run_config_path, run_id, subject_workers)

    rows: list[BatchItemResult] = []
    try:
        log_run_header(
            run_logger,
            title="CSFSeg batch log",
            values={
                "run_id": run_id,
                "input_paths": config.resolved_input_paths,
                "out_dir": out_dir,
                "checkpoint": config.resolved_checkpoint,
                "device": config.device,
                "subject_workers": subject_workers,
                "skip_existing": config.skip_existing,
                "total_inputs": len(records),
                "input_paths_used": input_paths_used_path,
                "run_config": run_config_path,
            },
        )

        total = len(records)
        for index, record in enumerate(records, start=1):
            row = _run_one_record(
                record,
                config,
                out_dir=out_dir,
                run_id=run_id,
                run_logger=run_logger,
            )
            rows.append(row)
            if progress_callback:
                progress_callback(index, total, row)

        report_path = write_batch_report(rows, out_dir / "reports" / "batch_report.csv")
        failed_path = write_failed_inputs(rows, out_dir / "reports" / "failed_inputs.csv")
        elapsed = time.perf_counter() - started
        failed = sum(1 for row in rows if row.status == "failed")
        skipped = sum(1 for row in rows if row.status == "skipped_existing")
        ok = sum(1 for row in rows if row.status == "ok")
        status = "ok" if failed == 0 else "completed_with_failures"
        log_values(
            run_logger,
            "batch_summary",
            {
                "status": status,
                "total": total,
                "ok": ok,
                "failed": failed,
                "skipped": skipped,
                "elapsed_seconds": f"{elapsed:.3f}",
                "batch_report": report_path,
                "failed_inputs": failed_path,
            },
        )
        return BatchRunResult(
            run_id=run_id,
            status=status,
            total=total,
            ok=ok,
            failed=failed,
            skipped=skipped,
            elapsed_seconds=elapsed,
            input_paths_used_path=input_paths_used_path,
            batch_report_path=report_path,
            failed_inputs_path=failed_path,
            run_config_path=run_config_path,
            run_log_path=run_log_path,
            rows=tuple(rows),
        )
    finally:
        close_logger(run_logger)


def resolve_subject_workers(value: int | str) -> int:
    """Resolve the current serial-batch subject worker setting."""

    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return 1
        try:
            parsed = int(text)
        except ValueError as exc:
            raise ValueError(f"subject_workers must be 'auto' or a positive integer, got {value!r}.") from exc
    else:
        parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"subject_workers must be positive, got {value!r}.")
    return parsed


def should_skip_existing(out_dir: str | Path, output_id: str) -> bool:
    """Return true if the core prediction outputs already exist."""

    paths = expected_core_outputs(out_dir, output_id)
    return all(path.exists() for path in paths.values())


def expected_core_outputs(out_dir: str | Path, output_id: str) -> dict[str, Path]:
    """Core files that define an already completed single-file prediction."""

    root = Path(out_dir).expanduser().resolve()
    return {
        "probability_path": root / "probabilities" / f"{output_id}_csf_prob.nii.gz",
        "mask_path": root / "masks" / f"{output_id}_csf_mask_bottom10.nii.gz",
        "prediction_qc_path": root / "qc" / f"{output_id}_prediction_qc.png",
        "voxel_table_path": root / "timeseries" / f"{output_id}_voxel_table.csv",
    }


def write_batch_report(rows: list[BatchItemResult], out_path: str | Path) -> Path:
    """Write the one-row-per-input batch report."""

    return _write_rows(rows, out_path, include_failures=True)


def write_failed_inputs(rows: list[BatchItemResult], out_path: str | Path) -> Path:
    """Write the smaller failure-only report."""

    failed_rows = [row for row in rows if row.status == "failed"]
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["output_id", "input_path", "stage", "error", "log_path"],
        )
        writer.writeheader()
        for row in failed_rows:
            writer.writerow(
                {
                    "output_id": row.output_id,
                    "input_path": str(row.input_path),
                    "stage": row.stage,
                    "error": row.error,
                    "log_path": str(row.log_path),
                }
            )
    return out_path


def _run_one_record(
    record: InputRecord,
    config: BatchRunConfig,
    *,
    out_dir: Path,
    run_id: str,
    run_logger,
) -> BatchItemResult:
    start = time.perf_counter()
    subject_logger, log_path, _ = setup_subject_logger(record.output_id, out_dir, run_id=run_id)
    try:
        log_run_header(
            subject_logger,
            title="CSFSeg batch subject",
            values={
                "run_id": run_id,
                "output_id": record.output_id,
                "input_path": record.input_path,
                "out_dir": out_dir,
                "checkpoint": config.resolved_checkpoint,
                "skip_existing": config.skip_existing,
                "manifest_row": record.row_number,
            },
        )

        if config.skip_existing and should_skip_existing(out_dir, record.output_id):
            elapsed = time.perf_counter() - start
            paths = expected_core_outputs(out_dir, record.output_id)
            log_values(
                subject_logger,
                "skipped_existing",
                {
                    "status": "skipped_existing",
                    **paths,
                },
            )
            log_values(
                run_logger,
                "skipped_existing",
                {"output_id": record.output_id, "input_path": record.input_path},
            )
            return BatchItemResult(
                output_id=record.output_id,
                input_path=record.input_path,
                status="skipped_existing",
                stage="skip_existing",
                auto_layer="",
                mask_voxels=None,
                selected_voxels=None,
                elapsed_seconds=elapsed,
                log_path=log_path,
                error="",
                **paths,
            )

        single_config = SingleRunConfig(
            input_path=record.input_path,
            out_dir=out_dir,
            checkpoint=config.checkpoint,
            output_id=record.output_id,
            device=config.device,
            threshold=config.threshold,
            min_valid_voxels=config.min_valid_voxels,
            dropout_warning_fraction=config.dropout_warning_fraction,
            model_in_channels=config.model_in_channels,
            model_base=config.model_base,
            model_out_channels=config.model_out_channels,
        )
        result = run_single(single_config, logger=subject_logger, log_path=log_path, run_id=run_id)
        log_values(
            run_logger,
            "subject_ok",
            {
                "output_id": result.output_id,
                "auto_layer": result.auto_layer or "none",
                "mask_voxels": result.mask_voxels,
                "selected_voxels": result.selected_voxels,
                "elapsed_seconds": f"{result.elapsed_seconds:.3f}",
                "log_path": log_path,
            },
        )
        return _row_from_single_result(result, log_path)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        subject_logger.exception("Batch item failed.")
        stage = _last_error_stage(log_path) or "run_single"
        error = f"{type(exc).__name__}: {exc}"
        log_values(
            run_logger,
            "subject_failed",
            {
                "output_id": record.output_id,
                "input_path": record.input_path,
                "stage": stage,
                "error": error,
                "log_path": log_path,
            },
        )
        return BatchItemResult(
            output_id=record.output_id,
            input_path=record.input_path,
            status="failed",
            stage=stage,
            auto_layer="",
            mask_voxels=None,
            selected_voxels=None,
            elapsed_seconds=elapsed,
            log_path=log_path,
            error=error,
        )
    finally:
        close_logger(subject_logger)


def _row_from_single_result(result: SingleRunResult, log_path: Path) -> BatchItemResult:
    return BatchItemResult(
        output_id=result.output_id,
        input_path=result.input_path,
        status="ok",
        stage="done",
        auto_layer=result.auto_layer,
        mask_voxels=result.mask_voxels,
        selected_voxels=result.selected_voxels,
        elapsed_seconds=result.elapsed_seconds,
        log_path=log_path,
        error="",
        probability_path=result.probability_path,
        mask_path=result.mask_path,
        prediction_qc_path=result.prediction_qc_path,
        voxel_table_path=result.voxel_table_path,
        selected_voxels_path=result.selected_voxels_path,
        timeseries_qc_path=result.timeseries_qc_path,
    )


def _write_rows(rows: list[BatchItemResult], out_path: str | Path, *, include_failures: bool) -> Path:
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "output_id",
        "input_path",
        "status",
        "stage",
        "auto_layer",
        "mask_voxels",
        "selected_voxels",
        "elapsed_seconds",
        "log_path",
        "error",
        "probability_path",
        "mask_path",
        "prediction_qc_path",
        "voxel_table_path",
        "selected_voxels_path",
        "timeseries_qc_path",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not include_failures and row.status != "failed":
                continue
            writer.writerow(_row_to_dict(row))
    return out_path


def _row_to_dict(row: BatchItemResult) -> dict[str, str | int | float]:
    return {
        "output_id": row.output_id,
        "input_path": str(row.input_path),
        "status": row.status,
        "stage": row.stage,
        "auto_layer": row.auto_layer,
        "mask_voxels": "" if row.mask_voxels is None else row.mask_voxels,
        "selected_voxels": "" if row.selected_voxels is None else row.selected_voxels,
        "elapsed_seconds": f"{row.elapsed_seconds:.3f}",
        "log_path": str(row.log_path),
        "error": row.error,
        "probability_path": _path_text(row.probability_path),
        "mask_path": _path_text(row.mask_path),
        "prediction_qc_path": _path_text(row.prediction_qc_path),
        "voxel_table_path": _path_text(row.voxel_table_path),
        "selected_voxels_path": _path_text(row.selected_voxels_path),
        "timeseries_qc_path": _path_text(row.timeseries_qc_path),
    }


def _path_text(path: Path | None) -> str:
    return "" if path is None else str(path)


def _write_run_config(config: BatchRunConfig, out_path: Path, run_id: str, subject_workers: int) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": run_id,
        "input_paths": str(config.resolved_input_paths),
        "out_dir": str(config.resolved_out_dir),
        "checkpoint": str(config.resolved_checkpoint),
        "device": config.device,
        "threshold": config.threshold,
        "min_valid_voxels": config.min_valid_voxels,
        "dropout_warning_fraction": config.dropout_warning_fraction,
        "subject_workers": subject_workers,
        "skip_existing": bool(config.skip_existing),
    }
    with out_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_path


def _last_error_stage(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    pattern = re.compile(r"\[ERROR\]\s+([A-Za-z0-9_-]+)")
    stage = ""
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            stage = match.group(1)
    return stage


__all__ = [
    "BatchItemResult",
    "BatchRunConfig",
    "BatchRunResult",
    "expected_core_outputs",
    "resolve_subject_workers",
    "run_batch",
    "should_skip_existing",
    "write_batch_report",
    "write_failed_inputs",
]
