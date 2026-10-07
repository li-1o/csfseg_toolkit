"""Single-input CSFSeg prediction pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csfseg.batch.manifest import strip_nifti_suffix
from csfseg.config import ensure_out_dir
from csfseg.utils.logging import log_run_header, log_stage, log_values


@dataclass(frozen=True)
class SingleRunConfig:
    """Configuration for one complete input fMRI run."""

    input_path: str | Path
    out_dir: str | Path
    checkpoint: str | Path
    output_id: str | None = None
    device: str = "auto"
    threshold: float = 0.5
    min_valid_voxels: int = 5
    dropout_warning_fraction: float = 0.20
    model_in_channels: int = 3
    model_base: int = 16
    model_out_channels: int = 1

    @property
    def resolved_input_path(self) -> Path:
        return Path(self.input_path).expanduser().resolve()

    @property
    def resolved_out_dir(self) -> Path:
        return Path(self.out_dir).expanduser().resolve()

    @property
    def resolved_checkpoint(self) -> Path:
        return Path(self.checkpoint).expanduser().resolve()

    def resolved_output_id(self) -> str:
        return self.output_id or strip_nifti_suffix(self.resolved_input_path)


@dataclass(frozen=True)
class SingleRunResult:
    """Important outputs and summary values from one pipeline run."""

    output_id: str
    status: str
    input_path: Path
    out_dir: Path
    checkpoint_path: Path
    device: str
    log_path: Path | None
    probability_path: Path
    mask_path: Path
    prediction_qc_path: Path
    voxel_table_path: Path
    selection_template_path: Path
    selected_voxels_path: Path | None
    timeseries_qc_path: Path | None
    auto_summary_path: Path | None
    auto_layer: str
    mask_voxels: int
    selected_voxels: int
    patch_shape: tuple[int, ...]
    probability_patch_shape: tuple[int, ...]
    probability_full_shape: tuple[int, ...]
    checkpoint_requires_finetuning: bool
    elapsed_seconds: float


def run_single(
    config: SingleRunConfig,
    *,
    logger: logging.Logger | None = None,
    log_path: str | Path | None = None,
    run_id: str | None = None,
) -> SingleRunResult:
    """Run the complete single-file prediction and extraction pipeline."""

    import numpy as np

    from csfseg.inference.postprocess import restore_prediction_to_full
    from csfseg.inference.predict import load_model_for_inference, predict_patch
    from csfseg.io.nifti import load_nifti, save_mask_like, save_probability_like
    from csfseg.preprocessing.bottom import extract_bottom_volume
    from csfseg.preprocessing.features import compute_temporal_features
    from csfseg.preprocessing.normalize import normalize_features
    from csfseg.preprocessing.patch import make_patch
    from csfseg.qc.plots import plot_prediction_qc, plot_timeseries_qc
    from csfseg.timeseries.components import analyze_mask_layers, normalize_layer_name
    from csfseg.timeseries.extract import (
        extract_voxel_table,
        select_voxels,
        update_selection_template,
        update_summary_table,
        write_voxel_table,
    )

    started = time.perf_counter()
    input_path = config.resolved_input_path
    out_dir = ensure_out_dir(config.out_dir)
    checkpoint_path = config.resolved_checkpoint
    output_id = config.resolved_output_id()
    resolved_log_path = Path(log_path).expanduser().resolve() if log_path else None

    log_run_header(
        logger,
        title="CSFSeg subject log",
        values={
            "run_id": run_id or "",
            "output_id": output_id,
            "input_path": input_path,
            "out_dir": out_dir,
            "checkpoint": checkpoint_path,
            "device_request": config.device,
            "threshold": config.threshold,
            "min_valid_voxels": config.min_valid_voxels,
            "dropout_warning_fraction": config.dropout_warning_fraction,
        },
    )

    with log_stage(logger, "load_nifti"):
        reference_img = load_nifti(input_path)
        fmri_data = np.asanyarray(reference_img.dataobj, dtype=np.float32)
        log_values(logger, "input_nifti", _nifti_log_values(input_path, reference_img, fmri_data))

    with log_stage(logger, "preprocessing"):
        features = compute_temporal_features(fmri_data)
        normalized = normalize_features(features.data, features.channels)
        bottom = extract_bottom_volume(normalized.data)
        patch = make_patch(bottom.data)
        log_values(
            logger,
            "preprocessing",
            {
                "feature_channels": features.channels,
                "features_shape": features.data.shape,
                "normalization_channels": normalized.channels,
                "bottom_z_start": bottom.z_start,
                "bottom_z_stop": bottom.z_stop,
                "bottom_shape": bottom.data.shape,
                "candidate_layers": bottom.candidate_layers,
                "patch_shape": patch.data.shape,
                "patch_out_hw": patch.out_hw,
            },
        )

    with log_stage(logger, "load_model"):
        loaded = load_model_for_inference(
            checkpoint_path,
            device=config.device,
            in_channels=config.model_in_channels,
            base=config.model_base,
            out_channels=config.model_out_channels,
        )
        requires_finetuning = bool(loaded.metadata.get("requires_finetuning") is True)
        log_values(
            logger,
            "model",
            {
                "checkpoint_path": loaded.checkpoint_path,
                "device": loaded.device,
                "model_in_channels": config.model_in_channels,
                "model_base": config.model_base,
                "model_out_channels": config.model_out_channels,
                "checkpoint_requires_finetuning": requires_finetuning,
                "checkpoint_metadata": loaded.metadata,
            },
        )

    with log_stage(logger, "inference"):
        probability_patch = predict_patch(patch.data, loaded)
        restored = restore_prediction_to_full(
            probability_patch,
            patch,
            bottom,
            tuple(int(v) for v in reference_img.shape[:3]),
            threshold=config.threshold,
        )
        log_values(
            logger,
            "prediction",
            {
                "probability_patch_shape": restored.probability_patch.shape,
                "probability_full_shape": restored.probability_full.shape,
                "mask_voxels": int(restored.mask_full.sum()),
                "threshold": restored.threshold,
            },
        )

    with log_stage(logger, "save_outputs"):
        prob_path = save_probability_like(
            restored.probability_full,
            reference_img,
            out_dir / "probabilities" / f"{output_id}_csf_prob.nii.gz",
        )
        mask_path = save_mask_like(
            restored.mask_full,
            reference_img,
            out_dir / "masks" / f"{output_id}_csf_mask_bottom10.nii.gz",
        )
        log_values(
            logger,
            "nifti_outputs",
            {
                "probability_path": prob_path,
                "mask_path": mask_path,
            },
        )

    with log_stage(logger, "prediction_qc"):
        component_analysis = analyze_mask_layers(
            restored.mask_bottom,
            layers=bottom.candidate_layers,
            min_valid_voxels=config.min_valid_voxels,
        )
        prediction_qc_path = plot_prediction_qc(
            bottom.data[0],
            restored.probability_bottom,
            restored.mask_bottom,
            out_dir / "qc" / f"{output_id}_prediction_qc.png",
            layers=bottom.candidate_layers,
            component_analysis=component_analysis,
        )
        auto_layer = normalize_layer_name(component_analysis.auto_layer)
        layer_stats = {
            stats.layer_name: {
                "mask_voxels": stats.mask_voxels,
                "isolated_voxels": stats.isolated_voxels,
                "valid_voxels": stats.valid_voxels,
                "is_candidate_layer": stats.is_candidate_layer,
                "is_auto_layer": stats.is_auto_layer,
            }
            for stats in component_analysis.stats
        }
        log_values(
            logger,
            "layer_analysis",
            {
                "auto_layer": auto_layer or "none",
                "min_valid_voxels": component_analysis.min_valid_voxels,
                "layer_stats": layer_stats,
                "prediction_qc_path": prediction_qc_path,
            },
        )

    with log_stage(logger, "extract_timeseries"):
        voxel_table = extract_voxel_table(
            fmri_data,
            component_analysis,
            output_id,
            z_start=bottom.z_start,
        )
        voxel_table_path = write_voxel_table(
            voxel_table,
            out_dir / "timeseries" / f"{output_id}_voxel_table.csv",
        )
        template_path = update_selection_template(
            out_dir / "reports" / "selection_template.csv",
            output_id=output_id,
            selected_layer=auto_layer,
            min_valid_voxels=config.min_valid_voxels,
            exclude=False,
            notes="" if auto_layer else "no_valid_layer",
        )
        log_values(
            logger,
            "timeseries",
            {
                "voxel_table_rows": len(voxel_table),
                "voxel_table_path": voxel_table_path,
                "selection_template_path": template_path,
            },
        )

    selected_path = None
    timeseries_qc_path = None
    auto_summary_path = None
    selected_voxels = 0
    with log_stage(logger, "auto_selection"):
        if auto_layer:
            selection = select_voxels(
                voxel_table,
                output_id=output_id,
                selected_layer=auto_layer,
                min_valid_voxels=config.min_valid_voxels,
                selection_name="auto",
                exclude=False,
                dropout_warning_fraction=config.dropout_warning_fraction,
            )
            selected_dir = out_dir / "selected" / "auto"
            selected_path = write_voxel_table(
                selection.selected,
                selected_dir / f"{output_id}_{auto_layer}_auto_selected_voxels.csv",
            )
            timeseries_qc_path = plot_timeseries_qc(
                selection.layer_table,
                selected_dir / f"{output_id}_{auto_layer}_auto_timeseries_qc.png",
                output_id=output_id,
                selected_layer=auto_layer,
                selection_name="auto",
                min_valid_voxels=config.min_valid_voxels,
                dropout_warning_fraction=config.dropout_warning_fraction,
            )
            auto_summary_path = update_summary_table(
                out_dir / "reports" / "auto_selection_summary.csv",
                [selection.summary],
            )
            selected_voxels = int(selection.summary.get("selected_voxels", len(selection.selected)))
            log_values(
                logger,
                "auto_selection",
                {
                    "selected_voxels_path": selected_path,
                    "timeseries_qc_path": timeseries_qc_path,
                    "auto_summary_path": auto_summary_path,
                    "selected_voxels": selected_voxels,
                    "selection_summary": selection.summary,
                },
            )
        else:
            log_values(
                logger,
                "auto_selection",
                {
                    "auto_layer": "none",
                    "selected_voxels": 0,
                    "message": "No valid auto layer; selected voxel export was skipped.",
                },
            )

    elapsed = time.perf_counter() - started
    result = SingleRunResult(
        output_id=output_id,
        status="ok",
        input_path=input_path,
        out_dir=out_dir,
        checkpoint_path=loaded.checkpoint_path,
        device=str(loaded.device),
        log_path=resolved_log_path,
        probability_path=prob_path,
        mask_path=mask_path,
        prediction_qc_path=prediction_qc_path,
        voxel_table_path=voxel_table_path,
        selection_template_path=template_path,
        selected_voxels_path=selected_path,
        timeseries_qc_path=timeseries_qc_path,
        auto_summary_path=auto_summary_path,
        auto_layer=auto_layer,
        mask_voxels=int(restored.mask_full.sum()),
        selected_voxels=selected_voxels,
        patch_shape=tuple(int(v) for v in patch.output_shape),
        probability_patch_shape=tuple(int(v) for v in restored.probability_patch.shape),
        probability_full_shape=tuple(int(v) for v in restored.probability_full.shape),
        checkpoint_requires_finetuning=requires_finetuning,
        elapsed_seconds=elapsed,
    )
    log_values(logger, "result", _result_log_values(result))
    return result


def _nifti_log_values(input_path: Path, reference_img: Any, fmri_data: Any) -> dict[str, Any]:
    header = reference_img.header
    zooms = tuple(float(value) for value in header.get_zooms())
    return {
        "input_path": input_path,
        "input_file_size_bytes": input_path.stat().st_size if input_path.exists() else "",
        "input_shape": tuple(int(value) for value in reference_img.shape),
        "loaded_array_shape": tuple(int(value) for value in fmri_data.shape),
        "dtype": str(header.get_data_dtype()),
        "voxel_size": zooms[: min(3, len(zooms))],
        "tr": zooms[3] if len(zooms) >= 4 and len(reference_img.shape) >= 4 else "",
        "qform_code": int(header["qform_code"]),
        "sform_code": int(header["sform_code"]),
    }


def _result_log_values(result: SingleRunResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "output_id": result.output_id,
        "auto_layer": result.auto_layer or "none",
        "mask_voxels": result.mask_voxels,
        "selected_voxels": result.selected_voxels,
        "elapsed_seconds": f"{result.elapsed_seconds:.3f}",
        "probability_path": result.probability_path,
        "mask_path": result.mask_path,
        "prediction_qc_path": result.prediction_qc_path,
        "voxel_table_path": result.voxel_table_path,
        "selection_template_path": result.selection_template_path,
        "selected_voxels_path": result.selected_voxels_path or "",
        "timeseries_qc_path": result.timeseries_qc_path or "",
        "auto_summary_path": result.auto_summary_path or "",
        "log_path": result.log_path or "",
    }


__all__ = [
    "SingleRunConfig",
    "SingleRunResult",
    "run_single",
]
