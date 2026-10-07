"""Command-line entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from csfseg.batch.manifest import load_input_paths, write_input_paths_used
from csfseg.config import ensure_out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csfseg", description="CSF segmentation toolkit for preprocessed 4D fMRI.")
    sub = parser.add_subparsers(dest="command", required=True)

    check_manifest = sub.add_parser("check-manifest", help="Validate an input path file without running the model.")
    check_manifest.add_argument(
        "--input-paths",
        required=True,
        help="TXT/CSV/TSV path file. CSV/TSV use input_path and optional output_id columns.",
    )
    check_manifest.add_argument("--out-dir", required=True, help="Output directory. Required so paths can be checked early.")

    check_preprocessing = sub.add_parser(
        "check-preprocessing",
        help="Run feature generation and normalization QC for one 4D fMRI file.",
    )
    check_preprocessing.add_argument("--input", required=True, help="Preprocessed 4D fMRI NIfTI file.")
    check_preprocessing.add_argument("--out-dir", required=True, help="Output directory for the QC figure.")
    check_preprocessing.add_argument(
        "--output-id",
        help="Optional output prefix. Defaults to the input filename without .nii/.nii.gz.",
    )

    init_3ch = sub.add_parser(
        "init-3ch-checkpoint",
        help="Create a 3-channel initialization checkpoint from a 1-channel checkpoint.",
    )
    init_3ch.add_argument("--source-checkpoint", required=True, help="Old 1-channel .pt checkpoint.")
    init_3ch.add_argument("--out-checkpoint", required=True, help="Where to write the new 3-channel .pt checkpoint.")
    init_3ch.add_argument("--metadata-path", help="Optional metadata JSON path. Defaults to <out-checkpoint>.json.")
    init_3ch.add_argument("--base", type=int, default=16, help="U-Net base channel count. Default: 16.")
    init_3ch.add_argument("--seed", type=int, default=0, help="Seed for std/tSNR channel initialization.")
    init_3ch.add_argument(
        "--random-scale",
        type=float,
        default=0.01,
        help="Std/tSNR random std as a fraction of old first-layer std. Default: 0.01.",
    )
    init_3ch.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output checkpoint or metadata file.",
    )

    predict = sub.add_parser(
        "predict",
        help="Run single-file CSF mask prediction for one preprocessed 4D fMRI NIfTI.",
    )
    predict.add_argument("--input", required=True, help="Preprocessed 4D fMRI NIfTI file.")
    predict.add_argument("--out-dir", required=True, help="Output directory for masks and probabilities.")
    predict.add_argument("--checkpoint", required=True, help="3-channel model checkpoint.")
    predict.add_argument(
        "--output-id",
        help="Optional output prefix. Defaults to the input filename without .nii/.nii.gz.",
    )
    predict.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    predict.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the binary mask. Default: 0.5.",
    )
    predict.add_argument(
        "--min-valid-voxels",
        type=int,
        default=5,
        help="Minimum non-isolated voxels required for a candidate layer. Default: 5.",
    )
    predict.add_argument(
        "--dropout-warning-fraction",
        type=float,
        default=0.20,
        help="Warn when this fraction of non-isolated voxels has zero dropout. Default: 0.20.",
    )

    batch = sub.add_parser(
        "batch",
        help="Run CSFSeg prediction for every input listed in a TXT/CSV/TSV path file.",
    )
    batch.add_argument("--input-paths", required=True, help="TXT/CSV/TSV path file with inputs to process.")
    batch.add_argument("--out-dir", required=True, help="Output directory for all batch outputs.")
    batch.add_argument("--checkpoint", required=True, help="3-channel model checkpoint.")
    batch.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    batch.add_argument(
        "--subject-workers",
        default="auto",
        help="How many complete input pipelines to run at once. First version supports auto or 1.",
    )
    batch.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an input when core prediction outputs already exist.",
    )
    batch.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the binary mask. Default: 0.5.",
    )
    batch.add_argument(
        "--min-valid-voxels",
        type=int,
        default=5,
        help="Minimum non-isolated voxels required for a candidate layer. Default: 5.",
    )
    batch.add_argument(
        "--dropout-warning-fraction",
        type=float,
        default=0.20,
        help="Warn when this fraction of non-isolated voxels has zero dropout. Default: 0.20.",
    )

    select = sub.add_parser(
        "select-voxels",
        help="Export selected voxel time series from existing voxel tables without rerunning the model.",
    )
    select.add_argument("--out-dir", required=True, help="Existing CSFSeg output directory.")
    select.add_argument(
        "--selection-file",
        help="CSV with output_id, selected_layer, min_valid_voxels, exclude, and notes columns.",
    )
    select.add_argument(
        "--selection-name",
        default="manual",
        help="Name for this selected export version. Default: manual.",
    )
    select.add_argument("--output-id", help="Single output id to export when no selection file is used.")
    select.add_argument("--layer", help="Single selected layer, such as L0 or L1.")
    select.add_argument(
        "--min-valid-voxels",
        type=int,
        default=5,
        help="Minimum non-isolated voxels used for warnings in selected export. Default: 5.",
    )
    select.add_argument(
        "--dropout-warning-fraction",
        type=float,
        default=0.20,
        help="Warn when this fraction of non-isolated voxels has zero dropout. Default: 0.20.",
    )
    select.add_argument(
        "--exclude",
        action="store_true",
        help="For single-output mode, mark this output as manually excluded.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check-manifest":
        out_dir = ensure_out_dir(args.out_dir)
        records = load_input_paths(args.input_paths)
        used_path = write_input_paths_used(records, out_dir / "input_paths_used.csv")
        print(f"OK: {len(records)} input file(s) validated.")
        print(f"Wrote resolved input table: {used_path}")
        return 0

    if args.command == "check-preprocessing":
        return _run_check_preprocessing(args)

    if args.command == "init-3ch-checkpoint":
        return _run_init_3ch_checkpoint(args)

    if args.command == "predict":
        return _run_predict(args)

    if args.command == "batch":
        return _run_batch(args)

    if args.command == "select-voxels":
        return _run_select_voxels(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_check_preprocessing(args: argparse.Namespace) -> int:
    """Run the preprocessing sanity check command.

    Heavy neuroimaging dependencies are imported here so lightweight commands
    such as ``check-manifest`` do not need to import nibabel/numpy at startup.
    """

    import numpy as np

    from csfseg.batch.manifest import strip_nifti_suffix
    from csfseg.io.nifti import load_nifti
    from csfseg.preprocessing.bottom import extract_bottom_volume
    from csfseg.preprocessing.features import compute_temporal_features
    from csfseg.preprocessing.normalize import normalize_features
    from csfseg.preprocessing.patch import make_patch
    from csfseg.qc.plots import (
        choose_preprocessing_qc_slice,
        choose_raw_time_indices,
        plot_bottom_volume_qc,
        plot_patch_qc,
        plot_preprocessing_qc,
    )

    input_path = Path(args.input).expanduser().resolve()
    out_dir = ensure_out_dir(args.out_dir)
    output_id = args.output_id or strip_nifti_suffix(input_path)

    img = load_nifti(input_path)
    fmri_data = np.asanyarray(img.dataobj, dtype=np.float32)
    features = compute_temporal_features(fmri_data)
    normalized = normalize_features(features.data, features.channels)
    bottom = extract_bottom_volume(normalized.data)
    patch = make_patch(bottom.data)
    raw_time_indices = choose_raw_time_indices(fmri_data.shape[-1])
    qc_slice_index = choose_preprocessing_qc_slice(features.data)

    preprocessing_figure_path = plot_preprocessing_qc(
        fmri_data,
        features.data,
        normalized.data,
        features.channels,
        out_dir / "qc" / f"{output_id}_preprocessing_qc.png",
        slice_index=qc_slice_index,
        raw_time_indices=raw_time_indices,
    )
    bottom_figure_path = plot_bottom_volume_qc(
        bottom.data,
        normalized.channels,
        out_dir / "qc" / f"{output_id}_bottom_qc.png",
        source_z_start=bottom.z_start,
        candidate_layers=bottom.candidate_layers,
    )
    patch_figure_path = plot_patch_qc(
        bottom.data,
        patch.data,
        normalized.channels,
        out_dir / "qc" / f"{output_id}_patch_qc.png",
        candidate_layers=bottom.candidate_layers,
    )

    print(f"QC files written for {input_path}")
    print(f"Wrote preprocessing figure: {preprocessing_figure_path}")
    print(f"Wrote bottom figure: {bottom_figure_path}")
    print(f"Wrote patch figure: {patch_figure_path}")
    print(f"Raw time indices shown: {raw_time_indices}")
    print(f"Axial slice shown: z={qc_slice_index}")
    print(f"Bottom source z range: {bottom.z_start}:{bottom.z_stop}")
    print(f"Patch output shape: {patch.output_shape}")

    return 0


def _run_init_3ch_checkpoint(args: argparse.Namespace) -> int:
    """Create a 3-channel initialization checkpoint for later finetuning."""

    from csfseg.inference.checkpoint import initialize_3ch_from_1ch_checkpoint

    out_checkpoint = Path(args.out_checkpoint).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_path).expanduser().resolve()
        if args.metadata_path
        else out_checkpoint.with_suffix(out_checkpoint.suffix + ".json")
    )

    if not args.overwrite:
        existing = [path for path in (out_checkpoint, metadata_path) if path.exists()]
        if existing:
            paths = ", ".join(str(path) for path in existing)
            print(f"Refusing to overwrite existing file(s): {paths}", file=sys.stderr)
            print("Pass --overwrite if you intentionally want to replace them.", file=sys.stderr)
            return 2

    result = initialize_3ch_from_1ch_checkpoint(
        args.source_checkpoint,
        out_checkpoint,
        metadata_path=metadata_path,
        base=args.base,
        seed=args.seed,
        random_scale=args.random_scale,
    )

    print(f"Wrote 3-channel initialization checkpoint: {result.checkpoint_path}")
    print(f"Wrote checkpoint metadata: {result.metadata_path}")
    print(f"First conv shape: {result.metadata['first_conv_shape']}")
    print(f"Channel order: {', '.join(result.metadata['channel_order'])}")
    print(f"Initialization strategy: {result.metadata['init_strategy']}")
    print("This checkpoint is an initialization for finetuning, not a final model.")
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    """Run the single-file prediction pipeline through ``pipeline.run_single``."""

    from csfseg.pipeline import SingleRunConfig, run_single
    from csfseg.utils.logging import close_logger, setup_subject_logger

    config = SingleRunConfig(
        input_path=args.input,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        output_id=args.output_id,
        device=args.device,
        threshold=args.threshold,
        min_valid_voxels=args.min_valid_voxels,
        dropout_warning_fraction=args.dropout_warning_fraction,
    )
    output_id = config.resolved_output_id()
    logger, log_path, run_id = setup_subject_logger(output_id, config.out_dir)

    try:
        result = run_single(
            config,
            logger=logger,
            log_path=log_path,
            run_id=run_id,
        )
    finally:
        close_logger(logger)

    if result.checkpoint_requires_finetuning:
        print(
            "WARNING: checkpoint metadata says requires_finetuning=true. "
            "Use this output for pipeline testing, not final analysis.",
            file=sys.stderr,
        )

    print(f"Prediction complete for {result.input_path}")
    print(f"Device: {result.device}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Patch shape: {result.patch_shape}")
    print(f"Probability patch shape: {result.probability_patch_shape}")
    print(f"Full probability shape: {result.probability_full_shape}")
    print(f"Mask voxels: {result.mask_voxels}")
    print(f"Auto selected layer: {result.auto_layer or 'none'}")
    print(f"Wrote probability map: {result.probability_path}")
    print(f"Wrote bottom10 mask: {result.mask_path}")
    print(f"Wrote prediction QC: {result.prediction_qc_path}")
    print(f"Wrote voxel table: {result.voxel_table_path}")
    print(f"Wrote selection template: {result.selection_template_path}")
    if result.selected_voxels_path:
        print(f"Wrote auto selected voxels: {result.selected_voxels_path}")
        print(f"Wrote auto timeseries QC: {result.timeseries_qc_path}")
        print(f"Wrote auto selection summary: {result.auto_summary_path}")
    else:
        print("No valid auto layer; selected voxel export was skipped.")
    print(f"Wrote detailed log: {result.log_path}")
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    """Run the manifest-driven batch pipeline."""

    from csfseg.batch.runner import BatchItemResult, BatchRunConfig, run_batch

    def progress(index: int, total: int, row: BatchItemResult) -> None:
        if row.status == "ok":
            layer = row.auto_layer or "none"
            voxels = "" if row.selected_voxels is None else row.selected_voxels
            print(
                f"[{index}/{total}] {row.output_id} OK "
                f"layer={layer} voxels={voxels} time={row.elapsed_seconds:.1f}s"
            )
        elif row.status == "skipped_existing":
            print(f"[{index}/{total}] {row.output_id} SKIPPED existing")
        else:
            print(
                f"[{index}/{total}] {row.output_id} FAILED "
                f"stage={row.stage} log={row.log_path}",
                file=sys.stderr,
            )

    config = BatchRunConfig(
        input_paths=args.input_paths,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        min_valid_voxels=args.min_valid_voxels,
        dropout_warning_fraction=args.dropout_warning_fraction,
        subject_workers=args.subject_workers,
        skip_existing=args.skip_existing,
    )

    try:
        result = run_batch(config, progress_callback=progress)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Batch complete: status={result.status} total={result.total} "
        f"ok={result.ok} failed={result.failed} skipped={result.skipped}"
    )
    print(f"Run id: {result.run_id}")
    print(f"Wrote resolved input table: {result.input_paths_used_path}")
    print(f"Wrote batch report: {result.batch_report_path}")
    print(f"Wrote failed-input report: {result.failed_inputs_path}")
    print(f"Wrote run config: {result.run_config_path}")
    print(f"Wrote run log: {result.run_log_path}")
    return 1 if result.failed else 0


def _run_select_voxels(args: argparse.Namespace) -> int:
    """Export selected voxel tables from existing full voxel tables."""

    import pandas as pd

    from csfseg.timeseries.components import normalize_layer_name
    from csfseg.timeseries.extract import (
        read_selection_file,
        read_voxel_table,
        select_voxels,
        update_summary_table,
        write_voxel_table,
    )
    from csfseg.qc.plots import plot_timeseries_qc

    out_dir = ensure_out_dir(args.out_dir)
    selection_name = _safe_name(args.selection_name)

    if args.selection_file:
        selection_file = Path(args.selection_file).expanduser()
        if not selection_file.is_absolute() and not selection_file.exists():
            selection_file = out_dir / selection_file
        selection_rows = read_selection_file(
            selection_file,
            default_min_valid_voxels=args.min_valid_voxels,
        )
    else:
        if not args.output_id or args.layer is None:
            print(
                "select-voxels needs either --selection-file or both --output-id and --layer.",
                file=sys.stderr,
            )
            return 2
        selection_rows = pd.DataFrame(
            [
                {
                    "output_id": args.output_id,
                    "selected_layer": normalize_layer_name(args.layer),
                    "min_valid_voxels": int(args.min_valid_voxels),
                    "exclude": bool(args.exclude),
                    "notes": "",
                }
            ]
        )

    summaries: list[dict[str, object]] = []
    selected_dir = out_dir / "selected" / selection_name
    failures = 0

    for row in selection_rows.to_dict(orient="records"):
        output_id = str(row["output_id"])
        selected_layer = normalize_layer_name(row.get("selected_layer"))
        min_valid_voxels = int(row.get("min_valid_voxels", args.min_valid_voxels))
        exclude = _parse_bool(row.get("exclude", False))

        voxel_table_path = out_dir / "timeseries" / f"{output_id}_voxel_table.csv"
        try:
            voxel_table = read_voxel_table(voxel_table_path)
            selection = select_voxels(
                voxel_table,
                output_id=output_id,
                selected_layer=selected_layer,
                min_valid_voxels=min_valid_voxels,
                selection_name=selection_name,
                exclude=exclude,
                dropout_warning_fraction=args.dropout_warning_fraction,
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI use
            failures += 1
            summaries.append(
                {
                    "output_id": output_id,
                    "selection_name": selection_name,
                    "selected_layer": selected_layer,
                    "min_valid_voxels": min_valid_voxels,
                    "exclude": exclude,
                    "status": "error",
                    "warning": str(exc),
                }
            )
            print(f"ERROR {output_id}: {exc}", file=sys.stderr)
            continue

        summaries.append(selection.summary)
        if selection.summary["status"] in {"excluded_by_user", "no_selected_layer"}:
            print(f"Skipped {output_id}: {selection.summary['status']}")
            continue

        layer = selected_layer or "none"
        selected_path = write_voxel_table(
            selection.selected,
            selected_dir / f"{output_id}_{layer}_{selection_name}_selected_voxels.csv",
        )
        qc_path = None
        if not selection.layer_table.empty:
            qc_path = plot_timeseries_qc(
                selection.layer_table,
                selected_dir / f"{output_id}_{layer}_{selection_name}_timeseries_qc.png",
                output_id=output_id,
                selected_layer=layer,
                selection_name=selection_name,
                min_valid_voxels=min_valid_voxels,
                dropout_warning_fraction=args.dropout_warning_fraction,
            )
        print(f"Selected {output_id}: {selected_path}")
        if qc_path:
            print(f"Wrote timeseries QC: {qc_path}")

    summary_path = update_summary_table(
        out_dir / "reports" / f"{selection_name}_selection_summary.csv",
        summaries,
    )
    print(f"Wrote selection summary: {summary_path}")
    return 1 if failures else 0


def _safe_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "manual"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


if __name__ == "__main__":
    sys.exit(main())
