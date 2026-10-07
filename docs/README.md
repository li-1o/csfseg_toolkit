# Documentation Map

Use this page as the table of contents for CSFSeg Toolkit.

## Start Here

- [quickstart.md](quickstart.md) walks through a first run: prepare paths, check inputs, run
  prediction, inspect QC, and re-export selected voxels if needed.
- [commands.md](commands.md) explains what each `csfseg` command does, what it writes, and
  what it does not do.

## User Guides

- [input_format.md](input_format.md) explains TXT/CSV/TSV input path files, the required
  `input_path` column, optional `output_id`, duplicate-name checks, and
  `input_paths_used.csv`.
- [outputs.md](outputs.md) explains masks, probability maps, voxel tables, selected voxel
  tables, QC figures, and summary tables.
- [qc.md](qc.md) explains how to read preprocessing QC, bottom QC, patch QC,
  prediction QC, and time-series carpet QC.
- [model_checkpoints.md](model_checkpoints.md) explains checkpoint files, sidecar metadata, and how
  to tell whether a checkpoint matches the 3-channel model input.
- [batch_processing.md](batch_processing.md) explains the first batch command, reports,
  logs, `--skip-existing`, and why the current batch version stays serial.

## Design And Planning

- [design.md](design.md) explains the pipeline design: three-channel input, bottom-volume
  extraction, crop/pad patching, prediction restore, layer selection, and
  voxel-level time-series extraction.
