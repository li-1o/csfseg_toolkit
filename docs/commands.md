# Commands

This page explains the current command-line interface. The main user commands
are `check-manifest`, `check-preprocessing`, `predict`, `batch`, and
`select-voxels`.

## `csfseg check-manifest`

Use this before a large run to catch path-table mistakes.

Example:

```bash
csfseg check-manifest \
  --input-paths input_paths.txt \
  --out-dir /data/csfseg_outputs
```

It checks:

- the input file can be read as TXT, CSV, or TSV
- input files exist
- CSV/TSV tables have an `input_path` column
- optional `output_id` values are not empty
- final output ids are unique
- output ids do not contain `/` or `\`

It writes:

```text
/data/csfseg_outputs/input_paths_used.csv
```

It does not read NIfTI image data, run preprocessing, run the model, or make a
mask.

## `csfseg check-preprocessing`

Use this to inspect what will enter the model.

Example:

```bash
csfseg check-preprocessing \
  --input /data/sub-001_ses-01_task-rest_run-01_bold.nii.gz \
  --out-dir /data/csfseg_outputs
```

It does:

- read the 4D fMRI NIfTI
- compute temporal mean, temporal std, and tSNR
- normalize each channel with nonzero z-score
- take bottom `z=0:10`
- crop or pad X/Y to `128x128`
- write preprocessing QC figures

It writes:

```text
qc/<output_id>_preprocessing_qc.png
qc/<output_id>_bottom_qc.png
qc/<output_id>_patch_qc.png
```

It does not run U-Net and does not need a checkpoint.

## `csfseg predict`

Use this to run the full single-file workflow.

Example:

```bash
csfseg predict \
  --input /data/sub-001_ses-01_task-rest_run-01_bold.nii.gz \
  --out-dir /data/csfseg_outputs \
  --checkpoint /models/best_3ch.pt
```

It does:

- prepare the three model input channels
- run the 3D U-Net
- threshold the probability map, default `0.5`
- restore the bottom10 prediction into the input NIfTI space
- save probability and mask NIfTI files
- make prediction QC
- extract all L0-L3 mask voxel time series
- automatically choose a candidate layer
- export an auto selected voxel table
- make selected time-series QC

Important options:

```text
--threshold 0.5
--min-valid-voxels 5
--dropout-warning-fraction 0.20
--device auto
```

It writes:

```text
probabilities/<output_id>_csf_prob.nii.gz
masks/<output_id>_csf_mask_bottom10.nii.gz
qc/<output_id>_prediction_qc.png
timeseries/<output_id>_voxel_table.csv
selected/auto/<output_id>_<layer>_auto_selected_voxels.csv
selected/auto/<output_id>_<layer>_auto_timeseries_qc.png
reports/selection_template.csv
reports/auto_selection_summary.csv
logs/subjects/<output_id>.log
```

Use `predict` when you want to test or rerun one input by itself. Use `batch`
when you already have an input path file and want reports across many inputs.

## `csfseg batch`

Use this to run the same prediction workflow for every row in a TXT/CSV/TSV
path file.

Example:

```bash
csfseg batch \
  --input-paths input_paths.txt \
  --out-dir /data/csfseg_outputs \
  --checkpoint /models/best_3ch.pt \
  --device auto \
  --skip-existing
```

It does:

- validate the path file before starting
- write `input_paths_used.csv`
- run the same one-input workflow as `csfseg predict`
- write one subject log per input
- continue to the next input if one input fails
- write batch-level reports

Important options:

```text
--subject-workers auto
--skip-existing
--threshold 0.5
--min-valid-voxels 5
--dropout-warning-fraction 0.20
--device auto
```

The first batch version is deliberately serial. `--subject-workers auto` means
`1`, and `--subject-workers 1` is the only explicit worker count accepted for
now. This keeps GPU use, disk reads, memory use, and error logs simple while
the main workflow is still being validated.

It writes:

```text
input_paths_used.csv
run_config.yaml
reports/batch_report.csv
reports/failed_inputs.csv
logs/run_<run_id>.log
logs/subjects/<output_id>.log
```

Each successful input also gets the normal prediction outputs:

```text
probabilities/<output_id>_csf_prob.nii.gz
masks/<output_id>_csf_mask_bottom10.nii.gz
qc/<output_id>_prediction_qc.png
timeseries/<output_id>_voxel_table.csv
selected/auto/<output_id>_<layer>_auto_selected_voxels.csv
selected/auto/<output_id>_<layer>_auto_timeseries_qc.png
```

`--skip-existing` is useful when a run was interrupted. If the core prediction
outputs for an input already exist, batch mode records that input as
`skipped_existing` and moves on.

## `csfseg select-voxels`

Use this after reviewing QC when you want to re-export selected voxels without
rerunning the model.

Example with a selection file:

```bash
csfseg select-voxels \
  --out-dir /data/csfseg_outputs \
  --selection-file reports/selection_template.csv \
  --selection-name manual_v1
```

The selection file should contain:

```csv
output_id,selected_layer,min_valid_voxels,exclude,notes
sub-001_ses-01_task-rest_run-01_bold,L1,5,false,use_L1_after_QC
sub-002_ses-01_task-rest_run-01_bold,L0,5,true,bad_motion
```

It reads existing files:

```text
timeseries/<output_id>_voxel_table.csv
```

It writes:

```text
selected/<selection_name>/<output_id>_<layer>_<selection_name>_selected_voxels.csv
selected/<selection_name>/<output_id>_<layer>_<selection_name>_timeseries_qc.png
reports/<selection_name>_selection_summary.csv
```

It can change the selected layer or exclude a run. It cannot fix a wrong model
mask.

## `csfseg init-3ch-checkpoint`

This is a development and finetuning helper, not a normal analysis command.

It creates a 3-channel initialization checkpoint from a compatible 1-channel
checkpoint. The output is a warm start for finetuning and should not be used as
a final analysis model unless it is later trained and released as such.
