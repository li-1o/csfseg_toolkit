# Quickstart

This guide shows the shortest practical path from one preprocessed fMRI file to
CSF mask, voxel time series, and QC.

## 1. Install

```bash
conda env create -f environment.yml
conda activate csfseg
pip install -e .
```

## 2. Put A Checkpoint Somewhere

You need a trained 3-channel checkpoint, for example:

```text
/models/best_3ch.pt
```

The checkpoint can live anywhere. You point to it with `--checkpoint`.

If you only want to check that the toolkit runs on your machine, use the smoke
test first:

```bash
python examples/run_smoke_test.py --overwrite
```

That script creates fake 4D fMRI files and a random checkpoint, then runs the
real commands. It is useful for testing installation and output layout. It is
not a segmentation-quality test, and its checkpoint should not be used for
analysis.

## 3. Check A Path List

If you have many files, start with a small path list:

```text
input_paths.txt
/data/sub-001_ses-01_task-rest_run-01_bold.nii.gz
/data/sub-002_ses-01_task-rest_run-01_bold.nii.gz
```

Validate it:

```bash
csfseg check-manifest \
  --input-paths input_paths.txt \
  --out-dir /data/csfseg_outputs
```

This checks that paths exist and output names will not collide. It also writes:

```text
/data/csfseg_outputs/input_paths_used.csv
```

That file is the resolved path table the toolkit understood.

For one file, this step is optional, but it is a good habit before a large run.

## 4. Check Model Input Preparation

Before running the model, inspect a few representative files:

```bash
csfseg check-preprocessing \
  --input /data/sub-001_ses-01_task-rest_run-01_bold.nii.gz \
  --out-dir /data/csfseg_outputs
```

This does not run U-Net. It only prepares model inputs and writes QC:

```text
qc/<output_id>_preprocessing_qc.png
qc/<output_id>_bottom_qc.png
qc/<output_id>_patch_qc.png
```

Use these figures to check that mean/std/tSNR, normalization, bottom extraction,
and crop/pad look reasonable.

## 5. Run Prediction

For one file:

```bash
csfseg predict \
  --input /data/sub-001_ses-01_task-rest_run-01_bold.nii.gz \
  --out-dir /data/csfseg_outputs \
  --checkpoint /models/best_3ch.pt
```

For many files:

```bash
csfseg batch \
  --input-paths input_paths.txt \
  --out-dir /data/csfseg_outputs \
  --checkpoint /models/best_3ch.pt \
  --skip-existing
```

The first batch version runs inputs one at a time. It is still useful for large
datasets because it validates paths, continues after failed inputs, writes one
log per input, and summarizes everything in `reports/batch_report.csv`.

Prediction writes:

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

Batch mode also writes:

```text
run_config.yaml
reports/batch_report.csv
reports/failed_inputs.csv
logs/run_<run_id>.log
input_paths_used.csv
```

For a contract-bearing ADNI baseline checkpoint, prediction also writes
`masks/<output_id>_csf_mask_bottom10_raw.nii.gz`. The raw file preserves the
bottom-ten threshold result for QC; the main mask and voxel table are limited
to `L0-L2`. Checkpoints without a contract use the legacy `L0-L3` candidate
default. The auto selected table keeps one approved layer, excluding isolated
voxels and zero-dropout voxels.

## 6. Review QC And Adjust If Needed

Open:

```text
qc/<output_id>_prediction_qc.png
selected/auto/<output_id>_<layer>_auto_timeseries_qc.png
```

If the mask is roughly right but the selected layer should change, edit:

```text
reports/selection_template.csv
```

Example:

```csv
output_id,selected_layer,min_valid_voxels,exclude,notes
sub-001_ses-01_task-rest_run-01_bold,L1,5,false,use_L1_after_QC
```

Then export a new selected version:

```bash
csfseg select-voxels \
  --out-dir /data/csfseg_outputs \
  --selection-file reports/selection_template.csv \
  --selection-name manual_v1
```

New files go into:

```text
selected/manual_v1/
reports/manual_v1_selection_summary.csv
```

If the model mask itself is wrong, `select-voxels` cannot fix that. In that
case, the checkpoint or training data need to be revisited.
