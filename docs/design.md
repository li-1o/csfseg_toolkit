# Design Notes

This toolkit focuses on CSF mask prediction and CSF time-series extraction from preprocessed 4D fMRI.

The boundary is deliberate:

- input: preprocessed 4D fMRI NIfTI
- output: CSF mask, CSF time series, QC, logs, reports
- not included: HCP preprocessing, FSL motion correction, gradient unwarping, group analysis

The main single-file flow is:

```text
4D fMRI
→ model input preprocessing
→ 3D U-Net prediction
→ mask postprocessing and saving
→ CSF time-series extraction
→ QC plots and reports
```

The batch flow is built on top of the same single-file pipeline. Batch code
should not duplicate segmentation logic. It should only decide what to run,
what to skip, how many subject pipelines to start, and how to report results.

Current code organization:

```text
cli.py
  parses commands and prints short terminal summaries

pipeline.py
  defines SingleRunConfig, SingleRunResult, and run_single()
  run_single() processes one 4D fMRI input from loading through outputs

batch/runner.py
  defines BatchRunConfig, BatchRunResult, and run_batch()
  calls pipeline.run_single() once per manifest row
  writes batch reports, run config, and run-level logs
```

## Input Philosophy

Different users store data in different ways. Some use BIDS. Some use project-specific folders. Some put session, run, task, and date in the filename.

The toolkit should not force a directory layout. The first version only asks for an input path file:

```text
/data/sub-001_ses-01_task-rest_run-01_bold.nii.gz
```

If output names would collide, users can switch to CSV or TSV and add `output_id`.

## Model Input

The default input layout uses three channels:

```text
mean
std
tsnr
```

These channels are computed from the 4D fMRI time axis before the data goes
into the model:

- `mean`: average signal over time, one 3D image
- `std`: temporal standard deviation, one 3D image
- `tsnr`: temporal mean divided by temporal standard deviation

The feature stack is channel-first: `(C, X, Y, Z)`. For the current design
target, `C = 3`.

After feature generation, each channel is normalized with nonzero z-score. In
plain terms: for one channel, use finite nonzero voxels to compute the mean and
standard deviation, convert values to `(value - mean) / std`, and keep
background voxels at zero. This avoids letting empty image space dominate the
normalization. The same rule must be used during training, finetuning, and
inference.

## Bottom Volume

The CSF signal targeted by this toolkit follows the bottom-slice inflow idea:
CSF near the bottom edge of the fMRI acquisition can show strong inflow-related
signal changes. The model therefore should not see the whole brain volume. It
should see a thin volume starting at the bottom of the image.

For the first version, the rule is deliberately simple:

- assume array `z=0` is the acquisition bottom
- take `z=0:10` as the model input depth
- keep the z direction as part of the 3D U-Net input, because the vertical
  structure carries useful information
- after prediction, keep the full bottom 10-layer output
- for QC and later final-layer selection, first inspect low layers `L0-L3`
- choose the final signal layer later from the model prediction, rather than
  assuming that `L0` must contain usable ROI
- rely on QC figures so users can catch cases where `z=0` is not the true
  bottom or has too few useful voxels

The training target should let the model learn whether the usable ROI starts at
L0 or shifts upward to L1 in subjects where the bottom slice is empty or weak.

## Patch

After bottom extraction, the data layout is:

```text
(C, D, X, Y)
```

where `D` is the 10-slice bottom depth.  Before the data enters the 3D U-Net,
the toolkit center crops or zero pads only the X/Y axes to `128x128`:

```text
(C, D, X, Y) -> (C, D, 128, 128)
```

This step does not resize and does not interpolate.  The z depth stays
unchanged because the vertical bottom structure is part of what the model is
expected to learn.  The crop/pad metadata is kept in memory so a future model
prediction can be placed back into the original X/Y grid before saving the mask
and extracting the time series.

## Prediction Restore

The model predicts in patch space:

```text
(D, 128, 128)
```

That output is restored in two steps:

```text
patch space          -> bottom image space -> full NIfTI space
(D, 128, 128)        -> (D, X, Y)          -> (X, Y, Z)
using PatchVolume    using BottomVolume
```

For the first version, `BottomVolume.z_start` is `0`, so the restored prediction
is placed into `z=0:10`.  Voxels outside the bottom block are zero because the
model did not see them.

## Prediction QC And Extraction

After prediction, the toolkit keeps the complete bottom10 mask as the main mask
output. It does not create a separate single-layer final mask by default. Layer
choice is represented as metadata and as a selected voxel table.

The post-prediction flow is:

```text
bottom10 mask
→ inspect L0-L3
→ label 2D connected components in each layer
→ tag isolated voxels where component_size == 1
→ suggest the first candidate layer, preferring L0 then L1
→ extract every L0-L3 mask voxel time series into voxel_table.csv
→ export an auto selected subset
```

A candidate layer is one where, after removing isolated voxels, at least
`min_valid_voxels` remain. The first version defaults to `5`.

The full voxel table keeps everything: candidate and non-candidate layers,
isolated voxels, and zero-dropout voxels. The selected table is a filtered
subset intended for analysis:

```text
layer == selected_layer
is_isolated == false
has_zero_dropout == false
```

Users can later edit `reports/selection_template.csv` to choose another layer,
raise or lower `min_valid_voxels`, or manually exclude a run. Re-exporting
selected voxels does not rerun the model.

## Logging

Detailed logs are written by default for prediction. Users should not need to
remember a debug flag just to find out why a subject failed.

For single-file prediction, each input gets:

```text
logs/subjects/<output_id>.log
```

The log identifies the input, output id, checkpoint, command settings, major
pipeline stages, key shapes, output paths, and traceback if a stage fails.
Terminal output can stay short. File logs keep the full story.

For batch prediction, the toolkit adds:

```text
logs/run_<run_id>.log
logs/subjects/<output_id>.log
reports/batch_report.csv
reports/failed_inputs.csv
run_config.yaml
```

The run log identifies the whole batch. The subject log identifies one input.
This separation matters when hundreds of files are processed and only a few
fail.
