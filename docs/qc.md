# QC

QC should help users answer a simple question: does this output look trustworthy enough to use?

The first QC figure should include:

- low-layer mask overlay
- isolated voxel count
- automatically selected layer
- voxel carpet plot for selected exports
- dropout warnings

CSV summaries should store the same important numbers so a large batch can be
filtered without opening every PNG.

QC should not be an afterthought. In a 700-subject run, it is the only practical way to find strange cases quickly.

For `csfseg predict`, each input also gets a detailed log:

```text
logs/subjects/<output_id>.log
```

The log records which input was used, which checkpoint was loaded, the major
pipeline stages, key shapes and parameters, output paths, and full error details
if something fails.

## Preprocessing QC

Before the model is connected, we can already check the model input
preprocessing:

```bash
csfseg check-preprocessing \
  --input /data/sub-001_ses-01_task-rest_run-01_bold.nii.gz \
  --out-dir /data/csfseg_outputs
```

This command computes the three current channels, normalizes them, extracts the
bottom model-input volume, creates the fixed 128x128 model patch, and writes
three figures:

```text
csfseg_outputs/qc/sub-001_ses-01_task-rest_run-01_bold_preprocessing_qc.png
csfseg_outputs/qc/sub-001_ses-01_task-rest_run-01_bold_bottom_qc.png
csfseg_outputs/qc/sub-001_ses-01_task-rest_run-01_bold_patch_qc.png
```

These are the first QC figures in the pipeline, before the model runs. They are
deliberately visual: the user looks at the images and decides whether the raw
input, feature generation, normalization, bottom extraction, and patching look
reasonable.

The preprocessing QC layout is:

```text
row 1: raw t=0        raw t=T//2       raw t=T-1
row 2: mean feature   std feature      tsnr feature
row 3: mean z-score   std z-score      tsnr z-score
```

The raw row gives a quick look at the preprocessed 4D input. The feature row
shows what the model-input feature generation produced. The normalized row
shows whether the same rough anatomy is still visible after scaling.

All panels use one axial slice. The slice is chosen from the `mean` feature:
the toolkit picks the z-slice with the most nonzero voxels, because that slice
usually gives the clearest brain/background outline.

One small nuance: after z-score, a few originally nonzero voxels can become
exactly zero if their value is exactly the channel mean. That is fine. What
would be suspicious is a large region disappearing, zero areas becoming nonzero,
or a normalized channel looking visually strange.

The bottom QC layout is:

```text
row 1: mean z-score, local L0-L9
row 2: std z-score,  local L0-L9
row 3: tsnr z-score, local L0-L9
```

This is the actual bottom volume that will be sent to the model. Low layers
Candidate layers are marked with `*`. A contract-bearing ADNI baseline
checkpoint uses `L0-L2`; the older checkpoint-free preprocessing view uses the
legacy `L0-L3` default.

The patch QC layout is:

```text
row 1: mean z-score, L0-L3 before/after patch
row 2: std z-score,  L0-L3 before/after patch
row 3: tsnr z-score, L0-L3 before/after patch
```

This figure answers a narrow but important question: did the center crop/pad to
`128x128` keep the useful bottom region in view?  If the input image is smaller
than 128 in X/Y, the after-patch panels should show zero padding around the
centered data.  If the input image is larger than 128, the after-patch panels
show the centered crop that the model will actually receive.

## Prediction QC

After `csfseg predict`, the toolkit also writes:

```text
csfseg_outputs/qc/sub-001_ses-01_task-rest_run-01_bold_prediction_qc.png
```

Prediction QC uses the checkpoint-approved layers (`L0-L2` for the ADNI
baseline; legacy default `L0-L3` when no contract is present):

```text
row 1: normalized mean image, L0 L1 L2 L3
row 2: mask overlay,          L0 L1 L2 L3
```

The binary mask uses the prediction threshold set in the command, default
`0.5`. Probability is saved as NIfTI, but the QC figure focuses on the mask
because this is where the user judges the usable ROI. The overlay title shows:

```text
n = all mask voxels
valid = non-isolated voxels
iso = isolated voxels
AUTO = automatically selected layer
```

Colors:

```text
red = non-isolated mask voxel
cyan = isolated voxel
```

This is intentionally not fixed to only `L0/L1`.  Some training labels may
start at L1 when the very bottom slice has little usable signal, so the QC
needs to show whether the model learned to shift the ROI upward.

## Time-Series QC

The full voxel table keeps every mask voxel from the approved candidate layers.
The selected export then uses one selected layer, excludes isolated voxels, and
excludes voxels with obvious zero dropout.

Zero dropout is tagged per voxel:

```text
zero_fraction = fraction of TRs where abs(value) < 1e-6
has_zero_dropout = true when zero_fraction > 0.10
```

The selected time-series QC is a voxel x time carpet plot. The displayed values
are z-scored within each voxel for visual contrast; the CSV stores raw values.
Rows are ordered as:

```text
selected voxels
dropout excluded voxels
isolated excluded voxels
```

The side color bar uses:

```text
green = selected
orange = dropout excluded
cyan = isolated excluded
purple = NaN/inf excluded
```

If many non-isolated voxels have zero dropout, the plot title and summary table
show a warning. The default warning threshold is `0.20`, meaning 20% of
non-isolated voxels. This warning never changes layers or excludes the run by
itself; the user decides whether to keep the layer, choose another layer, or
mark the run as excluded in the selection template.
