# Checkpoints

This folder is a convenient place to put model checkpoint files.

You can also keep checkpoints somewhere else and pass the path explicitly:

```bash
csfseg predict \
  --input /data/sub-001_bold.nii.gz \
  --out-dir /data/csfseg_outputs \
  --checkpoint /models/best_3ch.pt
```

## What File Do I Need?

For normal use, you need a trained 3-channel 3D U-Net checkpoint:

```text
best_3ch.pt
```

The expected input channels are:

```text
mean, std, tsnr
```

The checkpoint should match the toolkit model settings:

```text
in_channels = 3
base = 16
depth = 10
out_hw = 128 x 128
```

## Metadata

A released checkpoint may come with a small sidecar metadata file:

```text
best_3ch.pt
best_3ch.json
```

Metadata is not a segmentation output. It describes the checkpoint: model type,
channel order, normalization, threshold, and whether the checkpoint is ready for
analysis.

If metadata says `requires_finetuning = true`, treat that checkpoint as a
training starting point, not as a final analysis model.
