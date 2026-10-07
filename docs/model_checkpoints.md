# Model Checkpoints

A checkpoint is the trained model weight file, usually ending in `.pt` or
`.pth`.

The Python code defines the U-Net architecture. The checkpoint stores the
learned weights. They have to match.

## Expected Checkpoint

The current toolkit target is a 3-channel 3D U-Net:

```text
channels = mean, std, tsnr
in_channels = 3
base = 16
depth = 10
patch size = 10 x 128 x 128
```

Use a checkpoint trained or finetuned for this input layout.

## Why Metadata Helps

A checkpoint may come with a small JSON file:

```text
best_3ch.pt
best_3ch.json
```

That JSON is not produced by prediction. It is a description of the checkpoint
itself. A typical metadata file might say:

```json
{
  "model": "UNet3D_NoZDown",
  "in_channels": 3,
  "channels": ["mean", "std", "tsnr"],
  "base": 16,
  "depth": 10,
  "out_hw": [128, 128],
  "normalization": "nonzero_zscore",
  "threshold": 0.5,
  "requires_finetuning": false
}
```

This helps users avoid a quiet but serious mistake: using a checkpoint trained
for one input layout with a different preprocessing pipeline.

## Checking Channel Count

For a PyTorch 3D convolution, the first layer weight shape is:

```text
(out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
```

For this toolkit, a normal first layer should look like:

```text
[16, 3, 3, 3, 3]
```

The second number is the input channel count. If it is `1`, that checkpoint was
not trained as a 3-channel model.

## Fit For Analysis

Before using a checkpoint for analysis, check:

- it is 3-channel
- channel order is `mean, std, tsnr`
- normalization is `nonzero_zscore`
- metadata does not say `requires_finetuning = true`
