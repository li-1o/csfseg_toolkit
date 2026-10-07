"""3D U-Net used for bottom-slice CSF segmentation.

This is the no-z-downsampling architecture from the reference project.  The
input tensor convention is PyTorch's 3D layout:

``(N, C, D, H, W)``

For this toolkit, ``C`` is the feature channel count, ``D`` is the 10-slice
bottom depth, and ``H/W`` are the 128x128 patch axes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


def _center_crop_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Center-crop ``x`` so its spatial shape matches ``ref``."""

    _, _, depth, height, width = x.shape
    _, _, ref_depth, ref_height, ref_width = ref.shape

    depth_start = (depth - ref_depth) // 2
    height_start = (height - ref_height) // 2
    width_start = (width - ref_width) // 2
    return x[
        :,
        :,
        depth_start : depth_start + ref_depth,
        height_start : height_start + ref_height,
        width_start : width_start + ref_width,
    ]


class UNet3D_NoZDown(nn.Module):
    """3D U-Net that keeps the bottom z depth unchanged.

    Downsampling and upsampling happen only in X/Y.  That keeps thin bottom
    layers such as L0/L1 visible throughout the network.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base: int = 16,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.enc1 = _conv_block(in_channels, base)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc2 = _conv_block(base, base * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc3 = _conv_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.bottleneck = _conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(
            base * 8,
            base * 4,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.dec3 = _conv_block(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(
            base * 4,
            base * 2,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.dec2 = _conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(
            base * 2,
            base,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.dec1 = _conv_block(base * 2, base)

        self.out = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(N, out_channels, D, H, W)``."""

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        if dec3.shape[-3:] != enc3.shape[-3:]:
            dec3 = _center_crop_like(dec3, enc3)
        dec3 = self.dec3(torch.cat([enc3, dec3], dim=1))

        dec2 = self.up2(dec3)
        if dec2.shape[-3:] != enc2.shape[-3:]:
            dec2 = _center_crop_like(dec2, enc2)
        dec2 = self.dec2(torch.cat([enc2, dec2], dim=1))

        dec1 = self.up1(dec2)
        if dec1.shape[-3:] != enc1.shape[-3:]:
            dec1 = _center_crop_like(dec1, enc1)
        dec1 = self.dec1(torch.cat([enc1, dec1], dim=1))

        return self.out(dec1)


__all__ = ["UNet3D_NoZDown"]
