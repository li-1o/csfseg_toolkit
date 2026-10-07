"""Center crop/pad the bottom volume to the 3D U-Net patch size.

The model sees the bottom volume as ``(C, D, H, W)``.  In this project those
axes correspond to ``(channels, bottom z depth, X, Y)``.  The first toolkit
version keeps the z depth unchanged and only adjusts X/Y to a fixed 128x128
patch.  This is a crop/pad operation, not a resize, so no interpolation is
introduced before inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_PATCH_HW = (128, 128)


@dataclass(frozen=True)
class AxisTransform:
    """How one spatial axis was moved into the fixed-size patch.

    Attributes:
        orig_len: Length of the source axis before patching.
        out_len: Length of the axis after patching.
        crop_start: Source start index used when cropping.  This is ``0`` when
            the source axis is smaller than the output axis.
        pad_pre: Number of zeros added before the source data.
        pad_post: Number of zeros added after the source data.
    """

    orig_len: int
    out_len: int
    crop_start: int
    pad_pre: int
    pad_post: int

    @property
    def crop_stop(self) -> int:
        """Exclusive source stop index used by the crop step."""

        return min(self.orig_len, self.crop_start + self.out_len)


@dataclass(frozen=True)
class PatchVolume:
    """Fixed-size model patch plus in-memory mapping metadata.

    Attributes:
        data: Float32 array shaped ``(C, D, H, W)``.
        source_shape: Shape before patching, usually ``(C, D, X, Y)``.
        out_hw: Requested model patch size.  The first version uses
            ``(128, 128)``.
        x: Crop/pad metadata for the X axis.
        y: Crop/pad metadata for the Y axis.
    """

    data: np.ndarray
    source_shape: tuple[int, int, int, int]
    out_hw: tuple[int, int]
    x: AxisTransform
    y: AxisTransform

    @property
    def output_shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.data.shape)


def make_patch(
    bottom_volume: np.ndarray,
    out_hw: tuple[int, int] = DEFAULT_PATCH_HW,
) -> PatchVolume:
    """Center crop/pad a bottom volume to the model's fixed X/Y size.

    Args:
        bottom_volume: Array shaped ``(C, D, X, Y)`` from
            :func:`csfseg.preprocessing.bottom.extract_bottom_volume`.
        out_hw: Target X/Y shape.  Defaults to ``(128, 128)``.

    Returns:
        A :class:`PatchVolume` with data shaped ``(C, D, out_hw[0], out_hw[1])``.

    Notes:
        This function does not save anything to disk.  The transform metadata
        stays in memory so future prediction code can map a 128x128 mask back
        into the original bottom-volume X/Y grid.
    """

    data = _validate_patch_input(bottom_volume, out_hw)
    out_x, out_y = int(out_hw[0]), int(out_hw[1])

    x_transform = compute_axis_transform(int(data.shape[2]), out_x)
    y_transform = compute_axis_transform(int(data.shape[3]), out_y)

    cropped = data[
        :,
        :,
        x_transform.crop_start : x_transform.crop_stop,
        y_transform.crop_start : y_transform.crop_stop,
    ]

    patched = np.pad(
        cropped,
        pad_width=(
            (0, 0),
            (0, 0),
            (x_transform.pad_pre, x_transform.pad_post),
            (y_transform.pad_pre, y_transform.pad_post),
        ),
        mode="constant",
        constant_values=0,
    )
    patched = np.ascontiguousarray(patched, dtype=np.float32)

    return PatchVolume(
        data=patched,
        source_shape=tuple(int(v) for v in data.shape),
        out_hw=(out_x, out_y),
        x=x_transform,
        y=y_transform,
    )


def compute_axis_transform(orig_len: int, out_len: int) -> AxisTransform:
    """Return the centered crop/pad rule for one axis."""

    if orig_len <= 0:
        raise ValueError(f"orig_len must be positive, got {orig_len}.")
    if out_len <= 0:
        raise ValueError(f"out_len must be positive, got {out_len}.")

    if orig_len >= out_len:
        crop_start = (orig_len - out_len) // 2
        return AxisTransform(
            orig_len=int(orig_len),
            out_len=int(out_len),
            crop_start=int(crop_start),
            pad_pre=0,
            pad_post=0,
        )

    total_pad = out_len - orig_len
    pad_pre = total_pad // 2
    pad_post = total_pad - pad_pre
    return AxisTransform(
        orig_len=int(orig_len),
        out_len=int(out_len),
        crop_start=0,
        pad_pre=int(pad_pre),
        pad_post=int(pad_post),
    )


def _validate_patch_input(
    bottom_volume: np.ndarray,
    out_hw: tuple[int, int],
) -> np.ndarray:
    data = np.asarray(bottom_volume, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(
            f"Expected bottom volume shape (C, D, X, Y), got ndim={data.ndim}."
        )

    if len(out_hw) != 2:
        raise ValueError(f"out_hw must contain exactly two values, got {out_hw}.")

    out_x, out_y = int(out_hw[0]), int(out_hw[1])
    if out_x <= 0 or out_y <= 0:
        raise ValueError(f"out_hw values must be positive, got {out_hw}.")

    if data.shape[2] <= 0 or data.shape[3] <= 0:
        raise ValueError(f"Bottom X/Y axes must be nonempty, got shape {data.shape}.")

    return data
