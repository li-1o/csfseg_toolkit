"""Prediction thresholding and restoration to NIfTI space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csfseg.preprocessing.bottom import BottomVolume
from csfseg.preprocessing.patch import PatchVolume


@dataclass(frozen=True)
class RestoredPrediction:
    """Probability and mask arrays in patch, bottom, and full-image space."""

    probability_patch: np.ndarray
    probability_bottom: np.ndarray
    probability_full: np.ndarray
    mask_patch: np.ndarray
    mask_bottom: np.ndarray
    mask_full: np.ndarray
    threshold: float


def threshold_probability(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a probability map to a uint8 binary mask."""

    if threshold < 0 or threshold > 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
    prob = np.asarray(probability, dtype=np.float32)
    if prob.ndim != 3:
        raise ValueError(f"Expected probability shape (D, H, W), got {prob.shape}.")
    return (prob >= float(threshold)).astype(np.uint8)


def restore_patch_to_bottom(patch_map: np.ndarray, patch: PatchVolume) -> np.ndarray:
    """Map a ``(D, H, W)`` patch prediction back to bottom ``(D, X, Y)`` space."""

    data = _validate_patch_map(patch_map, patch)
    depth = int(data.shape[0])
    restored = np.zeros(
        (depth, patch.x.orig_len, patch.y.orig_len),
        dtype=data.dtype,
    )

    patch_x, bottom_x = _axis_restore_slices(patch.x)
    patch_y, bottom_y = _axis_restore_slices(patch.y)
    restored[:, bottom_x, bottom_y] = data[:, patch_x, patch_y]
    return restored


def restore_bottom_to_full(
    bottom_map: np.ndarray,
    bottom: BottomVolume,
    spatial_shape: tuple[int, int, int],
) -> np.ndarray:
    """Place a bottom ``(D, X, Y)`` map into full ``(X, Y, Z)`` image space."""

    data = np.asarray(bottom_map)
    if data.ndim != 3:
        raise ValueError(f"Expected bottom map shape (D, X, Y), got {data.shape}.")

    expected_bottom_shape = (
        bottom.depth,
        bottom.source_shape[1],
        bottom.source_shape[2],
    )
    if tuple(data.shape) != expected_bottom_shape:
        raise ValueError(
            f"Bottom map shape {data.shape} does not match expected "
            f"{expected_bottom_shape}."
        )

    expected_spatial = bottom.source_shape[1:]
    if tuple(spatial_shape) != expected_spatial:
        raise ValueError(
            f"Full spatial shape {spatial_shape} does not match bottom source "
            f"shape {expected_spatial}."
        )

    full = np.zeros(tuple(int(v) for v in spatial_shape), dtype=data.dtype)
    full[:, :, bottom.z_start : bottom.z_stop] = np.transpose(data, (1, 2, 0))
    return full


def restore_prediction_to_full(
    probability_patch: np.ndarray,
    patch: PatchVolume,
    bottom: BottomVolume,
    spatial_shape: tuple[int, int, int],
    *,
    threshold: float = 0.5,
) -> RestoredPrediction:
    """Threshold and restore one patch probability map into full-image space."""

    prob_patch = np.asarray(probability_patch, dtype=np.float32)
    probability_bottom = restore_patch_to_bottom(prob_patch, patch).astype(
        np.float32,
        copy=False,
    )
    probability_full = restore_bottom_to_full(
        probability_bottom,
        bottom,
        spatial_shape,
    ).astype(np.float32, copy=False)

    mask_patch = threshold_probability(prob_patch, threshold=threshold)
    mask_bottom = restore_patch_to_bottom(mask_patch, patch).astype(np.uint8, copy=False)
    mask_full = restore_bottom_to_full(mask_bottom, bottom, spatial_shape).astype(
        np.uint8,
        copy=False,
    )

    return RestoredPrediction(
        probability_patch=prob_patch,
        probability_bottom=probability_bottom,
        probability_full=probability_full,
        mask_patch=mask_patch,
        mask_bottom=mask_bottom,
        mask_full=mask_full,
        threshold=float(threshold),
    )


def _axis_restore_slices(axis) -> tuple[slice, slice]:
    if axis.orig_len >= axis.out_len:
        patch_slice = slice(0, axis.out_len)
        bottom_slice = slice(axis.crop_start, axis.crop_start + axis.out_len)
    else:
        patch_slice = slice(axis.pad_pre, axis.out_len - axis.pad_post)
        bottom_slice = slice(0, axis.orig_len)
    return patch_slice, bottom_slice


def _validate_patch_map(patch_map: np.ndarray, patch: PatchVolume) -> np.ndarray:
    data = np.asarray(patch_map)
    if data.ndim != 3:
        raise ValueError(f"Expected patch map shape (D, H, W), got {data.shape}.")

    expected = (
        patch.source_shape[1],
        patch.out_hw[0],
        patch.out_hw[1],
    )
    if tuple(data.shape) != expected:
        raise ValueError(f"Patch map shape {data.shape} does not match {expected}.")

    return data
