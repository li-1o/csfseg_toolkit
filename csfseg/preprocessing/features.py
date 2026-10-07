"""Build model input channels from a preprocessed 4D fMRI image.

The segmentation model does not take the raw 4D time series directly.  It
expects a small stack of 3D summary images.  For the current 3-channel model
that stack is:

- temporal mean
- temporal standard deviation
- temporal signal-to-noise ratio, computed as mean / std

This module only computes those channels.  Cropping, padding, bottom-slice
selection, and normalization stay in their own modules so each step remains
easy to inspect in logs and reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SUPPORTED_FEATURE_CHANNELS = ("mean", "std", "tsnr")
DEFAULT_TSNR_EPS = 1e-6


@dataclass(frozen=True)
class FeatureImage:
    """Channel-first feature image prepared from one 4D fMRI input.

    Attributes:
        data: Float32 array with shape ``(C, X, Y, Z)``.
        channels: Channel names in the same order as ``data``.
        stats: Small per-channel summary for reports and logs.
    """

    data: np.ndarray
    channels: tuple[str, ...]
    stats: dict[str, dict[str, float | int]]


def compute_temporal_features(
    fmri_4d: np.ndarray,
    channels: tuple[str, ...] = SUPPORTED_FEATURE_CHANNELS,
    tsnr_eps: float = DEFAULT_TSNR_EPS,
) -> FeatureImage:
    """Compute 3D feature channels from a 4D fMRI array.

    Args:
        fmri_4d: Input fMRI data with shape ``(X, Y, Z, T)``.
        channels: Feature channels to return.  The default is the current
            3-channel model input: ``("mean", "std", "tsnr")``.
        tsnr_eps: Small value added to the denominator when computing tSNR.

    Returns:
        A :class:`FeatureImage` with channel-first data, ``(C, X, Y, Z)``.

    Raises:
        ValueError: If the input is not 4D, the channel list is empty, a
            channel name is unknown, or ``tsnr_eps`` is not positive.
    """

    _validate_feature_request(fmri_4d, channels, tsnr_eps)

    # Use float32 because the network will receive float32 tensors later.
    # The reductions use float64 internally to reduce avoidable rounding drift.
    data = np.asarray(fmri_4d, dtype=np.float32)
    mean_img = np.mean(data, axis=-1, dtype=np.float64).astype(np.float32)
    std_img = np.std(data, axis=-1, dtype=np.float64).astype(np.float32)

    feature_map = {
        "mean": mean_img,
        "std": std_img,
        "tsnr": (mean_img / (std_img + tsnr_eps)).astype(np.float32),
    }

    stacked = np.stack([feature_map[channel] for channel in channels], axis=0)
    stats = {
        channel: summarize_feature_channel(feature_map[channel])
        for channel in channels
    }
    return FeatureImage(
        data=stacked.astype(np.float32, copy=False),
        channels=channels,
        stats=stats,
    )


def summarize_feature_channel(volume: np.ndarray) -> dict[str, float | int]:
    """Return compact numeric stats for one 3D feature channel.

    These values are useful for logs and QC reports.  They are not used to make
    segmentation decisions.
    """

    valid = np.isfinite(volume)
    valid_count = int(valid.sum())
    total_count = int(volume.size)

    if valid_count == 0:
        return {
            "finite_voxels": 0,
            "total_voxels": total_count,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }

    values = volume[valid]
    return {
        "finite_voxels": valid_count,
        "total_voxels": total_count,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64)),
    }


def _validate_feature_request(
    fmri_4d: np.ndarray,
    channels: tuple[str, ...],
    tsnr_eps: float,
) -> None:
    if np.ndim(fmri_4d) != 4:
        raise ValueError(
            "Expected a 4D fMRI array with shape (X, Y, Z, T), "
            f"got ndim={np.ndim(fmri_4d)}."
        )

    if not channels:
        raise ValueError("At least one feature channel is required.")

    if len(set(channels)) != len(channels):
        raise ValueError(f"Duplicate feature channels are not allowed: {channels}.")

    unknown = sorted(set(channels) - set(SUPPORTED_FEATURE_CHANNELS))
    if unknown:
        raise ValueError(
            f"Unknown feature channel(s): {unknown}. "
            f"Supported channels are {SUPPORTED_FEATURE_CHANNELS}."
        )

    if tsnr_eps <= 0:
        raise ValueError(f"tsnr_eps must be positive, got {tsnr_eps}.")
