"""Normalize model input channels before they are sent to the network.

Normalization is the step that puts each feature channel on a comparable
numeric scale.  Without it, a channel with large raw values could dominate a
channel with smaller raw values even if both contain useful anatomy.

The default method here is nonzero z-score normalization:

1. For each channel, look only at finite, nonzero voxels.
2. Compute that channel's mean and standard deviation from those voxels.
3. Convert the channel to ``(value - mean) / std``.
4. Keep background and invalid voxels at zero.

Training and inference must use the same normalization rule.  If we finetune a
3-channel checkpoint later, this is one of the details that should be written
into the checkpoint metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_NORM_EPS = 1e-6


@dataclass(frozen=True)
class ChannelNormStats:
    """Normalization values used for one feature channel."""

    channel: str
    mean: float
    std: float
    valid_voxels: int
    total_voxels: int
    eps: float


@dataclass(frozen=True)
class NormalizedFeatures:
    """Normalized feature stack plus the stats needed for reports/logs."""

    data: np.ndarray
    channels: tuple[str, ...]
    stats: dict[str, ChannelNormStats]
    method: str


def normalize_features(
    features: np.ndarray,
    channels: tuple[str, ...],
    method: str = "nonzero_zscore",
    eps: float = DEFAULT_NORM_EPS,
) -> NormalizedFeatures:
    """Normalize a channel-first feature stack.

    Args:
        features: Feature stack with shape ``(C, X, Y, Z)``.
        channels: Channel names matching the first dimension of ``features``.
        method: Normalization method.  Currently only ``"nonzero_zscore"``.
        eps: Small value used to define nonzero voxels and avoid division by
            zero in edge cases.

    Returns:
        A :class:`NormalizedFeatures` object with float32 data.
    """

    if method != "nonzero_zscore":
        raise ValueError(
            "Only normalization method 'nonzero_zscore' is currently supported."
        )

    data, channel_names = _validate_feature_stack(features, channels, eps)
    normalized, stats = nonzero_zscore(data, channel_names, eps=eps)
    return NormalizedFeatures(
        data=normalized,
        channels=channel_names,
        stats=stats,
        method=method,
    )


def nonzero_zscore(
    features: np.ndarray,
    channels: tuple[str, ...],
    eps: float = DEFAULT_NORM_EPS,
) -> tuple[np.ndarray, dict[str, ChannelNormStats]]:
    """Apply per-channel z-score normalization using only nonzero voxels.

    The nonzero mask is a practical background guard.  In typical fMRI NIfTI
    files, voxels outside the brain or outside the acquired field are zero.
    Including those zeros in the mean/std calculation would make the
    normalization depend heavily on how much empty space is present in the file.
    """

    data, channel_names = _validate_feature_stack(features, channels, eps)
    output = np.zeros_like(data, dtype=np.float32)
    stats: dict[str, ChannelNormStats] = {}

    for channel_index, channel in enumerate(channel_names):
        volume = data[channel_index]
        valid = np.isfinite(volume) & (np.abs(volume) > eps)
        valid_count = int(valid.sum())
        total_count = int(volume.size)

        if valid_count == 0:
            raise ValueError(
                f"Cannot normalize channel '{channel}': "
                "no finite nonzero voxels were found."
            )

        values = volume[valid].astype(np.float64, copy=False)
        mean = float(np.mean(values))
        std = float(np.std(values))

        if std <= eps:
            raise ValueError(
                f"Cannot normalize channel '{channel}': std={std:.6g} is too small."
            )

        normalized_channel = (
            (volume.astype(np.float32, copy=False) - mean) / (std + eps)
        ).astype(np.float32)
        normalized_channel[~valid] = 0.0
        output[channel_index] = normalized_channel
        stats[channel] = ChannelNormStats(
            channel=channel,
            mean=mean,
            std=std,
            valid_voxels=valid_count,
            total_voxels=total_count,
            eps=eps,
        )

    return output, stats


def _validate_feature_stack(
    features: np.ndarray,
    channels: tuple[str, ...],
    eps: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")

    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(
            f"Expected feature stack shape (C, X, Y, Z), got ndim={data.ndim}."
        )

    channel_names = tuple(channels)
    if data.shape[0] != len(channel_names):
        raise ValueError(
            f"Feature stack has {data.shape[0]} channel(s), "
            f"but {len(channel_names)} name(s) were provided."
        )

    if not channel_names:
        raise ValueError("At least one channel is required for normalization.")

    if len(set(channel_names)) != len(channel_names):
        raise ValueError(f"Duplicate channel names are not allowed: {channel_names}.")

    return data, channel_names
