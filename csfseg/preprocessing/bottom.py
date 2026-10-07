"""Extract the bottom thin volume used by the 3D U-Net.

The first toolkit version uses a deliberately simple bottom rule:

- array ``z=0`` is treated as the acquisition bottom
- the model sees ``depth`` consecutive slices starting at ``z_start``
- later QC and layer selection focus on low local layers, usually L0-L3

This module does not inspect the NIfTI affine and does not try to infer the
true anatomical inferior direction.  That assumption is checked visually by QC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_BOTTOM_DEPTH = 10
DEFAULT_Z_START = 0
DEFAULT_CANDIDATE_LAYERS = (0, 1, 2, 3)


@dataclass(frozen=True)
class BottomVolume:
    """Bottom model-input volume plus the in-memory mapping metadata.

    Attributes:
        data: Float32 array with shape ``(C, D, X, Y)``.
        z_start: First source z index included in ``data``.
        z_stop: Stop source z index, exclusive.
        depth: Number of z slices in ``data``.
        source_shape: Shape of the input feature stack, ``(C, X, Y, Z)``.
        candidate_layers: Low local z layers highlighted for QC and later
            final-layer selection.
    """

    data: np.ndarray
    z_start: int
    z_stop: int
    depth: int
    source_shape: tuple[int, int, int, int]
    candidate_layers: tuple[int, ...]

    @property
    def output_shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.data.shape)


def extract_bottom_volume(
    features: np.ndarray,
    depth: int = DEFAULT_BOTTOM_DEPTH,
    z_start: int = DEFAULT_Z_START,
    candidate_layers: tuple[int, ...] = DEFAULT_CANDIDATE_LAYERS,
) -> BottomVolume:
    """Take the bottom z block from a channel-first feature stack.

    Args:
        features: Normalized feature stack with shape ``(C, X, Y, Z)``.
        depth: Number of z slices to provide to the model.
        z_start: First source z index.  First version defaults to ``0``.
        candidate_layers: Low local z indices highlighted for QC and later
            signal-layer selection.  They are recorded here so downstream code
            can keep the layer convention explicit.

    Returns:
        A :class:`BottomVolume` whose ``data`` is shaped ``(C, D, X, Y)``.
    """

    feature_data = _validate_bottom_request(
        features,
        depth,
        z_start,
        candidate_layers,
    )
    z_stop = z_start + depth

    # NIfTI-like feature layout is (C, X, Y, Z).  PyTorch Conv3d expects a
    # single sample as (C, D, H, W), where D is our z direction.
    bottom_cxyd = feature_data[:, :, :, z_start:z_stop]
    bottom_cdxy = np.transpose(bottom_cxyd, (0, 3, 1, 2))
    bottom_cdxy = np.ascontiguousarray(bottom_cdxy, dtype=np.float32)

    return BottomVolume(
        data=bottom_cdxy,
        z_start=int(z_start),
        z_stop=int(z_stop),
        depth=int(depth),
        source_shape=tuple(int(v) for v in feature_data.shape),
        candidate_layers=tuple(int(v) for v in candidate_layers),
    )


def _validate_bottom_request(
    features: np.ndarray,
    depth: int,
    z_start: int,
    candidate_layers: tuple[int, ...],
) -> np.ndarray:
    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(
            f"Expected feature stack shape (C, X, Y, Z), got ndim={data.ndim}."
        )

    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}.")

    if z_start < 0:
        raise ValueError(f"z_start must be nonnegative, got {z_start}.")

    z_size = int(data.shape[3])
    z_stop = z_start + depth
    if z_stop > z_size:
        raise ValueError(
            f"Cannot extract z={z_start}:{z_stop} from feature stack with Z={z_size}."
        )

    if not candidate_layers:
        raise ValueError("At least one candidate layer is required.")

    if len(set(candidate_layers)) != len(candidate_layers):
        raise ValueError(
            f"Duplicate candidate layers are not allowed: {candidate_layers}."
        )

    invalid = [layer for layer in candidate_layers if layer < 0 or layer >= depth]
    if invalid:
        raise ValueError(
            f"Candidate layer(s) {invalid} are outside the extracted depth 0:{depth}."
        )

    return data
