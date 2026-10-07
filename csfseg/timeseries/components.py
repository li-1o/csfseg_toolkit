"""Connected-component labels for low-layer CSF mask candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as ndi_label


DEFAULT_SELECTION_LAYERS = (0, 1, 2, 3)
DEFAULT_MIN_VALID_VOXELS = 5


@dataclass(frozen=True)
class LayerComponentStats:
    """Component summary for one local bottom layer."""

    layer: int
    mask_voxels: int
    isolated_voxels: int
    valid_voxels: int
    n_components: int
    largest_component_size: int
    is_candidate_layer: bool
    is_auto_layer: bool
    component_sizes: dict[int, int]

    @property
    def layer_name(self) -> str:
        return f"L{self.layer}"


@dataclass(frozen=True)
class LayerComponentAnalysis:
    """Component labels and candidate-layer decision for a bottom mask."""

    labels: dict[int, np.ndarray]
    stats: tuple[LayerComponentStats, ...]
    auto_layer: int | None
    min_valid_voxels: int
    layers: tuple[int, ...]
    mask_shape: tuple[int, int, int]

    def stats_for(self, layer: int) -> LayerComponentStats:
        layer = int(layer)
        for item in self.stats:
            if item.layer == layer:
                return item
        raise KeyError(f"Layer L{layer} was not analyzed.")

    def component_size(self, layer: int, component_id: int) -> int:
        if component_id <= 0:
            return 0
        return int(self.stats_for(layer).component_sizes.get(int(component_id), 0))


def analyze_mask_layers(
    mask_bottom: np.ndarray,
    layers: tuple[int, ...] = DEFAULT_SELECTION_LAYERS,
    min_valid_voxels: int = DEFAULT_MIN_VALID_VOXELS,
) -> LayerComponentAnalysis:
    """Label 2D connected components in low bottom mask layers.

    Components are measured within each 2D slice using 8-neighbor connectivity.
    In the first toolkit version, an isolated voxel is exactly a component with
    size 1.  Larger components are kept as potentially meaningful mask regions;
    the layer becomes a candidate if enough non-isolated voxels remain.
    """

    mask_data, layer_indices = _validate_mask_layers(
        mask_bottom,
        layers,
        min_valid_voxels,
    )
    structure = np.ones((3, 3), dtype=np.uint8)

    labels_by_layer: dict[int, np.ndarray] = {}
    raw_stats: list[LayerComponentStats] = []

    for layer in layer_indices:
        layer_mask = mask_data[layer] > 0
        labels, n_components = ndi_label(layer_mask, structure=structure)
        labels = labels.astype(np.int32, copy=False)
        counts = np.bincount(labels.ravel())
        component_sizes = {
            int(component_id): int(size)
            for component_id, size in enumerate(counts)
            if component_id > 0 and size > 0
        }

        mask_voxels = int(layer_mask.sum())
        isolated_voxels = int(
            sum(size for size in component_sizes.values() if size == 1)
        )
        valid_voxels = int(mask_voxels - isolated_voxels)
        largest_component_size = int(max(component_sizes.values(), default=0))
        is_candidate = valid_voxels >= int(min_valid_voxels)

        labels_by_layer[layer] = labels
        raw_stats.append(
            LayerComponentStats(
                layer=layer,
                mask_voxels=mask_voxels,
                isolated_voxels=isolated_voxels,
                valid_voxels=valid_voxels,
                n_components=int(n_components),
                largest_component_size=largest_component_size,
                is_candidate_layer=is_candidate,
                is_auto_layer=False,
                component_sizes=component_sizes,
            )
        )

    auto_layer = next(
        (item.layer for item in raw_stats if item.is_candidate_layer),
        None,
    )
    final_stats = tuple(
        LayerComponentStats(
            layer=item.layer,
            mask_voxels=item.mask_voxels,
            isolated_voxels=item.isolated_voxels,
            valid_voxels=item.valid_voxels,
            n_components=item.n_components,
            largest_component_size=item.largest_component_size,
            is_candidate_layer=item.is_candidate_layer,
            is_auto_layer=(item.layer == auto_layer),
            component_sizes=item.component_sizes,
        )
        for item in raw_stats
    )

    return LayerComponentAnalysis(
        labels=labels_by_layer,
        stats=final_stats,
        auto_layer=auto_layer,
        min_valid_voxels=int(min_valid_voxels),
        layers=layer_indices,
        mask_shape=tuple(int(v) for v in mask_data.shape),
    )


def normalize_layer_name(layer: str | int | None) -> str:
    """Return layer strings in the canonical ``L0`` form."""

    if layer is None:
        return ""
    if isinstance(layer, float) and np.isnan(layer):
        return ""
    if isinstance(layer, str):
        value = layer.strip()
        if not value:
            return ""
        if value.upper().startswith("L"):
            number = value[1:]
        else:
            number = value
    else:
        number = str(int(layer))
    try:
        return f"L{int(number)}"
    except ValueError as exc:
        raise ValueError(f"Layer must look like L0, L1, 0, or 1; got {layer!r}.") from exc


def layer_name_to_index(layer: str | int) -> int:
    """Return the integer index for a layer name such as ``L1``."""

    name = normalize_layer_name(layer)
    if not name:
        raise ValueError("Layer cannot be empty.")
    return int(name[1:])


def _validate_mask_layers(
    mask_bottom: np.ndarray,
    layers: tuple[int, ...],
    min_valid_voxels: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    mask_data = np.asarray(mask_bottom)
    if mask_data.ndim != 3:
        raise ValueError(f"Expected mask_bottom shape (D, X, Y), got {mask_data.shape}.")
    if min_valid_voxels <= 0:
        raise ValueError(f"min_valid_voxels must be positive, got {min_valid_voxels}.")
    if not layers:
        raise ValueError("At least one layer is required.")

    depth = int(mask_data.shape[0])
    layer_indices = tuple(int(layer) for layer in layers)
    invalid = [layer for layer in layer_indices if layer < 0 or layer >= depth]
    if invalid:
        raise ValueError(f"Layer(s) {invalid} are outside mask depth 0:{depth}.")
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(f"Duplicate layers are not allowed: {layers}.")

    return (mask_data > 0).astype(np.uint8, copy=False), layer_indices


__all__ = [
    "DEFAULT_MIN_VALID_VOXELS",
    "DEFAULT_SELECTION_LAYERS",
    "LayerComponentAnalysis",
    "LayerComponentStats",
    "analyze_mask_layers",
    "layer_name_to_index",
    "normalize_layer_name",
]
