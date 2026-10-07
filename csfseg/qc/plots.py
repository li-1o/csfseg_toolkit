"""QC plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_preprocessing_qc(
    fmri_4d: np.ndarray,
    features: np.ndarray,
    normalized: np.ndarray,
    channels: tuple[str, ...],
    out_path: str | Path,
    slice_index: int | None = None,
    raw_time_indices: tuple[int, int, int] | None = None,
) -> Path:
    """Save a visual QC panel for feature generation and normalization.

    The figure is a 3x3 panel:

    - row 1: raw 4D snapshots at t=0, t=T//2, t=T-1
    - row 2: mean/std/tSNR feature images
    - row 3: normalized mean/std/tSNR feature images

    All panels use the same axial slice so users can compare the full
    preprocessing chain by eye.
    """

    raw_data, feature_data, norm_data, channel_names = _validate_plot_inputs(
        fmri_4d,
        features,
        normalized,
        channels,
    )
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z_index = _choose_slice_index(feature_data, slice_index)
    time_indices = choose_raw_time_indices(raw_data.shape[-1], raw_time_indices)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(11, 10), squeeze=False)

    for column, time_index in enumerate(time_indices):
        raw_slice = raw_data[:, :, z_index, time_index]
        _show_slice(
            axes[0, column],
            raw_slice,
            f"raw t={time_index}",
            cmap="gray",
        )

    for column, channel in enumerate(channel_names):
        feature_slice = feature_data[column, :, :, z_index]
        norm_slice = norm_data[column, :, :, z_index]
        _show_slice(
            axes[1, column],
            feature_slice,
            f"{channel} feature",
            cmap="gray",
        )
        _show_slice(
            axes[2, column],
            norm_slice,
            f"{channel} normalized",
            cmap="coolwarm",
        )

    fig.suptitle(f"Preprocessing QC, axial slice z={z_index}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_bottom_volume_qc(
    bottom_volume: np.ndarray,
    channels: tuple[str, ...],
    out_path: str | Path,
    source_z_start: int = 0,
    candidate_layers: tuple[int, ...] = (0, 1, 2, 3),
) -> Path:
    """Save a QC panel showing the actual bottom volume sent to the model.

    The expected layout is ``(C, D, X, Y)``.  Rows are feature channels and
    columns are local bottom layers L0-L9.  This lets users inspect all model
    input channels, not only the mean image.
    """

    bottom_data, channel_names = _validate_bottom_qc_inputs(bottom_volume, channels)
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n_channels, depth = bottom_data.shape[:2]
    fig, axes = plt.subplots(
        n_channels,
        depth,
        figsize=(2.0 * depth, 2.3 * n_channels),
        squeeze=False,
    )

    for row, channel in enumerate(channel_names):
        for local_z in range(depth):
            label = f"L{local_z}"
            if local_z in candidate_layers:
                label += "*"
            source_z = source_z_start + local_z
            title = f"{channel} {label}\nz={source_z}"
            _show_slice_no_colorbar(
                axes[row, local_z],
                bottom_data[row, local_z],
                title,
                cmap="coolwarm",
            )

    fig.suptitle(
        "Bottom volume QC: model input channels across local L0-L9\n"
        "* low-layer QC / later selection range",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_patch_qc(
    before_patch: np.ndarray,
    after_patch: np.ndarray,
    channels: tuple[str, ...],
    out_path: str | Path,
    candidate_layers: tuple[int, ...] = (0, 1, 2, 3),
) -> Path:
    """Save a QC panel for the center crop/pad step.

    The figure compares the bottom volume before patching with the fixed-size
    patch after center crop/pad.  Rows are the three feature channels.  Columns
    show the highlighted low layers before/after patching.
    """

    before_data, after_data, channel_names = _validate_patch_qc_inputs(
        before_patch,
        after_patch,
        channels,
        candidate_layers,
    )
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n_layers = len(candidate_layers)
    fig, axes = plt.subplots(3, n_layers * 2, figsize=(3.0 * n_layers, 8), squeeze=False)

    panel_specs = []
    for local_z in candidate_layers:
        panel_specs.append((local_z, before_data, "before"))
        panel_specs.append((local_z, after_data, "after"))

    for row, channel in enumerate(channel_names):
        for column, (local_z, data, stage) in enumerate(panel_specs):
            _show_slice_no_colorbar(
                axes[row, column],
                data[row, local_z],
                f"{channel} L{local_z} {stage}",
                cmap="coolwarm",
            )

    fig.suptitle(
        "Patch QC: low layers before and after center crop/pad to 128x128",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_prediction_qc(
    mean_bottom: np.ndarray,
    probability_bottom: np.ndarray,
    mask_bottom: np.ndarray,
    out_path: str | Path,
    layers: tuple[int, ...] = (0, 1, 2, 3),
    component_analysis=None,
) -> Path:
    """Save a prediction QC panel for the low bottom layers.

    The expected inputs are bottom-space arrays shaped ``(D, X, Y)``. Rows:

    - normalized mean image
    - normalized mean image with binary mask overlay

    Probability is still accepted so the call stays close to the prediction
    pipeline, but the first QC figure emphasizes the thresholded mask because
    users mainly need to judge the spatial ROI and isolated voxels by eye.
    """

    mean_data, prob_data, mask_data, layer_indices = _validate_prediction_qc_inputs(
        mean_bottom,
        probability_bottom,
        mask_bottom,
        layers,
    )
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, len(layer_indices), figsize=(3.2 * len(layer_indices), 5.4), squeeze=False)

    for column, local_z in enumerate(layer_indices):
        title = _prediction_layer_title(local_z, mask_data, component_analysis)
        labels = None
        stats = None
        if component_analysis is not None and local_z in component_analysis.labels:
            labels = component_analysis.labels[local_z]
            stats = component_analysis.stats_for(local_z)
        _show_slice_no_colorbar(
            axes[0, column],
            mean_data[local_z],
            f"mean z-score L{local_z}",
            cmap="coolwarm",
        )
        _show_component_mask_overlay(
            axes[1, column],
            mean_data[local_z],
            mask_data[local_z],
            title,
            labels=labels,
            component_sizes=stats.component_sizes if stats else None,
        )

    fig.suptitle(
        "Prediction QC: low-layer mask overlay and isolated voxels",
        fontsize=12,
    )
    legend_handles = [
        Patch(facecolor="red", alpha=0.55, label="non-isolated mask voxel"),
        Patch(facecolor="cyan", alpha=0.70, label="isolated voxel"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_timeseries_qc(
    voxel_table,
    out_path: str | Path,
    *,
    output_id: str,
    selected_layer: str,
    selection_name: str,
    min_valid_voxels: int,
    dropout_warning_fraction: float = 0.20,
) -> Path:
    """Save a carpet QC plot for one selected layer.

    Rows are voxel time series.  The display is per-voxel z-scored for visual
    contrast; the CSV tables keep the raw values.  Rows are ordered as selected
    voxels first, then zero-dropout voxels, then isolated voxels.
    """

    table, time_cols = _validate_timeseries_qc_inputs(voxel_table, selected_layer)
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    categories = _timeseries_qc_categories(table)
    order = np.argsort(categories, kind="stable")
    ordered_table = table.iloc[order].reset_index(drop=True)
    ordered_categories = categories[order]
    values = ordered_table[time_cols].to_numpy(dtype=np.float32)
    display = _per_voxel_zscore(values)

    valid_mask = ~_as_bool_array(table["is_isolated"])
    dropout_mask = valid_mask & _as_bool_array(table["has_zero_dropout"])
    invalid = (
        _as_bool_array(table["has_nan_or_inf"])
        if "has_nan_or_inf" in table.columns
        else np.zeros(len(table), dtype=bool)
    )
    invalid_mask = valid_mask & invalid
    selected_mask = valid_mask & ~dropout_mask & ~invalid_mask

    total_voxels = int(len(table))
    isolated_voxels = int(_as_bool_array(table["is_isolated"]).sum())
    valid_voxels = int(valid_mask.sum())
    dropout_voxels = int(dropout_mask.sum())
    selected_voxels = int(selected_mask.sum())
    dropout_fraction = float(dropout_voxels / valid_voxels) if valid_voxels else 0.0
    warning = dropout_fraction >= float(dropout_warning_fraction) and valid_voxels > 0

    fig = plt.figure(figsize=(12, max(4.5, 0.12 * max(total_voxels, 1) + 2.2)))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.035, 1.0], wspace=0.02)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])

    if total_voxels:
        ax.imshow(display, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3, interpolation="nearest")
        ax_bar.imshow(ordered_categories[:, None], aspect="auto", cmap=ListedColormap(["#2ca25f", "#fdae61", "#00bcd4", "#756bb1"]), vmin=0, vmax=3)
    else:
        ax.text(0.5, 0.5, "No voxels in selected layer", ha="center", va="center")

    ax.set_xlabel("Time (TR)")
    ax.set_ylabel("Voxel")
    ax.set_title(
        "voxel x time carpet (per-voxel z-score display)",
        fontsize=10,
    )
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])

    summary = (
        f"{output_id} | {selection_name} | {selected_layer} | "
        f"total={total_voxels}, selected={selected_voxels}, "
        f"isolated={isolated_voxels}, dropout={dropout_voxels}, "
        f"dropout_fraction={dropout_fraction:.1%}"
    )
    if warning:
        summary += " | WARNING: high dropout fraction"
    fig.suptitle(summary, fontsize=11)

    legend = [
        Patch(facecolor="#2ca25f", label="selected"),
        Patch(facecolor="#fdae61", label="dropout excluded"),
        Patch(facecolor="#00bcd4", label="isolated excluded"),
        Patch(facecolor="#756bb1", label="NaN/inf excluded"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def choose_raw_time_indices(
    timepoints: int,
    raw_time_indices: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    """Return the raw 4D snapshots shown in preprocessing QC."""

    if timepoints <= 0:
        raise ValueError(f"timepoints must be positive, got {timepoints}.")

    if raw_time_indices is None:
        return (0, timepoints // 2, timepoints - 1)

    if len(raw_time_indices) != 3:
        raise ValueError("raw_time_indices must contain exactly three indices.")

    indices = tuple(int(v) for v in raw_time_indices)
    for index in indices:
        if index < 0 or index >= timepoints:
            raise ValueError(
                f"raw time index must be in [0, {timepoints - 1}], got {index}."
            )
    return indices


def choose_preprocessing_qc_slice(
    features: np.ndarray,
    slice_index: int | None = None,
) -> int:
    """Choose the axial slice used in preprocessing QC."""

    feature_data = np.asarray(features, dtype=np.float32)
    if feature_data.ndim != 4:
        raise ValueError(
            f"Expected feature stack shape (C, X, Y, Z), got ndim={feature_data.ndim}."
        )
    return _choose_slice_index(feature_data, slice_index)


def _show_slice(ax, data: np.ndarray, title: str, cmap: str) -> None:
    finite = np.isfinite(data)
    if finite.any():
        values = data[finite]
        vmin, vmax = _robust_limits(values)
    else:
        vmin, vmax = 0.0, 1.0

    image = ax.imshow(np.rot90(data), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _show_slice_no_colorbar(ax, data: np.ndarray, title: str, cmap: str) -> None:
    finite = np.isfinite(data)
    if finite.any():
        values = data[finite]
        vmin, vmax = _robust_limits(values)
    else:
        vmin, vmax = 0.0, 1.0

    ax.imshow(np.rot90(data), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _show_probability_slice(ax, data: np.ndarray, title: str) -> None:
    ax.imshow(np.rot90(data), cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _show_mask_overlay(
    ax,
    base: np.ndarray,
    mask: np.ndarray,
    title: str,
) -> None:
    finite = np.isfinite(base)
    if finite.any():
        vmin, vmax = _robust_limits(base[finite])
    else:
        vmin, vmax = 0.0, 1.0

    ax.imshow(np.rot90(base), cmap="gray", vmin=vmin, vmax=vmax)
    mask_bool = np.asarray(mask) > 0
    if np.any(mask_bool):
        overlay = np.ma.masked_where(~mask_bool, mask_bool.astype(np.float32))
        ax.imshow(np.rot90(overlay), cmap="Reds", vmin=0.0, vmax=1.0, alpha=0.55)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _show_component_mask_overlay(
    ax,
    base: np.ndarray,
    mask: np.ndarray,
    title: str,
    *,
    labels: np.ndarray | None,
    component_sizes: dict[int, int] | None,
) -> None:
    finite = np.isfinite(base)
    if finite.any():
        vmin, vmax = _robust_limits(base[finite])
    else:
        vmin, vmax = 0.0, 1.0

    ax.imshow(np.rot90(base), cmap="gray", vmin=vmin, vmax=vmax)
    mask_bool = np.asarray(mask) > 0
    if np.any(mask_bool):
        if labels is None or component_sizes is None:
            valid_mask = mask_bool
            isolated_mask = np.zeros_like(mask_bool, dtype=bool)
        else:
            labels = np.asarray(labels)
            isolated_ids = {
                component_id
                for component_id, size in component_sizes.items()
                if size == 1
            }
            isolated_mask = np.isin(labels, list(isolated_ids)) & mask_bool
            valid_mask = mask_bool & ~isolated_mask

        if np.any(valid_mask):
            valid_overlay = np.ma.masked_where(~valid_mask, valid_mask.astype(np.float32))
            ax.imshow(np.rot90(valid_overlay), cmap="Reds", vmin=0.0, vmax=1.0, alpha=0.55)
        if np.any(isolated_mask):
            isolated_overlay = np.ma.masked_where(~isolated_mask, isolated_mask.astype(np.float32))
            ax.imshow(np.rot90(isolated_overlay), cmap="cool", vmin=0.0, vmax=1.0, alpha=0.75)

    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _prediction_layer_title(local_z: int, mask_data: np.ndarray, component_analysis) -> str:
    if component_analysis is None:
        mask_voxels = int(np.count_nonzero(mask_data[local_z] > 0))
        return f"L{local_z}\nn={mask_voxels}"

    stats = component_analysis.stats_for(local_z)
    status = "candidate" if stats.is_candidate_layer else "below"
    if stats.is_auto_layer:
        status = f"AUTO {status}"
    return (
        f"{stats.layer_name} {status}\n"
        f"n={stats.mask_voxels}, valid={stats.valid_voxels}, iso={stats.isolated_voxels}"
    )


def _validate_timeseries_qc_inputs(voxel_table, selected_layer: str):
    import pandas as pd

    table = pd.DataFrame(voxel_table).copy()
    if table.empty:
        time_cols: list[str] = []
    else:
        required = {"layer", "is_isolated", "has_zero_dropout"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Timeseries QC table is missing column(s): {missing}.")
        time_cols = _time_columns(table)
        if not time_cols:
            raise ValueError("Timeseries QC table has no t0/t1/... columns.")

    layer_name = selected_layer if str(selected_layer).upper().startswith("L") else f"L{int(selected_layer)}"
    table = table[table["layer"].astype(str).str.upper() == layer_name.upper()].copy()
    return table, time_cols


def _timeseries_qc_categories(table) -> np.ndarray:
    isolated = _as_bool_array(table["is_isolated"])
    dropout = _as_bool_array(table["has_zero_dropout"])
    invalid = (
        _as_bool_array(table["has_nan_or_inf"])
        if "has_nan_or_inf" in table.columns
        else np.zeros(len(table), dtype=bool)
    )
    categories = np.zeros(len(table), dtype=np.int32)
    categories[dropout & ~isolated] = 1
    categories[isolated] = 2
    categories[invalid & ~isolated] = 3
    return categories


def _as_bool_array(values) -> np.ndarray:
    if isinstance(values, bool):
        return np.array([values], dtype=bool)
    array = np.asarray(values)
    if array.dtype == bool:
        return array
    return np.array([_to_bool(value) for value in array], dtype=bool)


def _to_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _time_columns(table) -> list[str]:
    columns = [column for column in table.columns if column.startswith("t") and column[1:].isdigit()]
    return sorted(columns, key=lambda name: int(name[1:]))


def _per_voxel_zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    finite_values = np.where(np.isfinite(values), values, np.nan)
    mean = np.nanmean(finite_values, axis=1, keepdims=True)
    std = np.nanstd(finite_values, axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (finite_values - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _robust_limits(values: np.ndarray) -> tuple[float, float]:
    vmin, vmax = np.percentile(values, [1, 99])
    if np.isclose(vmin, vmax):
        center = float(vmin)
        return center - 1.0, center + 1.0
    return float(vmin), float(vmax)


def _choose_slice_index(features: np.ndarray, slice_index: int | None) -> int:
    depth = int(features.shape[-1])
    if slice_index is not None:
        if slice_index < 0 or slice_index >= depth:
            raise ValueError(
                f"slice_index must be in [0, {depth - 1}], got {slice_index}."
            )
        return int(slice_index)

    nonzero_counts = np.sum(
        np.isfinite(features[0]) & (np.abs(features[0]) > 1e-6),
        axis=(0, 1),
    )
    if np.any(nonzero_counts):
        return int(np.argmax(nonzero_counts))
    return depth // 2


def _validate_plot_inputs(
    fmri_4d: np.ndarray,
    features: np.ndarray,
    normalized: np.ndarray,
    channels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    raw_data = np.asarray(fmri_4d, dtype=np.float32)
    feature_data = np.asarray(features, dtype=np.float32)
    norm_data = np.asarray(normalized, dtype=np.float32)
    channel_names = tuple(channels)

    if raw_data.ndim != 4:
        raise ValueError(
            f"Expected raw fMRI shape (X, Y, Z, T), got ndim={raw_data.ndim}."
        )

    if feature_data.ndim != 4:
        raise ValueError(
            f"Expected feature stack shape (C, X, Y, Z), got ndim={feature_data.ndim}."
        )

    if raw_data.shape[:3] != feature_data.shape[1:]:
        raise ValueError(
            f"Raw spatial shape {raw_data.shape[:3]} does not match "
            f"feature spatial shape {feature_data.shape[1:]}."
        )

    if norm_data.shape != feature_data.shape:
        raise ValueError(
            f"Normalized stack shape {norm_data.shape} does not match "
            f"feature stack shape {feature_data.shape}."
        )

    if feature_data.shape[0] != len(channel_names):
        raise ValueError(
            f"Feature stack has {feature_data.shape[0]} channel(s), "
            f"but {len(channel_names)} name(s) were provided."
        )

    if len(channel_names) != 3:
        raise ValueError("Preprocessing QC figure expects exactly three channels.")

    return raw_data, feature_data, norm_data, channel_names


def _validate_bottom_qc_inputs(
    bottom_volume: np.ndarray,
    channels: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    bottom_data = np.asarray(bottom_volume, dtype=np.float32)
    channel_names = tuple(channels)

    if bottom_data.ndim != 4:
        raise ValueError(
            f"Expected bottom volume shape (C, D, X, Y), got ndim={bottom_data.ndim}."
        )

    if bottom_data.shape[0] != len(channel_names):
        raise ValueError(
            f"Bottom volume has {bottom_data.shape[0]} channel(s), "
            f"but {len(channel_names)} name(s) were provided."
        )

    if bottom_data.shape[1] != 10:
        raise ValueError(
            f"Bottom QC expects depth=10 for the first version, got D={bottom_data.shape[1]}."
        )

    if len(channel_names) != 3:
        raise ValueError("Bottom QC figure expects exactly three channels.")

    return bottom_data, channel_names


def _validate_patch_qc_inputs(
    before_patch: np.ndarray,
    after_patch: np.ndarray,
    channels: tuple[str, ...],
    candidate_layers: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    before_data = np.asarray(before_patch, dtype=np.float32)
    after_data = np.asarray(after_patch, dtype=np.float32)
    channel_names = tuple(channels)

    if before_data.ndim != 4:
        raise ValueError(
            f"Expected before_patch shape (C, D, X, Y), got ndim={before_data.ndim}."
        )

    if after_data.ndim != 4:
        raise ValueError(
            f"Expected after_patch shape (C, D, H, W), got ndim={after_data.ndim}."
        )

    if before_data.shape[:2] != after_data.shape[:2]:
        raise ValueError(
            f"Before patch C/D shape {before_data.shape[:2]} does not match "
            f"after patch C/D shape {after_data.shape[:2]}."
        )

    if before_data.shape[0] != len(channel_names):
        raise ValueError(
            f"Patch data has {before_data.shape[0]} channel(s), "
            f"but {len(channel_names)} name(s) were provided."
        )

    if len(channel_names) != 3:
        raise ValueError("Patch QC figure expects exactly three channels.")

    if not candidate_layers:
        raise ValueError("Patch QC expects at least one candidate layer.")

    depth = int(before_data.shape[1])
    invalid = [layer for layer in candidate_layers if layer < 0 or layer >= depth]
    if invalid:
        raise ValueError(
            f"Candidate layer(s) {invalid} are outside the patch depth 0:{depth}."
        )

    return before_data, after_data, channel_names


def _validate_prediction_qc_inputs(
    mean_bottom: np.ndarray,
    probability_bottom: np.ndarray,
    mask_bottom: np.ndarray,
    layers: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    mean_data = np.asarray(mean_bottom, dtype=np.float32)
    prob_data = np.asarray(probability_bottom, dtype=np.float32)
    mask_data = np.asarray(mask_bottom, dtype=np.uint8)
    layer_indices = tuple(int(layer) for layer in layers)

    if mean_data.ndim != 3:
        raise ValueError(f"Expected mean_bottom shape (D, X, Y), got {mean_data.shape}.")
    if prob_data.shape != mean_data.shape:
        raise ValueError(
            f"Probability shape {prob_data.shape} does not match mean shape {mean_data.shape}."
        )
    if mask_data.shape != mean_data.shape:
        raise ValueError(
            f"Mask shape {mask_data.shape} does not match mean shape {mean_data.shape}."
        )
    if not layer_indices:
        raise ValueError("Prediction QC expects at least one layer.")

    depth = int(mean_data.shape[0])
    invalid = [layer for layer in layer_indices if layer < 0 or layer >= depth]
    if invalid:
        raise ValueError(
            f"Prediction QC layer(s) {invalid} are outside depth 0:{depth}."
        )

    return mean_data, prob_data, mask_data, layer_indices
