"""Voxel-level CSF time-series tables and selected-table export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from csfseg.timeseries.components import (
    LayerComponentAnalysis,
    normalize_layer_name,
)


DEFAULT_ZERO_EPSILON = 1e-6
DEFAULT_ZERO_FRACTION_THRESHOLD = 0.10
DEFAULT_DROPOUT_WARNING_FRACTION = 0.20


@dataclass(frozen=True)
class SelectionResult:
    """Selected voxel table plus the summary row that explains the filter."""

    selected: pd.DataFrame
    layer_table: pd.DataFrame
    summary: dict[str, object]


def extract_voxel_table(
    fmri_4d: np.ndarray,
    analysis: LayerComponentAnalysis,
    output_id: str,
    *,
    z_start: int = 0,
    zero_epsilon: float = DEFAULT_ZERO_EPSILON,
    zero_fraction_threshold: float = DEFAULT_ZERO_FRACTION_THRESHOLD,
) -> pd.DataFrame:
    """Extract all L0-L3 mask voxel time series into one wide table.

    No rows are removed here.  Isolated voxels and zero-dropout voxels are only
    tagged so QC and later selected exports can stay transparent.
    """

    fmri = _validate_fmri(fmri_4d, analysis, z_start, zero_epsilon, zero_fraction_threshold)
    time_columns = [f"t{i}" for i in range(int(fmri.shape[3]))]
    rows: list[dict[str, object]] = []

    for stats in analysis.stats:
        labels = analysis.labels[stats.layer]
        coords = np.argwhere(labels > 0)
        for x, y in coords:
            component_id = int(labels[x, y])
            component_size = analysis.component_size(stats.layer, component_id)
            source_z = int(z_start + stats.layer)
            series = np.asarray(fmri[int(x), int(y), source_z, :], dtype=np.float32)
            finite = np.isfinite(series)
            zero_fraction = float(np.mean(np.abs(series) < zero_epsilon))
            has_nan_or_inf = bool(not np.all(finite))
            has_zero_dropout = bool(zero_fraction > zero_fraction_threshold)

            row: dict[str, object] = {
                "output_id": output_id,
                "layer": stats.layer_name,
                "layer_index": stats.layer,
                "x": int(x),
                "y": int(y),
                "z": source_z,
                "component_id": component_id,
                "component_size": component_size,
                "is_isolated": bool(component_size == 1),
                "is_candidate_layer": bool(stats.is_candidate_layer),
                "is_auto_layer": bool(stats.is_auto_layer),
                "zero_fraction": zero_fraction,
                "has_zero_dropout": has_zero_dropout,
                "has_nan_or_inf": has_nan_or_inf,
            }
            row.update({column: float(value) for column, value in zip(time_columns, series)})
            rows.append(row)

    columns = [
        "output_id",
        "layer",
        "layer_index",
        "x",
        "y",
        "z",
        "component_id",
        "component_size",
        "is_isolated",
        "is_candidate_layer",
        "is_auto_layer",
        "zero_fraction",
        "has_zero_dropout",
        "has_nan_or_inf",
        *time_columns,
    ]
    return pd.DataFrame(rows, columns=columns)


def select_voxels(
    voxel_table: pd.DataFrame,
    *,
    output_id: str,
    selected_layer: str | int | None,
    min_valid_voxels: int,
    selection_name: str,
    exclude: bool = False,
    dropout_warning_fraction: float = DEFAULT_DROPOUT_WARNING_FRACTION,
) -> SelectionResult:
    """Return the analysis-ready subset for one output id and selected layer."""

    table = _normalize_voxel_table(voxel_table)
    layer_name = normalize_layer_name(selected_layer)
    summary: dict[str, object] = {
        "output_id": output_id,
        "selection_name": selection_name,
        "selected_layer": layer_name,
        "min_valid_voxels": int(min_valid_voxels),
        "exclude": bool(exclude),
        "status": "ok",
        "total_voxels": 0,
        "valid_voxels": 0,
        "isolated_voxels": 0,
        "dropout_voxels": 0,
        "nan_or_inf_voxels": 0,
        "selected_voxels": 0,
        "dropout_fraction": 0.0,
        "warning": "",
    }

    if exclude:
        summary["status"] = "excluded_by_user"
        return SelectionResult(pd.DataFrame(columns=table.columns), pd.DataFrame(columns=table.columns), summary)

    if not layer_name:
        summary["status"] = "no_selected_layer"
        summary["warning"] = "no_selected_layer"
        return SelectionResult(pd.DataFrame(columns=table.columns), pd.DataFrame(columns=table.columns), summary)

    layer_table = table[table["layer"] == layer_name].copy()
    summary["total_voxels"] = int(len(layer_table))
    if layer_table.empty:
        summary["status"] = "empty_layer"
        summary["warning"] = "selected_layer_has_no_mask_voxels"
        return SelectionResult(pd.DataFrame(columns=table.columns), layer_table, summary)

    isolated = _bool_series(layer_table["is_isolated"])
    dropout = _bool_series(layer_table["has_zero_dropout"])
    invalid = _bool_series(layer_table["has_nan_or_inf"])
    valid_component = ~isolated

    valid_count = int(valid_component.sum())
    dropout_count = int((valid_component & dropout).sum())
    invalid_count = int((valid_component & invalid).sum())
    selected_mask = valid_component & ~dropout & ~invalid
    selected = layer_table[selected_mask].copy()
    dropout_fraction = float(dropout_count / valid_count) if valid_count else 0.0

    warnings: list[str] = []
    if valid_count < int(min_valid_voxels):
        warnings.append("selected_layer_below_min_valid_voxels")
    if dropout_fraction >= float(dropout_warning_fraction) and valid_count:
        warnings.append("high_dropout_fraction")
    if invalid_count:
        warnings.append("nan_or_inf_voxels_excluded")
    if selected.empty:
        warnings.append("no_selected_voxels_after_filters")

    summary.update(
        {
            "valid_voxels": valid_count,
            "isolated_voxels": int(isolated.sum()),
            "dropout_voxels": dropout_count,
            "nan_or_inf_voxels": invalid_count,
            "selected_voxels": int(len(selected)),
            "dropout_fraction": dropout_fraction,
            "warning": ";".join(warnings),
        }
    )
    if selected.empty:
        summary["status"] = "empty_after_filters"
    elif warnings:
        summary["status"] = "warning"

    return SelectionResult(selected, layer_table, summary)


def write_voxel_table(table: pd.DataFrame, out_path: str | Path) -> Path:
    """Write a voxel table CSV."""

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return out_path


def read_voxel_table(path: str | Path) -> pd.DataFrame:
    """Read a voxel table CSV."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Voxel table does not exist: {path}")
    return _normalize_voxel_table(pd.read_csv(path))


def update_selection_template(
    template_path: str | Path,
    *,
    output_id: str,
    selected_layer: str | int | None,
    min_valid_voxels: int,
    exclude: bool = False,
    notes: str = "",
) -> Path:
    """Add or replace one row in the user-editable selection template."""

    template_path = Path(template_path).expanduser().resolve()
    template_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "output_id": output_id,
        "selected_layer": normalize_layer_name(selected_layer),
        "min_valid_voxels": int(min_valid_voxels),
        "exclude": bool(exclude),
        "notes": notes,
    }
    if template_path.exists():
        table = pd.read_csv(template_path)
        table = table[table["output_id"] != output_id]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table.to_csv(template_path, index=False)
    return template_path


def read_selection_file(path: str | Path, default_min_valid_voxels: int) -> pd.DataFrame:
    """Read a user-edited selection file and normalize required columns."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Selection file does not exist: {path}")
    table = pd.read_csv(path)
    if "output_id" not in table.columns or "selected_layer" not in table.columns:
        raise ValueError("Selection file must contain output_id and selected_layer columns.")
    if "min_valid_voxels" not in table.columns:
        table["min_valid_voxels"] = int(default_min_valid_voxels)
    if "exclude" not in table.columns:
        table["exclude"] = False
    if "notes" not in table.columns:
        table["notes"] = ""
    table["selected_layer"] = [normalize_layer_name(value) for value in table["selected_layer"]]
    table["min_valid_voxels"] = table["min_valid_voxels"].fillna(default_min_valid_voxels).astype(int)
    table["exclude"] = _bool_series(table["exclude"])
    table["notes"] = table["notes"].fillna("")
    return table


def update_summary_table(summary_path: str | Path, rows: list[dict[str, object]]) -> Path:
    """Write or replace selection summary rows keyed by output_id."""

    summary_path = Path(summary_path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(rows)
    if summary_path.exists():
        old_rows = pd.read_csv(summary_path)
        if "output_id" in old_rows.columns:
            old_rows = old_rows[~old_rows["output_id"].isin(new_rows["output_id"])]
            new_rows = pd.concat([old_rows, new_rows], ignore_index=True)
    new_rows.to_csv(summary_path, index=False)
    return summary_path


def time_columns(table: pd.DataFrame) -> list[str]:
    """Return time-point columns in numeric order."""

    columns = [column for column in table.columns if column.startswith("t") and column[1:].isdigit()]
    return sorted(columns, key=lambda name: int(name[1:]))


def _validate_fmri(
    fmri_4d: np.ndarray,
    analysis: LayerComponentAnalysis,
    z_start: int,
    zero_epsilon: float,
    zero_fraction_threshold: float,
) -> np.ndarray:
    data = np.asarray(fmri_4d, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected fMRI shape (X, Y, Z, T), got {data.shape}.")
    expected_xy = analysis.mask_shape[1:]
    if tuple(data.shape[:2]) != expected_xy:
        raise ValueError(
            f"fMRI X/Y shape {data.shape[:2]} does not match mask X/Y shape {expected_xy}."
        )
    max_z = z_start + max(analysis.layers)
    if max_z >= data.shape[2]:
        raise ValueError(f"Requested layer z={max_z} outside fMRI Z={data.shape[2]}.")
    if data.shape[3] <= 0:
        raise ValueError("fMRI time dimension must be nonempty.")
    if zero_epsilon < 0:
        raise ValueError(f"zero_epsilon must be nonnegative, got {zero_epsilon}.")
    if not 0 <= zero_fraction_threshold <= 1:
        raise ValueError(
            f"zero_fraction_threshold must be in [0, 1], got {zero_fraction_threshold}."
        )
    return data


def _normalize_voxel_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "output_id",
        "layer",
        "is_isolated",
        "has_zero_dropout",
        "has_nan_or_inf",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Voxel table is missing required column(s): {missing}.")
    normalized = table.copy()
    normalized["layer"] = [normalize_layer_name(value) for value in normalized["layer"]]
    for column in ("is_isolated", "is_candidate_layer", "is_auto_layer", "has_zero_dropout", "has_nan_or_inf"):
        if column in normalized.columns:
            normalized[column] = _bool_series(normalized[column])
    return normalized


def _bool_series(series) -> pd.Series:
    if isinstance(series, pd.Series) and series.dtype == bool:
        return series
    if isinstance(series, pd.Series):
        return series.map(_to_bool).astype(bool)
    return pd.Series(series).map(_to_bool).astype(bool)


def _to_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


__all__ = [
    "DEFAULT_DROPOUT_WARNING_FRACTION",
    "DEFAULT_ZERO_EPSILON",
    "DEFAULT_ZERO_FRACTION_THRESHOLD",
    "SelectionResult",
    "extract_voxel_table",
    "read_selection_file",
    "read_voxel_table",
    "select_voxels",
    "time_columns",
    "update_selection_template",
    "update_summary_table",
    "write_voxel_table",
]
