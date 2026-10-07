"""Basic checks for input 4D fMRI NIfTI files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from csfseg.io.nifti import NiftiInfo, get_nifti_info


@dataclass(frozen=True)
class PreflightResult:
    input_path: Path
    ok: bool
    info: NiftiInfo | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "ok": self.ok,
            "info": self.info.to_dict() if self.info is not None else None,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def check_fmri_input(
    path: str | Path,
    min_timepoints: int = 10,
    min_spatial_dim: int = 8,
) -> PreflightResult:
    """Check whether a NIfTI file looks usable as a 4D fMRI input.

    This does not verify preprocessing provenance. It only checks basic file
    and header properties that the toolkit needs downstream.
    """
    input_path = Path(path).expanduser().resolve()
    warnings: list[str] = []
    errors: list[str] = []

    try:
        info = get_nifti_info(input_path)
    except Exception as exc:
        return PreflightResult(input_path=input_path, ok=False, info=None, errors=[str(exc)])

    # A 4D input is required because downstream steps compute temporal
    # features and extract a CSF time series. A 3D file is probably a mask,
    # anatomy, or already-derived image.
    if info.ndim != 4:
        errors.append(f"Expected a 4D fMRI NIfTI, got shape {info.shape}.")
    else:
        spatial_shape = info.shape[:3]
        n_timepoints = info.shape[3]
        # These are broad guardrails, not quality criteria. They catch clearly
        # wrong files while still allowing short task-fMRI runs and non-HCP data.
        if any(dim < min_spatial_dim for dim in spatial_shape):
            errors.append(f"Spatial shape looks too small for fMRI: {spatial_shape}.")
        if n_timepoints < min_timepoints:
            errors.append(f"Too few timepoints: {n_timepoints} < {min_timepoints}.")

    # Voxel size is not required for the neural network math itself, but it is
    # important for reporting and for warning when a dataset is far from the
    # model's reference resolution.
    if len(info.voxel_size) < 3:
        warnings.append("Voxel size is missing or incomplete in the NIfTI header.")
    elif any(v <= 0 for v in info.voxel_size):
        warnings.append(f"Voxel size contains non-positive values: {info.voxel_size}.")

    # TR is useful for reports and time-series QC. Segmentation can still run
    # without it, so missing or invalid TR is a warning rather than an error.
    if info.tr is None:
        warnings.append("TR is missing because the input is not reported as 4D in the header.")
    elif info.tr <= 0:
        warnings.append(f"TR is non-positive: {info.tr}.")

    # The output mask is a plain 3D array until we save it as NIfTI. A valid
    # affine lets viewers place that mask back into the same space as the input.
    if info.affine.shape != (4, 4) or not np.isfinite(info.affine).all():
        errors.append("Affine matrix is missing or contains non-finite values.")

    # qform/sform codes are not used for prediction, but missing spatial form
    # codes can make downstream overlay/viewing less reliable.
    if info.qform_code == 0 and info.sform_code == 0:
        warnings.append("Both qform_code and sform_code are 0; spatial orientation metadata may be incomplete.")

    return PreflightResult(input_path=input_path, ok=not errors, info=info, warnings=warnings, errors=errors)
