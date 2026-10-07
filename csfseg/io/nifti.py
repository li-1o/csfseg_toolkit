"""NIfTI input/output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class NiftiInfo:
    path: Path
    shape: tuple[int, ...]
    ndim: int
    dtype: str
    voxel_size: tuple[float, ...]
    tr: float | None
    affine: np.ndarray
    qform_code: int
    sform_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "shape": list(self.shape),
            "ndim": self.ndim,
            "dtype": self.dtype,
            "voxel_size": list(self.voxel_size),
            "tr": self.tr,
            "affine": self.affine.tolist(),
            "qform_code": self.qform_code,
            "sform_code": self.sform_code,
        }


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI image without reading the full data array into memory."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"NIfTI path is not a file: {path}")
    return nib.load(str(path))


def get_nifti_info(path: str | Path) -> NiftiInfo:
    """Read basic NIfTI metadata from the header."""
    path = Path(path).expanduser().resolve()
    img = load_nifti(path)
    header = img.header
    shape = tuple(int(v) for v in img.shape)
    zooms = tuple(float(v) for v in header.get_zooms())
    spatial_zooms = zooms[: min(3, len(zooms))]
    tr = float(zooms[3]) if len(zooms) >= 4 and len(shape) >= 4 else None

    return NiftiInfo(
        path=path,
        shape=shape,
        ndim=len(shape),
        dtype=str(header.get_data_dtype()),
        voxel_size=spatial_zooms,
        tr=tr,
        affine=np.array(img.affine, dtype=float),
        qform_code=int(header["qform_code"]),
        sform_code=int(header["sform_code"]),
    )


def save_volume_like(
    volume_data: np.ndarray,
    reference_img: nib.Nifti1Image,
    out_path: str | Path,
    dtype: np.dtype | type,
) -> Path:
    """Save a 3D volume using spatial metadata from a reference image."""

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if volume_data.ndim != 3:
        raise ValueError(f"Volume must be 3D, got shape {volume_data.shape}")

    header = reference_img.header.copy()
    header.set_data_dtype(dtype)

    out_img = nib.Nifti1Image(volume_data.astype(dtype), reference_img.affine, header)
    qform = reference_img.get_qform()
    sform = reference_img.get_sform()
    qform_code = int(reference_img.header["qform_code"])
    sform_code = int(reference_img.header["sform_code"])
    out_img.set_qform(qform, code=qform_code)
    out_img.set_sform(sform, code=sform_code)

    nib.save(out_img, str(out_path))
    return out_path


def save_mask_like(mask_data: np.ndarray, reference_img: nib.Nifti1Image, out_path: str | Path) -> Path:
    """Save a 3D binary mask using spatial metadata from a reference image."""

    return save_volume_like(mask_data, reference_img, out_path, np.uint8)


def save_probability_like(probability_data: np.ndarray, reference_img: nib.Nifti1Image, out_path: str | Path) -> Path:
    """Save a 3D probability map using spatial metadata from a reference image."""

    return save_volume_like(probability_data, reference_img, out_path, np.float32)
