from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from csfseg.checks.preflight import check_fmri_input
from csfseg.io.nifti import get_nifti_info, save_mask_like, save_probability_like


def make_nifti(path: Path, data: np.ndarray, zooms: tuple[float, ...]) -> None:
    img = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    img.header.set_zooms(zooms)
    nib.save(img, path)


def test_get_nifti_info_for_4d(tmp_path: Path):
    path = tmp_path / "sub-001_bold.nii.gz"
    make_nifti(path, np.zeros((9, 10, 11, 12), dtype=np.float32), (2.0, 2.0, 2.0, 0.8))

    info = get_nifti_info(path)

    assert info.shape == (9, 10, 11, 12)
    assert info.ndim == 4
    assert info.voxel_size == (2.0, 2.0, 2.0)
    assert info.tr == pytest.approx(0.8)


def test_preflight_accepts_basic_4d(tmp_path: Path):
    path = tmp_path / "sub-001_bold.nii.gz"
    make_nifti(path, np.zeros((9, 10, 11, 12), dtype=np.float32), (2.0, 2.0, 2.0, 0.8))

    result = check_fmri_input(path)

    assert result.ok
    assert result.info is not None


def test_preflight_rejects_3d(tmp_path: Path):
    path = tmp_path / "mask.nii.gz"
    make_nifti(path, np.zeros((9, 10, 11), dtype=np.float32), (2.0, 2.0, 2.0))

    result = check_fmri_input(path)

    assert not result.ok
    assert "Expected a 4D fMRI" in result.errors[0]


def test_save_mask_like_preserves_shape_and_affine(tmp_path: Path):
    ref_path = tmp_path / "sub-001_bold.nii.gz"
    make_nifti(ref_path, np.zeros((9, 10, 11, 12), dtype=np.float32), (2.0, 2.0, 2.0, 0.8))
    ref_img = nib.load(ref_path)
    mask = np.zeros((9, 10, 11), dtype=np.uint8)
    mask[1, 2, 3] = 1

    out_path = save_mask_like(mask, ref_img, tmp_path / "mask.nii.gz")
    out_img = nib.load(out_path)

    assert out_img.shape == (9, 10, 11)
    np.testing.assert_allclose(out_img.affine, ref_img.affine)


def test_save_probability_like_preserves_float_dtype(tmp_path: Path):
    ref_path = tmp_path / "sub-001_bold.nii.gz"
    make_nifti(ref_path, np.zeros((9, 10, 11, 12), dtype=np.float32), (2.0, 2.0, 2.0, 0.8))
    ref_img = nib.load(ref_path)
    probability = np.zeros((9, 10, 11), dtype=np.float32)
    probability[1, 2, 3] = 0.75

    out_path = save_probability_like(probability, ref_img, tmp_path / "prob.nii.gz")
    out_img = nib.load(out_path)

    assert out_img.shape == (9, 10, 11)
    assert out_img.header.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_allclose(out_img.affine, ref_img.affine)
