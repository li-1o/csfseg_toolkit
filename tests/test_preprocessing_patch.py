import numpy as np
import pytest

from csfseg.preprocessing.patch import compute_axis_transform, make_patch


def test_compute_axis_transform_for_center_crop():
    result = compute_axis_transform(orig_len=7, out_len=5)

    assert result.crop_start == 1
    assert result.crop_stop == 6
    assert result.pad_pre == 0
    assert result.pad_post == 0


def test_compute_axis_transform_for_center_pad():
    result = compute_axis_transform(orig_len=3, out_len=6)

    assert result.crop_start == 0
    assert result.crop_stop == 3
    assert result.pad_pre == 1
    assert result.pad_post == 2


def test_make_patch_center_pads_xy_axes():
    bottom = np.ones((1, 2, 2, 3), dtype=np.float32)

    result = make_patch(bottom, out_hw=(4, 5))

    assert result.data.shape == (1, 2, 4, 5)
    assert result.source_shape == (1, 2, 2, 3)
    assert result.x.pad_pre == 1
    assert result.x.pad_post == 1
    assert result.y.pad_pre == 1
    assert result.y.pad_post == 1
    assert np.array_equal(result.data[:, :, 1:3, 1:4], bottom)
    assert np.count_nonzero(result.data[:, :, 0, :]) == 0
    assert np.count_nonzero(result.data[:, :, -1, :]) == 0


def test_make_patch_center_crops_xy_axes():
    bottom = np.arange(1 * 2 * 6 * 7, dtype=np.float32).reshape(1, 2, 6, 7)

    result = make_patch(bottom, out_hw=(4, 5))

    assert result.data.shape == (1, 2, 4, 5)
    assert result.x.crop_start == 1
    assert result.y.crop_start == 1
    assert np.array_equal(result.data, bottom[:, :, 1:5, 1:6])


def test_make_patch_rejects_non_4d_input():
    with pytest.raises(ValueError, match="Expected bottom volume"):
        make_patch(np.zeros((3, 10, 8), dtype=np.float32))


def test_make_patch_rejects_bad_output_size():
    with pytest.raises(ValueError, match="positive"):
        make_patch(np.zeros((3, 10, 8, 8), dtype=np.float32), out_hw=(128, 0))
