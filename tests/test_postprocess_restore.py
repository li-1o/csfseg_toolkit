import numpy as np
import pytest

from csfseg.inference.postprocess import (
    restore_bottom_to_full,
    restore_patch_to_bottom,
    restore_prediction_to_full,
    threshold_probability,
)
from csfseg.preprocessing.bottom import extract_bottom_volume
from csfseg.preprocessing.patch import make_patch


def test_threshold_probability_returns_uint8_mask():
    probability = np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32)

    mask = threshold_probability(probability, threshold=0.5)

    assert mask.dtype == np.uint8
    assert mask.tolist() == [[[0, 1, 1]]]


def test_restore_patch_to_bottom_removes_padding():
    bottom_data = np.zeros((3, 2, 2, 3), dtype=np.float32)
    patch = make_patch(bottom_data, out_hw=(4, 5))
    patch_map = np.zeros((2, 4, 5), dtype=np.float32)
    patch_map[:, 1:3, 1:4] = 7.0

    restored = restore_patch_to_bottom(patch_map, patch)

    assert restored.shape == (2, 2, 3)
    assert np.all(restored == 7.0)


def test_restore_patch_to_bottom_places_crop_back_in_original_xy_grid():
    bottom_data = np.zeros((3, 2, 6, 7), dtype=np.float32)
    patch = make_patch(bottom_data, out_hw=(4, 5))
    patch_map = np.ones((2, 4, 5), dtype=np.float32)

    restored = restore_patch_to_bottom(patch_map, patch)

    assert restored.shape == (2, 6, 7)
    assert np.all(restored[:, 1:5, 1:6] == 1.0)
    assert np.count_nonzero(restored[:, 0, :]) == 0
    assert np.count_nonzero(restored[:, -1, :]) == 0
    assert np.count_nonzero(restored[:, :, 0]) == 0
    assert np.count_nonzero(restored[:, :, -1]) == 0


def test_restore_bottom_to_full_places_bottom_at_z_start():
    features = np.zeros((3, 4, 5, 8), dtype=np.float32)
    bottom = extract_bottom_volume(
        features,
        depth=2,
        z_start=1,
        candidate_layers=(0, 1),
    )
    bottom_map = np.ones((2, 4, 5), dtype=np.uint8)

    full = restore_bottom_to_full(bottom_map, bottom, spatial_shape=(4, 5, 8))

    assert full.shape == (4, 5, 8)
    assert np.count_nonzero(full[:, :, 0]) == 0
    assert np.all(full[:, :, 1:3] == 1)
    assert np.count_nonzero(full[:, :, 3:]) == 0


def test_restore_prediction_to_full_returns_probability_and_mask():
    features = np.zeros((3, 4, 5, 10), dtype=np.float32)
    bottom = extract_bottom_volume(
        features,
        depth=2,
        z_start=0,
        candidate_layers=(0, 1),
    )
    patch = make_patch(bottom.data, out_hw=(6, 7))
    probability_patch = np.zeros((2, 6, 7), dtype=np.float32)
    probability_patch[:, 1:5, 1:6] = 0.8

    result = restore_prediction_to_full(
        probability_patch,
        patch,
        bottom,
        spatial_shape=(4, 5, 10),
        threshold=0.5,
    )

    assert result.probability_full.shape == (4, 5, 10)
    assert result.mask_full.dtype == np.uint8
    assert int(result.mask_full.sum()) == 2 * 4 * 5


def test_restore_patch_to_bottom_rejects_wrong_shape():
    bottom_data = np.zeros((3, 2, 4, 5), dtype=np.float32)
    patch = make_patch(bottom_data, out_hw=(4, 5))

    with pytest.raises(ValueError, match="Patch map shape"):
        restore_patch_to_bottom(np.zeros((2, 5, 5), dtype=np.float32), patch)
