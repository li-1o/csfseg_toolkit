import numpy as np
import pytest

from csfseg.preprocessing.bottom import extract_bottom_volume


def test_extract_bottom_volume_reorders_z_to_depth_axis():
    features = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6)

    result = extract_bottom_volume(
        features,
        depth=2,
        z_start=1,
        candidate_layers=(0, 1),
    )

    assert result.data.shape == (3, 2, 4, 5)
    assert result.source_shape == (3, 4, 5, 6)
    assert result.z_start == 1
    assert result.z_stop == 3
    assert result.depth == 2
    assert result.candidate_layers == (0, 1)
    assert np.array_equal(result.data[:, 0, :, :], features[:, :, :, 1])
    assert np.array_equal(result.data[:, 1, :, :], features[:, :, :, 2])


def test_extract_bottom_volume_rejects_short_z_axis():
    features = np.zeros((3, 4, 5, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="Cannot extract"):
        extract_bottom_volume(features, depth=10)


def test_extract_bottom_volume_rejects_candidate_layers_outside_depth():
    features = np.zeros((3, 4, 5, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="outside"):
        extract_bottom_volume(features, depth=2, candidate_layers=(0, 2))
