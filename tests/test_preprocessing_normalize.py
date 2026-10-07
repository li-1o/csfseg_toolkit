import numpy as np
import pytest

from csfseg.preprocessing.normalize import normalize_features


def test_nonzero_zscore_keeps_background_zero():
    features = np.array(
        [
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
            ]
        ],
        dtype=np.float32,
    )

    result = normalize_features(features, channels=("mean",))

    assert result.data.shape == features.shape
    assert result.data[0, 0, 0, 0] == 0
    assert result.stats["mean"].valid_voxels == 7
    assert abs(float(result.data[0][features[0] != 0].mean())) < 1e-6


def test_nonzero_zscore_rejects_flat_channel():
    features = np.ones((1, 2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="too small"):
        normalize_features(features, channels=("mean",))
