import numpy as np

from csfseg.preprocessing.features import compute_temporal_features


def test_compute_temporal_features_returns_channel_first_stack():
    fmri = np.zeros((2, 3, 4, 5), dtype=np.float32)
    fmri[..., 0] = 1
    fmri[..., 1] = 2
    fmri[..., 2] = 3
    fmri[..., 3] = 4
    fmri[..., 4] = 5

    result = compute_temporal_features(fmri)

    assert result.channels == ("mean", "std", "tsnr")
    assert result.data.shape == (3, 2, 3, 4)
    assert np.allclose(result.data[0], 3.0)
    assert np.all(result.data[1] > 0)
    assert set(result.stats) == {"mean", "std", "tsnr"}


def test_compute_temporal_features_can_choose_channel_order():
    fmri = np.ones((2, 2, 2, 3), dtype=np.float32)

    result = compute_temporal_features(fmri, channels=("tsnr", "mean"))

    assert result.channels == ("tsnr", "mean")
    assert result.data.shape == (2, 2, 2, 2)
