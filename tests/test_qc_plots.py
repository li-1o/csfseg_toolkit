import numpy as np
import pytest

from csfseg.qc.plots import (
    choose_preprocessing_qc_slice,
    choose_raw_time_indices,
    plot_bottom_volume_qc,
    plot_patch_qc,
    plot_prediction_qc,
    plot_timeseries_qc,
)
from csfseg.timeseries.components import analyze_mask_layers


def test_choose_raw_time_indices_uses_start_middle_end():
    assert choose_raw_time_indices(10) == (0, 5, 9)


def test_choose_preprocessing_qc_slice_uses_mean_nonzero_count():
    features = np.zeros((3, 4, 4, 3), dtype=np.float32)
    features[0, :1, :1, 0] = 1
    features[0, :, :, 1] = 1
    features[0, :2, :2, 2] = 1

    assert choose_preprocessing_qc_slice(features) == 1


def test_plot_bottom_volume_qc_writes_png(tmp_path):
    bottom = np.random.default_rng(0).normal(size=(3, 10, 8, 8)).astype(np.float32)
    out_path = tmp_path / "bottom_qc.png"

    result = plot_bottom_volume_qc(bottom, ("mean", "std", "tsnr"), out_path)

    assert result == out_path.resolve()
    assert out_path.stat().st_size > 0


def test_plot_bottom_volume_qc_rejects_wrong_depth(tmp_path):
    bottom = np.zeros((3, 9, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="depth=10"):
        plot_bottom_volume_qc(bottom, ("mean", "std", "tsnr"), tmp_path / "bad.png")


def test_plot_patch_qc_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    before = rng.normal(size=(3, 10, 8, 9)).astype(np.float32)
    after = rng.normal(size=(3, 10, 128, 128)).astype(np.float32)
    out_path = tmp_path / "patch_qc.png"

    result = plot_patch_qc(before, after, ("mean", "std", "tsnr"), out_path)

    assert result == out_path.resolve()
    assert out_path.stat().st_size > 0


def test_plot_patch_qc_rejects_mismatched_depth(tmp_path):
    before = np.zeros((3, 10, 8, 8), dtype=np.float32)
    after = np.zeros((3, 9, 128, 128), dtype=np.float32)

    with pytest.raises(ValueError, match="does not match"):
        plot_patch_qc(before, after, ("mean", "std", "tsnr"), tmp_path / "bad.png")


def test_plot_prediction_qc_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    mean_bottom = rng.normal(size=(10, 8, 9)).astype(np.float32)
    probability = rng.random(size=(10, 8, 9)).astype(np.float32)
    mask = (probability > 0.8).astype(np.uint8)
    analysis = analyze_mask_layers(mask, min_valid_voxels=2)
    out_path = tmp_path / "prediction_qc.png"

    result = plot_prediction_qc(
        mean_bottom,
        probability,
        mask,
        out_path,
        component_analysis=analysis,
    )

    assert result == out_path.resolve()
    assert out_path.stat().st_size > 0


def test_plot_prediction_qc_rejects_bad_layer(tmp_path):
    mean_bottom = np.zeros((2, 8, 8), dtype=np.float32)
    probability = np.zeros((2, 8, 8), dtype=np.float32)
    mask = np.zeros((2, 8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="outside"):
        plot_prediction_qc(mean_bottom, probability, mask, tmp_path / "bad.png")


def test_plot_timeseries_qc_writes_png(tmp_path):
    table = {
        "layer": ["L0", "L0", "L0"],
        "is_isolated": [False, False, True],
        "has_zero_dropout": [False, True, False],
        "has_nan_or_inf": [False, False, False],
        "t0": [1.0, 1.0, 1.0],
        "t1": [2.0, 0.0, 1.5],
        "t2": [3.0, 0.0, 2.0],
    }
    out_path = tmp_path / "timeseries_qc.png"

    result = plot_timeseries_qc(
        table,
        out_path,
        output_id="sub001",
        selected_layer="L0",
        selection_name="auto",
        min_valid_voxels=2,
    )

    assert result == out_path.resolve()
    assert out_path.stat().st_size > 0
