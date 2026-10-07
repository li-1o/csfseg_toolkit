import numpy as np

from csfseg.timeseries.components import analyze_mask_layers, normalize_layer_name


def test_analyze_mask_layers_counts_isolated_and_auto_layer():
    mask = np.zeros((4, 6, 6), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 1:3, 1:4] = 1
    mask[2, 4, 4] = 1

    analysis = analyze_mask_layers(mask, layers=(0, 1, 2, 3), min_valid_voxels=5)

    l0 = analysis.stats_for(0)
    l1 = analysis.stats_for(1)
    assert l0.mask_voxels == 1
    assert l0.isolated_voxels == 1
    assert not l0.is_candidate_layer
    assert l1.mask_voxels == 6
    assert l1.valid_voxels == 6
    assert l1.is_candidate_layer
    assert analysis.auto_layer == 1


def test_normalize_layer_name_accepts_common_forms():
    assert normalize_layer_name("L1") == "L1"
    assert normalize_layer_name("1") == "L1"
    assert normalize_layer_name(2) == "L2"
    assert normalize_layer_name(None) == ""
