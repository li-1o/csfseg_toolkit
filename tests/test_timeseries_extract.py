import numpy as np

from csfseg.timeseries.components import analyze_mask_layers
from csfseg.timeseries.extract import extract_voxel_table, select_voxels


def test_extract_voxel_table_tags_isolated_and_dropout():
    fmri = np.ones((5, 5, 4, 10), dtype=np.float32)
    mask = np.zeros((4, 5, 5), dtype=np.uint8)
    mask[0, 4, 4] = 1
    mask[0, 1, 1] = 1
    mask[0, 1, 2] = 1
    fmri[1, 2, 0, :2] = 0.0

    analysis = analyze_mask_layers(mask, min_valid_voxels=2)
    table = extract_voxel_table(
        fmri,
        analysis,
        "sub001",
        zero_fraction_threshold=0.10,
    )

    assert len(table) == 3
    isolated = table[table["is_isolated"]]
    assert isolated[["x", "y"]].to_dict(orient="records") == [{"x": 4, "y": 4}]
    dropout = table[table["has_zero_dropout"]]
    assert dropout[["x", "y"]].to_dict(orient="records") == [{"x": 1, "y": 2}]


def test_select_voxels_excludes_isolated_and_zero_dropout():
    fmri = np.ones((5, 5, 4, 10), dtype=np.float32)
    mask = np.zeros((4, 5, 5), dtype=np.uint8)
    mask[0, 4, 4] = 1
    mask[0, 1, 1] = 1
    mask[0, 1, 2] = 1
    fmri[1, 2, 0, :2] = 0.0

    analysis = analyze_mask_layers(mask, min_valid_voxels=2)
    table = extract_voxel_table(fmri, analysis, "sub001")
    result = select_voxels(
        table,
        output_id="sub001",
        selected_layer="L0",
        min_valid_voxels=2,
        selection_name="auto",
    )

    assert len(result.selected) == 1
    assert result.selected.iloc[0]["x"] == 1
    assert result.selected.iloc[0]["y"] == 1
    assert result.summary["isolated_voxels"] == 1
    assert result.summary["dropout_voxels"] == 1
    assert result.summary["selected_voxels"] == 1
