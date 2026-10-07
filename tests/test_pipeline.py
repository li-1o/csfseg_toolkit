from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from csfseg.models.unet3d import UNet3D_NoZDown
from csfseg.pipeline import SingleRunConfig, run_single
from csfseg.utils.logging import close_logger, setup_subject_logger


def _write_nifti(path: Path, data: np.ndarray) -> None:
    img = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    img.header.set_zooms((2.0, 2.0, 2.0, 0.8))
    nib.save(img, path)


def _write_checkpoint(path: Path, base: int = 1) -> None:
    model = UNet3D_NoZDown(in_channels=3, base=base, out_channels=1)
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {
                "requires_finetuning": False,
                "in_channels": 3,
                "channels": ["mean", "std", "tsnr"],
            },
        },
        path,
    )


def test_run_single_writes_outputs_and_subject_log(tmp_path):
    rng = np.random.default_rng(0)
    input_path = tmp_path / "sub001_bold.nii.gz"
    checkpoint_path = tmp_path / "model.pt"
    out_dir = tmp_path / "outputs"
    data = rng.normal(size=(6, 7, 12, 6)).astype(np.float32)
    _write_nifti(input_path, data)
    _write_checkpoint(checkpoint_path)

    config = SingleRunConfig(
        input_path=input_path,
        out_dir=out_dir,
        checkpoint=checkpoint_path,
        output_id="sub001",
        device="cpu",
        threshold=0.0,
        min_valid_voxels=5,
        model_base=1,
    )
    logger, log_path, run_id = setup_subject_logger("sub001", out_dir, run_id="run-test")

    try:
        result = run_single(config, logger=logger, log_path=log_path, run_id=run_id)
    finally:
        close_logger(logger)

    assert result.status == "ok"
    assert result.output_id == "sub001"
    assert result.auto_layer == "L0"
    assert result.mask_voxels == 6 * 7 * 10
    assert result.selected_voxels == 6 * 7
    assert result.patch_shape == (3, 10, 128, 128)
    assert result.probability_path.exists()
    assert result.mask_path.exists()
    assert result.prediction_qc_path.exists()
    assert result.voxel_table_path.exists()
    assert result.selection_template_path.exists()
    assert result.selected_voxels_path is not None
    assert result.selected_voxels_path.exists()
    assert result.timeseries_qc_path is not None
    assert result.timeseries_qc_path.exists()
    assert result.auto_summary_path is not None
    assert result.auto_summary_path.exists()
    assert result.log_path == log_path.resolve()

    log_text = log_path.read_text()
    assert "output_id: sub001" in log_text
    assert "input_path:" in log_text
    assert "[START] load_nifti" in log_text
    assert "[DONE] auto_selection" in log_text
    assert "probability_path:" in log_text
    assert "mask_voxels: 420" in log_text
