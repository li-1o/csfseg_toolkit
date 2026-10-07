import numpy as np
import torch

from csfseg.inference.predict import load_model_for_inference, predict_patch
from csfseg.models.unet3d import UNet3D_NoZDown


def test_predict_patch_returns_probability_map(tmp_path):
    model = UNet3D_NoZDown(in_channels=3, base=4, out_channels=1)
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "metadata": {"requires_finetuning": False}}, checkpoint_path)

    loaded = load_model_for_inference(
        checkpoint_path,
        device="cpu",
        in_channels=3,
        base=4,
        out_channels=1,
    )
    patch = np.zeros((3, 2, 16, 16), dtype=np.float32)

    probability = predict_patch(patch, loaded)

    assert probability.shape == (2, 16, 16)
    assert probability.dtype == np.float32
    assert np.all(probability >= 0)
    assert np.all(probability <= 1)
