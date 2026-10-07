import torch

from csfseg.models.unet3d import UNet3D_NoZDown


def test_unet3d_no_z_down_preserves_patch_shape():
    model = UNet3D_NoZDown(in_channels=3, base=16, out_channels=1)
    x = torch.zeros((1, 3, 10, 128, 128), dtype=torch.float32)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 10, 128, 128)
    assert model.enc1[0].weight.shape == (16, 3, 3, 3, 3)
