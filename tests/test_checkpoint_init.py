import torch
import pytest

from csfseg.inference.checkpoint import (
    infer_checkpoint_in_channels,
    initialize_3ch_from_1ch_checkpoint,
    load_checkpoint,
)
from csfseg.models.unet3d import UNet3D_NoZDown


def test_initialize_3ch_from_1ch_checkpoint(tmp_path):
    source_model = UNet3D_NoZDown(in_channels=1, base=16, out_channels=1)
    source_state = source_model.state_dict()
    source_state["enc1.0.weight"] = torch.arange(
        16 * 1 * 3 * 3 * 3,
        dtype=torch.float32,
    ).reshape(16, 1, 3, 3, 3)

    source_path = tmp_path / "old_1ch.pt"
    out_path = tmp_path / "init_3ch.pt"
    torch.save(source_state, source_path)

    result = initialize_3ch_from_1ch_checkpoint(
        source_path,
        out_path,
        seed=123,
        random_scale=0.01,
    )

    ckpt = load_checkpoint(out_path)
    new_state = ckpt["model"]
    first = new_state["enc1.0.weight"]

    assert first.shape == (16, 3, 3, 3, 3)
    assert torch.equal(first[:, 0:1], source_state["enc1.0.weight"])
    assert torch.count_nonzero(first[:, 1:]).item() > 0
    assert torch.equal(new_state["enc1.1.weight"], source_state["enc1.1.weight"])
    assert result.metadata_path.exists()
    assert result.metadata["source_in_channels"] == 1
    assert result.metadata["target_in_channels"] == 3
    assert result.metadata["requires_finetuning"] is True
    assert infer_checkpoint_in_channels(ckpt) == 3

    target_model = UNet3D_NoZDown(in_channels=3, base=16, out_channels=1)
    target_model.load_state_dict(new_state, strict=True)


def test_initialize_3ch_rejects_non_1ch_source(tmp_path):
    source_model = UNet3D_NoZDown(in_channels=3, base=16, out_channels=1)
    source_path = tmp_path / "already_3ch.pt"
    torch.save(source_model.state_dict(), source_path)

    with pytest.raises(ValueError, match="must be 1-channel"):
        initialize_3ch_from_1ch_checkpoint(
            source_path,
            tmp_path / "bad.pt",
        )
