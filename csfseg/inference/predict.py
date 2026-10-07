"""Run 3D U-Net inference on a prepared model patch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from csfseg.inference.checkpoint import (
    extract_state_dict,
    infer_checkpoint_in_channels,
    load_checkpoint,
)
from csfseg.models.unet3d import UNet3D_NoZDown


@dataclass(frozen=True)
class LoadedModel:
    """Model object plus useful checkpoint context."""

    model: UNet3D_NoZDown
    device: torch.device
    checkpoint_path: Path
    metadata: dict[str, Any]


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto``/string device requests to a torch device."""

    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model_for_inference(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "auto",
    in_channels: int = 3,
    base: int = 16,
    out_channels: int = 1,
) -> LoadedModel:
    """Load a checkpoint into the toolkit 3D U-Net for inference."""

    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    torch_device = resolve_device(device)
    checkpoint = load_checkpoint(ckpt_path, map_location=torch_device)
    checkpoint_in_channels = infer_checkpoint_in_channels(checkpoint)
    if checkpoint_in_channels != in_channels:
        raise ValueError(
            f"Checkpoint expects {checkpoint_in_channels} input channel(s), "
            f"but this model expects {in_channels}."
        )

    state_dict = extract_state_dict(checkpoint)
    model = UNet3D_NoZDown(
        in_channels=in_channels,
        base=base,
        out_channels=out_channels,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(torch_device)
    model.eval()

    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    return LoadedModel(
        model=model,
        device=torch_device,
        checkpoint_path=ckpt_path,
        metadata=metadata,
    )


def predict_patch(
    patch_data: np.ndarray,
    loaded_model: LoadedModel | UNet3D_NoZDown,
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Predict a probability map for one prepared patch.

    Args:
        patch_data: Float array shaped ``(C, D, H, W)``.
        loaded_model: A :class:`LoadedModel` from :func:`load_model_for_inference`
            or a raw ``UNet3D_NoZDown`` model.
        device: Optional device override when passing a raw model.

    Returns:
        Probability array shaped ``(D, H, W)``.
    """

    patch = _validate_patch_data(patch_data)
    model, torch_device = _resolve_model_and_device(loaded_model, device)

    tensor = torch.from_numpy(patch).unsqueeze(0).to(torch_device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)

    if probabilities.ndim != 5 or probabilities.shape[0] != 1 or probabilities.shape[1] != 1:
        raise ValueError(
            "Model output must have shape (1, 1, D, H, W), "
            f"got {tuple(probabilities.shape)}."
        )

    return probabilities[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def _validate_patch_data(patch_data: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch_data, dtype=np.float32)
    if patch.ndim != 4:
        raise ValueError(f"Expected patch shape (C, D, H, W), got {patch.shape}.")
    if patch.shape[0] <= 0 or patch.shape[1] <= 0:
        raise ValueError(f"Patch C/D axes must be nonempty, got {patch.shape}.")
    if patch.shape[2] <= 0 or patch.shape[3] <= 0:
        raise ValueError(f"Patch H/W axes must be nonempty, got {patch.shape}.")
    return np.ascontiguousarray(patch, dtype=np.float32)


def _resolve_model_and_device(
    loaded_model: LoadedModel | UNet3D_NoZDown,
    device: str | torch.device | None,
) -> tuple[UNet3D_NoZDown, torch.device]:
    if isinstance(loaded_model, LoadedModel):
        return loaded_model.model, loaded_model.device

    torch_device = resolve_device(device or "auto")
    loaded_model.to(torch_device)
    loaded_model.eval()
    return loaded_model, torch_device


__all__ = [
    "LoadedModel",
    "load_model_for_inference",
    "predict_patch",
    "resolve_device",
]
