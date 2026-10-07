"""Checkpoint loading and 1-channel to 3-channel initialization helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from csfseg.models.unet3d import UNet3D_NoZDown


FIRST_CONV_KEY = "enc1.0.weight"
DEFAULT_CHANNEL_ORDER = ("mean", "std", "tsnr")


@dataclass(frozen=True)
class CheckpointInitResult:
    """Result returned after writing a migrated initialization checkpoint."""

    checkpoint_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a PyTorch checkpoint from disk.

    PyTorch checkpoints are pickle-backed, so they should only be loaded from a
    trusted source.  The toolkit asks for explicit checkpoint paths for this
    reason.
    """

    ckpt_path = Path(path).expanduser().resolve()
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=map_location)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Return the model state dict from common checkpoint layouts."""

    if isinstance(checkpoint, Mapping):
        if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
            return checkpoint["model"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
            return checkpoint["state_dict"]
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint

    raise ValueError(
        "Could not find a model state dict. Expected a direct state_dict or a "
        "checkpoint containing a 'model' or 'state_dict' mapping."
    )


def infer_checkpoint_in_channels(
    checkpoint_or_state_dict: Any,
    first_conv_key: str = FIRST_CONV_KEY,
) -> int:
    """Infer input channel count from the first convolution weight."""

    state_dict = extract_state_dict(checkpoint_or_state_dict)
    if first_conv_key not in state_dict:
        raise KeyError(f"Missing first convolution key: {first_conv_key}")

    weight = state_dict[first_conv_key]
    if not hasattr(weight, "shape") or len(tuple(weight.shape)) != 5:
        raise ValueError(
            f"{first_conv_key} must be a 5D Conv3d weight, got {type(weight).__name__}."
        )
    return int(weight.shape[1])


def initialize_3ch_from_1ch_checkpoint(
    source_checkpoint: str | Path,
    out_checkpoint: str | Path,
    *,
    channel_order: tuple[str, str, str] = DEFAULT_CHANNEL_ORDER,
    base: int = 16,
    out_channels: int = 1,
    seed: int = 0,
    random_scale: float = 0.01,
    first_conv_key: str = FIRST_CONV_KEY,
    metadata_path: str | Path | None = None,
) -> CheckpointInitResult:
    """Create a 3-channel initialization checkpoint from a 1-channel checkpoint.

    The first input channel receives the old 1-channel mean weights.  The new
    std/tSNR channels receive small random weights.  All other matching U-Net
    parameters are copied from the source checkpoint.
    """

    source_path = Path(source_checkpoint).expanduser().resolve()
    out_path = Path(out_checkpoint).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_ckpt = load_checkpoint(source_path, map_location="cpu")
    source_state = extract_state_dict(source_ckpt)
    _validate_source_first_conv(source_state, first_conv_key)

    target_model = UNet3D_NoZDown(
        in_channels=len(channel_order),
        base=base,
        out_channels=out_channels,
    )
    target_state = target_model.state_dict()
    migrated_state, copied_keys, random_std = _migrate_state_dict(
        source_state,
        target_state,
        seed=seed,
        random_scale=random_scale,
        first_conv_key=first_conv_key,
    )

    target_model.load_state_dict(migrated_state, strict=True)
    metadata = _build_init_metadata(
        source_path=source_path,
        out_path=out_path,
        source_state=source_state,
        migrated_state=migrated_state,
        copied_keys=copied_keys,
        channel_order=channel_order,
        base=base,
        out_channels=out_channels,
        seed=seed,
        random_scale=random_scale,
        random_std=random_std,
        first_conv_key=first_conv_key,
    )

    torch.save({"model": migrated_state, "metadata": metadata}, out_path)

    sidecar_path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else out_path.with_suffix(out_path.suffix + ".json")
    )
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return CheckpointInitResult(
        checkpoint_path=out_path,
        metadata_path=sidecar_path,
        metadata=metadata,
    )


def _migrate_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    seed: int,
    random_scale: float,
    first_conv_key: str,
) -> tuple[dict[str, torch.Tensor], list[str], float]:
    source_first = source_state[first_conv_key].detach().cpu()
    target_first = target_state[first_conv_key].detach().cpu()

    if target_first.shape[0] != source_first.shape[0]:
        raise ValueError(
            f"First conv output channels differ: source={tuple(source_first.shape)}, "
            f"target={tuple(target_first.shape)}."
        )

    if target_first.shape[2:] != source_first.shape[2:]:
        raise ValueError(
            f"First conv kernel size differs: source={tuple(source_first.shape)}, "
            f"target={tuple(target_first.shape)}."
        )

    random_std = _compute_random_std(source_first, random_scale)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    new_first = target_first.clone()
    new_first[:, 0:1] = source_first
    if new_first.shape[1] > 1:
        random_part = torch.empty_like(new_first[:, 1:])
        random_part.normal_(mean=0.0, std=random_std, generator=generator)
        new_first[:, 1:] = random_part

    migrated: dict[str, torch.Tensor] = {}
    copied_keys: list[str] = []
    missing_or_mismatched: list[str] = []

    for key, target_value in target_state.items():
        target_tensor = target_value.detach().cpu()
        if key == first_conv_key:
            migrated[key] = new_first.to(dtype=target_tensor.dtype)
            copied_keys.append(key)
            continue

        source_value = source_state.get(key)
        if source_value is None or tuple(source_value.shape) != tuple(target_tensor.shape):
            missing_or_mismatched.append(key)
            continue

        migrated[key] = source_value.detach().cpu().to(dtype=target_tensor.dtype).clone()
        copied_keys.append(key)

    if missing_or_mismatched:
        preview = ", ".join(missing_or_mismatched[:8])
        raise ValueError(
            "Source checkpoint does not match the expected U-Net architecture. "
            f"Missing/mismatched key(s): {preview}"
        )

    return migrated, copied_keys, random_std


def _validate_source_first_conv(
    source_state: Mapping[str, torch.Tensor],
    first_conv_key: str,
) -> None:
    if first_conv_key not in source_state:
        raise KeyError(f"Missing first convolution key: {first_conv_key}")

    weight = source_state[first_conv_key]
    shape = tuple(weight.shape)
    if len(shape) != 5:
        raise ValueError(f"{first_conv_key} must be a 5D Conv3d weight, got {shape}.")
    if shape[1] != 1:
        raise ValueError(
            "Source checkpoint must be 1-channel for this initializer. "
            f"Got {first_conv_key} shape {shape}."
        )


def _compute_random_std(source_first: torch.Tensor, random_scale: float) -> float:
    if random_scale <= 0:
        raise ValueError(f"random_scale must be positive, got {random_scale}.")

    source_std = float(source_first.float().std(unbiased=False).item())
    if source_std <= 0:
        source_std = 1.0
    return source_std * float(random_scale)


def _build_init_metadata(
    *,
    source_path: Path,
    out_path: Path,
    source_state: Mapping[str, torch.Tensor],
    migrated_state: Mapping[str, torch.Tensor],
    copied_keys: list[str],
    channel_order: tuple[str, str, str],
    base: int,
    out_channels: int,
    seed: int,
    random_scale: float,
    random_std: float,
    first_conv_key: str,
) -> dict[str, Any]:
    first = migrated_state[first_conv_key]
    norms = {
        channel: float(torch.linalg.vector_norm(first[:, idx]).item())
        for idx, channel in enumerate(channel_order)
    }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "UNet3D_NoZDown",
        "source_checkpoint": str(source_path),
        "output_checkpoint": str(out_path),
        "source_in_channels": int(source_state[first_conv_key].shape[1]),
        "target_in_channels": int(first.shape[1]),
        "channel_order": list(channel_order),
        "base": int(base),
        "out_channels": int(out_channels),
        "first_conv_key": first_conv_key,
        "first_conv_shape": list(first.shape),
        "init_strategy": "mean_copy_std_tsnr_small_random",
        "mean_channel": channel_order[0],
        "random_initialized_channels": list(channel_order[1:]),
        "random_seed": int(seed),
        "random_scale": float(random_scale),
        "random_std": float(random_std),
        "copied_parameter_count": len(copied_keys),
        "total_parameter_count": len(migrated_state),
        "first_layer_channel_norms": norms,
        "requires_finetuning": True,
    }
