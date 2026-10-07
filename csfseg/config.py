"""Default configuration values for CSFSeg Toolkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    base: int = 16
    out_channels: int = 1


@dataclass(frozen=True)
class PreprocessingConfig:
    channels: tuple[str, ...] = ("mean", "std", "tsnr")
    depth: int = 10
    out_hw: tuple[int, int] = (128, 128)
    normalization: str = "nonzero_zscore"


@dataclass(frozen=True)
class PostprocessConfig:
    threshold: float = 0.5
    candidate_layers: tuple[int, ...] = (0, 1, 2, 3)
    min_valid_voxels: int = 5
    zero_epsilon: float = 1e-6
    zero_fraction_threshold: float = 0.10
    dropout_warning_fraction: float = 0.20


@dataclass(frozen=True)
class BatchConfig:
    subject_workers: int | str = "auto"
    skip_existing: bool = False


@dataclass(frozen=True)
class ToolkitConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)


DEFAULT_CONFIG = ToolkitConfig()


def ensure_out_dir(out_dir: str | Path) -> Path:
    """Resolve and create the user-provided output directory."""
    path = Path(out_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
