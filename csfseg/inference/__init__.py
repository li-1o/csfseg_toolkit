"""Inference helpers."""

from csfseg.inference.checkpoint import (
    CheckpointInitResult,
    infer_checkpoint_in_channels,
    initialize_3ch_from_1ch_checkpoint,
    load_checkpoint,
)
from csfseg.inference.postprocess import (
    RestoredPrediction,
    restore_bottom_to_full,
    restore_patch_to_bottom,
    restore_prediction_to_full,
    threshold_probability,
)
from csfseg.inference.predict import (
    LoadedModel,
    load_model_for_inference,
    predict_patch,
    resolve_device,
)

__all__ = [
    "CheckpointInitResult",
    "LoadedModel",
    "RestoredPrediction",
    "infer_checkpoint_in_channels",
    "initialize_3ch_from_1ch_checkpoint",
    "load_checkpoint",
    "load_model_for_inference",
    "predict_patch",
    "resolve_device",
    "restore_bottom_to_full",
    "restore_patch_to_bottom",
    "restore_prediction_to_full",
    "threshold_probability",
]
