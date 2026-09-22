"""Compatibility export for the dependency-light frozen-state contract."""

from skillev_private.frozen_state import (
    FROZEN_EVALUATION_INPUT_FORMAT,
    FROZEN_INFERENCE_STATE_FORMAT,
    FinalTrainingArtifactResolver,
    FrozenEvaluationInput,
    FrozenInferenceState,
)

__all__ = [
    "FROZEN_EVALUATION_INPUT_FORMAT",
    "FROZEN_INFERENCE_STATE_FORMAT",
    "FinalTrainingArtifactResolver",
    "FrozenEvaluationInput",
    "FrozenInferenceState",
]
