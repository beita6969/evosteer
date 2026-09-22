"""Regression coverage for the formal protocol-to-rollout seed binding."""

from __future__ import annotations

import pytest
from skillev_private.experiments.benchmark_attempt_input import (
    PRIVATE_BENCHMARK_ATTEMPT_INPUT_FORMAT,
    PrivateBenchmarkAttemptInput,
)

from skillev.application import ApplicationConfig
from skillev.calibration import CalibrationConfig
from skillev.contracts import stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import AuthoringSamplingConfig, EvolutionConfig
from skillev.experiments import FIXED_SEED, ExperimentProtocol
from skillev.policy import QwenBackboneConfig
from skillev.runtime import AttemptBuilderKind
from skillev.training import (
    CheckpointConfig,
    OptimizerConfig,
    PolicyRolloutConfig,
    TrainerConfig,
    TrainingExecutionConfig,
    TTBMethodConfig,
    conservative_rollout_maximum,
)


def _application(*, seed: int) -> ApplicationConfig:
    rollout_maximum = conservative_rollout_maximum(
        max_turns=1,
        max_reasoning_tokens=1,
        max_action_tokens=1,
        max_model_input_tokens=64,
        max_tool_wall_time_milliseconds=10,
    )
    return ApplicationConfig(
        trainer=TrainerConfig(
            method=TTBMethodConfig(epsilon_min=0.01, temperature_beta=1.0),
            rollout=PolicyRolloutConfig(
                base_seed=seed,
                max_turns=1,
                max_reasoning_tokens=1,
                max_action_tokens=1,
                per_rollout_maximum=rollout_maximum,
            ),
            optimizer=OptimizerConfig(
                adapter_learning_rate=1e-3,
                z_learning_rate=1e-3,
                weight_decay=0.0,
            ),
            execution=TrainingExecutionConfig(experiment_id="formal-seed-test", batch_size=1),
            checkpoint=CheckpointConfig(every_n_steps=1),
        ),
        diagnostics=DiagnosticsConfig(window_size=1),
        calibration=CalibrationConfig(),
        evolution=EvolutionConfig(generate_min_absolute_log_importance=0.1),
        authoring_sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        maximum_h0_tokens=64,
    )


def _backbone() -> QwenBackboneConfig:
    return QwenBackboneConfig(
        base_model_path="/private/model",
        revision="unit-test",
        tokenizer_id="unit-test-tokenizer",
        tokenizer_content_hash=stable_hash("unit-test-tokenizer"),
        hidden_size=8,
        device="cpu",
        torch_dtype="float32",
        lora_rank=1,
        lora_alpha=1,
        lora_dropout=0.0,
        lora_target_modules=("q_proj",),
        z_hidden_width=4,
        eos_token_ids=(0,),
    )


def _unvalidated_protocol() -> ExperimentProtocol:
    """Supply the typed seed carrier needed before later input fields matter."""

    protocol = object.__new__(ExperimentProtocol)
    object.__setattr__(protocol, "seed", FIXED_SEED)
    return protocol


def _input_until_seed_gate(*, rollout_seed: int) -> PrivateBenchmarkAttemptInput:
    """Construct only the prefix inspected before the seed invariant fails."""

    value = object.__new__(PrivateBenchmarkAttemptInput)
    object.__setattr__(value, "format", PRIVATE_BENCHMARK_ATTEMPT_INPUT_FORMAT)
    object.__setattr__(value, "builder_kind", AttemptBuilderKind.FULL)
    object.__setattr__(value, "backbone", _backbone())
    object.__setattr__(value, "backbone_kind", "qwen-causal")
    object.__setattr__(value, "application", _application(seed=rollout_seed))
    object.__setattr__(value, "protocol", _unvalidated_protocol())
    object.__setattr__(value, "tokenizer_artifact", object())
    object.__setattr__(value, "implementation_build", object())
    # A deliberately wrong next field distinguishes passing the seed gate
    # from merely returning before it.
    object.__setattr__(value, "checkpoint_storage", object())
    return value


def test_formal_attempt_rejects_a_rollout_seed_other_than_the_protocol_seed() -> None:
    candidate = _input_until_seed_gate(rollout_seed=FIXED_SEED + 1)

    with pytest.raises(ValueError):
        candidate.__post_init__()


def test_matching_formal_seed_reaches_the_next_input_gate() -> None:
    candidate = _input_until_seed_gate(rollout_seed=FIXED_SEED)

    with pytest.raises(TypeError):
        candidate.__post_init__()
