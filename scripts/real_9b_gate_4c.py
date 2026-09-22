#!/usr/bin/env python3
"""One-shot child for the non-benchmark production-shape Gate 4c @6."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import time
import traceback
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import torch

from scripts.gate4c_durable_io import write_text_once_atomic
from scripts.gate4c_journal import Gate4cJournal
from scripts.gate4c_runtime_support import (
    GATE_4C_CONTEXT_ID,
    GATE_4C_ENVIRONMENT_ID,
    GATE_4C_FIXTURE_FORMAT,
    GATE_4C_FORMAT,
    GATE_4C_TASK_FAMILY,
    GATE_SEED,
    HORIZON,
    MAX_ACTION_TOKENS,
    MAX_REASONING_TOKENS,
    MAXIMUM_H0_TOKENS,
    MODEL_INPUT_CAP,
    PEAK_RESERVED_LIMIT_BYTES,
)
from scripts.gate4c_runtime_support import (
    backbone_config as _backbone_config,
)
from scripts.gate4c_runtime_support import (
    backend_identity as _backend_identity,
)
from scripts.gate4c_runtime_support import bind_initial_state as _bind_initial_state
from scripts.gate4c_runtime_support import (
    configure_determinism as _configure_determinism,
)
from scripts.gate4c_runtime_support import construct_backbone as _construct_backbone
from scripts.gate4c_runtime_support import (
    hardware_identity as _hardware_identity,
)
from scripts.gate4c_runtime_support import load_initial_checkpoint as _load_initial_checkpoint
from scripts.gate4c_runtime_support import (
    read_spec as _read_spec,
)
from scripts.gate4c_runtime_support import (
    verify_bundle as _verify_bundle,
)
from scripts.gate4c_runtime_support import (
    verify_initial_checkpoint as _verify_initial_checkpoint,
)
from skillev.calibration import CalibrationConfig, CalibrationEngine
from skillev.contracts import (
    InitialContext,
    ScientificSamplingCoordinate,
    SuccessRule,
    TerminalReward,
    TrainingStepCommit,
    TrajectoryRecord,
    TrajectoryStep,
    build_trajectory_record,
    canonical_json,
    stable_hash,
)
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution.detector import AwaitingDetectorSegment
from skillev.experiments import (
    Gate4cChildTerminal,
    Gate4cSpec,
    Gate4cStage,
)
from skillev.experiments.build_identity import sha256_file
from skillev.policy import (
    AuthoringTokenizerProtocol,
    QwenMultimodalBackboneConfig,
    QwenMultimodalPolicyBackbone,
)
from skillev.rollout import (
    CanonicalInitialContextAssembler,
    DecodingSnapshot,
    LocalPolicyGenerator,
    NoTerminalSubmission,
    RolloutArtifact,
    RolloutEngine,
    RolloutRequest,
    RolloutTask,
    StructuredJsonActionCodec,
    TerminalEvaluationRequest,
)
from skillev.rollout.generator import (
    RolloutGenerationRequest,
    RolloutGenerationResult,
    RolloutGenerator,
    RolloutTokenizerProtocol,
)
from skillev.rollout.prompt_profiles import InitialContextProfile
from skillev.runtime import (
    AttemptBuilderKind,
    BoundedAgent,
    BoundedAgentPolicy,
    BudgetLedger,
    BudgetVector,
    EnvironmentObservation,
    EventType,
    FullRuntimeExecutionState,
    LiveAttemptEventLog,
    OrderedTaskCursorState,
    RuntimeEventEmitter,
    RuntimeSnapshotIdentity,
    SkillLibraryState,
)
from skillev.runtime.attempt_run_plan import AttemptRunCursorState, ExactAttemptRunPlan
from skillev.scoring import (
    ScoringConfig,
    assembled_context_hash,
    backward_trajectory_delta_streaming,
    render_forward_prefix,
    render_hindsight_prefix,
    render_reasoning_prefix,
)
from skillev.training import (
    CollectedTrainingBatch,
    FilesystemTrainingCheckpointStore,
    MethodProjectionPipeline,
    PlannedRollout,
    TrainingBatchPlan,
    TrainingCheckpointSnapshot,
    TrainingStepSource,
)
from skillev.training.config import OptimizerConfig
from skillev.training.step_math import (
    apply_optimizer_step,
    create_ttb_optimizer,
    prepare_ttb_step,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--local-model-path", required=True)
    parser.add_argument("--work-directory", required=True)
    return parser.parse_args()


@dataclass(slots=True)
class _Clock:
    current: datetime = datetime(2026, 8, 11, tzinfo=UTC)

    def __call__(self) -> str:
        value = self.current.isoformat(timespec="microseconds").replace("+00:00", "Z")
        self.current += timedelta(microseconds=1)
        return value


@dataclass(slots=True)
class _PublicEnvironment:
    environment_id: str = GATE_4C_ENVIRONMENT_ID
    task_family: str = GATE_4C_TASK_FAMILY

    async def execute(self, action: object, *, step_index: int) -> EnvironmentObservation:
        del action
        return EnvironmentObservation(
            public_value={"step": step_index, "status": "synthetic-observation"},
            observation_status="success",
            budget_usage=BudgetVector(tool_calls=1),
        )

    def validate_completion(self, submission: object) -> bool:
        del submission
        return False


@dataclass(slots=True)
class _ZeroEvaluator:
    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if not isinstance(request.evaluation_input, NoTerminalSubmission):
            raise RuntimeError("Gate 4c completion must remain a non-terminal agent outcome")
        return TerminalReward(
            value=0.0,
            success=False,
            success_rule=SuccessRule.R_AT_THRESHOLD,
            success_threshold=1.0,
            native_metric_name="gate-4c-zero-reward",
            native_payload={"no_submission": True},
            environment_id=GATE_4C_ENVIRONMENT_ID,
            verifier_version="gate-4c-zero-evaluator@1",
        )


class _RecordingGenerator:
    def __init__(self, delegate: RolloutGenerator) -> None:
        self._delegate = delegate
        self.generated: list[tuple[int, ...]] = []
        self.content_spans: list[tuple[int, ...]] = []
        self.input_lengths: list[int] = []

    @property
    def tokenizer(self) -> RolloutTokenizerProtocol:
        return self._delegate.tokenizer

    def snapshot(self):
        return self._delegate.snapshot()

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        self._delegate.begin_episode(episode_id, expected_policy_snapshot_id)

    def end_episode(self, episode_id: str) -> None:
        self._delegate.end_episode(episode_id)

    async def generate(self, request: RolloutGenerationRequest) -> RolloutGenerationResult:
        self.input_lengths.append(len(request.input_ids))
        result = await self._delegate.generate(request)
        self.content_spans.append(result.content_token_ids)
        self.generated.append((*result.content_token_ids, *result.stop_token_ids))
        return result


class _TimedBackbone:
    """Measure production teacher-forced forwards without changing their graphs."""

    def __init__(self, delegate: QwenMultimodalPolicyBackbone) -> None:
        self._delegate = delegate
        self.teacher_forced_forward_seconds = 0.0

    def score(self, *args, **kwargs):
        started = time.monotonic()
        value = self._delegate.score(*args, **kwargs)
        self.teacher_forced_forward_seconds += time.monotonic() - started
        return value

    def z_value(self, *args, **kwargs):
        started = time.monotonic()
        value = self._delegate.z_value(*args, **kwargs)
        self.teacher_forced_forward_seconds += time.monotonic() - started
        return value

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def _near_target_text(
    tokenizer: AuthoringTokenizerProtocol,
    *,
    target: int,
    label: str,
    tolerance: int = 64,
) -> str:
    """Build a text-first synthetic fixture close to a token target."""

    if type(target) is not int or target < 1:
        raise ValueError("target must be a positive integer")
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError("tolerance must be a non-negative integer")
    unit = f" {label}"
    if not tokenizer.encode(unit):
        raise RuntimeError("synthetic text unit encoded to no tokens")

    minimum = max(0, target - tolerance)
    high = 1
    while len(tokenizer.encode(unit * high)) < minimum:
        high *= 2
        if high > target * 1_024:
            raise RuntimeError("synthetic text cannot reach the requested token range")

    low = 0
    best_text = ""
    best_count = len(tokenizer.encode(best_text))
    while low <= high:
        middle = (low + high) // 2
        candidate = unit * middle
        count = len(tokenizer.encode(candidate))
        if count <= target:
            if count >= best_count:
                best_text = candidate
                best_count = count
            low = middle + 1
        else:
            high = middle - 1
    if best_count < minimum:
        raise RuntimeError("tokenizer cannot construct the fixed near-cap synthetic text")
    return best_text


def build_gate_4c_synthetic_record(
    tokenizer: AuthoringTokenizerProtocol,
) -> tuple[str, TrajectoryRecord, dict[str, int]]:
    initial_text = _near_target_text(tokenizer, target=MAXIMUM_H0_TOKENS - 8, label="h0")
    reasoning = _near_target_text(tokenizer, target=MAX_REASONING_TOKENS, label="reason")
    action = _near_target_text(tokenizer, target=MAX_ACTION_TOKENS, label="action")
    action_ids = tuple(tokenizer.encode(action))
    observation = canonical_json({"status": "public-synthetic", "payload": "observation"})
    placeholders = tuple(
        TrajectoryStep(
            index=index,
            reasoning_text=reasoning,
            action_text=action,
            action_token_ids=action_ids,
            action_token_count=len(action_ids),
            observation_text=observation,
            observation_status="success",
            invoked_skill_ids=(),
            forward_prefix_hash=stable_hash({"placeholder": "forward", "step": index}),
            hindsight_prefix_hash=stable_hash({"placeholder": "backward", "step": index}),
        )
        for index in range(1, HORIZON + 1)
    )
    steps = tuple(
        replace(
            step,
            forward_prefix_hash=render_forward_prefix(
                initial_text, placeholders, step.index
            ).prefix_hash,
            hindsight_prefix_hash=render_hindsight_prefix(
                initial_text, placeholders, step.index
            ).prefix_hash,
        )
        for step in placeholders
    )
    reward = TerminalReward(
        value=0.0,
        success=False,
        success_rule=SuccessRule.R_AT_THRESHOLD,
        success_threshold=1.0,
        native_metric_name="gate-4c-synthetic",
        native_payload={"synthetic": True},
        environment_id=GATE_4C_ENVIRONMENT_ID,
        verifier_version=GATE_4C_FIXTURE_FORMAT,
    )
    context = InitialContext(
        query="public synthetic query",
        retrieved_skill_ids=(),
        active_skill_ids=(),
        meta={
            "environment_id": GATE_4C_ENVIRONMENT_ID,
            "task_family": GATE_4C_TASK_FAMILY,
        },
        assembler_version=GATE_4C_FIXTURE_FORMAT,
        assembled_hash=assembled_context_hash(initial_text),
        assembled_token_count=len(tokenizer.encode(initial_text)),
    )
    record = build_trajectory_record(
        tokenizer=tokenizer,
        trajectory_id="gate-4c-worst-shape",
        environment_id=GATE_4C_ENVIRONMENT_ID,
        task_family=GATE_4C_TASK_FAMILY,
        initial_context=context,
        steps=steps,
        horizon=HORIZON,
        reward=reward,
        shifted_reward=0.01,
        epsilon_min=0.01,
        tokenizer_id=tokenizer.tokenizer_id,
        decoding_snapshot_id="gate-4c-decoding@2",
        created_at="2026-08-11T00:00:00Z",
    )
    lengths = {
        "forward_prefix_plus_action": max(
            len(tokenizer.encode(render_forward_prefix(initial_text, steps, index).text))
            + steps[index - 1].action_token_count
            for index in range(1, HORIZON + 1)
        ),
        "hindsight_prefix_plus_action": max(
            len(tokenizer.encode(render_hindsight_prefix(initial_text, steps, index).text))
            + steps[index - 1].action_token_count
            for index in range(1, HORIZON + 1)
        ),
    }
    _require_input_lengths(lengths)
    return initial_text, record, lengths


def _gradient_metrics(backbone: QwenMultimodalPolicyBackbone) -> dict[str, float]:
    groups = backbone.parameter_groups()
    result: dict[str, float] = {}
    for name, parameters in (
        ("forward", groups.forward),
        ("backward", groups.backward),
        ("z", groups.z_head),
    ):
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients:
            raise RuntimeError(f"Gate 4c {name} component produced no gradients")
        norm = math.sqrt(
            math.fsum(float(value.detach().float().pow(2).sum()) for value in gradients)
        )
        if not math.isfinite(norm) or norm == 0.0:
            raise RuntimeError(f"Gate 4c {name} gradient norm is not finite and nonzero")
        result[name] = norm
    return result


def _run_worst_shape_from_fixture(
    backbone: QwenMultimodalPolicyBackbone,
    fixture: tuple[str, TrajectoryRecord, dict[str, int]],
) -> dict[str, object]:
    initial_text, record, lengths = fixture
    optimizer, groups = create_ttb_optimizer(
        backbone, OptimizerConfig(adapter_learning_rate=1e-5, z_learning_rate=1e-4)
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    score = backward_trajectory_delta_streaming(
        backbone, record, initial_text, ScoringConfig(temperature_beta=1.0)
    )
    for parameter in (*groups.forward, *groups.backward, *groups.z_head):
        if parameter.grad is not None:
            parameter.grad.mul_(score.gradient_coefficient)
    gradients = _gradient_metrics(backbone)
    optimizer.step()
    backbone.mark_policy_update(1)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    _require_peak("synthetic worst-shape", peak_reserved)
    return {
        "elapsed_seconds": elapsed,
        "gradient_norms": gradients,
        "h0_tokens": record.initial_context.assembled_token_count,
        "horizon": record.horizon,
        "input_lengths": lengths,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def _require_peak(label: str, peak_reserved: int) -> None:
    if peak_reserved > PEAK_RESERVED_LIMIT_BYTES:
        raise RuntimeError(f"Gate 4c {label} exceeded the fixed 64 GiB reserve limit")


def _require_input_lengths(lengths: dict[str, int]) -> None:
    if any(value > MODEL_INPUT_CAP for value in lengths.values()):
        raise RuntimeError("Gate 4c input exceeded the fixed model input cap")


def _query_for_near_cap_h0(
    assembler: CanonicalInitialContextAssembler,
    tokenizer: RolloutTokenizerProtocol,
    library_version: str,
) -> str:
    low, high = 1, MAXIMUM_H0_TOKENS
    best = "synthetic"
    while low <= high:
        middle = (low + high) // 2
        query = "synthetic " * middle
        task = RolloutTask(
            task_id="gate-4c-rollout-task",
            environment_id=GATE_4C_ENVIRONMENT_ID,
            task_family=GATE_4C_TASK_FAMILY,
            context_id=GATE_4C_CONTEXT_ID,
            query=query,
            available_tools=("debug.tool",),
            public_context={"fixture": "gate-4c"},
        )
        try:
            assembled = assembler.assemble(
                task=task,
                retrieved_skills=(),
                active_skill_ids=(),
                library_version=library_version,
                tokenizer=tokenizer,
            )
        except ValueError:
            high = middle - 1
        else:
            best = query
            if assembled.contract.assembled_token_count >= MAXIMUM_H0_TOKENS - 64:
                return best
            low = middle + 1
    return best


async def _run_rollout(
    *,
    backbone: QwenMultimodalPolicyBackbone,
    root: Path,
    cache_enabled: bool,
    trajectory_id: str,
    library_version: str,
) -> tuple[RolloutArtifact, dict[str, object], tuple[tuple[int, ...], ...]]:
    root.mkdir(parents=True, exist_ok=False)
    clock = _Clock()
    log = LiveAttemptEventLog(root / "events.jsonl", run_id="gate-4c", attempt_id=trajectory_id)
    emitter = RuntimeEventEmitter(log=log, producer_id="gate-4c", clock=clock)
    ledger = BudgetLedger(
        run_id="gate-4c",
        attempt_id=trajectory_id,
        cap=BudgetVector(
            input_tokens=2 * HORIZON * MODEL_INPUT_CAP,
            output_tokens=2 * HORIZON * MAX_ACTION_TOKENS,
            model_calls=2 * HORIZON,
            agent_turns=HORIZON,
            tool_calls=HORIZON,
            wall_time_milliseconds=HORIZON * 1_000,
        ),
    )
    environment = _PublicEnvironment()
    assembler = CanonicalInitialContextAssembler(maximum_h0_tokens=MAXIMUM_H0_TOKENS)
    generator = _RecordingGenerator(
        LocalPolicyGenerator(backbone, episode_cache_enabled=cache_enabled)
    )
    query = _query_for_near_cap_h0(assembler, generator.tokenizer, library_version)
    task = RolloutTask(
        task_id="gate-4c-rollout-task",
        environment_id=environment.environment_id,
        task_family=environment.task_family,
        context_id=GATE_4C_CONTEXT_ID,
        query=query,
        available_tools=("debug.tool",),
        public_context={"fixture": "gate-4c"},
    )
    engine = RolloutEngine(
        generator=generator,
        context_assembler=assembler,
        action_codec=StructuredJsonActionCodec(),
        bounded_agent=BoundedAgent(
            environment=environment,
            policy=BoundedAgentPolicy(max_turns=HORIZON),
            ledger=ledger,
            tool_call_maximum=BudgetVector(tool_calls=1, wall_time_milliseconds=1_000),
            emitter=emitter,
        ),
        terminal_evaluator=_ZeroEvaluator(),
        ledger=ledger,
        reasoning_call_maximum=BudgetVector(
            input_tokens=MODEL_INPUT_CAP,
            output_tokens=MAX_REASONING_TOKENS,
            model_calls=1,
        ),
        action_call_maximum=BudgetVector(
            input_tokens=MODEL_INPUT_CAP,
            output_tokens=MAX_ACTION_TOKENS,
            model_calls=1,
            agent_turns=1,
        ),
        emitter=emitter,
        clock=clock,
    )
    request = RolloutRequest(
        trajectory_id=trajectory_id,
        task=task,
        retrieved_skills=(),
        active_skill_ids=(),
        library_version=library_version,
        sampling_coordinate=ScientificSamplingCoordinate(
            sampling_schedule_hash=stable_hash({"gate": "4c@2"}),
            schedule_purpose="non-benchmark-gate",
            ordered_sequence_hash=stable_hash({"tasks": [task.task_id]}),
            sequence_position=0,
            task_id=task.task_id,
            optimizer_step_or_anchor_ordinal=1,
        ),
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=MAX_REASONING_TOKENS,
            max_action_tokens=MAX_ACTION_TOKENS,
            base_seed=GATE_SEED,
        ),
        epsilon_min=0.01,
        condition_id="skillev-cold-start-neutral@1",
        initial_context_profile=InitialContextProfile.SKILLEV_COLD_START,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    artifact = await engine.run(request)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    stateless_prefill = math.fsum(
        len(
            generator.tokenizer.encode(
                render_reasoning_prefix(
                    artifact.initial_context.text,
                    artifact.record.steps[: step.index - 1],
                    step.index,
                ).text
            )
        )
        + len(
            generator.tokenizer.encode(
                render_forward_prefix(
                    artifact.initial_context.text, artifact.record.steps, step.index
                ).text
            )
        )
        for step in artifact.record.steps
    )
    peak_reserved = torch.cuda.max_memory_reserved()
    _require_peak("cached rollout" if cache_enabled else "stateless rollout", peak_reserved)
    if artifact.record.horizon != HORIZON or len(generator.generated) != 2 * HORIZON:
        raise RuntimeError("Gate 4c rollout did not execute exactly 15 turns / 30 calls")
    sampled_action_spans = tuple(generator.content_spans[1::2])
    recorded_action_spans = tuple(step.action_token_ids for step in artifact.record.steps)
    if recorded_action_spans != sampled_action_spans:
        raise RuntimeError("Gate 4c rollout replaced a sampled action token span")
    restored = RolloutArtifact.from_value(artifact.to_value(), tokenizer=generator.tokenizer)
    if restored != artifact:
        raise RuntimeError("Gate 4c rollout artifact did not restore exactly")
    input_lengths = {
        "reasoning_generation_input": max(generator.input_lengths[0::2]),
        "action_generation_input": max(generator.input_lengths[1::2]),
    }
    _require_input_lengths(input_lengths)
    return (
        artifact,
        {
            "actual_prefill_token_positions": (
                backbone._last_completed_policy_episode_prefill_token_count
                if cache_enabled
                else int(stateless_prefill)
            ),
            "elapsed_seconds": elapsed,
            "horizon": artifact.record.horizon,
            "input_lengths": input_lengths,
            "model_calls": len(generator.generated),
            "artifact_restore_exact": True,
            "sampled_action_spans_preserved": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": peak_reserved,
            "stateless_prefill_token_positions": int(stateless_prefill),
        },
        tuple(generator.generated),
    )


def _records_equivalent(left: RolloutArtifact, right: RolloutArtifact) -> bool:
    return (
        left.initial_context.text == right.initial_context.text
        and left.record.steps == right.record.steps
        and left.record.reward == right.record.reward
        and left.manifest.reasoning_token_counts == right.manifest.reasoning_token_counts
    )


def _optimizer_states_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left.cpu(), cast(torch.Tensor, right).cpu())
    if isinstance(left, dict):
        other = cast(dict[object, object], right)
        return left.keys() == other.keys() and all(
            _optimizer_states_equal(value, other[key]) for key, value in left.items()
        )
    if isinstance(left, list | tuple):
        other_sequence = cast(list[object] | tuple[object, ...], right)
        return len(left) == len(other_sequence) and all(
            _optimizer_states_equal(a, b) for a, b in zip(left, other_sequence, strict=True)
        )
    return left == right


def _optimizer_state_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _optimizer_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_optimizer_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_optimizer_state_to_cpu(item) for item in value)
    return value


def _runtime_identity(
    spec: Gate4cSpec,
    backbone: QwenMultimodalPolicyBackbone,
    library: SkillLibraryState,
    run_plan: ExactAttemptRunPlan,
) -> RuntimeSnapshotIdentity:
    return RuntimeSnapshotIdentity(
        builder_kind=AttemptBuilderKind.FULL,
        public_identity_content_hash=stable_hash({"gate": "4c@3", "kind": "public"}),
        application_config_hash=spec.production_config_hash,
        method_identity_hash=stable_hash({"gate": "4c@3", "method": "full"}),
        protocol_hash=spec.content_hash,
        protocol_freeze_id=stable_hash({"gate": "4c@3", "spec": spec.content_hash}),
        run_plan_hash=run_plan.content_hash,
        initial_library_version=library.current_version,
        initial_skill_library_state_hash=library.state_hash,
        initial_trainable_state_hash=backbone.initial_trainable_state_hash,
        ordered_task_sequence_hash=stable_hash({"tasks": ["gate-4c-rollout-task"]}),
        sampling_schedule_algorithm="skillev-scientific-sampling@1",
        sampling_schedule_hash=stable_hash({"gate": "4c@3"}),
        base_model_artifact_hash=spec.base_model_artifact.content_hash,
        tokenizer_artifact_hash=spec.tokenizer_artifact.content_hash,
        implementation_build_hash=spec.source_tree_hash,
    )


def _run_production_step(
    *,
    backbone: QwenMultimodalPolicyBackbone,
    artifact: RolloutArtifact,
    work: Path,
    spec: Gate4cSpec,
    config: QwenMultimodalBackboneConfig,
    initial_policy: Path,
    journal: Gate4cJournal,
) -> dict[str, object]:
    clock = _Clock(datetime(2026, 8, 11, 1, tzinfo=UTC))
    optimizer, groups = create_ttb_optimizer(
        backbone, OptimizerConfig(adapter_learning_rate=1e-5, z_learning_rate=1e-4)
    )
    snapshot_before = LocalPolicyGenerator(backbone).snapshot()
    batch = CollectedTrainingBatch(
        batch_id="gate-4c-production-batch",
        optimizer_step=1,
        policy_snapshot_id=snapshot_before.snapshot_id,
        library_version=artifact.manifest.library_version,
        artifacts=(artifact,),
    )
    library = SkillLibraryState.from_seed_documents(())
    if library.current_version != batch.library_version:
        raise RuntimeError("Gate 4c artifact and checkpoint library identities differ")
    projections = MethodProjectionPipeline.fresh(
        diagnostics_config=DiagnosticsConfig(),
        calibration=CalibrationEngine(CalibrationConfig()),
        library_version=library.current_version,
    )
    run_plan = ExactAttemptRunPlan(phase_search_steps=1, closure_steps=1, maximum_cycles=1)
    run_cursor = AttemptRunCursorState.fresh(run_plan).after_training_step(run_plan)
    identity = _runtime_identity(spec, backbone, library, run_plan)
    source_log = LiveAttemptEventLog(
        work / "production-events.jsonl",
        run_id="gate-4c@6",
        attempt_id="gate-4c-production-step",
    )
    emitter = RuntimeEventEmitter(log=source_log, producer_id="gate-4c", clock=clock)
    timed_backbone = _TimedBackbone(backbone)

    torch.cuda.reset_peak_memory_stats()
    with journal.stage(Gate4cStage.PRODUCTION_SCORE):
        started = time.monotonic()
        prepared = prepare_ttb_step(
            backbone=timed_backbone,
            optimizer=optimizer,
            parameters=groups,
            batch=batch,
            snapshot_before=snapshot_before,
            temperature_beta=1.0,
            clock=clock,
        )
        torch.cuda.synchronize()
        scoring_backward_seconds = time.monotonic() - started
        scoring_forward_seconds = timed_backbone.teacher_forced_forward_seconds
        backward_seconds = scoring_backward_seconds - scoring_forward_seconds
        if backward_seconds <= 0.0:
            raise RuntimeError("Gate 4c did not measure a positive backward duration")
        gradients = _gradient_metrics(backbone)
        source = TrainingStepSource(
            batch_id=batch.batch_id,
            optimizer_step=batch.optimizer_step,
            artifacts=batch.artifacts,
            stats=prepared.stats,
            edge_records=prepared.edges,
        )
        transition = projections.preview(source)
    with journal.stage(Gate4cStage.OPTIMIZER_STEP):
        optimizer_started = time.monotonic()
        report = apply_optimizer_step(
            optimizer=optimizer, backbone=backbone, prepared=prepared, clock=clock
        )
        torch.cuda.synchronize()
        optimizer_seconds = time.monotonic() - optimizer_started
    snapshot_after = LocalPolicyGenerator(backbone).snapshot()
    commit = TrainingStepCommit(
        batch_id=batch.batch_id,
        optimizer_step=batch.optimizer_step,
        policy_snapshot_before=snapshot_before.snapshot_id,
        policy_snapshot_after=snapshot_after.snapshot_id,
        library_version=batch.library_version,
        records=(artifact.record,),
        edge_records=prepared.edges,
        stats=prepared.stats,
        posterior_batch=transition.posterior_batch,
        report=report,
        run_cursor_after=run_cursor.to_source_value(),
    )
    with journal.stage(Gate4cStage.SOURCE_COMMIT):
        emitter.emit(EventType.TRAINING_STEP_COMMITTED, commit.to_value())
        projections.commit(transition)
        source_commit_count = sum(
            1
            for line in source_log.path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event_type"] == EventType.TRAINING_STEP_COMMITTED.value
        )
        edge_record_count = len(commit.edge_records)
        residual_count = len(commit.stats.residuals)
        if source_commit_count != 1 or edge_record_count != 2 * HORIZON:
            raise RuntimeError("Gate 4c production source commit is incomplete")

    expected_policy = backbone.trainable_state_identity
    expected_optimizer = _optimizer_state_to_cpu(optimizer.state_dict())
    execution_state = FullRuntimeExecutionState(
        task_cursor=OrderedTaskCursorState(curriculum_id="gate-4c@6", cursor=1),
        run_cursor=run_cursor,
        library=library,
        projections=projections.runtime_state(),
        detector=AwaitingDetectorSegment(library.current_version),
    )
    checkpoint_store = FilesystemTrainingCheckpointStore(root=work / "checkpoints")
    with journal.stage(Gate4cStage.CHECKPOINT_SAVE):
        checkpoint_started = time.monotonic()
        checkpoint = checkpoint_store.save_as(
            TrainingCheckpointSnapshot(
                optimizer_step=1,
                experiment_id="gate-4c@6",
                identity=identity,
                backbone=backbone,
                optimizer=optimizer,
                execution_state=execution_state,
            ),
            name="step-00000001",
        )
        checkpoint_seconds = time.monotonic() - checkpoint_started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    _require_peak("production training step", peak_reserved)

    del prepared, source, transition, commit, groups, optimizer, backbone
    gc.collect()
    with journal.stage(Gate4cStage.CHECKPOINT_RESTORE):
        restored = _construct_backbone(config, initialization_seed=spec.initialization_seed)
        _load_initial_checkpoint(backbone=restored, initial_policy=initial_policy, spec=spec)
        _bind_initial_state(backbone=restored, spec=spec)
        restored_optimizer, _ = create_ttb_optimizer(
            restored, OptimizerConfig(adapter_learning_rate=1e-5, z_learning_rate=1e-4)
        )
        restored_snapshot = checkpoint_store.restore(
            checkpoint,
            backbone=restored,
            optimizer=restored_optimizer,
            expected_experiment_id="gate-4c@6",
            expected_identity=identity,
        )
        if restored.trainable_state_identity != expected_policy:
            raise RuntimeError("Gate 4c checkpoint changed policy tensor identity")
        if not _optimizer_states_equal(restored_optimizer.state_dict(), expected_optimizer):
            raise RuntimeError("Gate 4c checkpoint changed AdamW state")
        if restored_snapshot.execution_state != execution_state:
            raise RuntimeError("Gate 4c checkpoint changed runtime cursor/state")
        next_snapshot = LocalPolicyGenerator(restored).snapshot()
        next_plan = TrainingBatchPlan(
            batch_id="gate-4c-next-batch-plan",
            optimizer_step=2,
            policy_snapshot_id=next_snapshot.snapshot_id,
            library_version=library.current_version,
            rollouts=(
                PlannedRollout(
                    position=1,
                    task=RolloutTask(
                        task_id="gate-4c-next-task",
                        environment_id=GATE_4C_ENVIRONMENT_ID,
                        task_family=GATE_4C_TASK_FAMILY,
                        context_id=GATE_4C_CONTEXT_ID,
                        query="public next-step construction",
                        available_tools=("debug.tool",),
                        public_context={"fixture": "gate-4c-next"},
                    ),
                    trajectory_id="gate-4c-next-trajectory",
                    decoding=DecodingSnapshot.create(
                        max_reasoning_tokens=MAX_REASONING_TOKENS,
                        max_action_tokens=MAX_ACTION_TOKENS,
                        base_seed=GATE_SEED,
                    ),
                ),
            ),
        )
        if next_plan.optimizer_step != 2:
            raise RuntimeError("Gate 4c restore cannot construct the next training batch")
    return {
        "checkpoint_optimizer_identity_exact": True,
        "checkpoint_policy_identity_exact": True,
        "checkpoint_runtime_cursor_exact": True,
        "checkpoint_seconds": checkpoint_seconds,
        "edge_record_count": edge_record_count,
        "gradient_norms": gradients,
        "optimizer_seconds": optimizer_seconds,
        "optimizer_step": report.optimizer_step,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "residual_count": residual_count,
        "scoring_forward_seconds": scoring_forward_seconds,
        "backward_seconds": backward_seconds,
        "scoring_backward_seconds": scoring_backward_seconds,
        "source_commit_count": source_commit_count,
    }


def _load_backbone(
    *,
    config: QwenMultimodalBackboneConfig,
    spec: Gate4cSpec,
    initial_policy: Path,
    journal: Gate4cJournal,
) -> QwenMultimodalPolicyBackbone:
    with journal.stage(Gate4cStage.BACKBONE_CONSTRUCT):
        backbone = _construct_backbone(config, initialization_seed=spec.initialization_seed)
    with journal.stage(Gate4cStage.INITIAL_CHECKPOINT_LOAD):
        _load_initial_checkpoint(backbone=backbone, initial_policy=initial_policy, spec=spec)
    with journal.stage(Gate4cStage.INITIAL_STATE_BIND):
        _bind_initial_state(backbone=backbone, spec=spec)
    return backbone


def execute_gate_4c(
    *, spec_path: Path, model_path: Path, work: Path, journal: Gate4cJournal
) -> tuple[Gate4cSpec, dict[str, object]]:
    with journal.stage(Gate4cStage.SPEC_READ):
        spec = _read_spec(spec_path)
    with journal.stage(Gate4cStage.DETERMINISM_CONFIGURED):
        _configure_determinism(spec.initialization_seed)
    with journal.stage(Gate4cStage.HARDWARE_ATTESTED):
        hardware = _hardware_identity()
        if hardware != spec.expected_hardware_identity:
            raise ValueError("execution hardware differs from Gate 4c spec")
    with journal.stage(Gate4cStage.BACKEND_ATTESTED):
        backend = _backend_identity()
        if backend != spec.deterministic_backend_identity:
            raise ValueError("deterministic backend differs from Gate 4c spec")
    with journal.stage(Gate4cStage.BUNDLE_VERIFIED):
        initial_policy = _verify_bundle(spec, spec_path, work / "scratch")
    with journal.stage(Gate4cStage.CHECKPOINT_METADATA_VERIFIED):
        _verify_initial_checkpoint(spec=spec, initial_policy=initial_policy)
    with journal.stage(Gate4cStage.MODEL_CONFIG_CONSTRUCTED):
        config = _backbone_config(spec, model_path)

    worst_backbone = _load_backbone(
        config=config, spec=spec, initial_policy=initial_policy, journal=journal
    )
    with journal.stage(Gate4cStage.WORST_SHAPE_FIXTURE):
        fixture = build_gate_4c_synthetic_record(worst_backbone.tokenizer)
    with journal.stage(Gate4cStage.WORST_SHAPE_SCORE):
        worst_shape = _run_worst_shape_from_fixture(worst_backbone, fixture)
    del worst_backbone
    gc.collect()

    backbone = _load_backbone(
        config=config, spec=spec, initial_policy=initial_policy, journal=journal
    )
    library_version = SkillLibraryState.from_seed_documents(()).current_version
    with journal.stage(Gate4cStage.STATELESS_ROLLOUT):
        stateless, stateless_metrics, stateless_tokens = asyncio.run(
            _run_rollout(
                backbone=backbone,
                root=work / "scratch" / "stateless-rollout",
                cache_enabled=False,
                trajectory_id="gate-4c-stateless",
                library_version=library_version,
            )
        )
    with journal.stage(Gate4cStage.CACHED_ROLLOUT):
        cached, cached_metrics, cached_tokens = asyncio.run(
            _run_rollout(
                backbone=backbone,
                root=work / "scratch" / "cached-rollout",
                cache_enabled=True,
                trajectory_id="gate-4c-cached",
                library_version=library_version,
            )
        )
    with journal.stage(Gate4cStage.CACHE_EQUIVALENCE):
        if not _records_equivalent(stateless, cached) or stateless_tokens != cached_tokens:
            raise RuntimeError("cached rollout differs from stateless full-prefix execution")
        if int(cached_metrics["actual_prefill_token_positions"]) >= int(
            stateless_metrics["actual_prefill_token_positions"]
        ):
            raise RuntimeError("Gate 4c episode cache did not reduce prefill positions")

    production_step = _run_production_step(
        backbone=backbone,
        artifact=cached,
        work=work / "scratch",
        spec=spec,
        config=config,
        initial_policy=initial_policy,
        journal=journal,
    )
    rollout_seconds = float(cached_metrics["elapsed_seconds"])
    core_step_seconds = rollout_seconds + float(production_step["scoring_backward_seconds"])
    core_step_seconds += float(production_step["optimizer_seconds"])
    result = {
        "cached_rollout": cached_metrics,
        "cached_stateless_observations_exact": True,
        "cached_stateless_token_ids_exact": True,
        "cached_stateless_prefill_ratio": int(cached_metrics["actual_prefill_token_positions"])
        / int(stateless_metrics["actual_prefill_token_positions"]),
        "deterministic_backend_identity": backend.to_value(),
        "execution_hardware_identity": hardware.to_value(),
        "execution_identity": {
            "base_model_artifact_hash": spec.base_model_artifact.content_hash,
            "initial_checkpoint_tree_hash": spec.initial_checkpoint_tree_hash,
            "initial_trainable_state_hash": spec.initial_trainable_state.content_hash,
            "lockfile_sha256": spec.lockfile_sha256,
            "private_wheel_sha256": spec.private_wheel_sha256,
            "production_config_hash": spec.production_config_hash,
            "public_wheel_sha256": spec.public_wheel_sha256,
            "qwen_deployment_hash": spec.qwen_deployment_hash,
            "source_archive_sha256": spec.source_archive_sha256,
            "source_commit": spec.source_commit,
            "source_tree_hash": spec.source_tree_hash,
            "tokenizer_artifact_hash": spec.tokenizer_artifact.content_hash,
        },
        "format": GATE_4C_FORMAT,
        "gate_passed": True,
        "production_step": production_step,
        "projected_core_compute_lower_bound_per_arm_seconds": core_step_seconds * 4_608,
        "projected_core_compute_lower_bound_two_gpu_seconds": core_step_seconds * 4_608 * 4,
        "spec_content_hash": spec.content_hash,
        "stateless_rollout": stateless_metrics,
        "steps_per_day_core_compute_lower_bound": 86_400 / core_step_seconds,
        "turns_per_minute_cached": HORIZON / (rollout_seconds / 60),
        "worst_shape": worst_shape,
    }
    return spec, result


def run_child(
    *,
    spec_path: Path,
    model_path: Path,
    work: Path,
    execute=execute_gate_4c,
    process_start_hook=None,
) -> Gate4cChildTerminal:
    public = work / "public"
    private = work / "private"
    scratch = work / "scratch"
    if not all(path.is_dir() for path in (public, private, scratch)):
        raise FileNotFoundError("Gate 4c operator directories are missing")
    journal = Gate4cJournal(private / "stage-journal.jsonl")
    started_ns = time.monotonic_ns()
    spec: Gate4cSpec | None = None
    try:
        with journal.stage(Gate4cStage.PROCESS_START):
            if process_start_hook is not None:
                process_start_hook()
        spec, result = execute(
            spec_path=spec_path, model_path=model_path, work=work, journal=journal
        )
        result_path = public / "gate-4c-result.json"
        with journal.stage(Gate4cStage.RESULT_WRITE):
            write_text_once_atomic(result_path, canonical_json(result) + "\n")
        with journal.stage(Gate4cStage.PROCESS_SUCCESS):
            pass
        terminal = Gate4cChildTerminal.passed(
            spec_content_hash=spec.content_hash,
            final_stage=Gate4cStage.PROCESS_SUCCESS,
            result_sha256=sha256_file(result_path),
            stage_journal_sha256=sha256_file(journal.path),
            elapsed_ns=time.monotonic_ns() - started_ns,
        )
    except BaseException as error:
        traceback_path = private / "traceback.txt"
        write_text_once_atomic(traceback_path, traceback.format_exc())
        terminal = Gate4cChildTerminal.failed(
            spec_content_hash=spec.content_hash if spec is not None else None,
            final_stage=journal.current_stage,
            exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
            traceback_sha256=sha256_file(traceback_path),
            stage_journal_sha256=sha256_file(journal.path),
            elapsed_ns=time.monotonic_ns() - started_ns,
        )
        write_text_once_atomic(
            public / "child-terminal.json", canonical_json(terminal.to_value()) + "\n"
        )
        raise
    write_text_once_atomic(
        public / "child-terminal.json", canonical_json(terminal.to_value()) + "\n"
    )
    return terminal


def main() -> None:
    args = _args()
    run_child(
        spec_path=Path(args.spec),
        model_path=Path(args.local_model_path),
        work=Path(args.work_directory),
    )


if __name__ == "__main__":
    main()
