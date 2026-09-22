"""Shared scientific assembly and evidence/reporting for seven-domain training.

Formal and diagnostic launchers both use SKILLEVApplication; this module has no
admission/attestation gate and never substitutes rollout or terminal evaluation.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from skillev.application import SKILLEVApplication
from skillev.application_config import ApplicationConfig
from skillev.calibration import CalibrationConfig
from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import AuthoringSamplingConfig, EvolutionConfig, SkillAuthoringAuthority
from skillev.experiments import FormalMethodV10
from skillev.experiments._evolution_preflight_seed import planned_seed_documents
from skillev.policy import PrivateInitialCheckpointBinding, QwenMultimodalBackboneConfig
from skillev.rollout import RolloutTask
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptRunCursorState,
    BudgetVector,
    RuntimeSnapshotIdentity,
    SkillLibraryState,
)
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.training import (
    CheckpointConfig,
    OptimizerConfig,
    PolicyRolloutConfig,
    TrainerConfig,
    TrainingExecutionConfig,
    TTBMethodConfig,
    conservative_rollout_maximum,
)
from skillev.training.deadline import forecast_deadline
from skillev.training.inflight import durable_json
from skillev.training.metrics_export import MetricsStore, export_available
from skillev_private.benchmarks.mbpp_scoring import MBPPScorerProfile
from skillev_private.benchmarks.protocol_v13_seven_training import (
    SEVEN_DOMAIN_CONDITION,
    TRAJECTORIES_PER_QUESTION,
)

from .protocol_v10_application_input import ProtocolV10ApplicationIdentity

_PREPARATION_FORMAT: Final = "skillev-private-protocol13-training-debug-preparation@1"


def _clock() -> str:
    return datetime.now(UTC).isoformat()


def _object(value: object, *, fields: set[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _read_preparation(
    path: Path,
) -> tuple[QwenMultimodalBackboneConfig, PrivateInitialCheckpointBinding]:
    if not path.is_absolute() or not path.is_file():
        raise ValueError("preparation path must be an existing absolute file")
    from .warmup_initialization import WARMUP_PREPARATION_FORMAT, initialization_condition

    raw = json.loads(path.read_text(encoding="utf-8"))
    warmup = isinstance(raw, dict) and raw.get("format") == WARMUP_PREPARATION_FORMAT
    data = _object(
        raw,
        fields={"backbone", "format", "initial_checkpoint"}
        | ({"initialization"} if warmup else set()),
        label="training preparation",
    )
    if warmup:
        initialization_condition(path)
    elif data["format"] != _PREPARATION_FORMAT:
        raise ValueError("unsupported Protocol 13 debug preparation")
    backbone = QwenMultimodalBackboneConfig.from_value(data["backbone"])
    checkpoint = PrivateInitialCheckpointBinding.from_value(data["initial_checkpoint"])
    return backbone, checkpoint


def _validated_worker_interpreter(path: Path) -> Path:
    """Validate an absolute venv entrypoint without resolving away its environment."""

    expanded = path.expanduser()
    if not expanded.is_absolute() or not expanded.is_file():
        raise ValueError("MBPP worker interpreter must be an existing absolute file")
    return expanded


def build_application_config(
    *,
    run_id: str,
    steps: int,
    run_plan: ExactAttemptRunPlan,
    batch_size: int,
    checkpoint_every: int = 10,
    max_turns: int = 50,
    max_reasoning_tokens: int = 1024,
    max_action_tokens: int = 2048,
    max_input_tokens: int = 65_536,
) -> tuple[ApplicationConfig, ExactAttemptRunPlan]:
    if run_plan.total_training_steps != steps:
        raise ValueError("training steps differ from the declared run plan")
    rollout_maximum = conservative_rollout_maximum(
        max_turns=max_turns,
        max_reasoning_tokens=max_reasoning_tokens,
        max_action_tokens=max_action_tokens,
        max_model_input_tokens=max_input_tokens,
        max_tool_wall_time_milliseconds=120_000,
    )
    application = ApplicationConfig(
        trainer=TrainerConfig(
            method=TTBMethodConfig(epsilon_min=0.1, temperature_beta=1.0),
            rollout=PolicyRolloutConfig(
                base_seed=0,
                max_turns=max_turns,
                max_reasoning_tokens=max_reasoning_tokens,
                max_action_tokens=max_action_tokens,
                per_rollout_maximum=rollout_maximum,
                format="skillev-policy-rollout@4",
                reasoning_native_thinking=True,
            ),
            optimizer=OptimizerConfig(
                adapter_learning_rate=1e-4,
                z_learning_rate=1e-4,
                weight_decay=0.0,
            ),
            execution=TrainingExecutionConfig(
                experiment_id=run_id,
                batch_size=batch_size,
            ),
            checkpoint=CheckpointConfig(every_n_steps=checkpoint_every),
        ),
        # Production windows are deliberately retained.  A bounded debug run should
        # exercise diagnostics/calibration without manufacturing an artificial phase.
        diagnostics=DiagnosticsConfig(),
        calibration=CalibrationConfig(),
        evolution=EvolutionConfig(generate_min_absolute_log_importance=0.1),
        authoring_sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        maximum_h0_tokens=32_768,
    )
    return application, run_plan


def _public_identity(
    *,
    application: ApplicationConfig,
    run_plan: ExactAttemptRunPlan,
    task_ids: tuple[str, ...],
    checkpoint: PrivateInitialCheckpointBinding,
    mbpp_profile: MBPPScorerProfile,
    sampling_condition: str = SEVEN_DOMAIN_CONDITION,
    entry_kind: str = "debug",
    healthbench_judge: str = "qwen-local@1",
    initial_skill_profile: str = "public-advisory@2",
    alfworld_goal_binding: str = "catalog-instruction@1",
    format_review_from_step: int | None = None,
) -> ProtocolV10ApplicationIdentity:
    from skillev.evaluation.healthbench_luna_profile import (
        LEGACY_TRAINING_JUDGE,
        healthbench_condition,
    )
    from skillev.evolution.task_features import TASK_FEATURE_MAPPING_VERSION

    terminal_condition_extensions: dict[str, JsonValue] = (
        {"healthbench": healthbench_condition(healthbench_judge)}
        if healthbench_judge != LEGACY_TRAINING_JUDGE
        else {}
    )
    if format_review_from_step is not None:
        from skillev.evaluation.format_content_review import format_review_condition

        terminal_condition_extensions["format_content_review"] = format_review_condition(
            format_review_from_step
        )

    from skillev_private.benchmarks.alfworld_public_goal import (
        LEGACY_GOAL_BINDING,
        require_goal_binding,
    )

    require_goal_binding(alfworld_goal_binding)
    # The runtime session freezes this public reset contract alongside scorers.
    # Preserve the exact historical projection when the legacy binding is used.
    if alfworld_goal_binding != LEGACY_GOAL_BINDING:
        terminal_condition_extensions["alfworld_goal_binding"] = alfworld_goal_binding

    seeds = planned_seed_documents(initial_skill_profile)
    library = SkillLibraryState.from_seed_documents(seeds)
    task_sequence = stable_hash(
        {
            "algorithm": sampling_condition,
            "format": f"seven-training-{entry_kind}-task-sequence@1",
            "task_ids": list(task_ids),
        }
    )
    method_identity = stable_hash(
        {
            "authority": "idea.tex",
            "format": f"seven-bayesian-improve-{entry_kind}-method@1",
            "method": FormalMethodV10.BAYESIAN_IMPROVE_FULL.value,
        }
    )
    protocol_identity = stable_hash(
        {
            "effective_batch_size": application.trainer.execution.batch_size,
            "domain_count": application.trainer.execution.batch_size // TRAJECTORIES_PER_QUESTION,
            "format": f"seven-training-{entry_kind}@1",
            "questions_per_step": application.trainer.execution.batch_size
            // TRAJECTORIES_PER_QUESTION,
            "steps": run_plan.total_training_steps,
            "trajectories_per_question": TRAJECTORIES_PER_QUESTION,
        }
    )
    snapshot = RuntimeSnapshotIdentity(
        builder_kind=AttemptBuilderKind.FULL,
        public_identity_content_hash=stable_hash(
            {
                "application": application.to_value(),
                "method": method_identity,
                "protocol": protocol_identity,
                "run_plan": run_plan.to_value(),
                "tasks": task_sequence,
                **(
                    {"terminal_evaluation_conditions": terminal_condition_extensions}
                    if terminal_condition_extensions
                    else {}
                ),
            }
        ),
        application_config_hash=application.content_hash,
        method_identity_hash=method_identity,
        protocol_hash=protocol_identity,
        protocol_freeze_id=stable_hash(
            {"format": f"seven-{entry_kind}-condition@1", "protocol": protocol_identity}
        ),
        run_plan_hash=run_plan.content_hash,
        initial_library_version=library.current_version,
        initial_skill_library_state_hash=library.state_hash,
        initial_trainable_state_hash=checkpoint.trainable_state.content_hash,
        ordered_task_sequence_hash=task_sequence,
        sampling_schedule_algorithm=sampling_condition,
        sampling_schedule_hash=task_sequence,
        tokenizer_artifact_hash=None,
        base_model_artifact_hash=None,
        implementation_build_hash=None,
        terminal_evaluation_conditions_json=canonical_json(
            {"mbpp-plus": mbpp_profile.to_value(), **terminal_condition_extensions}
        ),
        task_feature_mapping_version=TASK_FEATURE_MAPPING_VERSION,
    )
    return ProtocolV10ApplicationIdentity(
        method=FormalMethodV10.BAYESIAN_IMPROVE_FULL,
        application_config=application,
        run_plan=run_plan,
        initial_run_cursor=AttemptRunCursorState.fresh(run_plan),
        snapshot_identity=snapshot,
        phase_checkpoint_cycle_ordinals=tuple(range(1, run_plan.maximum_cycles + 1)),
    )


def _phi_per_cycle(application: ApplicationConfig, calls: int) -> BudgetVector:
    """Resolve the declared per-cycle authoring envelope, not a hidden cycle cap."""
    if type(calls) is not int or calls < 1:
        raise ValueError("authoring calls per cycle must be a positive integer")
    return BudgetVector(
        input_tokens=calls * application.evolution.max_authoring_prompt_tokens,
        output_tokens=calls * application.evolution.max_authoring_completion_tokens,
        model_calls=calls,
    )


def _authoring_authority(tasks: tuple[RolloutTask, ...]) -> SkillAuthoringAuthority:
    return SkillAuthoringAuthority(
        input_schema_id="skillev-public-task@1",
        output_schema_id="skillev-structured-action@1",
        license_id="CC0-1.0",
        allowed_task_families=tuple(sorted({task.task_family for task in tasks})),
        allowed_tools=tuple(sorted({tool for task in tasks for tool in task.available_tools})),
    )


def _start_progress_monitor(
    application: SKILLEVApplication,
    *,
    total_steps: int,
    performance_path: Path,
    run_started: float,
    planning_hours: float = 72.0,
    target_steps_per_hour: float = 4.2,
    elapsed_since_run_start: Callable[[], float] | None = None,
    metrics_condition_id: str | None = None,
    metrics_condition_starts: dict[int, str] | None = None,
    controller_observe: Callable[[], dict[str, Any]] | None = None,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    initial_step = application.training_loop.optimizer_step
    branch = performance_path.parent / "branch-source.json"
    metrics_after = json.loads(branch.read_text())["optimizer_step"] if branch.exists() else 0
    previous: dict[int, dict[str, JsonValue]] = {}
    incomplete_monitor_records = 0
    if performance_path.exists():
        previous_text = performance_path.read_text(encoding="utf-8")
        if previous_text and not previous_text.endswith("\n"):
            with performance_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")  # Preserve partial text; isolate the next advisory row.
        for line in previous_text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Advisory output may end mid-line after a crash. Method recovery
                # is owned by the snapshot/journal, never by this timing sidecar.
                incomplete_monitor_records += 1
                continue
            if row.get("committed") is True and row["optimizer_step"] <= initial_step:
                previous[row["optimizer_step"]] = row
    prior_wall = sum(float(str(row["step_wall_seconds"])) for row in previous.values())

    def monitor_updates(metrics: MetricsStore | None) -> None:
        offset = 0
        recorded = 0
        measured = [
            float(str(row["step_wall_seconds"]))
            for row in previous.values()
            if row.get("warmup") is False
        ]
        next_live = 0.0
        while True:
            controller = None if controller_observe is None else controller_observe()
            timings = application.training_loop.finalized_timings
            if metrics is not None:
                assert metrics_condition_id is not None
                offset, added = export_available(
                    performance_path.parent / "events.jsonl",
                    metrics,
                    condition_id=metrics_condition_id,
                    condition_starts=metrics_condition_starts,
                    offset=offset,
                    committed_through=timings[-1].optimizer_step if timings else initial_step,
                    committed_after=metrics_after,
                )
                if added or not (performance_path.parent / "committed-metrics.jsonl").exists():
                    metrics.publish(performance_path.parent / "committed-metrics.jsonl")
            for timing in timings[recorded:]:
                elapsed = time.monotonic() - run_started
                planning_elapsed = (
                    prior_wall + elapsed
                    if elapsed_since_run_start is None
                    else elapsed_since_run_start()
                )
                wall = timing.committed - timing.started
                if not timing.is_process_warmup:
                    measured.append(wall)
                payload = timing.to_value()
                if timing.gradient_detail is not None:
                    durable_json(
                        performance_path.parent / f"gradient-work-{timing.optimizer_step:08d}.json",
                        normalize_json(timing.gradient_detail),
                    )
                if timing.rollout_detail is not None:
                    durable_json(
                        performance_path.parent
                        / f"rollout-phases-{timing.optimizer_step:08d}.json",
                        timing.rollout_detail,
                    )
                if controller is not None:
                    gpu_count = controller["resources"]["reserved_gpu_count"]
                    payload["step_reserved_gpu_hours"] = wall * gpu_count / 3600
                    payload["reserved_gpu_count"] = gpu_count
                    payload["resource_accounting"] = "step-reservation-overlaps-process-reservation"
                payload["run_elapsed_seconds"] = planning_elapsed
                payload["process_elapsed_seconds"] = elapsed
                payload["incomplete_monitor_records"] = incomplete_monitor_records
                payload["planning_elapsed_basis"] = (
                    "prior-committed-wall-plus-current-process"
                    if elapsed_since_run_start is None
                    else "persistent-wall-origin-including-failures-and-downtime"
                )
                payload["rate_scope"] = "complete-transactions-including-any-evolution"
                payload["planning_hours_not_hard_deadline"] = planning_hours
                payload["prior_committed_wall_seconds"] = prior_wall
                payload["forecast"] = forecast_deadline(
                    completed_steps=timing.optimizer_step,
                    elapsed_seconds=planning_elapsed,
                    measured_step_seconds=tuple(measured),
                    future_extra_seconds=0.0,
                    total_steps=total_steps,
                    deadline_seconds=planning_hours * 3600,
                ).to_value()
                payload["stress_forecast"] = forecast_deadline(
                    completed_steps=timing.optimizer_step,
                    elapsed_seconds=planning_elapsed,
                    measured_step_seconds=tuple(measured),
                    future_extra_seconds=0.0,
                    total_steps=total_steps,
                    deadline_seconds=planning_hours * 3600,
                    growth_factor=1.1,
                ).to_value()
                rate = 3600 * len(measured) / sum(measured) if measured else None
                payload["post_warmup_steps_per_hour"] = rate
                payload["formal_target_steps_per_hour"] = target_steps_per_hour
                payload["post_warmup_step_seconds"] = list(measured)
                payload["acceptance_purpose"] = (
                    "four-step-correctness-not-throughput" if total_steps == 4 else "declared-run"
                )
                physical = getattr(application.generator, "physical_usage", {})
                payload["physical_generation_usage_cumulative"] = dict(physical)
                payload["throughput_gate"] = (
                    (
                        "insufficient-measured-steps"
                        if timing.optimizer_step == total_steps
                        else "warming-up"
                    )
                    if len(measured) < 3
                    else "pass"
                    if rate is not None and rate >= target_steps_per_hour
                    else "needs-human-decision"
                )
                with performance_path.open("a", encoding="utf-8") as handle:
                    handle.write(canonical_json(payload) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                print(
                    canonical_json(
                        {
                            "status": "committed",
                            "optimizer_step": timing.optimizer_step,
                            "steps_per_hour": rate,
                            "step_wall_seconds": wall,
                            "remaining_eta_seconds": None
                            if rate is None
                            else (total_steps - timing.optimizer_step) * 3600 / rate,
                            "forecast": payload["forecast"],
                        }
                    ),
                    flush=True,
                )
            recorded = len(timings)
            now = time.monotonic()
            if now >= next_live:
                durable_json(
                    performance_path.parent / "rollout-progress.json",
                    normalize_json(application.training_loop.rollout_progress),
                )
            if now >= next_live and application.training_loop.gradient_progress is not None:
                from .training_observability import serving_metrics

                service_config = getattr(application.generator, "config", None)
                endpoint = getattr(service_config, "endpoint_base", None)
                if endpoint is not None:
                    durable_json(
                        performance_path.parent / "serving-metrics.json",
                        normalize_json(serving_metrics(endpoint)),
                    )
                live = application.training_loop.gradient_progress
                assert live is not None
                durable_json(performance_path.parent / "stream-progress.json", normalize_json(live))
                print(json.dumps({"status": "gradient-progress", **live}), flush=True)
                next_live = now + 60
            if stop.wait(5.0):
                # A final durable step must not be lost between the last poll and stop.
                if len(application.training_loop.finalized_timings) == recorded:
                    return

    def monitor() -> None:
        metrics = None
        try:
            if metrics_condition_id is not None:
                metrics = MetricsStore(performance_path.parent / "committed-metrics.sqlite3")
            monitor_updates(metrics)
        except Exception as error:
            # Monitoring loss is not a bad answer or a partial optimizer abort.
            # Preserve the current transaction, then pause before another batch.
            durable_json(
                performance_path.parent / "monitor-failure.json",
                {
                    "error_type": type(error).__name__,
                    "action": "pause-after-complete-checkpoint",
                },
            )
            (performance_path.parent / "STOP_AFTER_CHECKPOINT").touch()
        finally:
            if metrics is not None:
                metrics.close()

    thread = threading.Thread(target=monitor, name="training-progress", daemon=True)
    thread.start()
    return stop, thread
