"""Run a bounded, real Protocol 13 BayesianImprove training debug attempt.

This entrypoint intentionally bypasses the formal-admission gate while retaining the
production application, rollout, evaluator, adapter-publication, distributed-gradient,
TTB, diagnostic, calibration, and checkpoint paths.  It is therefore an integration
debug run, never a publishable formal experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import torch
import torch.distributed as dist

from skillev.application import SKILLEVApplication, TerminalComponents
from skillev.application_config import ApplicationConfig
from skillev.contracts import JsonValue, canonical_json
from skillev.contracts.identity import validate_identifier
from skillev.evaluation.current_iid.protocol13.catalog import (
    CHECKPOINT_EVERY_STEPS,
    TRAINING_STEPS,
)
from skillev.evaluation.healthbench_luna_profile import LEGACY_TRAINING_JUDGE, PROFILE_ID
from skillev.evolution import (
    PhiBudgetAuthority,
)
from skillev.experiments._evolution_preflight_seed import planned_seed_documents
from skillev.policy import (
    PrivateInitialCheckpointBinding,
    QwenMultimodalBackboneConfig,
    build_qwen_policy_backbone,
    qwen_tokenizer_artifact_identity,
)
from skillev.rollout import ExternalSGLangRolloutConfig
from skillev.runtime import (
    BudgetLedger,
    LiveAttemptEventLog,
)
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.runtime.formal_sglang_runtime import (
    BoundFormalSGLangRuntime,
    FormalSGLangRuntimeBinding,
)
from skillev.runtime.sglang_gateway import SGLangGatewayConfig
from skillev.training import (
    FixedAttemptBudgetPlan,
    PrivateCheckpointStorageBinding,
)
from skillev.training.deadline import TrainingDeadline
from skillev.training.distributed_ttb import (
    DistributedTTBGradientCoordinator,
    DistributedTTBTopology,
    initialize_distributed_ttb,
    serve_distributed_ttb_worker,
)
from skillev.training.performance_config import TrainingPerformanceConfig
from skillev_private.benchmarks.mbpp_scoring import resolve_mbpp_profile
from skillev_private.benchmarks.protocol_v13_seven_training import (
    EFFECTIVE_BATCH_SIZE,
    QUESTIONS_PER_STEP,
    SEVEN_DOMAIN_CONDITION,
    load_seven_domain_training_sources,
    seven_domain_training_trajectories,
)
from skillev_private.benchmarks.protocol_v13_training_sessions import (
    build_protocol13_training_sessions,
)

from .bayesian_training_setup import (
    _authoring_authority as _authoring_authority,
)
from .bayesian_training_setup import (
    _clock as _clock,
)
from .bayesian_training_setup import (
    _object as _object,
)
from .bayesian_training_setup import (
    _phi_per_cycle as _phi_per_cycle,
)
from .bayesian_training_setup import (
    _public_identity as _public_identity,
)
from .bayesian_training_setup import (
    _read_preparation as _read_preparation,
)
from .bayesian_training_setup import (
    _start_progress_monitor as _start_progress_monitor,
)
from .bayesian_training_setup import (
    _validated_worker_interpreter as _validated_worker_interpreter,
)
from .bayesian_training_setup import (
    build_application_config,
)
from .protocol_v10_attempt_builder import ProtocolV10OrderedTaskProvider

_PREPARATION_FORMAT: Final = "skillev-private-protocol13-training-debug-preparation@1"
_RUN_FORMAT: Final = "skillev-private-protocol13-training-debug-result@3"
_DEFAULT_DEBUG_STEPS: Final = 8
_GLOBAL_MAX_TURNS: Final = 50
_MAX_REASONING_TOKENS: Final = 1024
_MAX_ACTION_TOKENS: Final = 2048
_MAX_MODEL_INPUT_TOKENS: Final = 65_536
_MAX_TOOL_WALL_MILLISECONDS: Final = 120_000
_FORMAL_HARD_STEPS_PER_HOUR: Final = 250.0 / 72.0
_FORMAL_TARGET_STEPS_PER_HOUR: Final = 4.2
_PERFORMANCE_WARMUP_STEPS: Final = 2
_PERFORMANCE_MEASURED_STEPS: Final = 3
_DEFAULT_MAX_RESIDENT_TRAJECTORIES: Final = 16
_DEFAULT_MAX_INFLIGHT_MODEL_REQUESTS: Final = 8
_DEFAULT_MAX_INFLIGHT_ENVIRONMENT_CALLS: Final = 8
_DEFAULT_MAX_INFLIGHT_TERMINAL_EVALUATIONS: Final = 8
_DEFAULT_MAX_INFLIGHT_PROCESS_GRADERS: Final = 4
_DEFAULT_TRANSPORT_WORKER_THREADS: Final = 16
# SGLang 0.5.17 fuses Qwen3.5 attention/MLP projections but has no dynamic-LoRA
# buffers for the Hugging Face model's in_proj_qkv/z/b/a GDN projections.  Train
# exactly the common subset so the sampled and teacher-forced policies remain the
# same adapter rather than silently dropping trainable weights at serving time.
_SGLANG_COMPATIBLE_LORA_TARGET_MODULES: Final = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _prepare(arguments: argparse.Namespace) -> None:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    model_path = arguments.model_path.resolve()
    output_root = arguments.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = raw.get("text_config")
    if not isinstance(text_config, dict) or type(text_config.get("hidden_size")) is not int:
        raise ValueError("Qwen3.5 config lacks text hidden_size")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=arguments.model_revision,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("Protocol 13 training requires the pinned fast tokenizer")
    tokenizer_identity = qwen_tokenizer_artifact_identity(
        tokenizer=tokenizer,
        tokenizer_id=arguments.tokenizer_id,
        # QwenBackboneConfig carries one frozen revision for both local model and
        # tokenizer loading; bind the tokenizer content to that same revision.
        revision=arguments.model_revision,
    )
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("Qwen3.5 tokenizer lacks the required <|im_end|> token")
    backbone_config = QwenMultimodalBackboneConfig(
        base_model_path=str(model_path),
        revision=arguments.model_revision,
        tokenizer_id=tokenizer_identity.tokenizer_id,
        tokenizer_path=str(model_path),
        tokenizer_content_hash=tokenizer_identity.content_hash,
        hidden_size=cast(int, text_config["hidden_size"]),
        device="cuda",
        torch_dtype="bfloat16",
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=_SGLANG_COMPATIBLE_LORA_TARGET_MODULES,
        z_hidden_width=32,
        eos_token_ids=(eos_token_id,),
        # The disposable runtime overlay carries the repository-pinned FLA kernel.
        # Checkpointing plus exact saved-tensor CPU offload bounds long-trajectory
        # teacher-forcing memory without changing the TTB objective or token spans.
        teacher_forced_gradient_checkpointing=True,
        attention_implementation="sdpa",
    )
    output_root.mkdir(parents=True, mode=0o700)
    checkpoint_directory = output_root / "initial-policy"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    backbone = build_qwen_policy_backbone(backbone_config)
    initial_state = backbone.trainable_state_identity
    backbone.bind_initial_trainable_state(initial_state)
    backbone.save_checkpoint(str(checkpoint_directory))
    binding = PrivateInitialCheckpointBinding(
        directory=str(checkpoint_directory),
        trainable_state=initial_state,
    )
    value = {
        "backbone": backbone_config.to_value(),
        "format": _PREPARATION_FORMAT,
        "initial_checkpoint": binding.to_value(),
    }
    (output_root / "preparation.json").write_text(
        canonical_json(value) + "\n",
        encoding="utf-8",
    )
    print(
        canonical_json(
            {
                "checkpoint": str(checkpoint_directory),
                "format": _PREPARATION_FORMAT,
                "status": "prepared",
            }
        ),
        flush=True,
    )


def _application_config(
    *,
    run_id: str,
    steps: int = _DEFAULT_DEBUG_STEPS,
    run_plan: ExactAttemptRunPlan | None = None,
) -> tuple[ApplicationConfig, ExactAttemptRunPlan]:
    if not 2 <= steps <= TRAINING_STEPS:
        raise ValueError("Protocol 13 debug steps must lie within the formal schedule")
    if run_plan is None:
        if steps > _DEFAULT_DEBUG_STEPS:
            raise ValueError(
                "long training requires an explicit run plan, not the short debug default"
            )
        run_plan = ExactAttemptRunPlan(
            phase_search_steps=steps - 1,
            closure_steps=1,
            maximum_cycles=1,
        )
    if run_plan.total_training_steps != steps:
        raise ValueError("requested steps differ from the declared run plan")
    return build_application_config(
        run_id=run_id,
        steps=steps,
        run_plan=run_plan,
        batch_size=EFFECTIVE_BATCH_SIZE,
        checkpoint_every=CHECKPOINT_EVERY_STEPS,
    )


def _performance(arguments: argparse.Namespace) -> TrainingPerformanceConfig:
    profile = TrainingPerformanceConfig.load(arguments.performance_config)
    overrides = {
        field: getattr(arguments, argument, None)
        for field, argument in {
            "resident_trajectories": "max_resident_trajectories",
            "actor_requests": "max_inflight_model_requests",
            "environment_calls": "max_inflight_environment_calls",
            "terminal_evaluations": "max_inflight_terminal_evaluations",
            "process_graders": "max_inflight_process_graders",
            "transport_threads": "transport_worker_threads",
            "coordinator_participates": "coordinator_gradient_shard",
        }.items()
    }
    return replace(profile, **{key: value for key, value in overrides.items() if value is not None})


def _runtime_binding(arguments: argparse.Namespace, run_root: Path) -> FormalSGLangRuntimeBinding:
    profile = _performance(arguments)
    workflow = profile.workflow()
    return FormalSGLangRuntimeBinding(
        gateway=SGLangGatewayConfig(
            endpoint_base=arguments.endpoint,
            base_model=arguments.base_model,
            supervisor_adapter=f"{arguments.adapter_namespace}-supervisor",
            seed=0,
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=4096,
            request_timeout_seconds=600.0,
        ),
        rollout=ExternalSGLangRolloutConfig(
            endpoint_base=arguments.endpoint,
            request_timeout_seconds=600.0,
            transport_worker_threads=profile.transport_threads,
        ),
        workflow=workflow,
        adapter_export_root=run_root / "adapters",
        adapter_namespace=arguments.adapter_namespace,
        adapter_keep_recent=3,
        performance=profile,
    )


async def _run_coordinator(
    arguments: argparse.Namespace,
    topology: DistributedTTBTopology,
) -> None:
    run_started = time.monotonic()
    run_root = arguments.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True, mode=0o700)
    backbone_config, checkpoint = _read_preparation(arguments.preparation.resolve())
    mbpp_interpreter = _validated_worker_interpreter(arguments.mbpp_interpreter)
    mbpp_source_root = arguments.mbpp_source_root.resolve()
    mbpp_profile = resolve_mbpp_profile(
        {
            "source_root": str(mbpp_source_root),
            "profile": {}
            if arguments.mbpp_profile is None
            else json.loads(arguments.mbpp_profile.read_text()),
        }
    )
    (run_root / "terminal-evaluation.json").write_text(
        canonical_json({"mbpp-plus": mbpp_profile.to_value()}) + "\n",
        encoding="utf-8",
    )
    records = load_seven_domain_training_sources(arguments.dataset.resolve())
    required_questions = arguments.steps * QUESTIONS_PER_STEP
    selected = seven_domain_training_trajectories(records, steps=arguments.steps)
    required_trajectories = arguments.steps * EFFECTIVE_BATCH_SIZE
    if len(selected) != required_trajectories:
        raise AssertionError("Protocol 13 rollout expansion differs from the effective batch")
    (run_root / "execution-profile.json").write_text(
        canonical_json(_performance(arguments).to_value()) + "\n", encoding="utf-8"
    )
    coordinator = DistributedTTBGradientCoordinator(
        topology,
        coordinator_participates=_performance(arguments).coordinator_participates,
        pipeline_mode=_performance(arguments).pipeline_mode,
    )
    runtime = BoundFormalSGLangRuntime.build(
        binding=_runtime_binding(arguments, run_root),
        gradient_preparer=coordinator,
    )
    application: SKILLEVApplication | None = None
    stop: threading.Event | None = None
    thread: threading.Thread | None = None
    try:
        hydration_started = time.monotonic()
        tasks, sessions = await build_protocol13_training_sessions(
            selected,
            deployments_path=arguments.deployments.resolve(),
            endpoint_base=arguments.endpoint,
            base_model=arguments.base_model,
            resources=runtime.resources,
            mbpp_interpreter=mbpp_interpreter,
            mbpp_source_root=mbpp_source_root,
            mbpp_profile=mbpp_profile,
            healthbench_judge=getattr(arguments, "healthbench_judge", PROFILE_ID),
        )
        hydration_seconds = time.monotonic() - hydration_started
        application_config, run_plan = _application_config(
            run_id=run_root.name,
            steps=arguments.steps,
            run_plan=None
            if arguments.run_plan is None
            else ExactAttemptRunPlan.from_value(
                json.loads(arguments.run_plan.read_text(encoding="utf-8"))
            ),
        )
        identity = _public_identity(
            application=application_config,
            run_plan=run_plan,
            task_ids=tuple(task.task_id for task in tasks),
            checkpoint=checkpoint,
            mbpp_profile=mbpp_profile,
            healthbench_judge=getattr(arguments, "healthbench_judge", PROFILE_ID),
        )
        phi_per_cycle = _phi_per_cycle(
            application_config,
            arguments.phi_calls_per_cycle,
        )
        attempt_budget = FixedAttemptBudgetPlan.from_trainer_and_run_plan(
            trainer=application_config.trainer,
            run_plan=run_plan,
            phi_per_cycle_maximum=phi_per_cycle,
        ).required()
        (run_root / "resolved-run-plan.json").write_text(
            canonical_json(
                {
                    "application": application_config.to_value(),
                    "terminal_evaluation_conditions": {"mbpp-plus": mbpp_profile.to_value()},
                    "task_feature_mapping_version": sessions.task_feature_mapping_version,
                    "run_plan": run_plan.to_value(),
                    "phi_per_cycle_maximum": phi_per_cycle.to_value(),
                    "total_attempt_budget": attempt_budget.to_value(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        terminal = TerminalComponents(
            ledger=BudgetLedger(
                run_id=run_root.name,
                attempt_id=f"real-{arguments.steps}-step-debug",
                cap=attempt_budget,
            ),
            authoring_authority=_authoring_authority(tasks),
            phi_budget=PhiBudgetAuthority(phi_per_cycle),
        )
        event_log = LiveAttemptEventLog(
            run_root / "events.jsonl",
            run_id=run_root.name,
            attempt_id=f"real-{arguments.steps}-step-debug",
        )
        model_started = time.monotonic()
        application = SKILLEVApplication.build_formal(
            backbone_config=backbone_config,
            task_provider=ProtocolV10OrderedTaskProvider(
                tasks,
                SEVEN_DOMAIN_CONDITION,
            ),
            base_session_factory=sessions,
            seed_documents=planned_seed_documents(),
            terminal_components=terminal,
            checkpoint_storage=PrivateCheckpointStorageBinding(
                directory=str(run_root / "checkpoints")
            ),
            initial_checkpoint=checkpoint,
            public_identity=identity,
            event_log=event_log,
            clock=_clock,
            runtime=runtime.dependencies(),
        )
        (run_root / "preparation-timing.json").write_text(
            canonical_json(
                {
                    "environment_evaluator_preparation_seconds": hydration_seconds,
                    "model_application_startup_seconds": time.monotonic() - model_started,
                    "total_preparation_seconds": time.monotonic() - run_started,
                    "sampling_condition": SEVEN_DOMAIN_CONDITION,
                    "jit_warmup": "first-two-steps-of-each-process-all-wall-times-retained",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        application.training_loop.execution_deadline = TrainingDeadline(run_started)
        initial_state = application.backbone.trainable_state_identity
        stop, thread = _start_progress_monitor(
            application,
            total_steps=run_plan.total_training_steps,
            performance_path=run_root / "performance.jsonl",
            run_started=run_started,
        )
        started = time.monotonic()
        summary = await application.evolution_loop.run(run_plan)
        elapsed = time.monotonic() - started
        application.training_loop.ledger.assert_fully_settled()
        if summary.final_optimizer_step != arguments.steps:
            raise RuntimeError("Protocol 13 debug did not commit the requested optimizer steps")
        changed = application.backbone.trainable_state_identity != initial_state
        result: dict[str, JsonValue] = {
            "completed_under_wall_time_deadline": time.monotonic() - run_started <= 72 * 3600,
            "run_elapsed_seconds": time.monotonic() - run_started,
            "elapsed_seconds": elapsed,
            "formal_estimated_hours": TRAINING_STEPS / (arguments.steps / elapsed * 3600.0),
            "formal_hard_steps_per_hour": _FORMAL_HARD_STEPS_PER_HOUR,
            "formal_target_steps_per_hour": _FORMAL_TARGET_STEPS_PER_HOUR,
            "format": _RUN_FORMAT,
            "mode": "nonformal-real-training-debug",
            "questions_consumed": required_questions,
            "summary": summary.to_value(),
            "trajectories_consumed": required_trajectories,
            "trainable_state_changed": changed,
            "steps_per_hour": arguments.steps / elapsed * 3600.0,
        }
        (run_root / "summary.json").write_text(
            canonical_json(result) + "\n",
            encoding="utf-8",
        )
        print(
            canonical_json(
                {
                    "elapsed_seconds": round(elapsed, 3),
                    "final_optimizer_step": summary.final_optimizer_step,
                    "status": "passed",
                    "trainable_state_changed": changed,
                }
            ),
            flush=True,
        )
    finally:
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=5.0)
        coordinator.close()


def _run_gradient_worker(
    arguments: argparse.Namespace,
    topology: DistributedTTBTopology,
) -> None:
    backbone_config, checkpoint = _read_preparation(arguments.preparation.resolve())
    backbone = build_qwen_policy_backbone(backbone_config, performance=_performance(arguments))
    backbone.load_checkpoint(checkpoint.directory)
    backbone.bind_initial_trainable_state(checkpoint.trainable_state)
    serve_distributed_ttb_worker(topology=topology, backbone=backbone)


def _require_two_gradient_workers(profile: TrainingPerformanceConfig, world_size: int) -> None:
    """Training uses one separate serving card and two real gradient owners."""
    if world_size != 2 or not profile.coordinator_participates:
        raise ValueError(
            "training requires two gradient ranks with coordinator participation, "
            "plus a separate inference GPU; no one-gradient fallback"
        )


def _run(arguments: argparse.Namespace) -> None:
    if not 2 <= arguments.steps <= TRAINING_STEPS:
        raise ValueError("debug steps require a search slot and a closure slot")
    validate_identifier(arguments.run_root.resolve().name)
    resolved, _ = _application_config(
        run_id=arguments.run_root.resolve().name,
        steps=arguments.steps,
        run_plan=None
        if getattr(arguments, "run_plan", None) is None
        else ExactAttemptRunPlan.from_value(
            json.loads(arguments.run_plan.read_text(encoding="utf-8"))
        ),
    )
    _phi_per_cycle(resolved, getattr(arguments, "phi_calls_per_cycle", 64))
    profile = _performance(arguments)
    _require_two_gradient_workers(profile, int(os.environ.get("WORLD_SIZE", "1")))
    profile.configure_process()
    topology = initialize_distributed_ttb(timeout_minutes=180)
    try:
        if topology.rank == 0:
            asyncio.run(_run_coordinator(arguments, topology))
        else:
            _run_gradient_worker(arguments, topology)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--model-path", type=Path, required=True)
    prepare.add_argument("--model-revision", required=True)
    prepare.add_argument("--tokenizer-id", default="Qwen/Qwen3.5-9B")
    prepare.add_argument("--output-root", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--preparation", type=Path, required=True)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--deployments", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--base-model", required=True)
    run.add_argument(
        "--healthbench-judge", choices=(PROFILE_ID, LEGACY_TRAINING_JUDGE), default=PROFILE_ID
    )
    run.add_argument("--mbpp-interpreter", type=Path, required=True)
    run.add_argument("--mbpp-source-root", type=Path, required=True)
    run.add_argument(
        "--mbpp-profile",
        type=Path,
        help="Explicit custom scoring profile; native IID defaults when omitted",
    )
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--adapter-namespace", required=True)
    run.add_argument(
        "--performance-config",
        type=Path,
        default=Path("configs/training/protocol13_performance.yaml"),
    )
    run.add_argument("--steps", type=int, default=_DEFAULT_DEBUG_STEPS)
    run.add_argument(
        "--phi-calls-per-cycle",
        type=int,
        default=64,
        help="Declared maximum base-author calls per cycle; total budget scales by cycle capacity",
    )
    run.add_argument(
        "--run-plan",
        type=Path,
        help="Explicit phase-search/closure/cycle budget JSON; required beyond short debug runs",
    )
    run.add_argument(
        "--coordinator-gradient-shard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the profile: let rank zero compute an exact TTB gradient shard",
    )
    run.add_argument(
        "--max-resident-trajectories",
        type=int,
        default=None,
    )
    run.add_argument(
        "--max-inflight-model-requests",
        type=int,
        default=None,
    )
    run.add_argument(
        "--max-inflight-environment-calls",
        type=int,
        default=None,
    )
    run.add_argument(
        "--max-inflight-terminal-evaluations",
        type=int,
        default=None,
    )
    run.add_argument(
        "--max-inflight-process-graders",
        type=int,
        default=None,
    )
    run.add_argument(
        "--transport-worker-threads",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.command == "prepare":
        _prepare(arguments)
    else:
        _run(arguments)


if __name__ == "__main__":
    main()


__all__ = ["main"]
