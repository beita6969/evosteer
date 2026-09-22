"""Bounded real training transactions with one local gradient owner and external SGLang.

Owner-authorized diagnostic topology only, NOT formal admission or an A0 result.
No IID/development episode loader, forced skill read, mutation, or fallback policy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

from skillev.application import SKILLEVApplication, TerminalComponents
from skillev.application_reporting import resolved_method_state
from skillev.evolution import PhiBudgetAuthority
from skillev.experiments._evolution_preflight_seed import planned_seed_documents
from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator
from skillev.runtime import BudgetLedger, LiveAttemptEventLog, StepTransactionJournal
from skillev.runtime.formal_sglang_runtime import BoundFormalSGLangRuntime
from skillev.runtime.serving_profile import require_training_service
from skillev.training import FixedAttemptBudgetPlan, PrivateCheckpointStorageBinding
from skillev.training.inflight import durable_json
from skillev.training.performance_config import TrainingPerformanceConfig
from skillev.training.run_observer import CommittedRunObserver
from skillev.training.stopping import StopAfterCheckpoint, TrainingPausedError
from skillev_private.benchmarks.mbpp_scoring import MBPPScorerProfile, resolve_mbpp_profile
from skillev_private.benchmarks.protocol_v13_seven_training import (
    load_seven_domain_training_sources,
    seven_domain_training_trajectories,
)
from skillev_private.benchmarks.protocol_v13_training_sessions import (
    build_protocol13_training_sessions,
)

from .bayesian_improve_training import FormalTrainingBindings
from .bayesian_training_config import BayesianFormalConfig
from .bayesian_training_setup import (
    _authoring_authority,
    _clock,
    _phi_per_cycle,
    _public_identity,
    _read_preparation,
    _start_progress_monitor,
    _validated_worker_interpreter,
)
from .fresh_restart import load_fresh_config, require_clean_initial_application
from .local_diagnostic_continuation import require_append
from .protocol_v10_attempt_builder import (
    ProtocolV10OrderedTaskProvider,
    ProtocolV10TaskProviderFactory,
)
from .training_controller import TrainingController
from .training_data_condition import data_condition_scientific
from .warmup_initialization import initialization_condition, require_initialization_candidate

FORMAT = "skillev-single-local-gradient-diagnostic@1"


def diagnostic_config(
    source: BayesianFormalConfig, *, total_steps: int = 3
) -> BayesianFormalConfig:
    """Only shorten the declared schedule; retain all actual candidate method fields."""
    if (
        source.cold_start is None
        or source.cold_start.min_batches != 2
        or source.window != 50
        or source.batch_size != 28
    ):
        raise ValueError("diagnostic requires the declared B28/window50/cold-start candidate")
    if source.skill_exposure != "catalog-then-read@1":
        raise ValueError("diagnostic requires autonomous catalog/read, not a forced-read quota")
    if source.max_turns > 25 or source.static_max_turns != 8:
        raise ValueError("diagnostic must preserve static8 and ALF at most25")
    if total_steps not in (3, 5, 10):
        raise ValueError("only fresh3/fresh5 or explicitly appended total10 is supported")
    return replace(source, steps=total_steps, closure_steps=1, maximum_cycles=2, checkpoint_every=1)


def load_bindings(path: Path) -> FormalTrainingBindings:
    value = json.loads(path.read_text())
    if value.get("format") != FORMAT or value.get("topology") is not None:
        raise ValueError("explicit single-local diagnostic bindings required, not formal topology")
    if any(key in value for key in ("resume", "iid_baselines", "development_records")):
        raise ValueError("diagnostic starts fresh on real training records only")
    bindings = FormalTrainingBindings.load(path)
    devices = bindings.training_gpu_uuids
    if len(devices) != 1 or not devices[0].startswith("GPU-"):
        raise ValueError("exactly one explicit physical gradient GPU UUID is required")
    if not bindings.serving_gpu_uuid.startswith("GPU-") or bindings.serving_gpu_uuid in devices:
        raise ValueError("the existing external actor must own a different physical GPU")
    if bindings.data_condition is None or bindings.evidence_mirror_root is None:
        raise ValueError("declare actual training data and an isolated persistent evidence mirror")
    return bindings


def _gpu_uuid(value: object) -> UUID:
    """Normalize Torch/NVML display forms without changing device identity."""
    return UUID(str(value).removeprefix("GPU-"))


def initialize_cuda_owner(expected_uuid: str) -> dict[str, Any]:
    """Small context only, before HF weight loading; called only by the live CLI."""
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip() or "," in visible or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("diagnostic must expose exactly its single gradient device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("the declared local CUDA owner is not available")
    torch.cuda.set_device(0)
    torch.cuda.init()  # type: ignore[no-untyped-call]
    # Observed Torch 2.11 _CUuuid omits NVML's GPU- prefix. Compare the
    # underlying UUID, not display formatting; another physical GPU still fails.
    actual = torch.cuda.get_device_properties(0).uuid
    if _gpu_uuid(actual) != _gpu_uuid(expected_uuid):
        raise ValueError("visible CUDA owner differs from the explicitly bound GPU UUID")
    return {"owner_pid": os.getpid(), "gradient_gpu_uuid": expected_uuid, "cuda_initialized": True}


async def owner_handoff(
    identity: dict[str, Any],
    ready: Path | None,
    release: Path | None,
    stop: StopAfterCheckpoint,
) -> None:
    """Optional ten-minute operator handoff; never stops another process itself."""
    if ready is None and release is None:
        return
    if ready is None or release is None or not ready.is_absolute() or not release.is_absolute():
        raise ValueError("declare both absolute diagnostic handoff paths")
    if ready.exists() or release.exists() or ready == release:
        raise ValueError("handoff paths must be new, never an old release")
    durable_json(ready, {**identity, "state": "cuda-ready-before-backbone", "timeout_seconds": 600})
    deadline = time.monotonic() + 600
    while not release.exists():
        if stop():
            raise InterruptedError("owner stopped before backbone loading; no training began")
        if time.monotonic() >= deadline:
            raise TimeoutError("diagnostic CUDA handoff expired before backbone loading")
        await asyncio.sleep(0.25)
    value = json.loads(release.read_text())
    if any(value.get(key) != identity[key] for key in ("owner_pid", "gradient_gpu_uuid")):
        raise ValueError("handoff release belongs to a different CUDA owner")


def build_application(
    *,
    runtime: BoundFormalSGLangRuntime,
    storage: PrivateCheckpointStorageBinding,
    resume: Path | None = None,
    **kwargs: Any,
) -> SKILLEVApplication:
    """Same application/transaction machinery, explicit existing local gradient mode."""
    if runtime.gradient_preparer is not None:
        raise ValueError("local diagnostic cannot silently select distributed gradients")
    shared = runtime.shared_dependencies()
    constructor = SKILLEVApplication.build if resume is None else SKILLEVApplication.resume
    if resume is not None:
        kwargs["snapshot_directory"] = resume
    return constructor(
        **kwargs,
        checkpoint_storage=storage,
        generator_factory=shared.rollout_generator_factory,
        gradient_preparer=None,  # Explicit local prepare_ttb_step and LocalGradientStepStream.
        workflow_binding=shared.workflow_resources.binding,
        workflow_resources=shared.workflow_resources,
        step_adapter_publisher_factory=shared.step_adapter_publisher_factory,
        skill_author_factory=shared.skill_author_factory,
        step_transaction_journal=StepTransactionJournal(
            (Path(storage.directory) / "step-transactions").resolve()
        ),
    )


async def run(
    *,
    config_path: Path,
    bindings_path: Path,
    root: Path,
    resume: Path | None = None,
    total_steps: int = 3,
    cuda_owner_ready: Path | None = None,
    cuda_owner_release: Path | None = None,
) -> None:
    if (resume is None and total_steps not in (3, 5)) or (resume is not None and total_steps != 10):
        raise ValueError("fresh runs are three/five steps; only explicit Step3 append reaches ten")
    config = diagnostic_config(load_fresh_config(config_path), total_steps=total_steps)
    bindings = load_bindings(bindings_path)
    require_initialization_candidate(
        bindings.preparation, sampling=config.sampling_config.to_value()
    )
    initialization = initialization_condition(bindings.preparation)
    backbone, checkpoint = _read_preparation(bindings.preparation)
    config.require_backbone(backbone)
    if backbone.device not in {"cuda", "cuda:0"}:
        raise ValueError("diagnostic deployment preparation must bind its one CUDA owner")
    profile = TrainingPerformanceConfig.load(config.performance_profile)
    records = load_seven_domain_training_sources(bindings.dataset)
    selected = seven_domain_training_trajectories(records, steps=total_steps)
    if len(selected) != total_steps * 28:
        raise ValueError("diagnostic must retain every complete seven-domain B28 batch")
    append = (
        None
        if resume is None
        else require_append(
            root=root, resume=resume, config=config, bindings=bindings, selected=selected
        )
    )
    interpreter = _validated_worker_interpreter(bindings.evalplus_python)
    mbpp = resolve_mbpp_profile(
        {
            "source_root": str(bindings.evalplus_source_root),
            "profile": MBPPScorerProfile().to_value(),
        }
    )
    if append is None:
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
    output = root if append is None else root / f"continuation-step3-to10-{os.getpid()}"
    if append is not None:
        output.mkdir(mode=0o700, exist_ok=False)
    for name, path in (
        ("source-config.yaml", config_path),
        ("bindings-private.json", bindings_path),
    ):
        (output / name).write_bytes(path.read_bytes())
    durable_json(
        output / "diagnostic-condition.json",
        {
            "format": FORMAT,
            "purpose": "real-training-diagnostic-not-formal",
            "a0_admission": "not-claimed",
            "formal": config.to_value(),
            "execution_plan": (config.run_plan if append is None else append.plan).to_value(),
            "plan_semantics": "fresh" if append is None else "append-after-original-closure",
            "local_gradient_backend": "prepare_ttb_step-or-LocalGradientStepStream",
            "training_gpu_uuids": list(bindings.training_gpu_uuids),
            "serving_gpu_uuid": bindings.serving_gpu_uuid,
            **initialization,
            **data_condition_scientific(bindings.data_condition, selected),
        },
    )
    started = time.monotonic()
    runtime = BoundFormalSGLangRuntime.build_local_diagnostic(
        binding=bindings.runtime(root, profile)
    )
    evidence: CommittedRunObserver | None = None
    controller = TrainingController(
        root,
        resource_roles={
            "mode": FORMAT,
            **initialization,
            "external_actor": bindings.serving_gpu_uuid,
            "local_gradient": bindings.training_gpu_uuids[0],
        },
        gpu_uuids=(bindings.serving_gpu_uuid, *bindings.training_gpu_uuids),
        evidence=evidence,
        initial_committed_step=0 if append is None else append.snapshot.optimizer_step,
    )
    stop = StopAfterCheckpoint(root / "STOP_AFTER_CHECKPOINT")
    monitor_stop = thread = application = None
    try:
        with stop.signals():
            actual = await asyncio.to_thread(runtime.gateway.read_serving_profile)
            require_training_service(
                actual,
                model_path=backbone.base_model_path,
                tokenizer_path=backbone.tokenizer_path or backbone.base_model_path,
                base_model=bindings.base_model,
                minimum_context=config.max_input_tokens
                + max(config.maximum_reasoning_tokens, config.max_action_tokens),
                actor=True,
            )
            runtime.gateway.bind_serving_profile(actual)
            durable_json(output / "serving-runtime.json", actual)
            tasks, sessions = await build_protocol13_training_sessions(
                selected,
                deployments_path=bindings.deployments,
                endpoint_base=bindings.endpoint,
                request_journal_path=root / "requests.sqlite3",
                base_model=bindings.base_model,
                resources=runtime.resources,
                mbpp_interpreter=interpreter,
                mbpp_source_root=bindings.evalplus_source_root,
                mbpp_profile=mbpp,
                hotpot_deliberation=config.hotpot_deliberation,
                rollout_budget=config.task_budget,
                static_rollout_budget=config.static_task_budget,
                domain_rollout_budgets=config.domain_task_budgets,
                lazy_environments=True,
                healthbench_judge=config.healthbench_judge,
            )
            app_config = config.application_config(root.name)
            plan = config.run_plan if append is None else append.plan
            identity = _public_identity(
                application=app_config,
                run_plan=plan,
                task_ids=tuple(t.task_id for t in tasks),
                checkpoint=checkpoint,
                mbpp_profile=mbpp,
                sampling_condition=config.condition,
                entry_kind="local-training-diagnostic",
                healthbench_judge=config.healthbench_judge,
                initial_skill_profile=config.initial_skill_profile,
                alfworld_goal_binding=sessions.alfworld_goal_binding,
            )
            transition = None
            if append is not None:
                source_identity = _public_identity(
                    application=append.source_application,
                    run_plan=append.source_plan,
                    task_ids=tuple(t.task_id for t in tasks[:84]),
                    checkpoint=checkpoint,
                    mbpp_profile=mbpp,
                    sampling_condition=config.condition,
                    entry_kind="local-training-diagnostic",
                    healthbench_judge=config.healthbench_judge,
                    initial_skill_profile=config.initial_skill_profile,
                    alfworld_goal_binding=sessions.alfworld_goal_binding,
                )
                identity, transition = append.identity(
                    source_identity, identity, tuple(t.task_id for t in tasks)
                )
            phi = _phi_per_cycle(app_config, config.phi_calls_per_cycle)
            cap = FixedAttemptBudgetPlan.from_trainer_and_run_plan(
                trainer=app_config.trainer, run_plan=plan, phi_per_cycle_maximum=phi
            ).required()
            terminal = TerminalComponents(
                ledger=BudgetLedger(run_id=root.name, attempt_id="diagnostic-seed0", cap=cap),
                authoring_authority=_authoring_authority(tasks),
                phi_budget=PhiBudgetAuthority(phi),
            )
            if append is not None:
                terminal.ledger.restore_completed(append.entries)
            durable_json(
                output / "resolved-run-plan.json",
                {
                    "mode": FORMAT,
                    "application": app_config.to_value(),
                    "run_plan": plan.to_value(),
                    "total_attempt_budget": cap.to_value(),
                    "phi_per_cycle_maximum": phi.to_value(),
                    "performance": profile.to_value(),
                    **initialization,
                    "continuation": None
                    if append is None
                    else {
                        "source_checkpoint": str(resume),
                        "source_plan": append.source_plan.to_value(),
                        "original_closure_preserved_without_recheck": True,
                        "restored_optimizer_step": append.snapshot.optimizer_step,
                        "restored_task_cursor": append.snapshot.execution_state.task_cursor.cursor,
                        "restored_ledger_entries": len(append.entries),
                    },
                },
            )
            owner = initialize_cuda_owner(bindings.training_gpu_uuids[0])
            await owner_handoff(owner, cuda_owner_ready, cuda_owner_release, stop)
            if stop():
                raise InterruptedError("owner stopped before model construction")
            mode_kwargs = (
                {
                    "task_provider": ProtocolV10OrderedTaskProvider(tasks, config.condition),
                    "seed_documents": planned_seed_documents(config.initial_skill_profile),
                }
                if append is None
                else {
                    "task_provider_factory": ProtocolV10TaskProviderFactory(
                        tasks, config.condition
                    ),
                    "plan_continuation": transition,
                }
            )
            event_constructor = (
                LiveAttemptEventLog if append is None else LiveAttemptEventLog.resume
            )
            application = build_application(
                runtime=runtime,
                storage=PrivateCheckpointStorageBinding(directory=str(root / "checkpoints")),
                backbone_config=backbone,
                base_session_factory=sessions,
                terminal_components=terminal,
                initial_checkpoint=checkpoint,
                public_identity=identity,
                event_log=event_constructor(
                    root / "events.jsonl", run_id=root.name, attempt_id="diagnostic-seed0"
                ),
                clock=_clock,
                resume=resume,
                **mode_kwargs,
            )
            if append is None:
                require_clean_initial_application(
                    application,
                    preparation=bindings.preparation,
                    checkpoint_directory=Path(checkpoint.directory),
                    root=root,
                    initial_skill_profile=config.initial_skill_profile,
                )
            assert bindings.evidence_mirror_root is not None
            evidence = CommittedRunObserver(
                root,
                run_id=root.name,
                condition_id=config.condition,
                mirror_root=bindings.evidence_mirror_root,
                expected_batch_size=28,
                max_pending_steps=bindings.evidence_max_pending_steps,
                minimum_free_bytes=bindings.evidence_minimum_free_bytes,
            )
            controller.evidence = evidence
            application.training_loop.configure_inflight(
                root / "inflight",
                condition={
                    "diagnostic": FORMAT,
                    "formal": config.to_value(),
                    "performance": profile.to_value(),
                },
            )
            application.training_loop.enable_update_observation()
            controller.attach(application.training_loop)
            controller.set_state(None)
            durable_json(output / "initial-method-state.json", resolved_method_state(application))
            monitor_stop, thread = _start_progress_monitor(
                application,
                total_steps=total_steps,
                performance_path=root / "performance.jsonl",
                run_started=started,
                metrics_condition_id=config.condition,
                controller_observe=controller.publish,
            )

            async def boundary() -> bool:
                evidence_pause = await asyncio.to_thread(controller.checkpoint_boundary)
                return stop() or evidence_pause

            try:
                summary = await application.evolution_loop.run(plan, stop_requested=boundary)
            except TrainingPausedError as paused:
                application.training_loop.ledger.assert_fully_settled()
                durable_json(
                    output / "paused.json",
                    {
                        "status": "paused-after-complete-checkpoint",
                        "optimizer_step": paused.optimizer_step,
                        "checkpoint": str(paused.checkpoint),
                        "planned_steps": total_steps,
                        "mode": FORMAT,
                    },
                )
                controller.set_state("paused")
            else:
                if summary.final_optimizer_step != total_steps:
                    raise RuntimeError(
                        "diagnostic did not finish the declared complete transactions"
                    )
                application.training_loop.ledger.assert_fully_settled()
                await asyncio.to_thread(controller.checkpoint_boundary)
                durable_json(
                    output / "summary.json",
                    {
                        "mode": FORMAT,
                        "status": "finished",
                        "summary": summary.to_value(),
                        "a0_admission": "not-claimed",
                        "natural_evolution": (
                            "actual event evidence only; no forced mutation or skill call"
                        ),
                    },
                )
                controller.set_state("finished")
            durable_json(output / "final-method-state.json", resolved_method_state(application))
            durable_json(
                output / "final-evidence-status.json",
                await asyncio.to_thread(controller.finish_evidence),
            )
    except BaseException as error:
        controller.set_state("failed")
        durable_json(
            output / "failure-private.json",
            {
                "error_type": type(error).__name__,
                "traceback": "".join(traceback.format_exception(error)),
                "optimizer_step": None
                if application is None
                else application.training_loop.optimizer_step,
                "note": "partial work is not a committed diagnostic step; no resampling or retry",
            },
        )
        raise
    finally:
        if monitor_stop is not None:
            monitor_stop.set()
        if thread is not None:
            thread.join(timeout=15)
        try:
            if evidence is not None:
                evidence.close()
        finally:
            if application is not None and isinstance(
                application.generator, ExternalSGLangRolloutGenerator
            ):
                application.generator.close()
            controller.close_resources()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--runroot", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-steps", type=int, default=3, choices=(3, 5, 10))
    parser.add_argument("--cuda-owner-ready", type=Path)
    parser.add_argument("--cuda-owner-release", type=Path)
    args = parser.parse_args()
    asyncio.run(
        run(
            config_path=args.config.resolve(),
            bindings_path=args.bindings.resolve(),
            root=args.runroot.resolve(),
            resume=None if args.resume is None else args.resume.resolve(),
            total_steps=args.total_steps,
            cuda_owner_ready=args.cuda_owner_ready,
            cuda_owner_release=args.cuda_owner_release,
        )
    )


if __name__ == "__main__":
    main()
