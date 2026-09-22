"""Distributed Protocol 10 attempt worker.

Rank zero owns the formal application and all durable state.  Every other
rank constructs only the teacher-forced policy backbone and serves the closed
TTB gradient protocol.  The worker never imports the historical Protocol 9
catalog or its semantic selection.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch.distributed as dist

from skillev.application import SKILLEVApplication
from skillev.contracts.identity import utc_timestamp
from skillev.experiments.arms.builders import (
    FlowOnlyApplication,
    FlowOnlyArmApplicationInputs,
    build_no_bayesian_calibration_application,
)
from skillev.experiments.protocol_v10 import FormalMethodV10, load_active_protocol_v10
from skillev.experiments.protocol_v10_formal import load_protocol_v10_formal_experiment
from skillev.policy.hf_backbone import build_qwen_policy_backbone
from skillev.runtime import (
    AttemptRunProgress,
    EventType,
    FlowOnlyRuntimeExecutionState,
    LiveAttemptEventLog,
    RuntimeEventEmitter,
    StepTransactionReconciler,
    StepTransactionRecord,
    StepTransactionState,
)
from skillev.runtime.formal_sglang_runtime import BoundFormalSGLangRuntime
from skillev.runtime.sglang_gateway import SGLangGateway
from skillev.skillflow_baseline import (
    ExactSkillFlowApplication,
    ExactSkillFlowProtocolConfig,
    ExactSkillFlowTaskAdapter,
    ProtocolV10SkillFlowEpisodeRunner,
    build_exact_skillflow_protocol_v10_application,
    build_protocol_v10_skillflow_trainer,
)
from skillev.training import FilesystemTrainingCheckpointStore, TaskProvider
from skillev.training.distributed_ttb import (
    DistributedTTBGradientCoordinator,
    DistributedTTBTopology,
    initialize_distributed_ttb,
    serve_distributed_ttb_worker,
)
from skillev_private.benchmarks.protocol_v10_materialization import (
    load_protocol_v10_materialized_catalog,
)
from skillev_private.benchmarks.protocol_v10_session_deployments import (
    ProtocolV10SessionDeployments,
)

from .protocol_v10_application_input import (
    ExactSkillFlowApplicationContract,
    ProtocolV10ApplicationInput,
)
from .protocol_v10_attempt_builder import (
    ProtocolV10ApplicationInputs,
    ProtocolV10AttemptBuilder,
    ProtocolV10IntegrationSmokeBuilder,
    ProtocolV10MethodBuilders,
)
from .protocol_v10_attempt_input import ProtocolV10AdmissionMode, ProtocolV10AttemptInput


class ProtocolV10MethodUnavailableError(RuntimeError):
    """A formal method has no completed executable composition yet."""


_FORMAL_PROCESS_GROUP_TIMEOUT_MINUTES = 180


def _clock() -> str:
    return utc_timestamp(datetime.now(UTC))


def _bind_skillflow_authoring_service(exact: ProtocolV10AttemptInput) -> None:
    """Bind the upstream authoring client to the frozen base-model service."""
    if exact.method is not FormalMethodV10.SKILLFLOW_BASELINE:
        return
    expected = {
        "SKILL_CREATOR_API_BASE": exact.sglang.gateway.api_root + "/v1/messages",
        "SKILL_CREATOR_MODEL": exact.sglang.gateway.base_model,
    }
    for name, value in expected.items():
        inherited = os.environ.get(name)
        if inherited is not None and inherited != value:
            raise RuntimeError(f"formal SkillFlow inherited a conflicting {name}")
        os.environ[name] = value


def _initialize_formal_topology() -> DistributedTTBTopology:
    """Cover coordinator-only rollout and evolution phases between collectives."""
    return initialize_distributed_ttb(
        timeout_minutes=_FORMAL_PROCESS_GROUP_TIMEOUT_MINUTES,
    )


def _bind_skillflow_distributed_gradient(
    trainer_object: Any,
    *,
    resume_snapshot: Path | None,
) -> None:
    """Join the worker startup barrier before issuing any gradient collective."""
    from training.distributed_gradient import DistributedGradientClient, coordinator_barrier

    model = trainer_object.shared_model
    values = trainer_object.config
    client = DistributedGradientClient.create(
        model,
        initial_micro_batch=int(values.get("distributed_micro_batch", 4)),
        minimum_micro_batch=int(values.get("distributed_minimum_micro_batch", 1)),
    )
    coordinator_barrier()
    trainer_object.distributed_gradient_client = client
    if resume_snapshot is not None:
        trainer_object.resume(str(resume_snapshot))


def _skillflow_runtime_values(
    *,
    config: ExactSkillFlowProtocolConfig,
    application_input: ProtocolV10ApplicationInput,
    exact: ProtocolV10AttemptInput,
) -> dict[str, object]:
    """Preserve the split model/tokenizer deployment in the upstream runtime."""
    values = config.runtime_values(
        base_model_path=application_input.backbone.base_model_path,
        output_directory=str((exact.attempt_root / "skillflow-runtime").resolve()),
        supervisor_api_base=exact.sglang.gateway.openai_base,
        supervisor_model="supervisor_theta",
    )
    values.update(
        {
            "executor_api_base": exact.sglang.gateway.openai_base,
            "executor_model": exact.sglang.gateway.base_model,
            "executor_request_timeout_seconds": (exact.sglang.rollout.request_timeout_seconds),
            "supervisor_adapter_prefix": f"{exact.sglang.adapter_namespace}-step-",
            "supervisor_request_timeout_seconds": (exact.sglang.rollout.request_timeout_seconds),
            "tokenizer_path": application_input.backbone.tokenizer_path,
            "distributed_micro_batch": 1,
            "distributed_minimum_micro_batch": 1,
        }
    )
    return values


def _method_builders(
    *,
    exact: ProtocolV10AttemptInput,
    application_input: ProtocolV10ApplicationInput,
    experiment: object,
    event_log: LiveAttemptEventLog,
    run_id: str,
    attempt_id: str,
    gateway: SGLangGateway,
) -> ProtocolV10MethodBuilders[object]:
    from skillev.experiments.protocol_v10_formal import ProtocolV10FormalExperimentSpec

    if not isinstance(experiment, ProtocolV10FormalExperimentSpec):
        raise TypeError("Protocol 10 method builders require the formal experiment")

    def full(inputs: ProtocolV10ApplicationInputs) -> SKILLEVApplication:
        identity = application_input.identity(method=inputs.method, experiment=experiment)
        terminal = application_input.terminal_components(
            run_id=run_id,
            attempt_id=attempt_id,
        )
        if exact.resume_snapshot is None:
            return SKILLEVApplication.build_formal(
                backbone_config=application_input.backbone,
                task_provider=inputs.task_provider_factory.fresh(),
                base_session_factory=inputs.base_session_factory,
                seed_documents=application_input.seed_documents,
                terminal_components=terminal,
                checkpoint_storage=application_input.checkpoint_storage,
                initial_checkpoint=application_input.initial_checkpoint,
                public_identity=identity,
                event_log=event_log,
                clock=_clock,
                runtime=inputs.runtime,
            )
        return SKILLEVApplication.resume_formal(
            snapshot_directory=exact.resume_snapshot,
            backbone_config=application_input.backbone,
            task_provider_factory=inputs.task_provider_factory,
            base_session_factory=inputs.base_session_factory,
            terminal_components=terminal,
            checkpoint_storage=application_input.checkpoint_storage,
            initial_checkpoint=application_input.initial_checkpoint,
            public_identity=identity,
            event_log=event_log,
            clock=_clock,
            runtime=inputs.runtime,
        )

    def skillflow_baseline(
        inputs: ProtocolV10ApplicationInputs,
    ) -> ExactSkillFlowApplication:
        contract = application_input.method_contract
        if not isinstance(contract, ExactSkillFlowApplicationContract):
            raise TypeError("exact SkillFlow builder requires its tagged input")
        projection = contract.official_configuration_projection
        if not isinstance(projection, dict):  # guarded by the tagged contract
            raise TypeError("exact SkillFlow configuration projection is unavailable")
        config = ExactSkillFlowProtocolConfig(
            values=projection,
            upstream_revision=contract.upstream_revision,
            parity_contract=contract.parity_contract,
        )
        runtime_values = _skillflow_runtime_values(
            config=config,
            application_input=application_input,
            exact=exact,
        )
        baseline_emitter = RuntimeEventEmitter(
            event_log,
            producer_id="skillev-exact-skillflow-baseline",
            clock=_clock,
        )

        def publish_committed_step(committed_step: int) -> None:
            baseline_emitter.emit(
                EventType.EXACT_SKILLFLOW_STEP_COMMITTED,
                {
                    "committed_optimizer_step": committed_step,
                    "parity_contract": contract.parity_contract,
                    "upstream_revision": contract.upstream_revision,
                },
            )

        trainer = build_protocol_v10_skillflow_trainer(
            runtime_values,
            episode_runner=ProtocolV10SkillFlowEpisodeRunner(
                inputs.base_session_factory,
                gateway,
            ),
            checkpoint_callback=publish_committed_step,
        )

        def bind_distributed_gradient(bound: Any) -> None:
            _bind_skillflow_distributed_gradient(
                bound,
                resume_snapshot=exact.resume_snapshot,
            )

        return build_exact_skillflow_protocol_v10_application(
            trainer=trainer,
            ordered_tasks=tuple(
                ExactSkillFlowTaskAdapter(task, position)
                for position, task in enumerate(inputs.training_mix.tasks)
            ),
            config=config,
            after_setup=bind_distributed_gradient,
            emitter=baseline_emitter,
        )

    def no_calibration(inputs: ProtocolV10ApplicationInputs) -> FlowOnlyApplication:
        identity = application_input.identity(method=inputs.method, experiment=experiment)
        terminal = application_input.terminal_components(
            run_id=run_id,
            attempt_id=attempt_id,
        )
        restored: FlowOnlyRuntimeExecutionState | None = None
        provider: TaskProvider = inputs.task_provider_factory.fresh()
        run_progress = AttemptRunProgress.from_state(
            experiment.run_plan,
            identity.initial_run_cursor,
        )
        if exact.resume_snapshot is not None:
            metadata = FilesystemTrainingCheckpointStore(
                root=application_input.checkpoint_storage.directory
            ).load_metadata(exact.resume_snapshot)
            if metadata.identity != identity.runtime_snapshot_identity():
                raise ValueError("no-calibration snapshot identity differs from the attempt")
            if not isinstance(metadata.execution_state, FlowOnlyRuntimeExecutionState):
                raise TypeError("no-calibration resume requires flow-only runtime state")
            restored = metadata.execution_state
            provider = inputs.task_provider_factory.from_exact_state(restored.task_cursor)
            run_progress = AttemptRunProgress.from_state(
                experiment.run_plan,
                restored.run_cursor,
            )
        application = build_no_bayesian_calibration_application(
            FlowOnlyArmApplicationInputs(
                backbone_config=application_input.backbone,
                task_provider=provider,
                base_session_factory=inputs.base_session_factory,
                seed_documents=application_input.seed_documents,
                terminal_components=terminal,
                checkpoint_storage=application_input.checkpoint_storage,
                initial_checkpoint=application_input.initial_checkpoint,
                public_identity=identity,
                run_progress=run_progress,
                snapshot_identity=identity.runtime_snapshot_identity(),
                event_log=event_log,
                clock=_clock,
                rollout_workflow=inputs.runtime.workflow_resources.binding,
                arm_event_log=None,
                generator_factory=inputs.runtime.rollout_generator_factory,
                gradient_preparer=inputs.runtime.gradient_preparer,
                workflow_resources=inputs.runtime.workflow_resources,
                step_adapter_publisher_factory=(inputs.runtime.step_adapter_publisher_factory),
                skill_author_factory=inputs.runtime.skill_author_factory,
                resume_snapshot_directory=exact.resume_snapshot,
                restored_execution_state=restored,
            )
        )
        inputs.runtime.require_flow_only_bound(application)
        if exact.resume_snapshot is not None:
            resume_snapshot = exact.resume_snapshot
            journal = application.evolution_loop.step_transaction_journal
            if journal is None:
                raise RuntimeError("no-calibration resume has no step transaction journal")
            pending = journal.pending()
            if len(pending) > 1:
                raise RuntimeError("no-calibration resume found multiple unfinished steps")
            if pending:
                record = pending[0]
                if record.state in {
                    StepTransactionState.PREPARED,
                    StepTransactionState.OPTIMIZER_APPLIED,
                    StepTransactionState.PROJECTION_INSTALLED,
                    StepTransactionState.EVOLUTION_RESOLVED,
                }:
                    journal.rollback_before_checkpoint(record)
                else:
                    if record.checkpoint_name != resume_snapshot.name:
                        raise RuntimeError(
                            "no-calibration resume checkpoint differs from transaction"
                        )

                    def require_checkpoint(record: StepTransactionRecord) -> None:
                        del record
                        if not resume_snapshot.is_dir():
                            raise RuntimeError("no-calibration checkpoint is unavailable")

                    def ensure_adapter(record: StepTransactionRecord) -> str:
                        del record
                        generation = application.evolution_loop.publish_restored_adapter()
                        if generation is None:
                            raise RuntimeError("no-calibration adapter publisher is absent")
                        return generation.adapter_revision

                    StepTransactionReconciler(
                        journal=journal,
                        emitter=application.emitter,
                        require_checkpoint=require_checkpoint,
                        ensure_adapter=ensure_adapter,
                    ).reconcile(record)
        return application

    return ProtocolV10MethodBuilders(
        skillflow_baseline=skillflow_baseline,
        bayesian_improve_full=full,
        bayesian_improve_no_calibration=no_calibration,
    )


async def _run_coordinator(
    exact: ProtocolV10AttemptInput,
    topology: DistributedTTBTopology,
) -> None:
    _bind_skillflow_authoring_service(exact)
    protocol = load_active_protocol_v10(exact.protocol_config)
    experiment = load_protocol_v10_formal_experiment(exact.experiment_config)
    loaded = load_protocol_v10_materialized_catalog(protocol, exact.catalog_root)
    application_input = ProtocolV10ApplicationInput.read(exact.application_input)
    application_input.require_method(exact.method)
    coordinator = DistributedTTBGradientCoordinator(topology)
    runtime = BoundFormalSGLangRuntime.build(
        binding=exact.sglang,
        gradient_preparer=coordinator,
    )
    deployments = ProtocolV10SessionDeployments.read(exact.session_deployments)
    sessions = deployments.build_registry(loaded, runtime.resources, runtime.gateway)
    exact.attempt_root.mkdir(parents=True, exist_ok=True)
    run_id = exact.attempt_root.parent.name
    attempt_id = exact.attempt_root.name
    event_path = exact.attempt_root / "events.jsonl"
    event_log = (
        LiveAttemptEventLog.resume(event_path, run_id=run_id, attempt_id=attempt_id)
        if exact.resume_snapshot is not None
        else LiveAttemptEventLog(event_path, run_id=run_id, attempt_id=attempt_id)
    )
    methods = _method_builders(
        exact=exact,
        application_input=application_input,
        experiment=experiment,
        event_log=event_log,
        run_id=run_id,
        attempt_id=attempt_id,
        gateway=runtime.gateway,
    )
    builder = ProtocolV10AttemptBuilder(
        experiment=experiment,
        catalog=loaded.catalog,
        selection=loaded.selection,
        sessions=sessions,
        runtime=runtime.dependencies(),
        workflow_binding=exact.sglang.workflow,
        methods=methods,
    )
    try:
        built = (
            builder.build(exact.method)
            if exact.admission_mode is ProtocolV10AdmissionMode.FORMAL
            else ProtocolV10IntegrationSmokeBuilder(builder).build(exact.method)
        )
        application = built.application
        if not isinstance(
            application,
            SKILLEVApplication | FlowOnlyApplication | ExactSkillFlowApplication,
        ):
            raise TypeError("Protocol 10 method returned an incompatible application")
        if isinstance(application, ExactSkillFlowApplication):
            final_optimizer_step = (
                await application.run(
                    experiment.run_plan,
                    maximum_steps_this_attempt=exact.integration_smoke_steps,
                )
            ).final_optimizer_step
        else:
            summary = await application.evolution_loop.run(
                experiment.run_plan,
                maximum_steps_this_attempt=exact.integration_smoke_steps,
            )
            application.training_loop.ledger.assert_fully_settled()
            final_optimizer_step = summary.final_optimizer_step
        print(
            json.dumps(
                {
                    "final_optimizer_step": final_optimizer_step,
                    "method": exact.method.value,
                    "status": (
                        "passed"
                        if exact.admission_mode is ProtocolV10AdmissionMode.FORMAL
                        else "integration-smoke-passed"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if exact.method is FormalMethodV10.SKILLFLOW_BASELINE:
            application = locals().get("application")
            if isinstance(application, ExactSkillFlowApplication):
                client = getattr(application.trainer, "distributed_gradient_client", None)
                if client is not None:
                    client.close()
        else:
            coordinator.close()


def _run_gradient_worker(
    exact: ProtocolV10AttemptInput,
    topology: DistributedTTBTopology,
) -> None:
    application_input = ProtocolV10ApplicationInput.read(exact.application_input)
    application_input.require_method(exact.method)
    if exact.method is FormalMethodV10.SKILLFLOW_BASELINE:
        contract = application_input.method_contract
        if not isinstance(contract, ExactSkillFlowApplicationContract):
            raise TypeError("SkillFlow worker requires the exact input tag")
        projection = contract.official_configuration_projection
        if not isinstance(projection, dict):
            raise TypeError("SkillFlow worker configuration is unavailable")
        worker_config: dict[str, object] = {
            **projection,
            "base_model": application_input.backbone.base_model_path,
        }
        from training.distributed_gradient import worker_loop

        worker_loop(worker_config)
        return
    backbone = build_qwen_policy_backbone(application_input.backbone)
    backbone.load_checkpoint(application_input.initial_checkpoint.directory)
    backbone.bind_initial_trainable_state(application_input.initial_checkpoint.trainable_state)
    serve_distributed_ttb_worker(topology=topology, backbone=backbone)


def main(path: str) -> None:
    exact_path = Path(path)
    if not exact_path.is_absolute():
        raise ValueError("Protocol 10 attempt input route must be absolute")
    exact = ProtocolV10AttemptInput.read(exact_path)
    environment_method = os.environ.get("SKILLEV_PROTOCOL_V10_METHOD")
    if environment_method != exact.method.value:
        raise ValueError("Protocol 10 method environment differs from exact input")
    topology = _initialize_formal_topology()
    try:
        if topology.rank == 0:
            asyncio.run(_run_coordinator(exact, topology))
        else:
            _run_gradient_worker(exact, topology)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m skillev_private.experiments.protocol_v10_attempt_worker INPUT.json"
        )
    main(sys.argv[1])


__all__ = ["ProtocolV10MethodUnavailableError", "main"]
