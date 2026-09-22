"""Formal single-owner BayesianImprove training, from one declared run configuration.

Run with two torchrun ranks and a separate SGLang GPU. The existing application
owns TTB, posterior, evolution, publication and recovery; this entrypoint only
binds private data/deployment and the explicit seven-domain condition. No debug
step cap, admission receipts, fallback model or implicit training deadline.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch.distributed as dist

from skillev.application import SKILLEVApplication, TerminalComponents
from skillev.application_continuation import (
    ActionWireContinuation,
    HorizonContinuation,
    ReasoningContinuation,
    SkillColdStartContinuation,
    SkillExposureContinuation,
    TokenBudgetNoticeContinuation,
)
from skillev.application_reporting import resolved_method_state
from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.domain_subset_continuation import DomainSubsetContinuation
from skillev.evolution import PhiBudgetAuthority
from skillev.experiments._evolution_preflight_seed import planned_seed_documents
from skillev.format_review_continuation import FormatReviewContinuation
from skillev.healthbench_judge_continuation import HealthBenchJudgeContinuation
from skillev.policy import build_qwen_policy_backbone as build_qwen_policy_backbone
from skillev.runtime import BudgetLedger, LiveAttemptEventLog, SkillLibraryState
from skillev.runtime.formal_sglang_runtime import (
    BoundFormalSGLangRuntime,
    FormalSGLangRuntimeBinding,
)
from skillev.runtime.service_topology import InferenceService, ServiceTopology
from skillev.runtime.sglang_gateway import SGLangGatewayConfig
from skillev.training import FixedAttemptBudgetPlan, PrivateCheckpointStorageBinding
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore
from skillev.training.distributed_ttb import (
    DistributedTTBGradientCoordinator,
    DistributedTTBTopology,
)
from skillev.training.distributed_ttb import (
    initialize_distributed_ttb as initialize_distributed_ttb,
)
from skillev.training.distributed_ttb import (
    serve_distributed_ttb_worker as serve_distributed_ttb_worker,
)
from skillev.training.performance_config import TrainingPerformanceConfig
from skillev.training.quality_monitor import (
    QualityCheckpointStop,
    QualityProbeCoordinator,
    load_quality_policy,
)
from skillev.training.run_clock import RunWallClock
from skillev.training.run_condition import EffectiveRunCondition
from skillev.training.stopping import StopAfterCheckpoint, TrainingPausedError
from skillev_private.benchmarks.mbpp_scoring import MBPPScorerProfile, resolve_mbpp_profile
from skillev_private.benchmarks.protocol_v13_seven_training import (
    load_seven_domain_training_sources,
    seven_domain_training_trajectories,
)
from skillev_private.benchmarks.protocol_v13_training_sessions import (
    build_protocol13_training_sessions,
    native_scorer_contracts,
)

from .bayesian_condition_transition import (
    _bind_run_directory as _bind_run_directory,
)
from .bayesian_condition_transition import (
    condition_configs,
    observer_batch_size_starts,
    observer_condition_starts,
    sampling_condition,
    save_condition_transition,
)
from .bayesian_serving import register_serving
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
from .initial_quality_probe import import_initial_quality_probe
from .protocol_v10_attempt_builder import (
    ProtocolV10OrderedTaskProvider,
    ProtocolV10TaskProviderFactory,
)
from .quality_collection import FormalQualityCollector, QualityCollectionBinding
from .training_data_condition import (
    data_condition_scientific,
    inline_data_condition,
    require_same_run_data_condition,
)
from .training_domain_schedule import schedule_summary, training_schedule

# Retain the original entry module's CLI dependency/monkeypatch surface.
__all__ = [
    "BayesianFormalConfig",
    "TrainingPerformanceConfig",
    "_read_preparation",
    "dist",
    "load_seven_domain_training_sources",
    "seven_domain_training_trajectories",
]


def _write(path: Path, value: object) -> None:
    path.write_text(canonical_json(normalize_json(value)) + "\n", encoding="utf-8")


def _emit(status: str, **fields: JsonValue) -> None:
    print(canonical_json({"time": _clock(), "status": status, **fields}), flush=True)


@dataclass(frozen=True, slots=True)
class FormalTrainingBindings:
    """Private paths and explicitly assigned physical devices, never public config."""

    preparation: Path
    dataset: Path
    deployments: Path
    evalplus_python: Path
    evalplus_source_root: Path
    endpoint: str
    base_model: str
    adapter_namespace: str
    training_gpu_uuids: tuple[str, ...]
    serving_gpu_uuid: str
    topology: ServiceTopology | None = None
    data_condition: dict[str, JsonValue] | None = None
    evidence_mirror_root: Path | None = None
    evidence_max_pending_steps: int = 2
    evidence_minimum_free_bytes: int = 1_073_741_824

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_condition", inline_data_condition(self.data_condition))
        if self.evidence_max_pending_steps < 1 or self.evidence_minimum_free_bytes < 0:
            raise ValueError("declare positive mirror backlog and nonnegative storage headroom")

    @property
    def roles(self) -> ServiceTopology:
        if self.topology is not None:
            return self.topology
        return ServiceTopology(
            (InferenceService("shared", self.endpoint, self.serving_gpu_uuid),),
            ("shared",),
            ("shared",),
            ("shared",),
            self.training_gpu_uuids,
        )

    def require_device_mapping(self, visible: str, world_size: int) -> None:
        self.roles.require_device_mapping(visible, world_size)
        if self.topology is not None and (
            self.training_gpu_uuids != self.topology.gradient_workers
            or self.endpoint != self.topology.members("actor")[0].endpoint
            or self.serving_gpu_uuid != self.topology.members("actor")[0].gpu_uuid
        ):
            raise ValueError("legacy aliases disagree with explicit role topology")

    @classmethod
    def load(cls, path: Path) -> FormalTrainingBindings:
        value = json.loads(path.read_text(encoding="utf-8"))
        topology = ServiceTopology.from_value(value["topology"]) if "topology" in value else None
        actor = topology.members("actor")[0] if topology is not None else None
        return cls(
            topology=topology,
            data_condition=inline_data_condition(value.get("data_condition")),
            evidence_mirror_root=Path(value["evidence_mirror_root"])
            if value.get("evidence_mirror_root") is not None
            else None,
            evidence_max_pending_steps=value.get("evidence_max_pending_steps", 2),
            evidence_minimum_free_bytes=value.get("evidence_minimum_free_bytes", 1_073_741_824),
            preparation=Path(value["preparation"]),
            dataset=Path(value["dataset"]),
            deployments=Path(value["deployments"]),
            evalplus_python=Path(value["evalplus_python"]),
            evalplus_source_root=Path(value["evalplus_source_root"]),
            endpoint=value["endpoint"] if actor is None else actor.endpoint,
            base_model=value["base_model"],
            adapter_namespace=value["adapter_namespace"],
            training_gpu_uuids=tuple(value["training_gpu_uuids"])
            if topology is None
            else topology.gradient_workers,
            serving_gpu_uuid=value["serving_gpu_uuid"] if actor is None else actor.gpu_uuid,
        )

    def runtime(self, root: Path, profile: TrainingPerformanceConfig) -> FormalSGLangRuntimeBinding:
        from .formal_episode_config import formal_actor_transport

        binding = FormalSGLangRuntimeBinding(
            gateway=SGLangGatewayConfig(
                endpoint_base=self.endpoint,
                base_model=self.base_model,
                supervisor_adapter=f"{self.adapter_namespace}-supervisor",
                seed=0,
                temperature=0.0,
                top_p=1.0,
                max_output_tokens=4096,
                request_timeout_seconds=600.0,
            ),
            rollout=formal_actor_transport(self.endpoint, worker_threads=profile.transport_threads),
            workflow=profile.workflow(),
            adapter_export_root=root / "adapters",
            adapter_namespace=self.adapter_namespace,
            adapter_keep_recent=3,
            performance=profile,
            request_journal_path=root / "requests.sqlite3",
        )
        if self.topology is None:
            return binding
        return replace(
            binding,
            actor_replicas=tuple(
                replace(binding.gateway, endpoint_base=v.endpoint)
                for v in self.roles.members("actor")
            ),
            author_replicas=tuple(
                replace(binding.gateway, endpoint_base=v.endpoint)
                for v in self.roles.members("author")
            ),
        )


def require_formal_execution(profile: TrainingPerformanceConfig) -> None:
    if not profile.coordinator_participates or profile.pipeline_mode != "within-step":
        raise ValueError("both gradient ranks must participate with within-step rollout overlap")


async def _stop_at_committed_boundary(
    request: StopAfterCheckpoint,
    probes: QualityProbeCoordinator | None,
    *,
    optimizer_step: int,
    policy_snapshot_id: str,
    pause_at_step: int | None,
    library_snapshot_id: str | None = None,
) -> bool:
    """A scheduled handoff still runs the due quality probe before saving."""
    if request():
        return True
    quality_pause = probes is not None and await probes.check(
        policy_step=optimizer_step,
        policy_snapshot_id=policy_snapshot_id,
        **({"library_snapshot_id": library_snapshot_id} if library_snapshot_id is not None else {}),
    )
    # A requested stop can arrive while the fixed quality collection is awaited.
    # Finish that collection, then honor it before sampling another full batch.
    return (
        request()
        or quality_pause
        or (pause_at_step is not None and optimizer_step >= pause_at_step)
    )


async def run_coordinator(
    *,
    config: BayesianFormalConfig,
    bindings: FormalTrainingBindings,
    profile: TrainingPerformanceConfig,
    root: Path,
    resume: Path | None,
    topology: DistributedTTBTopology,
    allow_new_horizons: bool = False,
    allow_new_reasoning: bool = False,
    allow_new_action_wire: bool = False,
    allow_token_budget_notice: bool = False,
    allow_catalog_read: bool = False,
    allow_domain_subset: bool = False,
    allow_skill_cold_start: bool = False,
    allow_healthbench_judge: bool = False,
    allow_format_review: bool = False,
    quality_policy: Path | None = None,
    quality_panel: Path | None = None,
    initial_quality_probe: Path | None = None,
    pause_at_step: int | None = None,
    iid_baselines: Path | None = None,
    observation_steps: int | None = None,
    observation_sources: Path | None = None,
    unqualified_training_sources: Path | None = None,
) -> None:
    from .autonomous_ttb import (
        autonomous_training_sources,
        require_autonomous_initialization,
        source_coverage_report,
    )
    from .training_observation import (
        observation_condition,
        observation_config,
        require_observation_sources,
        require_training_sources,
        unqualified_training_condition,
    )
    from .warmup_initialization import initialization_condition, require_initialization_candidate

    original_config = config
    config = observation_config(
        config,
        steps=observation_steps,
        resume=resume,
        sources=observation_sources,
        continuation=any(
            (
                allow_new_horizons,
                allow_new_reasoning,
                allow_new_action_wire,
                allow_token_budget_notice,
                allow_catalog_read,
                allow_domain_subset,
                allow_skill_cold_start,
                allow_healthbench_judge,
                allow_format_review,
            )
        ),
    )
    observation = observation_condition(observation_steps)
    unqualified = unqualified_training_condition(
        config,
        sources=unqualified_training_sources,
        root=root,
        resume=resume,
        iid_baselines=iid_baselines,
        observation_steps=observation_steps,
        observation_sources=observation_sources,
        proactive_catalog_continuation=allow_catalog_read,
        domain_subset_continuation=allow_domain_subset,
        healthbench_judge_continuation=allow_healthbench_judge,
        format_review_continuation=allow_format_review,
        continuation=any(
            (
                allow_new_horizons,
                allow_new_reasoning,
                allow_new_action_wire,
                allow_token_budget_notice,
                allow_catalog_read,
                allow_domain_subset,
                allow_skill_cold_start,
                allow_healthbench_judge,
                allow_format_review,
            )
        ),
    )
    unqualified_records = None
    if unqualified:
        assert unqualified_training_sources is not None
        unqualified_records = autonomous_training_sources(
            config, load_seven_domain_training_sources(bindings.dataset), bindings.data_condition
        )
        authorized_schedule = training_schedule(
            config, unqualified_records, root=root, resume=resume
        )
        require_training_sources(
            unqualified_training_sources,
            authorized_schedule,
            expected_trajectories=len(authorized_schedule),
        )
    if observation or unqualified:
        require_formal_execution(profile)
        bindings.require_device_mapping(
            os.environ.get("CUDA_VISIBLE_DEVICES", ""), topology.world_size
        )
        if topology.rank != 0:
            raise ValueError("observation coordinator must be the participating rank zero")
        if iid_baselines is not None:
            raise ValueError("observation does not import or claim IID admission")
    if pause_at_step is not None and (
        type(pause_at_step) is not int or not 1 <= pause_at_step < config.steps
    ):
        raise ValueError("scheduled pause must be an intermediate committed optimizer step")
    if (quality_policy is None) != (quality_panel is None):
        raise ValueError("declare both quality policy and isolated panel collection binding")
    if initial_quality_probe is not None and quality_policy is None:
        raise ValueError("initial quality import requires a quality policy and panel")
    if config.format.endswith("@6") and bindings.evidence_mirror_root is None:
        raise ValueError("fresh architecture runs require an explicit persistent evidence mirror")
    fresh_reference = None
    if config.format.endswith("@6"):
        from .fresh_restart import require_run_iid_baselines

        if bindings.data_condition is None:
            raise ValueError("fresh architecture runs require explicit training sources")
        if not observation and not unqualified:
            fresh_reference = require_run_iid_baselines(
                requested=iid_baselines, root=root, config=config, resuming=resume is not None
            )
    started = time.monotonic()
    source_condition = _bind_run_directory(
        root,
        config,
        resume,
        allow_new_horizons=allow_new_horizons,
        allow_new_reasoning=allow_new_reasoning,
        allow_new_action_wire=allow_new_action_wire,
        allow_token_budget_notice=allow_token_budget_notice,
        allow_catalog_read=allow_catalog_read,
        allow_domain_subset=allow_domain_subset,
        allow_skill_cold_start=allow_skill_cold_start,
        allow_healthbench_judge=allow_healthbench_judge,
        allow_format_review=allow_format_review,
    )
    if observation:
        _write(root / "observation-candidate-config.json", original_config.to_value())
        _write(root / "observation-condition.json", observation)
    if unqualified and resume is None:
        assert unqualified_training_sources is not None
        _write(root / "unqualified-training-condition.json", unqualified)
        (root / "unqualified-training-sources-private.json").write_bytes(
            unqualified_training_sources.read_bytes()
        )
    if fresh_reference is not None and resume is None:
        assert iid_baselines is not None
        # Only result-directory references; no IID tasks, answers or trajectories.
        _write(root / "iid-baselines-private.json", json.loads(iid_baselines.read_text()))
    quality = (
        QualityCheckpointStop(root / "quality", load_quality_policy(quality_policy))
        if quality_policy is not None
        else None
    )
    if quality is None and (root / "quality" / "policy.json").exists():
        raise ValueError("resume cannot disable this run's quality policy")
    if fresh_reference is not None and (
        quality is None
        or quality.policy.baseline_step != 0
        or quality.policy.initial_admission_override is not None
        or quality.policy.architecture_id != fresh_reference["architecture_id"]
    ):
        raise ValueError(
            "fresh restart requires its own architecture-bound quality baseline "
            "without an old override"
        )
    run_clock = RunWallClock(root)
    backbone_config, checkpoint = _read_preparation(bindings.preparation)
    require_autonomous_initialization(config, bindings.preparation)
    config.require_backbone(backbone_config)
    initialization = {}
    if config.format.endswith("@6"):
        require_initialization_candidate(
            bindings.preparation, sampling=config.sampling_config.to_value()
        )
        initialization = initialization_condition(bindings.preparation)
    interpreter = _validated_worker_interpreter(bindings.evalplus_python)
    mbpp_profile = resolve_mbpp_profile(
        {
            "source_root": str(bindings.evalplus_source_root),
            "profile": MBPPScorerProfile().to_value(),
        }
    )
    coordinator = DistributedTTBGradientCoordinator(
        topology,
        coordinator_participates=True,
        pipeline_mode="within-step",
    )
    runtime = BoundFormalSGLangRuntime.build(
        binding=bindings.runtime(root, profile),
        gradient_preparer=coordinator,
    )
    from .training_controller import TrainingController

    evidence = None
    controller = TrainingController(
        root,
        resource_roles=bindings.roles.to_value(),
        gpu_uuids=tuple(v.gpu_uuid for v in bindings.roles.services) + bindings.training_gpu_uuids,
        evidence=evidence,
    )
    stop = thread = None
    try:
        actual_services = await asyncio.to_thread(
            register_serving,
            runtime.gateway,
            bindings.roles,
            root=root,
            model_path=backbone_config.base_model_path,
            tokenizer_path=backbone_config.tokenizer_path or backbone_config.base_model_path,
            minimum_context=config.max_input_tokens
            + max(config.maximum_reasoning_tokens, config.max_action_tokens),
        )
        if fresh_reference is not None:
            from skillev.runtime.serving_profile import require_same_profile

            from .formal_episode_config import formal_actor_transport

            controls = cast(dict[str, JsonValue], fresh_reference["controls"])
            transport = formal_actor_transport(
                bindings.endpoint, worker_threads=profile.transport_threads
            ).to_value()
            transport.pop("endpoint_base")
            if transport != controls["actor_transport"]:
                raise ValueError(
                    "formal actor transport differs from the accepted IID architecture"
                )
            for service in bindings.roles.members("actor"):
                profile_value = actual_services[service.endpoint.rstrip("/").removesuffix("/v1")]
                require_same_profile(
                    cast(dict[str, JsonValue], controls["serving"]),
                    cast(dict[str, JsonValue], profile_value),
                )
        if bindings.topology is not None:
            for service in bindings.roles.services:
                runtime.resources.configure_model_endpoint(
                    service.endpoint,
                    capacity=service.request_capacity,
                    token_capacity=service.token_capacity,
                )
        _emit(
            "preparing-domain-sessions", domains=list(config.domains), batch_size=config.batch_size
        )
        records = (
            unqualified_records
            if unqualified_records is not None
            else autonomous_training_sources(
                config,
                load_seven_domain_training_sources(bindings.dataset),
                bindings.data_condition,
            )
        )
        quality_binding = None
        if quality is not None and quality_panel is not None:
            if quality.policy.condition_id != config.condition:
                raise ValueError("quality policy must name this training condition")
            quality_binding = QualityCollectionBinding.load(
                quality_panel, training=records, policy=quality.policy, batch_size=config.batch_size
            )
            quality_binding.freeze(quality.root)
        selected = training_schedule(config, records, root=root, resume=resume)
        if config.learning_protocol is not None:
            assert bindings.data_condition is not None
            _write(
                root / "source-coverage-private.json",
                source_coverage_report(bindings.data_condition),
            )
        if observation:
            assert observation_sources is not None
            require_observation_sources(observation_sources, selected)
            (root / "observation-sources-private.json").write_bytes(
                observation_sources.read_bytes()
            )
        if fresh_reference is not None:
            from .fresh_restart import require_restart_inputs

            require_restart_inputs(
                fresh_reference,
                backbone=backbone_config,
                library=SkillLibraryState.from_seed_documents(
                    planned_seed_documents(config.initial_skill_profile)
                ),
                scorer_contracts=native_scorer_contracts(
                    mbpp_profile, config.healthbench_judge, domains=config.domains
                ),
                training_sources=frozenset(
                    (r.episode.benchmark.value, r.episode.source_id) for r in selected
                ),
            )
        tasks, sessions = await build_protocol13_training_sessions(
            selected,
            deployments_path=bindings.deployments,
            endpoint_base=bindings.roles.members("judge")[0].endpoint,
            judge_endpoints=tuple(v.endpoint for v in bindings.roles.members("judge")),
            request_journal_path=root / "requests.sqlite3",
            base_model=bindings.base_model,
            resources=runtime.resources,
            mbpp_interpreter=interpreter,
            mbpp_source_root=bindings.evalplus_source_root,
            mbpp_profile=mbpp_profile,
            hotpot_deliberation=config.hotpot_deliberation,
            rollout_budget=config.task_budget,
            static_rollout_budget=config.static_task_budget,
            domain_rollout_budgets=config.domain_task_budgets,
            lazy_environments=True,
            healthbench_judge=config.healthbench_judge,
            format_review_from_step=config.format_review_from_step,
        )
        hydration_seconds = time.monotonic() - started
        application_config = config.application_config(root.name)
        plan = config.run_plan
        curriculum = sampling_condition(
            root, config.condition if source_condition is None else source_condition.condition
        )
        identity = _public_identity(
            application=application_config,
            run_plan=plan,
            task_ids=tuple(task.task_id for task in tasks),
            checkpoint=checkpoint,
            mbpp_profile=mbpp_profile,
            sampling_condition=curriculum,
            entry_kind="formal",
            healthbench_judge=config.healthbench_judge,
            format_review_from_step=config.format_review_from_step,
            initial_skill_profile=config.initial_skill_profile,
            alfworld_goal_binding=sessions.alfworld_goal_binding,
        )
        # A readable projection of the SAME snapshot/launch inputs, not another
        # state counter or an assertion that initial LoRA equals the base model.
        effective = EffectiveRunCondition.create(
            condition_id=config.condition,
            scientific={
                **data_condition_scientific(bindings.data_condition, selected),
                **observation,
                **unqualified,
                **initialization,
                "formal": {
                    key: value
                    for key, value in config.to_value().items()
                    if key
                    not in {
                        "performance_profile",
                        "planning_hours",
                        "target_steps_per_hour",
                        "checkpoint_every",
                    }
                },
                "batch_size": config.batch_size,
                "maximum_h0_tokens": application_config.maximum_h0_tokens,
                "rollout": application_config.trainer.rollout.to_value(),
                "evolution": application_config.evolution.to_value(),
                "initial_partition": checkpoint.trainable_state.to_value(),
                "tokenizer_id": backbone_config.tokenizer_id,
                "base_revision": backbone_config.revision,
                "model_artifact": (
                    backbone_config.base_model_artifact.to_value()
                    if backbone_config.base_model_artifact is not None
                    else None
                ),
                "mbpp_scorer": mbpp_profile.to_value(),
            },
            execution={
                "performance": profile.to_value(),
                "roles": bindings.roles.to_value(),
                "checkpoint_every": config.checkpoint_every,
                "planning_hours": config.planning_hours,
                "target_steps_per_hour": config.target_steps_per_hour,
                **(
                    {"optimizer_transition_observation": "actual-trainable-adam@1"}
                    if config.format.endswith("@6")
                    else {}
                ),
                **({"pause_at_optimizer_step": pause_at_step} if pause_at_step is not None else {}),
            },
        )
        if config.format.endswith("@6"):
            from .fresh_restart import resolve_effective_run_condition, resolved_input_profiles

            expanded = resolve_effective_run_condition(
                config=config,
                backbone=backbone_config,
                initial_library=SkillLibraryState.from_seed_documents(
                    planned_seed_documents(config.initial_skill_profile)
                ),
                data_condition=cast(
                    dict[str, JsonValue],
                    data_condition_scientific(bindings.data_condition, selected)["data_condition"],
                ),
                input_profiles=resolved_input_profiles(tasks),
                scorer_contracts=native_scorer_contracts(
                    mbpp_profile, config.healthbench_judge, domains=config.domains
                ),
                execution=effective.execution,
                condition_id=config.condition,
            )
            effective = EffectiveRunCondition.create(
                condition_id=config.condition,
                scientific={
                    **expanded.scientific,
                    **observation,
                    **unqualified,
                    **initialization,
                    "initial_partition": checkpoint.trainable_state.to_value(),
                },
                execution=expanded.execution,
            )
        if resume is not None:
            history = condition_configs(root)
            declared_data = None
            if any(value.domains != config.domains for value in history.values()):
                declared_data = {
                    value.condition: data_condition_scientific(
                        bindings.data_condition, training_schedule(value, records, root=root)
                    ).get("data_condition")
                    for value in history.values()
                }
                declared_data[config.condition] = effective.scientific.get("data_condition")
            require_same_run_data_condition(
                root, effective, declared_data_by_condition=declared_data
            )
        _write(
            root / f"effective-condition-process-{os.getpid()}.json",
            {
                **effective.to_value(),
                "source_snapshot_identity": identity.snapshot_identity.to_value(),
                "initial_adapter_base_equivalence": "requires-I0-model-comparison",
            },
        )
        phi = _phi_per_cycle(application_config, config.phi_calls_per_cycle)
        cap = FixedAttemptBudgetPlan.from_trainer_and_run_plan(
            trainer=application_config.trainer,
            run_plan=plan,
            phi_per_cycle_maximum=phi,
        ).required()
        attempt_id = "formal-seed0"
        terminal = TerminalComponents(
            ledger=BudgetLedger(run_id=root.name, attempt_id=attempt_id, cap=cap),
            authoring_authority=_authoring_authority(tasks),
            phi_budget=PhiBudgetAuthority(phi),
        )
        event_factory = LiveAttemptEventLog if resume is None else LiveAttemptEventLog.resume
        event_log = event_factory(root / "events.jsonl", run_id=root.name, attempt_id=attempt_id)
        storage = PrivateCheckpointStorageBinding(directory=str(root / "checkpoints"))
        _write(
            root / "resolved-run-plan.json",
            {
                "mode": "five-step-training-observation"
                if observation
                else "owner-authorized-unqualified-training"
                if unqualified
                else "formal-bayesian-improve-single-owner",
                **observation,
                **unqualified,
                "schedule": schedule_summary(config, selected),
                "application": application_config.to_value(),
                "run_plan": plan.to_value(),
                "terminal_evaluation_conditions": json.loads(
                    sessions.terminal_evaluation_conditions_json
                ),
                "task_feature_mapping_version": sessions.task_feature_mapping_version,
                "phi_per_cycle_maximum": phi.to_value(),
                "total_attempt_budget": cap.to_value(),
                "cadence_steps": list(
                    range(config.checkpoint_every, config.steps + 1, config.checkpoint_every)
                ),
                "recovery_snapshots": "every-transaction-rolling-three-plus-cadence-phase-final",
                "performance": profile.to_value(),
                "training_gpu_uuids": list(bindings.training_gpu_uuids),
                "serving_gpu_uuid": bindings.serving_gpu_uuid,
                "role_topology": bindings.roles.to_value(),
                "planning_hours_not_hard_deadline": config.planning_hours,
            },
        )
        _emit("building-formal-application", resume=resume is not None)
        model_started = time.monotonic()
        if resume is None:
            application = SKILLEVApplication.build_formal(
                backbone_config=backbone_config,
                task_provider=ProtocolV10OrderedTaskProvider(tasks, curriculum),
                base_session_factory=sessions,
                seed_documents=planned_seed_documents(config.initial_skill_profile),
                terminal_components=terminal,
                checkpoint_storage=storage,
                initial_checkpoint=checkpoint,
                public_identity=identity,
                event_log=event_log,
                clock=_clock,
                runtime=runtime.dependencies(),
            )
            if fresh_reference is not None or observation or unqualified or initialization:
                from .fresh_restart import require_clean_initial_application

                require_clean_initial_application(
                    application,
                    preparation=bindings.preparation,
                    checkpoint_directory=Path(checkpoint.directory),
                    root=root,
                    initial_skill_profile=config.initial_skill_profile,
                )
        else:
            continuation: (
                FormatReviewContinuation
                | HealthBenchJudgeContinuation
                | SkillColdStartContinuation
                | SkillExposureContinuation
                | TokenBudgetNoticeContinuation
                | ActionWireContinuation
                | ReasoningContinuation
                | HorizonContinuation
                | DomainSubsetContinuation
                | None
            ) = None
            if source_condition is not None:
                previous_tasks = tuple(
                    r.input.task_id for r in training_schedule(source_condition, records, root=root)
                )
                old_identity = _public_identity(
                    application=source_condition.application_config(root.name),
                    run_plan=plan,
                    task_ids=previous_tasks,
                    checkpoint=checkpoint,
                    mbpp_profile=mbpp_profile,
                    sampling_condition=curriculum,
                    entry_kind="formal",
                    healthbench_judge=source_condition.healthbench_judge,
                    format_review_from_step=source_condition.format_review_from_step,
                    initial_skill_profile=source_condition.initial_skill_profile,
                    alfworld_goal_binding=sessions.alfworld_goal_binding,
                )
                saved = FilesystemTrainingCheckpointStore(root=root / "checkpoints").load_metadata(
                    resume
                )
                continuation = (
                    FormatReviewContinuation
                    if allow_format_review
                    else HealthBenchJudgeContinuation
                    if allow_healthbench_judge
                    else SkillColdStartContinuation
                    if allow_skill_cold_start
                    else SkillExposureContinuation
                    if allow_catalog_read
                    else TokenBudgetNoticeContinuation
                    if allow_token_budget_notice
                    else ActionWireContinuation
                    if allow_new_action_wire
                    else ReasoningContinuation
                    if allow_new_reasoning
                    else HorizonContinuation
                )(old_identity, saved.optimizer_step)
                if allow_domain_subset:
                    continuation = DomainSubsetContinuation(
                        old_identity,
                        saved.optimizer_step,
                        saved.execution_state.task_cursor.cursor,
                        previous_tasks,
                        tuple(task.task_id for task in tasks),
                    )
            application = SKILLEVApplication.resume_formal(
                snapshot_directory=resume,
                backbone_config=backbone_config,
                task_provider_factory=ProtocolV10TaskProviderFactory(tasks, curriculum),
                base_session_factory=sessions,
                terminal_components=terminal,
                checkpoint_storage=storage,
                initial_checkpoint=checkpoint,
                public_identity=identity,
                event_log=event_log,
                clock=_clock,
                runtime=runtime.dependencies(),
                format_review_continuation=(
                    continuation if isinstance(continuation, FormatReviewContinuation) else None
                ),
                healthbench_judge_continuation=(
                    continuation if isinstance(continuation, HealthBenchJudgeContinuation) else None
                ),
                horizon_continuation=(
                    continuation if isinstance(continuation, HorizonContinuation) else None
                ),
                reasoning_continuation=(
                    continuation if isinstance(continuation, ReasoningContinuation) else None
                ),
                token_budget_continuation=(
                    continuation
                    if isinstance(continuation, TokenBudgetNoticeContinuation)
                    else None
                ),
                domain_subset_continuation=(
                    continuation if isinstance(continuation, DomainSubsetContinuation) else None
                ),
                skill_exposure_continuation=(
                    continuation if isinstance(continuation, SkillExposureContinuation) else None
                ),
                action_wire_continuation=(
                    continuation if isinstance(continuation, ActionWireContinuation) else None
                ),
            )
            if source_condition is not None:
                save_condition_transition(
                    application,
                    root=root,
                    source_snapshot=resume,
                    source=source_condition,
                    target=config,
                    reasoning=allow_new_reasoning,
                    action_wire=allow_new_action_wire,
                    token_budget_notice=allow_token_budget_notice,
                    catalog_read=allow_catalog_read,
                    domain_subset=allow_domain_subset,
                    skill_cold_start=allow_skill_cold_start,
                    healthbench_judge=allow_healthbench_judge,
                    format_review=allow_format_review,
                )
        if bindings.evidence_mirror_root is not None:
            from skillev.training.run_observer import CommittedRunObserver

            evidence = CommittedRunObserver(
                root,
                run_id=root.name,
                condition_id=config.condition,
                mirror_root=bindings.evidence_mirror_root,
                condition_starts=observer_condition_starts(root, config),
                code_revision=os.environ.get("SKILLEV_CODE_REVISION"),
                expected_batch_size=config.batch_size,
                batch_size_starts=observer_batch_size_starts(root, config),
                max_pending_steps=bindings.evidence_max_pending_steps,
                minimum_free_bytes=bindings.evidence_minimum_free_bytes,
            )
        controller.evidence = evidence
        application.training_loop.configure_inflight(
            root / "inflight",
            condition={
                "formal": config.to_value(),
                "performance": profile.to_value(),
                **observation,
                **unqualified,
                **initialization,
            },
        )
        if config.format.endswith("@6"):
            application.training_loop.enable_update_observation()
        controller.attach(application.training_loop)
        initial_step = application.training_loop.optimizer_step
        if initial_quality_probe is not None:
            assert quality is not None
            assert quality_binding is not None
            import_initial_quality_probe(
                initial_quality_probe,
                monitor=quality,
                panel=quality_binding.panel,
                optimizer_step=initial_step,
                policy_snapshot_id=application.training_loop.policy_snapshot_id,
            )
        if quality is not None and source_condition is not None:
            if quality.policy.baseline_step != initial_step:
                raise ValueError("changed condition requires a baseline at the restored step")
        process = f"{os.getpid()}-from-step-{initial_step:08d}"
        _write(
            root / f"preparation-{process}.json",
            {
                "environment_evaluator_preparation_seconds": hydration_seconds,
                "model_or_resume_seconds": time.monotonic() - model_started,
                "total_preparation_seconds": time.monotonic() - started,
                "process_id": os.getpid(),
                "initial_optimizer_step": initial_step,
                "service_instance_id": os.environ.get("SKILLEV_SERVICE_INSTANCE_ID"),
                "warmup": "first-two-steps-per-process-all-times-retained",
            },
        )
        _write(root / f"initial-method-state-{process}.json", resolved_method_state(application))
        # No TrainingDeadline: 72 hours is a planning estimate, not a stopping rule.
        stop, thread = _start_progress_monitor(
            application,
            total_steps=config.steps,
            performance_path=root / "performance.jsonl",
            run_started=started,
            planning_hours=config.planning_hours,
            target_steps_per_hour=config.target_steps_per_hour,
            elapsed_since_run_start=run_clock.elapsed,
            metrics_condition_id=config.condition,
            metrics_condition_starts=observer_condition_starts(root, config),
            controller_observe=controller.publish,
        )
        _emit(
            "formal-training-started",
            initial_optimizer_step=initial_step,
            total_steps=config.steps,
            batch_size=config.batch_size,
        )
        request = StopAfterCheckpoint(root / "STOP_AFTER_CHECKPOINT")
        probes = (
            QualityProbeCoordinator(
                quality,
                FormalQualityCollector(
                    quality_binding,
                    application,
                    runtime,
                    bindings,
                    config,
                    profile,
                    effective,
                    interpreter,
                    mbpp_profile,
                    quality.policy,
                ),
            )
            if quality is not None and quality_binding is not None
            else None
        )

        async def stop_at_boundary() -> bool:
            if await asyncio.to_thread(controller.checkpoint_boundary):
                (root / "STOP_AFTER_CHECKPOINT").touch()
            controller.set_state("validating" if probes is not None else None)
            try:
                return await _stop_at_committed_boundary(
                    request,
                    probes,
                    optimizer_step=application.training_loop.optimizer_step,
                    policy_snapshot_id=application.training_loop.policy_snapshot_id,
                    pause_at_step=pause_at_step,
                    library_snapshot_id=(
                        application.library.state.current_version
                        if quality is not None
                        and quality.policy.library_axis == "combined-policy-library"
                        else None
                    ),
                )
            finally:
                controller.set_state(None)

        with request.signals():
            try:
                summary = await application.evolution_loop.run(
                    plan, stop_requested=stop_at_boundary
                )
            except TrainingPausedError as paused:
                application.training_loop.ledger.assert_fully_settled()
                evidence_status = await asyncio.to_thread(controller.finish_evidence)
                _write(
                    root / "paused.json",
                    {
                        "status": "paused-after-complete-checkpoint",
                        "optimizer_step": paused.optimizer_step,
                        "checkpoint": str(paused.checkpoint),
                        "planned_steps": config.steps,
                        "process_id": os.getpid(),
                        "evidence": evidence_status,
                    },
                )
                _emit("formal-training-paused", optimizer_step=paused.optimizer_step)
                controller.set_state("paused")
                return
        application.training_loop.ledger.assert_fully_settled()
        if summary.final_optimizer_step != config.steps:
            raise RuntimeError("formal run did not finish the declared number of complete steps")
        await asyncio.to_thread(controller.checkpoint_boundary)
        controller.set_state("validating" if probes is not None else None)
        final_quality_paused = probes is not None and await probes.check(
            policy_step=application.training_loop.optimizer_step,
            policy_snapshot_id=application.training_loop.policy_snapshot_id,
            library_snapshot_id=(
                application.library.state.current_version
                if quality is not None and quality.policy.library_axis == "combined-policy-library"
                else None
            ),
        )
        _write(root / "final-method-state.json", resolved_method_state(application))
        evidence_status = await asyncio.to_thread(controller.finish_evidence)
        final_paused = final_quality_paused or evidence_status["pause_required"]
        _write(
            root / "summary.json",
            {
                "mode": "five-step-training-observation"
                if observation
                else "owner-authorized-unqualified-training"
                if unqualified
                else "formal-bayesian-improve-single-owner",
                **observation,
                **unqualified,
                "status": "paused-after-complete-checkpoint" if final_paused else "finished",
                "summary": summary.to_value(),
                "evidence": evidence_status,
                "schedule": schedule_summary(config, selected),
                "current_process_initial_step": initial_step,
                "current_process_elapsed_seconds": time.monotonic() - started,
                "natural_evolution": "read-actual-mutations-not-assumed-from-step-count",
                "final_quality": (
                    "failed-or-unverified"
                    if final_quality_paused
                    else "declared-checks-passed"
                    if probes is not None
                    else "not-configured"
                ),
            },
        )
        _emit(
            "formal-training-paused" if final_paused else "formal-training-complete",
            optimizer_step=summary.final_optimizer_step,
        )
        controller.set_state("paused" if final_paused else "finished")
    except BaseException as exc:
        controller.set_state("failed")
        _emit("formal-training-failed-no-partial-batch-commit", error_type=type(exc).__name__)
        raise
    finally:
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=10.0)
        try:
            if evidence is not None:
                evidence.close()
        finally:
            try:
                coordinator.close()
            finally:
                controller.close_resources()


def main() -> None:
    from .bayesian_training_cli import main as run_cli

    run_cli()


if __name__ == "__main__":
    main()
