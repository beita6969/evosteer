"""Run a declared source-disjoint panel through the real, read-only collector.

This is private deployment wiring. Targets never enter panel metadata or the
actor; native scorer, budgets, wire and current library are the training ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from skillev.contracts import normalize_json
from skillev.evaluation.input_metric_contracts import HISTORICAL_IID_BENCHMARKS, IID_BENCHMARKS
from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.training.quality_gate import ProtocolProbe, QualityGatePolicy
from skillev.training.rollout_workflow import RolloutWorkflowResources
from skillev.training.run_condition import EffectiveRunCondition
from skillev_private.benchmarks.mbpp_scoring import MBPPScorerProfile
from skillev_private.benchmarks.protocol_v13_training import Protocol13TrainingRecord
from skillev_private.benchmarks.protocol_v13_training_sessions import (
    build_protocol13_training_sessions,
    native_scorer_contracts,
)

from .quality_panel import FixedQualityPanel, PanelSlot
from .zero_update_bridge import collect_training_condition

if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication
    from skillev.runtime.formal_sglang_runtime import BoundFormalSGLangRuntime
    from skillev.training.performance_config import TrainingPerformanceConfig

    from .bayesian_improve_training import FormalTrainingBindings
    from .bayesian_training_config import BayesianFormalConfig


def _slot(record: Protocol13TrainingRecord) -> PanelSlot:
    return PanelSlot(
        record.input.task_id,
        record.episode.benchmark.value,
        record.episode.population_id,
        record.episode.source_id,
    )


@dataclass(frozen=True)
class QualityCollectionBinding:
    panel: FixedQualityPanel
    records: tuple[Protocol13TrainingRecord, ...]
    excluded_sources: frozenset[tuple[str, str, str]]
    sampling_schedule_id: str
    ordered_task_sequence_id: str

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        training: tuple[Protocol13TrainingRecord, ...],
        policy: QualityGatePolicy,
        batch_size: int,
    ) -> QualityCollectionBinding:
        value = json.loads(path.read_text())
        record_path = Path(value["records"])
        if not record_path.is_absolute():
            record_path = path.parent / record_path
        with record_path.open() as stream:
            records = tuple(
                Protocol13TrainingRecord.from_value(json.loads(line))
                for line in stream
                if line.strip()
            )
        panel = FixedQualityPanel(
            value["panel_id"], value["condition_id"], tuple(_slot(record) for record in records)
        )
        if (panel.panel_id, panel.condition_id) != (policy.panel_id, policy.condition_id):
            raise ValueError("quality source binding differs from declared policy")
        # A frozen historical quality panel keeps its original condition/IDs.
        # New six-domain panels are distinct, never a filtered/relabelled old panel.
        if batch_size < 1 or {s.benchmark_id for s in panel.slots} not in (
            set(IID_BENCHMARKS),
            set(HISTORICAL_IID_BENCHMARKS),
        ):
            raise ValueError("quality collection requires a complete declared IID catalog")
        if len({s.canonical_source for s in panel.slots}) < policy.minimum_source_questions:
            raise ValueError("quality panel has too few distinct source questions")
        final_sources = value["final_evaluation_sources"]
        excluded = frozenset(_slot(record).source for record in training) | frozenset(
            (s["benchmark_id"], s["population_id"], s["source_question_id"]) for s in final_sources
        )
        panel.require_disjoint(excluded)
        schedule, order = value["sampling_schedule_id"], value["ordered_task_sequence_id"]
        if not all(isinstance(v, str) and v for v in (schedule, order)):
            raise ValueError("quality sampling coordinates must be declared")
        return cls(panel, records, excluded, schedule, order)

    def freeze(self, root: Path) -> None:
        """Preserve private input/target bindings, not merely the display panel ID."""
        value = {
            "panel": asdict(self.panel),
            "records": [record.to_value() for record in self.records],
            "excluded_sources": sorted(self.excluded_sources),
            "sampling_schedule_id": self.sampling_schedule_id,
            "ordered_task_sequence_id": self.ordered_task_sequence_id,
        }
        value = json.loads(json.dumps(value))
        path = root / "collection-binding-private.json"
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("quality panel or targets changed across resume")
        else:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            with path.open("x") as stream:
                json.dump(value, stream, allow_nan=False)
            path.chmod(0o600)


@dataclass
class FormalQualityCollector:
    binding: QualityCollectionBinding
    application: SKILLEVApplication
    runtime: BoundFormalSGLangRuntime
    bindings: FormalTrainingBindings
    config: BayesianFormalConfig
    profile: TrainingPerformanceConfig
    condition: EffectiveRunCondition
    mbpp_interpreter: Path
    mbpp_profile: MBPPScorerProfile
    quality_policy: QualityGatePolicy | None = None

    async def __call__(self, root: Path, step: int, snapshot_id: str) -> ProtocolProbe:
        application = self.application
        snapshot = application.generator.snapshot()
        if snapshot.snapshot_id != snapshot_id:
            raise ValueError("quality collection must use the committed forward policy")
        resources = RolloutWorkflowResources(self.profile.workflow())
        if self.bindings.topology is not None:
            for service in self.bindings.roles.services:
                resources.configure_model_endpoint(
                    service.endpoint,
                    capacity=service.request_capacity,
                    token_capacity=service.token_capacity,
                )
        journal = root.parent / (root.name + "-requests.sqlite3")
        from skillev_private.benchmarks.evaluation_episode import (
            EvaluationEpisodeRecord,
            EvaluationSource,
            EvaluationTarget,
        )

        # Quality sources are diagnostic inputs, not trainable source records.
        diagnostic_records = tuple(
            EvaluationEpisodeRecord(
                EvaluationSource(
                    r.episode.benchmark,
                    r.episode.population_id,
                    r.episode.source_id,
                    r.input.task_id,
                ),
                r.input,
                EvaluationTarget(r.output.target),
            )
            for r in self.binding.records
        )
        tasks, sessions = await build_protocol13_training_sessions(
            diagnostic_records,
            deployments_path=self.bindings.deployments,
            endpoint_base=self.bindings.roles.members("judge")[0].endpoint,
            judge_endpoints=tuple(s.endpoint for s in self.bindings.roles.members("judge")),
            request_journal_path=journal,
            base_model=self.bindings.base_model,
            resources=resources,
            mbpp_interpreter=self.mbpp_interpreter,
            mbpp_source_root=self.bindings.evalplus_source_root,
            mbpp_profile=self.mbpp_profile,
            hotpot_deliberation=self.config.hotpot_deliberation,
            rollout_budget=self.config.task_budget,
            static_rollout_budget=self.config.static_task_budget,
            domain_rollout_budgets=self.config.domain_task_budgets,
            lazy_environments=True,
            healthbench_judge=self.config.healthbench_judge,
        )
        generator = ExternalSGLangRolloutGenerator(
            config=self.runtime.binding.rollout,
            tokenizer=application.backbone.tokenizer,
            gateway=self.runtime.gateway,
            snapshot_provider=lambda: snapshot,
            request_journal=DurableRequestJournal(journal),
        )
        app_config = self.config.application_config("quality-collect-only")
        library = application.library.state
        if (
            self.quality_policy is not None
            and self.quality_policy.library_axis == "fixed-initial-library"
        ):
            from skillev.experiments._evolution_preflight_seed import planned_seed_documents
            from skillev.runtime import SkillLibraryState

            library = SkillLibraryState.from_seed_documents(
                planned_seed_documents(self.config.initial_skill_profile)
            )
        controls = normalize_json(
            {
                "execution_machine": "skillev.rollout.episode_executor.execute_episode@1",
                "rollout": self.config.sampling_config.to_value(),
                "maximum_h0_tokens": app_config.maximum_h0_tokens,
                "native_scorers": native_scorer_contracts(
                    self.mbpp_profile, self.config.healthbench_judge, domains=self.config.domains
                ),
                "public_tasks": [task.to_value() for task in tasks],
                "model_tokenizer": application.backbone.tokenizer.tokenizer_id,
                "library_axis": None
                if self.quality_policy is None
                else self.quality_policy.library_axis,
            }
        )
        assert isinstance(controls, dict)
        try:
            await collect_training_condition(
                root=root,
                condition=self.condition,
                tasks=tasks,
                generator=generator,
                base_sessions=sessions,
                library_state=library,
                trainer=app_config.trainer,
                maximum_h0_tokens=app_config.maximum_h0_tokens,
                workflow=resources.binding,
                workflow_resources=resources,
                sampling_schedule_id=self.binding.sampling_schedule_id,
                ordered_task_sequence_id=self.binding.ordered_task_sequence_id,
                sampled_policy_step=step,
                quality_panel=self.binding.panel,
                excluded_quality_sources=self.binding.excluded_sources,
                architecture_id=None
                if self.quality_policy is None
                else self.quality_policy.architecture_id,
                execution_controls=controls,
            )
        finally:
            generator.close()
        return ProtocolProbe(**json.loads((root / f"probe-{step:08d}.json").read_text()))
