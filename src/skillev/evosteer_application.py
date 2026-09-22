"""EvoSteer composition root on the existing SKILLEV runtime infrastructure.

This is a new method identity. It never sends AnchorTB histories through the
old hindsight-TTB trainer, Beta calibration, or partition-reset phase machine.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from skillev.contracts.canonical import canonical_json, stable_hash
from skillev.contracts.evosteer import EvoTask, EvoTrajectory
from skillev.contracts.evosteer_risk import TrajectoryRiskAssessment
from skillev.evolution.paired_trials import (
    ComparisonContext,
    PairedTrialOutcome,
    PairScheduler,
    TrialArmSpec,
    TrialInitialCondition,
    TrialOutcome,
)
from skillev.evolution.evosteer_author import (
    SkillAuthorCallError,
    SkillAuthorOutputError,
    select_author_evidence,
)
from skillev.evolution.validated_admission import AdmissionConfig, AdmissionLedger, SkillStatus
from skillev.orchestration.actions import GraphAction, GraphActionKind
from skillev.orchestration.budget import EvoBudgetLedger
from skillev.orchestration.evosteer_features import FEATURE_VERSION
from skillev.orchestration.execution import GraphRuntime, TerminalEvaluator
from skillev.orchestration.graph import NodeExecutor, RoleSpec
from skillev.policy.state_value import StateValueHeads
from skillev.rollout.evosteer import collect_episode
from skillev.rollout.evosteer_risk import EvoSteerRiskError, require_assessed
from skillev.runtime import BudgetVector
from skillev.training.evosteer import EvoOptimizerConfig, EvoTrainer, structure_visit
from skillev.training.reference_statistics import (
    ReferenceObservation,
    ReferenceStatistics,
    ReferenceStatisticsConfig,
    StructureReferenceObservation,
)

CHECKPOINT_FORMAT = "evosteer-checkpoint@2"
AUTHOR_EVIDENCE_FORMAT = "evosteer-author-evidence-pool@1"
# Paired arms start from an action forced by the admission test, so the pool
# keeps only rollouts whose every action the orchestrator chose itself.
_NATURAL_SOURCES = frozenset({"current", "natural_reference"})
# Fields appended after runs were checkpointed. At these defaults they stay out
# of EvoSteerConfig.identity, so an unchanged configuration keeps the identity
# its existing checkpoints and value-snapshot IDs are bound to.
_IDENTITY_OPTIONAL_DEFAULTS = {
    "author_start_batch": 1,
    "author_start_families": 1,
    "author_evidence_pool": 0,
    "allow_repeated_pair_tasks": False,
    "validation_interval": 20,
}
# The same, one level down: asdict nests the optimizer, so an appended
# EvoOptimizerConfig field is dropped from that mapping rather than the top level.
_IDENTITY_OPTIONAL_OPTIMIZER_DEFAULTS = {
    "actor_beta1": 0.9,
    "head_beta1": 0.9,
}
_BATCH_ID = re.compile(r"batch-(\d{6,})")


@dataclass(frozen=True, slots=True)
class EvoSteerConfig:
    task_families: tuple[str, ...]
    roles: tuple[RoleSpec, ...]
    optimizer: EvoOptimizerConfig = field(default_factory=EvoOptimizerConfig)
    max_nodes: int | None = None
    max_actions: int | None = None
    current_rollouts: int = 2
    reference_rollouts: int = 2
    seed: int = 0
    author_interval: int = 1
    retirement_interval: int = 10
    # Every batch records its new pairs, but a look is spent only on a validation
    # batch. The per-look level shrinks with the look count, so testing on every
    # discordant pair drives the boundary far below any reachable sign-test
    # p-value and no comparison ever promotes or retires; batching the looks
    # spends the same total alpha over far fewer, better-powered tests.
    validation_interval: int = 20
    value_refresh_interval: int = 1
    strict_value_context: bool = False
    structure_source_mode: str = "reference_with_paired"
    total_token_cap: int = 16_384
    episode_budget: BudgetVector | None = None
    # Paper Table 4 keeps one candidate per task type; more slots keep the author
    # proposing while earlier candidates are still under sequential test.
    max_candidates_per_family: int = 1
    # Paired evidence is independent per task by default: a task counts once per
    # comparison. With a small training pool a candidate exhausts its family's
    # tasks within a few batches and can never reach a test boundary; setting this
    # lets a task be re-paired under a later policy/seed, which must be disclosed
    # because the sign test then treats repeated tasks as fresh observations.
    allow_repeated_pair_tasks: bool = False
    # Disclosed experimental schedule, not the paper's: batches before
    # author_start_batch open no author window, so no skill reaches any rollout
    # before batch author_start_batch + 1 while the orchestrator learns its
    # structure. The first open window then authors for up to
    # author_start_families families at once; later windows author one family.
    author_start_batch: int = 1
    author_start_families: int = 1
    # Per-family cap on natural trajectories collected before the first open
    # author window and shown to that window only, next to its own batch; the
    # window consumes the pool and every later window sees only its own batch
    # (the paper). 0 disables it.
    author_evidence_pool: int = 0

    def __post_init__(self) -> None:
        if not self.task_families or len(set(self.task_families)) != len(self.task_families):
            raise ValueError("task-family universe must be nonempty and unique")
        if not self.roles or len({role.role_id for role in self.roles}) != len(self.roles):
            raise ValueError("roles must have unique IDs")
        for name in (
            "current_rollouts",
            "reference_rollouts",
            "author_interval",
            "retirement_interval",
            "validation_interval",
            "value_refresh_interval",
            "total_token_cap",
            "max_candidates_per_family",
            "author_start_batch",
            "author_start_families",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if type(self.allow_repeated_pair_tasks) is not bool:
            raise TypeError("allow_repeated_pair_tasks must be boolean")
        if self.author_start_families > len(self.task_families):
            raise ValueError("author_start_families cannot exceed the task-family universe")
        pool = self.author_evidence_pool
        if type(pool) is not int or pool < 0 or pool == 1:
            # Like the author's own material capacity, a pool must be able to
            # hold one same-task success/failure contrast.
            raise ValueError("author_evidence_pool must be 0 (off) or an integer >= 2")
        if pool and self.first_author_batch == 1:
            # The pool feeds only the first open window; at batch 1 no earlier
            # batch exists to collect from, so the setting could never act.
            raise ValueError("author_evidence_pool needs a batch before the first author window")
        for name, minimum in (("max_nodes", 1), ("max_actions", 3)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be null or an integer >= {minimum}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("invalid horizon or seed")
        if type(self.strict_value_context) is not bool:
            raise ValueError("strict_value_context must be a boolean")
        if self.structure_source_mode not in {"reference_with_paired", "natural_only"}:
            raise ValueError("unsupported structure reference source mode")
        if self.episode_budget is None:
            # Each real model call consumes at least one input token. These
            # counters therefore cannot bind before the paper's shared token
            # cap; graph size/repair depth have no separate default cutoff.
            object.__setattr__(
                self,
                "episode_budget",
                BudgetVector(
                    input_tokens=self.total_token_cap,
                    output_tokens=self.total_token_cap,
                    model_calls=self.total_token_cap,
                    agent_turns=2 * self.total_token_cap + 50,
                    tool_calls=50,
                    wall_time_milliseconds=600_000,
                ),
            )
        elif not isinstance(self.episode_budget, BudgetVector):
            raise TypeError("episode_budget must be a BudgetVector")

    @property
    def resource_cap(self) -> BudgetVector:
        assert self.episode_budget is not None
        return self.episode_budget

    @property
    def first_author_batch(self) -> int:
        """The first batch that opens an author window (the multi-family window)."""
        interval = self.author_interval
        return -(-self.author_start_batch // interval) * interval

    @property
    def identity(self) -> str:
        value = asdict(self)
        for name, default in _IDENTITY_OPTIONAL_DEFAULTS.items():
            if value[name] == default:
                del value[name]
        for name, default in _IDENTITY_OPTIONAL_OPTIMIZER_DEFAULTS.items():
            if value["optimizer"][name] == default:
                del value["optimizer"][name]
        return str(stable_hash(value))


@dataclass(slots=True)
class TaskSession:
    """An isolated episode's shared executor/world and trusted terminal evaluator."""

    executor: NodeExecutor
    evaluator: TerminalEvaluator
    close: Callable[[], Any] | None = None
    reset_receipt: ResetReceipt | None = None
    risk_assessor: Callable[[EvoTrajectory], Any] | None = None


@dataclass(frozen=True, slots=True)
class SessionRequest:
    task: EvoTask
    seed: int
    pair_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResetReceipt:
    """Adapter-attested reset, with a fresh world ID and initial-state digest.

    Interactive adapters must derive initial_state_id from their reset result.
    It is not a clone of a mid-episode state or a guarantee supplied by the model.
    """

    task_identity: str
    reset_id: str
    environment_config_id: str
    initial_state_id: str
    session_id: str
    seed: int

    def __post_init__(self) -> None:
        for name in (
            "task_identity",
            "reset_id",
            "environment_config_id",
            "initial_state_id",
            "session_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"reset receipt {name} must be nonempty")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("reset receipt seed must be a nonnegative integer")

    def validate(self, request: SessionRequest) -> None:
        if (
            self.task_identity != request.task.identity
            or self.reset_id != request.task.reset_id
            or self.environment_config_id != request.task.environment_config_id
            or self.seed != request.seed
        ):
            raise ValueError("observed environment reset differs from the requested task/seed")


@dataclass(frozen=True, slots=True)
class TaskBinding:
    task: EvoTask
    executor_id: str
    session_factory: Callable[[SessionRequest], TaskSession]
    replay_safe: bool = False


@dataclass(frozen=True, slots=True)
class EvoBatchResult:
    batch_id: str
    trajectories: tuple[EvoTrajectory, ...]
    metrics: dict[str, Any]
    admission_decisions: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class _RolloutFrame:
    """State one collection reads, frozen at its start."""

    receipts: dict[str, ResetReceipt]
    value_heads: Any
    value_snapshot_id: str


@dataclass(frozen=True, slots=True)
class CollectedBatch:
    batch_number: int
    batch_id: str
    trajectories: tuple[EvoTrajectory, ...]
    paired_outcomes: tuple[PairedTrialOutcome, ...]
    receipts: dict[str, ResetReceipt]
    staged_admission: AdmissionLedger
    admission_base: AdmissionLedger


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    collected: CollectedBatch
    author_result: dict[str, Any]
    decisions: list[dict[str, Any]]
    skipped_pair_evidence: dict[str, int]
    # Staged author evidence pool; committed only when the batch's update is.
    evidence_pool: dict[str, tuple[EvoTrajectory, ...]] | None = None


def _report_rollout(trajectory: EvoTrajectory, seconds: float) -> None:
    """One progress line per finished episode; batches otherwise stay silent until committed."""
    kinds = [json.loads(d.action_json).get("kind") for d in trajectory.decisions]
    print(
        json.dumps(
            {
                "event": "rollout",
                "sample_id": trajectory.sample_id,
                "source": trajectory.source,
                "family": trajectory.task.family,
                "reward": round(trajectory.reward, 4),
                "actions": kinds,
                "seconds": round(seconds, 1),
            }
        ),
        file=sys.stdout,
        flush=True,
    )


def _updated_evidence_pool(
    pool: Mapping[str, tuple[EvoTrajectory, ...]],
    trajectories: tuple[EvoTrajectory, ...],
    families: tuple[str, ...],
    capacity: int,
) -> dict[str, tuple[EvoTrajectory, ...]]:
    """Re-rank each family's pool together with this batch's natural rollouts.

    The author's own contrast-first rule decides what stays, so the pool keeps
    the most informative same-task success/failure contrasts seen so far, and
    a mixed task can pair a success and a failure from different batches.
    """
    return {
        family: select_author_evidence(
            (
                *pool.get(family, ()),
                *(
                    item
                    for item in trajectories
                    if item.source in _NATURAL_SOURCES and item.task.family == family
                ),
            ),
            family,
            capacity,
        )
        for family in families
    }


def _evidence_pool_value(
    pool: Mapping[str, tuple[EvoTrajectory, ...]], families: tuple[str, ...], capacity: int
) -> dict[str, Any]:
    # Full records: risk_evidence_id binds every field, decisions included, so
    # a reduced record would fail the author's evidence binding unless its risk
    # assessment were re-signed, which would forge the trusted assessor's word.
    return {
        "format": AUTHOR_EVIDENCE_FORMAT,
        "capacity": capacity,
        "families": {family: [item.to_value() for item in pool[family]] for family in families},
    }


def _evidence_pool_from_value(
    value: object,
    *,
    families: tuple[str, ...],
    capacity: int,
    batch_index: int,
    retired: bool,
) -> dict[str, tuple[EvoTrajectory, ...]]:
    """Validate a saved pool completely before any of it is restored.

    ``retired`` means the first open author window is committed: it consumed
    the pool, so any record left would be stale evidence for a later window.
    """
    if not isinstance(value, dict) or set(value) != {"format", "capacity", "families"}:
        raise ValueError("checkpoint author evidence pool has incompatible fields")
    if value["format"] != AUTHOR_EVIDENCE_FORMAT:
        raise ValueError("unsupported checkpoint author evidence pool format")
    if type(value["capacity"]) is not int or value["capacity"] != capacity:
        raise ValueError("checkpoint author evidence capacity differs from the configuration")
    rows = value["families"]
    if not isinstance(rows, dict) or set(rows) != set(families):
        raise ValueError("checkpoint author evidence families differ from the family universe")
    pool: dict[str, tuple[EvoTrajectory, ...]] = {}
    seen: set[str] = set()
    for family in families:
        items = rows[family]
        if not isinstance(items, list) or len(items) > capacity:
            raise ValueError("checkpoint author evidence exceeds its per-family capacity")
        if retired and items:
            raise ValueError("checkpoint author evidence outlived the first author window")
        restored = []
        for item in items:
            trajectory = EvoTrajectory.from_value(item)
            if trajectory.to_value() != item:
                raise ValueError("checkpoint author evidence is not a canonical trajectory")
            batch = _BATCH_ID.fullmatch(trajectory.batch_id)
            if (
                trajectory.task.family != family
                or trajectory.source not in _NATURAL_SOURCES
                or batch is None
                or not 1 <= int(batch.group(1)) <= batch_index
                or not trajectory.sample_id.startswith(f"{trajectory.batch_id}/")
            ):
                raise ValueError(
                    "checkpoint author evidence is not a committed natural rollout of its family"
                )
            if trajectory.sample_id in seen:
                raise ValueError("checkpoint author evidence repeats a sample ID")
            seen.add(trajectory.sample_id)
            if select_author_evidence((trajectory,), family, 1) != (trajectory,):
                raise ValueError("checkpoint author evidence is not accepted, bound evidence")
            restored.append(trajectory)
        pool[family] = tuple(restored)
    return pool


class EvoSteerApplication:
    def __init__(
        self,
        policy: Any,
        config: EvoSteerConfig,
        *,
        admission: AdmissionLedger | None = None,
        author: Any = None,
        heads: Any = None,
    ) -> None:
        import torch

        self.policy = policy
        self.config = config
        self.admission = admission or AdmissionLedger(
            AdmissionConfig(
                max_candidates_per_family=config.max_candidates_per_family,
                allow_repeated_tasks=config.allow_repeated_pair_tasks,
            )
        )
        if config.author_start_batch > 1 and self.admission.skills:
            # Otherwise an injected seed or candidate would reach warmup rollouts.
            raise ValueError("a skill-free warmup requires an empty initial skill library")
        self.author = author
        self._evidence_pool: dict[str, tuple[EvoTrajectory, ...]] = {
            family: () for family in config.task_families
        }
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.heads = (
                heads
                if heads is not None
                else StateValueHeads(
                    encoding_dim=policy.encoding_dim, num_task_types=len(config.task_families)
                )
            )
        self.trainer = EvoTrainer(policy, self.heads, config.task_families, config.optimizer)
        self.statistics: dict[str, ReferenceStatistics] = {}
        self.batch_index = 0
        self.value_version = 0
        self._value_heads = copy.deepcopy(self.heads).cpu().eval()
        self._value_digest = self._hash_value_head()
        self.total_usage = BudgetVector()
        self._poisoned = False
        self._busy = False
        # Episodes are sampled by this policy; a separate replica enables prefetch.
        self.rollout_policy = policy

    @property
    def author_evidence(self) -> dict[str, tuple[EvoTrajectory, ...]]:
        """Committed per-family warmup evidence the first open author window adds."""
        return dict(self._evidence_pool)

    @property
    def value_snapshot_id(self) -> str:
        return f"{self.config.identity}/reference-value/{self.value_version}/{self._value_digest}"

    def _hash_value_head(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self._value_heads.runtime_head.state_dict().items()):
            digest.update(
                canonical_json(
                    {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                ).encode()
            )
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _context_id(
        self, binding: TaskBinding, bodies: dict[str, str], value_snapshot_id: str
    ) -> str:
        # Like the paper's lagged buckets, default pooling spans updated value
        # snapshots but never model/menu/role/environment changes. This is an
        # explicit approximation, not a claim of exact stationary calibration.
        return str(
            stable_hash(
                {
                    "reference": self.policy.configuration_id,
                    "executor": binding.executor_id,
                    "skills": bodies,
                    "roles": [role.to_value() for role in self.config.roles],
                    "environment": binding.task.environment_config_id,
                    "features": FEATURE_VERSION,
                    "pool": "strict-value-context"
                    if self.config.strict_value_context
                    else "pool-value-versions@1",
                    "value": value_snapshot_id if self.config.strict_value_context else None,
                    "max_nodes": self.config.max_nodes,
                    "max_actions": self.config.max_actions,
                    "budget": self.config.resource_cap.to_value(),
                    "total_token_cap": self.config.total_token_cap,
                    "structure_source_mode": self.config.structure_source_mode,
                }
            )
        )

    async def _rollout(
        self,
        binding: TaskBinding,
        *,
        batch_id: str,
        source: str,
        index: int,
        menu: Any,
        frame: _RolloutFrame,
        arm: TrialArmSpec | None = None,
    ) -> EvoTrajectory:
        sample_id = f"{batch_id}/{binding.task.task_id}/{source}/{index}"
        bodies = dict(menu.bodies)
        if arm is not None:
            for skill_id in arm.excluded_skill_ids:
                bodies.pop(skill_id, None)
        seed_key = arm.pair_id if arm is not None else sample_id
        seed = (
            self.config.seed + int(hashlib.sha256(seed_key.encode()).hexdigest()[:15], 16)
        ) % 2**63
        ledger = EvoBudgetLedger(
            run_id="evosteer",
            attempt_id=sample_id,
            cap=self.config.resource_cap,
            total_token_cap=self.config.total_token_cap,
        )
        request = SessionRequest(binding.task, seed, arm.pair_id if arm is not None else None)
        session = binding.session_factory(request)
        if not isinstance(session, TaskSession):
            raise TypeError("task session factory must return TaskSession")
        try:
            if session.risk_assessor is None:
                raise EvoSteerRiskError(
                    "task adapter must provide an explicit trajectory risk assessor"
                )
            if session.executor.frozen_identity != binding.executor_id:
                raise ValueError("task session differs from its frozen executor identity")
            if session.reset_receipt is not None:
                session.reset_receipt.validate(request)
                if any(
                    r.session_id == session.reset_receipt.session_id
                    for r in frame.receipts.values()
                ):
                    raise ValueError("episodes must use distinct isolated environment sessions")
                frame.receipts[sample_id] = session.reset_receipt
            elif arm is not None:
                raise ValueError("paired evidence requires an observed reset receipt")
            runtime = GraphRuntime(
                task_prompt=binding.task.prompt,
                roles=self.config.roles,
                skills=bodies,
                executor=session.executor,
                ledger=ledger,
                evaluator=session.evaluator,
                max_nodes=self.config.max_nodes,
                max_actions=self.config.max_actions,
                seed=seed,
                runtime_id=sample_id,
            )
            first = (
                None
                if arm is None
                else GraphAction(
                    GraphActionKind.ADD_AGENT,
                    node_id="n0",
                    role_id=arm.first_role,
                    skill_id=arm.skill_id if arm.arm == "positive" else None,
                )
            )
            trajectory = await collect_episode(
                policy=self.rollout_policy,
                runtime=runtime,
                ledger=ledger,
                task=binding.task,
                sample_id=sample_id,
                batch_id=batch_id,
                source=source,
                menu_id=menu.menu_id,
                value_snapshot_id=frame.value_snapshot_id,
                statistics_context=self._context_id(binding, bodies, frame.value_snapshot_id),
                value_function=lambda features, family: self.trainer.value(
                    features, family, heads=frame.value_heads
                ),
                seed=seed,
                forced_first_action=first,
                pair_id=arm.pair_id if arm is not None else None,
                candidate_id=arm.skill_id if arm is not None else None,
            )
            assessment = session.risk_assessor(trajectory)
            if inspect.isawaitable(assessment):
                assessment = await assessment
            if not isinstance(assessment, TrajectoryRiskAssessment):
                raise EvoSteerRiskError(
                    "risk assessor must return a typed evidence-bound assessment"
                )
            assessed = replace(trajectory, risk=assessment)
            require_assessed(assessed)
            return assessed
        finally:
            if session.close is not None:
                result = session.close()
                if inspect.isawaitable(result):
                    await result

    def _comparison(
        self,
        ledger: AdmissionLedger,
        binding: TaskBinding,
        menu: Any,
        *,
        value_snapshot_id: str,
        reevaluation: bool = False,
    ) -> Any:
        expected = SkillStatus.VALIDATED if reevaluation else SkillStatus.CANDIDATE
        candidates = [
            item
            for item in ledger.trial_priority(binding.task.family, reevaluation_due=reevaluation)
            if item.status is expected and any(m.skill_id == item.skill_id for m in menu.entries)
        ]
        if not candidates or not binding.replay_safe:
            return None

        def evidence_count(item: Any) -> int:
            return sum(
                c.wins + c.losses + c.ties
                for c in ledger.comparisons
                if c.skill_id == item.skill_id
            )

        skill = min(candidates, key=lambda item: (evidence_count(item), item.skill_id))
        context = ComparisonContext(
            task_family=binding.task.family,
            reference_snapshot_id=self.policy.reference_id,
            executor_snapshot_id=binding.executor_id,
            background_menu_id=menu.background_id((skill.skill_id,)),
            value_snapshot_id=value_snapshot_id,
            environment_config_id=binding.task.environment_config_id,
        )
        open_comparisons = [
            c for c in ledger.comparisons if c.skill_id == skill.skill_id and not c.closed
        ]
        matching = None
        for comparison in open_comparisons:
            previous = comparison.context
            # The value head is refreshed every batch. Both arms of a pair share
            # one snapshot, and under H0 every discordant pair is a fair coin
            # whatever snapshot produced it, so the sign test stays valid when
            # evidence accumulates across refreshes. Superseding on each refresh
            # would cap every comparison at one batch of pairs, too few for any
            # alpha-spending look to ever promote or retire a skill.
            if (
                previous.reference_snapshot_id != context.reference_snapshot_id
                or previous.background_menu_id != context.background_menu_id
                or comparison.reevaluation != reevaluation
            ):
                ledger.supersede_comparison(
                    comparison.comparison_id, "frozen comparison context changed"
                )
            elif replace(previous, value_snapshot_id=value_snapshot_id) == context:
                matching = comparison
            # Distinct environment/executor strata may coexist in one frozen
            # batch. Do not close a comparison before its already collected
            # pairs reach this batch's validation round.
        if matching is not None:
            return matching
        # Decisions may have changed status earlier in this same rollout batch.
        # The menu remains frozen; an in-flight trial is allowed, but do not
        # register a new candidate test against an already promoted skill.
        expected = SkillStatus.VALIDATED if reevaluation else SkillStatus.CANDIDATE
        if ledger.status(skill.skill_id) is not expected:
            return None
        identity = ledger.register_comparison(skill.skill_id, context, reevaluation=reevaluation)
        return ledger.comparison(identity)

    async def train_batch(self, bindings: tuple[TaskBinding, ...]) -> EvoBatchResult:
        if self._poisoned or self._busy:
            raise RuntimeError("application requires a healthy, idle committed boundary")
        if not bindings or len({b.task.task_id for b in bindings}) != len(bindings):
            raise ValueError("batch requires unique task records")
        if any(b.task.family not in self.config.task_families for b in bindings):
            raise ValueError("batch contains a task family outside the frozen universe")
        self._busy = True
        try:
            collected = await self._collect(bindings, self.batch_index + 1)
            return self._finish(self._prepare(collected))
        except BaseException:
            # Model updates/external effects are never rolled back or retried.
            # A caller can construct a NEW application from a completed checkpoint.
            self._poisoned = True
            raise
        finally:
            self._busy = False

    async def _gather_rollouts(
        self,
        batch_id: str,
        jobs: list[tuple[TaskBinding, str, int, Any, TrialArmSpec | None]],
        frame: _RolloutFrame,
    ) -> list[EvoTrajectory]:
        """Episodes run concurrently up to EVOSTEER_ROLLOUT_CONCURRENCY (default 1).

        Policy calls are synchronous, so orchestrator decisions never overlap on
        one model; only thread-safe executor generation proceeds in parallel.
        Results keep the job order. Actions and rewards match a sequential run;
        wall-clock features differ, as they do between any two runs.
        """
        limit = asyncio.Semaphore(int(os.environ.get("EVOSTEER_ROLLOUT_CONCURRENCY", "1")))

        async def one(
            binding: TaskBinding, source: str, index: int, menu: Any, arm: TrialArmSpec | None
        ) -> EvoTrajectory:
            async with limit:
                started = time.monotonic()
                trajectory = await self._rollout(
                    binding,
                    batch_id=batch_id,
                    source=source,
                    index=index,
                    menu=menu,
                    frame=frame,
                    arm=arm,
                )
                _report_rollout(trajectory, time.monotonic() - started)
                return trajectory

        return list(await asyncio.gather(*(one(*job) for job in jobs)))

    def _frame(self) -> _RolloutFrame:
        # Frozen per collection: a concurrent update may publish a new value head.
        return _RolloutFrame({}, self._value_heads, self.value_snapshot_id)

    def set_rollout_policy(self, sampler: Any) -> None:
        """Sample episodes with a separate replica of the actor (e.g. on another GPU)."""
        if sampler.configuration_id != self.policy.configuration_id:
            raise ValueError("rollout replica must share the actor's exact configuration")
        self.rollout_policy = sampler
        self.sync_rollout_policy()

    def sync_rollout_policy(self) -> None:
        """Publish the trained adapter to the rollout replica (Algorithm 1, step 3)."""
        if self.rollout_policy is not self.policy:
            self.rollout_policy.load_adapter_state(self.policy.adapter_state())
            self.rollout_policy.version = self.policy.version

    async def evaluate(
        self,
        bindings: tuple[TaskBinding, ...],
        *,
        sources: tuple[str, ...] = ("current",),
        samples: int = 1,
    ) -> list[EvoTrajectory]:
        """Roll out fixed held-out tasks with no update and no admission writes.

        Sample IDs, and therefore seeds, depend only on task, source and index,
        so every checkpoint is evaluated under the same common random numbers.
        """
        admission = AdmissionLedger.from_value(self.admission.to_value())
        menus = {
            family: admission.freeze_menu("heldout", family) for family in self.config.task_families
        }
        return await self._gather_rollouts(
            "heldout",
            [
                (binding, source, index, menus[binding.task.family], None)
                for binding in bindings
                for source in sources
                for index in range(samples)
            ],
            self._frame(),
        )

    async def collect_batch(
        self, bindings: tuple[TaskBinding, ...], *, batch_number: int
    ) -> CollectedBatch:
        """Algorithm 1 step 2: all natural and paired rollouts of one batch.

        It reads the committed admission ledger and a frozen value snapshot and
        writes no application state, so it may overlap the previous batch's
        update (the paper's asynchronous prefetch).
        """
        if self._poisoned:
            raise RuntimeError("application requires a healthy committed boundary")
        if not bindings or len({b.task.task_id for b in bindings}) != len(bindings):
            raise ValueError("batch requires unique task records")
        if any(b.task.family not in self.config.task_families for b in bindings):
            raise ValueError("batch contains a task family outside the frozen universe")
        if batch_number not in (self.batch_index + 1, self.batch_index + 2):
            raise ValueError("a batch may be collected at most one batch ahead")
        if batch_number == self.batch_index + 2 and self.rollout_policy is self.policy:
            raise ValueError("prefetching while the actor trains requires a rollout replica")
        # Collection writes no application state, so a failed collection leaves
        # the committed boundary (and any concurrent update) intact.
        return await self._collect(bindings, batch_number)

    async def _collect(self, bindings: tuple[TaskBinding, ...], batch_number: int) -> CollectedBatch:
        batch_id = f"batch-{batch_number:06d}"
        frame = self._frame()
        admission_base = self.admission
        staged_admission = AdmissionLedger.from_value(admission_base.to_value())
        menus = {
            family: staged_admission.freeze_menu(batch_id, family)
            for family in self.config.task_families
        }
        jobs: list[tuple[TaskBinding, str, int, Any, TrialArmSpec | None]] = [
            (binding, source, index, menus[binding.task.family], None)
            for binding in bindings
            for source, count in (
                ("current", self.config.current_rollouts),
                ("natural_reference", self.config.reference_rollouts),
            )
            for index in range(count)
        ]
        # Registering comparisons mutates the staged ledger, so it runs in task
        # order before any episode; pair outcomes are only recorded after the
        # batch, so running the planned pairs concurrently changes no decision.
        pairs: list[tuple[TaskBinding, Any, Any]] = []
        for binding in bindings:
            menu = menus[binding.task.family]
            trial_modes = [False]
            if batch_number % self.config.retirement_interval == 0:
                trial_modes.append(True)
            # A retirement round adds a validated-skill pair; it never starves
            # the current candidate merely because all validated slots are full.
            for trial_index, reevaluation in enumerate(trial_modes):
                comparison = self._comparison(
                    staged_admission,
                    binding,
                    menu,
                    reevaluation=reevaluation,
                    value_snapshot_id=frame.value_snapshot_id,
                )
                if comparison is None:
                    continue
                spec = PairScheduler.plan(
                    comparison.comparison_id,
                    comparison.skill_id,
                    TrialInitialCondition(
                        binding.task.task_id,
                        binding.task.reset_id,
                        self.config.roles[0].role_id,
                        comparison.context,
                    ),
                    pair_id=f"{batch_id}/{binding.task.task_id}/{comparison.comparison_id}",
                )
                pairs.append((binding, comparison, spec))
                jobs.append((binding, "paired_treatment", trial_index, menu, spec.positive))
                jobs.append((binding, "paired_control", trial_index, menu, spec.negative))
        trajectories = await self._gather_rollouts(batch_id, jobs, frame)
        natural_count = len(jobs) - 2 * len(pairs)
        paired_outcomes: list[PairedTrialOutcome] = []
        for pair_index, (binding, comparison, spec) in enumerate(pairs):
            positive = trajectories[natural_count + 2 * pair_index]
            negative = trajectories[natural_count + 2 * pair_index + 1]
            observed_positive = frame.receipts[positive.sample_id]
            observed_negative = frame.receipts[negative.sample_id]
            if observed_positive.initial_state_id != observed_negative.initial_state_id:
                raise ValueError("paired environments did not reset to the same initial state")
            paired_outcomes.append(
                PairedTrialOutcome(
                    TrialOutcome(
                        spec.positive,
                        positive.reward,
                        side_effect_free=require_assessed(positive).side_effect_free,
                        observed_initial_condition=TrialInitialCondition(
                            binding.task.task_id,
                            observed_positive.reset_id,
                            spec.positive.first_role,
                            comparison.context,
                        ),
                    ),
                    TrialOutcome(
                        spec.negative,
                        negative.reward,
                        side_effect_free=require_assessed(negative).side_effect_free,
                        observed_initial_condition=TrialInitialCondition(
                            binding.task.task_id,
                            observed_negative.reset_id,
                            spec.negative.first_role,
                            comparison.context,
                        ),
                    ),
                )
            )
        return CollectedBatch(
            batch_number,
            batch_id,
            tuple(trajectories),
            tuple(paired_outcomes),
            frame.receipts,
            staged_admission,
            admission_base,
        )

    def prepare_batch(self, collected: CollectedBatch) -> PreparedBatch:
        """Algorithm 1 step 8 (author window, pair accumulation, sign tests).

        Both consume only the scored pre-update batch, never the new adapter, so
        running them before the step-7 update changes no outcome. Committing the
        ledger first lets the next batch be collected against it during the update.
        """
        if self._poisoned or self._busy:
            raise RuntimeError("application requires a healthy, idle committed boundary")
        if (
            collected.batch_number != self.batch_index + 1
            or collected.admission_base is not self.admission
        ):
            raise ValueError("collected batch is stale for this committed boundary")
        try:
            return self._prepare(collected)
        except BaseException:
            self._poisoned = True
            raise

    def _prepare(self, collected: CollectedBatch) -> PreparedBatch:
        # Algorithm 1's full-batch gate precedes ALL learning, baseline and
        # admission writes. Assessments are bound to each exact saved history.
        complete = collected.trajectories
        for trajectory in complete:
            require_assessed(trajectory)
        batch_id = collected.batch_id
        batch_number = collected.batch_number
        staged_admission = collected.staged_admission
        # Staged before any author spend; every batch before the first open
        # window adds its natural rollouts, and that window consumes the pool.
        # Kept later, the author's tie-breaks (task hash, then earliest sample ID)
        # would let the same pre-skill contrasts outrank every later batch.
        # _finish commits the staged pool together with the update.
        evidence_pool: dict[str, tuple[EvoTrajectory, ...]] | None = None
        if self.config.author_evidence_pool:
            evidence_pool = (
                _updated_evidence_pool(
                    self._evidence_pool,
                    complete,
                    self.config.task_families,
                    self.config.author_evidence_pool,
                )
                if batch_number < self.config.first_author_batch
                else {family: () for family in self.config.task_families}
            )
        decisions: list[dict[str, Any]] = []
        author_result: dict[str, Any] = {"status": "not_scheduled"}
        if self.author is not None and batch_number < self.config.author_start_batch:
            author_result = {"status": "warmup", "opens_at_batch": self.config.first_author_batch}
        elif self.author is not None and batch_number % self.config.author_interval == 0:
            eligible = [
                family
                for family in self.config.task_families
                if staged_admission.can_propose(family)
            ]
            if eligible:
                # Paper Appendix C prefers families with few skills. Ties used to
                # fall to config order, which authored ceiling families (all runs
                # in the window succeeded) first. Among the fewest-skill families:
                # a family whose earlier windows produced no candidate yields to
                # the others (so one family cannot hold every window); a family
                # absent from the batch has no evidence and ranks last; then the
                # family with the most same-task success/failure contrasts, then
                # the one whose natural rollouts fail most, so the author sees
                # failures next to successes to learn a correction from.
                natural: dict[str, list[float]] = {}
                by_task: dict[str, dict[str, list[float]]] = {}
                for trajectory in complete:
                    if trajectory.source in {"current", "natural_reference"}:
                        family_name = trajectory.task.family
                        natural.setdefault(family_name, []).append(trajectory.reward)
                        by_task.setdefault(family_name, {}).setdefault(
                            trajectory.task.task_id, []
                        ).append(trajectory.reward)
                threshold = staged_admission.config.success_threshold
                reports = getattr(self.author, "reports", ())

                def author_priority(candidate: str) -> tuple[int, int, bool, int, float, int, int]:
                    rewards = natural.get(candidate, [])
                    contrasts = sum(
                        max(runs) >= threshold > min(runs)
                        for runs in by_task.get(candidate, {}).values()
                    )
                    return (
                        sum(s.family == candidate for s in staged_admission.skills),
                        sum(r.family == candidate and r.status != "candidate" for r in reports),
                        not rewards,
                        -contrasts,
                        sum(rewards) / len(rewards) if rewards else 1.0,
                        -sum(reward < threshold for reward in rewards),
                        self.config.task_families.index(candidate),
                    )

                # The first open window authors for several families at once from
                # the evidence pooled during warmup; every later window keeps the
                # paper's one proposal per window from its own batch only. The
                # keys are unique (the last is the config index), so the first
                # family equals the min().
                first_window = batch_number == self.config.first_author_batch
                count = self.config.author_start_families if first_window else 1
                windows: list[dict[str, Any]] = []
                for family in sorted(eligible, key=author_priority)[:count]:
                    # Each window needs its own ID in the ledger, the author
                    # report history and the budget reservation.
                    window_id = f"{batch_id}/{family}" if windows else batch_id
                    pooled = self._evidence_pool.get(family, ()) if first_window else ()
                    windows.append(
                        self._author_window(
                            staged_admission, complete, pooled, family, window_id
                        )
                    )
                # The first window keeps the single-window keys readers expect.
                author_result = dict(windows[0])
                if count > 1:
                    author_result["windows"] = windows
            else:
                author_result = {"status": "no_capacity"}
        # One statistical look per comparison per validation round, after all
        # its new pairs are accumulated. Ties alone never spend another alpha.
        comparisons_to_validate: set[str] = set()
        skipped_pair_evidence = {"repeated_task": 0, "not_side_effect_free": 0}
        for outcome in collected.paired_outcomes:
            arm_spec = outcome.positive.spec
            if not outcome.eligible:
                skipped_pair_evidence["not_side_effect_free"] += 1
                continue
            if not staged_admission.config.allow_repeated_tasks and staged_admission.has_task(
                arm_spec.comparison_id, arm_spec.initial_condition.task_id
            ):
                # Still generate/train the paper's pair on a repeated record;
                # do not pretend it is a new independent sign-test observation.
                skipped_pair_evidence["repeated_task"] += 1
                continue
            staged_admission.record_pair(outcome, validate=False)
            comparisons_to_validate.add(outcome.positive.spec.comparison_id)
        if batch_number % self.config.validation_interval == 0:
            for comparison_id in sorted(comparisons_to_validate):
                decision = staged_admission.validate_comparison(comparison_id)
                if decision is not None:
                    decisions.append(asdict(decision))
        self.admission = staged_admission
        return PreparedBatch(
            collected, author_result, decisions, skipped_pair_evidence, evidence_pool
        )

    def _author_window(
        self,
        ledger: AdmissionLedger,
        complete: tuple[EvoTrajectory, ...],
        pooled: tuple[EvoTrajectory, ...],
        family: str,
        window_id: str,
    ) -> dict[str, Any]:
        """One author window for one family; returns its batch-report entry."""
        if not ledger.can_propose(family):
            # An earlier window of this batch may have filled the total library;
            # do not pay for a call whose candidate could not be installed.
            return {"status": "no_capacity", "family": family, "window": window_id}
        # The first open window adds the committed warmup pool (earlier batches
        # only) to the whole current batch; any other window sees its batch alone.
        evidence = (*pooled, *complete) if pooled else complete
        try:
            entry = self.author.propose(
                evidence,
                family=family,
                window_id=window_id,
                seed=self.config.seed + self.batch_index,
            )
        except (SkillAuthorCallError, SkillAuthorOutputError) as error:
            # A failed or malformed author call proposes nothing in this window;
            # the author's own report records the failure. Other families'
            # windows in the same batch still run.
            entry = None
            author_error = f"{type(error).__name__}: {str(error)[:300]}"
        else:
            author_error = None
        result: dict[str, Any] = {"status": "no_proposal", "family": family, "window": window_id}
        if author_error is not None:
            result.update(status="author_error", error=author_error)
        if entry is not None:
            if entry.family != family:
                raise ValueError("author returned a candidate for another family")
            duplicate = any(
                entry.content_hash == skill.entry.content_hash for skill in ledger.skills
            )
            if duplicate:
                result["status"] = "duplicate_filtered"
            else:
                ledger.propose(entry, author_window_id=window_id)
                result.update(status="candidate_proposed", skill_id=entry.skill_id)
        return result

    def finish_batch(self, prepared: PreparedBatch) -> EvoBatchResult:
        """Algorithm 1 steps 4-7: baselines and the AnchorTB/value update."""
        if self._poisoned or self._busy:
            raise RuntimeError("application requires a healthy, idle committed boundary")
        if prepared.collected.batch_number != self.batch_index + 1:
            raise ValueError("prepared batch is not the next committed batch")
        self._busy = True
        try:
            return self._finish(prepared)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._busy = False

    def _finish(self, prepared: PreparedBatch) -> EvoBatchResult:
        collected = prepared.collected
        complete = collected.trajectories
        batch_id = collected.batch_id
        staged_statistics = {
            key: ReferenceStatistics.from_value(store.to_value())
            for key, store in self.statistics.items()
        }
        snapshots = {}
        for context in sorted({item.statistics_context for item in complete}):
            store = staged_statistics.setdefault(
                context,
                ReferenceStatistics(
                    context,
                    ReferenceStatisticsConfig(
                        beta=self.config.optimizer.beta,
                        structure_source_mode=self.config.structure_source_mode,
                    ),
                ),
            )
            observations = tuple(
                ReferenceObservation(
                    observation_id=item.sample_id,
                    task_id=item.task.task_id,
                    task_family=item.task.family,
                    reward=item.reward,
                    context_version=context,
                    prior_rate=item.task.prior_rate,
                    prior_count=item.task.prior_count,
                    visits=(
                        *(structure_visit(step.state_json) for step in item.decisions),
                        structure_visit(item.terminal_state_json),
                    ),
                )
                for item in complete
                if item.source == "natural_reference" and item.statistics_context == context
            )
            structure_observations = tuple(
                StructureReferenceObservation(
                    observation_id=item.sample_id,
                    task_id=item.task.task_id,
                    task_family=item.task.family,
                    reward=item.reward,
                    context_version=context,
                    visits=(
                        *(structure_visit(step.state_json) for step in item.decisions),
                        structure_visit(item.terminal_state_json),
                    ),
                    source=item.source,
                    pair_id=item.pair_id,
                    candidate_id=item.candidate_id,
                    forced_prefix_indices=(0,),
                    prior_rate=item.task.prior_rate,
                    prior_count=item.task.prior_count,
                )
                for item in complete
                if item.source in {"paired_treatment", "paired_control"}
                and item.statistics_context == context
            )
            snapshot = store.snapshot(
                batch_id, observations, structure_observations=structure_observations
            )
            snapshots[context] = snapshot
            store.commit_batch(snapshot)
        metrics: dict[str, Any] = self.trainer.update(complete, snapshots)
        self.statistics = staged_statistics
        self.batch_index += 1
        if prepared.evidence_pool is not None:
            # Committed with the statistics and the batch counter: a batch that
            # fails before its update leaves the pool exactly as it was.
            self._evidence_pool = prepared.evidence_pool
        if self.batch_index % self.config.value_refresh_interval == 0:
            self._value_heads = copy.deepcopy(self.heads).cpu().eval()
            self.value_version += 1
            self._value_digest = self._hash_value_head()
        usage = BudgetVector()
        for item in complete:
            usage = usage.add(BudgetVector.from_value(json.loads(item.usage_json)))
        self.total_usage = self.total_usage.add(usage)
        # Keep the aggregate training statistic for compatibility, but expose
        # source- and family-conditioned rewards as well.  ``mean_reward``
        # mixes current actor rollouts with frozen natural references (and,
        # on admission rounds, paired trials), so it is not an accuracy curve
        # for the actor.  These decomposed values make the plotted signal
        # auditable without changing the optimizer objective or the rewards.
        def reward_summary(items: list[EvoTrajectory]) -> dict[str, Any]:
            if not items:
                return {"count": 0, "mean": None, "std": None}
            values = [float(item.reward) for item in items]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            return {
                "count": len(values),
                "mean": mean,
                "std": variance**0.5,
            }

        current = [item for item in complete if item.source == "current"]
        natural_reference = [
            item for item in complete if item.source == "natural_reference"
        ]
        reward_by_family: dict[str, dict[str, Any]] = {}
        for family in self.config.task_families:
            reward_by_family[family] = {
                "current": reward_summary(
                    [item for item in current if item.task.family == family]
                ),
                "natural_reference": reward_summary(
                    [item for item in natural_reference if item.task.family == family]
                ),
            }
        metrics.update(
            {
                "author": prepared.author_result,
                "reset_receipts": {
                    sample_id: asdict(receipt)
                    for sample_id, receipt in collected.receipts.items()
                },
                "batch_index": self.batch_index,
                "pair_count": sum(x.source == "paired_treatment" for x in complete),
                "skipped_pair_evidence": prepared.skipped_pair_evidence,
                "alpha_spent": self.admission.alpha_spent,
                "usage": usage.to_value(),
                "cumulative_usage": self.total_usage.to_value(),
                "source_counts": {
                    source: sum(x.source == source for x in complete)
                    for source in sorted({x.source for x in complete})
                },
                "current_reward": reward_summary(current),
                "natural_reference_reward": reward_summary(natural_reference),
                "reward_by_family": reward_by_family,
                "skill_status": {
                    skill.skill_id: skill.status.value for skill in self.admission.skills
                },
            }
        )
        if self.config.author_evidence_pool:
            metrics["author_evidence_pool"] = {
                family: len(items) for family, items in self._evidence_pool.items()
            }
        return EvoBatchResult(batch_id, complete, metrics, tuple(prepared.decisions))

    def save_checkpoint(self, directory: str | Path) -> Path:
        import torch

        if self._poisoned or self._busy:
            raise RuntimeError("only healthy committed application state can be saved")
        destination = Path(directory)
        if destination.exists():
            raise FileExistsError("checkpoint destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".evosteer-", dir=destination.parent))
        try:
            tensor_path = temporary / "trainable.pt"
            torch.save(
                {
                    "actor": self.policy.adapter_state(),
                    "heads": self.heads.state_dict(),
                    "published_value": self._value_heads.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                },
                tensor_path,
            )
            state = {
                "format": CHECKPOINT_FORMAT,
                "config": self.config.identity,
                "policy": self.policy.configuration_id,
                "reference": self.policy.reference_id,
                "batch_index": self.batch_index,
                "actor_version": self.policy.version,
                "value_version": self.value_version,
                "admission": self.admission.to_value(),
                "value_digest": self._value_digest,
                "total_usage": self.total_usage.to_value(),
                "author": self.author.state_dict() if self.author is not None else None,
                "statistics": {key: store.to_value() for key, store in self.statistics.items()},
                "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
            }
            if self.config.author_evidence_pool:
                # Absent at the default, so default checkpoints are unchanged.
                state["author_evidence"] = _evidence_pool_value(
                    self._evidence_pool,
                    self.config.task_families,
                    self.config.author_evidence_pool,
                )
            payload = canonical_json(state)
            (temporary / "state.json").write_text(payload + "\n", encoding="utf-8")
            (temporary / "state.sha256").write_text(
                hashlib.sha256(payload.encode()).hexdigest() + "\n"
            )
            os.rename(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def load_checkpoint(self, directory: str | Path) -> None:
        import torch

        if self._busy or self._poisoned or self.batch_index or self.policy.version:
            raise RuntimeError("restore requires a fresh application")
        directory = Path(directory)
        payload = (directory / "state.json").read_text(encoding="utf-8").rstrip("\n")
        if (
            hashlib.sha256(payload.encode()).hexdigest()
            != (directory / "state.sha256").read_text().strip()
        ):
            raise ValueError("checkpoint state checksum differs")
        state = json.loads(payload)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                "unsupported EvoSteer checkpoint format: this method requires @2 "
                "with diagnostic heads and separate paired structure statistics"
            )
        if (
            state["config"] != self.config.identity
            or state["policy"] != self.policy.configuration_id
            or state["reference"] != self.policy.reference_id
        ):
            raise ValueError("checkpoint belongs to another model or EvoSteer configuration")
        tensor_path = directory / "trainable.pt"
        if hashlib.sha256(tensor_path.read_bytes()).hexdigest() != state["tensor_sha256"]:
            raise ValueError("checkpoint tensor checksum differs")
        admission = AdmissionLedger.from_value(state["admission"])
        statistics = {
            key: ReferenceStatistics.from_value(value) for key, value in state["statistics"].items()
        }
        if any(key != value.context_version for key, value in statistics.items()):
            raise ValueError("checkpoint statistic context key differs")
        for key in ("batch_index", "actor_version", "value_version"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError("invalid checkpoint version counter")
        if state["actor_version"] != state["batch_index"]:
            raise ValueError("checkpoint optimizer/actor batch counters differ")
        if state["value_version"] != state["batch_index"] // self.config.value_refresh_interval:
            raise ValueError("checkpoint value refresh counter differs")
        if (state["author"] is None) != (self.author is None):
            raise ValueError("checkpoint author presence differs")
        if (
            self.config.author_start_batch > 1
            and state["batch_index"] < self.config.author_start_batch
            and admission.skills
        ):
            raise ValueError("checkpoint inside the skill-free warmup contains skills")
        if ("author_evidence" in state) != bool(self.config.author_evidence_pool):
            raise ValueError("checkpoint author evidence pool presence differs")
        evidence_pool = (
            _evidence_pool_from_value(
                state["author_evidence"],
                families=self.config.task_families,
                capacity=self.config.author_evidence_pool,
                batch_index=state["batch_index"],
                retired=state["batch_index"] >= self.config.first_author_batch,
            )
            if self.config.author_evidence_pool
            else {family: () for family in self.config.task_families}
        )
        usage = BudgetVector.from_value(state["total_usage"])
        tensors = torch.load(tensor_path, map_location=self.policy.device, weights_only=True)
        try:
            self.policy.load_adapter_state(tensors["actor"])
            self.heads.load_state_dict(tensors["heads"], strict=True)
            self._value_heads.load_state_dict(tensors["published_value"], strict=True)
            if self._hash_value_head() != state["value_digest"]:
                raise ValueError("checkpoint published value identity differs")
            self.trainer.optimizer.load_state_dict(tensors["optimizer"])
            # torch restores the checkpoint's own parameter-group
            # hyperparameters next to its moments, so this run's
            # configuration is applied over them; the moments and the
            # step counts stay exactly as saved.
            self.trainer.apply_configured_hyperparameters()
            if self.author is not None:
                self.author.load_state_dict(state["author"])
            self.admission = admission
            self.statistics = statistics
            self._evidence_pool = evidence_pool
            self.batch_index = state["batch_index"]
            self.policy.version = state["actor_version"]
            self.value_version = state["value_version"]
            self._value_digest = state["value_digest"]
            self.total_usage = usage
        except BaseException:
            self._poisoned = True
            raise
