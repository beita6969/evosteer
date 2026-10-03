from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
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
    describe_existing_skill,
    select_author_evidence,
)
from skillev.evolution.validated_admission import (
    AdmissionConfig,
    AdmissionLedger,
    SkillStatus,
)
from skillev.orchestration.actions import GraphAction, GraphActionKind
from skillev.orchestration.budget import EvoBudgetLedger
from skillev.orchestration.evosteer_features import FEATURE_VERSION
from skillev.orchestration.execution import GraphRuntime, TerminalEvaluator
from skillev.orchestration.graph import NodeExecutor, RoleSpec
from skillev.orchestration.text_tools import KNOWN_TEXT_TOOLS, TextToolsConfig
from skillev.policy.state_value import StateValueHeads
from skillev.rollout.evosteer import collect_episode
from skillev.rollout.evosteer_risk import EvoSteerRiskError, require_assessed
from skillev.runtime.contracts import BudgetVector
from skillev.scoring.anchor_tb import log_terminal_reward
from skillev.training.evosteer import EvoOptimizerConfig, EvoTrainer, structure_visit
from skillev.training.reference_statistics import (
    ReferenceObservation,
    ReferenceStatistics,
    ReferenceStatisticsConfig,
    StructureReferenceObservation,
)

CHECKPOINT_FORMAT = "evosteer-checkpoint@2"
AUTHOR_EVIDENCE_FORMAT = "evosteer-author-evidence-pool@1"
_NATURAL_SOURCES = frozenset({"current", "natural_reference"})
_RECORD_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "text_tools": None,
    "controller_wall_milliseconds": 120_000,
    "non_output_roles": (),
    "stop_gate_families": (),
}
STATISTICS_POOL = "paper-run-pool@1"
_IDENTITY_OPTIONAL_DEFAULTS = {
    "author_start_batch": 1,
    "author_start_families": 1,
    "author_evidence_pool": 0,
    "allow_repeated_pair_tasks": False,
    "validation_interval": 20,
    "family_role_maximum": None,
    **_RECORD_OPTIONAL_DEFAULTS,
}
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
    validation_interval: int = 20
    value_refresh_interval: int = 1
    structure_source_mode: str = "reference_with_paired"
    total_token_cap: int = 16_384
    episode_budget: BudgetVector | None = None
    family_role_maximum: tuple[tuple[str, BudgetVector], ...] | None = None
    text_tools: TextToolsConfig | None = None
    controller_wall_milliseconds: int = 120_000
    non_output_roles: tuple[str, ...] = ()
    stop_gate_families: tuple[str, ...] = ()
    max_candidates_per_family: int = 1
    allow_repeated_pair_tasks: bool = False
    author_start_batch: int = 1
    author_start_families: int = 1
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
            "controller_wall_milliseconds",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if type(self.allow_repeated_pair_tasks) is not bool:
            raise TypeError("allow_repeated_pair_tasks must be boolean")
        if self.author_start_families > len(self.task_families):
            raise ValueError("author_start_families cannot exceed the task-family universe")
        pool = self.author_evidence_pool
        if type(pool) is not int or pool < 0 or pool == 1:
            raise ValueError("author_evidence_pool must be 0 (off) or an integer >= 2")
        if pool and self.first_author_batch == 1:
            raise ValueError("author_evidence_pool needs a batch before the first author window")
        for name, minimum in (("max_nodes", 1), ("max_actions", 3)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be null or an integer >= {minimum}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("invalid horizon or seed")
        if self.structure_source_mode not in {"reference_with_paired", "natural_only"}:
            raise ValueError("unsupported structure reference source mode")
        if self.episode_budget is None:
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
        if self.family_role_maximum is not None:
            overrides = self.family_role_maximum
            if not isinstance(overrides, tuple) or not overrides or any(
                not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[1], BudgetVector)
                for item in overrides
            ):
                raise TypeError("family_role_maximum must be a nonempty tuple of (family, BudgetVector)")
            names = [family for family, _ in overrides]
            if names != sorted(set(names)) or not set(names) <= set(self.task_families):
                raise ValueError("family_role_maximum must name distinct task families, sorted")
            for family, maximum in overrides:
                if maximum.model_calls < 1:
                    raise ValueError(f"{family} envelope must reserve a model call")
                if (
                    not maximum.fits_within(self.episode_budget)
                    or maximum.input_tokens + maximum.output_tokens >= self.total_token_cap
                ):
                    raise ValueError(f"{family} envelope cannot fit one episode budget")
        for name, universe, label in (
            ("non_output_roles", {role.role_id for role in self.roles}, "roles"),
            ("stop_gate_families", set(self.task_families), "task families"),
        ):
            names = getattr(self, name)
            if (
                not isinstance(names, tuple)
                or any(type(item) is not str for item in names)
                or list(names) != sorted(set(names))
            ):
                raise ValueError(f"{name} must be a sorted tuple of distinct names")
            if not set(names) <= universe:
                raise ValueError(f"{name} must name configured {label}: {sorted(names)}")
        if self.roles[0].role_id in self.non_output_roles:
            raise ValueError("the first role (paired trials start from it) cannot be non-output")
        self._check_text_tools()

    def _check_text_tools(self) -> None:
        declared = {tool for role in self.roles for tool in role.tools}
        unknown = declared - KNOWN_TEXT_TOOLS
        if unknown:
            raise ValueError(
                f"role tools must be among {sorted(KNOWN_TEXT_TOOLS)}: {sorted(unknown)}"
            )
        settings = self.text_tools
        if settings is None:
            if declared:
                raise ValueError("roles declare tools but text_tools is not configured")
            return
        if not isinstance(settings, TextToolsConfig):
            raise TypeError("text_tools must be a TextToolsConfig")
        families = [family for family, _ in settings.family_tools]
        if not set(families) <= set(self.task_families):
            raise ValueError("text_tools.family_tools must name configured task families")
        overridden = {family for family, _ in self.family_role_maximum or ()}
        if overridden & set(families):
            raise ValueError(
                "text_tools.family_tools and family_role_maximum must name different families"
            )
        by_id = {role.role_id: role for role in self.roles}
        for role_id, _ in settings.role_maximum:
            if role_id not in by_id or not by_id[role_id].tools:
                raise ValueError(f"text_tools.role_maximum names {role_id!r}, not a tool role")
        for role_id in settings.tool_only_roles:
            if role_id not in by_id or not by_id[role_id].tools:
                raise ValueError(f"text_tools.tool_only_roles names {role_id!r}, not a tool role")
            if role_id == self.roles[0].role_id:
                raise ValueError("the first role (paired trials start from it) cannot be tool-only")
        for role_id, pairs in settings.role_tool_instructions:
            if role_id not in by_id or not {tool for tool, _ in pairs} <= set(
                by_id[role_id].tools
            ):
                raise ValueError(
                    f"text_tools.role_tool_instructions names {role_id!r} or a tool "
                    "it does not declare"
                )
        for family, pairs in settings.family_role_instructions:
            offered = set(settings.tools_for(family))
            for role_id, _ in pairs:
                if role_id not in by_id or not offered & set(by_id[role_id].tools):
                    raise ValueError(
                        f"text_tools.family_role_instructions names {role_id!r}, which has no "
                        f"tool in {family}"
                    )
        for family in self.task_families:
            for role in self.roles_for(family):
                if not role.tools:
                    continue
                maximum = role.model_maximum
                if maximum.model_calls < 2 or maximum.tool_calls < 1:
                    raise ValueError(
                        f"{role.role_id} runs tools in {family} and needs a text_tools."
                        "role_maximum envelope with >= 2 model calls and >= 1 tool call"
                    )
                if (
                    not maximum.fits_within(self.resource_cap)
                    or maximum.input_tokens + maximum.output_tokens >= self.total_token_cap
                ):
                    raise ValueError(
                        f"{role.role_id} tool envelope cannot fit one {family} episode budget"
                    )

    def roles_for(self, family: str) -> tuple[RoleSpec, ...]:
        tool_only = set(self.text_tools.tool_only_roles) if self.text_tools else set()
        for name, maximum in self.family_role_maximum or ():
            if name == family:
                return tuple(
                    replace(role, model_maximum=maximum, tools=())
                    for role in self.roles
                    if role.role_id not in tool_only
                )
        if self.text_tools is None:
            return self.roles
        offered = set(self.text_tools.tools_for(family))
        roles = []
        for role in self.roles:
            tools = tuple(tool for tool in role.tools if tool in offered)
            if not tools and role.role_id in tool_only:
                continue
            maximum = self.text_tools.maximum_for(role.role_id) if tools else None
            sentences = dict(self.text_tools.tool_instructions_for(role.role_id))
            instruction = (
                self.text_tools.family_instruction_for(family, role.role_id) if tools else None
            ) or " ".join(
                (role.instruction, *(sentences[tool] for tool in tools if tool in sentences))
            )
            if tools == role.tools and maximum is None and instruction == role.instruction:
                roles.append(role)
            else:
                roles.append(
                    replace(
                        role,
                        instruction=instruction,
                        tools=tools,
                        **({"model_maximum": maximum} if maximum else {}),
                    )
                )
        return tuple(roles)

    def non_output_roles_for(self, family: str) -> tuple[str, ...]:
        present = {role.role_id for role in self.roles_for(family)}
        return tuple(role_id for role_id in self.non_output_roles if role_id in present)

    @property
    def resource_cap(self) -> BudgetVector:
        assert self.episode_budget is not None
        return self.episode_budget

    @property
    def first_author_batch(self) -> int:
        interval = self.author_interval
        return -(-self.author_start_batch // interval) * interval

    @property
    def identity(self) -> str:
        value = _without_empty_tool_fields(asdict(self))
        for name, default in _IDENTITY_OPTIONAL_DEFAULTS.items():
            if value[name] == default:
                del value[name]
        for name, default in _IDENTITY_OPTIONAL_OPTIMIZER_DEFAULTS.items():
            if value["optimizer"][name] == default:
                del value["optimizer"][name]
        return str(stable_hash(value))

    def record(self) -> dict[str, Any]:
        value = _without_empty_tool_fields(asdict(self))
        for name, default in _RECORD_OPTIONAL_DEFAULTS.items():
            if value[name] == default:
                del value[name]
        return value


STOP_GATE_RULE = "environment-done@1"


def environment_stop_gate(executor: Any) -> Callable[[], bool]:
    def stop_allowed() -> bool:
        done = getattr(executor, "environment_done", None)
        if done is not None and type(done) is not bool:
            raise TypeError("executor environment_done must be a boolean or None")
        return done is not False

    return stop_allowed


def _without_empty_tool_fields(value: dict[str, Any]) -> dict[str, Any]:
    for role in value["roles"]:
        if not role["tools"]:
            del role["tools"]
    text_tools = value.get("text_tools")
    if text_tools is not None:
        for name in ("role_tool_instructions", "family_role_instructions"):
            if not text_tools[name]:
                del text_tools[name]
    return value


@dataclass(slots=True)
class TaskSession:
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
    receipts: dict[str, ResetReceipt]
    value_heads: Any
    value_snapshot_id: str
    value_fitted: bool = False


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
    evidence_pool: dict[str, tuple[EvoTrajectory, ...]] | None = None


def _paper_statistics_metrics(
    snapshots: Mapping[str, Any], complete: tuple[EvoTrajectory, ...], beta: float
) -> dict[str, Any]:
    (snapshot,) = snapshots.values()
    excess: dict[str, list[float]] = {}
    for item in complete:
        if item.source != "natural_reference":
            continue
        anchor = snapshot.task_anchor(
            item.task.task_id,
            item.task.family,
            exclude_observation_id=item.sample_id,
            prior_rate=item.task.prior_rate,
            prior_count=item.task.prior_count,
        )
        excess.setdefault(item.task.family, []).append(
            math.expm1(log_terminal_reward(item.reward, beta) - anchor)
        )
    return {
        "natural_observations": len(snapshot.observations),
        "lagged_buckets": len(snapshot.structure_buckets),
        "lagged_buckets_at_minimum": sum(
            bucket.visits >= snapshot.config.structure_minimum
            for bucket in snapshot.structure_buckets
        ),
        "natural_excess_ratio_by_family": {
            family: sum(values) / len(values) for family, values in sorted(excess.items())
        },
    }


def _report_rollout(trajectory: EvoTrajectory, seconds: float) -> None:
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
        self.rollout_policy = policy

    @property
    def author_evidence(self) -> dict[str, tuple[EvoTrajectory, ...]]:
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

    def _context_id(self) -> str:
        return str(
            stable_hash(
                {
                    "scope": STATISTICS_POOL,
                    "reference": self.policy.configuration_id,
                    "config": self.config.identity,
                    "features": FEATURE_VERSION,
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
            roles = self.config.roles_for(binding.task.family)
            if any(role.tools for role in roles) and not getattr(
                session.executor, "supports_role_tools", False
            ):
                raise ValueError("roles declare tools but the task executor cannot run them")
            runtime = GraphRuntime(
                task_prompt=binding.task.prompt,
                roles=roles,
                skills=bodies,
                executor=session.executor,
                ledger=ledger,
                evaluator=session.evaluator,
                max_nodes=self.config.max_nodes,
                max_actions=self.config.max_actions,
                seed=seed,
                runtime_id=sample_id,
                non_output_roles=self.config.non_output_roles_for(binding.task.family),
                stop_gate=(
                    environment_stop_gate(session.executor)
                    if binding.task.family in self.config.stop_gate_families
                    else None
                ),
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
                statistics_context=self._context_id(),
                value_function=lambda features, family: self.trainer.value(
                    features, family, heads=frame.value_heads
                ),
                value_batch_function=lambda rows, family: self.trainer.values(
                    rows, family, heads=frame.value_heads
                ),
                value_fitted=frame.value_fitted,
                seed=seed,
                forced_first_action=first,
                controller_deadline_ms=self.config.controller_wall_milliseconds,
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
        if matching is not None:
            return matching
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
        return _RolloutFrame(
            {}, self._value_heads, self.value_snapshot_id, self.value_version > 0
        )

    def set_rollout_policy(self, sampler: Any) -> None:
        if sampler.configuration_id != self.policy.configuration_id:
            raise ValueError("rollout replica must share the actor's exact configuration")
        self.rollout_policy = sampler
        self.sync_rollout_policy()

    def sync_rollout_policy(self) -> None:
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
        pairs: list[tuple[TaskBinding, Any, Any]] = []
        for binding in bindings:
            menu = menus[binding.task.family]
            trial_modes = [False]
            if batch_number % self.config.retirement_interval == 0:
                trial_modes.append(True)
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
        complete = collected.trajectories
        for trajectory in complete:
            require_assessed(trajectory)
        batch_id = collected.batch_id
        batch_number = collected.batch_number
        staged_admission = collected.staged_admission
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

                first_window = batch_number == self.config.first_author_batch
                count = self.config.author_start_families if first_window else 1
                windows: list[dict[str, Any]] = []
                for family in sorted(eligible, key=author_priority)[:count]:
                    window_id = f"{batch_id}/{family}" if windows else batch_id
                    pooled = self._evidence_pool.get(family, ()) if first_window else ()
                    windows.append(
                        self._author_window(
                            staged_admission, complete, pooled, family, window_id
                        )
                    )
                author_result = dict(windows[0])
                if count > 1:
                    author_result["windows"] = windows
            else:
                author_result = {"status": "no_capacity"}
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
        if not ledger.can_propose(family):
            return {"status": "no_capacity", "family": family, "window": window_id}
        evidence = (*pooled, *complete) if pooled else complete
        existing = tuple(
            describe_existing_skill(item.entry, item.status.value)
            for item in ledger.skills
            if item.family == family
        )
        try:
            entry = self.author.propose(
                evidence,
                family=family,
                window_id=window_id,
                seed=self.config.seed + self.batch_index,
                existing_skills=existing,
            )
        except (SkillAuthorCallError, SkillAuthorOutputError) as error:
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
        staged_statistics = {key: store.clone() for key, store in self.statistics.items()}
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
            self._evidence_pool = prepared.evidence_pool
        if self.batch_index % self.config.value_refresh_interval == 0:
            self._value_heads = copy.deepcopy(self.heads).cpu().eval()
            self.value_version += 1
            self._value_digest = self._hash_value_head()
        usage = BudgetVector()
        for item in complete:
            usage = usage.add(BudgetVector.from_value(json.loads(item.usage_json)))
        self.total_usage = self.total_usage.add(usage)
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
        metrics["reference_statistics"] = _paper_statistics_metrics(
            snapshots, complete, self.config.optimizer.beta
        )
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
        if set(statistics) - {self._context_id()}:
            raise ValueError("checkpoint statistics are not this run's single pool")
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
