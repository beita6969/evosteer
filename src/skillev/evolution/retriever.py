"""Deterministic full-content retrieval from the active skill library."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

from skillev.contracts import JsonValue, TokenizerProtocol, canonical_json
from skillev.rollout import (
    RolloutSessionBundle,
    RolloutTask,
    UnskilledRolloutSessionBundle,
)
from skillev.runtime import (
    FullRetrievedSkillContext,
    RetrievalInclusionReason,
    SkillApplicability,
    SkillDocument,
    SkillLibrary,
    SkillMetadata,
    model_visible_skill_content,
)

RETRIEVAL_FORMAT = "skillev-skill-retrieval@3"


@dataclass(frozen=True, slots=True)
class TaskRetrievalFeatures:
    """Pre-action public coordinates. Never a retrospective ContextFeature:
    no final horizon, terminal outcome, or future observation is admissible.
    """

    task_id: str
    task_family: str
    context: str
    available_tools: tuple[str, ...]
    benchmark_id: str | None = None
    root_query: str = ""


class SkillExposureMode(StrEnum):
    """Whether H0 exposes a complete skill or a lazy-invocation summary."""

    SUMMARY_AND_LAZY_INVOKE = "summary-and-lazy-invoke"
    FULL_INLINE_NO_INVOKE = "full-inline-no-invoke"


@dataclass(frozen=True, slots=True)
class SkillApplicabilityV2:
    """Step-0 applicability richer than the persisted training document schema."""

    skill_id: str
    benchmark_ids: tuple[str, ...]
    task_families: tuple[str, ...]
    required_tools_any: tuple[str, ...] = ()
    required_tools_all: tuple[str, ...] = ()
    excluded_benchmarks: tuple[str, ...] = ()
    completion_kinds: tuple[str, ...] = ()
    prior_confidence_milli: int = 500

    def __post_init__(self) -> None:
        if not self.skill_id.strip() or not self.benchmark_ids:
            raise ValueError("Step-0 skill applicability requires an ID and benchmark")
        for name in (
            "benchmark_ids",
            "task_families",
            "required_tools_any",
            "required_tools_all",
            "excluded_benchmarks",
            "completion_kinds",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values or any(not item.strip() for item in values):
                raise ValueError(f"{name} must be sorted unique non-empty text")
        if set(self.benchmark_ids) & set(self.excluded_benchmarks):
            raise ValueError("included and excluded benchmarks overlap")
        if not 0 <= self.prior_confidence_milli <= 1000:
            raise ValueError("prior confidence must lie in [0, 1000]")


@dataclass(frozen=True, slots=True)
class StepZeroRetrievalPolicy:
    policy_id: str
    routes: tuple[SkillApplicabilityV2, ...]
    top_k: int = 1
    minimum_score: int = 1000
    exposure_mode: SkillExposureMode = SkillExposureMode.FULL_INLINE_NO_INVOKE

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.routes:
            raise ValueError("Step-0 retrieval policy identity and routes are required")
        if not 1 <= self.top_k <= 2:
            raise ValueError("Step-0 retrieval top_k must lie in [1, 2]")
        skill_ids = tuple(route.skill_id for route in self.routes)
        if len(set(skill_ids)) != len(skill_ids):
            raise ValueError("Step-0 retrieval routes must have unique skill IDs")


@dataclass(frozen=True, slots=True)
class SkillRejection:
    skill_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class SkillRetrievalDecision:
    selected: tuple[FullRetrievedSkillContext, ...]
    rejected: tuple[SkillRejection, ...]
    abstained: bool
    total_instruction_tokens: int
    policy_id: str
    exposure_mode: SkillExposureMode

    def __post_init__(self) -> None:
        if self.abstained != (not self.selected):
            raise ValueError("retrieval abstention must match the selected set")
        if self.total_instruction_tokens < 0 or not self.policy_id.strip():
            raise ValueError("retrieval decision identity is invalid")


@dataclass(frozen=True, slots=True)
class SkillRetrievalEvidence:
    skill_id: str
    task_family_match: bool
    context_match: bool
    required_tools_satisfied: bool
    specificity: int

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "context_match": self.context_match,
            "required_tools_satisfied": self.required_tools_satisfied,
            "skill_id": self.skill_id,
            "specificity": self.specificity,
            "task_family_match": self.task_family_match,
        }


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    task_id: str
    library_version: str
    candidates: tuple[SkillRetrievalEvidence, ...]
    selected_skill_ids: tuple[str, ...]
    assembled_skill_token_count: int
    format: str = RETRIEVAL_FORMAT

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "assembled_skill_token_count": self.assembled_skill_token_count,
            "candidates": [item.to_value() for item in self.candidates],
            "format": self.format,
            "library_version": self.library_version,
            "selected_skill_ids": list(self.selected_skill_ids),
            "task_id": self.task_id,
        }


def task_retrieval_features(task: RolloutTask) -> TaskRetrievalFeatures:
    benchmark_id: str | None = None
    if isinstance(task.public_context, dict):
        candidate = task.public_context.get("benchmark_id")
        if type(candidate) is str and candidate:
            benchmark_id = candidate
    return TaskRetrievalFeatures(
        task_id=task.task_id,
        task_family=task.task_family,
        context=task.context_id,
        available_tools=task.available_tools,
        benchmark_id=benchmark_id,
        root_query=task.query,
    )


def applicability_mismatch_reasons(
    applicability: SkillApplicability,
    features: TaskRetrievalFeatures,
) -> tuple[str, ...]:
    if not isinstance(features, TaskRetrievalFeatures):
        raise TypeError("retrieval requires pre-action public features, not post-hoc calibration")
    reasons: list[str] = []
    if features.context in applicability.excluded_contexts:
        reasons.append("context-excluded")
    if (
        "*" not in applicability.task_families
        and features.task_family not in applicability.task_families
    ):
        reasons.append("task-family-mismatch")
    if "*" not in applicability.contexts and features.context not in applicability.contexts:
        reasons.append("context-mismatch")
    if not set(applicability.required_tools) <= set(features.available_tools):
        reasons.append("required-tools-missing")
    return tuple(reasons)


def applicability_matches(
    applicability: SkillApplicability,
    features: TaskRetrievalFeatures,
    *,
    skill_id: str,
) -> SkillRetrievalEvidence | None:
    if applicability_mismatch_reasons(applicability, features):
        return None
    return SkillRetrievalEvidence(
        skill_id=skill_id,
        task_family_match=True,
        context_match=True,
        required_tools_satisfied=True,
        specificity=2 + len(applicability.required_tools),
    )


class TaskConditionedSkillRetriever:
    """Return every applicable active skill with full instructions."""

    def __init__(
        self,
        *,
        library: SkillLibrary,
    ) -> None:
        self._library = library

    @property
    def library(self) -> SkillLibrary:
        return self._library

    def retrieve(self, task: RolloutTask) -> tuple[FullRetrievedSkillContext, ...]:
        return self.retrieve_features(task_retrieval_features(task))

    def retrieve_features(
        self, features: TaskRetrievalFeatures
    ) -> tuple[FullRetrievedSkillContext, ...]:
        """The same applicability rule for public runtime and evaluation tasks."""
        matches: list[tuple[SkillRetrievalEvidence, SkillDocument]] = []
        for document in self._library.active_documents():
            evidence = applicability_matches(
                document.applicability,
                features,
                skill_id=document.manifest.skill_id,
            )
            if evidence is not None:
                matches.append((evidence, document))
        matches.sort(key=lambda item: (-item[0].specificity, item[0].skill_id))
        return tuple(
            FullRetrievedSkillContext(
                metadata=SkillMetadata.from_document(document),
                content=model_visible_skill_content(document),
                inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
            )
            for _, document in matches
        )

    def retrieve_step_zero(
        self,
        task: RolloutTask,
        *,
        policy: StepZeroRetrievalPolicy,
        tokenizer: TokenizerProtocol,
        instruction_token_budget: int,
    ) -> SkillRetrievalDecision:
        """Select a bounded benchmark-specific seed set, including valid abstention."""

        if not isinstance(policy, StepZeroRetrievalPolicy):
            raise TypeError("Step-0 retrieval requires a typed policy")
        if type(instruction_token_budget) is not int or instruction_token_budget < 1:
            raise ValueError("Step-0 skill token budget must be positive")
        features = task_retrieval_features(task)
        documents = {item.manifest.skill_id: item for item in self._library.active_documents()}
        query_terms = _lexical_terms(features.root_query)
        ranked: list[tuple[int, int, str, SkillDocument, str]] = []
        rejected: list[SkillRejection] = []
        tools = set(features.available_tools)
        for route in policy.routes:
            document = documents.get(route.skill_id)
            if document is None:
                rejected.append(SkillRejection(route.skill_id, "inactive"))
                continue
            if features.benchmark_id is None or features.benchmark_id in route.excluded_benchmarks:
                rejected.append(SkillRejection(route.skill_id, "benchmark-excluded"))
                continue
            if features.benchmark_id not in route.benchmark_ids:
                rejected.append(SkillRejection(route.skill_id, "benchmark-mismatch"))
                continue
            if route.task_families and features.task_family not in route.task_families:
                rejected.append(SkillRejection(route.skill_id, "task-family-mismatch"))
                continue
            if route.required_tools_all and not set(route.required_tools_all) <= tools:
                rejected.append(SkillRejection(route.skill_id, "required-tools-all-missing"))
                continue
            if route.required_tools_any and not set(route.required_tools_any) & tools:
                rejected.append(SkillRejection(route.skill_id, "required-tools-any-missing"))
                continue
            lexical = len(query_terms & _lexical_terms(f"{document.title} {document.summary}"))
            score = 10_000 + 1_000 * lexical + route.prior_confidence_milli
            content = (
                model_visible_skill_content(document)
                if policy.exposure_mode is SkillExposureMode.FULL_INLINE_NO_INVOKE
                else canonical_json({"summary": document.summary, "title": document.title})
            )
            token_cost = len(tokenizer.encode(content))
            ranked.append((score, -token_cost, route.skill_id, document, content))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected: list[FullRetrievedSkillContext] = []
        total = 0
        for score, negative_cost, skill_id, document, content in ranked:
            token_cost = -negative_cost
            if score < policy.minimum_score:
                rejected.append(SkillRejection(skill_id, "below-threshold"))
                continue
            if len(selected) >= policy.top_k:
                rejected.append(SkillRejection(skill_id, "top-k"))
                continue
            if total + token_cost > instruction_token_budget:
                rejected.append(SkillRejection(skill_id, "token-budget"))
                continue
            selected.append(
                FullRetrievedSkillContext(
                    metadata=SkillMetadata.from_document(document),
                    content=content,
                    inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
                )
            )
            total += token_cost
        return SkillRetrievalDecision(
            selected=tuple(selected),
            rejected=tuple(sorted(rejected, key=lambda item: (item.skill_id, item.reason))),
            abstained=not selected,
            total_instruction_tokens=total,
            policy_id=policy.policy_id,
            exposure_mode=policy.exposure_mode,
        )


_LEXICAL_TERM = re.compile(r"[a-z0-9]+")


def _lexical_terms(text: str) -> frozenset[str]:
    return frozenset(_LEXICAL_TERM.findall(text.casefold()))


class BaseRolloutSessionFactory(Protocol):
    def create(self, task: RolloutTask) -> UnskilledRolloutSessionBundle: ...


class RetrievingRolloutSessionFactory:
    def __init__(
        self,
        *,
        base_factory: BaseRolloutSessionFactory,
        retriever: TaskConditionedSkillRetriever,
    ) -> None:
        self._base_factory = base_factory
        self._retriever = retriever

    async def prepare_tasks(self, tasks: tuple[RolloutTask, ...]) -> tuple[RolloutTask, ...]:
        prepare = getattr(self._base_factory, "prepare_tasks", None)
        return tasks if prepare is None else cast(tuple[RolloutTask, ...], await prepare(tasks))

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        base = self._base_factory.create(task)
        return RolloutSessionBundle(
            environment=base.environment,
            evaluator=base.evaluator,
            retrieved_skills=self._retriever.retrieve(task),
            cleanup=base.cleanup,
        )


__all__ = [
    "RETRIEVAL_FORMAT",
    "BaseRolloutSessionFactory",
    "RetrievalDecision",
    "RetrievingRolloutSessionFactory",
    "SkillApplicabilityV2",
    "SkillExposureMode",
    "SkillRejection",
    "SkillRetrievalDecision",
    "SkillRetrievalEvidence",
    "StepZeroRetrievalPolicy",
    "TaskConditionedSkillRetriever",
    "TaskRetrievalFeatures",
    "applicability_matches",
    "task_retrieval_features",
]
