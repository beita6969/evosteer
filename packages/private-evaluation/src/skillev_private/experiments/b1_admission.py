"""Result-blind exact admissions required before the B1 protocol freeze.

The complete private catalog is loaded by the existing B0 authority.  This
module only inspects its answer-free :class:`~skillev.rollout.RolloutTask`
projections.  It neither selects tasks nor opens an environment/evaluator.

Two facts are established here:

* every catalog task's canonical ``H_0`` fits when encoded by the pinned Qwen
  tokenizer; the task/wrapper portion is measured alongside the separate
  F2/F3 capacity proof but is not a second per-task admission limit;
* every task-derived Generate group has exactly one legal authoring authority,
  with at most one matching group for that task in a cycle.

No top-k selection, truncation, task replacement, or inferred tokenizer is
available at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from skillev.contracts import JsonValue, normalize_json, stable_hash, validate_sha256
from skillev.evolution import (
    SkillAuthoringAuthority,
    TaskConditionedSkillRetriever,
)
from skillev.experiments import ACTIVE_BENCHMARKS, Benchmark
from skillev.policy import (
    QwenDeploymentConfig,
    TokenizerArtifactIdentity,
)
from skillev.rollout import (
    AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP,
    MAXIMUM_APPLICABLE_SKILL_POSITION,
    CanonicalInitialContextAssembler,
    RolloutTask,
    RolloutTokenizerProtocol,
    maximum_retrieved_skill_block_token_count,
)
from skillev.runtime import (
    FullRetrievedSkillContext,
    RetrievalInclusionReason,
    SkillDocument,
    SkillLibrary,
    SkillLibraryState,
    SkillMetadata,
    model_visible_skill_content,
)
from skillev_private.benchmarks.catalog import PrivateBenchmarkCatalog

B1_EXACT_ADMISSION_FORMAT = "skillev-private-b1-exact-admission@2"


class _AdmissionTokenizer(RolloutTokenizerProtocol, Protocol):
    @property
    def artifact_identity(self) -> TokenizerArtifactIdentity: ...

    def encode(self, text: str) -> list[int]: ...


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _sha256(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    validate_sha256(value)
    return value


@dataclass(frozen=True, slots=True)
class B1BenchmarkPopulationCount:
    benchmark: Benchmark
    task_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("B1 population benchmark must be Benchmark")
        _positive_int(self.task_count, field="B1 benchmark task_count")

    def to_value(self) -> dict[str, JsonValue]:
        return {"benchmark": self.benchmark.value, "task_count": self.task_count}

    @classmethod
    def from_value(cls, value: object) -> B1BenchmarkPopulationCount:
        data = normalize_json(value)
        if not isinstance(data, dict) or set(data) != {"benchmark", "task_count"}:
            raise ValueError("B1 benchmark population count has incompatible fields")
        if type(data["benchmark"]) is not str:
            raise TypeError("B1 benchmark population benchmark must be text")
        return cls(
            benchmark=Benchmark(data["benchmark"]),
            task_count=_positive_int(data["task_count"], field="B1 benchmark task_count"),
        )


@dataclass(frozen=True, slots=True)
class B1ExactAdmissionReport:
    """Path-free, answer-free commitment to the two exact B1 admissions."""

    catalog_freeze_hash: str
    tokenizer_artifact: TokenizerArtifactIdentity
    initial_library_version: str
    initial_library_state_hash: str
    maximum_h0_tokens: int
    task_wrapper_reserve_tokens: int
    task_population_count: int
    task_population_hash: str
    benchmark_counts: tuple[B1BenchmarkPopulationCount, ...]
    exact_h0_admission_hash: str
    maximum_exact_h0_tokens: int
    maximum_task_wrapper_tokens: int
    maximum_initial_retrieved_skills: int
    maximum_applicable_skills_per_task: int
    maximum_authored_skill_full_block_tokens: int
    maximum_future_complete_h0_tokens: int
    generate_authority_hash: str
    generate_authority_group_count: int
    maximum_generate_authority_matches_per_task_per_cycle: int
    authoring_authority_hash: str
    format: str = B1_EXACT_ADMISSION_FORMAT

    def __post_init__(self) -> None:
        if self.format != B1_EXACT_ADMISSION_FORMAT:
            raise ValueError("unsupported B1 exact-admission format")
        for field in (
            "catalog_freeze_hash",
            "initial_library_version",
            "initial_library_state_hash",
            "task_population_hash",
            "exact_h0_admission_hash",
            "generate_authority_hash",
            "authoring_authority_hash",
        ):
            _sha256(getattr(self, field), field=field)
        if not isinstance(self.tokenizer_artifact, TokenizerArtifactIdentity):
            raise TypeError("B1 admission requires a tokenizer artifact")
        _positive_int(self.maximum_h0_tokens, field="maximum_h0_tokens")
        _positive_int(self.task_wrapper_reserve_tokens, field="task_wrapper_reserve_tokens")
        _positive_int(self.task_population_count, field="task_population_count")
        _positive_int(self.maximum_exact_h0_tokens, field="maximum_exact_h0_tokens")
        _positive_int(self.maximum_task_wrapper_tokens, field="maximum_task_wrapper_tokens")
        _non_negative_int(
            self.maximum_initial_retrieved_skills,
            field="maximum_initial_retrieved_skills",
        )
        _positive_int(
            self.maximum_applicable_skills_per_task,
            field="maximum_applicable_skills_per_task",
        )
        _positive_int(
            self.maximum_authored_skill_full_block_tokens,
            field="maximum_authored_skill_full_block_tokens",
        )
        _positive_int(
            self.maximum_future_complete_h0_tokens,
            field="maximum_future_complete_h0_tokens",
        )
        _positive_int(
            self.generate_authority_group_count,
            field="generate_authority_group_count",
        )
        if self.maximum_generate_authority_matches_per_task_per_cycle != 1:
            raise ValueError("B1 requires exactly one matching Generate authority per task")
        if self.maximum_exact_h0_tokens > self.maximum_h0_tokens:
            raise ValueError("an exact H0 exceeds its formal token limit")
        if self.maximum_applicable_skills_per_task != MAXIMUM_APPLICABLE_SKILL_POSITION:
            raise ValueError("B1 applicable-skill bound differs from the production cap")
        if self.maximum_authored_skill_full_block_tokens != AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP:
            raise ValueError("B1 authored-skill block bound differs from the production cap")
        expected_future_h0 = self.maximum_task_wrapper_tokens + (
            self.maximum_applicable_skills_per_task * self.maximum_authored_skill_full_block_tokens
        )
        if self.maximum_future_complete_h0_tokens != expected_future_h0:
            raise ValueError("B1 future complete H0 bound is inconsistent")
        if self.maximum_future_complete_h0_tokens > self.maximum_h0_tokens:
            raise ValueError("a future complete H0 can exceed its formal token limit")
        expected_reserve = self.maximum_h0_tokens - (
            self.maximum_applicable_skills_per_task * self.maximum_authored_skill_full_block_tokens
        )
        if self.task_wrapper_reserve_tokens != expected_reserve:
            raise ValueError("B1 task/wrapper reserve differs from its complete H0 envelope")
        if self.maximum_task_wrapper_tokens > self.task_wrapper_reserve_tokens:
            raise ValueError("a measured task/wrapper exceeds its formal reserve")
        if tuple(item.benchmark for item in self.benchmark_counts) != ACTIVE_BENCHMARKS:
            raise ValueError("B1 admission must cover the complete benchmark order")
        if sum(item.task_count for item in self.benchmark_counts) != self.task_population_count:
            raise ValueError("B1 benchmark counts differ from the admitted population")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "authoring_authority_hash": self.authoring_authority_hash,
            "benchmark_counts": [item.to_value() for item in self.benchmark_counts],
            "catalog_freeze_hash": self.catalog_freeze_hash,
            "exact_h0_admission_hash": self.exact_h0_admission_hash,
            "format": self.format,
            "generate_authority_group_count": self.generate_authority_group_count,
            "generate_authority_hash": self.generate_authority_hash,
            "initial_library_state_hash": self.initial_library_state_hash,
            "initial_library_version": self.initial_library_version,
            "maximum_exact_h0_tokens": self.maximum_exact_h0_tokens,
            "maximum_generate_authority_matches_per_task_per_cycle": (
                self.maximum_generate_authority_matches_per_task_per_cycle
            ),
            "maximum_h0_tokens": self.maximum_h0_tokens,
            "maximum_applicable_skills_per_task": self.maximum_applicable_skills_per_task,
            "maximum_authored_skill_full_block_tokens": (
                self.maximum_authored_skill_full_block_tokens
            ),
            "maximum_future_complete_h0_tokens": self.maximum_future_complete_h0_tokens,
            "maximum_initial_retrieved_skills": self.maximum_initial_retrieved_skills,
            "maximum_task_wrapper_tokens": self.maximum_task_wrapper_tokens,
            "task_population_count": self.task_population_count,
            "task_population_hash": self.task_population_hash,
            "task_wrapper_reserve_tokens": self.task_wrapper_reserve_tokens,
            "tokenizer_artifact": self.tokenizer_artifact.to_value(),
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @classmethod
    def from_value(cls, value: object) -> B1ExactAdmissionReport:
        data = normalize_json(value)
        fields = {
            "authoring_authority_hash",
            "benchmark_counts",
            "catalog_freeze_hash",
            "exact_h0_admission_hash",
            "format",
            "generate_authority_group_count",
            "generate_authority_hash",
            "initial_library_state_hash",
            "initial_library_version",
            "maximum_exact_h0_tokens",
            "maximum_applicable_skills_per_task",
            "maximum_authored_skill_full_block_tokens",
            "maximum_future_complete_h0_tokens",
            "maximum_generate_authority_matches_per_task_per_cycle",
            "maximum_h0_tokens",
            "maximum_initial_retrieved_skills",
            "maximum_task_wrapper_tokens",
            "task_population_count",
            "task_population_hash",
            "task_wrapper_reserve_tokens",
            "tokenizer_artifact",
        }
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("B1 exact-admission report has incompatible fields")
        raw_counts = data["benchmark_counts"]
        if not isinstance(raw_counts, list):
            raise TypeError("B1 benchmark counts must be an array")
        text_fields = (
            "authoring_authority_hash",
            "catalog_freeze_hash",
            "exact_h0_admission_hash",
            "format",
            "generate_authority_hash",
            "initial_library_state_hash",
            "initial_library_version",
            "task_population_hash",
        )
        if any(type(data[field]) is not str for field in text_fields):
            raise TypeError("B1 exact-admission identity fields must be text")
        integer_fields = (
            "generate_authority_group_count",
            "maximum_applicable_skills_per_task",
            "maximum_authored_skill_full_block_tokens",
            "maximum_exact_h0_tokens",
            "maximum_future_complete_h0_tokens",
            "maximum_generate_authority_matches_per_task_per_cycle",
            "maximum_h0_tokens",
            "maximum_initial_retrieved_skills",
            "maximum_task_wrapper_tokens",
            "task_population_count",
            "task_wrapper_reserve_tokens",
        )
        if any(type(data[field]) is not int for field in integer_fields):
            raise TypeError("B1 exact-admission count fields must be integers")
        return cls(
            catalog_freeze_hash=data["catalog_freeze_hash"],
            tokenizer_artifact=TokenizerArtifactIdentity.from_value(data["tokenizer_artifact"]),
            initial_library_version=data["initial_library_version"],
            initial_library_state_hash=data["initial_library_state_hash"],
            maximum_h0_tokens=data["maximum_h0_tokens"],
            task_wrapper_reserve_tokens=data["task_wrapper_reserve_tokens"],
            task_population_count=data["task_population_count"],
            task_population_hash=data["task_population_hash"],
            benchmark_counts=tuple(
                B1BenchmarkPopulationCount.from_value(item) for item in raw_counts
            ),
            exact_h0_admission_hash=data["exact_h0_admission_hash"],
            maximum_exact_h0_tokens=data["maximum_exact_h0_tokens"],
            maximum_task_wrapper_tokens=data["maximum_task_wrapper_tokens"],
            maximum_initial_retrieved_skills=data["maximum_initial_retrieved_skills"],
            maximum_applicable_skills_per_task=data["maximum_applicable_skills_per_task"],
            maximum_authored_skill_full_block_tokens=data[
                "maximum_authored_skill_full_block_tokens"
            ],
            maximum_future_complete_h0_tokens=data["maximum_future_complete_h0_tokens"],
            generate_authority_hash=data["generate_authority_hash"],
            generate_authority_group_count=data["generate_authority_group_count"],
            maximum_generate_authority_matches_per_task_per_cycle=data[
                "maximum_generate_authority_matches_per_task_per_cycle"
            ],
            authoring_authority_hash=data["authoring_authority_hash"],
            format=data["format"],
        )


def _ordered_complete_tasks(
    catalog: PrivateBenchmarkCatalog,
) -> tuple[tuple[Benchmark, RolloutTask], ...]:
    if not isinstance(catalog, PrivateBenchmarkCatalog):
        raise TypeError("B1 admission requires PrivateBenchmarkCatalog")
    if tuple(item.benchmark for item in catalog.workloads) != ACTIVE_BENCHMARKS:
        raise ValueError("B1 admission requires the complete ordered benchmark catalog")
    return tuple(
        (workload.benchmark, task)
        for workload in catalog.workloads
        for task in sorted(workload.tasks, key=lambda item: item.task_id)
    )


def _require_model_visible_tools(task: RolloutTask) -> None:
    """Bind the Generate tool surface to bytes that actually enter ``H_0``."""

    context = task.public_context
    if not isinstance(context, dict):
        raise ValueError("formal task public_context must be an object")
    raw_tools = context.get("tools")
    if not task.available_tools:
        if raw_tools is not None and (not isinstance(raw_tools, dict) or raw_tools):
            raise ValueError("tool-free task exposes a different model-visible tool surface")
        return
    if not isinstance(raw_tools, dict):
        raise ValueError("task tools are absent from its model-visible public context")
    if tuple(sorted(raw_tools)) != task.available_tools:
        raise ValueError("task available_tools differ from the model-visible tool surface")


def _admit_loaded_catalog(
    *,
    catalog: PrivateBenchmarkCatalog,
    catalog_freeze_hash: str,
    tokenizer: _AdmissionTokenizer,
    seed_documents: tuple[SkillDocument, ...],
    authoring_authority: SkillAuthoringAuthority,
    maximum_h0_tokens: int,
    task_wrapper_reserve_tokens: int,
) -> B1ExactAdmissionReport:
    _sha256(catalog_freeze_hash, field="catalog_freeze_hash")
    _positive_int(maximum_h0_tokens, field="maximum_h0_tokens")
    _positive_int(task_wrapper_reserve_tokens, field="task_wrapper_reserve_tokens")
    if task_wrapper_reserve_tokens >= maximum_h0_tokens:
        raise ValueError("task/wrapper reserve must leave a positive future-skill envelope")
    expected_reserve = maximum_h0_tokens - (
        MAXIMUM_APPLICABLE_SKILL_POSITION * AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP
    )
    if task_wrapper_reserve_tokens != expected_reserve:
        raise ValueError("task/wrapper reserve differs from the complete H0 envelope")
    if not isinstance(authoring_authority, SkillAuthoringAuthority):
        raise TypeError("B1 admission requires SkillAuthoringAuthority")
    if not isinstance(seed_documents, tuple) or not seed_documents:
        raise ValueError("B1 admission requires the complete initial skill library")
    if any(not isinstance(item, SkillDocument) for item in seed_documents):
        raise TypeError("B1 seed_documents contain an invalid skill")
    for document in seed_documents:
        seed_context = FullRetrievedSkillContext(
            metadata=SkillMetadata.from_document(document),
            content=model_visible_skill_content(document),
            inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
        )
        if (
            maximum_retrieved_skill_block_token_count(
                seed_context,
                maximum_position=MAXIMUM_APPLICABLE_SKILL_POSITION,
                tokenizer=tokenizer,
            )
            > AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP
        ):
            raise ValueError("an initial skill exceeds the complete rendered-block cap")

    ordered = _ordered_complete_tasks(catalog)
    library_state = SkillLibraryState.from_seed_documents(seed_documents)
    library = SkillLibrary(library_state)
    retriever = TaskConditionedSkillRetriever(library=library)
    assembler = CanonicalInitialContextAssembler(maximum_h0_tokens=maximum_h0_tokens)

    h0_entries: list[dict[str, JsonValue]] = []
    task_values: list[JsonValue] = []
    groups: dict[tuple[str, str, tuple[str, ...]], dict[str, JsonValue]] = {}
    maximum_h0 = 0
    maximum_wrapper = 0
    maximum_retrieved = 0
    counts: dict[Benchmark, int] = dict.fromkeys(Benchmark, 0)

    for benchmark, task in ordered:
        counts[benchmark] += 1
        task_values.append(task.to_value())
        _require_model_visible_tools(task)
        if task.task_family not in authoring_authority.allowed_task_families:
            raise ValueError("formal task family is outside the Generate authority")
        if not set(task.available_tools) <= set(authoring_authority.allowed_tools):
            raise ValueError("formal task tools are outside the Generate authority")

        group_key = (task.task_family, task.context_id, task.available_tools)
        group_value: dict[str, JsonValue] = {
            "available_tools": list(task.available_tools),
            "context_id": task.context_id,
            "input_schema_id": authoring_authority.input_schema_id,
            "license_id": authoring_authority.license_id,
            "output_schema_id": authoring_authority.output_schema_id,
            "task_family": task.task_family,
        }
        existing = groups.setdefault(group_key, group_value)
        if existing != group_value:  # pragma: no cover - one frozen authority supplies all values
            raise ValueError("one Generate group resolves to multiple authorities")

        retrieved = retriever.retrieve(task)
        assembled = assembler.assemble(
            task=task,
            retrieved_skills=retrieved,
            active_skill_ids=library.active_skill_ids,
            library_version=library.current_version,
            tokenizer=tokenizer,
        )
        wrapper = assembler.assemble(
            task=task,
            retrieved_skills=(),
            active_skill_ids=library.active_skill_ids,
            library_version=library.current_version,
            tokenizer=tokenizer,
        )
        wrapper_tokens = wrapper.contract.assembled_token_count
        h0_tokens = assembled.contract.assembled_token_count
        maximum_h0 = max(maximum_h0, h0_tokens)
        maximum_wrapper = max(maximum_wrapper, wrapper_tokens)
        maximum_retrieved = max(maximum_retrieved, len(retrieved))
        h0_entries.append(
            {
                "assembled_hash": assembled.contract.assembled_hash,
                "assembled_token_count": h0_tokens,
                "retrieved_skill_ids": list(assembled.contract.retrieved_skill_ids),
                "task_id": task.task_id,
                "task_wrapper_hash": wrapper.contract.assembled_hash,
                "task_wrapper_token_count": wrapper_tokens,
            }
        )

    group_keys = tuple(sorted(groups))
    maximum_group_matches = 0
    for _, task in ordered:
        matching = sum(
            1
            for family, context, required_tools in group_keys
            if family == task.task_family
            and context == task.context_id
            and required_tools == task.available_tools
        )
        if matching != 1:
            raise ValueError("formal task does not have exactly one matching Generate authority")
        maximum_group_matches = max(maximum_group_matches, matching)

    benchmark_counts = tuple(
        B1BenchmarkPopulationCount(benchmark=benchmark, task_count=counts[benchmark])
        for benchmark in ACTIVE_BENCHMARKS
    )
    return B1ExactAdmissionReport(
        catalog_freeze_hash=catalog_freeze_hash,
        tokenizer_artifact=tokenizer.artifact_identity,
        initial_library_version=library.current_version,
        initial_library_state_hash=library.state.state_hash,
        maximum_h0_tokens=maximum_h0_tokens,
        task_wrapper_reserve_tokens=task_wrapper_reserve_tokens,
        task_population_count=len(ordered),
        task_population_hash=stable_hash(task_values),
        benchmark_counts=benchmark_counts,
        exact_h0_admission_hash=stable_hash(h0_entries),
        maximum_exact_h0_tokens=maximum_h0,
        maximum_task_wrapper_tokens=maximum_wrapper,
        maximum_initial_retrieved_skills=maximum_retrieved,
        maximum_applicable_skills_per_task=MAXIMUM_APPLICABLE_SKILL_POSITION,
        maximum_authored_skill_full_block_tokens=AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP,
        maximum_future_complete_h0_tokens=(
            maximum_wrapper
            + (MAXIMUM_APPLICABLE_SKILL_POSITION * AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP)
        ),
        generate_authority_hash=stable_hash([groups[key] for key in group_keys]),
        generate_authority_group_count=len(groups),
        maximum_generate_authority_matches_per_task_per_cycle=maximum_group_matches,
        authoring_authority_hash=stable_hash(authoring_authority.to_value()),
    )


def build_b1_exact_admission(
    *,
    catalog: PrivateBenchmarkCatalog,
    catalog_freeze_hash: str,
    backbone: QwenDeploymentConfig,
    tokenizer_artifact: TokenizerArtifactIdentity,
    seed_documents: tuple[SkillDocument, ...],
    authoring_authority: SkillAuthoringAuthority,
    maximum_h0_tokens: int,
    task_wrapper_reserve_tokens: int,
) -> B1ExactAdmissionReport:
    """Run exact admission with only the tokenizer named by the pinned Qwen config."""

    from skillev.policy import QwenTokenizerAdapter

    tokenizer = QwenTokenizerAdapter.from_config(backbone)
    if tokenizer.artifact_identity != tokenizer_artifact:
        raise ValueError("measured tokenizer differs from the planned tokenizer artifact")
    return _admit_loaded_catalog(
        catalog=catalog,
        catalog_freeze_hash=catalog_freeze_hash,
        tokenizer=tokenizer,
        seed_documents=seed_documents,
        authoring_authority=authoring_authority,
        maximum_h0_tokens=maximum_h0_tokens,
        task_wrapper_reserve_tokens=task_wrapper_reserve_tokens,
    )


__all__ = [
    "B1_EXACT_ADMISSION_FORMAT",
    "B1BenchmarkPopulationCount",
    "B1ExactAdmissionReport",
    "build_b1_exact_admission",
]
