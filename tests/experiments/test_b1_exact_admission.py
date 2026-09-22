from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pytest
from skillev_private.benchmarks.catalog import (
    PrivateBenchmarkCatalog,
    PrivateBenchmarkWorkload,
)
from skillev_private.experiments.b1_admission import (
    B1ExactAdmissionReport,
    _admit_loaded_catalog,
    build_b1_exact_admission,
)

from skillev.contracts import JsonValue, stable_hash
from skillev.evolution import SkillAuthoringAuthority
from skillev.experiments import ACTIVE_BENCHMARKS, Benchmark
from skillev.experiments.evolution_preflight import planned_seed_documents
from skillev.policy import (
    PublicTokenizerKind,
    QwenBackboneConfig,
    QwenTokenizerAdapter,
    TokenizerArtifactIdentity,
)
from skillev.rollout import RolloutTask
from tests.v3_helpers import make_skill_document

if TYPE_CHECKING:
    from skillev.training import RolloutSessionBundle


@dataclass(frozen=True, slots=True)
class _UnusedSessionFactory:
    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        raise AssertionError(f"B1 admission opened an environment for {task.task_id}")


@dataclass(frozen=True, slots=True)
class _Tokenizer:
    artifact_identity: TokenizerArtifactIdentity

    @property
    def tokenizer_id(self) -> str:
        return self.artifact_identity.tokenizer_id

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return bytes(token_ids).decode("utf-8")


def _artifact(label: str = "planned") -> TokenizerArtifactIdentity:
    return TokenizerArtifactIdentity.create(
        kind=PublicTokenizerKind.QWEN,
        tokenizer_id="qwen-fixture",
        revision="fixture-revision",
        backend_serialization_hash=stable_hash({"backend": label}),
        tokenizer_config_hash=stable_hash({"config": label}),
        chat_template_hash=stable_hash({"chat": label}),
        special_tokens_hash=stable_hash({"special": label}),
        added_tokens_hash=stable_hash({"added": label}),
        transformers_version="fixture",
        tokenizers_version="fixture",
    )


def _task(
    benchmark: Benchmark,
    *,
    suffix: str = "one",
    context: str | None = None,
    tools: tuple[str, ...] = (),
    public_tools: tuple[str, ...] | None = None,
) -> RolloutTask:
    visible_tools = tools if public_tools is None else public_tools
    public_context: dict[str, JsonValue] = {"benchmark_id": benchmark.value}
    if visible_tools:
        public_context["tools"] = {tool: {"arguments": []} for tool in visible_tools}
    return RolloutTask(
        task_id=f"{benchmark.value}-{suffix}",
        environment_id=f"benchmark:{benchmark.value}",
        task_family=f"{benchmark.value}/fixture",
        context_id=context or f"{benchmark.value}:fixture",
        query="Answer-free synthetic admission query.",
        available_tools=tools,
        public_context=public_context,
    )


def _catalog(
    *,
    replacement: tuple[Benchmark, tuple[RolloutTask, ...]] | None = None,
) -> PrivateBenchmarkCatalog:
    return PrivateBenchmarkCatalog(
        workloads=tuple(
            PrivateBenchmarkWorkload(
                benchmark=benchmark,
                tasks=(
                    replacement[1]
                    if replacement is not None and replacement[0] is benchmark
                    else (_task(benchmark),)
                ),
                session_factory=_UnusedSessionFactory(),
            )
            for benchmark in ACTIVE_BENCHMARKS
        )
    )


def _authority(catalog: PrivateBenchmarkCatalog) -> SkillAuthoringAuthority:
    tasks = tuple(task for workload in catalog.workloads for task in workload.tasks)
    return SkillAuthoringAuthority(
        input_schema_id="formal-input@1",
        output_schema_id="formal-output@1",
        license_id="project-owned",
        allowed_task_families=tuple(sorted({task.task_family for task in tasks})),
        allowed_tools=tuple(sorted({tool for task in tasks for tool in task.available_tools})),
    )


def _admit(catalog: PrivateBenchmarkCatalog) -> B1ExactAdmissionReport:
    return _admit_loaded_catalog(
        catalog=catalog,
        catalog_freeze_hash=stable_hash({"catalog": "fixture"}),
        tokenizer=_Tokenizer(_artifact()),
        seed_documents=planned_seed_documents(),
        authoring_authority=_authority(catalog),
        maximum_h0_tokens=100_000,
        task_wrapper_reserve_tokens=83_000,
    )


def test_b1_exact_admission_covers_complete_catalog_without_opening_sessions() -> None:
    catalog = _catalog()

    report = _admit(catalog)

    assert report.task_population_count == len(ACTIVE_BENCHMARKS)
    assert tuple(item.benchmark for item in report.benchmark_counts) == ACTIVE_BENCHMARKS
    assert all(item.task_count == 1 for item in report.benchmark_counts)
    assert report.generate_authority_group_count == len(ACTIVE_BENCHMARKS)
    assert report.maximum_generate_authority_matches_per_task_per_cycle == 1
    assert report.maximum_exact_h0_tokens <= report.maximum_h0_tokens
    assert report.maximum_task_wrapper_tokens <= report.task_wrapper_reserve_tokens
    assert report.maximum_initial_retrieved_skills == len(planned_seed_documents())
    assert report.maximum_applicable_skills_per_task == 5
    assert report.maximum_authored_skill_full_block_tokens == 3_400
    assert report.maximum_future_complete_h0_tokens == (
        report.maximum_task_wrapper_tokens + 5 * 3_400
    )
    assert report.maximum_future_complete_h0_tokens <= report.maximum_h0_tokens
    assert B1ExactAdmissionReport.from_value(report.to_value()) == report


def test_b1_admission_rejects_tools_absent_from_model_visible_context() -> None:
    benchmark = next(iter(Benchmark))
    task = _task(benchmark, tools=("search",), public_tools=())
    catalog = _catalog(replacement=(benchmark, (task,)))

    with pytest.raises(ValueError):
        _admit(catalog)


def test_b1_admission_distinguishes_generate_groups_by_exact_tool_surface() -> None:
    benchmark = next(iter(Benchmark))
    context = "shared-context"
    catalog = _catalog(
        replacement=(
            benchmark,
            (
                _task(benchmark, suffix="narrow", context=context, tools=("read",)),
                _task(
                    benchmark,
                    suffix="wide",
                    context=context,
                    tools=("read", "search"),
                ),
            ),
        )
    )

    report = _admit(catalog)

    assert report.generate_authority_group_count == len(ACTIVE_BENCHMARKS) + 1
    assert report.maximum_generate_authority_matches_per_task_per_cycle == 1


def test_b1_admission_rejects_a_task_outside_authoring_authority() -> None:
    catalog = _catalog()
    authority = _authority(catalog)

    with pytest.raises(ValueError):
        _admit_loaded_catalog(
            catalog=catalog,
            catalog_freeze_hash=stable_hash({"catalog": "fixture"}),
            tokenizer=_Tokenizer(_artifact()),
            seed_documents=planned_seed_documents(),
            authoring_authority=replace(authority, allowed_task_families=("other/fixture",)),
            maximum_h0_tokens=100_000,
            task_wrapper_reserve_tokens=83_000,
        )


def test_b1_admission_requires_exact_reserve_and_rejects_any_h0_overflow() -> None:
    catalog = _catalog()

    with pytest.raises(ValueError):
        _admit_loaded_catalog(
            catalog=catalog,
            catalog_freeze_hash=stable_hash({"catalog": "fixture"}),
            tokenizer=_Tokenizer(_artifact()),
            seed_documents=planned_seed_documents(),
            authoring_authority=_authority(catalog),
            maximum_h0_tokens=100_000,
            task_wrapper_reserve_tokens=1,
        )

    with pytest.raises(ValueError):
        _admit_loaded_catalog(
            catalog=catalog,
            catalog_freeze_hash=stable_hash({"catalog": "fixture"}),
            tokenizer=_Tokenizer(_artifact()),
            seed_documents=planned_seed_documents(),
            authoring_authority=_authority(catalog),
            maximum_h0_tokens=100,
            task_wrapper_reserve_tokens=50,
        )

    with pytest.raises(ValueError):
        _admit_loaded_catalog(
            catalog=catalog,
            catalog_freeze_hash=stable_hash({"catalog": "fixture"}),
            tokenizer=_Tokenizer(_artifact()),
            seed_documents=planned_seed_documents(),
            authoring_authority=_authority(catalog),
            maximum_h0_tokens=17_100,
            task_wrapper_reserve_tokens=100,
        )


def test_b1_admission_exact_tokenizes_every_initial_skill_complete_block() -> None:
    catalog = _catalog()
    oversized = make_skill_document(
        "oversized-seed",
        instructions="X" * 4_000,
        task_family="*",
    )

    with pytest.raises(ValueError):
        _admit_loaded_catalog(
            catalog=catalog,
            catalog_freeze_hash=stable_hash({"catalog": "fixture"}),
            tokenizer=_Tokenizer(_artifact()),
            seed_documents=(*planned_seed_documents(), oversized),
            authoring_authority=_authority(catalog),
            maximum_h0_tokens=100_000,
            task_wrapper_reserve_tokens=83_000,
        )


def test_b1_formal_entrypoint_loads_only_the_pinned_qwen_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    planned = _artifact()
    tokenizer = _Tokenizer(planned)
    config = QwenBackboneConfig(
        base_model_path="/private/pinned-qwen",
        revision=planned.revision,
        tokenizer_id=planned.tokenizer_id,
        tokenizer_content_hash=planned.content_hash,
        hidden_size=8,
        device="cpu",
        torch_dtype="float32",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=("q_proj",),
        z_hidden_width=4,
        eos_token_ids=(1,),
    )
    loaded: list[QwenBackboneConfig] = []

    def fake_from_config(cls: type[QwenTokenizerAdapter], source: QwenBackboneConfig) -> _Tokenizer:
        del cls
        loaded.append(source)
        return tokenizer

    monkeypatch.setattr(QwenTokenizerAdapter, "from_config", classmethod(fake_from_config))

    report = build_b1_exact_admission(
        catalog=catalog,
        catalog_freeze_hash=stable_hash({"catalog": "fixture"}),
        backbone=config,
        tokenizer_artifact=planned,
        seed_documents=planned_seed_documents(),
        authoring_authority=_authority(catalog),
        maximum_h0_tokens=100_000,
        task_wrapper_reserve_tokens=83_000,
    )

    assert loaded == [config]
    assert report.tokenizer_artifact == planned

    with pytest.raises(ValueError):
        build_b1_exact_admission(
            catalog=catalog,
            catalog_freeze_hash=stable_hash({"catalog": "fixture"}),
            backbone=config,
            tokenizer_artifact=_artifact("other"),
            seed_documents=planned_seed_documents(),
            authoring_authority=_authority(catalog),
            maximum_h0_tokens=100_000,
            task_wrapper_reserve_tokens=83_000,
        )
