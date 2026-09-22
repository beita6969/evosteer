"""Private, answer-separated population loaders."""

from __future__ import annotations

import csv
import gzip
import json
import random
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from skillev.evaluation.direct_baseline.config import DirectBenchmark, DirectReferenceProtocol
from skillev.evaluation.direct_baseline.parsing import PARSER_REGISTRY
from skillev.evaluation.direct_baseline.prompts import (
    render_interactive_messages,
    render_static_messages,
)
from skillev.evaluation.direct_baseline.runner import DirectTask
from skillev_private.benchmarks.converters import convert_mind2web_row
from skillev_private.benchmarks.hotpot_context_audit import audit_hotpot_context
from skillev_private.benchmarks.math_answers import last_boxed_answer
from skillev_private.benchmarks.mind2web_scores import load_mind2web_candidate_rankings

from .manifests import (
    PopulationManifest,
    select_manifest_rows,
    validate_manifest_task_ids,
)
from .skillflow_iid import (
    HotpotRenderMode,
    SkillFlowIIDRecord,
    parse_skillflow_iid_record,
    render_hotpot_input,
    trivia_aliases_from_record,
)


@dataclass(frozen=True, slots=True)
class PrivateDirectCase:
    """A model-visible task paired with verifier-only material."""

    public_task: DirectTask
    target: object
    private_metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class PrivateQATarget:
    accepted_answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrivateMathTarget:
    boxed_answer: str


@dataclass(frozen=True, slots=True)
class PrivateHumanEvalTarget:
    prompt: str
    test: str
    entry_point: str


_SOURCE_TO_BENCHMARK = {
    "HotpotQA": DirectBenchmark.HOTPOT_QA,
    "TriviaQA": DirectBenchmark.TRIVIA_QA,
    "AIME 2026": DirectBenchmark.AIME_2026,
    "MedQA": DirectBenchmark.MED_QA,
    "SWE-bench": DirectBenchmark.SWE_BENCH,
}


def load_skillflow_iid_cases(
    path: Path,
    *,
    protocol: DirectReferenceProtocol,
    include: frozenset[DirectBenchmark] | None = None,
    manifests: Mapping[DirectBenchmark, PopulationManifest] | None = None,
) -> tuple[PrivateDirectCase, ...]:
    """Load the released 798-record IID panel without leaking targets."""

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError("SkillFlow IID population must be a JSON array")
    source_indices: dict[DirectBenchmark, int] = {}
    cases: list[PrivateDirectCase] = []
    for source_position, raw_value in enumerate(value):
        record = parse_skillflow_iid_record(raw_value, source_position=source_position)
        raw = record.raw
        extra = record.extra
        source = record.source
        if source in {"WebShop", "ALFWorld"}:
            continue
        try:
            benchmark = _SOURCE_TO_BENCHMARK[source]
        except KeyError as exc:
            raise ValueError(f"unsupported SkillFlow IID source {source}") from exc
        if include is not None and benchmark not in include:
            continue
        spec = protocol.benchmark(benchmark)
        question = record.question
        target: object = record.answer
        if benchmark is DirectBenchmark.HOTPOT_QA:
            mode = (
                HotpotRenderMode.RAW_TEN_PASSAGE_CONTEXT
                if spec.prompt_profile
                in {
                    "hotpotqa-full-context@1",
                    "hotpotqa-raw-ten-passage@2",
                }
                else HotpotRenderMode.RELEASED_VISIBLE_QUESTION
            )
            question = render_hotpot_input(record, mode=mode)
        index = source_indices.get(benchmark, 0)
        source_indices[benchmark] = index + 1
        task_id = f"skillflow-iid-v3:{benchmark.value}:{index:03d}"
        source_identity = _iid_source_identity(record, source_position)
        metadata: dict[str, object] = {
            "source": source,
            "source_identity": source_identity,
            "source_position": source_position,
            "source_index": index,
            "source_split": str(extra.get("split", "released-iid")),
            "source_format": "skillflow-released-iid-v3",
        }
        if benchmark is DirectBenchmark.HOTPOT_QA:
            context = raw.get("context")
            if not isinstance(context, list) or any(not isinstance(item, str) for item in context):
                raise ValueError("HotpotQA context differs from its rendering contract")
            audit = audit_hotpot_context(
                task_id=task_id,
                rendered_question=question,
                public_context=tuple(cast(list[str], context)),
                supporting_facts=raw.get("supporting_facts"),
            )
            if not audit.structurally_valid:
                raise ValueError("HotpotQA public context fails structural integrity")
            if (
                audit.supporting_fact_count is not None
                and audit.supporting_fact_present_count != audit.supporting_fact_count
            ):
                raise ValueError("HotpotQA public context omits a declared supporting fact")
            metadata.update(
                {
                    "context_passage_count": audit.passage_count,
                    "context_evidence_status": audit.evidence_status.value,
                    "supporting_fact_count": audit.supporting_fact_count,
                    "supporting_fact_present_count": audit.supporting_fact_present_count,
                }
            )
        if benchmark is DirectBenchmark.TRIVIA_QA:
            aliases = trivia_aliases_from_record(record)
            target = PrivateQATarget(aliases.aliases)
            metadata.update(
                {
                    "alias_count": len(aliases.aliases),
                    "alias_source": aliases.source,
                    "aliases_potentially_truncated": aliases.potentially_truncated,
                }
            )
        if benchmark is DirectBenchmark.AIME_2026:
            try:
                numeric_target = int(cast(str, target))
            except ValueError as exc:
                raise ValueError("AIME target must be an integer") from exc
            if numeric_target < 0 or numeric_target > 999:
                raise ValueError("AIME target must lie in [0, 999]")
            target = str(numeric_target)
        if benchmark is DirectBenchmark.MED_QA:
            correct_option = extra.get("correct_option")
            if type(correct_option) is not str or correct_option not in {"A", "B", "C", "D"}:
                raise ValueError("MedQA record lacks a valid correct option")
            target = correct_option
        if benchmark is DirectBenchmark.SWE_BENCH:
            instance_id = extra.get("instance_id")
            if type(instance_id) is not str or not instance_id.strip():
                raise ValueError("SWE-bench record lacks instance_id")
            repo = extra.get("repo")
            base_commit = extra.get("base_commit")
            if type(repo) is not str or not repo.strip():
                raise ValueError("SWE-bench record lacks repo")
            if type(base_commit) is not str or not base_commit.strip():
                raise ValueError("SWE-bench record lacks base_commit")
            problem_statement = question
            code_files = raw.get("code_files")
            if not isinstance(code_files, dict) or not code_files:
                raise ValueError("SWE-bench record lacks public repository source excerpts")
            excerpts: list[str] = []
            for code_path, source in code_files.items():
                if type(code_path) is not str or not code_path.strip() or type(source) is not str:
                    raise ValueError("SWE-bench source excerpts have incompatible fields")
                excerpts.append(f"### {code_path}\n```\n{source}\n```")
            question = (
                f"{question}\n\nThe following files are from the pinned repository snapshot:\n\n"
                + "\n\n".join(excerpts)
            )
            metadata.update(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": base_commit,
                    "problem_statement": problem_statement,
                }
            )
            target = instance_id
        parser = PARSER_REGISTRY[spec.parser_profile]
        cases.append(
            PrivateDirectCase(
                public_task=DirectTask(
                    task_id=task_id,
                    benchmark=benchmark,
                    messages=render_static_messages(spec.prompt_profile, question),
                    profile=protocol.profile(spec.decoding_profile),
                    parser=parser,
                    prompt_profile_id=spec.prompt_profile,
                    parser_profile_id=spec.parser_profile,
                    population_id=spec.population,
                    run_seed=spec.seed_aggregation.seeds[0],
                ),
                target=target,
                private_metadata=metadata,
            )
        )
    expected = {
        DirectBenchmark.HOTPOT_QA: 128,
        DirectBenchmark.TRIVIA_QA: 128,
        DirectBenchmark.AIME_2026: 30,
        DirectBenchmark.MED_QA: 128,
        DirectBenchmark.SWE_BENCH: 128,
    }
    if manifests is not None:
        manifest_selected: list[PrivateDirectCase] = []
        for benchmark in expected:
            if include is not None and benchmark not in include:
                continue
            try:
                manifest = manifests[benchmark]
            except KeyError as exc:
                raise ValueError(f"missing frozen manifest for {benchmark.value}") from exc
            candidates = {
                cast(str, case.private_metadata["source_identity"]): case
                for case in cases
                if case.public_task.benchmark is benchmark
            }
            for entry in manifest.entries:
                try:
                    manifest_selected.append(candidates[entry.source_identity])
                except KeyError as exc:
                    raise ValueError(
                        f"manifest source identity missing: {entry.source_identity}"
                    ) from exc
        cases = manifest_selected
    for benchmark, count in expected.items():
        if include is None or benchmark in include:
            actual = sum(case.public_task.benchmark is benchmark for case in cases)
            if actual != count:
                raise ValueError(f"{benchmark.value} panel has {actual} records, expected {count}")
            if manifests is not None:
                try:
                    manifest = manifests[benchmark]
                except KeyError as exc:
                    raise ValueError(f"missing frozen manifest for {benchmark.value}") from exc
                benchmark_tasks = tuple(
                    case.public_task.task_id
                    for case in cases
                    if case.public_task.benchmark is benchmark
                )
                benchmark_sources = tuple(
                    cast(str, case.private_metadata["source_identity"])
                    for case in cases
                    if case.public_task.benchmark is benchmark
                )
                validate_manifest_task_ids(
                    manifest,
                    population_id=protocol.benchmark(benchmark).population,
                    task_ids=benchmark_tasks,
                    source_identities=benchmark_sources,
                    dataset_revision=protocol.benchmark(benchmark).dataset_revision,
                    selection_rule=protocol.benchmark(benchmark).selection_rule,
                )
    return tuple(cases)


def _iid_source_identity(parsed: SkillFlowIIDRecord, source_position: int) -> str:
    for key in ("instance_id", "question_id", "problem_id", "id"):
        value = parsed.extra.get(key)
        if type(value) is str and value.strip():
            return f"{parsed.source}:{value}"
    return f"{parsed.source}:source-position:{source_position}"


def _sample(values: list[object], *, count: int = 128) -> list[object]:
    ordered = list(values)
    random.Random(42).shuffle(ordered)  # noqa: S311 -- frozen benchmark sampling
    return ordered[: min(count, len(ordered))]


def load_nq_open_cases(
    path: Path, *, protocol: DirectReferenceProtocol
) -> tuple[PrivateDirectCase, ...]:
    rows: list[object] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    cases: list[PrivateDirectCase] = []
    for index, raw in enumerate(_sample(rows)):
        if not isinstance(raw, dict):
            raise ValueError("NQ-Open row must be an object")
        question, answers = raw.get("question"), raw.get("answer")
        if type(question) is not str or not isinstance(answers, list):
            raise ValueError("NQ-Open row has incompatible fields")
        accepted = tuple(value for value in answers if type(value) is str and value.strip())
        if not accepted:
            raise ValueError("NQ-Open row requires accepted answers")
        cases.append(
            _static_case(
                benchmark=DirectBenchmark.NQ_OPEN,
                index=index,
                question=question,
                target=PrivateQATarget(accepted),
                protocol=protocol,
                metadata={"source_split": "dev"},
            )
        )
    return tuple(cases)


def load_musique_cases(
    archive: Path, *, protocol: DirectReferenceProtocol
) -> tuple[PrivateDirectCase, ...]:
    with zipfile.ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist() if name.endswith("musique_ans_v1.0_dev.jsonl")]
        if len(names) != 1:
            raise ValueError("MuSiQue archive lacks one answerable dev file")
        rows = [json.loads(line) for line in zipped.read(names[0]).decode("utf-8").splitlines()]
    cases: list[PrivateDirectCase] = []
    for index, raw in enumerate(_sample(cast(list[object], rows))):
        if not isinstance(raw, dict) or raw.get("answerable") is not True:
            raise ValueError("MuSiQue sample must be answerable")
        question, answer, aliases, paragraphs = (
            raw.get("question"),
            raw.get("answer"),
            raw.get("answer_aliases"),
            raw.get("paragraphs"),
        )
        if type(question) is not str or type(answer) is not str or not isinstance(aliases, list):
            raise ValueError("MuSiQue row has incompatible answer fields")
        if not isinstance(paragraphs, list):
            raise ValueError("MuSiQue row has incompatible paragraphs")
        rendered: list[str] = []
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                raise ValueError("MuSiQue paragraph must be an object")
            title, text = paragraph.get("title"), paragraph.get("paragraph_text")
            if type(title) is not str or type(text) is not str:
                raise ValueError("MuSiQue paragraph has incompatible fields")
            rendered.append(f"[{title}] {text}")
        alias_values = tuple(value for value in aliases if type(value) is str and value.strip())
        prompt = "Based on the following passages, answer the question.\n\n" + "\n\n".join(
            (*rendered, f"Question: {question}")
        )
        cases.append(
            _static_case(
                benchmark=DirectBenchmark.MUSIQUE,
                index=index,
                question=prompt,
                target=PrivateQATarget((answer, *alias_values)),
                protocol=protocol,
                metadata={"source_split": "validation"},
            )
        )
    return tuple(cases)


def load_math_hard_cases(
    path: Path, *, protocol: DirectReferenceProtocol
) -> tuple[PrivateDirectCase, ...]:
    rows = cast(list[object], pq.read_table(path).to_pylist())
    hard_rows = [row for row in rows if isinstance(row, dict) and row.get("level") == "Level 5"]
    cases: list[PrivateDirectCase] = []
    for index, raw in enumerate(_sample(cast(list[object], hard_rows))):
        if not isinstance(raw, dict):
            raise ValueError("MATH-Hard row must be an object")
        problem, solution, level = raw.get("problem"), raw.get("solution"), raw.get("level")
        if type(problem) is not str or type(solution) is not str or level != "Level 5":
            raise ValueError("MATH-Hard row has incompatible fields")
        cases.append(
            _static_case(
                benchmark=DirectBenchmark.MATH_HARD,
                index=index,
                question=problem,
                target=PrivateMathTarget(last_boxed_answer(solution)),
                protocol=protocol,
                metadata={"source_split": "test", "subject": raw.get("type")},
            )
        )
    return tuple(cases)


def load_humaneval_cases(
    path: Path,
    *,
    protocol: DirectReferenceProtocol,
    manifest: PopulationManifest | None = None,
) -> tuple[PrivateDirectCase, ...]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows: list[object] = [json.loads(line) for line in stream]
    identity_by_position: dict[str, int] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("task_id")) is not str:
            raise ValueError("HumanEval row lacks a task ID")
        task_id = cast(str, row["task_id"])
        if task_id in identity_by_position:
            raise ValueError("HumanEval task IDs must be unique")
        identity_by_position[task_id] = position
    selected = (
        select_manifest_rows(
            rows,
            manifest,
            source_identity=lambda row, _position: cast(
                str, cast(dict[str, object], row)["task_id"]
            ),
        )
        if manifest is not None
        else _sample(rows)
    )
    cases: list[PrivateDirectCase] = []
    for index, raw in enumerate(selected):
        if not isinstance(raw, dict):
            raise ValueError("HumanEval row must be an object")
        prompt, test, entry_point, row_task_id = (
            raw.get("prompt"),
            raw.get("test"),
            raw.get("entry_point"),
            raw.get("task_id"),
        )
        if not all(
            isinstance(value, str) and value.strip()
            for value in (prompt, test, entry_point, row_task_id)
        ):
            raise ValueError("HumanEval row has incompatible fields")
        assert isinstance(prompt, str)
        assert isinstance(test, str)
        assert isinstance(entry_point, str)
        assert isinstance(row_task_id, str)
        spec = protocol.benchmark(DirectBenchmark.HUMAN_EVAL)
        cases.append(
            PrivateDirectCase(
                public_task=DirectTask(
                    task_id=row_task_id,
                    benchmark=DirectBenchmark.HUMAN_EVAL,
                    messages=render_static_messages(spec.prompt_profile, prompt),
                    profile=protocol.profile(spec.decoding_profile),
                    parser=PARSER_REGISTRY[spec.parser_profile],
                    prompt_profile_id=spec.prompt_profile,
                    parser_profile_id=spec.parser_profile,
                    population_id=spec.population,
                    run_seed=spec.seed_aggregation.seeds[0],
                ),
                target=PrivateHumanEvalTarget(
                    prompt=prompt,
                    test=test,
                    entry_point=entry_point,
                ),
                private_metadata={
                    "source_split": "test",
                    "source_identity": row_task_id,
                    "source_position": identity_by_position[row_task_id],
                    "sample_index": index,
                },
            )
        )
    result = tuple(cases)
    if manifest is not None:
        spec = protocol.benchmark(DirectBenchmark.HUMAN_EVAL)
        validate_manifest_task_ids(
            manifest,
            population_id=spec.population,
            task_ids=tuple(case.public_task.task_id for case in result),
            source_identities=tuple(
                cast(str, case.private_metadata["source_identity"]) for case in result
            ),
            dataset_revision=spec.dataset_revision,
            selection_rule=spec.selection_rule,
        )
    return result


def load_gpqa_cases(
    path: Path, *, protocol: DirectReferenceProtocol
) -> tuple[PrivateDirectCase, ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows: list[object] = list(csv.DictReader(stream))
    cases: list[PrivateDirectCase] = []
    for index, raw in enumerate(_sample(rows)):
        if not isinstance(raw, dict):
            raise ValueError("GPQA row must be an object")
        question, answer = raw.get("Question"), raw.get("Correct Answer")
        record_id = raw.get("Record ID")
        distractors = [raw.get(f"Incorrect Answer {number}") for number in range(1, 4)]
        if (
            type(question) is not str
            or type(answer) is not str
            or type(record_id) is not str
            or not record_id.strip()
            or not all(type(value) is str for value in distractors)
        ):
            raise ValueError("GPQA row has incompatible fields")
        choices = [answer, *cast(list[str], distractors)]
        random.Random(f"gpqa:{record_id}:42").shuffle(choices)  # noqa: S311 -- frozen option order
        labels = ("A", "B", "C", "D")
        target = labels[choices.index(answer)]
        options = "\n".join(
            f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
        )
        cases.append(
            _static_case(
                benchmark=DirectBenchmark.GPQA_DIAMOND,
                index=index,
                question=f"{question}\n\n{options}",
                target=target,
                protocol=protocol,
                metadata={
                    "source_split": "test",
                    "source_identity": record_id,
                    "option_order": tuple(choices),
                },
            )
        )
    return tuple(cases)


def load_mind2web_cases(
    archive: Path,
    score_path: Path,
    *,
    protocol: DirectReferenceProtocol,
    archive_password: bytes,
) -> tuple[PrivateDirectCase, ...]:
    if not archive_password:
        raise ValueError("Mind2Web archive password is required")
    raw_steps: list[tuple[str, dict[str, object], dict[str, object], str]] = []
    with zipfile.ZipFile(archive) as zipped:
        for name in sorted(item for item in zipped.namelist() if item.endswith(".json")):
            split = name.split("/", 1)[0]
            values = json.loads(zipped.read(name, pwd=archive_password))
            if not isinstance(values, list):
                raise ValueError("Mind2Web split shard must be a JSON array")
            for row in values:
                if not isinstance(row, dict) or not isinstance(row.get("actions"), list):
                    raise ValueError("Mind2Web row has incompatible actions")
                annotation_id = row.get("annotation_id")
                if type(annotation_id) is not str:
                    raise ValueError("Mind2Web row lacks annotation_id")
                for action in row["actions"]:
                    if not isinstance(action, dict) or type(action.get("action_uid")) is not str:
                        raise ValueError("Mind2Web action lacks action_uid")
                    action_id = f"{annotation_id}_{action['action_uid']}"
                    raw_steps.append((split, row, action, action_id))
    sampled = cast(
        list[tuple[str, dict[str, object], dict[str, object], str]],
        _sample(cast(list[object], raw_steps)),
    )
    action_ids = frozenset(value[3] for value in sampled)
    rankings = load_mind2web_candidate_rankings(score_path.resolve(), action_ids=action_ids)
    cases: list[PrivateDirectCase] = []
    for split, row, action, action_id in sampled:
        actions = cast(list[object], row["actions"])
        representations = row.get("action_reprs")
        if not isinstance(representations, list) or len(representations) != len(actions):
            raise ValueError("Mind2Web action representations are misaligned")
        position = next(index for index, item in enumerate(actions) if item is action)
        trimmed = {**row, "actions": [action], "action_reprs": [representations[position]]}
        converted = convert_mind2web_row(
            trimmed,
            dataset_revision="17ece8eb89862368edc0cc806acee6fca5163474",
            split=split,
            candidate_rankings={action_id: rankings[action_id]},
        )
        if len(converted) != 1:
            raise RuntimeError("trimmed Mind2Web row must produce one step")
        case = converted[0]
        context = case.public.public_context
        if not isinstance(context, dict):
            raise ValueError("Mind2Web public candidates are unavailable")
        candidates = context.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Mind2Web public candidates are unavailable")
        candidate_lines: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("Mind2Web public candidate must be an object")
            candidate_lines.append(
                f"[{candidate['backend_node_id']}] <{candidate['tag']}> {candidate['attributes']}"
            )
        question = (
            f"Task: {case.public.query}\n"
            f"Website: {context['website']}\n"
            "Candidate elements:\n"
            + "\n".join(candidate_lines)
            + "\nReturn `Action: <CLICK|TYPE|SELECT> <backend_node_id> <value>`. "
            "Use an empty value for CLICK."
        )
        spec = protocol.benchmark(DirectBenchmark.MIND2WEB)
        cases.append(
            PrivateDirectCase(
                public_task=DirectTask(
                    task_id=case.public.task_id,
                    benchmark=DirectBenchmark.MIND2WEB,
                    messages=render_interactive_messages(
                        spec.prompt_profile,
                        task=case.public.query,
                        current_observation=question,
                    ),
                    profile=protocol.profile(spec.decoding_profile),
                    parser=PARSER_REGISTRY[spec.parser_profile],
                    prompt_profile_id=spec.prompt_profile,
                    parser_profile_id=spec.parser_profile,
                    population_id=spec.population,
                    run_seed=spec.seed_aggregation.seeds[0],
                ),
                target=case.target,
                private_metadata={
                    "source_split": split,
                    "source_identity": action_id,
                    "source_position": len(cases),
                },
            )
        )
    return tuple(cases)


def _static_case(
    *,
    benchmark: DirectBenchmark,
    index: int,
    question: str,
    target: object,
    protocol: DirectReferenceProtocol,
    metadata: dict[str, object],
) -> PrivateDirectCase:
    spec = protocol.benchmark(benchmark)
    normalized_metadata = dict(metadata)
    normalized_metadata.setdefault(
        "source_identity", f"{benchmark.value}:selection-position:{index}"
    )
    normalized_metadata.setdefault("source_position", index)
    normalized_metadata.setdefault("source_split", "unspecified")
    return PrivateDirectCase(
        public_task=DirectTask(
            task_id=f"direct-reference:{benchmark.value}:{index:03d}",
            benchmark=benchmark,
            messages=render_static_messages(spec.prompt_profile, question),
            profile=protocol.profile(spec.decoding_profile),
            parser=PARSER_REGISTRY[spec.parser_profile],
            prompt_profile_id=spec.prompt_profile,
            parser_profile_id=spec.parser_profile,
            population_id=spec.population,
            run_seed=spec.seed_aggregation.seeds[0],
        ),
        target=target,
        private_metadata=normalized_metadata,
    )
