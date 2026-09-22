"""Source adapters for the private Protocol 13 formal-training dataset.

The adapters intentionally accept explicit paths.  No server path, credential, question,
answer, rubric, test, or per-item identity is compiled into the repository.  Public model
inputs and trusted evaluator targets are separated as each source row is converted.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev.contracts import JsonValue, normalize_json
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.rollout import ModelVisibleMessage, RolloutTask

from .protocol_v13_training import Protocol13TrainingSourceCase

_HOTPOT_VERSION = "skillflow-dataset-07bb38bcc62fa8bebab6af86c39ba23b0293c97d"
_TRIVIA_VERSION = _HOTPOT_VERSION
_AIME_VERSION = "project-released"
_HEALTH_VERSION = "openai-full-2025-05-07"
_WEBSHOP_VERSION = "webshop-64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
_ALFWORLD_VERSION = "alfworld-aaba6870f86c5be6a08a491f32a50b906227bc3e"
_MBPP_VERSION = "evalplus-mbppplus-v0.2.0"
_HUMANEVAL_VERSION = "openai-humaneval-v1"
NONFINITE_FLOAT_FORMAT = "skillev-private-nonfinite-float@1"


class Protocol13TrainingSourceError(ValueError):
    """A private source cannot be safely projected into the formal dataset."""


@dataclass(frozen=True, slots=True)
class Protocol13TrainingSourcePaths:
    joint_qa_train: Path
    released_final_source: Path
    aime_train: Path
    healthbench: Path
    webshop_goals: Path
    webshop_final_manifest: Path
    alfworld_train: Path
    alfworld_final_manifest: Path
    mbpp_plus: Path
    mbpp_plus_final_manifest: Path
    humaneval: Path
    humaneval_final_manifest: Path

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            path = getattr(self, field)
            if not isinstance(path, Path) or not path.is_absolute() or not path.is_file():
                raise Protocol13TrainingSourceError(
                    f"{field} must be an absolute existing private file"
                )


@dataclass(frozen=True, slots=True)
class Protocol13TrainingSourceBundle:
    sources: Mapping[Protocol13Benchmark, tuple[Protocol13TrainingSourceCase, ...]]
    source_pool_counts: Mapping[Protocol13Benchmark, int]
    excluded_final_counts: Mapping[Protocol13Benchmark, int]
    heldout_validation_counts: Mapping[Protocol13Benchmark, int]

    def __post_init__(self) -> None:
        expected = set(Protocol13Benchmark)
        if set(self.sources) != expected:
            raise Protocol13TrainingSourceError("source bundle does not cover Protocol 13")
        if (
            set(self.source_pool_counts) != expected
            or set(self.excluded_final_counts) != expected
            or set(self.heldout_validation_counts) != expected
        ):
            raise Protocol13TrainingSourceError("source diagnostics do not cover Protocol 13")
        for benchmark, cases in self.sources.items():
            if not cases or any(case.benchmark is not benchmark for case in cases):
                raise Protocol13TrainingSourceError("source bundle has an empty or mislabeled lane")
            if self.source_pool_counts[benchmark] != len(cases):
                raise Protocol13TrainingSourceError("source bundle count differs from its lane")


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise Protocol13TrainingSourceError(f"{field} must be non-empty text without NUL")
    return value


def _object(value: object, *, field: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise Protocol13TrainingSourceError(f"{field} must be a JSON object")
    return normalized


def _array(value: object, *, field: str) -> list[JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, list):
        raise Protocol13TrainingSourceError(f"{field} must be a JSON array")
    return normalized


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _nonfinite_constant(token: str) -> dict[str, str]:
    values = {"NaN": "nan", "Infinity": "+inf", "-Infinity": "-inf"}
    try:
        value = values[token]
    except KeyError as error:
        raise Protocol13TrainingSourceError("unsupported non-finite JSON constant") from error
    return {"format": NONFINITE_FLOAT_FORMAT, "value": value}


def restore_nonfinite_evaluator_values(value: object) -> object:
    """Restore tagged EvalPlus constants immediately before trusted evaluation."""

    if isinstance(value, dict):
        if set(value) == {"format", "value"} and value.get("format") == NONFINITE_FLOAT_FORMAT:
            encoded = value.get("value")
            if encoded == "nan":
                return float("nan")
            if encoded == "+inf":
                return float("inf")
            if encoded == "-inf":
                return float("-inf")
            raise Protocol13TrainingSourceError("invalid tagged non-finite value")
        return {key: restore_nonfinite_evaluator_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_nonfinite_evaluator_values(item) for item in value]
    return value


def _jsonl_rows(lines: Iterable[str]) -> tuple[dict[str, JsonValue], ...]:
    rows: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(
                _object(
                    json.loads(line, parse_constant=_nonfinite_constant),
                    field=f"row {line_number}",
                )
            )
        except json.JSONDecodeError as error:
            raise Protocol13TrainingSourceError(f"invalid JSONL at line {line_number}") from error
    if not rows:
        raise Protocol13TrainingSourceError("source JSONL is empty")
    return tuple(rows)


def _jsonl(path: Path) -> tuple[dict[str, JsonValue], ...]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return _jsonl_rows(stream)
    with path.open("r", encoding="utf-8") as stream:
        return _jsonl_rows(stream)


def _json(path: Path) -> JsonValue:
    try:
        return cast(JsonValue, normalize_json(json.loads(path.read_text(encoding="utf-8"))))
    except json.JSONDecodeError as error:
        raise Protocol13TrainingSourceError("source JSON is invalid") from error


def _task(
    *,
    benchmark: Protocol13Benchmark,
    source_version: str,
    source_id: str,
    task_family_suffix: str,
    query: str,
    public_payload: JsonValue,
    available_tools: tuple[str, ...] = (),
    environment_suffix: str | None = None,
    messages: tuple[ModelVisibleMessage, ...] = (),
) -> RolloutTask:
    environment = f"benchmark:{benchmark.value}@{source_version}"
    if environment_suffix is not None:
        environment += f":{environment_suffix}"
    return RolloutTask(
        task_id=source_id,
        environment_id=environment,
        task_family=f"{benchmark.value}/{task_family_suffix}",
        context_id=f"{benchmark.value}:training",
        query=_text(query, field="public query"),
        available_tools=available_tools,
        public_context={
            "benchmark_id": benchmark.value,
            "dataset_revision": source_version,
            "payload": normalize_json(public_payload),
            "split": "training",
        },
        model_visible_messages=messages,
    )


def _case(
    *,
    benchmark: Protocol13Benchmark,
    population_id: str,
    source_version: str,
    source_id: str,
    task: RolloutTask,
    evaluator_kind: str,
    target: Mapping[str, object],
) -> Protocol13TrainingSourceCase:
    return Protocol13TrainingSourceCase(
        benchmark=benchmark,
        population_id=population_id,
        source_version=source_version,
        source_id=source_id,
        task=task,
        evaluator_kind=evaluator_kind,
        evaluator_target=_object(target, field="private evaluator target"),
    )


def _final_public_queries(path: Path) -> Mapping[str, frozenset[str]]:
    raw = _json(path)
    if not isinstance(raw, list):
        raise Protocol13TrainingSourceError("released final source must be a JSON array")
    by_type: dict[str, set[str]] = {}
    for value in raw:
        row = _object(value, field="released final row")
        task_type = _text(row.get("task_type"), field="released final task type")
        question = _text(row.get("question"), field="released final question")
        by_type.setdefault(task_type, set()).add(_normalized_text(question))
    return {key: frozenset(values) for key, values in by_type.items()}


def _accepted_answers(row: Mapping[str, JsonValue]) -> tuple[str, ...]:
    values: list[str] = []
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        payload = metadata.get("evaluator_payload")
        raw_answers = payload.get("accepted_answers") if isinstance(payload, dict) else None
        if isinstance(raw_answers, list):
            values.extend(_text(value, field="accepted answer") for value in raw_answers)
    answer = _text(row.get("answer"), field="source answer")
    values.append(answer)
    return tuple(dict.fromkeys(values))


def _joint_qa_cases(
    path: Path,
    final_queries: Mapping[str, frozenset[str]],
) -> tuple[
    Mapping[Protocol13Benchmark, tuple[Protocol13TrainingSourceCase, ...]],
    Mapping[Protocol13Benchmark, int],
]:
    lanes: dict[Protocol13Benchmark, list[Protocol13TrainingSourceCase]] = {
        Protocol13Benchmark.HOTPOT_QA: [],
        Protocol13Benchmark.TRIVIA_QA: [],
    }
    excluded: Counter[Protocol13Benchmark] = Counter()
    definitions = {
        "multi_hop_qa": (
            Protocol13Benchmark.HOTPOT_QA,
            "hotpotqa-train-v13",
            _HOTPOT_VERSION,
            "multi-hop-qa",
            "hotpotqa-official-em-f1",
        ),
        "factual_qa": (
            Protocol13Benchmark.TRIVIA_QA,
            "triviaqa-train-v13",
            _TRIVIA_VERSION,
            "factual-qa",
            "triviaqa-official-alias-em-f1",
        ),
    }
    seen: dict[Protocol13Benchmark, set[str]] = {key: set() for key in lanes}
    for row in _jsonl(path):
        definition = definitions.get(str(row.get("task_type")))
        if definition is None:
            continue
        benchmark, population_id, version, family, evaluator = definition
        query = _text(row.get("question"), field="QA question")
        normalized = _normalized_text(query)
        if normalized in final_queries.get(str(row["task_type"]), frozenset()):
            excluded[benchmark] += 1
            continue
        if normalized in seen[benchmark]:
            raise Protocol13TrainingSourceError("QA training population repeats public content")
        seen[benchmark].add(normalized)
        source_id = _text(row.get("task_id"), field="QA source ID")
        answers = list(_accepted_answers(row))
        if benchmark is Protocol13Benchmark.HOTPOT_QA:
            context = _array(row.get("context"), field="HotpotQA context")
            if not context or any(type(paragraph) is not str for paragraph in context):
                raise Protocol13TrainingSourceError(
                    "HotpotQA context must contain non-empty text paragraphs"
                )
            metadata = _object(row.get("metadata"), field="HotpotQA metadata")
            evaluator_payload = _object(
                metadata.get("evaluator_payload"), field="HotpotQA evaluator payload"
            )
            query = f"{query}\n\nEvidence:\n" + "\n\n".join(
                _text(paragraph, field="HotpotQA paragraph") for paragraph in context
            )
            public_payload: JsonValue = {"context_in_query": True}
            target: Mapping[str, object] = {
                "accepted_answers": answers,
                "supporting_facts": evaluator_payload.get("supporting_facts"),
            }
        else:
            if _array(row.get("context"), field="TriviaQA context"):
                raise Protocol13TrainingSourceError(
                    "TriviaQA training input must not contain direct context"
                )
            public_payload = {"initial_context": "none"}
            target = {"accepted_answers": answers}
        task = _task(
            benchmark=benchmark,
            source_version=version,
            source_id=source_id,
            task_family_suffix=family,
            query=query,
            public_payload=public_payload,
        )
        lanes[benchmark].append(
            _case(
                benchmark=benchmark,
                population_id=population_id,
                source_version=version,
                source_id=source_id,
                task=task,
                evaluator_kind=evaluator,
                target=target,
            )
        )
    return (
        {key: tuple(value) for key, value in lanes.items()},
        {key: excluded[key] for key in lanes},
    )


def _aime_cases(
    path: Path,
    final_queries: Mapping[str, frozenset[str]],
) -> tuple[tuple[Protocol13TrainingSourceCase, ...], int]:
    cases: list[Protocol13TrainingSourceCase] = []
    excluded = 0
    seen: set[str] = set()
    for row in _jsonl(path):
        if row.get("task_type") != "math_reasoning":
            raise Protocol13TrainingSourceError("AIME training source has another task type")
        query = _text(row.get("question"), field="AIME problem")
        normalized = _normalized_text(query)
        if normalized in final_queries.get("math_reasoning", frozenset()):
            excluded += 1
            continue
        if normalized in seen:
            raise Protocol13TrainingSourceError("AIME training population repeats a problem")
        seen.add(normalized)
        source_id = _text(row.get("task_id"), field="AIME source ID")
        task = _task(
            benchmark=Protocol13Benchmark.AIME_2026,
            source_version=_AIME_VERSION,
            source_id=source_id,
            task_family_suffix="integer-answer",
            query=query,
            public_payload={"benchmark_slice": "pre-2026"},
        )
        cases.append(
            _case(
                benchmark=Protocol13Benchmark.AIME_2026,
                population_id="aime-train-v13",
                source_version=_AIME_VERSION,
                source_id=source_id,
                task=task,
                evaluator_kind="integer-exact",
                target={"accepted_answers": list(_accepted_answers(row))},
            )
        )
    return tuple(cases), excluded


def _render_messages(messages: Sequence[Mapping[str, JsonValue]]) -> str:
    return "\n\n".join(
        f"{_text(message.get('role'), field='message role').title()}: "
        f"{_text(message.get('content'), field='message content')}"
        for message in messages
    )


def _healthbench_cases(
    path: Path,
) -> tuple[tuple[Protocol13TrainingSourceCase, ...], int]:
    rows = list(_jsonl(path))
    if len(rows) != 5_000:
        raise Protocol13TrainingSourceError("HealthBench Full must contain 5,000 rows")
    prompt_ids = tuple(_text(row.get("prompt_id"), field="HealthBench prompt ID") for row in rows)
    if len(prompt_ids) != len(set(prompt_ids)):
        raise Protocol13TrainingSourceError("HealthBench repeats a prompt ID")
    final_indices = set(random.Random(0).sample(range(len(rows)), 128))  # noqa: S311
    cases: list[Protocol13TrainingSourceCase] = []
    for index, row in enumerate(rows):
        if index in final_indices:
            continue
        source_id = prompt_ids[index]
        raw_messages = _array(row.get("prompt"), field="HealthBench prompt")
        messages: list[ModelVisibleMessage] = []
        message_values: list[Mapping[str, JsonValue]] = []
        for raw_message in raw_messages:
            message = _object(raw_message, field="HealthBench message")
            role = _text(message.get("role"), field="HealthBench message role")
            content = _text(message.get("content"), field="HealthBench message content")
            messages.append(ModelVisibleMessage(role=role, content=content))
            message_values.append(message)
        task = _task(
            benchmark=Protocol13Benchmark.HEALTHBENCH,
            source_version=_HEALTH_VERSION,
            source_id=source_id,
            task_family_suffix="health-dialogue",
            query=_render_messages(message_values),
            public_payload={"message_count": len(messages)},
            messages=tuple(messages),
        )
        cases.append(
            _case(
                benchmark=Protocol13Benchmark.HEALTHBENCH,
                population_id="healthbench-train-v13",
                source_version=_HEALTH_VERSION,
                source_id=source_id,
                task=task,
                evaluator_kind="simple-evals-rubric",
                target={
                    "grader_kind": "healthbench-qwen35-local-simple-evals",
                    "prompt": raw_messages,
                    "rubrics": _array(row.get("rubrics"), field="HealthBench rubrics"),
                },
            )
        )
    return tuple(cases), len(final_indices)


def _manifest_cases(path: Path, *, benchmark: str) -> tuple[dict[str, JsonValue], ...]:
    manifest = _object(_json(path), field=f"{benchmark} final manifest")
    cases = _array(manifest.get("cases"), field=f"{benchmark} final cases")
    selected = tuple(
        _object(value, field=f"{benchmark} final case")
        for value in cases
        if isinstance(value, dict) and value.get("benchmark") == benchmark
    )
    if len(selected) != 128:
        raise Protocol13TrainingSourceError(f"{benchmark} final manifest must contain 128 cases")
    return selected


def _webshop_cases(
    goals_path: Path,
    final_manifest_path: Path,
) -> tuple[tuple[Protocol13TrainingSourceCase, ...], int]:
    goals = list(_jsonl(goals_path))
    if len(goals) != 12_087:
        raise Protocol13TrainingSourceError("WebShop official goal inventory must contain 12,087")
    random.Random(233).shuffle(goals)  # noqa: S311 - official WebShop server order
    final = _manifest_cases(final_manifest_path, benchmark="webshop")
    final_indices: set[int] = set()
    for row in final:
        raw_index = _object(row.get("payload"), field="WebShop final payload").get("goal_index")
        if type(raw_index) is not int or raw_index < 0:
            raise Protocol13TrainingSourceError("WebShop final goal index is invalid")
        final_indices.add(raw_index)
    final_queries = {
        _normalized_text(_text(row.get("task"), field="WebShop final task")) for row in final
    }
    cases: list[Protocol13TrainingSourceCase] = []
    seen_queries: set[str] = set()
    excluded = 0
    for goal_index in range(1_500, len(goals)):
        goal = goals[goal_index]
        query = _text(goal.get("instruction_text"), field="WebShop instruction")
        normalized = _normalized_text(query)
        if goal_index in final_indices or normalized in final_queries:
            excluded += 1
            continue
        if normalized in seen_queries:
            continue
        seen_queries.add(normalized)
        source_id = f"WebShop/goal-{goal_index}"
        task = _task(
            benchmark=Protocol13Benchmark.WEB_SHOP,
            source_version=_WEBSHOP_VERSION,
            source_id=source_id,
            task_family_suffix="shopping",
            query=query,
            public_payload={
                "category": _text(goal.get("category"), field="WebShop category"),
                "observation_format": "official-text",
            },
            available_tools=("click", "purchase", "search"),
            environment_suffix="official-environment",
        )
        cases.append(
            _case(
                benchmark=Protocol13Benchmark.WEB_SHOP,
                population_id="webshop-train-v13",
                source_version=_WEBSHOP_VERSION,
                source_id=source_id,
                task=task,
                evaluator_kind="webshop-native-reward",
                target={
                    "goal_id": f"goal-{goal_index}",
                    "goal_index": goal_index,
                    "native_split": "train",
                },
            )
        )
    return tuple(cases), excluded


def _alfworld_final_directories(path: Path) -> tuple[set[str], set[str]]:
    manifest = _object(_json(path), field="ALFWorld final manifest")
    selected = _manifest_cases(path, benchmark="alfworld")
    deployments = _object(manifest.get("deployments"), field="ALFWorld deployments")
    final_directories: set[str] = set()
    final_queries: set[str] = set()
    for row in selected:
        deployment_id = _text(row.get("deployment"), field="ALFWorld deployment ID")
        deployment = _object(deployments.get(deployment_id), field="ALFWorld deployment")
        games = _object(deployment.get("games"), field="ALFWorld deployment games")
        payload = _object(row.get("payload"), field="ALFWorld final payload")
        game_id = _text(payload.get("game_id"), field="ALFWorld final game ID")
        game = _object(games.get(game_id), field="ALFWorld final game")
        final_directories.add(str(Path(_text(game.get("data_directory"), field="game directory"))))
        final_queries.add(_normalized_text(_text(row.get("task"), field="ALFWorld final task")))
    return final_directories, final_queries


def _alfworld_cases(
    path: Path,
    final_manifest_path: Path,
) -> tuple[tuple[Protocol13TrainingSourceCase, ...], int]:
    final_directories, final_queries = _alfworld_final_directories(final_manifest_path)
    cases: list[Protocol13TrainingSourceCase] = []
    excluded = 0
    seen_games: set[str] = set()
    for row in _jsonl(path):
        query = _text(row.get("question"), field="ALFWorld instruction")
        extra = _object(row.get("extra"), field="ALFWorld extra")
        canonical = _text(extra.get("canonical_instruction"), field="canonical instruction")
        if query != canonical:
            raise Protocol13TrainingSourceError("ALFWorld task differs from canonical instruction")
        env = _object(row.get("env_config"), field="ALFWorld environment route")
        game_file = Path(_text(env.get("game_file"), field="ALFWorld game file"))
        if not game_file.is_absolute() or not game_file.is_file():
            raise Protocol13TrainingSourceError("ALFWorld game file is absent")
        game_directory = str(game_file.parent)
        if game_directory in final_directories or _normalized_text(query) in final_queries:
            excluded += 1
            continue
        if game_directory in seen_games:
            raise Protocol13TrainingSourceError("ALFWorld training source repeats a game")
        seen_games.add(game_directory)
        source_id = _text(row.get("task_id"), field="ALFWorld source ID")
        max_steps = env.get("max_steps")
        if type(max_steps) is not int or max_steps < 1:
            raise Protocol13TrainingSourceError("ALFWorld max_steps must be positive")
        family = _text(extra.get("task_family"), field="ALFWorld task family")
        task = _task(
            benchmark=Protocol13Benchmark.ALF_WORLD,
            source_version=_ALFWORLD_VERSION,
            source_id=source_id,
            task_family_suffix=family,
            query=query,
            public_payload={"max_steps": max_steps, "observation_format": "official-text"},
            available_tools=("act",),
            environment_suffix="official-environment",
        )
        cases.append(
            _case(
                benchmark=Protocol13Benchmark.ALF_WORLD,
                population_id="alfworld-train-v13",
                source_version=_ALFWORLD_VERSION,
                source_id=source_id,
                task=task,
                evaluator_kind="alfworld-success",
                target={
                    "environment_route": env,
                    "target_won": True,
                },
            )
        )
    return tuple(cases), excluded


def _manifest_source_ids(path: Path, *, expected_count: int) -> frozenset[str]:
    manifest = _object(_json(path), field="final population manifest")
    entries = _array(manifest.get("entries"), field="final population entries")
    identities = frozenset(
        _text(
            _object(value, field="final population entry").get("source_identity"),
            field="final source identity",
        )
        for value in entries
    )
    if len(entries) != expected_count or len(identities) != expected_count:
        raise Protocol13TrainingSourceError("final population manifest count differs")
    return identities


def _validation_source_ids(
    rows: Sequence[Mapping[str, JsonValue]],
    *,
    count: int,
) -> frozenset[str]:
    ordered = sorted(_text(row.get("task_id"), field="code source ID") for row in rows)
    if len(ordered) <= count or len(ordered) != len(set(ordered)):
        raise Protocol13TrainingSourceError("code source cannot form a disjoint validation split")
    random.Random(0).shuffle(ordered)  # noqa: S311 - frozen scientific split
    return frozenset(ordered[:count])


def _mbpp_cases(
    path: Path,
    final_manifest_path: Path,
) -> tuple[tuple[Protocol13TrainingSourceCase, ...], int, int]:
    rows = _jsonl(path)
    if len(rows) != 378:
        raise Protocol13TrainingSourceError("MBPP+ v0.2.0 must contain 378 tasks")
    final_ids = _manifest_source_ids(final_manifest_path, expected_count=128)
    remaining = tuple(
        row for row in rows if _text(row.get("task_id"), field="MBPP+ source ID") not in final_ids
    )
    validation_ids = _validation_source_ids(remaining, count=32)
    cases: list[Protocol13TrainingSourceCase] = []
    for row in remaining:
        source_id = _text(row.get("task_id"), field="MBPP+ source ID")
        if source_id in validation_ids:
            continue
        query = _text(row.get("prompt"), field="MBPP+ prompt")
        task = _task(
            benchmark=Protocol13Benchmark.MBPP_PLUS,
            source_version=_MBPP_VERSION,
            source_id=source_id,
            task_family_suffix="code-generation",
            query=query,
            public_payload={"language": "python", "test_suite": "evalplus-base-plus-v0.2.0"},
        )
        target = {
            key: row.get(key)
            for key in (
                "assertion",
                "atol",
                "base_input",
                "canonical_solution",
                "contract",
                "entry_point",
                "plus_input",
            )
        }
        cases.append(
            _case(
                benchmark=Protocol13Benchmark.MBPP_PLUS,
                population_id="mbpp-plus-train-v13",
                source_version=_MBPP_VERSION,
                source_id=source_id,
                task=task,
                evaluator_kind="evalplus-base-plus",
                target=target,
            )
        )
    if len(cases) + len(final_ids) + len(validation_ids) != 378:
        raise Protocol13TrainingSourceError("MBPP+ final IDs do not belong to the source")
    return tuple(cases), len(final_ids), len(validation_ids)


def _humaneval_cases(
    path: Path,
    final_manifest_path: Path,
) -> tuple[tuple[Protocol13TrainingSourceCase, ...], int, int]:
    rows = _jsonl(path)
    if len(rows) != 164:
        raise Protocol13TrainingSourceError("HumanEval must contain 164 tasks")
    final_ids = _manifest_source_ids(final_manifest_path, expected_count=128)
    remaining = tuple(
        row
        for row in rows
        if _text(row.get("task_id"), field="HumanEval source ID") not in final_ids
    )
    validation_ids = _validation_source_ids(remaining, count=4)
    cases: list[Protocol13TrainingSourceCase] = []
    for row in remaining:
        source_id = _text(row.get("task_id"), field="HumanEval source ID")
        if source_id in validation_ids:
            continue
        query = _text(row.get("prompt"), field="HumanEval prompt")
        task = _task(
            benchmark=Protocol13Benchmark.HUMAN_EVAL,
            source_version=_HUMANEVAL_VERSION,
            source_id=source_id,
            task_family_suffix="code-generation",
            query=query,
            public_payload={"language": "python", "test_suite": "humaneval-original-v1"},
        )
        cases.append(
            _case(
                benchmark=Protocol13Benchmark.HUMAN_EVAL,
                population_id="humaneval-train-v13",
                source_version=_HUMANEVAL_VERSION,
                source_id=source_id,
                task=task,
                evaluator_kind="humaneval-native",
                target={
                    "canonical_solution": row.get("canonical_solution"),
                    "entry_point": row.get("entry_point"),
                    "test": row.get("test"),
                },
            )
        )
    if len(cases) + len(final_ids) + len(validation_ids) != 164:
        raise Protocol13TrainingSourceError("HumanEval final IDs do not belong to the source")
    return tuple(cases), len(final_ids), len(validation_ids)


def load_protocol13_training_sources(
    paths: Protocol13TrainingSourcePaths,
) -> Protocol13TrainingSourceBundle:
    """Load all exact-eight lanes and enforce final-panel isolation before sampling."""

    final_queries = _final_public_queries(paths.released_final_source)
    qa, qa_excluded = _joint_qa_cases(paths.joint_qa_train, final_queries)
    aime, aime_excluded = _aime_cases(paths.aime_train, final_queries)
    health, health_excluded = _healthbench_cases(paths.healthbench)
    webshop, webshop_excluded = _webshop_cases(paths.webshop_goals, paths.webshop_final_manifest)
    alfworld, alfworld_excluded = _alfworld_cases(
        paths.alfworld_train, paths.alfworld_final_manifest
    )
    mbpp, mbpp_excluded, mbpp_validation = _mbpp_cases(
        paths.mbpp_plus, paths.mbpp_plus_final_manifest
    )
    humaneval, humaneval_excluded, humaneval_validation = _humaneval_cases(
        paths.humaneval, paths.humaneval_final_manifest
    )
    sources = {
        Protocol13Benchmark.HOTPOT_QA: qa[Protocol13Benchmark.HOTPOT_QA],
        Protocol13Benchmark.TRIVIA_QA: qa[Protocol13Benchmark.TRIVIA_QA],
        Protocol13Benchmark.AIME_2026: aime,
        Protocol13Benchmark.HEALTHBENCH: health,
        Protocol13Benchmark.WEB_SHOP: webshop,
        Protocol13Benchmark.ALF_WORLD: alfworld,
        Protocol13Benchmark.MBPP_PLUS: mbpp,
        Protocol13Benchmark.HUMAN_EVAL: humaneval,
    }
    excluded = {
        Protocol13Benchmark.HOTPOT_QA: qa_excluded[Protocol13Benchmark.HOTPOT_QA],
        Protocol13Benchmark.TRIVIA_QA: qa_excluded[Protocol13Benchmark.TRIVIA_QA],
        Protocol13Benchmark.AIME_2026: aime_excluded,
        Protocol13Benchmark.HEALTHBENCH: health_excluded,
        Protocol13Benchmark.WEB_SHOP: webshop_excluded,
        Protocol13Benchmark.ALF_WORLD: alfworld_excluded,
        Protocol13Benchmark.MBPP_PLUS: mbpp_excluded,
        Protocol13Benchmark.HUMAN_EVAL: humaneval_excluded,
    }
    validation = dict.fromkeys(Protocol13Benchmark, 0)
    validation[Protocol13Benchmark.MBPP_PLUS] = mbpp_validation
    validation[Protocol13Benchmark.HUMAN_EVAL] = humaneval_validation
    return Protocol13TrainingSourceBundle(
        sources=sources,
        source_pool_counts={key: len(value) for key, value in sources.items()},
        excluded_final_counts=excluded,
        heldout_validation_counts=validation,
    )


def source_diagnostics(bundle: Protocol13TrainingSourceBundle) -> dict[str, JsonValue]:
    return {
        "format": "skillev-private-protocol13-source-diagnostics@1",
        "lanes": [
            {
                "benchmark": benchmark.value,
                "excluded_final_or_overlap_count": bundle.excluded_final_counts[benchmark],
                "heldout_validation_count": bundle.heldout_validation_counts[benchmark],
                "source_pool_count": bundle.source_pool_counts[benchmark],
            }
            for benchmark in Protocol13Benchmark
        ],
    }


__all__ = [
    "NONFINITE_FLOAT_FORMAT",
    "Protocol13TrainingSourceBundle",
    "Protocol13TrainingSourceError",
    "Protocol13TrainingSourcePaths",
    "load_protocol13_training_sources",
    "restore_nonfinite_evaluator_values",
    "source_diagnostics",
]
