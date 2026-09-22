"""Pinned server-side source readers for the nine Protocol 10 domains.

The readers in this module are intentionally deployment-facing.  They accept
an already populated private data root, never download data, and return
separated source records for :mod:`protocol_v10_materialization`.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from datasets import DatasetDict, load_from_disk  # type: ignore[import-untyped]

from skillev.benchmarks.protocol_v10_action import protocol_v10_action_contract
from skillev.contracts import JsonValue, normalize_json
from skillev.experiments.protocol_v10 import (
    ActiveBenchmarkProtocolV10,
    BenchmarkPopulation,
)
from skillev.rollout import ModelVisibleMessage, RolloutTask

from .protocol_v10_materialization import ProtocolV10SourceRecord


class ProtocolV10SourceError(RuntimeError):
    """A frozen source is absent or has an incompatible schema."""


ALFWORLD_PUBLIC_RESET_FORMAT = "skillev-protocol-v10-alfworld-public-resets@1"
WEBSHOP_PUBLIC_RESET_FORMAT = "skillev-protocol-v10-webshop-public-resets@1"


def _load_webshop_public_resets(path: Path) -> dict[int, dict[str, JsonValue]]:
    """Load answer-free observations captured from the pinned official reset."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "records"}
        or value["format"] != WEBSHOP_PUBLIC_RESET_FORMAT
        or not isinstance(value["records"], dict)
    ):
        raise ProtocolV10SourceError("WebShop public reset cache is incompatible")
    resets: dict[int, dict[str, JsonValue]] = {}
    for raw_index, raw in value["records"].items():
        if type(raw_index) is not str or not raw_index.isdecimal() or not isinstance(raw, dict):
            raise ProtocolV10SourceError("WebShop public reset route is incompatible")
        index = int(raw_index)
        normalized = _json(raw, label="WebShop public reset")
        if not isinstance(normalized, dict) or set(normalized) != {
            "available_actions",
            "instruction_text",
            "observation_text",
        }:
            raise ProtocolV10SourceError("WebShop public reset fields differ")
        actions = normalized["available_actions"]
        if (
            not isinstance(actions, list)
            or not actions
            or any(type(action) is not str or not action.strip() for action in actions)
            or len(set(actions)) != len(actions)
            or type(normalized["instruction_text"]) is not str
            or not normalized["instruction_text"].strip()
            or type(normalized["observation_text"]) is not str
            or not normalized["observation_text"].strip()
        ):
            raise ProtocolV10SourceError("WebShop public reset values differ")
        resets[index] = normalized
    if not resets:
        raise ProtocolV10SourceError("WebShop public reset cache is empty")
    return resets


def _load_alfworld_public_resets(path: Path) -> dict[str, dict[str, JsonValue]]:
    """Load model-visible official reset projections prepared server-side."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "records"}
        or value["format"] != ALFWORLD_PUBLIC_RESET_FORMAT
        or not isinstance(value["records"], dict)
    ):
        raise ProtocolV10SourceError("ALFWorld public reset cache is incompatible")
    resets: dict[str, dict[str, JsonValue]] = {}
    for relative_path, raw in value["records"].items():
        if type(relative_path) is not str or not isinstance(raw, dict):
            raise ProtocolV10SourceError("ALFWorld public reset route is incompatible")
        normalized = _json(raw, label="ALFWorld public reset")
        if not isinstance(normalized, dict) or set(normalized) != {
            "admissible_commands",
            "game_id",
            "initial_observation",
            "instruction_text",
            "max_steps",
            "seed",
        }:
            raise ProtocolV10SourceError("ALFWorld public reset fields differ")
        commands = normalized["admissible_commands"]
        if (
            not isinstance(commands, list)
            or not commands
            or any(type(command) is not str or not command.strip() for command in commands)
            or type(normalized["game_id"]) is not str
            or type(normalized["initial_observation"]) is not str
            or type(normalized["instruction_text"]) is not str
            or type(normalized["max_steps"]) is not int
            or normalized["max_steps"] < 1
            or type(normalized["seed"]) is not int
            or normalized["seed"] < 0
        ):
            raise ProtocolV10SourceError("ALFWorld public reset values differ")
        resets[relative_path] = normalized
    if not resets:
        raise ProtocolV10SourceError("ALFWorld public reset cache is empty")
    return resets


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProtocolV10SourceError(f"{label} must be non-empty text")
    return value


def _json(value: object, *, label: str) -> JsonValue:
    try:
        return cast(JsonValue, normalize_json(value))
    except (TypeError, ValueError) as error:
        raise ProtocolV10SourceError(f"{label} is not JSON-compatible") from error


def _spec(protocol: ActiveBenchmarkProtocolV10, population_id: str) -> BenchmarkPopulation:
    matches = tuple(
        population
        for benchmark in protocol.benchmarks
        for population in benchmark.populations
        if population.population_id == population_id
    )
    if len(matches) != 1:
        raise ProtocolV10SourceError("Protocol 10 population ID is not unique")
    return matches[0]


def _task(
    spec: BenchmarkPopulation,
    *,
    raw_source_id: str,
    query: str,
    public_context: Mapping[str, JsonValue] | None = None,
    available_tools: tuple[str, ...] = (),
    task_family_suffix: str = "completion",
    environment_suffix: str = "completion",
    model_visible_messages: tuple[ModelVisibleMessage, ...] = (),
) -> RolloutTask:
    source_id = f"{spec.benchmark.value}/{raw_source_id}"
    context: dict[str, JsonValue] = {
        "benchmark_id": spec.benchmark.value,
        "population_id": spec.population_id,
        "source_version": spec.source_version,
    }
    if public_context is not None:
        context.update(public_context)
    max_steps = context.get("max_steps")
    action_surface, budget_profile = protocol_v10_action_contract(
        spec.benchmark.value,
        max_steps=max_steps if type(max_steps) is int else None,
    )
    return RolloutTask(
        task_id=source_id,
        environment_id=(
            f"benchmark:{spec.benchmark.value}@{spec.source_version}:{environment_suffix}"
        ),
        task_family=f"{spec.benchmark.value}/{task_family_suffix}",
        context_id=f"{spec.benchmark.value}:{spec.population_id}",
        query=_text(query, label="public query"),
        available_tools=available_tools,
        public_context=context,
        action_surface=action_surface,
        budget_profile=budget_profile,
        model_visible_messages=model_visible_messages,
    )


def _record(
    spec: BenchmarkPopulation,
    *,
    raw_source_id: str,
    query: str,
    private_payload: object,
    public_context: Mapping[str, JsonValue] | None = None,
    available_tools: tuple[str, ...] = (),
    task_family_suffix: str = "completion",
    environment_suffix: str = "completion",
    model_visible_messages: tuple[ModelVisibleMessage, ...] = (),
) -> ProtocolV10SourceRecord:
    task = _task(
        spec,
        raw_source_id=raw_source_id,
        query=query,
        public_context=public_context,
        available_tools=available_tools,
        task_family_suffix=task_family_suffix,
        environment_suffix=environment_suffix,
        model_visible_messages=model_visible_messages,
    )
    return ProtocolV10SourceRecord(
        source_id=task.task_id,
        task=task,
        private_payload=_json(private_payload, label="private evaluator payload"),
    )


def _partition(
    records: Sequence[ProtocolV10SourceRecord],
    *,
    validation_count: int,
) -> tuple[tuple[ProtocolV10SourceRecord, ...], tuple[ProtocolV10SourceRecord, ...]]:
    ordered = sorted(records, key=lambda record: record.source_id)
    if len(ordered) <= validation_count:
        raise ProtocolV10SourceError("source population is too small for its held-out split")
    shuffled = list(ordered)
    random.Random(0).shuffle(shuffled)  # noqa: S311 - frozen scientific split
    validation = tuple(sorted(shuffled[:validation_count], key=lambda item: item.source_id))
    training = tuple(sorted(shuffled[validation_count:], key=lambda item: item.source_id))
    return training, validation


def _unique_public_queries(
    records: Iterable[ProtocolV10SourceRecord],
) -> tuple[ProtocolV10SourceRecord, ...]:
    """Keep the first stable source row for identical model-visible tasks."""

    seen: set[str] = set()
    unique: list[ProtocolV10SourceRecord] = []
    for record in sorted(records, key=lambda item: item.source_id):
        key = " ".join(record.task.query.casefold().split())
        if key not in seen:
            unique.append(record)
            seen.add(key)
    return tuple(unique)


def _hotpot_records(
    spec: BenchmarkPopulation, rows: Iterable[Mapping[str, object]]
) -> tuple[ProtocolV10SourceRecord, ...]:
    records = []
    for row in rows:
        raw_id = _text(row.get("id"), label="HotpotQA ID")
        question = _text(row.get("question"), label="HotpotQA question")
        context = row.get("context")
        rendered: list[str] = [question, "\nEvidence:"]
        if isinstance(context, Mapping):
            titles = context.get("title")
            sentences = context.get("sentences")
            if isinstance(titles, Sequence) and isinstance(sentences, Sequence):
                for title, paragraph in zip(titles, sentences, strict=True):
                    if isinstance(title, str) and isinstance(paragraph, Sequence):
                        rendered.append(f"[{title}] " + " ".join(map(str, paragraph)))
        answer = _text(row.get("answer"), label="HotpotQA answer")
        records.append(
            _record(
                spec,
                raw_source_id=raw_id,
                query="\n".join(rendered),
                private_payload={
                    "accepted_answers": [answer],
                    "scoring_rule": "token-f1",
                },
                public_context={"question_type": str(row.get("type", "unknown"))},
            )
        )
    return tuple(records)


def _trivia_records(
    spec: BenchmarkPopulation, rows: Iterable[Mapping[str, object]]
) -> tuple[ProtocolV10SourceRecord, ...]:
    records = []
    for row in rows:
        raw_id = _text(row.get("question_id"), label="TriviaQA ID")
        answer = row.get("answer")
        if not isinstance(answer, Mapping):
            raise ProtocolV10SourceError("TriviaQA answer must be an object")
        aliases = answer.get("aliases")
        accepted = tuple(str(item) for item in aliases) if isinstance(aliases, Sequence) else ()
        value = answer.get("value")
        if isinstance(value, str):
            accepted = (value, *accepted)
        accepted = tuple(dict.fromkeys(item for item in accepted if item.strip()))
        if not accepted:
            raise ProtocolV10SourceError("TriviaQA row has no accepted answer")
        records.append(
            _record(
                spec,
                raw_source_id=raw_id,
                query=_text(row.get("question"), label="TriviaQA question"),
                private_payload={
                    "accepted_answers": list(accepted),
                    "scoring_rule": "token-f1",
                },
                public_context={"question_source": str(row.get("question_source", "unknown"))},
            )
        )
    return tuple(records)


def _aime_records(
    spec: BenchmarkPopulation, rows: Iterable[Mapping[str, object]]
) -> tuple[ProtocolV10SourceRecord, ...]:
    records = []
    for index, row in enumerate(rows):
        problem = row.get("problem", row.get("question"))
        answer = row.get("answer")
        if answer is None:
            raise ProtocolV10SourceError("AIME row lacks its official answer field")
        try:
            answer_int = int(str(answer).strip())
        except ValueError as error:
            raise ProtocolV10SourceError("AIME answer must be an integer") from error
        if not 0 <= answer_int <= 999:
            raise ProtocolV10SourceError("AIME answer lies outside the official range")
        raw = row.get("problem_idx", row.get("id", row.get("task_id", index)))
        year = str(row.get("year", spec.source_version))
        records.append(
            _record(
                spec,
                raw_source_id=f"{year}/{raw}",
                query=_text(problem, label="AIME problem"),
                private_payload={
                    "accepted_answers": [str(answer_int)],
                    "scoring_rule": "integer",
                },
                public_context={"year": year},
            )
        )
    return tuple(records)


def _render_messages(messages: Sequence[Mapping[str, object]]) -> str:
    rendered = []
    for message in messages:
        role = _text(message.get("role"), label="conversation role")
        content = _text(message.get("content"), label="conversation content")
        rendered.append(f"{role}: {content}")
    return "\n".join(rendered)


def _model_messages(
    messages: Sequence[Mapping[str, object]],
) -> tuple[ModelVisibleMessage, ...]:
    role_map = {
        "doctor": "assistant",
        "patient": "user",
        "human": "user",
        "model": "assistant",
    }
    rendered = []
    for message in messages:
        raw_role = _text(message.get("role"), label="conversation role").casefold()
        role = role_map.get(raw_role, raw_role)
        rendered.append(
            ModelVisibleMessage(
                role=role,
                content=_text(message.get("content"), label="conversation content"),
            )
        )
    return tuple(rendered)


def _prompt_text(value: object, *, label: str) -> str:
    """Normalize either a plain prompt or a chat-style prompt column."""

    if isinstance(value, str):
        return _text(value, label=label)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        messages = []
        for message in value:
            if not isinstance(message, Mapping):
                raise ProtocolV10SourceError(f"{label} messages must be objects")
            messages.append(message)
        return _text(_render_messages(messages), label=label)
    raise ProtocolV10SourceError(f"{label} must be text or chat messages")


def _health_training_records(
    spec: BenchmarkPopulation, parquet_path: Path
) -> tuple[ProtocolV10SourceRecord, ...]:
    rows = pq.read_table(parquet_path).to_pylist()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw in rows:
        row = cast(dict[str, object], raw)
        grouped[_text(row.get("conversation_id"), label="health conversation ID")].append(row)
    records = []
    for conversation_id, messages in grouped.items():
        ordered = sorted(messages, key=lambda item: int(cast(int, item["message_id"])))
        if str(ordered[-1].get("sender")).casefold() != "doctor":
            raise ProtocolV10SourceError("health conversation does not end with a doctor response")
        prompt = [
            {"role": str(item["sender"]), "content": str(item["content"])} for item in ordered[:-1]
        ]
        reference = _text(ordered[-1].get("content"), label="doctor response")
        records.append(
            _record(
                spec,
                raw_source_id=conversation_id,
                query=_render_messages(prompt),
                private_payload={
                    "grader_kind": "reference-health-dialogue",
                    "prompt": prompt,
                    "reference_response": reference,
                },
                public_context={"message_count": len(prompt)},
                task_family_suffix="health-dialogue",
                model_visible_messages=_model_messages(prompt),
            )
        )
    return _unique_public_queries(records)


def _healthbench_records(
    spec: BenchmarkPopulation, path: Path
) -> tuple[ProtocolV10SourceRecord, ...]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, list):
                raise ProtocolV10SourceError("HealthBench prompt must be a message list")
            raw_id = _text(row.get("prompt_id"), label="HealthBench prompt ID")
            records.append(
                _record(
                    spec,
                    raw_source_id=raw_id,
                    query=_render_messages(prompt),
                    private_payload={
                        "grader_kind": "official-healthbench",
                        "prompt": prompt,
                        "rubrics": row.get("rubrics"),
                    },
                    public_context={"message_count": len(prompt)},
                    task_family_suffix="health-dialogue",
                    model_visible_messages=_model_messages(prompt),
                )
            )
    return tuple(records)


def _webshop_records(
    spec: BenchmarkPopulation,
    goals: Sequence[object],
    indices: Sequence[int],
    resets: Mapping[int, Mapping[str, JsonValue]],
) -> tuple[ProtocolV10SourceRecord, ...]:
    records: list[ProtocolV10SourceRecord] = []
    for index in indices:
        reset = resets.get(index)
        if reset is None:
            raise ProtocolV10SourceError("WebShop public reset cache is incomplete")
        query = _text(goals[index], label="WebShop goal")
        if reset["instruction_text"] != query:
            raise ProtocolV10SourceError("WebShop reset instruction differs from goal inventory")
        records.append(
            _record(
                spec,
                raw_source_id=f"goal-{index:05d}",
                query=query,
                private_payload={"goal_index": index},
                public_context={
                    "initial_available_actions": reset["available_actions"],
                    "initial_observation": reset["observation_text"],
                    "scenario_id": f"webshop/goal-{index:05d}",
                },
                available_tools=("click", "purchase", "search"),
                task_family_suffix="shopping",
                environment_suffix="official-environment",
            )
        )
    return tuple(records)


def _unique_webshop_goal_indices(goals: Sequence[object]) -> list[int]:
    """Return one stable source index per model-visible WebShop goal."""

    indices: list[int] = []
    seen: set[str] = set()
    for index, goal in enumerate(goals):
        text = _text(goal, label="WebShop goal")
        identity = " ".join(text.casefold().split())
        if identity not in seen:
            seen.add(identity)
            indices.append(index)
    return indices


def _load_webshop_runtime_goals(path: Path) -> tuple[str, ...]:
    """Load goals in the exact order exposed by the official environment.

    The pinned WebShop server shuffles its goal objects with seed 233 before a
    numeric session index is resolved.  Protocol 10 therefore has to build its
    public task and private ``goal_index`` route from that same post-shuffle
    inventory.  Reading the older standalone ``human_goals.json`` would pair a
    public instruction with a different official session.
    """

    if not path.is_file():
        raise ProtocolV10SourceError("prepared WebShop runtime goals are absent")
    goals: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="\n") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolV10SourceError("prepared WebShop goal stream is invalid") from error
            if not isinstance(value, dict):
                raise ProtocolV10SourceError("prepared WebShop goal must be an object")
            goals.append(value)
    if len(goals) != 12_087:
        raise ProtocolV10SourceError("WebShop official goal count differs")
    random.Random(233).shuffle(goals)  # noqa: S311 - pinned official server order
    return tuple(
        _text(goal.get("instruction_text"), label="WebShop runtime instruction") for goal in goals
    )


def _alfworld_records(
    spec: BenchmarkPopulation,
    paths: Sequence[Path],
    data_root: Path,
    public_resets: Mapping[str, Mapping[str, JsonValue]],
) -> tuple[ProtocolV10SourceRecord, ...]:
    records = []
    for path in paths:
        # ALFWorld's published trajectory archive includes unsuccessful build
        # attempts without a compiled TextWorld game.  They are not runnable
        # benchmark items and therefore never enter a population.
        if len(tuple(path.parent.glob("game.tw-pddl"))) != 1:
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        annotations = row.get("turk_annotations")
        if not isinstance(annotations, Mapping) or not isinstance(annotations.get("anns"), list):
            raise ProtocolV10SourceError("ALFWorld trajectory lacks annotations")
        anns = cast(list[object], annotations["anns"])
        if not anns or not isinstance(anns[0], Mapping):
            raise ProtocolV10SourceError("ALFWorld trajectory has no task description")
        query = _text(cast(Mapping[str, object], anns[0]).get("task_desc"), label="ALFWorld task")
        relative = path.parent.relative_to(data_root).as_posix()
        reset = public_resets.get(relative)
        if reset is None or reset.get("instruction_text") != query:
            raise ProtocolV10SourceError("ALFWorld public reset differs from its source task")
        scenario = path.parent.parent.name
        records.append(
            _record(
                spec,
                raw_source_id=relative,
                query=query,
                private_payload={
                    "game_id": reset["game_id"],
                    "max_steps": reset["max_steps"],
                    "seed": reset["seed"],
                    "trajectory_relative_path": relative,
                },
                public_context={
                    "admissible_commands": reset["admissible_commands"],
                    "initial_observation": reset["initial_observation"],
                    "max_steps": reset["max_steps"],
                    "scenario_id": f"alfworld/{scenario}",
                },
                available_tools=("act",),
                task_family_suffix=scenario.split("-", 1)[0],
                environment_suffix="official-environment",
            )
        )
    return _unique_alfworld_scenarios(records)


def _unique_alfworld_scenarios(
    records: Iterable[ProtocolV10SourceRecord],
) -> tuple[ProtocolV10SourceRecord, ...]:
    """Keep one stable trial for each public ALFWorld game configuration."""

    unique: list[ProtocolV10SourceRecord] = []
    seen: set[str] = set()
    for record in sorted(records, key=lambda item: item.source_id):
        context = record.task.public_context
        if not isinstance(context, dict) or type(context.get("scenario_id")) is not str:
            raise ProtocolV10SourceError("ALFWorld record lacks a scenario identity")
        scenario_id = cast(str, context["scenario_id"])
        if scenario_id not in seen:
            seen.add(scenario_id)
            unique.append(record)
    return tuple(unique)


def _exclude_alfworld_evaluation_scenarios(
    training: Iterable[ProtocolV10SourceRecord],
    evaluation: Iterable[ProtocolV10SourceRecord],
) -> tuple[ProtocolV10SourceRecord, ...]:
    """Keep official evaluation games intact and remove their training twins."""

    protected = {
        cast(dict[str, JsonValue], record.task.public_context)["scenario_id"]
        for record in evaluation
    }
    return tuple(
        record
        for record in training
        if cast(dict[str, JsonValue], record.task.public_context)["scenario_id"] not in protected
    )


def _spreadsheet_training_records(
    spec: BenchmarkPopulation, path: Path
) -> tuple[ProtocolV10SourceRecord, ...]:
    records = []
    for raw in pq.read_table(path).to_pylist():
        row = cast(dict[str, object], raw)
        extra = row.get("extra_info")
        if not isinstance(extra, Mapping):
            raise ProtocolV10SourceError("spreadsheet training row lacks extra_info")
        reward_model = row.get("reward_model")
        if not isinstance(reward_model, Mapping):
            raise ProtocolV10SourceError("spreadsheet training row lacks its workbook route")
        workbook_route = _text(
            reward_model.get("ground_truth"),
            label="spreadsheet training workbook route",
        )
        raw_id = str(extra.get("id"))
        records.append(
            _record(
                spec,
                raw_source_id=raw_id,
                query=_prompt_text(row.get("prompt"), label="spreadsheet instruction"),
                private_payload={
                    "answer_position": extra.get("answer_position"),
                    "spreadsheet_task_route": workbook_route,
                },
                public_context=cast(
                    dict[str, JsonValue],
                    {
                        "tool_schema": {
                            "spreadsheet.execute": {
                                "arguments": {"code": "Python code editing WORKBOOK_PATH"}
                            }
                        },
                        "workbook_structure_id": f"spreadsheet-training/{raw_id}",
                    },
                ),
                available_tools=("spreadsheet.execute", "submit"),
                task_family_suffix="spreadsheet-editing",
                environment_suffix="spreadsheet-runtime",
            )
        )
    return _unique_public_queries(records)


def _spreadsheetbench_records(
    spec: BenchmarkPopulation, path: Path
) -> tuple[ProtocolV10SourceRecord, ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ProtocolV10SourceError("SpreadsheetBench dataset must be an array")
    records = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolV10SourceError("SpreadsheetBench row must be an object")
        raw_id = str(row.get("id"))
        workbook = _text(row.get("spreadsheet_path"), label="spreadsheet path")
        records.append(
            _record(
                spec,
                raw_source_id=raw_id,
                query=_text(row.get("instruction"), label="SpreadsheetBench instruction"),
                private_payload={
                    "answer_position": row.get("answer_position"),
                    "answer_sheet": row.get("answer_sheet"),
                    "data_position": row.get("data_position"),
                    "spreadsheet_relative_path": workbook,
                },
                public_context=cast(
                    dict[str, JsonValue],
                    {
                        "tool_schema": {
                            "spreadsheet.execute": {
                                "arguments": {"code": "Python code editing WORKBOOK_PATH"}
                            }
                        },
                        "workbook_structure_id": f"spreadsheetbench/{workbook}",
                    },
                ),
                available_tools=("spreadsheet.execute", "submit"),
                task_family_suffix="spreadsheet-editing",
                environment_suffix="official-oj",
            )
        )
    return tuple(records)


def _appworld_records(
    spec: BenchmarkPopulation, data_root: Path, split: str
) -> tuple[ProtocolV10SourceRecord, ...]:
    split_path = data_root / "datasets" / f"{split}.txt"
    source_ids = tuple(line.strip() for line in split_path.read_text().splitlines() if line.strip())
    public_api_surface = _appworld_public_api_surface(data_root)
    records = []
    for source_id in source_ids:
        specs = json.loads((data_root / "tasks" / source_id / "specs.json").read_text())
        if not isinstance(specs, Mapping):
            raise ProtocolV10SourceError("AppWorld task specs must be an object")
        generator_id = source_id.rsplit("_", 1)[0]
        records.append(
            _record(
                spec,
                raw_source_id=source_id,
                query=_text(specs.get("instruction"), label="AppWorld instruction"),
                private_payload={"appworld_source_id": source_id, "split": split},
                public_context={
                    "public_api_surface": public_api_surface,
                    "scenario_id": f"appworld/{generator_id}",
                },
                available_tools=("appworld.execute", "submit"),
                task_family_suffix=generator_id,
                environment_suffix="official-environment",
            )
        )
    return tuple(records)


def _appworld_public_api_surface(data_root: Path) -> JsonValue:
    """Load compact public signatures from official AppWorld API docs."""

    docs_root = data_root / "api_docs" / "standard"
    files = tuple(sorted(docs_root.glob("*.json")))
    if not files:
        raise ProtocolV10SourceError("AppWorld public API docs are absent")
    apps: list[JsonValue] = []
    for path in files:
        app_name = _text(path.stem, label="AppWorld app name")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or not raw:
            raise ProtocolV10SourceError("AppWorld public API doc is incompatible")
        signatures: list[JsonValue] = []
        for api_name, doc in sorted(raw.items()):
            if type(api_name) is not str or not isinstance(doc, Mapping):
                raise ProtocolV10SourceError("AppWorld public API entry is incompatible")
            raw_parameters = doc.get("parameters", [])
            if not isinstance(raw_parameters, Sequence) or isinstance(
                raw_parameters, str | bytes | bytearray
            ):
                raise ProtocolV10SourceError("AppWorld API parameters are incompatible")
            signature_parts: list[str] = []
            for raw_parameter in raw_parameters:
                if not isinstance(raw_parameter, Mapping):
                    raise ProtocolV10SourceError("AppWorld API parameter is incompatible")
                name = _text(raw_parameter.get("name"), label="AppWorld API parameter")
                parameter_type = _text(
                    raw_parameter.get("type", raw_parameter.get("schema", "value")),
                    label="AppWorld API parameter type",
                )
                suffix = "" if raw_parameter.get("required") is True else "=..."
                signature_parts.append(f"{name}:{parameter_type}{suffix}")
            signatures.append(f"apis.{app_name}.{api_name}({', '.join(signature_parts)})")
        apps.append({"name": app_name, "signatures": signatures})
    return {
        "apps": apps,
        "execution_examples": [
            "result = apis.<app>.<api>(parameter=value)",
            "print(result)",
        ],
    }


_SIGNATURE = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)")


def _code_signature(row: Mapping[str, object]) -> str:
    for field in ("code", "prompt"):
        value = row.get(field)
        if isinstance(value, str) and (match := _SIGNATURE.search(value)) is not None:
            arguments = re.sub(r"\s+", "", match.group(2))
            return f"{match.group(1)}({arguments})"
    raise ProtocolV10SourceError("MBPP row lacks a public function signature")


def _mbpp_records(
    spec: BenchmarkPopulation, rows: Iterable[Mapping[str, object]]
) -> tuple[ProtocolV10SourceRecord, ...]:
    records = []
    for row in rows:
        raw_id = str(row.get("task_id"))
        official_row = {
            field: row.get(field)
            for field in (
                "code",
                "prompt",
                "source_file",
                "task_id",
                "test",
                "test_imports",
                "test_list",
            )
        }
        records.append(
            _record(
                spec,
                raw_source_id=raw_id,
                query=_text(row.get("prompt"), label="MBPP prompt"),
                private_payload={
                    "canonical_code": row.get("code"),
                    "official_row": official_row,
                    "test_imports": row.get("test_imports"),
                    "tests": row.get("test", row.get("test_list")),
                },
                public_context={"code_signature": _code_signature(row)},
                task_family_suffix="python-code",
                environment_suffix="evalplus-sandbox",
            )
        )
    return tuple(records)


def _humaneval_records(
    spec: BenchmarkPopulation, rows: Iterable[Mapping[str, object]]
) -> tuple[ProtocolV10SourceRecord, ...]:
    records = []
    for row in rows:
        prompt = _text(row.get("prompt"), label="HumanEval prompt")
        canonical_solution = _text(
            row.get("canonical_solution"), label="HumanEval canonical solution"
        )
        test = _text(row.get("test"), label="HumanEval tests")
        entry_point = _text(row.get("entry_point"), label="HumanEval entry point")
        records.append(
            _record(
                spec,
                raw_source_id=_text(row.get("task_id"), label="HumanEval task ID"),
                query=prompt,
                private_payload={
                    "canonical_code": f"{prompt}{canonical_solution}",
                    "entry_point": entry_point,
                    "evaluator_kind": "humaneval-sandbox",
                    "tests": test,
                },
                public_context={"code_signature": _code_signature(row)},
                task_family_suffix="python-code",
                environment_suffix="python-sandbox",
            )
        )
    return tuple(records)


def _unique_code_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Keep one result-blind row per public prompt and function signature."""

    unique: list[Mapping[str, object]] = []
    prompts: set[str] = set()
    signatures: set[str] = set()
    for row in sorted(rows, key=lambda item: str(item.get("task_id"))):
        prompt = " ".join(_text(row.get("prompt"), label="code prompt").casefold().split())
        signature = _code_signature(row)
        if prompt in prompts or signature in signatures:
            continue
        prompts.add(prompt)
        signatures.add(signature)
        unique.append(row)
    return tuple(unique)


def load_protocol_v10_source_records(
    protocol: ActiveBenchmarkProtocolV10,
    data_root: Path,
) -> tuple[tuple[BenchmarkPopulation, tuple[ProtocolV10SourceRecord, ...]], ...]:
    """Load every frozen population from one already prepared server data root."""

    if not isinstance(protocol, ActiveBenchmarkProtocolV10):
        raise TypeError("source materialization requires the active Protocol 10")
    if not data_root.is_absolute() or not data_root.is_dir():
        raise ValueError("Protocol 10 data root must be an existing absolute directory")

    raw_root = data_root / "imports-20260817" / "protocol10-raw-local"
    hotpot = cast(DatasetDict, load_from_disk(str(raw_root / "hotpotqa")))
    hotpot_train_spec = _spec(protocol, "hotpotqa-v1.1-train-main")
    hotpot_val_spec = _spec(protocol, "hotpotqa-v1.1-train-holdout")
    hotpot_final_spec = _spec(protocol, "hotpotqa-v1.1-dev-distractor")
    hotpot_all = _hotpot_records(hotpot_train_spec, hotpot["train"])
    hotpot_train, hotpot_val = _partition(hotpot_all, validation_count=512)
    hotpot_val = tuple(
        ProtocolV10SourceRecord(
            item.source_id,
            _task(
                hotpot_val_spec,
                raw_source_id=item.source_id.split("/", 1)[1],
                query=item.task.query,
                public_context={
                    "question_type": cast(dict[str, JsonValue], item.task.public_context)[
                        "question_type"
                    ]
                },
            ),
            item.private_payload,
        )
        for item in hotpot_val
    )
    hotpot_final = _hotpot_records(hotpot_final_spec, hotpot["validation"])

    trivia_train_spec = _spec(protocol, "triviaqa-v1.0-unfiltered-train-main")
    trivia_val_spec = _spec(protocol, "triviaqa-v1.0-unfiltered-train-holdout")
    trivia_final_spec = _spec(protocol, "triviaqa-v1.0-unfiltered-dev")
    trivia_rows = pq.read_table(data_root / "sources/triviaqa/train.parquet").to_pylist()
    trivia_all = _trivia_records(trivia_train_spec, trivia_rows)
    trivia_train, trivia_val_raw = _partition(trivia_all, validation_count=512)
    trivia_validation_ids = {item.source_id for item in trivia_val_raw}
    trivia_val = _trivia_records(
        trivia_val_spec,
        [row for row in trivia_rows if f"triviaqa/{row['question_id']}" in trivia_validation_ids],
    )
    trivia_final = _trivia_records(
        trivia_final_spec,
        pq.read_table(data_root / "sources/triviaqa/validation.parquet").to_pylist(),
    )

    health_train_spec = _spec(protocol, "independent-health-dialogue-train-v1")
    health_val_spec = _spec(protocol, "independent-health-dialogue-validation-v1")
    health_final_spec = _spec(protocol, "healthbench-full-5000")
    health_all = _health_training_records(
        health_train_spec, data_root / "sources/health-training/train.parquet"
    )
    health_train, health_val_raw = _partition(health_all, validation_count=512)
    health_val = tuple(
        ProtocolV10SourceRecord(
            item.source_id,
            _task(
                health_val_spec,
                raw_source_id=item.source_id.split("/", 1)[1],
                query=item.task.query,
                public_context={
                    "message_count": cast(dict[str, JsonValue], item.task.public_context)[
                        "message_count"
                    ]
                },
                task_family_suffix="health-dialogue",
                model_visible_messages=item.task.model_visible_messages,
            ),
            item.private_payload,
        )
        for item in health_val_raw
    )
    health_snapshot = next(
        (
            data_root
            / "imports-20260817/hf-protocol10-local/hub/datasets--openai--healthbench/snapshots"
        ).iterdir()
    )
    health_final = _healthbench_records(
        health_final_spec,
        health_snapshot / "2025-05-07-06-14-12_oss_eval.jsonl",
    )

    goals = _load_webshop_runtime_goals(
        data_root / "prepared/webshop-full-streaming-v1/goals.jsonl"
    )
    webshop_resets = _load_webshop_public_resets(
        data_root / "prepared/webshop-full-streaming-v1/protocol_v10_public_resets.json"
    )
    order = _unique_webshop_goal_indices(goals)
    if len(order) <= 11_000:
        raise ProtocolV10SourceError("WebShop has too few unique goals for frozen splits")
    random.Random(0).shuffle(order)  # noqa: S311 - frozen result-blind split
    webshop_specs = (
        (_spec(protocol, "webshop-goals-train-v1"), order[:10_000]),
        (_spec(protocol, "webshop-goals-validation-v1"), order[10_000:11_000]),
        (_spec(protocol, "webshop-goals-test-v1"), order[11_000:]),
    )

    alf_root = data_root / "prepared/alfworld/json_2.1.1"
    alf_resets = _load_alfworld_public_resets(
        data_root / "prepared/alfworld/protocol_v10_public_resets.json"
    )
    alf_train_spec = _spec(protocol, "alfworld-train-main")
    alf_val_spec = _spec(protocol, "alfworld-train-holdout")
    alf_seen_spec = _spec(protocol, "alfworld-valid-seen")
    alf_unseen_spec = _spec(protocol, "alfworld-valid-unseen")
    alf_seen = _alfworld_records(
        alf_seen_spec,
        tuple((alf_root / "valid_seen").glob("*/*/traj_data.json")),
        alf_root,
        alf_resets,
    )
    alf_unseen = _alfworld_records(
        alf_unseen_spec,
        tuple((alf_root / "valid_unseen").glob("*/*/traj_data.json")),
        alf_root,
        alf_resets,
    )
    alf_train_all = _exclude_alfworld_evaluation_scenarios(
        _alfworld_records(
            alf_train_spec,
            tuple((alf_root / "train").glob("*/*/traj_data.json")),
            alf_root,
            alf_resets,
        ),
        (*alf_seen, *alf_unseen),
    )
    alf_train, alf_val_raw = _partition(alf_train_all, validation_count=512)
    alf_val = _alfworld_records(
        alf_val_spec,
        tuple(
            alf_root
            / _text(
                cast(dict[str, JsonValue], item.private_payload)["trajectory_relative_path"],
                label="ALFWorld trajectory route",
            )
            / "traj_data.json"
            for item in alf_val_raw
        ),
        alf_root,
        alf_resets,
    )
    sheet_train_spec = _spec(protocol, "independent-spreadsheet-train-v1")
    sheet_val_spec = _spec(protocol, "independent-spreadsheet-validation-v1")
    sheet_final_spec = _spec(protocol, "spreadsheetbench-v1-verified-400")
    sheet_train_all = _spreadsheet_training_records(
        sheet_train_spec, data_root / "sources/spreadsheet-training/train.parquet"
    )
    sheet_train, sheet_val_raw = _partition(sheet_train_all, validation_count=512)
    sheet_val = tuple(
        ProtocolV10SourceRecord(
            item.source_id,
            _task(
                sheet_val_spec,
                raw_source_id=item.source_id.split("/", 1)[1],
                query=item.task.query,
                public_context={
                    "tool_schema": cast(dict[str, JsonValue], item.task.public_context)[
                        "tool_schema"
                    ],
                    "workbook_structure_id": cast(dict[str, JsonValue], item.task.public_context)[
                        "workbook_structure_id"
                    ],
                },
                available_tools=("spreadsheet.execute", "submit"),
                task_family_suffix="spreadsheet-editing",
                environment_suffix="spreadsheet-runtime",
            ),
            item.private_payload,
        )
        for item in sheet_val_raw
    )
    sheet_final = _spreadsheetbench_records(
        sheet_final_spec,
        data_root
        / "prepared/spreadsheetbench-v1-verified-400/spreadsheetbench_verified_400/dataset.json",
    )

    app_root = data_root / "prepared/appworld/data"
    app_populations = tuple(
        (
            _spec(protocol, population_id),
            _appworld_records(_spec(protocol, population_id), app_root, split),
        )
        for population_id, split in (
            ("appworld-train", "train"),
            ("appworld-dev", "dev"),
            ("appworld-test-normal", "test_normal"),
            ("appworld-test-challenge", "test_challenge"),
        )
    )

    humaneval = cast(DatasetDict, load_from_disk(str(raw_root / "humaneval")))
    mbpp_plus = cast(DatasetDict, load_from_disk(str(raw_root / "mbppplus")))
    mbpp_train_spec = _spec(protocol, "independent-code-train-excluding-all-mbpp-plus")
    mbpp_val_spec = _spec(protocol, "independent-code-validation-excluding-all-mbpp-plus")
    mbpp_final_spec = _spec(protocol, "mbpp-plus-fixed-100-v1")
    plus_rows = tuple(mbpp_plus["test"])
    plus_signatures = {_code_signature(row) for row in plus_rows}
    plus_prompts = {
        " ".join(_text(row.get("prompt"), label="MBPP+ prompt").casefold().split())
        for row in plus_rows
    }
    independent_rows = _unique_code_rows(
        row
        for row in humaneval["test"]
        if _code_signature(row) not in plus_signatures
        and " ".join(_text(row.get("prompt"), label="HumanEval prompt").casefold().split())
        not in plus_prompts
    )
    independent = _humaneval_records(mbpp_train_spec, independent_rows)
    mbpp_train, mbpp_val_raw = _partition(independent, validation_count=32)
    mbpp_val = tuple(
        ProtocolV10SourceRecord(
            item.source_id,
            _task(
                mbpp_val_spec,
                raw_source_id=item.source_id.split("/", 1)[1],
                query=item.task.query,
                public_context={
                    "code_signature": cast(dict[str, JsonValue], item.task.public_context)[
                        "code_signature"
                    ]
                },
                task_family_suffix="python-code",
                environment_suffix="python-sandbox",
            ),
            item.private_payload,
        )
        for item in mbpp_val_raw
    )
    fixed = sorted(
        plus_rows,
        key=lambda row: int(str(row["task_id"]).rsplit("/", 1)[-1]),
    )[:100]
    mbpp_final = _mbpp_records(mbpp_final_spec, fixed)

    aime_history = data_root / "sources/aime/aime_1983_2025.parquet"
    aime_2026 = data_root / "sources/AIME2026/aime_2026_problems.parquet"
    if not aime_history.is_file() or not aime_2026.is_file():
        raise ProtocolV10SourceError("frozen AIME history or AIME 2026 source is absent")
    history = pq.read_table(aime_history).to_pylist()
    aime_train_spec = _spec(protocol, "aime-1983-2024")
    aime_val_spec = _spec(protocol, "aime-2025")
    aime_final_spec = _spec(protocol, "aime-2026-all-30")
    aime_train = _aime_records(
        aime_train_spec, (row for row in history if int(row["year"]) <= 2024)
    )
    aime_val = _aime_records(aime_val_spec, (row for row in history if int(row["year"]) == 2025))
    aime_final = _aime_records(aime_final_spec, pq.read_table(aime_2026).to_pylist())

    populations = [
        (hotpot_train_spec, hotpot_train),
        (hotpot_val_spec, hotpot_val),
        (hotpot_final_spec, hotpot_final),
        (trivia_train_spec, trivia_train),
        (trivia_val_spec, trivia_val),
        (trivia_final_spec, trivia_final),
        (aime_train_spec, aime_train),
        (aime_val_spec, aime_val),
        (aime_final_spec, aime_final),
        (health_train_spec, health_train),
        (health_val_spec, health_val),
        (health_final_spec, health_final),
        *[
            (spec, _webshop_records(spec, goals, indices, webshop_resets))
            for spec, indices in webshop_specs
        ],
        (alf_train_spec, alf_train),
        (alf_val_spec, alf_val),
        (alf_seen_spec, alf_seen),
        (alf_unseen_spec, alf_unseen),
        (sheet_train_spec, sheet_train),
        (sheet_val_spec, sheet_val),
        (sheet_final_spec, sheet_final),
        *app_populations,
        (mbpp_train_spec, mbpp_train),
        (mbpp_val_spec, mbpp_val),
        (mbpp_final_spec, mbpp_final),
    ]
    expected = tuple(
        population for benchmark in protocol.benchmarks for population in benchmark.populations
    )
    routes = {spec.population_id: (spec, records) for spec, records in populations}
    if set(routes) != {spec.population_id for spec in expected}:
        raise ProtocolV10SourceError("source readers do not cover Protocol 10")
    return tuple(routes[spec.population_id] for spec in expected)


__all__ = [
    "ALFWORLD_PUBLIC_RESET_FORMAT",
    "ProtocolV10SourceError",
    "load_protocol_v10_source_records",
]
