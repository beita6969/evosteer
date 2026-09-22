"""Contract-checked loading for scorer-only direct-reference replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev.evaluation.direct_baseline.config import DirectBenchmark, PaperBenchmarkSpec


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"generation journal {label} must be an integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"generation journal {label} must be non-negative")
    return result


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"generation journal {label} must be text or null")
    return value


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    task_id: str
    benchmark: DirectBenchmark
    raw_text: str | None
    response_model: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    prompt_profile_id: str
    parser_profile_id: str
    decoding_profile_id: str
    population_id: str
    run_seed: int
    request_attempt: int
    infrastructure_error: str | None


def load_generation_records(path: Path) -> tuple[GenerationRecord, ...]:
    completion_path = path.parent / "attempt-complete.json"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("replay requires a complete formal attempt") from exc
    if not isinstance(completion, dict) or completion.get("format") != "skillev-direct-attempt@1":
        raise ValueError("replay attempt completion marker is incompatible")
    if completion.get("status") != "complete":
        raise ValueError("replay requires a complete formal attempt")
    records: list[GenerationRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid generation journal line {line_number}") from exc
            if type(value) is not dict:
                raise ValueError("generation journal rows must be objects")
            raw = cast(dict[str, object], value)
            records.append(
                GenerationRecord(
                    task_id=_text(raw.get("task_id"), "task_id"),
                    benchmark=DirectBenchmark(_text(raw.get("benchmark"), "benchmark")),
                    raw_text=_optional_text(raw.get("raw_text"), "raw_text"),
                    response_model=_optional_text(raw.get("response_model"), "response_model"),
                    finish_reason=_optional_text(raw.get("finish_reason"), "finish_reason"),
                    prompt_tokens=_optional_integer(raw.get("prompt_tokens"), "prompt_tokens"),
                    completion_tokens=_optional_integer(
                        raw.get("completion_tokens"), "completion_tokens"
                    ),
                    prompt_profile_id=_text(raw.get("prompt_profile_id"), "prompt profile"),
                    parser_profile_id=_text(raw.get("parser_profile_id"), "parser profile"),
                    decoding_profile_id=_text(raw.get("decoding_profile_id"), "decoding profile"),
                    population_id=_text(raw.get("population_id"), "population"),
                    run_seed=_integer(raw.get("run_seed"), "run seed"),
                    request_attempt=_integer(raw.get("request_attempt"), "request attempt"),
                    infrastructure_error=_optional_text(
                        raw.get("infrastructure_error"), "infrastructure_error"
                    ),
                )
            )
    declared_count = _integer(completion.get("final_record_count"), "final record count")
    if declared_count != len(records):
        raise ValueError("generation journal count differs from completion marker")
    return tuple(records)


def collapse_generation_attempts(
    records: tuple[GenerationRecord, ...],
) -> tuple[GenerationRecord, ...]:
    grouped: dict[str, list[GenerationRecord]] = {}
    order: list[str] = []
    for record in records:
        if record.task_id not in grouped:
            grouped[record.task_id] = []
            order.append(record.task_id)
        grouped[record.task_id].append(record)
    collapsed: list[GenerationRecord] = []
    for task_id in order:
        attempts = grouped[task_id]
        if [record.request_attempt for record in attempts] != list(range(1, len(attempts) + 1)):
            raise ValueError("generation attempts must be consecutive from one")
        if any(record.infrastructure_error is None for record in attempts[:-1]):
            raise ValueError("candidate response cannot have a later retry")
        if any(record.raw_text is not None for record in attempts[:-1]):
            raise ValueError("infrastructure retry cannot carry a candidate")
        collapsed.append(attempts[-1])
    return tuple(collapsed)


def validate_replay_contract(
    record: GenerationRecord,
    spec: PaperBenchmarkSpec,
    *,
    served_model_name: str,
) -> None:
    if record.benchmark is not spec.benchmark:
        raise ValueError("generation benchmark differs from replay contract")
    if record.prompt_profile_id != spec.prompt_profile:
        raise ValueError("generation used another prompt profile")
    if record.parser_profile_id != spec.parser_profile:
        raise ValueError("generation used another parser profile")
    allowed_decoding_profiles = {
        spec.decoding_profile,
        f"{spec.decoding_profile}/seed-{record.run_seed}",
    }
    if record.decoding_profile_id not in allowed_decoding_profiles:
        raise ValueError("generation used another decoding profile")
    if record.population_id != spec.population:
        raise ValueError("generation used another population")
    if record.run_seed not in spec.seed_aggregation.seeds:
        raise ValueError("generation used an undeclared seed")
    if record.infrastructure_error is None:
        if record.raw_text is None:
            raise ValueError("successful generation lacks raw text")
        if record.response_model != served_model_name:
            raise ValueError("generation used another served model")
    elif record.raw_text is not None:
        raise ValueError("infrastructure record cannot contain final candidate")


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"generation journal {label} must be non-empty text")
    return value
