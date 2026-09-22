"""Frozen IID public inputs and native terminal cases, never training records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from skillev.contracts import JsonValue, normalize_json
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.input_metric_contracts import (
    HISTORICAL_IID_BENCHMARKS,
    IID_BENCHMARKS,
    PublicTaskView,
)
from skillev.evolution.task_features import public_task_features
from skillev.rollout import ModelVisibleMessage, RolloutTask
from skillev_private.benchmarks.evaluation_episode import (
    EvaluationEpisodeRecord,
    EvaluationSource,
    EvaluationTarget,
)

from .integrity_sources import SourcePanel


@dataclass(frozen=True, slots=True)
class IIDSourceIdentity:
    task_id: str
    benchmark_id: str
    source_question_id: str
    population_id: str
    dataset_revision: str
    split: str

    def __post_init__(self) -> None:
        # A saved source identity remains readable after a catalog migration.
        # Admission to newly loaded panels is governed by integrity_sources.
        if self.benchmark_id not in HISTORICAL_IID_BENCHMARKS or not all(
            (
                self.task_id,
                self.source_question_id,
                self.population_id,
                self.dataset_revision,
                self.split,
            )
        ):
            raise ValueError(
                "native IID source revision, split and canonical identity are required"
            )

    @property
    def canonical_source(self) -> tuple[str, str]:
        return self.benchmark_id, self.source_question_id


def canonical_source_key(
    source: tuple[str, str], aliases: Mapping[str, Mapping[str, str]] | None = None
) -> tuple[str, str]:
    """Resolve native manifest aliases, not guessed or renamed populations.

    HealthBench's released UUID and training ``healthbench/<UUID>`` key need
    the same declared source mapping, as do ALFWorld source and game IDs.
    """
    domain, identity = source
    return domain, (aliases or {}).get(domain, {}).get(identity, identity)


def require_source_isolation(
    entries: tuple[PublicTaskView, ...],
    identities: tuple[IIDSourceIdentity, ...],
    excluded: frozenset[tuple[str, str]],
    aliases: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    if tuple((entry.task_id, entry.benchmark) for entry in entries) != tuple(
        (row.task_id, row.benchmark_id) for row in identities
    ):
        raise ValueError("source declaration must preserve every frozen panel position")
    canonical = [canonical_source_key(row.canonical_source, aliases) for row in identities]
    if len(set(canonical)) != len(canonical):
        raise ValueError("IID baseline requires distinct source questions, not renamed populations")
    if set(canonical) & {canonical_source_key(row, aliases) for row in excluded}:
        raise ValueError("IID baseline overlaps declared training or development sources")


def public_episode_task(entry: PublicTaskView, source: IIDSourceIdentity) -> RolloutTask:
    """Only PublicTaskView is available here, not a source row or evaluator target."""
    if entry.benchmark != source.benchmark_id or entry.task_id != source.task_id:
        raise ValueError("public input and native source identity differ")
    features = public_task_features(entry.benchmark)
    values = dict(entry.fields)
    messages: tuple[ModelVisibleMessage, ...] = ()
    if entry.input_profile != "released-iid-source@1":
        raise ValueError("architecture-matched IID requires native released evaluation inputs")
    if entry.benchmark == "healthbench":
        messages = tuple(ModelVisibleMessage(**row) for row in entry.conversation())
        query = "\n\n".join(f"{m.role}: {m.content}" for m in messages)
        payload: JsonValue = {"message_count": len(messages)}
    elif entry.benchmark in {"mbpp-plus", "humaneval"}:
        query, payload = values["prompt"], {"language": "python"}
    elif entry.benchmark == "aime-2026":
        query, payload = values["problem"], {"benchmark_slice": "AIME2026"}
    elif entry.benchmark == "alfworld":
        query, payload = values["task"], {"observation_format": "official-text"}
    else:
        # Preserve every released passage in order. In particular Trivia's
        # reading context must not silently become closed-book training input.
        query, payload = entry.render(), {"context_in_query": True}
    return RolloutTask(
        task_id=entry.task_id,
        environment_id=f"benchmark:{entry.benchmark}@{source.dataset_revision}",
        task_family=features.task_family,
        context_id=features.context_id,
        query=query,
        available_tools=("act",) if entry.benchmark == "alfworld" else (),
        public_context={
            "benchmark_id": entry.benchmark,
            "dataset_revision": source.dataset_revision,
            "split": source.split,
            "payload": payload,
            "input_profile": entry.input_profile,
        },
        model_visible_messages=messages,
    )


def _alf_route(specification: dict[str, Any], *, max_turns: int, seed: int) -> dict[str, JsonValue]:
    from skillev_private.benchmarks.official_process import _single_alfworld_game

    case, manifest = specification["case"], specification["manifest"]
    deployment = manifest["deployments"][case["deployment"]]
    game = deployment["games"][case["payload"]["game_id"]]
    return {
        "environment_route": {
            "game_file": str(_single_alfworld_game(Path(game["data_directory"]))),
            "config_file": deployment["config_path"],
            "seed": seed,
            "max_steps": max_turns,
            "mode": game["train_eval"],
        }
    }


def evaluation_records(
    source: SourcePanel,
    identities: tuple[IIDSourceIdentity, ...],
    *,
    max_turns: int,
    seed: int,
) -> tuple[EvaluationEpisodeRecord, ...]:
    require_source_isolation(source.panel.entries, identities, frozenset())
    records = []
    for entry, identity in zip(source.panel.entries, identities, strict=True):
        benchmark = Protocol13Benchmark(entry.benchmark)
        task = public_episode_task(entry, identity)
        target = source.targets.get(entry.task_id, {})
        if entry.benchmark == "alfworld":
            target = _alf_route(source.interactive[entry.task_id], max_turns=max_turns, seed=seed)
        elif entry.benchmark == "aime-2026":
            target = {"accepted_answers": [target["answer"]]}
        elif entry.benchmark == "mbpp-plus":
            target = target["private_target"]
        elif entry.benchmark == "healthbench":
            target = {"prompt": target["prompt"], "rubrics": target["rubrics"]}
        records.append(
            EvaluationEpisodeRecord(
                EvaluationSource(
                    benchmark, identity.population_id, identity.source_question_id, entry.task_id
                ),
                task,
                EvaluationTarget(cast(dict[str, JsonValue], normalize_json(target))),
            )
        )
    return tuple(records)


def source_identities_from_config(
    config: dict[str, Any], source: SourcePanel
) -> tuple[IIDSourceIdentity, ...]:
    """Native manifests plus explicit split/revision, never generated task-ID guesses."""
    declarations = config["population_provenance"]
    aliases: dict[tuple[str, str], str] = {}
    for name, path in config["manifests"].items():
        if name not in IID_BENCHMARKS:
            continue
        manifest = json.loads(Path(path).read_text())
        for row in manifest.get("entries", []):
            aliases[name, str(row["task_id"])] = str(row["source_identity"])
        for row in manifest.get("cases", []):
            if name == "alfworld":
                aliases[name, str(row["task_id"])] = str(row["payload"]["game_id"])
    return tuple(
        IIDSourceIdentity(
            entry.task_id,
            entry.benchmark,
            entry.task_id
            if entry.benchmark == "healthbench"
            else aliases[entry.benchmark, entry.task_id],
            declarations[entry.benchmark]["population_id"],
            declarations[entry.benchmark]["dataset_revision"],
            declarations[entry.benchmark]["split"],
        )
        for entry in source.panel.entries
    )
