"""Run native owners on the *same* training sources, without exporting targets.

This is an explicitly labelled diagnostic population, not an IID manifest or a
replacement for released TriviaQA RC. Native environment bindings remain trusted
deployment inputs; the actor sees only PublicTaskView and its real reset response.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any, cast

from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS, PublicTaskView
from skillev.evaluation.integrity_pipeline import FrozenPanel
from skillev_private.benchmarks.protocol_v13_training import Protocol13TrainingRecord
from skillev_private.benchmarks.static import parse_aime_answer
from skillev_private.evaluation.integrity_sources import SourcePanel


def native_training_source_panel(
    records: tuple[Protocol13TrainingRecord, ...],
    *,
    interactive: dict[str, dict[str, Any]],
) -> SourcePanel:
    """Keep canonical slots and source IDs, including repeated source questions.

    Only verifier payload *carriers* change to the existing native scorer API. No
    candidate, answer selection, new sampling, or training update happens here.
    Scorer execution profiles must still be frozen and compared by the caller.
    """
    if not records or len({r.input.task_id for r in records}) != len(records):
        raise ValueError("native source bridge requires distinct nonempty rollout slots")
    entries = []
    targets: dict[str, dict[str, Any]] = {}
    slots = []
    environment_ids = set()
    for record in records:
        domain, task = record.episode.benchmark.value, record.input
        if domain not in IID_BENCHMARKS:
            raise ValueError("native bridge accepts only the current IID domains")
        view = PublicTaskView.from_training_task(task)
        if view.benchmark != domain:
            raise ValueError("public input and private source identify different domains")
        entries.append(view)
        slots.append(
            {
                "task_id": task.task_id,
                "benchmark": domain,
                "population_id": record.episode.population_id,
                "source_question_id": record.episode.source_id,
            }
        )
        target = copy.deepcopy(record.output.target)
        if domain == "alfworld":
            environment_ids.add(task.task_id)
            spec = interactive.get(task.task_id, {})
            case = spec.get("case", {})
            if (
                case.get("task_id") != task.task_id
                or case.get("benchmark") != domain
                or case.get("task") != task.query
                or case.get("payload", {}).get("game_id") != record.episode.source_id
            ):
                raise ValueError("native environment must bind the original task and game")
        elif domain == "aime-2026":
            aliases = target.get("accepted_answers")
            if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
                raise ValueError("AIME source requires its original integer aliases")
            values = {parse_aime_answer(a) for a in cast(list[str], aliases)}
            if len(values) != 1 or None in values:
                raise ValueError("AIME native verifier requires one unambiguous integer")
            targets[task.task_id] = {"answer": next(iter(values))}
        elif domain == "mbpp-plus":
            targets[task.task_id] = {
                "private_target": target,
                "reference_prompt": task.query,
                "source_task_id": record.episode.source_id,
            }
        elif domain == "humaneval":
            targets[task.task_id] = {**target, "prompt": task.query}
        else:
            targets[task.task_id] = target
    if set(interactive) != environment_ids:
        raise ValueError("native bridge environment bindings differ from its selected slots")
    provenance: dict[str, object] = {
        "purpose": "zero-update-native-training-source-bridge",
        "input_profile": "training-public-source-bridge@1",
        "sample_unit": "rollout-slot-not-independent-source-question",
        "slots": slots,
        "unique_source_questions": len(
            {(s["benchmark"], s["population_id"], s["source_question_id"]) for s in slots}
        ),
        "target_exposure": "trusted-native-scorer-only",
        "triviaqa_condition": "original-training-public-input-not-released-RC-IID",
    }
    return SourcePanel(
        FrozenPanel(
            tuple(entries),
            exposure="training-source-diagnostic-not-IID",
            source_provenance=json.dumps(provenance, sort_keys=True),
            sample_counts=tuple(Counter(entry.benchmark for entry in entries).items()),
        ),
        targets,
        copy.deepcopy(interactive),
        provenance,
    )
