#!/usr/bin/env python3
"""Aggregate private ALFWorld traces by official split and six-task taxonomy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev_private.benchmarks.alfworld_taxonomy import (
    ALFWorldManifestCase,
    ALFWorldTaskType,
    classify_alfworld_game_id,
    validate_protocol13_alfworld_panel,
)


@dataclass(frozen=True, slots=True)
class ALFWorldAggregateKey:
    split: str
    task_type: str


def aggregate_alfworld_diagnostics(
    manifest_by_task: dict[str, ALFWorldManifestCase],
    rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if set(manifest_by_task) != {str(row.get("task_id")) for row in rows}:
        raise ValueError("ALFWorld result IDs differ from the manifest")
    buckets: dict[ALFWorldAggregateKey, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    for row in rows:
        case = manifest_by_task[str(row["task_id"])]
        bucket = buckets[ALFWorldAggregateKey(case.split, case.task_type.value)]
        for target in (bucket, overall):
            target["planned"] += 1
            target["success"] += row.get("success") is True
            target["candidate_invalid"] += row.get("termination_reason") == "candidate-invalid"
            target["horizon"] += row.get("terminated_by_horizon") is True
            target["budget_exhausted"] += row.get("budget_exhausted") is True
            target["steps"] += int(cast(int, row.get("steps", 0)))
        for step in cast(list[dict[str, object]], row.get("trace") or []):
            action = step.get("parsed_action")
            if isinstance(action, str):
                for target in (bucket, overall):
                    target["inventory"] += action.strip() == "inventory"
                    target["look"] += action.strip() == "look"
                    target["unlisted"] += step.get("action_listed_before") is False
        diagnostics = classify_alfworld_failure(row, case.task_type)
        for target in (bucket, overall):
            target.update(diagnostics)
    return {
        "overall": dict(overall),
        "by_split_and_task": {
            f"{key.split}/{key.task_type}": dict(value)
            for key, value in sorted(
                buckets.items(), key=lambda item: (item[0].split, item[0].task_type)
            )
        },
    }


def classify_alfworld_failure(
    row: dict[str, object],
    task_type: ALFWorldTaskType,
) -> Counter[str]:
    """Return content-free, non-corrective failure diagnostics from public traces."""

    result: Counter[str] = Counter()
    if row.get("success") is not False:
        return result
    trace = cast(list[dict[str, object]], row.get("trace") or [])
    actions = [
        str(step["parsed_action"]).strip()
        for step in trace
        if isinstance(step.get("parsed_action"), str)
    ]
    lowered = [action.casefold() for action in actions]
    result["failure_episode"] += 1
    result["repeated_action_count"] += len(lowered) - len(set(lowered))
    locations = [action.removeprefix("go to ") for action in lowered if action.startswith("go to ")]
    result["location_revisit_count"] += len(locations) - len(set(locations))
    result["unchanged_state_count"] += sum(
        _same_observation(
            str(step.get("observation_before") or ""),
            str(step.get("observation_after") or ""),
        )
        for step in trace
        if step.get("observation_after") is not None
    )
    takes = [action for action in lowered if action.startswith("take ")]
    moves = [action for action in lowered if action.startswith(("move ", "put "))]
    opened = [action.removeprefix("open ") for action in lowered if action.startswith("open ")]
    result["target_object_not_found"] += not takes and not any(
        action.startswith("examine ") for action in lowered
    )
    result["receptacle_not_opened"] += _needed_open_was_skipped(trace, opened)
    process_prefix = {
        ALFWorldTaskType.CLEAN_THEN_PLACE: "clean ",
        ALFWorldTaskType.HEAT_THEN_PLACE: "heat ",
        ALFWorldTaskType.COOL_THEN_PLACE: "cool ",
    }.get(task_type)
    process_done = process_prefix is None or any(
        action.startswith(process_prefix) for action in lowered
    )
    if process_prefix is not None:
        result[f"forgot_{process_prefix.strip()}"] += not process_done
        result["processed_but_wrong_destination"] += process_done and bool(moves)
        process_indexes = [
            index for index, action in enumerate(lowered) if action.startswith(process_prefix)
        ]
        appliance = {
            ALFWorldTaskType.CLEAN_THEN_PLACE: "sinkbasin",
            ALFWorldTaskType.HEAT_THEN_PLACE: "microwave",
            ALFWorldTaskType.COOL_THEN_PLACE: "fridge",
        }[task_type]
        stored_indexes = [
            index
            for index, action in enumerate(lowered)
            if action.startswith(("move ", "put ")) and appliance in action
        ]
        result["stored_object_in_transformation_appliance"] += bool(stored_indexes)
        result["transformation_attempt_after_storage"] += bool(
            process_indexes and stored_indexes and min(stored_indexes) < min(process_indexes)
        )
        result["unlisted_transformation_action"] += sum(
            isinstance(step.get("parsed_action"), str)
            and str(step["parsed_action"]).casefold().startswith(process_prefix)
            and step.get("action_listed_before") is False
            for step in trace
        )
        result["wrong_transformation_action"] += any(
            action.startswith(("clean ", "heat ", "cool ", "cook "))
            and not action.startswith(process_prefix)
            for action in lowered
        )
        result["processed_without_later_placement"] += bool(process_indexes) and not any(
            index > max(process_indexes) and action.startswith(("move ", "put "))
            for index, action in enumerate(lowered)
        )
        appliance_open_close = [
            action
            for action in lowered
            if action.startswith((f"open {appliance}", f"close {appliance}"))
        ]
        result["transformation_appliance_open_close_loop"] += len(appliance_open_close) >= 4
    moved_objects = {action.split(" to ", 1)[0].split(" in/on ", 1)[0] for action in moves}
    result["two_object_only_one_completed"] += (
        task_type is ALFWorldTaskType.PICK_TWO_OBJECTS and len(moved_objects) == 1
    )
    result["inventory_or_world_state_error"] += (
        any(
            step.get("action_listed_before") is False or step.get("official_action_valid") is False
            for step in trace
        )
        or sum(action == "inventory" for action in lowered) > 1
    )
    result["unlisted_action_count"] += sum(
        step.get("action_listed_before") is False for step in trace
    )
    result["simulator_terminal_at_outer_budget"] += (
        row.get("budget_exhausted") is True
        and row.get("terminal_reached") is True
        and row.get("terminated_by_horizon") is not True
    )
    subgoal = _unfinished_subgoal(task_type, lowered, moved_objects)
    result[f"failure_subgoal/{subgoal}"] += 1
    return result


def _same_observation(before: str, after: str) -> bool:
    before = before.split("\n\nAdmissible actions:", 1)[0]
    return " ".join(before.split()).casefold() == " ".join(after.split()).casefold()


def _needed_open_was_skipped(
    trace: list[dict[str, object]],
    opened: list[str],
) -> bool:
    opened_set = set(opened)
    for step in trace:
        available = tuple(
            str(item).casefold()
            for item in cast(list[str], step.get("available_actions_before") or [])
        )
        openable = [
            action.removeprefix("open ") for action in available if action.startswith("open ")
        ]
        action = str(step.get("parsed_action") or "").casefold()
        if openable and action.startswith("take ") and not opened_set.intersection(openable):
            return True
    return False


def _unfinished_subgoal(
    task_type: ALFWorldTaskType,
    actions: list[str],
    moved_objects: set[str],
) -> str:
    if not any(action.startswith("take ") for action in actions):
        return "locate_or_take_target"
    process = {
        ALFWorldTaskType.CLEAN_THEN_PLACE: "clean ",
        ALFWorldTaskType.HEAT_THEN_PLACE: "heat ",
        ALFWorldTaskType.COOL_THEN_PLACE: "cool ",
    }.get(task_type)
    if process is not None and not any(action.startswith(process) for action in actions):
        return process.strip()
    if task_type is ALFWorldTaskType.LOOK_AT_OBJECT and not any(
        action.startswith("examine ") for action in actions
    ):
        return "examine_target"
    if task_type is ALFWorldTaskType.PICK_TWO_OBJECTS and len(moved_objects) < 2:
        return "complete_second_object"
    if task_type is not ALFWorldTaskType.LOOK_AT_OBJECT and not moved_objects:
        return "place_target"
    return "verify_correct_destination_or_object"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    cases: list[ALFWorldManifestCase] = []
    for row in manifest["cases"]:
        if row["benchmark"] != "alfworld":
            continue
        payload = row["payload"]
        game_id = str(payload["game_id"])
        cases.append(
            ALFWorldManifestCase(
                task_id=str(row["task_id"]),
                game_id=game_id,
                split=str(payload["split"]),
                task_type=ALFWorldTaskType(str(payload["task_type"])),
                seed=int(payload["seed"]),
                max_steps=int(row["max_steps"]),
                environment_id=str(payload["environment_id"]),
                instruction=str(row["task"]),
                deployment=str(row["deployment"]),
            )
        )
        if classify_alfworld_game_id(game_id) is not cases[-1].task_type:
            raise AssertionError("taxonomy constructor should have rejected this row")
    panel = tuple(cases)
    validate_protocol13_alfworld_panel(panel)
    rows = tuple(
        cast(dict[str, object], json.loads(line))
        for line in arguments.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    value = aggregate_alfworld_diagnostics({item.task_id: item for item in panel}, rows)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
