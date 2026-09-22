#!/usr/bin/env python3
"""Build deterministic train-only WebShop/ALFWorld ReAct validation panels.

The script consumes private official-runtime manifests and prompt assets but
writes no benchmark content to the repository.  WebShop uses train-range goal
indices; ALFWorld uses train games with final scenarios and demonstrations
excluded.  The resulting manifests remain private runner inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import cast

import yaml

_TASK_TYPES = {
    "pick_and_place_simple": "pick_and_place",
    "pick_two_obj_and_place": "pick_two_obj",
    "pick_clean_then_place_in_recep": "pick_clean_then_place",
    "pick_heat_then_place_in_recep": "pick_heat_then_place",
    "pick_cool_then_place_in_recep": "pick_cool_then_place",
    "look_at_obj_in_light": "look_at_obj",
}


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _examples(path: Path) -> tuple[dict[str, object], ...]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("examples"), list):
        raise ValueError("interactive prompt asset is invalid")
    return tuple(_mapping(item, "prompt example") for item in value["examples"])


def _alfworld_task_text(row: dict[str, object]) -> str:
    annotations = _mapping(row.get("turk_annotations"), "trajectory annotations")
    values = annotations.get("anns")
    if not isinstance(values, list) or not values:
        raise ValueError("trajectory task description is absent")
    first = _mapping(values[0], "trajectory annotation")
    task = first.get("task_desc")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("trajectory task description is invalid")
    return task


def build_webshop_validation(
    final: dict[str, object],
    *,
    asset_path: Path,
    count: int,
    selection_seed: int,
) -> dict[str, object]:
    if count < 1:
        raise ValueError("WebShop validation count must be positive")
    result = copy.deepcopy(final)
    cases = final.get("cases")
    deployments = _mapping(final.get("deployments"), "WebShop deployments")
    if not isinstance(cases, list):
        raise ValueError("WebShop final cases are absent")
    deployment = _mapping(deployments.get("webshop"), "WebShop deployment")
    goals_path = Path(str(deployment["goals_path"]))
    goals = [json.loads(line) for line in goals_path.read_text(encoding="utf-8").splitlines()]
    random.Random(233).shuffle(goals)  # noqa: S311 - frozen benchmark ordering
    final_indices = {
        _integer(
            _mapping(case.get("payload"), "final WebShop payload")["goal_index"],
            "final WebShop goal index",
        )
        for case in cases
        if isinstance(case, dict) and case.get("benchmark") == "webshop"
    }
    demo_indices = {
        int(str(example["source_entry_id"]).removeprefix("goal-"))
        for example in _examples(asset_path)
    }
    candidates = [
        index
        for index in range(1500, len(goals))
        if index not in final_indices and index not in demo_indices
    ]
    random.Random(selection_seed).shuffle(candidates)  # noqa: S311 - frozen panel selection
    selected = candidates[:count]
    if len(selected) != count:
        raise ValueError("insufficient final-disjoint WebShop train goals")
    rows: list[dict[str, object]] = []
    for panel_index, goal_index in enumerate(selected):
        goal = _mapping(goals[goal_index], "WebShop train goal")
        instruction = goal.get("instruction_text")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("WebShop train instruction is invalid")
        rows.append(
            {
                "benchmark": "webshop",
                "deployment": "webshop",
                "max_steps": 10,
                "payload": {
                    "environment_id": "webshop-official",
                    "goal_id": f"goal-{goal_index}",
                    "goal_index": goal_index,
                    "session_id": f"react-v6-validation-{panel_index:03d}",
                    "split": "train-holdout",
                },
                "source_identity": f"WebShop/goal-{goal_index}",
                "task": instruction,
                "task_id": f"webshop:react-v6-validation:{panel_index:03d}",
            }
        )
    result.update(
        {
            "benchmark": "webshop",
            "population_id": f"webshop-train-final-disjoint-{count}-react-v6",
            "selection_rule": f"random-{selection_seed}-train-final-and-demo-disjoint",
            "panel_manifest_id": f"webshop-react-validation-{count}-v6",
            "cases": rows,
        }
    )
    contracts = _mapping(result["benchmarks"], "benchmark contracts")
    _mapping(contracts["webshop"], "WebShop contract")["expected_count"] = count
    return result


def build_alfworld_validation(
    final: dict[str, object],
    *,
    asset_path: Path,
    count: int,
    selection_seed: int,
) -> dict[str, object]:
    if count < len(_TASK_TYPES):
        raise ValueError("ALFWorld validation must cover all six task types")
    result = copy.deepcopy(final)
    cases = final.get("cases")
    deployments = _mapping(final.get("deployments"), "ALFWorld deployments")
    if not isinstance(cases, list):
        raise ValueError("ALFWorld final cases are absent")
    deployment = _mapping(deployments.get("alfworld"), "ALFWorld deployment")
    games = _mapping(deployment.get("games"), "ALFWorld final games")
    first_game = _mapping(next(iter(games.values())), "ALFWorld final game")
    root = Path(str(first_game["data_directory"])).parents[2]
    final_scenarios = {
        str(_mapping(case.get("payload"), "final ALFWorld payload")["game_id"]).split("/", 2)[1]
        for case in cases
        if isinstance(case, dict) and case.get("benchmark") == "alfworld"
    }
    demo_ids = {str(example["source_entry_id"]) for example in _examples(asset_path)}
    grouped: dict[str, list[tuple[str, Path, str]]] = {
        task_type: [] for task_type in _TASK_TYPES.values()
    }
    for path in sorted((root / "train").glob("*/*/traj_data.json")):
        if not (path.parent / "game.tw-pddl").is_file():
            continue
        relative = path.parent.relative_to(root / "train").as_posix()
        scenario = relative.split("/", 1)[0]
        if relative in demo_ids or scenario in final_scenarios:
            continue
        trajectory = _load_json(path)
        task_type = _TASK_TYPES.get(str(trajectory.get("task_type")))
        if task_type is not None:
            grouped[task_type].append((relative, path.parent, _alfworld_task_text(trajectory)))

    ordered_types = tuple(_TASK_TYPES.values())
    base, remainder = divmod(count, len(ordered_types))
    selected: list[tuple[str, str, Path, str]] = []
    for type_index, task_type in enumerate(ordered_types):
        candidates = grouped[task_type]
        random.Random(selection_seed + type_index).shuffle(  # noqa: S311 - frozen panel
            candidates
        )
        needed = base + (type_index < remainder)
        if len(candidates) < needed:
            raise ValueError(f"insufficient ALFWorld train games for {task_type}")
        selected.extend((task_type, *item) for item in candidates[:needed])
    random.Random(selection_seed).shuffle(selected)  # noqa: S311 - frozen panel selection

    rows: list[dict[str, object]] = []
    selected_games: dict[str, object] = {}
    for panel_index, (task_type, relative, directory, instruction) in enumerate(selected):
        game_id = f"train/{relative}"
        selected_games[game_id] = {
            "data_directory": str(directory),
            "instruction_text": instruction,
            "train_eval": "train",
        }
        rows.append(
            {
                "benchmark": "alfworld",
                "deployment": "alfworld",
                "max_steps": 20,
                "payload": {
                    "environment_id": "alfworld-official",
                    "game_id": game_id,
                    "seed": 0,
                    "split": "train-holdout",
                    "task_type": task_type,
                },
                "source_identity": f"ALFWorld/{game_id}",
                "task": instruction,
                "task_id": f"alfworld:react-v6-validation:{panel_index:03d}",
            }
        )
    result.update(
        {
            "benchmark": "alfworld",
            "population_id": f"alfworld-train-final-scenario-disjoint-{count}-react-v6",
            "selection_rule": (
                f"stratified-random-{selection_seed}-train-final-scenario-and-demo-disjoint"
            ),
            "panel_manifest_id": f"alfworld-react-validation-{count}-v6",
            "cases": rows,
        }
    )
    selected_deployment = _mapping(
        _mapping(result["deployments"], "result deployments")["alfworld"],
        "result ALFWorld deployment",
    )
    selected_deployment["games"] = selected_games
    contracts = _mapping(result["benchmarks"], "benchmark contracts")
    _mapping(contracts["alfworld"], "ALFWorld contract")["expected_count"] = count
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--webshop-count", type=int, default=64)
    parser.add_argument("--alfworld-count", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    values = {
        "webshop": build_webshop_validation(
            _load_json(arguments.manifest_dir / "webshop.runtime-v6-22048.json"),
            asset_path=arguments.asset_dir / "webshop_native_react_v6.yaml",
            count=arguments.webshop_count,
            selection_seed=arguments.selection_seed,
        ),
        "alfworld": build_alfworld_validation(
            _load_json(arguments.manifest_dir / "alfworld.runtime-v6-22048.json"),
            asset_path=arguments.asset_dir / "alfworld_native_react_v6.yaml",
            count=arguments.alfworld_count,
            selection_seed=arguments.selection_seed,
        ),
    }
    for benchmark, value in values.items():
        path = arguments.output_dir / f"{benchmark}.react-validation-v6.json"
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(benchmark, len(cast(list[object], value["cases"])))


if __name__ == "__main__":
    main()
