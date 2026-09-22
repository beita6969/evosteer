#!/usr/bin/env python3
"""Prepare model-visible Protocol 10 ALFWorld reset context on the server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from skillev_private.benchmarks.official_process import (
    ALFWorldGameDeployment,
    OfficialALFWorldProcessFactory,
    PinnedOfficialProcess,
)
from skillev_private.benchmarks.official_process_preparation import (
    enumerate_official_alfworld_tasks,
    load_official_alfworld_task_families,
)
from skillev_private.benchmarks.protocol_v10_sources import ALFWORLD_PUBLIC_RESET_FORMAT


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def _train_eval(relative_path: str) -> str:
    split = relative_path.split("/", 1)[0]
    if split == "train":
        return "train"
    if split == "valid_seen":
        return "eval_in_distribution"
    if split == "valid_unseen":
        return "eval_out_of_distribution"
    raise ValueError("ALFWorld source contains an unsupported split")


def main() -> None:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    alf_root = (args.data_root / "prepared/alfworld/json_2.1.1").resolve()
    source_trajectories = sorted(
        (
            *alf_root.glob("train/*/*/traj_data.json"),
            *alf_root.glob("valid_seen/*/*/traj_data.json"),
            *alf_root.glob("valid_unseen/*/*/traj_data.json"),
        )
    )
    trajectories = tuple(
        trajectory
        for trajectory in source_trajectories
        if len(tuple(trajectory.parent.glob("game.tw-pddl"))) == 1
    )
    if not trajectories:
        raise FileNotFoundError("Protocol 10 ALFWorld trajectories are absent")
    games_by_split: dict[str, dict[str, ALFWorldGameDeployment]] = {
        "train": {},
        "eval_in_distribution": {},
        "eval_out_of_distribution": {},
    }
    for trajectory in trajectories:
        relative = trajectory.parent.relative_to(alf_root).as_posix()
        value = json.loads(trajectory.read_text(encoding="utf-8"))
        instruction = value["turk_annotations"]["anns"][0]["task_desc"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("ALFWorld task instruction is unavailable")
        train_eval = _train_eval(relative)
        games_by_split[train_eval][relative] = ALFWorldGameDeployment(
            data_directory=trajectory.parent,
            train_eval=train_eval,
            instruction_text=instruction,
        )
    runtime = PinnedOfficialProcess(
        interpreter_path=args.interpreter,
        source_root=args.source_root,
        source_revision=args.source_revision,
        request_timeout_seconds=args.timeout_seconds,
    )
    records = []
    completed_before_split = 0
    total_games = sum(len(games) for games in games_by_split.values())
    started_at = time.monotonic()
    for games in games_by_split.values():
        if not games:
            continue
        factory = OfficialALFWorldProcessFactory(
            runtime=runtime,
            config_path=args.config,
            games=games,
            seed=0,
        )
        split_start = completed_before_split

        def report_progress(
            completed: int,
            _split_total: int,
            split_offset: int = split_start,
        ) -> None:
            total_completed = split_offset + completed
            elapsed = max(time.monotonic() - started_at, 1e-9)
            rate = total_completed / elapsed
            eta = (total_games - total_completed) / rate if rate > 0.0 else None
            print(
                json.dumps(
                    {
                        "completed": total_completed,
                        "eta_seconds": eta,
                        "rate_games_per_second": rate,
                        "total": total_games,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )

        records.extend(
            enumerate_official_alfworld_tasks(
                factory,
                task_families=load_official_alfworld_task_families(factory),
                max_steps=args.max_steps,
                progress_observer=report_progress,
            )
        )
        completed_before_split += len(games)
    output = {
        "format": ALFWORLD_PUBLIC_RESET_FORMAT,
        "records": {
            record.game_id: {
                "admissible_commands": list(record.admissible_commands),
                "game_id": record.game_id,
                "initial_observation": record.initial_observation,
                "instruction_text": record.query,
                "max_steps": record.max_steps,
                "seed": 0,
            }
            for record in records
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
    print(json.dumps({"prepared_records": len(records)}))


if __name__ == "__main__":
    main()
