from __future__ import annotations

import json
from pathlib import Path

from skillev_private.benchmarks.protocol_v10_sources import (
    ALFWORLD_PUBLIC_RESET_FORMAT,
    _alfworld_records,
    _load_alfworld_public_resets,
)

from skillev.experiments import BenchmarkV10, PopulationRole, load_active_protocol_v10

ROOT = Path(__file__).parents[2]


def test_alfworld_source_binds_official_public_reset_without_private_truth(
    tmp_path: Path,
) -> None:
    protocol = load_active_protocol_v10(ROOT / "configs/evaluation/protocol_v10.yaml")
    spec = protocol.benchmark(BenchmarkV10.ALFWORLD).population(PopulationRole.TRAINING)[0]
    alf_root = tmp_path / "json_2.1.1"
    trajectory = alf_root / "train" / "pick_and_place-simple" / "trial-1" / "traj_data.json"
    trajectory.parent.mkdir(parents=True)
    (trajectory.parent / "game.tw-pddl").write_text("compiled game", encoding="utf-8")
    trajectory.write_text(
        json.dumps(
            {"turk_annotations": {"anns": [{"task_desc": "Place the public object on the table."}]}}
        ),
        encoding="utf-8",
    )
    resets_path = tmp_path / "public-resets.json"
    resets_path.write_text(
        json.dumps(
            {
                "format": ALFWORLD_PUBLIC_RESET_FORMAT,
                "records": {
                    "train/pick_and_place-simple/trial-1": {
                        "admissible_commands": ["look", "put object on table"],
                        "game_id": "train/pick_and_place-simple/trial-1",
                        "initial_observation": "You are in a public room.",
                        "instruction_text": "Place the public object on the table.",
                        "max_steps": 12,
                        "seed": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    records = _alfworld_records(
        spec,
        (trajectory,),
        alf_root,
        _load_alfworld_public_resets(resets_path),
    )

    task = records[0].task
    assert task.public_context["initial_observation"] == "You are in a public room."
    assert task.public_context["admissible_commands"] == ["look", "put object on table"]
    assert "success" not in json.dumps(task.to_value())
    assert records[0].private_payload["game_id"] == "train/pick_and_place-simple/trial-1"


def test_alfworld_source_excludes_uncompiled_trajectory(tmp_path: Path) -> None:
    protocol = load_active_protocol_v10(ROOT / "configs/evaluation/protocol_v10.yaml")
    spec = protocol.benchmark(BenchmarkV10.ALFWORLD).population(PopulationRole.TRAINING)[0]
    alf_root = tmp_path / "json_2.1.1"
    trajectory = alf_root / "train" / "pick_and_place-simple" / "trial-1" / "traj_data.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps({"turk_annotations": {"anns": [{"task_desc": "An uncompiled task."}]}}),
        encoding="utf-8",
    )

    assert _alfworld_records(spec, (trajectory,), alf_root, {}) == ()
