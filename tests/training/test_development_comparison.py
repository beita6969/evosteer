"""Synthetic paired development evidence; no model/scorer requests."""

import copy
import json

import pytest
from skillev_private.experiments import development_collection as collection
from skillev_private.experiments import development_comparison as report

from tests.training.test_development_collection import request_fixture, write
from tests.v3_helpers import make_artifact


def pair_fixture(tmp_path):
    request, _, _ = request_fixture(tmp_path)
    _, frozen = collection.selection(request)
    first = frozen["source_manifest"]["ordered_sources"][0]
    second = dict(first, source_id="synthetic-2", task_id="synthetic-task-2")
    frozen["source_manifest"]["ordered_sources"].append(second)
    roots = [tmp_path / name for name in ("before", "after")]
    for side, root in enumerate(roots):
        episodes = root / "collection/episodes"
        episodes.mkdir(parents=True)
        write(root / "selection-private.json", frozen)
        write(
            root / "controls-private.json",
            {"selection": frozen, "formal": {"reasoning_tokens": 100 * (side + 1)}},
        )
        write(root / "collection/collection.json", {"synthetic": True})
        traces = []
        for i, source in enumerate(frozen["source_manifest"]["ordered_sources"]):
            artifact = make_artifact(f"synthetic-{i}").to_value()
            artifact["manifest"]["task_id"] = source["task_id"]
            reward = artifact["record"]["reward"]
            reward.update(
                value=0.2 + side * 0.3,
                success=bool(side),
                native_metric_name="synthetic-rubric",
                native_payload={"public_metrics": {"synthetic-rubric": 0.2 + side * 0.3}},
            )
            horizon = len(artifact["record"]["steps"])
            artifact["manifest"].update(
                reasoning_token_counts=[5] * horizon,
                reasoning_finish_reasons=["length"] * horizon,
                action_finish_reasons=["stop"] * horizon,
            )
            write(
                episodes / f"episode-{i:06d}-private.json",
                {
                    "position": i,
                    "task_id": source["task_id"],
                    "execution_status": "completed",
                    "infrastructure_error": None,
                    "artifact": artifact,
                },
            )
            traces.append(
                {
                    "trajectory_id": artifact["record"]["trajectory_id"],
                    "canonical_position": i,
                    "phases": [
                        {
                            "phase": "reasoning",
                            "input_tokens": 12,
                            "output_tokens": 5,
                            "elapsed_seconds": 2,
                            "finish_reason": "length",
                        }
                    ],
                }
            )
        write(episodes / "rollout-progress.json", {"trajectories": traces})
        write(root / "summary.json", collection.aggregate(root, frozen, elapsed_seconds=10 + side))
    return roots


def test_all_sources_paired_native_scores_and_original_timing(tmp_path):
    before, after = pair_fixture(tmp_path)
    result = report.compare(before, after)
    assert result["complete_paired_comparison"]
    assert len(result["pairs"]) == 2
    assert all(
        p["reward_delta"] == pytest.approx(0.3) and p["success_delta"] == 1 for p in result["pairs"]
    )
    assert result["domain_deltas"]["healthbench"]["native_metrics"][
        "synthetic-rubric"
    ] == pytest.approx(0.3)
    assert result["before"]["elapsed_seconds"] == 10
    assert result["after"]["elapsed_seconds"] == 11
    assert (
        result["before"]["phase_totals"]["reasoning"]["measurements"]["output_tokens"]["sum"] == 10
    )
    actions = result["pairs"][0]["before"]["actions"]
    assert actions["reasoning_length_cap_hits"] > 0
    assert actions["action_output_tokens"] > 0
    assert actions["admitted_count"] is None
    assert actions["body_visible_input_count"] is None
    assert result["declared_condition_differences"][0]["path"] == "/formal/reasoning_tokens"
    assert "baseline_accepted" not in json.dumps(result)
    assert "synthetic_target" not in json.dumps(result)


@pytest.mark.parametrize("change", ["source", "order", "policy", "arm"])
def test_selection_changes_cannot_be_called_paired(tmp_path, change):
    before, after = pair_fixture(tmp_path)
    selection = json.loads((after / "selection-private.json").read_text())
    if change == "source":
        selection["source_manifest"]["ordered_sources"][0]["source_id"] = "changed"
    elif change == "order":
        selection["source_manifest"]["ordered_sources"].reverse()
    else:
        selection[change] = "changed"
    write(after / "selection-private.json", selection)
    with pytest.raises(ValueError):
        report.compare(before, after)


def test_missing_infrastructure_rows_remain_in_full_denominator(tmp_path, monkeypatch):
    before, after = pair_fixture(tmp_path)
    path = after / "collection/episodes/episode-000001-private.json"
    row = json.loads(path.read_text())
    row.update(
        artifact=None,
        execution_status="not-started",
        infrastructure_error="PreparationInfrastructure",
        preparation_chunk_start=1,
    )
    write(path, row)
    write(
        after / "failure-private.json",
        {"error_type": "PreparationInfrastructure", "message": "private detail"},
    )
    result = report.compare(before, after)
    assert not result["complete_paired_comparison"]
    assert len(result["pairs"]) == 2
    assert result["domain_deltas"] is None
    assert result["pairs"][1]["reward_delta"] is None
    assert result["after"]["domains"]["healthbench"]["reward_mean"] is None
    assert result["pairs"][1]["after"]["infrastructure_error"] == "PreparationInfrastructure"
    output = tmp_path / "comparison-private.json"
    monkeypatch.setattr(
        "sys.argv",
        ["comparison", "--before", str(before), "--after", str(after), "--output", str(output)],
    )
    with pytest.raises(SystemExit) as error:
        report.main()
    assert error.value.code == 2
    assert output.is_file()
    assert "private detail" not in output.read_text()
    with pytest.raises(FileExistsError):
        report.main()


def test_assessments_require_exact_original_edge_and_missing_phase_stays_unknown(tmp_path):
    before, after = pair_fixture(tmp_path)
    (before / "collection/episodes/rollout-progress.json").unlink()
    row = json.loads((before / "collection/episodes/episode-000000-private.json").read_text())[
        "artifact"
    ]["record"]
    events = []
    for turn, step in enumerate(row["steps"], 1):
        events.append(
            {
                "run_id": "collection",
                "payload": {
                    "trajectory_id": row["trajectory_id"],
                    "turn": turn,
                    "action_token_ids": step["action_token_ids"],
                    "observation_text": step["observation_text"],
                    "assessment": {
                        "admitted": True,
                        "executed": True,
                        "accepted_submission": False,
                        "environment_terminal": False,
                        "execution_status": "success",
                    },
                },
            }
        )
    wrong = copy.deepcopy(events[0])
    wrong["payload"]["action_token_ids"] = [-999]
    wrong["payload"]["assessment"]["admitted"] = False
    events.append(wrong)
    (before / "collection/events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    result = report.compare(before, after)
    assert result["before"]["phase_totals"] is None
    assert result["pairs"][0]["before"]["actions"]["admitted_count"] == len(row["steps"])
    assert result["pairs"][1]["before"]["actions"]["admitted_count"] is None


def test_real_skill_receipt_and_body_visibility_are_separate(tmp_path):
    before, after = pair_fixture(tmp_path)
    for root in (before, after):
        path = root / "collection/episodes/episode-000000-private.json"
        outcome = json.loads(path.read_text())
        artifact = outcome["artifact"]
        record = artifact["record"]
        meta = record["initial_context"]["meta"]
        meta["skill_exposure"] = "catalog-then-read@1"
        step = record["steps"][0]
        step.update(
            action_text=json.dumps(
                {
                    "kind": "skill",
                    "name": "read",
                    "resource_id": "skills",
                    "skill_id": "synthetic-skill",
                    "arguments": {},
                }
            ),
            invoked_skill_ids=["synthetic-skill"],
            observation_status="success",
            observation_text=json.dumps(
                {
                    "status": "skill-read",
                    "skill_id": "synthetic-skill",
                    "version": "v1",
                    "library_version": meta["library_version"],
                    "content": "Public synthetic method, not a benchmark answer.",
                }
            ),
        )
        artifact["skill_input_evidence"] = [
            {
                "phase": "action",
                "step_index": 2,
                "visible_skill_body_refs": [{"skill_id": "synthetic-skill", "read_step_index": 1}],
            }
        ]
        write(path, outcome)
    result = report.compare(before, after)
    action = result["pairs"][0]["before"]["actions"]
    assert action["skill_body_returned_count"] == 1
    assert action["body_visible_input_count"] == 1
    assert "Public synthetic method" not in json.dumps(result)


def test_changed_actual_sampling_coordinates_rejected(tmp_path):
    before, after = pair_fixture(tmp_path)
    path = after / "collection/episodes/episode-000000-private.json"
    row = json.loads(path.read_text())
    row["artifact"]["manifest"]["sampling_coordinate"]["sequence_position"] += 1
    write(path, row)
    with pytest.raises(ValueError):
        report.compare(before, after)
