"""Operational recovery preserves successful execution, not successful answers."""

import asyncio
import json
from pathlib import Path

import pytest
from skillev_private.evaluation.ood_recovery import (
    aggregate_recovery,
    plan_recovery,
    recovery_reason,
)

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.sealed_candidates import CandidateJournal, EventOrigin, FinalCandidate
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_integrity_broker_boundary import runtime


def recipe():
    return json.loads(Path("configs/evaluation/ood_step0_8000.json").read_text())


@pytest.mark.parametrize(
    ("tokens", "wall", "reason"),
    [
        (8000, None, None),
        (8001, None, "output-token-limit"),
        (20, 601, "episode-wall-limit"),
        (20, 600, None),
    ],
)
def test_metadata_only_recovery_includes_boundary_and_unknown_time(tmp_path, tokens, wall, reason):
    journal = CandidateJournal(tmp_path / "candidates.sqlite")
    scope = ("run", "arm", "task")
    journal.seal(FinalCandidate(*scope, "policy", "final", "WRONG", "parser", 5, tokens))
    if wall is not None:
        journal.record(
            scope, "episode-timing", {"wall_seconds": wall}, origin=EventOrigin.MODEL_TRANSPORT
        )
    journal.record(scope, "native-verdict", {"correct": False}, origin=EventOrigin.SCORER)
    try:
        assert recovery_reason(journal, scope, token_limit=8000, wall_limit=600) == reason
        assert (
            recovery_reason(journal, ("run", "arm", "unstarted"), token_limit=8000, wall_limit=600)
            == "unstarted"
        )
        journal.start_attempt(("run", "arm", "interrupted"), policy_id="policy")
        assert (
            recovery_reason(
                journal, ("run", "arm", "interrupted"), token_limit=8000, wall_limit=600
            )
            == "unfinished-attempt"
        )
    finally:
        journal.close()


def test_plan_and_native_aggregate_keep_wrong_answer_and_fill_only_missing(tmp_path):
    entries = [
        PublicTaskView.from_record(f"task-{i}", "nq-open", {"question": "Color?"}) for i in range(2)
    ]
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old = runtime(old_dir, entries[0], ["Final answer: red"])
    asyncio.run(old.generate(entries[0], InferenceArm("old"), "old-run"))
    old.journal.close()
    data = tmp_path / "source.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": e.task_id,
                    "public": {"question": "Color?"},
                    "target": {"accepted_answers": ["blue"]},
                }
            )
            for e in entries
        )
    )
    config = tmp_path / "original.json"
    config.write_text(
        json.dumps(
            {
                "arms": [InferenceArm("old").to_value()],
                "ood_sources": {"nq-open": str(data)},
                "evaluation_sample_counts": {"nq-open": 2},
                "ood_provenance": {"split": "synthetic"},
                "public_source": str(Path("src").resolve()),
                "scorers": {"omni-math": {}, "livecodebench": {}, "apps-introductory": {}},
            }
        )
    )
    output = tmp_path / "recovery"
    result = plan_recovery(
        original_config=config,
        old_journal=old_dir / "candidate.sqlite",
        old_run_id="old-run",
        new_run_id="new-run",
        new_arm_id="new",
        recipe=Path("configs/evaluation/ood_step0_8000.json"),
        output=output,
    )
    assert result["replacement_count"] == 1
    subset = [
        json.loads(line) for line in (output / "nq-open-private.jsonl").read_text().splitlines()
    ]
    assert [row["task_id"] for row in subset] == [entries[1].task_id]
    new_dir = output / "run"
    new_dir.mkdir()
    new = runtime(new_dir, entries[1], ["Final answer: blue"])
    new.journal.close()
    new.journal = CandidateJournal(new_dir / "candidates-private.sqlite")
    asyncio.run(new.generate(entries[1], InferenceArm("new"), "new-run"))
    new.journal.close()
    aggregate = asyncio.run(aggregate_recovery(output / "plan-private.json"))
    assert aggregate["domains"]["nq-open"]["count"] == 2
    assert aggregate["domains"]["nq-open"]["value"] == 0.5
    assert aggregate["domains"]["nq-open"]["source_run_counts"] == {"old-run": 1, "new-run": 1}
    again = plan_recovery(
        original_config=config,
        old_journal=old_dir / "candidate.sqlite",
        old_run_id="old-run",
        new_run_id="third-run",
        new_arm_id="third",
        recipe=Path("configs/evaluation/ood_step0_8000.json"),
        output=tmp_path / "again",
        previous_plan=output / "plan-private.json",
    )
    assert again["replacement_count"] == 0


def test_8000_is_cumulative_across_actual_owner_format_repairs(tmp_path):
    entry = PublicTaskView.from_record("case", "nq-open", {"question": "Synthetic?"})
    malformed = "Message to solver:\n" + "x" * 6900
    instance = runtime(tmp_path, entry, [malformed, "Final answer: blue"])
    instance.config.update(recipe())
    try:
        final = asyncio.run(instance.generate(entry, InferenceArm("ood"), "run"))
        profiles = instance.generators[0].profiles
        assert profiles[0].max_new_tokens == 8000
        assert profiles[1].max_new_tokens == 8000 - len(malformed)
        assert final.completion_tokens == len(malformed) + len("Final answer: blue") <= 8000
        assert final.intervention_counts["peer_model_calls"] == 0
    finally:
        instance.journal.close()
