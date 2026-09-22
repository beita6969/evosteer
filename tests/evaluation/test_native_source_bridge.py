import json
from dataclasses import asdict, replace

import pytest
from skillev_private.benchmarks.protocol_v13_training import (
    Protocol13TrainingEpisode,
    Protocol13TrainingOutput,
    Protocol13TrainingRecord,
)
from skillev_private.experiments.native_source_bridge import native_training_source_panel

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from tests.evaluation.test_training_public_bridge import task


def record(domain, verifier, target):
    return Protocol13TrainingRecord(
        Protocol13TrainingEpisode(
            Protocol13Benchmark(domain), "synthetic-population", "synthetic", "source-1", 0, 0, 1, 0
        ),
        task(domain),
        Protocol13TrainingOutput(verifier, target),
    )


@pytest.mark.parametrize(
    ("domain", "verifier", "target"),
    [
        ("hotpotqa", "hotpotqa-official-em-f1", {"accepted_answers": ["PRIVATE_SENTINEL"]}),
        ("triviaqa", "triviaqa-official-alias-em-f1", {"accepted_answers": ["PRIVATE_SENTINEL"]}),
        ("healthbench", "simple-evals-rubric", {"rubrics": [{"text": "PRIVATE_SENTINEL"}]}),
        ("mbpp-plus", "evalplus-base-plus", {"canonical_solution": "PRIVATE_SENTINEL"}),
    ],
)
def test_native_panel_keeps_targets_private_and_counts_repeated_questions(domain, verifier, target):
    first = record(domain, verifier, target)
    second = replace(
        first,
        episode=replace(first.episode, episode_id="second-slot"),
        input=replace(first.input, task_id="second-slot"),
    )
    panel = native_training_source_panel((first, second), interactive={})
    panel.panel.validate(canary=False)
    public = json.dumps(asdict(panel.panel))
    assert "PRIVATE_SENTINEL" not in public
    assert "PRIVATE_SENTINEL" in json.dumps(panel.targets)
    assert panel.provenance["unique_source_questions"] == 1
    assert tuple(e.task_id for e in panel.panel.entries) == ("synthetic", "second-slot")
    assert first.input.query in panel.panel.entries[0].render()
    first.output.target.clear()
    assert "PRIVATE_SENTINEL" in json.dumps(panel.targets)


def test_aime_native_carrier_preserves_original_integer_without_answer_selection():
    first = record("aime-2026", "integer-exact", {"accepted_answers": ["007", "7"]})
    panel = native_training_source_panel((first,), interactive={})
    assert panel.targets["synthetic"]["answer"] == "7"
    first.output.target["accepted_answers"] = ["7", "8"]
    with pytest.raises(ValueError):
        native_training_source_panel((first,), interactive={})


def test_retired_humaneval_cannot_enter_a_new_current_native_panel():
    first = record(
        "humaneval", "humaneval-native", {"test": "PRIVATE_SENTINEL", "entry_point": "f"}
    )
    with pytest.raises(ValueError):
        native_training_source_panel((first,), interactive={})


def test_native_alfworld_binding_must_use_same_public_task_and_original_game():
    first = record("alfworld", "alfworld-success", {"target_won": True})
    spec = {
        "case": {
            "task_id": "synthetic",
            "benchmark": "alfworld",
            "task": first.input.query,
            "payload": {"game_id": "source-1"},
        },
        "manifest": {"deployments": {"private-path": "not actor input"}},
    }
    result = native_training_source_panel((first,), interactive={"synthetic": spec})
    assert result.targets == {}
    assert "private-path" not in json.dumps(asdict(result.panel))
    spec["case"]["payload"]["game_id"] = "different-game"
    assert result.interactive["synthetic"]["case"]["payload"]["game_id"] == "source-1"
    with pytest.raises(ValueError):
        native_training_source_panel((first,), interactive={"synthetic": spec})
    with pytest.raises(ValueError):
        native_training_source_panel((first,), interactive={})
