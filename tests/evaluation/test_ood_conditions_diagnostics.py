"""Independent conditions and future holdout exclusions never select high scores."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.prepare_ood_condition import prepare_condition
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.panel_isolation import exclude_seen_public_problems
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.evaluation.trajectory_diagnostics import trajectory_diagnostics


def test_reserve_is_a_new_condition_without_changing_total_or_original_configuration():
    base = {
        "arms": [InferenceArm("old").to_value()],
        "evaluation_sample_counts": {"livecodebench": 64},
        "budgets": {"livecodebench": {"total_output_tokens": 12000}},
        "decoding": {"livecodebench": {"max_new_tokens": 12000, "seed": 0}},
    }
    unchanged = deepcopy(base)
    new = prepare_condition(base, "code-finalization-reserve", "new")
    assert base == unchanged
    assert new["budget_overrides"]["livecodebench"]["finalization_reserve_tokens"] == 1000
    assert new["budgets"] == base["budgets"]
    assert new["decoding"] == base["decoding"]
    assert new["evaluation_sample_counts"] == base["evaluation_sample_counts"]


def test_independent_generation_does_not_point_at_previous_judge_attempts():
    base = {
        "arms": [InferenceArm("old").to_value()],
        "scorers": {"omni-math": {"cache_directory": "/private/old-judge"}},
    }
    new = prepare_condition(base, "training-shared-semantics-step0", "new/condition")
    assert base["scorers"]["omni-math"]["cache_directory"] == "/private/old-judge"
    path = new["scorers"]["omni-math"]["cache_directory"]
    assert path != base["scorers"]["omni-math"]["cache_directory"]
    assert path.endswith("new%2Fcondition")
    base["scorers"]["omni-math"]["judgements_path"] = "/private/old-judgements.json"
    with pytest.raises(ValueError):
        prepare_condition(base, "training-shared-semantics-step0", "new")


@pytest.mark.parametrize("condition", ["code-finalization-reserve", "scienceworld-public-state"])
def test_named_intervention_requires_its_actual_task_population(condition):
    base = {
        "arms": [InferenceArm("old").to_value()],
        "evaluation_sample_counts": {"nq-open": 64},
    }
    with pytest.raises(ValueError):
        prepare_condition(base, condition, "new")


def test_new_ids_and_different_context_do_not_make_an_old_question_a_holdout():
    old = PublicTaskView.from_record(
        "old", "musique", {"question": "Which Tree?", "context": "Old context"}
    )
    duplicate = PublicTaskView.from_record(
        "new-id", "musique", {"question": " WHICH  tree? ", "context": "Other context"}
    )
    new = PublicTaskView.from_record(
        "other", "musique", {"question": "Which flower?", "context": "Context"}
    )
    assert exclude_seen_public_problems([duplicate, new], [old]) == (new,)


def test_environment_holdout_uses_source_variation_not_placeholder_or_new_task_id():
    def task(task_id):
        return PublicTaskView.from_record(task_id, "scienceworld", {"task": "Provided on reset"})

    old, duplicate, new = task("old"), task("renamed"), task("new")
    identities = {
        "old": ("task-name", "variation-1", "simplification-none"),
        "renamed": ("task-name", "variation-1", "simplification-none"),
        "new": ("task-name", "variation-2", "simplification-none"),
    }
    with pytest.raises(ValueError):
        exclude_seen_public_problems([duplicate, new], [old])
    assert exclude_seen_public_problems([duplicate, new], [old], source_identities=identities) == (
        new,
    )


def test_diagnostics_report_missing_history_without_reconstructing_success():
    candidate = SimpleNamespace(
        text="An explanatory response.",
        submission={"projection_id": "owner-unlabelled-short-answer@5"},
        terminal_status=SimpleNamespace(value="submitted"),
    )
    reader = SimpleNamespace(
        get=lambda *scope: candidate,
        model_outputs=lambda scope: (),
        traces=lambda *args, **kwargs: (),
    )
    report = trajectory_diagnostics(reader, (("r", "a", "t"),), benchmark="nq-open")
    assert report["counts"]["whole_text_unlabelled"] == 1
    assert report["counts"]["missing_source_identity"] == 1
    assert report["counts"]["missing_input_receipt"] == 1
    assert report["counts"]["consultant_calls"] == 0
