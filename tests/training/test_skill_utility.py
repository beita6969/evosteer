import json

import pytest
from skillev_private.experiments import skill_utility as utility

from tests.training.test_development_collection import write
from tests.training.test_development_comparison import pair_fixture
from tests.v3_helpers import CharacterTokenizer, make_artifact


def utility_pair(tmp_path):
    roots = pair_fixture(tmp_path)
    for arm, root in zip(("skills-off", "initial-library"), roots, strict=True):
        selection = json.loads((root / "selection-private.json").read_text())
        selection.update(
            arm=arm,
            skill_utility_policy={
                "minimum_reviewed_sources": 2,
                "minimum_success_rate_gain": 0.1,
                "minimum_output_token_saving_fraction": 0.1,
            },
        )
        write(root / "selection-private.json", selection)
        write(
            root / "controls-private.json",
            {
                "selection": selection,
                "formal": {"reasoning_tokens": 100},
                **dict.fromkeys(
                    ("backbone", "policy", "service", "scorers", "deployments", "workflow"), "same"
                ),
            },
        )
    return roots


def test_high_reward_without_application_evidence_does_not_qualify_library(tmp_path, monkeypatch):
    off, on = utility_pair(tmp_path)
    monkeypatch.setattr(utility, "_artifact", lambda *_: make_artifact("synthetic"))
    result = utility.compare_skill_utility(off, on, [], tokenizer=CharacterTokenizer())
    assert result["all_source_success_rate_gain"] == 1
    assert result["reviewed_successful_sources"] == 0
    assert result["observed_library_utility_qualified"] is False
    assert len(result["sources"]) == result["planned_source_count"] == 2
    assert result["training_evidence_writes"] == 0


@pytest.mark.parametrize(
    "changed", ["formal", "scorers", "policy", "service", "development_practice"]
)
def test_skill_contrast_cannot_hide_another_changed_axis(tmp_path, changed):
    off, on = utility_pair(tmp_path)
    value = json.loads((on / "controls-private.json").read_text())
    value[changed] = "changed"
    write(on / "controls-private.json", value)
    with pytest.raises(ValueError):
        utility.compare_skill_utility(off, on, [])


def test_missing_native_outcome_is_not_a_negative_example(tmp_path):
    off, on = utility_pair(tmp_path)
    (on / "collection/episodes/episode-000001-private.json").unlink()
    with pytest.raises(ValueError):
        utility.compare_skill_utility(off, on, [])


def test_a_review_cannot_invent_a_read_or_visible_application():
    artifact = make_artifact("no-read")
    with pytest.raises(ValueError):
        utility.application_evidence(
            artifact,
            [
                {
                    "skill_id": "made-up",
                    "read_step": 1,
                    "decision": "applied-procedure",
                    "application_steps": [2],
                    "reviewer": "fixture",
                    "public_rationale": "synthetic review",
                    "evidence_reference": "not-real",
                }
            ],
        )
