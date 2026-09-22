"""Native public observations do not imply oracle actions or framework planning."""

import asyncio
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.official_environment_worker import ScienceWorldWorker
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldStepResult
from skillev_private.evaluation.ood_scienceworld import OODScienceWorldEnvironment

from skillev.evaluation.journaled_environment import JournaledEnvironment
from skillev.evaluation.scienceworld_observation import containment_layout, public_containment_view
from skillev.evaluation.sealed_candidates import CandidateJournal, EventOrigin


def test_worker_forwards_only_existing_public_info_without_an_extra_action():
    calls = []
    info = {
        "score": -100,
        "moves": 8,
        "look": "Public room",
        "inv": "Public inventory",
        "taskDesc": "Public goal",
        "valid": ["oracle action"],
        "gold": ["private plan"],
    }
    worker = object.__new__(ScienceWorldWorker)
    worker.env = SimpleNamespace(
        step=lambda action: calls.append(action) or ("Public feedback", -100, True, info)
    )
    result = worker.step("focus on wrong object")
    assert calls == ["focus on wrong object"]
    assert result["score"] == -100
    assert result["native_moves"] == 8
    assert result["public_state"]["current_look"] == "Public room"
    assert "oracle" not in str(result["public_state"])
    assert "private" not in str(result["public_state"])


@pytest.mark.parametrize("profile", ["public-state@1", "public-state@2"])
def test_enriched_observation_and_transition_keep_raw_score_private(tmp_path, profile):
    calls = []
    state = {
        "current_look": "A public room\n  a pot (containing a plant)",
        "inventory": "A public item",
        "task_description": "A goal",
    }
    native = SimpleNamespace(
        step=lambda action: calls.append(action)
        or OfficialScienceWorldStepResult(
            "Native negative terminal feedback",
            -100,
            True,
            public_state=state,
            native_moves=5,
        )
    )
    env = OODScienceWorldEnvironment(native, "synthetic", 0, 0, 20, observation_profile=profile)
    env._raw_score = 45
    journal = CandidateJournal(tmp_path / "private.sqlite")
    wrapped = JournaledEnvironment(env, journal, ("run", "arm", "case"), owner="owner")
    wrapped.revision = 1
    try:
        result = asyncio.run(wrapped.execute("owner", "decision", "focus on object", 1))
        assert calls == ["focus on object"]
        assert "A public room" in result.observation
        assert "A public item" in result.observation
        assert state["current_look"] in result.observation
        assert ("Public containment layout" in result.observation) == (profile == "public-state@2")
        assert "-100" not in result.observation
        assert result.available_actions == ("act",)
        receipt = journal.traces(
            ("run", "arm", "case"), "environment-transition", origin=EventOrigin.ENVIRONMENT
        )[0]
        assert receipt["command"] == calls[0]
        assert receipt["category"] == "native-negative-terminal"
        assert receipt["native_score"] == -100
        assert receipt["simulator_moves"] == 5
        outcome = asyncio.run(env.outcome())
        assert outcome.native_steps == 1
        assert outcome.simulator_moves == 5
        assert not outcome.success
    finally:
        journal.close()


@pytest.mark.parametrize("profile", ["public-state@1", "public-state@2"])
def test_selected_enriched_profile_does_not_fall_back_to_text_only(profile):
    env = OODScienceWorldEnvironment(None, "synthetic", 0, 0, 20, observation_profile=profile)
    with pytest.raises(RuntimeError):
        env.public_observation("Public feedback")
    env.observation_profile = "text-only@1"
    assert env.public_observation("Public feedback") == "Public feedback"


def test_containment_view_is_literal_text_not_a_resolved_object_catalogue():
    line = "  a tray (containing a bowl (containing water), a spoon)"
    layout = containment_layout(line)
    assert layout.count("\n") == 2
    assert layout.split() == line.split()
    assert containment_layout("a cupboard. Its door is closed.") == ""
    assert containment_layout("an unfamiliar unstructured observation") == ""
    assert "containing nothing" in containment_layout("a pot (containing nothing)")
    assert public_containment_view("a closed chest", "empty") == ""
    view = public_containment_view(line, "a jar (containing a liquid)")
    assert "water" in view
    assert "a liquid" in view
    assert "focus on" not in view
