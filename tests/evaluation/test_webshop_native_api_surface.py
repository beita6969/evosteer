"""The public surface follows the deployed native API, not HTML widget presence."""

import pytest
from skillev_private.benchmarks import official_environment_worker
from skillev_private.benchmarks.official_environment_worker import available_webshop_actions

from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)
from tests.evaluation.test_integrity_native_tool_calls import call


@pytest.mark.parametrize("has_search_bar", [False, True])
def test_search_is_available_on_all_native_pages_without_selecting_a_query(has_search_bar):
    native = {"has_search_bar": has_search_bar, "clickables": ["item-1", "buy now"]}
    actions = tuple(available_webshop_actions(native))
    surface = PublicSurface("webshop", 4, actions)
    request = call("search", query="A model-chosen Query")
    assert normalize_decision(ExplicitDecision(request, 4), surface).action == (
        "search[A model-chosen Query]"
    )
    assert all(f"click[{name}]" in actions for name in native["clickables"])
    assert native["has_search_bar"] is has_search_bar


@pytest.mark.parametrize("target", ["Buy Now", "BUY NOW", "buy now", "click[Buy Now]"])
def test_native_click_argument_lowercasing_does_not_rewrite_the_owner_wire_text(target):
    surface = PublicSurface("webshop", 3, ("search", "click[buy now]"))
    action = normalize_decision(ExplicitDecision(call("click", target=target), 3), surface).action
    expected = target if target.startswith("click[") else f"click[{target}]"
    assert action == expected
    # Exactly the native step's argument transform, with no semantic target search.
    assert action[6:-1].lower() == "buy now"


@pytest.mark.parametrize("target", ["Buy", "Buy Now!", " Buy Now ", '"Buy Now"', "Buy Later"])
def test_native_equivalence_never_changes_spacing_punctuation_or_selects_another_control(target):
    surface = PublicSurface("webshop", 3, ("search", "click[buy now]"))
    assert (
        normalize_decision(ExplicitDecision(call("click", target=target), 3), surface).action
        is None
    )


def test_native_click_casing_does_not_expand_other_benchmark_commands():
    surface = PublicSurface("alfworld", 3, ("look",))
    assert (
        normalize_decision(ExplicitDecision(call("act", command="LOOK"), 3), surface).action is None
    )


def test_search_form_does_not_advertise_the_native_click_search_noop():
    actions = available_webshop_actions(
        {"has_search_bar": True, "clickables": ["search", "item-1", "buy now"]}
    )
    assert "search" in actions
    assert "click[search]" not in actions
    assert "click[item-1]" in actions
    assert "click[buy now]" in actions


@pytest.mark.parametrize("mode", ["text", "text_rich"])
def test_explicit_native_renderer_reaches_both_reset_and_action_observations(monkeypatch, mode):
    class Environment:
        observation_mode = "text"

        def reset(self, session):
            return self.observation_mode + ": public reset", None

        def step(self, action):
            return self.observation_mode + ": public selected option", 0.0, False, None

        def get_instruction_text(self):
            return "Instruction: Synthetic shopping request."

        def get_available_actions(self):
            return {"has_search_bar": False, "clickables": ["option", "buy now"]}

    monkeypatch.setattr(
        official_environment_worker, "create_webshop_env", lambda *args: Environment()
    )
    worker = official_environment_worker.WebShopWorker(
        "synthetic-source", {}, {"goal_index": 0, "observation_mode": mode}
    )
    assert worker.reset()["observation_text"] == mode + ": public reset"
    assert worker.step("click[option]")["observation_text"] == mode + ": public selected option"
