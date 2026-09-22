from skillev.evaluation.direct_baseline.prompts import (
    PublicInteractiveTurn,
    render_interactive_messages,
)


def test_webshop_source_prompt_has_no_react_or_demonstration() -> None:
    messages = render_interactive_messages(
        "webshop-skillflow-action-only-v1@1",
        task="buy a blue shirt",
        current_observation="search page",
        available_actions=("search", "click[Blue shirt]"),
    )
    assert messages[0]["role"] == "user"
    prompt = messages[0]["content"]
    assert "search[<your query>]" in prompt
    assert "click[Blue shirt]" in prompt
    assert "Thought:" not in prompt
    assert "Demonstration" not in prompt


def test_source_history_contains_actions_and_public_observations_not_reasoning() -> None:
    history = (
        PublicInteractiveTurn(
            "private reasoning must not be rendered",
            "click[item]",
            "before",
            "result",
            ("click[Buy Now]",),
        ),
    )
    messages = render_interactive_messages(
        "webshop-skillflow-action-only-v1@1",
        task="buy item",
        current_observation="product page",
        history=history,
        available_actions=("click[Buy Now]",),
    )
    prompt = messages[0]["content"]
    assert "click[item]" in prompt
    assert "result" in prompt
    assert "before" in prompt
    assert "private reasoning" not in prompt


def test_alfworld_source_prompt_uses_exact_action_only_instruction() -> None:
    messages = render_interactive_messages(
        "alfworld-skillflow-action-only-v1@1",
        task="put mug on shelf",
        current_observation="You are in the kitchen.",
        available_actions=("go to cabinet 1", "help"),
    )
    prompt = messages[0]["content"]
    assert "'go to cabinet 1'," in prompt
    assert "'help'" not in prompt
    assert "Output ONLY the action you choose. No explanation, no reasoning" in prompt
    assert "Your task is to: put mug on shelf" in prompt
