from skillev.evaluation.direct_baseline.prompts import (
    PublicInteractiveTurn,
    WebShopPublicCatalogCandidate,
    render_interactive_messages,
)


def test_visible_react_history_is_public_and_hidden_reasoning_is_absent() -> None:
    messages = render_interactive_messages(
        "webshop-native-react-v3@1",
        task="buy an item",
        current_observation="latest page",
        history=(
            PublicInteractiveTurn(
                "Thought: use search\nAction: search[item]",
                "search[item]",
                "initial page",
                "search results page",
                ("click[item]",),
            ),
        ),
    )
    text = messages[-1]["content"]
    assert "Thought: use search" in text
    assert "Environment returned state: search results page" in text
    assert "click[item]" in text
    assert "latest page" in text
    assert "server hidden reasoning" not in text


def test_history_truncates_whole_turns_and_preserves_latest_observation() -> None:
    messages = render_interactive_messages(
        "alfworld-native-react-v3@1",
        task="task",
        current_observation="latest",
        history=(
            PublicInteractiveTurn("Action: first", "first", "x" * 100),
            PublicInteractiveTurn("Action: second", "second", "recent"),
        ),
        history_maximum_characters=35,
        remaining_steps=7,
    )
    text = messages[-1]["content"]
    assert "Action: first" not in text
    assert "Action: second" in text
    assert text.count("latest") == 1
    assert "Remaining action budget: 7" in text


def test_history_records_each_action_and_returned_public_state_in_order() -> None:
    messages = render_interactive_messages(
        "alfworld-native-react-v4@1",
        task="put the mug on the shelf",
        current_observation="The mug is in your inventory and you face shelf 1.",
        history=(
            PublicInteractiveTurn(
                "Thought: take the mug\nAction: take mug 1 from table 1",
                "take mug 1 from table 1",
                "You face table 1.",
                "You take mug 1. It is now in your inventory.",
                ("go to shelf 1",),
            ),
        ),
        available_actions=("move mug 1 to shelf 1",),
        remaining_steps=4,
    )
    text = messages[-1]["content"]
    before = text.index("State before action: You face table 1.")
    action = text.index("Action: take mug 1 from table 1")
    returned = text.index(
        "Environment returned state: You take mug 1. It is now in your inventory."
    )
    current = text.index("Observation: The mug is in your inventory and you face shelf 1.")
    assert before < action < returned < current
    assert "go to shelf 1" in text


def test_structured_prompt_separates_training_goal_from_authoritative_current_task() -> None:
    messages = render_interactive_messages(
        "alfworld-native-react-memory-v7@1",
        task="put current-object on destination-a",
        task_type="pick_and_place",
        current_observation="You are in the middle of a room.",
        demonstrations=(
            "Demonstration (training example):\nTask: put demonstration-object on destination-b",
        ),
        available_actions=("go to countertop 1",),
        remaining_steps=20,
    )
    text = messages[-1]["content"]

    assert "BEGIN TRAINING DEMONSTRATION" in text
    assert "END TRAINING DEMONSTRATION" in text
    assert "AUTHORITATIVE CURRENT TASK (not a demonstration)" in text
    assert text.rindex("put current-object on destination-a") > text.index(
        "put demonstration-object on destination-b"
    )
    assert "Do not copy a demonstration Goal into current Memory" in text
    assert "FINAL AUTHORITATIVE CURRENT-TASK ANCHOR" in text
    assert "if the environment says NOT TERMINAL" in text


def test_webshop_v18_is_surface_grounded_full_react_memory() -> None:
    messages = render_interactive_messages(
        "webshop-native-react-memory-v18@1",
        task="buy synthetic product alpha below 40 dollars",
        current_observation="Search page",
        available_actions=("search", "click[search]"),
        remaining_steps=20,
    )

    assert "one concise Thought" in messages[0]["content"]
    assert "then output one Thought and one Action" in messages[-1]["content"]
    assert "CURRENT ACTION SURFACE" in messages[-1]["content"]


def test_step_zero_interactive_source_profiles_leave_strategy_to_retrieved_skill() -> None:
    webshop = render_interactive_messages(
        "webshop-step0-skill-owned-react-memory@1",
        task="buy a public item",
        current_observation="Search page",
        structured_memory="public factual ledger",
        available_actions=("search", "click[search]"),
        remaining_steps=20,
    )
    alfworld = render_interactive_messages(
        "alfworld-step0-skill-owned-react-memory@1",
        task="put a public object on a destination",
        task_type="pick_and_place",
        current_observation="Room state",
        structured_memory="public factual ledger",
        demonstrations=("external source strategy that must not enter Step-0",),
        available_actions=("go to countertop 1",),
        remaining_steps=40,
    )

    for messages in (webshop, alfworld):
        combined = "\n".join(message["content"] for message in messages)
        assert "retrieved skill" in combined
        assert "response-format policy" in combined
        assert "Task decomposition:" not in combined
        assert "Current decision policy:" not in combined
        assert "advance one remaining subgoal" not in combined
        assert "Accumulated structured memory before this action" not in combined
        assert "external source strategy" not in combined
    assert "CURRENT ACTION SURFACE" in webshop[-1]["content"]
    assert "Admissible actions now" in alfworld[-1]["content"]


def test_step_zero_webshop_catalog_is_not_duplicated_into_source_messages() -> None:
    messages = render_interactive_messages(
        "webshop-step0-skill-owned-react-memory@1",
        task="buy a public item",
        current_observation="Search page",
        structured_memory="public factual ledger",
        available_actions=("search", "click[search]"),
        remaining_steps=20,
        webshop_catalog_candidates=(
            WebShopPublicCatalogCandidate(
                product_id="PUBLIC0001",
                title="Public candidate",
                price=9.0,
                suggested_search_query="Public candidate",
            ),
        ),
    )

    text = messages[-1]["content"]
    assert "PUBLIC CATALOG LOOKUP TOOL OUTPUT" not in text
    assert "Public candidate" not in text
    assert "Begin with Candidate 1" not in text
    assert "move to a later candidate only" not in text
