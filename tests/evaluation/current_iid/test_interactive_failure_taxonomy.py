from skillev_private.benchmarks.alfworld_taxonomy import ALFWorldTaskType

from scripts.analyze_qwen35_alfworld_traces import classify_alfworld_failure
from scripts.analyze_qwen35_webshop_traces import classify_webshop_episode


def test_webshop_failure_taxonomy_separates_purchase_budget_and_surface_signals() -> None:
    row = {
        "success": False,
        "reward": 0.25,
        "steps": 3,
        "terminal_reached": True,
        "terminated_by_horizon": False,
        "budget_exhausted": False,
        "termination_reason": "official-terminal",
        "trace": [
            {
                "parsed_action": "search[blue shoe]",
                "observation_before": "home",
                "observation_after": "results",
                "action_listed_before": True,
            },
            {
                "parsed_action": "click[item-1]",
                "observation_before": "results",
                "observation_after": "Price: $25.00\nBuy Now\nDescription",
                "action_listed_before": True,
            },
            {
                "parsed_action": "click[buy now]",
                "observation_before": "Price: $25.00\nBuy Now",
                "observation_after": "done",
                "action_listed_before": True,
            },
        ],
    }
    counts = classify_webshop_episode(row, task="buy a blue shoe under $20")
    assert counts["early_terminal_failure"] == 1
    assert counts["premature_purchase"] == 1
    assert counts["price_exceeded"] == 1
    assert counts["probable_product_selection_error"] == 1
    assert counts["budget_exhausted"] == 0


def test_alfworld_failure_taxonomy_reports_missing_transformation_subgoal() -> None:
    row = {
        "success": False,
        "trace": [
            {
                "parsed_action": "take mug 1 from table 1",
                "observation_before": "mug visible",
                "observation_after": "mug taken",
                "available_actions_before": ["take mug 1 from table 1"],
                "action_listed_before": True,
                "official_action_valid": True,
            }
        ],
    }
    counts = classify_alfworld_failure(row, ALFWorldTaskType.CLEAN_THEN_PLACE)
    assert counts["forgot_clean"] == 1
    assert counts["failure_subgoal/clean"] == 1


def test_alfworld_failure_taxonomy_detects_one_of_two_objects_completed() -> None:
    row = {
        "success": False,
        "trace": [
            {
                "parsed_action": "take apple 1 from table 1",
                "observation_before": "apple visible",
                "observation_after": "apple taken",
                "available_actions_before": ["take apple 1 from table 1"],
                "action_listed_before": True,
                "official_action_valid": True,
            },
            {
                "parsed_action": "move apple 1 to bowl 1",
                "observation_before": "holding apple",
                "observation_after": "apple moved",
                "available_actions_before": ["move apple 1 to bowl 1"],
                "action_listed_before": True,
                "official_action_valid": True,
            },
        ],
    }
    counts = classify_alfworld_failure(row, ALFWorldTaskType.PICK_TWO_OBJECTS)
    assert counts["two_object_only_one_completed"] == 1
    assert counts["failure_subgoal/complete_second_object"] == 1
