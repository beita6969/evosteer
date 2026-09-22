from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from skillev_private.benchmarks import official_environment_worker
from skillev_private.benchmarks.alfworld_taxonomy import ALFWorldTaskType
from skillev_private.direct_reference.protocol13_runner import (
    load_protocol13_environment_manifest_identity,
)

from scripts.analyze_qwen35_alfworld_traces import classify_alfworld_failure
from scripts.analyze_qwen35_webshop_traces import classify_webshop_episode
from skillev.evaluation.current_iid.protocol13.config import load_execution_contracts_v3
from skillev.evaluation.direct_baseline.prompts import (
    PublicInteractiveTurn,
    render_interactive_messages,
)
from skillev.evaluation.interactive_prompt_assets import (
    InteractivePromptAsset,
    InteractivePromptExample,
    InteractiveReplayStep,
    load_interactive_prompt_asset,
)
from skillev.experiments.protocol_v13 import load_protocol_v13

ROOT = Path(__file__).parents[3]


def test_webshop_v6_separates_unknowns_from_violations_and_enters_commit_phase() -> None:
    messages = render_interactive_messages(
        "webshop-native-react-memory-v6@1",
        task="buy a blue mug below 20 dollars",
        current_observation="Product page with price $18",
        available_actions=("click[blue]", "click[buy now]"),
        remaining_steps=2,
    )
    text = messages[-1]["content"]
    assert "Candidate ledger" in text
    assert "COMMIT PHASE" in text
    assert "Unknown descriptive attributes are not known violations" in text
    assert "actions reserved for options+purchase" in text


def test_webshop_v7_expires_historical_surfaces_and_makes_current_surface_authoritative() -> None:
    messages = render_interactive_messages(
        "webshop-native-react-memory-v7@1",
        task="buy a blue mug below 20 dollars",
        current_observation="Description subpage for the current candidate.",
        available_actions=("click[Back to Search]", "click[< Prev]", "click[Next >]"),
        history=(
            PublicInteractiveTurn(
                "Action: click[description]",
                "click[description]",
                "Product page with price $18",
                "Description subpage for the current candidate.",
                ("click[Back to Search]", "click[< Prev]", "click[Next >]"),
                ("click[description]", "click[buy now]"),
                structured_memory_after="Ready to purchase: no",
            ),
        ),
        remaining_steps=3,
    )
    text = messages[-1]["content"]
    assert "HISTORICAL action surface" in text
    assert "expired; do not choose from it now" in text
    assert "CURRENT ACTION SURFACE (authoritative for this turn; choose only here)" in text
    assert "CURRENT SURFACE TYPE: subpage/navigation" in text
    assert "Buy Now is not currently legal" in text
    assert "Never output Buy Now when that line is absent" in text


def test_webshop_v7_classifies_result_surface_without_using_private_task_data() -> None:
    messages = render_interactive_messages(
        "webshop-native-react-memory-v7@1",
        task="buy an item",
        current_observation="Public search results",
        available_actions=("click[b012345678]", "click[b087654321]", "click[Next >]"),
        remaining_steps=8,
    )
    text = messages[-1]["content"]
    assert "CURRENT SURFACE TYPE: results" in text
    assert "exact currently listed product ID" in text
    assert "DISCOVERY PHASE" in text


def test_webshop_v8_keeps_v7_grounding_and_uses_the_short_published_workflow() -> None:
    messages = render_interactive_messages(
        "webshop-native-react-memory-v8@1",
        task="buy a large blue yoga shirt below 20 dollars",
        current_observation="Product page with price $18 and selectable blue and large values.",
        available_actions=(
            "click[back to search]",
            "click[description]",
            "click[buy now]",
            "click[blue]",
            "click[large]",
        ),
        history=(
            PublicInteractiveTurn(
                "Action: search[blue large yoga shirt]",
                "search[blue large yoga shirt]",
                "Search page",
                "Results page",
                ("click[b012345678]",),
                ("search",),
            ),
            PublicInteractiveTurn(
                "Action: click[b012345678]",
                "click[b012345678]",
                "Results page",
                "Product page",
                ("click[buy now]", "click[blue]", "click[large]"),
                ("click[b012345678]",),
            ),
        ),
        remaining_steps=4,
    )
    text = messages[-1]["content"]
    assert "HISTORICAL action surface" in text
    assert "CURRENT ACTION SURFACE (authoritative for this turn; choose only here)" in text
    assert "PUBLIC PROGRESS: searches used 1/2; products opened 1" in text
    assert "PRODUCT COMMITMENT" in text
    assert "select only values explicitly required by the task, then buy" in text
    assert "Description, Features, Reviews, Back to Search" in text
    assert "Search attempts (hard plan maximum 2)" in text


def test_webshop_v8_marks_the_second_search_as_the_final_planned_retry() -> None:
    messages = render_interactive_messages(
        "webshop-native-react-memory-v8@1",
        task="buy a blue mug",
        current_observation="Search page",
        available_actions=("search", "click[search]"),
        history=(("search[blue mug]", "No acceptable result was opened."),),
        remaining_steps=6,
    )
    text = messages[-1]["content"]
    assert "PUBLIC PROGRESS: searches used 1/2" in text
    assert "FINAL SEARCH RETRY" in text
    assert "commit from the next result page" in text


def test_webshop_v9_tracks_state_but_omits_thought_from_current_and_history_turns() -> None:
    messages = render_interactive_messages(
        "webshop-native-stateact-memory-v9@1",
        task="buy a large blue yoga shirt below 20 dollars",
        current_observation="Product page with price $18.",
        available_actions=("click[blue]", "click[large]", "click[buy now]"),
        history=(
            PublicInteractiveTurn(
                "Memory: old state\nThought: inspect\nAction: click[b012345678]",
                "click[b012345678]",
                "Results page",
                "Product page",
                ("click[blue]", "click[large]", "click[buy now]"),
                ("click[b012345678]",),
                structured_memory_after="Current location / surface: product",
                visible_thought="inspect",
            ),
        ),
        remaining_steps=4,
    )
    text = "\n".join(message["content"] for message in messages)
    assert "Goal / hard constraints" in text
    assert "Current location / surface" in text
    assert "Current selection" in text
    assert "PRODUCT COMMITMENT" in text
    assert "Thought:" not in text
    assert "output Memory and one Action only" in text


def test_webshop_v10_commits_after_two_searches_without_changing_model_action() -> None:
    history = tuple(
        PublicInteractiveTurn(
            f"Action: {action}",
            action,
            "public state before",
            "public state after",
            ("click[next]",),
            ("click[current]",),
            structured_memory_after="Ready to purchase: no",
        )
        for action in (
            "search[blue yoga shirt]",
            "click[b012345678]",
            "click[back to search]",
            "search[large blue yoga shirt]",
            "click[b087654321]",
        )
    )
    messages = render_interactive_messages(
        "webshop-native-stateact-memory-v10@1",
        task="buy a large blue yoga shirt below 20 dollars",
        current_observation="Product page with price $18.",
        available_actions=("click[back to search]", "click[large]", "click[buy now]"),
        history=history,
        remaining_steps=3,
    )
    text = "\n".join(message["content"] for message in messages)
    assert "PUBLIC PROGRESS: searches used 2/2" in text
    assert "LEGAL ACTION TYPE THIS TURN: click[...] only" in text
    assert "FINAL CANDIDATE AFTER TWO SEARCHES" in text
    assert "Do not leave this candidate" in text
    assert "Thought:" not in text


def test_webshop_v11_teaches_one_search_one_product_without_gating_the_action() -> None:
    history = (
        PublicInteractiveTurn(
            "Action: search[large blue yoga shirt]",
            "search[large blue yoga shirt]",
            "Search page",
            "Results page",
            ("click[b012345678]", "click[b087654321]"),
            ("search",),
            structured_memory_after="Search attempts (planned maximum 1): 1",
        ),
        PublicInteractiveTurn(
            "Action: click[b012345678]",
            "click[b012345678]",
            "Results page",
            "Product page",
            ("click[large]", "click[buy now]", "click[back to search]"),
            ("click[b012345678]", "click[b087654321]"),
            structured_memory_after="Committed product / visible price: current / $18",
        ),
    )
    messages = render_interactive_messages(
        "webshop-native-stateact-memory-v11@1",
        task="buy a large blue yoga shirt below 20 dollars",
        current_observation="Product page with price $18.",
        available_actions=(
            "click[back to search]",
            "click[description]",
            "click[large]",
            "click[buy now]",
        ),
        history=history,
        remaining_steps=3,
    )
    text = "\n".join(message["content"] for message in messages)
    assert "search once" in text.casefold()
    assert "SINGLE-PRODUCT COMMITMENT" in text
    assert "Do not click Back to Search or an information tab" in text
    assert "CURRENT ACTION SURFACE" in text
    assert "click[back to search]" in text
    assert "Thought:" not in text


def test_webshop_v13_keeps_v10_policy_but_retrieves_lexical_train_examples() -> None:
    arguments = {
        "task": "buy red waterproof leather hiking boots",
        "current_observation": "Product page with price $18.",
        "available_actions": ("click[red]", "click[buy now]"),
        "remaining_steps": 4,
    }
    assert render_interactive_messages(
        "webshop-native-stateact-memory-v13@1", **arguments
    ) == render_interactive_messages("webshop-native-stateact-memory-v10@1", **arguments)

    tasks = (
        "buy a red ceramic coffee mug",
        "buy a braided usb charging cable",
        "buy red waterproof leather hiking boots",
        "buy blue trail running shoes",
    )
    examples = tuple(
        InteractivePromptExample(
            example_id=f"train-example-{index}",
            source_entry_id=f"goal-{1500 + index}",
            source_split="train",
            task=task,
            turns=("Observation: public state\nAction: search[query]",),
        )
        for index, task in enumerate(tasks)
    )
    asset = InteractivePromptAsset(
        benchmark="webshop",
        asset_id="test-train-replays",
        source_kind="official-train-replay",
        source_split="train",
        source_revision="test",
        reasoning_mode="action-only",
        examples=examples,
    )
    demonstrations = asset.render_demonstrations(
        task="find waterproof red leather boots for hiking",
        selection_profile="lexical-bm25@1",
    )
    assert len(demonstrations) == 2
    assert f"Task: {tasks[2]}" in demonstrations[0]
    single = asset.render_demonstrations(
        task="find waterproof red leather boots for hiking",
        selection_profile="lexical-bm25-top1@1",
    )
    assert len(single) == 1
    assert f"Task: {tasks[2]}" in single[0]


def test_webshop_v14_keeps_v10_public_state_policy() -> None:
    arguments = {
        "task": "buy red waterproof leather hiking boots",
        "current_observation": "Product page with price $18.",
        "available_actions": ("click[red]", "click[buy now]"),
        "remaining_steps": 4,
    }
    assert render_interactive_messages(
        "webshop-native-stateact-memory-v14@1", **arguments
    ) == render_interactive_messages("webshop-native-stateact-memory-v10@1", **arguments)
    assert render_interactive_messages(
        "webshop-native-stateact-memory-v16@1", **arguments
    ) == render_interactive_messages("webshop-native-stateact-memory-v10@1", **arguments)
    assert render_interactive_messages(
        "webshop-native-stateact-memory-v17@1", **arguments
    ) == render_interactive_messages("webshop-native-stateact-memory-v10@1", **arguments)


def test_webshop_v15_omits_expired_demo_action_lists_without_changing_actions() -> None:
    step = InteractiveReplayStep(
        observation_before="Search page with [Search]",
        available_actions_before=("search",),
        action="search[red mug]",
        observation_after="Results with [item1]",
        available_actions_after=("click[item1]",),
    )
    asset = InteractivePromptAsset(
        benchmark="webshop",
        asset_id="webshop-official-train-replay-memory-v5",
        source_kind="official-train-replay",
        source_split="train",
        source_revision="test",
        reasoning_mode="visible-react",
        examples=(
            InteractivePromptExample(
                example_id="train/example",
                source_entry_id="goal-1500",
                source_split="train",
                task="buy a red mug",
                turns=(),
                replay_steps=(step,),
            ),
        ),
    )
    rendered = asset.render_demonstrations(
        task="buy a red mug",
        structured_memory=True,
        include_thought=False,
        compact_action_surfaces=True,
    )[0]
    assert "Action:\nsearch[red mug]" in rendered
    assert "Admissible actions before" not in rendered
    assert "Returned admissible actions" not in rendered


def test_alfworld_v6_heat_recipe_keeps_the_object_in_inventory() -> None:
    messages = render_interactive_messages(
        "alfworld-native-react-memory-v6@1",
        task="heat an object and place it",
        task_type="pick_heat_then_place",
        current_observation="You are at microwave 1 carrying mug 1.",
        available_actions=("heat mug 1 with microwave 1", "go to cabinet 1"),
        remaining_steps=12,
    )
    text = messages[-1]["content"]
    assert "KEEP IT IN INVENTORY" in text
    assert "Do NOT move the object into the microwave" in text
    assert "heat mug 1 with microwave 1" in text
    assert "Next admissible subgoal action" in text


@pytest.mark.parametrize(
    "task_type",
    [
        "pick_and_place",
        "pick_two_obj",
        "pick_clean_then_place",
        "pick_cool_then_place",
        "look_at_obj",
    ],
)
def test_alfworld_v7_preserves_v5_prompt_for_non_heat_tasks(task_type: str) -> None:
    arguments = {
        "task": "complete the public household task",
        "task_type": task_type,
        "current_observation": "You are in a room.",
        "available_actions": ("go to countertop 1",),
        "remaining_steps": 20,
    }
    assert render_interactive_messages(
        "alfworld-native-react-memory-v7@1", **arguments
    ) == render_interactive_messages("alfworld-native-react-memory-v5@1", **arguments)


def test_alfworld_v7_uses_v6_recipe_for_heat() -> None:
    arguments = {
        "task": "heat a mug and place it",
        "task_type": "pick_heat_then_place",
        "current_observation": "You are at microwave 1 carrying mug 1.",
        "available_actions": ("heat mug 1 with microwave 1",),
        "remaining_steps": 12,
    }
    assert render_interactive_messages(
        "alfworld-native-react-memory-v7@1", **arguments
    ) == render_interactive_messages("alfworld-native-react-memory-v6@1", **arguments)


@pytest.mark.parametrize(
    "asset_id",
    [
        "webshop-official-train-replay-memory-v6",
        "webshop-official-train-efficient-memory-v7",
    ],
)
def test_v6_v7_webshop_asset_requires_concrete_annotations_and_hides_unchosen_surfaces(
    tmp_path: Path,
    asset_id: str,
) -> None:
    examples = []
    for index in range(4):
        examples.append(
            {
                "example_id": f"train/goal-{1600 + index}",
                "source_entry_id": f"goal-{1600 + index}",
                "source_split": "train",
                "task": "buy a train-only item",
                "steps": [
                    {
                        "observation_before": "search page",
                        "available_actions_before": ["search", "click[decoy]"],
                        "action": f"search[item {index}]",
                        "observation_after": "results page",
                        "available_actions_after": ["click[item]", "click[other]"],
                        "memory": (
                            "Hard constraints (verbatim): train-only item\n"
                            "Known violations: none\nUnknown constraints: details\n"
                            "Ready to purchase: no"
                        ),
                        "thought": "Preserve the product core and search once.",
                    }
                ],
            }
        )
    path = tmp_path / "webshop_native_react_v6.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "format": "skillev-public-interactive-prompt@2",
                "benchmark": "webshop",
                "asset_id": asset_id,
                "source_kind": "official-train-replay",
                "source_split": "train",
                "source_revision": "revision",
                "reasoning_mode": "visible-react",
                "examples": examples,
            }
        ),
        encoding="utf-8",
    )
    asset = load_interactive_prompt_asset(path)
    rendered = asset.render_demonstrations(
        task="buy a train-only item",
        structured_memory=True,
    )
    assert len(rendered) == 2
    assert "Preserve the product core" in rendered[0]
    if asset_id.endswith("memory-v6"):
        assert "replay-verified against the native public action surface" in rendered[0]
        assert "click[decoy]" not in rendered[0]
    else:
        assert "Admissible actions before action 1" in rendered[0]
        assert "click[decoy]" in rendered[0]
    thought_free = asset.render_demonstrations(
        task="buy a train-only item",
        structured_memory=True,
        include_thought=False,
    )
    assert "Thought:" not in thought_free[0]
    assert "Memory:" in thought_free[0]
    assert "Action:\nsearch[item 0]" in thought_free[0]

    del examples[0]["steps"][0]["memory"]
    path.write_text(
        yaml.safe_dump(
            {
                "format": "skillev-public-interactive-prompt@2",
                "benchmark": "webshop",
                "asset_id": asset_id,
                "source_kind": "official-train-replay",
                "source_split": "train",
                "source_revision": "revision",
                "reasoning_mode": "visible-react",
                "examples": examples,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_interactive_prompt_asset(path)


def test_webshop_v6_failure_taxonomy_detects_commitment_collapse() -> None:
    row: dict[str, object] = {
        "success": False,
        "reward": 0,
        "steps": 10,
        "max_steps": 10,
        "budget_exhausted": True,
        "terminated_by_horizon": True,
        "terminal_reached": False,
        "termination_reason": "horizon",
        "trace": [
            {
                "step_index": 1,
                "parsed_action": "search[blue mug]",
                "observation_before": "search",
                "observation_after": "results",
                "available_actions_after": ["click[b012345678]"],
                "action_listed_before": True,
            },
            {
                "step_index": 2,
                "parsed_action": "click[b012345678]",
                "observation_before": "results",
                "observation_after": "price: $18 description buy now",
                "available_actions_after": ["click[buy now]"],
                "action_listed_before": True,
            },
            {
                "step_index": 9,
                "parsed_action": "search[unrelated]",
                "observation_before": "product",
                "observation_after": "results",
                "available_actions_after": ["click[b087654321]"],
                "action_listed_before": True,
            },
        ],
    }
    counts = classify_webshop_episode(row, task="buy a blue mug below 20 dollars")
    assert counts["commitment_failure"] == 1
    assert counts["product_inspected_without_purchase"] == 1
    assert counts["horizon_without_purchase"] == 1
    assert counts["late_search"] == 1


def test_webshop_failure_taxonomy_detects_stale_surface_cascade() -> None:
    row: dict[str, object] = {
        "success": False,
        "reward": 0,
        "steps": 3,
        "max_steps": 10,
        "budget_exhausted": False,
        "terminated_by_horizon": False,
        "terminal_reached": False,
        "trace": [
            {
                "step_index": 1,
                "parsed_action": "click[b012345678]",
                "available_actions_before": ["click[b012345678]"],
                "available_actions_after": ["click[buy now]", "click[description]"],
                "action_listed_before": True,
            },
            {
                "step_index": 2,
                "parsed_action": "click[description]",
                "available_actions_before": ["click[buy now]", "click[description]"],
                "available_actions_after": ["click[< Prev]"],
                "action_listed_before": True,
            },
            {
                "step_index": 3,
                "parsed_action": "click[buy now]",
                "available_actions_before": ["click[< Prev]"],
                "available_actions_after": ["click[< Prev]"],
                "action_listed_before": False,
                "unchanged_state": True,
            },
        ],
    }
    counts = classify_webshop_episode(row)
    assert counts["episode_with_unlisted_action"] == 1
    assert counts["first_unlisted_action"] == 1
    assert counts["stale_historical_surface_action"] == 1
    assert counts["unlisted_unchanged_state"] == 1


def test_alfworld_v6_failure_taxonomy_detects_heat_storage_loop() -> None:
    row: dict[str, object] = {
        "success": False,
        "budget_exhausted": True,
        "terminal_reached": True,
        "terminated_by_horizon": False,
        "trace": [
            {"parsed_action": "move mug 1 to microwave 1", "action_listed_before": True},
            {"parsed_action": "close microwave 1", "action_listed_before": True},
            {"parsed_action": "heat mug 1 with microwave 1", "action_listed_before": False},
            {"parsed_action": "open microwave 1", "action_listed_before": True},
            {"parsed_action": "close microwave 1", "action_listed_before": True},
            {"parsed_action": "open microwave 1", "action_listed_before": True},
        ],
    }
    counts = classify_alfworld_failure(row, ALFWorldTaskType.HEAT_THEN_PLACE)
    assert counts["stored_object_in_transformation_appliance"] == 1
    assert counts["transformation_attempt_after_storage"] == 1
    assert counts["unlisted_transformation_action"] == 1
    assert counts["transformation_appliance_open_close_loop"] == 1
    assert counts["simulator_terminal_at_outer_budget"] == 1


def _alfworld_execution():
    protocol = load_protocol_v13(
        ROOT / "configs/evaluation/protocol_v13.yaml",
        ROOT / "configs/evaluation/protocol_v13_sources.yaml",
    )
    executions = load_execution_contracts_v3(
        ROOT / "configs/evaluation/protocol_v13_conditions.yaml",
        protocol=protocol,
    )
    return next(item for item in executions.values() if item.benchmark.value == "alfworld")


def _environment_manifest(simulator_max_steps: int) -> dict[str, object]:
    return {
        "format": "skillev-qwen35-direct-interactive@3",
        "runtimes": {"runtime": {"source_revision": "revision"}},
        "deployments": {
            "alfworld": {
                "kind": "alfworld",
                "runtime": "runtime",
                "simulator_max_steps": simulator_max_steps,
                "games": {},
            }
        },
        "cases": [
            {
                "benchmark": "alfworld",
                "task_id": f"alfworld:{index}",
                "source_identity": f"ALFWorld/train/game-{index}",
                "deployment": "alfworld",
                "max_steps": 20,
                "payload": {"game_id": f"train/game-{index}"},
            }
            for index in range(128)
        ],
    }


def test_alfworld_manifest_separates_outer_and_simulator_horizons(tmp_path: Path) -> None:
    path = tmp_path / "alfworld.json"
    path.write_text(json.dumps(_environment_manifest(50)), encoding="utf-8")
    identity = load_protocol13_environment_manifest_identity(
        path,
        execution=_alfworld_execution(),
    )
    assert identity["deployment"]["simulator_max_steps"] == 50

    path.write_text(json.dumps(_environment_manifest(20)), encoding="utf-8")
    with pytest.raises(ValueError):
        load_protocol13_environment_manifest_identity(path, execution=_alfworld_execution())


def test_alfworld_worker_uses_simulator_horizon_not_outer_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}

    def create_environment(
        source_root: str,
        *,
        config: object,
        data_directory: str,
        max_steps: int,
        seed: int,
        train_eval: str,
    ) -> tuple[object, str]:
        del source_root, config, data_directory, seed, train_eval
        observed["max_steps"] = max_steps
        return object(), "game"

    monkeypatch.setattr(official_environment_worker, "load_alfworld_config", lambda path: {})
    monkeypatch.setattr(
        official_environment_worker,
        "create_alfworld_env",
        create_environment,
    )
    worker = official_environment_worker.ALFWorldWorker(
        "source",
        {
            "config_path": "config",
            "data_directory": "data",
            "instruction_text": "task",
            "seed": 0,
            "train_eval": "train",
        },
        {"game_id": "game", "max_steps": 20, "simulator_max_steps": 50},
    )
    assert worker.outer_max_steps == 20
    assert worker.simulator_max_steps == 50
    assert observed["max_steps"] == 50
