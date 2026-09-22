from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from scripts.run_qwen35_direct_interactive import (
    _compact_demonstration_action_surfaces,
    _demonstration_selection_profile,
    _prompt_asset_version,
    _uses_task_specific_demonstrations,
)
from scripts.run_qwen35_react_memory_ablation import (
    ReactAblationVariant,
    _configure_cases,
    ablation_settings,
    require_no_infrastructure_failures,
    validate_final_disjointness,
)
from skillev.evaluation.direct_baseline.config import (
    DirectBenchmark,
    DirectDecodingProfile,
)
from skillev.evaluation.direct_baseline.interactive_tasks import (
    InvalidCandidatePolicy,
    NativeInteractiveTask,
)
from skillev.evaluation.interactive_prompt_assets import load_interactive_prompt_asset


def _manifest(source: str, *, task_id: str = "diagnostic:0") -> dict[str, object]:
    return {
        "format": "skillev-qwen35-direct-interactive@3",
        "cases": [
            {
                "benchmark": "webshop",
                "task_id": task_id,
                "source_identity": f"WebShop/{source}",
                "payload": {"goal_id": source},
            }
        ],
    }


def test_ablation_ladder_changes_only_the_declared_method_components() -> None:
    current = ablation_settings(DirectBenchmark.WEB_SHOP, ReactAblationVariant.CURRENT_V4)
    history = ablation_settings(DirectBenchmark.WEB_SHOP, ReactAblationVariant.FULL_HISTORY)
    memory = ablation_settings(DirectBenchmark.WEB_SHOP, ReactAblationVariant.STRUCTURED_MEMORY)
    demos = ablation_settings(DirectBenchmark.WEB_SHOP, ReactAblationVariant.TASK_SPECIFIC_DEMOS)

    assert current.history_window_steps == 8
    assert current.history_maximum_characters == 30_000
    assert history == replace(current, history_window_steps=None, history_maximum_characters=None)
    assert memory.prompt_profile == "webshop-native-react-memory-v5@1"
    assert memory.parser_profile == "native-action-memory-v5@1"
    assert memory.demonstration_mode == "none"
    assert demos == replace(memory, demonstration_mode="targeted-safe-train-replays")


def test_alfworld_current_window_is_sixteen_and_all_variants_use_model_actions() -> None:
    current = ablation_settings(DirectBenchmark.ALF_WORLD, ReactAblationVariant.CURRENT_V4)
    memory = ablation_settings(DirectBenchmark.ALF_WORLD, ReactAblationVariant.STRUCTURED_MEMORY)
    assert current.history_window_steps == 16
    assert current.invalid_candidate_policy is InvalidCandidatePolicy.TERMINATE_ZERO
    assert memory.invalid_candidate_policy is InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE


def test_v6_ablation_variants_keep_webshop_and_alfworld_changes_separate() -> None:
    state = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V6_STATE_MEMORY,
    )
    budget = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V6_HORIZON_POLICY,
    )
    heat_recipe = ablation_settings(
        DirectBenchmark.ALF_WORLD,
        ReactAblationVariant.V6_TASK_RECIPE,
    )
    assert state.prompt_profile == "webshop-native-react-memory-v6-state@1"
    assert budget == replace(state, prompt_profile="webshop-native-react-memory-v6@1")
    assert heat_recipe.prompt_profile == "alfworld-native-react-memory-v6@1"
    assert heat_recipe.demonstration_mode == "none"
    with pytest.raises(ValueError):
        ablation_settings(DirectBenchmark.ALF_WORLD, ReactAblationVariant.V6_STATE_MEMORY)


def test_webshop_v7_reuses_v6_train_evidence_and_only_changes_public_policy() -> None:
    settings = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V7_SURFACE_GROUNDING,
    )
    assert settings.prompt_profile == "webshop-native-react-memory-v7@1"
    assert settings.parser_profile == "native-action-memory-v6@1"
    assert settings.prompt_asset_version == "v6"
    assert settings.demonstration_mode == "targeted-safe-train-replays"
    assert _prompt_asset_version(settings.prompt_profile) == "v6"
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V7_SURFACE_GROUNDING,
        )


def test_webshop_v8_reuses_v6_train_evidence_and_preserves_model_action_ownership() -> None:
    settings = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V8_PUBLISHED_WORKFLOW,
    )
    assert settings.prompt_profile == "webshop-native-react-memory-v8@1"
    assert settings.parser_profile == "native-action-memory-v6@1"
    assert settings.prompt_asset_version == "v6"
    assert settings.invalid_candidate_policy is InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE
    assert _prompt_asset_version(settings.prompt_profile) == "v6"
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V8_PUBLISHED_WORKFLOW,
        )


def test_webshop_v9_reuses_train_evidence_but_removes_visible_thoughts() -> None:
    settings = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V9_STATEACT_NO_THOUGHT,
    )
    assert settings.prompt_profile == "webshop-native-stateact-memory-v9@1"
    assert settings.parser_profile == "native-action-memory-v6@1"
    assert settings.prompt_asset_version == "v6"
    assert settings.demonstration_mode == "targeted-safe-train-replays"
    assert not settings.include_thought
    assert _prompt_asset_version(settings.prompt_profile) == "v6"
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V9_STATEACT_NO_THOUGHT,
        )


def test_webshop_v10_adds_only_the_bounded_public_commitment_policy() -> None:
    v9 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V9_STATEACT_NO_THOUGHT,
    )
    v10 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V10_BOUNDED_COMMITMENT,
    )
    assert v10 == replace(v9, prompt_profile="webshop-native-stateact-memory-v10@1")
    assert _prompt_asset_version(v10.prompt_profile) == "v6"
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V10_BOUNDED_COMMITMENT,
        )


def test_webshop_v11_reuses_train_evidence_and_adds_single_product_policy() -> None:
    v10 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V10_BOUNDED_COMMITMENT,
    )
    v11 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V11_SINGLE_PRODUCT_COMMITMENT,
    )
    assert v11 == replace(
        v10,
        prompt_profile="webshop-native-stateact-memory-v11@1",
    )
    assert _prompt_asset_version(v11.prompt_profile) == "v6"
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V11_SINGLE_PRODUCT_COMMITMENT,
        )


def test_webshop_v12_is_the_released_skillflow_action_only_control() -> None:
    settings = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V12_SKILLFLOW_SOURCE_ACTION_ONLY,
    )
    assert settings.prompt_profile == "webshop-skillflow-action-only-v1@1"
    assert settings.parser_profile == "native-action@2"
    assert settings.history_window_steps is None
    assert settings.history_maximum_characters is None
    assert settings.demonstration_mode == "none"
    assert not settings.include_thought
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V12_SKILLFLOW_SOURCE_ACTION_ONLY,
        )


def test_webshop_v13_changes_v10_only_by_train_demo_retrieval() -> None:
    v10 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V10_BOUNDED_COMMITMENT,
    )
    v13 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V13_EXPEL_BM25_DEMONSTRATIONS,
    )
    assert v13 == replace(
        v10,
        prompt_profile="webshop-native-stateact-memory-v13@1",
        demonstration_selection_profile="lexical-bm25@1",
    )
    assert _prompt_asset_version(v13.prompt_profile) == "v6"
    assert _demonstration_selection_profile(v13.prompt_profile) == "lexical-bm25@1"
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V13_EXPEL_BM25_DEMONSTRATIONS,
        )


def test_webshop_v14_changes_v10_only_to_short_source_proven_replays() -> None:
    v10 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V10_BOUNDED_COMMITMENT,
    )
    v14 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V14_EFFICIENT_TRAIN_REPLAYS,
    )
    assert v14 == replace(
        v10,
        prompt_profile="webshop-native-stateact-memory-v14@1",
        prompt_asset_version="v5",
    )
    assert _prompt_asset_version(v14.prompt_profile) == "v5"
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V14_EFFICIENT_TRAIN_REPLAYS,
        )


def test_webshop_v15_compacts_only_the_v14_demonstration_surface() -> None:
    v14 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V14_EFFICIENT_TRAIN_REPLAYS,
    )
    v15 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V15_EFFICIENT_COMPACT_TRAIN_REPLAYS,
    )
    assert v15 == replace(
        v14,
        prompt_profile="webshop-native-stateact-memory-v15@1",
        compact_demonstration_action_surfaces=True,
    )
    assert _prompt_asset_version(v15.prompt_profile) == "v5"
    assert _compact_demonstration_action_surfaces(v15.prompt_profile)
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V15_EFFICIENT_COMPACT_TRAIN_REPLAYS,
        )


def test_webshop_v16_combines_diverse_efficient_replays_with_bm25() -> None:
    v14 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V14_EFFICIENT_TRAIN_REPLAYS,
    )
    v16 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V16_DIVERSE_EFFICIENT_BM25_REPLAYS,
    )
    assert v16 == replace(
        v14,
        prompt_profile="webshop-native-stateact-memory-v16@1",
        prompt_asset_version="v7",
        demonstration_selection_profile="lexical-bm25@1",
    )
    assert _prompt_asset_version(v16.prompt_profile) == "v7"
    assert _demonstration_selection_profile(v16.prompt_profile) == "lexical-bm25@1"
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V16_DIVERSE_EFFICIENT_BM25_REPLAYS,
        )


def test_webshop_v17_retrieves_one_full_efficient_replay() -> None:
    v16 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V16_DIVERSE_EFFICIENT_BM25_REPLAYS,
    )
    v17 = ablation_settings(
        DirectBenchmark.WEB_SHOP,
        ReactAblationVariant.V17_SINGLE_EFFICIENT_BM25_REPLAY,
    )
    assert v17 == replace(
        v16,
        prompt_profile="webshop-native-stateact-memory-v17@1",
        demonstration_selection_profile="lexical-bm25-top1@1",
    )
    assert _prompt_asset_version(v17.prompt_profile) == "v7"
    assert _demonstration_selection_profile(v17.prompt_profile) == "lexical-bm25-top1@1"
    with pytest.raises(ValueError):
        ablation_settings(
            DirectBenchmark.ALF_WORLD,
            ReactAblationVariant.V17_SINGLE_EFFICIENT_BM25_REPLAY,
        )


def test_alfworld_v7_reuses_v6_evidence_and_limits_demos_to_heat() -> None:
    profile = "alfworld-native-react-memory-v7@1"
    assert _prompt_asset_version(profile) == "v6"
    assert _uses_task_specific_demonstrations(profile, "pick_heat_then_place")
    assert not _uses_task_specific_demonstrations(profile, "pick_clean_then_place")


def test_validation_panel_rejects_final_task_source_or_environment_overlap() -> None:
    final = _manifest("goal-7", task_id="final:0")
    assert (
        validate_final_disjointness(_manifest("goal-8"), final, benchmark="webshop")[0]["task_id"]
        == "diagnostic:0"
    )
    with pytest.raises(ValueError, match="overlaps"):
        validate_final_disjointness(_manifest("goal-7"), final, benchmark="webshop")


def test_webshop_v5_asset_rejects_mislabeled_non_train_goal(tmp_path: Path) -> None:
    examples = []
    for index in (1499, 2000):
        examples.append(
            {
                "example_id": f"train/goal-{index}",
                "source_entry_id": f"goal-{index}",
                "source_split": "train",
                "task": "buy item",
                "steps": [
                    {
                        "observation_before": "search page",
                        "available_actions_before": ["search"],
                        "action": "search[item]",
                        "observation_after": "results",
                        "available_actions_after": ["click[item]"],
                    }
                ],
            }
        )
    path = tmp_path / "webshop_native_react_v5.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "format": "skillev-public-interactive-prompt@2",
                "benchmark": "webshop",
                "asset_id": "webshop-official-train-replay-memory-v5",
                "source_kind": "official-train-replay",
                "source_split": "train",
                "source_revision": "revision",
                "reasoning_mode": "visible-react",
                "examples": examples,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official train range"):
        load_interactive_prompt_asset(path)


def test_variant_configuration_preserves_task_action_execution_identity() -> None:
    # The case factory is a sentinel: configuring the prompt/memory ladder must
    # not wrap or replace the official environment/action path.
    from scripts import run_qwen35_direct_interactive as legacy

    profile = DirectDecodingProfile("p", False, 0.7, 0.8, 20, 0, 1.5, 1, 32)
    task = NativeInteractiveTask(
        "webshop:test",
        DirectBenchmark.WEB_SHOP,
        "buy item",
        profile,
        10,
        "webshop-native-react-memory-v5@1",
        "native-action-memory-v5@1",
        "diagnostic",
        42,
        demonstrations=("targeted",),
    )

    def sentinel() -> None:
        return None

    case = legacy._Case(task, sentinel)
    settings = ablation_settings(DirectBenchmark.WEB_SHOP, ReactAblationVariant.STRUCTURED_MEMORY)

    class Counter:
        def count(self, messages: object, *, enable_thinking: bool) -> int:
            return 1

    configured = _configure_cases(
        (case,),
        settings=settings,
        all_safe_demonstrations=("old",),
        counter=Counter(),  # type: ignore[arg-type]
        context_length=98_304,
        replica_count=2,
    )[0]
    assert configured.create_environment is sentinel
    assert configured.task.demonstrations == ()
    assert configured.task.max_steps == 10
    assert configured.task.service_slot == 0


def test_diagnostic_sequence_stops_after_infrastructure_failure() -> None:
    from skillev.evaluation.direct_baseline.interactive_tasks import NativeInteractiveAttempt

    failed = NativeInteractiveAttempt(
        "webshop:test",
        DirectBenchmark.WEB_SHOP,
        None,
        None,
        0,
        0,
        0,
        False,
        False,
        "EnvironmentCreationError",
    )
    with pytest.raises(RuntimeError, match="stop before next variant"):
        require_no_infrastructure_failures((failed,))
