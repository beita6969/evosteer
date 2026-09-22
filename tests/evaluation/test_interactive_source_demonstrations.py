from pathlib import Path

import pytest
import yaml

from skillev.evaluation.interactive_prompt_assets import load_interactive_prompt_asset


def _webshop_asset(*, invalid_action: bool = False) -> dict[str, object]:
    examples = []
    for index in range(4):
        action = "click[missing]" if invalid_action and index == 0 else "click[item]"
        examples.append(
            {
                "example_id": f"webshop-train-demo-{index}",
                "source_entry_id": f"goal-{index}",
                "source_split": "train",
                "task": "buy the visible item",
                "steps": [
                    {
                        "observation_before": "results page",
                        "available_actions_before": ["click[item]"],
                        "action": action,
                        "observation_after": "product page",
                    }
                ],
            }
        )
    return {
        "format": "skillev-public-interactive-prompt@2",
        "benchmark": "webshop",
        "asset_id": "webshop-official-train-replay-v4",
        "source_kind": "official-train-replay",
        "source_split": "train",
        "source_revision": "official-source-revision",
        "reasoning_mode": "visible-react",
        "examples": examples,
    }


def test_webshop_formal_demonstrations_require_actions_from_replayed_surface(
    tmp_path: Path,
) -> None:
    path = tmp_path / "webshop.yaml"
    path.write_text(yaml.safe_dump(_webshop_asset()), encoding="utf-8")
    asset = load_interactive_prompt_asset(path)
    assert len(asset.examples) == 4
    assert all(example.source_split == "train" for example in asset.examples)
    asset.validate_final_isolation(frozenset({"final-goal"}))

    path.write_text(yaml.safe_dump(_webshop_asset(invalid_action=True)), encoding="utf-8")
    with pytest.raises(ValueError):
        load_interactive_prompt_asset(path)


def test_source_entry_identity_prevents_train_final_overlap(tmp_path: Path) -> None:
    path = tmp_path / "webshop.yaml"
    path.write_text(yaml.safe_dump(_webshop_asset()), encoding="utf-8")
    asset = load_interactive_prompt_asset(path)
    with pytest.raises(ValueError):
        asset.validate_final_isolation(frozenset({"goal-2"}))
