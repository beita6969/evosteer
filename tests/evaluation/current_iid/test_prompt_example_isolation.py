from pathlib import Path

import pytest

from skillev.evaluation.interactive_prompt_assets import load_interactive_prompt_asset

ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize(
    "name",
    ["webshop_native_react_v3.yaml", "alfworld_native_react_v3.yaml"],
)
def test_interactive_examples_are_train_only(name: str) -> None:
    asset = load_interactive_prompt_asset(ROOT / "configs/evaluation/prompts" / name)
    assert asset.source_split == "train"
    assert len(asset.examples) >= (4 if asset.benchmark == "webshop" else 2)
    assert asset.render_demonstrations()
    asset.validate_final_isolation(frozenset({"final/example"}))
    with pytest.raises(ValueError):
        asset.validate_final_isolation(frozenset({asset.examples[0].example_id}))
