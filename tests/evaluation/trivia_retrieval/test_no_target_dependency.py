from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]


def test_formal_retrieval_has_no_auxiliary_model_or_target_dependency() -> None:
    raw = yaml.safe_load(
        (ROOT / "configs/evaluation/qwen35_trivia_public_retrieval.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert raw["auxiliary_models"] == []
    assert raw["target_dependency"] == "forbidden"
    assert raw["tools"]["search_limit"] == 2
    assert raw["tools"]["read_limit"] == 2
