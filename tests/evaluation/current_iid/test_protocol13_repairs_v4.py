from decimal import Decimal
from pathlib import Path

import pytest
from skillev_private.benchmarks.alfworld_taxonomy import (
    ALFWorldTaskType,
    classify_alfworld_game_id,
)

from scripts import run_qwen35_mbpp_plus as mbpp_runner
from skillev.evaluation.current_iid.protocol13.aime_extraction import (
    AIMEExtractionReason,
    extract_aime_boxed,
)
from skillev.evaluation.current_iid.protocol13.healthbench_grading import (
    RubricCriterion,
    RubricVerdict,
    calculate_healthbench_score,
)
from skillev.evaluation.current_iid.protocol13.interactive_demo_assets import (
    ALFWorldDemoStep,
    WebShopDemoStep,
)
from skillev.evaluation.current_iid.protocol13.native_agent_metrics import (
    alfworld_success_rate,
    webshop_panel_metrics,
)


def test_mbpp_evalplus_preserves_virtualenv_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base-python"
    base.touch()
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(base)
    monkeypatch.setattr(mbpp_runner.sys, "executable", str(launcher))

    assert mbpp_runner._current_python_executable() == launcher.absolute()
    assert mbpp_runner._current_python_executable() != launcher.resolve()


def test_aime_parser_rejects_float_and_incomplete_box() -> None:
    assert extract_aime_boxed(r"final \boxed{42}").value == 42
    assert extract_aime_boxed(r"final \boxed{42.0}").reason is (
        AIMEExtractionReason.INVALID_INTEGER
    )
    assert extract_aime_boxed(r"final \boxed{42").reason is (AIMEExtractionReason.INCOMPLETE_BOX)


def test_healthbench_negative_rubric_formula_is_preserved() -> None:
    assert calculate_healthbench_score(
        (
            RubricCriterion("positive", Decimal("2")),
            RubricCriterion("negative", Decimal("-1")),
        ),
        (RubricVerdict(True, "met"), RubricVerdict(True, "met")),
    ) == Decimal("0.5")


def test_demo_actions_must_come_from_public_action_surface() -> None:
    WebShopDemoStep(
        "before",
        ("search",),
        "search[blue cotton shirt]",
        "after",
        (),
        Decimal(0),
        False,
    )
    with pytest.raises(ValueError, match="not available"):
        WebShopDemoStep("before", ("click[a]",), "click[b]", "after", (), Decimal(0), False)
    with pytest.raises(ValueError, match="not admissible"):
        ALFWorldDemoStep("before", ("look",), "take mug", "after", (), False, None)


def test_native_agent_headlines_use_exact_128_denominator() -> None:
    assert webshop_panel_metrics((Decimal(1),) * 128).success_rate_percent == Decimal(100)
    assert alfworld_success_rate((True,) * 128) == Decimal(100)


def test_alfworld_taxonomy_accepts_official_split_prefixed_game_ids() -> None:
    assert (
        classify_alfworld_game_id(
            "valid_seen/look_at_obj_in_light-Bowl-None-DeskLamp-316/trial-example"
        )
        is ALFWorldTaskType.LOOK_AT_OBJECT
    )
