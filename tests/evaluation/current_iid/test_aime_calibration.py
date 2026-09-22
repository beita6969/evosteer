from pathlib import Path

import pytest

from scripts.calibrate_qwen35_aime_protocol import validate_calibration_year
from skillev.evaluation.current_iid.protocol13.aime_profiles import (
    load_aime_profile_registry,
)

ROOT = Path(__file__).parents[3]


def test_aime_final_panel_cannot_calibrate_decoding() -> None:
    validate_calibration_year(2025)
    with pytest.raises(ValueError):
        validate_calibration_year(2026)


def test_protocol13_aime_profile_is_typed_pre_final_evidence_not_author_exact() -> None:
    registry = load_aime_profile_registry(ROOT / "configs/evaluation/aime_protocol_candidates.yaml")
    profile = registry.require("qwen35-thinking-aime-boxed-candidate@3", calibration_year=2025)
    assert registry.final_population == "aime-2026-all-30-v13"
    assert profile.decoding_profile == "qwen35-thinking-math-reference@2"
    assert not profile.formal_reference_eligible
