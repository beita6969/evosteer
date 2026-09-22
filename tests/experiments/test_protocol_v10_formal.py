from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from skillev.experiments.protocol_v10 import FormalMethodV10, ProtocolV10Error
from skillev.experiments.protocol_v10_formal import (
    FormalApplicationV10,
    load_protocol_v10_formal_experiment,
    parse_protocol_v10_formal_experiment,
)

ROOT = Path(__file__).parents[2]
FORMAL_PATH = ROOT / "configs" / "experiments" / "protocol_v10_formal.yaml"


def _raw() -> dict[str, object]:
    value = yaml.safe_load(FORMAL_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_formal_protocol_consumes_all_4608_episodes_in_288_steps() -> None:
    spec = load_protocol_v10_formal_experiment(FORMAL_PATH)

    assert spec.batch_size == 16
    assert spec.total_steps == 288
    assert spec.total_episodes == 4_608
    assert spec.batch_size * spec.total_steps == spec.total_episodes
    assert spec.phase_search_steps == 272
    assert spec.closure_steps == 16
    assert spec.run_plan.total_training_steps == 288
    assert spec.executable is True


def test_formal_methods_have_distinct_non_alias_application_semantics() -> None:
    spec = load_protocol_v10_formal_experiment(FORMAL_PATH)

    assert tuple(binding.method for binding in spec.methods) == tuple(FormalMethodV10)
    assert tuple(binding.application for binding in spec.methods) == (
        FormalApplicationV10.EXACT_SKILLFLOW_BASELINE,
        FormalApplicationV10.BAYESIAN_IMPROVE_FULL,
        FormalApplicationV10.BAYESIAN_IMPROVE_NO_CALIBRATION,
    )


def test_workflow_concurrency_is_not_part_of_formal_method_identity() -> None:
    raw = _raw()

    assert "workflow" not in raw
    assert "concurrency" not in raw
    assert "gpu" not in raw


def test_formal_reader_rejects_incomplete_episode_shape_and_method_alias() -> None:
    wrong_shape = deepcopy(_raw())
    training = wrong_shape["training"]
    assert isinstance(training, dict)
    training["total_steps"] = 287
    with pytest.raises(ProtocolV10Error):
        parse_protocol_v10_formal_experiment(wrong_shape)

    alias = deepcopy(_raw())
    methods = alias["methods"]
    assert isinstance(methods, list)
    third = methods[2]
    assert isinstance(third, dict)
    third["application"] = "exact-skillflow-baseline"
    with pytest.raises(ProtocolV10Error):
        parse_protocol_v10_formal_experiment(alias)
