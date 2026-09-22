from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from skillev.evaluation.direct_baseline.protocol import (
    load_direct_reference_protocol,
    validate_runtime_registries,
)

CONFIG = Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")


def test_protocol_loads_all_contracts_and_registries() -> None:
    protocol = load_direct_reference_protocol(CONFIG)
    assert len(protocol.benchmarks) == 14
    assert protocol.parity.max_gap_pp_exclusive == Decimal("7.0")
    validate_runtime_registries(protocol)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("formal_run", "retries_after_candidate_failure"), 1),
        (("model", "adapters"), "optional"),
        (("benchmarks", 0, "n"), 127),
    ],
)
def test_protocol_rejects_formal_identity_drift(
    tmp_path: Path, path: tuple[object, ...], value: object
) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_direct_reference_protocol(candidate)


@pytest.mark.parametrize(
    ("path", "value", "delete"),
    [
        (("upstream", "revision"), None, True),
        (("parity", "metric"), "not-a-parity-metric", False),
        (("benchmarks", 0, "evidence", "prompt"), None, True),
        (("profiles", "qwen35-humaneval-deterministic@1", "temperature"), 0.1, False),
        (("profiles", "qwen35-thinking-general@1", "temperature"), 0.0, False),
        (("benchmarks", 0, "dataset_revision"), "", False),
        (("benchmarks", 0, "selection_rule"), "", False),
        (("benchmarks", 4, "environment_contract"), 7, False),
    ],
)
def test_protocol_rejects_incomplete_scientific_contract(
    tmp_path: Path,
    path: tuple[object, ...],
    value: object,
    delete: bool,
) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    target = raw
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises((KeyError, ValueError)):
        load_direct_reference_protocol(candidate)


def test_protocol_rejects_static_benchmark_bound_to_interactive_prompt(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["benchmarks"][0]["prompt"] = "webshop-native-react@2"
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(raw), encoding="utf-8")
    protocol = load_direct_reference_protocol(candidate)
    with pytest.raises(ValueError, match="prompt kind"):
        validate_runtime_registries(protocol)
