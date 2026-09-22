"""Snapshot substitution cannot turn OOD into training or change evaluation semantics."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from skillev_private.evaluation.architecture_matched import (
    FORMAT,
    evaluation_isolation,
    freeze_architecture,
    require_architecture_match,
    substitute_snapshot,
)
from skillev_private.evaluation.integrity_sources import SourcePanel, order_source_panel

from skillev.evaluation.input_metric_contracts import OOD_BENCHMARKS, PublicTaskView
from skillev.evaluation.integrity_pipeline import FrozenPanel
from skillev.evaluation.step0_integrity import InferenceArm, SkillMode
from skillev.evolution.skill_access import ACTIVE_APPLICABILITY_RULE
from tests.evaluation.test_integrity_policy_runtime_controls import production_runtime
from tests.evaluation.test_step0_integrity_pipeline import panel


def architecture():
    return {
        "format": FORMAT,
        "runtime_config": {
            "catalog": "ood",
            "arms": [InferenceArm("base").to_value()],
            "evaluation_sample_counts": {"livecodebench": 64},
            "budgets": {"livecodebench": {"total_output_tokens": 12000}},
            "scorers": {"omni-math": {"judge_model": "gpt-5.6-luna"}},
            "thinking_policy": {"thinking_by_benchmark": {"livecodebench": False}},
        },
    }


def test_policy_selection_does_not_mutate_any_evaluation_configuration(tmp_path):
    source = architecture()
    original = deepcopy(source)
    selected = substitute_snapshot(
        source,
        {
            "arm_id": "trained",
            "policy_id": "explicit-step16",
            "optimizer_steps": 16,
            "policy": {"checkpoint_directory": "/private/explicit16", "adapter_name": "forward16"},
        },
        tmp_path,
    )
    assert source == original
    for field in ("catalog", "evaluation_sample_counts", "budgets", "thinking_policy"):
        assert selected[field] == source["runtime_config"][field]
    assert selected["trained_policies"]["explicit-step16"]["adapter_name"] == "forward16"
    assert selected["scorers"]["omni-math"]["judge_model"] == "gpt-5.6-luna"
    assert selected["scorers"]["omni-math"]["cache_directory"].startswith(str(tmp_path))


@pytest.mark.parametrize("field", ["budgets", "decoding", "endpoints", "training_batch", "scorers"])
def test_snapshot_does_not_accept_architecture_or_training_overrides(tmp_path, field):
    with pytest.raises(ValueError):
        substitute_snapshot(architecture(), {"arm_id": "new", field: {}}, tmp_path)


def test_library_snapshot_is_not_a_skill_on_intervention(tmp_path):
    with pytest.raises(ValueError):
        substitute_snapshot(architecture(), {"arm_id": "new", "library_id": "evolved"}, tmp_path)


def test_library_aware_substitution_keeps_retrieval_surface(tmp_path):
    source = architecture()
    source["runtime_config"]["arms"] = [
        InferenceArm(
            "initial",
            skill_mode=SkillMode.LIBRARY,
            skill_library_id="initial",
            skill_retrieval_rule=ACTIVE_APPLICABILITY_RULE,
        ).to_value()
    ]
    selection = {
        "arm_id": "evolved",
        "library_id": "evolved",
        "library": {
            "kind": "evolved",
            "checkpoint_directory": "/private/explicit16",
            "retrieval_rule": ACTIVE_APPLICABILITY_RULE,
        },
    }
    selected = substitute_snapshot(source, selection, tmp_path)
    assert selected["arms"][0]["skill_library_id"] == "evolved"
    assert selected["trained_policies"] == {}
    selection["library"]["retrieval_rule"] = "changed"
    with pytest.raises(ValueError):
        substitute_snapshot(source, selection, tmp_path)


def test_freeze_copies_inputs_and_scorer_not_historical_predictions(tmp_path):
    source = architecture()["runtime_config"]
    data, checker = tmp_path / "data.jsonl", tmp_path / "checker.py"
    data.write_text('{"task_id":"synthetic-public-question"}\n')
    checker.write_text("# synthetic official checker fixture\n")
    source["ood_sources"] = {"livecodebench": str(data)}
    source["scorers"]["livecodebench"] = {"checker_path": str(checker)}
    target = freeze_architecture(source, tmp_path / "frozen", "X")
    frozen = json.loads(target.read_text())["runtime_config"]
    data.write_text("changed source after freezing")
    assert (
        "synthetic-public-question" in (tmp_path / "frozen/livecodebench-private.jsonl").read_text()
    )
    assert frozen["scorers"]["livecodebench"]["checker_path"] != str(checker)
    assert frozen["architecture_id"] == "X"


def test_actual_policy_controls_match_but_budget_and_native_scorer_drift_do_not(
    tmp_path, monkeypatch
):
    runtime, _, (base, trained) = production_runtime(tmp_path, monkeypatch)
    try:
        left = runtime.controls(base, panel().entries)
        right = runtime.controls(trained, panel().entries)
        assert require_architecture_match(left, right, base, trained) == ("forward-policy",)
        for field in ("budgets", "tools", "sampling", "evaluator", "parser", "public_inputs"):
            changed = (("changed-input", "text"),) if field == "public_inputs" else {"changed": 1}
            with pytest.raises(ValueError):
                require_architecture_match(left, replace(right, **{field: changed}), base, trained)
        isolation = right.parser["evaluation_isolation"]
        assert isolation == evaluation_isolation()
        assert isolation["training_updates"] == isolation["posterior_updates"] == 0
        assert isolation["skill_evolution"] == isolation["training_evidence_writes"] == 0
        assert (
            runtime.journal.connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == 0
        )
    finally:
        asyncio.run(runtime.close())


def test_same_snapshot_remains_matched_and_rejects_silent_semantics_change(tmp_path, monkeypatch):
    runtime, _, (base, _) = production_runtime(tmp_path, monkeypatch)
    try:
        controls = runtime.controls(base, panel().entries)
        assert (
            require_architecture_match(controls, controls, base, replace(base, arm_id="again"))
            == ()
        )
        changed = replace(controls, service={**controls.service, "request_timeout_seconds": 7})
        with pytest.raises(ValueError):
            require_architecture_match(controls, changed, base, replace(base, arm_id="again"))
    finally:
        asyncio.run(runtime.close())


def test_frozen_ood_scheduling_uses_ood_catalog_not_iid():
    entries = (
        PublicTaskView.from_record("nq", "nq-open", {"question": "Synthetic question?"}),
        PublicTaskView.from_record("mu", "musique", {"question": "Other?", "context": "Public"}),
    )
    frozen = FrozenPanel(entries, "development", "synthetic", catalog=OOD_BENCHMARKS)
    source = SourcePanel(frozen, {}, {}, {})
    ordered = order_source_panel(source, OOD_BENCHMARKS)
    assert ordered.panel.entries[0].benchmark == "musique"
    assert set(ordered.panel.entries) == set(entries)
