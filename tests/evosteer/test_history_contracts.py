from dataclasses import replace

import pytest

from skillev.contracts.canonical import canonical_json, stable_hash
from skillev.contracts.evosteer import DecisionRecord, EvoTask, EvoTrajectory


def decision(*, forced=False, history=()):
    state = {"graph": {"nodes": []}, "history": list(history)}
    return DecisionRecord(
        stable_hash(state),
        canonical_json(state),
        '{"kind":"STOP"}',
        (1, 2),
        (3, 0),
        ((3, 0), (4, 0)),
        (0.0,) * 30,
        (),
        0.5,
        forced,
    )


def trajectory(**overrides):
    values = {
        "sample_id": "sample-1",
        "batch_id": "batch-1",
        "task": EvoTask("task-1", "math", "Public problem only"),
        "source": "current",
        "behavior_policy_id": "actor-1",
        "reference_id": "rho-1",
        "executor_id": "executor-1",
        "menu_id": "skills-1",
        "value_snapshot_id": "value-1",
        "statistics_context": "context-1",
        "decisions": (decision(),),
        "terminal_state_json": '{"stopped":true}',
        "reward": 1.0,
        "output": "answer",
        "usage_json": '{"output_tokens":2}',
    }
    return EvoTrajectory(**{**values, **overrides})


def test_records_roundtrip_exact_tokens_and_empty_deferred_encoding():
    history = trajectory()
    restored = EvoTrajectory.from_value(history.to_value())
    assert restored == history
    assert restored.identity == history.identity
    assert restored.decisions[0].reference_encoding == ()


def test_same_graph_with_different_history_has_distinct_identity():
    first = decision(history=("a",))
    second = decision(history=("b",))
    assert first.state_id != second.state_id
    with pytest.raises(ValueError):
        replace(first, state_id=second.state_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forced", 1),
        ("features", [0.0] * 30),
        ("features", (True,) * 30),
        ("value_estimate", True),
        ("prompt_ids", (True,)),
        ("prompt_ids", [1]),
        ("reference_encoding", (float("inf"),)),
        ("state_json", "[]"),
        ("action_json", '{ "kind":"STOP"}'),
        ("legal_token_paths", ((3, 0), (3,))),
    ],
)
def test_decision_rejects_invalid_typed_fields(field, value):
    with pytest.raises(ValueError):
        replace(decision(), **{field: value})


def test_from_value_rejects_extra_fields_bad_arrays_and_invalid_nested_task():
    raw = decision().to_value()
    with pytest.raises(ValueError):
        DecisionRecord.from_value({**raw, "unknown": 1})
    with pytest.raises(ValueError):
        DecisionRecord.from_value({**raw, "prompt_ids": "12"})
    with pytest.raises(ValueError):
        DecisionRecord.from_value({**raw, "legal_token_paths": [[3, 0], [True]]})
    with pytest.raises(ValueError):
        EvoTrajectory.from_value({**trajectory().to_value(), "task": {"task_id": "q"}})


@pytest.mark.parametrize(
    "overrides",
    [
        {"pair_id": "p"},
        {"candidate_id": "c"},
        {"sample_id": ""},
        {"reward": True},
        {"decisions": [decision()]},
        {"terminal_state_json": "null"},
        {"terminal_state_json": '{"stopped":false}'},
        {"usage_json": '{"output_tokens":-1}'},
        {"source": "natural_reference", "behavior_policy_id": "actor"},
        {"source": "paired_treatment", "pair_id": "p", "candidate_id": "c"},
    ],
)
def test_trajectory_provenance_and_resource_validation(overrides):
    with pytest.raises(ValueError):
        trajectory(**overrides)


def test_paired_history_forces_exactly_first_action_under_frozen_behavior():
    paired = trajectory(
        source="paired_control",
        behavior_policy_id="rho-1",
        pair_id="p",
        candidate_id="c",
        decisions=(decision(forced=True), decision(history=("first",))),
    )
    assert EvoTrajectory.from_value(paired.to_value()) == paired
    with pytest.raises(ValueError):
        replace(
            paired, decisions=(decision(forced=True), decision(forced=True, history=("first",)))
        )
