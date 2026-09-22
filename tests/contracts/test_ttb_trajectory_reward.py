from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.contracts.canonical import canonical_json, parse_canonical_json, stable_hash
from skillev.contracts.ttb_reward import SuccessRule, TerminalReward
from skillev.contracts.ttb_trajectory import (
    InitialContext,
    TrajectoryRecord,
    TrajectoryStep,
    build_trajectory_record,
)


class CharacterTokenizer:
    @property
    def tokenizer_id(self) -> str:
        return "character-tokenizer@1"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class CountingTokenizer(CharacterTokenizer):
    def __init__(self) -> None:
        self.identity_reads = 0
        self.decoded_ids: list[tuple[int, ...]] = []

    @property
    def tokenizer_id(self) -> str:
        self.identity_reads += 1
        return super().tokenizer_id

    def decode(self, token_ids: tuple[int, ...]) -> str:
        self.decoded_ids.append(token_ids)
        return super().decode(token_ids)


class NonTextDecodingTokenizer(CharacterTokenizer):
    def decode(self, token_ids: tuple[int, ...]) -> str:
        del token_ids
        return ["not", "text"]  # type: ignore[return-value]


class ManyToOneTokenizer(CharacterTokenizer):
    sampled_ids = (41, 42)
    canonical_ids = (99,)
    action_text = "act"

    def encode(self, text: str) -> list[int]:
        if text == self.action_text:
            return list(self.canonical_ids)
        return super().encode(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        if token_ids in {self.sampled_ids, self.canonical_ids}:
            return self.action_text
        return super().decode(token_ids)


def _context() -> InitialContext:
    return InitialContext(
        query="solve the task",
        retrieved_skill_ids=("skill-alpha",),
        active_skill_ids=("skill-alpha",),
        meta={
            "environment_id": "debug-environment",
            "task_family": "debug-family",
            "tools": ["calculator"],
        },
        assembler_version="assembler-v1",
        assembled_hash=stable_hash({"assembled": "H0"}),
        assembled_token_count=12,
    )


def _reward(
    *,
    value: float = 1.0,
    success: bool = True,
    success_rule: SuccessRule = SuccessRule.R_EQUALS_ONE,
    success_threshold: float | None = None,
    native_payload: object | None = None,
) -> TerminalReward:
    payload = {"score": value} if native_payload is None else native_payload
    assert isinstance(payload, dict)
    return TerminalReward(
        value=value,
        success=success,
        success_rule=success_rule,
        success_threshold=success_threshold,
        native_metric_name="binary",
        native_payload=payload,
        environment_id="debug-environment",
        verifier_version="verifier-v1",
    )


def _step(
    *,
    index: int = 1,
    action_text: str = "act",
    action_token_ids: tuple[int, ...] | None = None,
    action_token_count: int | None = None,
    observation_status: str = "success",
    invoked_skill_ids: tuple[str, ...] = (),
) -> TrajectoryStep:
    token_ids = (
        tuple(ord(character) for character in action_text)
        if action_token_ids is None
        else action_token_ids
    )
    return TrajectoryStep(
        index=index,
        reasoning_text="choose a tool",
        action_text=action_text,
        action_token_ids=token_ids,
        action_token_count=len(token_ids) if action_token_count is None else action_token_count,
        observation_text="tool completed",
        observation_status=observation_status,
        invoked_skill_ids=invoked_skill_ids,
        forward_prefix_hash=stable_hash({"prefix": "forward", "index": index}),
        hindsight_prefix_hash=stable_hash({"prefix": "hindsight", "index": index}),
    )


def _record(
    *,
    tokenizer: CharacterTokenizer | None = None,
    steps: tuple[TrajectoryStep, ...] | None = None,
    horizon: int | None = None,
    reward: TerminalReward | None = None,
    shifted_reward: float | None = None,
    epsilon_min: float = 0.01,
    tokenizer_id: str = "character-tokenizer@1",
    trajectory_id: str = "trajectory-one",
    environment_id: str = "debug-environment",
    task_family: str = "debug-family",
    decoding_snapshot_id: str = "decode-v1",
    created_at: str = "2026-07-19T00:00:00Z",
) -> TrajectoryRecord:
    actual_steps = (_step(),) if steps is None else steps
    actual_reward = _reward() if reward is None else reward
    actual_shifted = actual_reward.value + epsilon_min if shifted_reward is None else shifted_reward
    initial_context = replace(
        _context(),
        meta={
            "environment_id": environment_id,
            "task_family": task_family,
            "tools": ["calculator"],
        },
    )
    return build_trajectory_record(
        tokenizer=CharacterTokenizer() if tokenizer is None else tokenizer,
        trajectory_id=trajectory_id,
        environment_id=environment_id,
        task_family=task_family,
        initial_context=initial_context,
        steps=actual_steps,
        horizon=len(actual_steps) if horizon is None else horizon,
        reward=actual_reward,
        shifted_reward=actual_shifted,
        epsilon_min=epsilon_min,
        tokenizer_id=tokenizer_id,
        decoding_snapshot_id=decoding_snapshot_id,
        created_at=created_at,
    )


def _revalidate(record: TrajectoryRecord) -> TrajectoryRecord:
    return build_trajectory_record(
        tokenizer=CharacterTokenizer(),
        trajectory_id=record.trajectory_id,
        environment_id=record.environment_id,
        task_family=record.task_family,
        initial_context=record.initial_context,
        steps=record.steps,
        horizon=record.horizon,
        reward=record.reward,
        shifted_reward=record.shifted_reward,
        epsilon_min=record.epsilon_min,
        tokenizer_id=record.tokenizer_id,
        decoding_snapshot_id=record.decoding_snapshot_id,
        created_at=record.created_at,
    )


def test_initial_context_accepts_separate_h0_components() -> None:
    context = _context()

    assert context.query
    assert context.retrieved_skill_ids == ("skill-alpha",)
    assert context.assembled_token_count > 0


def _structured_action_text(
    *,
    kind: str,
    skill_id: str | None,
) -> str:
    return canonical_json(
        {
            "arguments": {},
            "kind": kind,
            "name": "debug-action",
            "resource_id": None if kind == "complete" else "debug.resource",
            "skill_id": skill_id,
        }
    )


def test_trajectory_skill_credit_is_derived_from_action_and_h0_membership() -> None:
    valid_skill = _step(
        action_text=_structured_action_text(kind="skill", skill_id="skill-alpha"),
        invoked_skill_ids=("skill-alpha",),
    )
    record = _record(steps=(valid_skill,))

    assert record.steps[0].invoked_skill_ids == ("skill-alpha",)

    unavailable = _step(
        action_text=_structured_action_text(kind="skill", skill_id="skill-beta"),
        observation_status="schema_invalid",
    )
    assert _record(steps=(unavailable,)).steps[0].invoked_skill_ids == ()

    invalid_credit_steps = (
        replace(valid_skill, invoked_skill_ids=()),
        replace(
            valid_skill,
            action_text=_structured_action_text(kind="tool", skill_id=None),
            invoked_skill_ids=("skill-alpha",),
        ),
        replace(
            valid_skill,
            action_text=_structured_action_text(kind="complete", skill_id=None),
            invoked_skill_ids=("skill-alpha",),
        ),
        replace(
            valid_skill,
            action_text=_structured_action_text(kind="skill", skill_id="skill-beta"),
            observation_status="success",
        ),
        replace(valid_skill, action_text="not-json", invoked_skill_ids=("skill-alpha",)),
    )
    for step in invalid_credit_steps:
        with pytest.raises(ValueError):
            replace(record, steps=(step,))


@pytest.mark.parametrize(
    "wrapper", ["{}", "Action: {}", "```json\n{}\n```", "Action:\n```\n{}\n```"]
)
def test_executed_wrapped_skill_survives_record_admission_without_rewriting_tokens(wrapper):
    from skillev.rollout import ActionParseStatus, StructuredJsonActionCodec

    text = wrapper.format(_structured_action_text(kind="skill", skill_id="skill-alpha"))
    parsed = StructuredJsonActionCodec().parse(text)
    assert parsed.status is ActionParseStatus.VALID
    step = _step(action_text=text, invoked_skill_ids=("skill-alpha",))
    record = _record(steps=(step,))
    assert record.steps[0].action_text == text
    assert record.steps[0].action_token_ids == tuple(map(ord, text))
    assert record.steps[0].invoked_skill_ids == ("skill-alpha",)
    with pytest.raises(ValueError):
        _record(steps=(replace(step, invoked_skill_ids=()),))
    with pytest.raises(ValueError):
        _record(steps=(_step(action_text="prose " + text, invoked_skill_ids=("skill-alpha",)),))

    # A historical codec rejection is not retroactively upgraded to execution.
    rejected = replace(step, observation_status="parse_error", invoked_skill_ids=())
    restored = _record(steps=(rejected,))
    assert restored.steps[0].invoked_skill_ids == ()
    assert restored.steps[0].action_token_ids == step.action_token_ids


@pytest.mark.parametrize(
    ("meta_field", "replacement"),
    [
        ("environment_id", "another-environment"),
        ("task_family", "another-family"),
    ],
)
def test_trajectory_binds_environment_and_task_family_to_initial_context(
    meta_field: str,
    replacement: str,
) -> None:
    record = _record()
    altered_meta = dict(record.initial_context.meta)
    altered_meta[meta_field] = replacement

    with pytest.raises(ValueError):
        replace(record, initial_context=replace(record.initial_context, meta=altered_meta))


@pytest.mark.parametrize(
    "mutation",
    [
        {"query": ""},
        {"query": None},
        {"retrieved_skill_ids": ["skill-alpha"]},
        {"retrieved_skill_ids": ("invalid skill id",)},
        {"retrieved_skill_ids": (1,)},
        {"meta": []},
        {"meta": {"unsupported": object()}},
        {"assembler_version": ""},
        {"assembled_hash": "not-a-hash"},
        {"assembled_hash": 123},
        {"assembled_token_count": 0},
        {"assembled_token_count": True},
    ],
)
def test_initial_context_rejects_each_invalid_component(mutation: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(_context(), **mutation)


def test_trajectory_step_accepts_empty_reasoning_but_complete_action_feedback() -> None:
    step = replace(_step(), reasoning_text="")

    assert step.reasoning_text == ""
    assert step.action_token_count == len(step.action_token_ids)


@pytest.mark.parametrize(
    "mutation",
    [
        {"index": 0},
        {"index": True},
        {"reasoning_text": None},
        {"action_text": ""},
        {"action_text": None},
        {"action_token_ids": (), "action_token_count": 0},
        {"action_token_ids": [97, 99, 116]},
        {"action_token_ids": (-1,), "action_token_count": 1},
        {"action_token_ids": (True,), "action_token_count": 1},
        {"action_token_count": 2},
        {"action_token_count": True},
        {"observation_text": ""},
        {"observation_text": None},
        {"observation_status": "unknown"},
        {"invoked_skill_ids": ["skill-alpha"]},
        {"invoked_skill_ids": ("invalid skill id",)},
        {"invoked_skill_ids": (1,)},
        {"invoked_skill_ids": ("skill-alpha", "skill-alpha")},
        {"forward_prefix_hash": "bad"},
        {"forward_prefix_hash": 123},
        {"hindsight_prefix_hash": "bad"},
    ],
)
def test_trajectory_step_rejects_each_broken_edge_invariant(
    mutation: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        replace(_step(), **mutation)


@pytest.mark.parametrize(
    ("case", "recorded_ids"),
    [
        ("boundary-shifted-by-one", (99, 116, 33)),
        ("one-token-missing", (97, 99)),
        ("one-token-extra", (97, 99, 116, 33)),
    ],
    ids=["offset-by-one", "missing-one", "extra-one"],
)
def test_factory_rejects_each_token_boundary_error(
    case: str,
    recorded_ids: tuple[int, ...],
) -> None:
    step = _step(
        action_token_ids=recorded_ids,
        action_token_count=len(recorded_ids),
    )

    with pytest.raises(ValueError):
        _record(steps=(step,))
    assert case


def test_factory_decode_admits_two_contiguous_sampled_spans_once_each() -> None:
    steps = (_step(), _step(index=2, action_text="done"))
    tokenizer = CountingTokenizer()

    record = _record(tokenizer=tokenizer, steps=steps)

    assert record.horizon == 2
    assert tuple(step.index for step in record.steps) == (1, 2)
    assert tokenizer.identity_reads == 1
    assert tokenizer.decoded_ids == [steps[0].action_token_ids, steps[1].action_token_ids]


def test_factory_accepts_many_to_one_sampled_span_without_retokenizing() -> None:
    tokenizer = ManyToOneTokenizer()
    step = _step(
        action_text=tokenizer.action_text,
        action_token_ids=tokenizer.sampled_ids,
        action_token_count=len(tokenizer.sampled_ids),
    )

    record = _record(tokenizer=tokenizer, steps=(step,), tokenizer_id=tokenizer.tokenizer_id)

    assert record.steps[0].action_token_ids is tokenizer.sampled_ids
    assert tuple(tokenizer.encode(record.steps[0].action_text)) == tokenizer.canonical_ids


def test_factory_rejects_text_or_span_tampering_by_decode_identity() -> None:
    tokenizer = ManyToOneTokenizer()
    sampled = _step(
        action_text=tokenizer.action_text,
        action_token_ids=tokenizer.sampled_ids,
        action_token_count=len(tokenizer.sampled_ids),
    )

    with pytest.raises(ValueError):
        _record(
            tokenizer=tokenizer,
            steps=(replace(sampled, action_text="tampered"),),
            tokenizer_id=tokenizer.tokenizer_id,
        )
    with pytest.raises(ValueError):
        _record(
            tokenizer=tokenizer,
            steps=(replace(sampled, action_token_ids=(43,), action_token_count=1),),
            tokenizer_id=tokenizer.tokenizer_id,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": ()},
        {"steps": (_step(index=2),)},
        {"horizon": 2},
        {"horizon": True},
        {"epsilon_min": 0.0},
        {"epsilon_min": float("inf")},
        {"shifted_reward": 1.5},
        {"shifted_reward": float("nan")},
        {"trajectory_id": "invalid trajectory id"},
        {"trajectory_id": 123},
        {"environment_id": ""},
        {"task_family": ""},
        {"tokenizer_id": ""},
        {"decoding_snapshot_id": ""},
        {"created_at": "2026-07-19"},
        {"created_at": 123},
    ],
)
def test_trajectory_rejects_each_self_contained_invariant(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _record(**kwargs)


def test_factory_rejects_wrong_tokenizer_identity() -> None:
    with pytest.raises(ValueError):
        _record(tokenizer_id="other-tokenizer@1")


def test_factory_rejects_non_text_decode_output() -> None:
    with pytest.raises(ValueError):
        _record(tokenizer=NonTextDecodingTokenizer())


def test_trajectory_rejects_invalid_nested_record_types() -> None:
    record = _record()
    with pytest.raises(ValueError):
        replace(record, initial_context="not-a-context")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(record, steps=[record.steps[0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(record, steps=("not-a-step",))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(record, reward="not-a-reward")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "reward",
    [
        _reward(value=1.0, success=True),
        _reward(value=0.4, success=False),
        _reward(
            value=0.7,
            success=True,
            success_rule=SuccessRule.R_AT_THRESHOLD,
            success_threshold=0.6,
        ),
    ],
)
def test_terminal_reward_accepts_each_explicit_success_rule(
    reward: TerminalReward,
) -> None:
    assert 0.0 <= reward.value <= 1.0
    assert reward.native_payload


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": -0.1, "success": False},
        {"value": 1.1, "success": True},
        {"value": float("nan"), "success": False},
        {"value": 10**400, "success": False},
        {"value": 0.9, "success": True},
        {"value": 1.0, "success": 1},
        {"value": 1.0, "success": True, "success_rule": "r-equals-one"},
        {
            "value": 0.7,
            "success": True,
            "success_rule": SuccessRule.R_AT_THRESHOLD,
            "success_threshold": None,
        },
        {
            "value": 0.7,
            "success": False,
            "success_rule": SuccessRule.R_AT_THRESHOLD,
            "success_threshold": 0.6,
        },
        {
            "value": 0.7,
            "success": True,
            "success_rule": SuccessRule.R_AT_THRESHOLD,
            "success_threshold": 0.0,
        },
        {
            "value": 0.7,
            "success": True,
            "success_rule": SuccessRule.R_AT_THRESHOLD,
            "success_threshold": float("inf"),
        },
        {"value": 1.0, "success": True, "success_threshold": 0.5},
    ],
)
def test_terminal_reward_rejects_each_rule_or_range_mismatch(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _reward(**kwargs)


def test_terminal_reward_rejects_empty_identity_and_noncanonical_payload() -> None:
    good = _reward()
    with pytest.raises(ValueError):
        replace(good, environment_id="")
    with pytest.raises(ValueError):
        replace(good, native_metric_name="")
    with pytest.raises(ValueError):
        replace(good, verifier_version="")
    with pytest.raises(ValueError):
        replace(good, native_payload={"unsupported": object()})
    with pytest.raises(ValueError):
        replace(good, native_payload=[])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("native_metric_name", 1),
        ("environment_id", 1),
        ("verifier_version", 1),
        ("success_rule", 1),
        ("value", 10**400),
    ],
)
def test_terminal_reward_from_value_rejects_malformed_wire_types(
    field: str,
    bad_value: object,
) -> None:
    payload = _reward().to_value()
    payload[field] = bad_value  # type: ignore[assignment]

    with pytest.raises(ValueError):
        TerminalReward.from_value(payload)


def test_every_trajectory_record_family_round_trips_through_canonical_json() -> None:
    record = _record(steps=(_step(), _step(index=2, action_text="done")))
    context_payload = parse_canonical_json(canonical_json(record.initial_context.to_value()))
    step_payload = parse_canonical_json(canonical_json(record.steps[0].to_value()))
    reward_payload = parse_canonical_json(canonical_json(record.reward.to_value()))
    trajectory_payload = parse_canonical_json(canonical_json(record.to_value()))

    rebuilt_context = InitialContext.from_value(context_payload)
    rebuilt_step = TrajectoryStep.from_value(step_payload)
    rebuilt_reward = TerminalReward.from_value(reward_payload)
    rebuilt_record = TrajectoryRecord.from_value(trajectory_payload)

    assert rebuilt_context == record.initial_context
    assert rebuilt_context.content_hash == record.initial_context.content_hash
    assert rebuilt_step == record.steps[0]
    assert rebuilt_step.content_hash == record.steps[0].content_hash
    assert rebuilt_reward == record.reward
    assert rebuilt_reward.content_hash == record.reward.content_hash
    assert rebuilt_record == record
    assert rebuilt_record.content_hash == record.content_hash
    assert _revalidate(rebuilt_record) == record


def test_created_at_is_serialized_but_excluded_from_trajectory_content_hash() -> None:
    record = _record()
    later = replace(record, created_at="2026-07-19T01:00:00+00:00")

    assert record.to_value() != later.to_value()
    assert record.content_hash == later.content_hash


def test_scientific_content_changes_trajectory_hash() -> None:
    record = _record()
    changed = _record(task_family="another-family")

    assert record.content_hash != changed.content_hash
