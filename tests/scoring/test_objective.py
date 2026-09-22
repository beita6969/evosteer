from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from dataclasses import FrozenInstanceError, dataclass, replace
from functools import reduce
from typing import Any

import pytest
import torch
from torch import nn

from skillev.contracts import (
    EdgeLogprobRecord,
    TrajectoryResidual,
    stable_hash,
)
from skillev.policy import AdapterRole, PublicTokenizerIdentity, QwenPolicyBackbone
from skillev.scoring import (
    ScoringConfig,
    ScoringDirection,
    backward_trajectory_delta_streaming,
    edge_logprob_mean,
    materialize_edge_records,
    materialize_residual,
    render_forward_prefix,
    render_hindsight_prefix,
    score_trajectory,
)


@dataclass(frozen=True, slots=True)
class _ScoreCall:
    prefix_ids: tuple[int, ...]
    action_ids: tuple[int, ...]
    role: AdapterRole


class _RecordingTokenizer:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.encoded_texts: list[str] = []
        self.rollout_encoded_texts: list[str] = []

    @property
    def tokenizer_id(self) -> str:
        return self._delegate.tokenizer_id

    @property
    def public_identity(self) -> PublicTokenizerIdentity:
        return self._delegate.public_identity

    def encode(self, text: str) -> list[int]:
        self.encoded_texts.append(text)
        return self._delegate.encode(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return self._delegate.decode(token_ids)

    def encode_rollout_prompt(self, text: str) -> list[int]:
        self.rollout_encoded_texts.append(text)
        return self._delegate.encode_rollout_prompt(text)

    def encode_authoring_prompt(self, text: str) -> list[int]:
        return self._delegate.encode_authoring_prompt(text)


class _RecordingBackbone:
    def __init__(self, delegate: QwenPolicyBackbone) -> None:
        self._delegate = delegate
        self._tokenizer = _RecordingTokenizer(delegate.tokenizer)
        self.score_calls: list[_ScoreCall] = []
        self.tokenizer_reads = 0

    @property
    def tokenizer(self) -> _RecordingTokenizer:
        self.tokenizer_reads += 1
        return self._tokenizer

    def score(
        self,
        prefix_ids: tuple[int, ...],
        action_ids: tuple[int, ...],
        role: AdapterRole,
    ) -> torch.Tensor:
        self.score_calls.append(_ScoreCall(prefix_ids, action_ids, role))
        return self._delegate.score(prefix_ids, action_ids, role)

    def z_value(self, query_ids: tuple[int, ...]) -> torch.Tensor:
        return self._delegate.z_value(query_ids)


class _WrongLengthBackbone(_RecordingBackbone):
    def score(
        self,
        prefix_ids: tuple[int, ...],
        action_ids: tuple[int, ...],
        role: AdapterRole,
    ) -> torch.Tensor:
        return super().score(prefix_ids, action_ids, role)[:-1]


def _parameters(
    backbone: QwenPolicyBackbone,
    component: AdapterRole | str,
) -> tuple[nn.Parameter, ...]:
    groups = backbone.parameter_groups()
    if component is AdapterRole.FORWARD_POLICY:
        return groups.forward
    if component is AdapterRole.BACKWARD_POLICY:
        return groups.backward
    if component == "z-head":
        return groups.z_head
    raise ValueError("unknown test parameter group")


def _base_parameters(backbone: QwenPolicyBackbone) -> tuple[nn.Parameter, ...]:
    trainable_parameter_ids = {
        id(parameter)
        for component in (*tuple(AdapterRole), "z-head")
        for parameter in _parameters(backbone, component)
    }
    return tuple(
        parameter
        for parameter in backbone._model.parameters()
        if id(parameter) not in trainable_parameter_ids
    )


def _clear_gradients(backbone: QwenPolicyBackbone) -> None:
    for parameter in (
        *tuple(backbone._model.parameters()),
        *_parameters(backbone, "z-head"),
    ):
        parameter.grad = None


def _trainable_parameters(backbone: QwenPolicyBackbone) -> tuple[nn.Parameter, ...]:
    groups = backbone.parameter_groups()
    return (*groups.forward, *groups.backward, *groups.z_head)


def _gradient_snapshot(backbone: QwenPolicyBackbone) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter.grad.detach().clone()
        for parameter in _trainable_parameters(backbone)
        if parameter.grad is not None
    )


def _replace_adapter(
    backbone: QwenPolicyBackbone,
    role: AdapterRole,
    *,
    scale: float,
) -> None:
    with torch.no_grad():
        for parameter_index, parameter in enumerate(_parameters(backbone, role), start=1):
            ramp = torch.arange(
                1,
                parameter.numel() + 1,
                device=parameter.device,
                dtype=parameter.dtype,
            ).reshape_as(parameter)
            parameter.copy_(
                ramp * (scale / parameter.numel()) + scale * parameter_index,
            )


def _adapter_snapshot(
    backbone: QwenPolicyBackbone,
    role: AdapterRole,
) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in _parameters(backbone, role))


def _restore_adapter(
    backbone: QwenPolicyBackbone,
    role: AdapterRole,
    snapshot: tuple[torch.Tensor, ...],
) -> None:
    with torch.no_grad():
        for parameter, saved in zip(
            _parameters(backbone, role),
            snapshot,
            strict=True,
        ):
            parameter.copy_(saved)


def _assert_all_gradients_none(parameters: Iterable[nn.Parameter]) -> None:
    assert all(parameter.grad is None for parameter in parameters)


def _manual_edge_mean(
    backbone: QwenPolicyBackbone,
    case: Any,
    step_index: int,
    role: AdapterRole,
) -> torch.Tensor:
    if role is AdapterRole.FORWARD_POLICY:
        rendered = render_forward_prefix(
            case.initial_text,
            case.record.steps,
            step_index,
        )
    else:
        rendered = render_hindsight_prefix(
            case.initial_text,
            case.record.steps,
            step_index,
        )
    step = case.record.steps[step_index - 1]
    prefix_ids = tuple(backbone.tokenizer.encode_rollout_prompt(rendered.text))
    raw = backbone.score(prefix_ids, step.action_token_ids, role)
    return raw.sum() / step.action_token_count


def test_scoring_config_normalizes_positive_beta_and_is_immutable() -> None:
    config = ScoringConfig(temperature_beta=2)

    assert config.temperature_beta == 2.0
    assert type(config.temperature_beta) is float
    with pytest.raises(FrozenInstanceError):
        config.temperature_beta = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "temperature_beta",
    [0.0, -1.0, math.inf, -math.inf, math.nan, True, "1.0"],
)
def test_scoring_config_rejects_non_positive_or_non_finite_beta(
    temperature_beta: object,
) -> None:
    with pytest.raises(ValueError):
        ScoringConfig(temperature_beta=temperature_beta)  # type: ignore[arg-type]


def test_score_trajectory_matches_independent_manual_edges_delta_and_loss_exactly(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    beta = 1.75
    query_ids = tuple(scoring_backbone.tokenizer.encode(scoring_case.record.initial_context.query))
    manual_z = scoring_backbone.z_value(query_ids)
    manual_forward = tuple(
        _manual_edge_mean(
            scoring_backbone,
            scoring_case,
            step_index,
            AdapterRole.FORWARD_POLICY,
        )
        for step_index in range(1, scoring_case.record.horizon + 1)
    )
    manual_backward = tuple(
        _manual_edge_mean(
            scoring_backbone,
            scoring_case,
            step_index,
            AdapterRole.BACKWARD_POLICY,
        )
        for step_index in range(1, scoring_case.record.horizon + 1)
    )
    manual_delta = (
        manual_z
        + reduce(operator.add, manual_forward)
        - beta * math.log(scoring_case.record.shifted_reward)
        - reduce(operator.add, manual_backward)
    )
    manual_loss = (manual_delta / scoring_case.record.horizon) ** 2

    actual = score_trajectory(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        ScoringConfig(temperature_beta=beta),
    )

    torch.testing.assert_close(actual.delta, manual_delta, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.loss, manual_loss, rtol=0.0, atol=0.0)
    assert actual.log_z == float(manual_z.detach().item())
    assert actual.forward_per_token_means == tuple(
        float(value.detach().item()) for value in manual_forward
    )
    assert actual.backward_per_token_means == tuple(
        float(value.detach().item()) for value in manual_backward
    )
    assert actual.log_shifted_reward == math.log(scoring_case.record.shifted_reward)


@pytest.mark.parametrize("horizon", [1, 2, 15])
@pytest.mark.parametrize("reward_value", [0.0, 0.7])
def test_streaming_delta_backward_matches_monolithic_loss_values_and_all_gradients(
    scoring_backbone: QwenPolicyBackbone,
    make_scoring_case: Any,
    horizon: int,
    reward_value: float,
) -> None:
    reasoning = tuple("reasonone thoughtone" for _ in range(horizon))
    actions = tuple("alpha beta" if index % 2 else "delta epsilon zeta" for index in range(horizon))
    observations = tuple("observeone resultone" for _ in range(horizon))
    case = make_scoring_case(
        reasoning_texts=reasoning,
        action_texts=actions,
        observation_texts=observations,
        reward_value=reward_value,
    )
    config = ScoringConfig(temperature_beta=1.75)

    _clear_gradients(scoring_backbone)
    monolithic = score_trajectory(
        scoring_backbone,
        case.record,
        case.initial_text,
        config,
    )
    monolithic.loss.backward()
    expected_gradients = _gradient_snapshot(scoring_backbone)
    expected_residual = materialize_residual(monolithic, raw_reward=reward_value)
    expected_edges = materialize_edge_records(
        monolithic,
        forward_adapter_version="forward@test",
        backward_adapter_version="backward@test",
    )

    _clear_gradients(scoring_backbone)
    streaming = backward_trajectory_delta_streaming(
        scoring_backbone,
        case.record,
        case.initial_text,
        config,
    )
    for parameter in _trainable_parameters(scoring_backbone):
        if parameter.grad is not None:
            parameter.grad.mul_(streaming.gradient_coefficient)
    actual_gradients = _gradient_snapshot(scoring_backbone)

    assert streaming.loss == pytest.approx(float(monolithic.loss.detach()), rel=2e-5, abs=2e-5)
    assert streaming.delta == pytest.approx(float(monolithic.delta.detach()), rel=2e-5, abs=2e-5)
    assert streaming.log_z == monolithic.log_z
    assert streaming.forward_per_token_means == monolithic.forward_per_token_means
    assert streaming.backward_per_token_means == monolithic.backward_per_token_means
    assert materialize_residual(streaming, raw_reward=reward_value) == expected_residual
    assert (
        materialize_edge_records(
            streaming,
            forward_adapter_version="forward@test",
            backward_adapter_version="backward@test",
        )
        == expected_edges
    )
    assert len(actual_gradients) == len(expected_gradients)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_log_z_is_query_only_and_invariant_to_retrieved_skill_text(
    scoring_backbone: QwenPolicyBackbone,
    make_scoring_case: Any,
) -> None:
    first = make_scoring_case(initial_text="query\nretrieved skill alpha\n")
    second = make_scoring_case(initial_text="query\nretrieved skill radically different beta\n")

    first_score = score_trajectory(
        scoring_backbone,
        first.record,
        first.initial_text,
        ScoringConfig(temperature_beta=1.0),
    )
    second_score = score_trajectory(
        scoring_backbone,
        second.record,
        second.initial_text,
        ScoringConfig(temperature_beta=1.0),
    )

    assert first_score.log_z == second_score.log_z


def test_hindsight_score_excludes_current_reasoning_and_known_wrong_prefix_differs(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    step_index = 2
    step = scoring_case.record.steps[step_index - 1]
    correct = edge_logprob_mean(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        step_index,
        ScoringDirection.HINDSIGHT,
    )

    # The known semantic error scores phi under the forward r_t condition.
    wrong_prefix = render_forward_prefix(
        scoring_case.initial_text,
        scoring_case.record.steps,
        step_index,
    )
    wrong_raw = scoring_backbone.score(
        tuple(scoring_backbone.tokenizer.encode_rollout_prompt(wrong_prefix.text)),
        step.action_token_ids,
        AdapterRole.BACKWARD_POLICY,
    )
    wrong = wrong_raw.sum() / step.action_token_count

    assert not torch.equal(correct, wrong)
    full_score = score_trajectory(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        ScoringConfig(),
    )
    assert full_score.backward_per_token_means[step_index - 1] == float(correct.detach().item())
    assert full_score.backward_per_token_means[step_index - 1] != float(wrong.detach().item())


def test_edges_use_recorded_action_ids_same_k_and_fixed_direction_adapter_binding(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    recording = _RecordingBackbone(scoring_backbone)
    step = scoring_case.record.steps[0]
    forward = edge_logprob_mean(
        recording,
        scoring_case.record,
        scoring_case.initial_text,
        1,
        ScoringDirection.FORWARD,
    )
    hindsight = edge_logprob_mean(
        recording,
        scoring_case.record,
        scoring_case.initial_text,
        1,
        ScoringDirection.HINDSIGHT,
    )

    assert [call.role for call in recording.score_calls] == [
        AdapterRole.FORWARD_POLICY,
        AdapterRole.BACKWARD_POLICY,
    ]
    assert all(call.action_ids == step.action_token_ids for call in recording.score_calls)
    assert all(len(call.action_ids) == step.action_token_count for call in recording.score_calls)
    assert recording.tokenizer.encoded_texts == []
    assert recording.tokenizer.rollout_encoded_texts == [
        render_forward_prefix(scoring_case.initial_text, scoring_case.record.steps, 1).text,
        render_hindsight_prefix(scoring_case.initial_text, scoring_case.record.steps, 1).text,
    ]
    for actual, call in zip(
        (forward, hindsight),
        recording.score_calls,
        strict=True,
    ):
        per_token = scoring_backbone.score(call.prefix_ids, call.action_ids, call.role)
        expected = per_token.sum() / step.action_token_count
        known_wrong = per_token.sum() / (len(call.prefix_ids) + step.action_token_count)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(actual, known_wrong, rtol=0.0, atol=0.0)


def test_forward_and_backward_scoring_never_reencode_recorded_action_text(
    scoring_case: Any,
) -> None:
    sampled_ids = (7, 8)
    original_step = scoring_case.record.steps[0]
    sampled_step = replace(
        original_step,
        action_token_ids=sampled_ids,
        action_token_count=len(sampled_ids),
    )
    record = replace(
        scoring_case.record,
        steps=(sampled_step, *scoring_case.record.steps[1:]),
    )

    class ManyToOneScoringTokenizer:
        tokenizer_id = "many-to-one-scoring@1"

        def encode(self, text: str) -> list[int]:
            if text == sampled_step.action_text:
                return [99]
            return [ord(character) for character in text]

        def decode(self, token_ids: tuple[int, ...]) -> str:
            if token_ids in {sampled_ids, (99,)}:
                return sampled_step.action_text
            return "".join(chr(token_id) for token_id in token_ids)

    class RecordingBackbone:
        def __init__(self) -> None:
            self.tokenizer = ManyToOneScoringTokenizer()
            self.calls: list[_ScoreCall] = []

        def score(
            self,
            prefix_ids: tuple[int, ...],
            action_ids: tuple[int, ...],
            role: AdapterRole,
        ) -> torch.Tensor:
            self.calls.append(_ScoreCall(prefix_ids, action_ids, role))
            return torch.zeros(len(action_ids), dtype=torch.float32)

        def z_value(self, query_ids: tuple[int, ...]) -> torch.Tensor:
            del query_ids
            return torch.tensor(0.0)

    backbone = RecordingBackbone()

    edge_logprob_mean(backbone, record, scoring_case.initial_text, 1, ScoringDirection.FORWARD)
    edge_logprob_mean(backbone, record, scoring_case.initial_text, 1, ScoringDirection.HINDSIGHT)

    assert [call.action_ids for call in backbone.calls] == [sampled_ids, sampled_ids]
    assert tuple(backbone.tokenizer.encode(sampled_step.action_text)) == (99,)


def test_score_trajectory_reads_the_backbone_tokenizer_property_once(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    recording = _RecordingBackbone(scoring_backbone)

    score_trajectory(
        recording,
        scoring_case.record,
        scoring_case.initial_text,
        ScoringConfig(),
    )

    assert recording.tokenizer_reads == 1


def test_edge_rejects_a_backbone_score_with_wrong_action_token_count(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    wrong_length = _WrongLengthBackbone(scoring_backbone)

    with pytest.raises(ValueError) as captured:
        edge_logprob_mean(
            wrong_length,
            scoring_case.record,
            scoring_case.initial_text,
            1,
            ScoringDirection.FORWARD,
        )

    location = str(captured.value).lower()
    assert scoring_case.record.trajectory_id in location
    assert "step 1" in location
    assert "forward" in location


def test_adapter_perturbations_change_only_the_bound_direction(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    forward_snapshot = _adapter_snapshot(
        scoring_backbone,
        AdapterRole.FORWARD_POLICY,
    )
    baseline_forward = edge_logprob_mean(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        1,
        ScoringDirection.FORWARD,
    ).detach()
    baseline_backward = edge_logprob_mean(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        1,
        ScoringDirection.HINDSIGHT,
    ).detach()

    _replace_adapter(
        scoring_backbone,
        AdapterRole.FORWARD_POLICY,
        scale=0.03125,
    )
    changed_forward = edge_logprob_mean(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        1,
        ScoringDirection.FORWARD,
    ).detach()
    unchanged_backward = edge_logprob_mean(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        1,
        ScoringDirection.HINDSIGHT,
    ).detach()

    assert not torch.equal(changed_forward, baseline_forward)
    assert torch.equal(unchanged_backward, baseline_backward)

    _restore_adapter(
        scoring_backbone,
        AdapterRole.FORWARD_POLICY,
        forward_snapshot,
    )
    restored_forward = edge_logprob_mean(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        1,
        ScoringDirection.FORWARD,
    ).detach()
    _replace_adapter(
        scoring_backbone,
        AdapterRole.BACKWARD_POLICY,
        scale=-0.015625,
    )
    unchanged_forward = edge_logprob_mean(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        1,
        ScoringDirection.FORWARD,
    ).detach()
    changed_backward = edge_logprob_mean(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        1,
        ScoringDirection.HINDSIGHT,
    ).detach()

    assert torch.equal(restored_forward, baseline_forward)
    assert torch.equal(unchanged_forward, baseline_forward)
    assert not torch.equal(changed_backward, baseline_backward)


def test_loss_backward_reaches_both_adapters_and_z_but_not_frozen_base(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    _clear_gradients(scoring_backbone)
    forward_parameters = _parameters(
        scoring_backbone,
        AdapterRole.FORWARD_POLICY,
    )
    backward_parameters = _parameters(
        scoring_backbone,
        AdapterRole.BACKWARD_POLICY,
    )
    z_parameters = _parameters(scoring_backbone, "z-head")
    base_parameters = _base_parameters(scoring_backbone)

    score = score_trajectory(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        ScoringConfig(),
    )
    score.loss.backward()

    assert forward_parameters
    assert backward_parameters
    assert z_parameters
    assert all(parameter.grad is not None for parameter in forward_parameters)
    assert all(parameter.grad is not None for parameter in backward_parameters)
    assert all(parameter.grad is not None for parameter in z_parameters)
    _assert_all_gradients_none(base_parameters)


def test_score_rejects_tokenizer_assembled_hash_and_assembled_count_mismatches(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    tokenizer_mismatch = replace(
        scoring_case.record,
        tokenizer_id="different-tokenizer@1",
    )
    with pytest.raises(ValueError) as tokenizer_error:
        score_trajectory(
            scoring_backbone,
            tokenizer_mismatch,
            scoring_case.initial_text,
            ScoringConfig(),
        )
    assert scoring_case.record.trajectory_id in str(tokenizer_error.value)
    assert "tokenizer" in str(tokenizer_error.value).lower()

    with pytest.raises(ValueError) as hash_error:
        score_trajectory(
            scoring_backbone,
            scoring_case.record,
            scoring_case.initial_text + "changed",
            ScoringConfig(),
        )
    assert scoring_case.record.trajectory_id in str(hash_error.value)
    assert "assembled" in str(hash_error.value).lower()

    count_mismatch = replace(
        scoring_case.record,
        initial_context=replace(
            scoring_case.record.initial_context,
            assembled_token_count=(scoring_case.record.initial_context.assembled_token_count + 1),
        ),
    )
    with pytest.raises(ValueError) as count_error:
        score_trajectory(
            scoring_backbone,
            count_mismatch,
            scoring_case.initial_text,
            ScoringConfig(),
        )
    assert scoring_case.record.trajectory_id in str(count_error.value)
    assert "token" in str(count_error.value).lower()
    assert "count" in str(count_error.value).lower()


@pytest.mark.parametrize(
    ("mutated_field", "direction"),
    [
        ("reasoning_text", ScoringDirection.FORWARD),
        ("observation_text", ScoringDirection.HINDSIGHT),
    ],
)
def test_changed_current_condition_is_rejected_at_the_exact_edge(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
    mutated_field: str,
    direction: ScoringDirection,
) -> None:
    step_index = 2
    step = scoring_case.record.steps[step_index - 1]
    changed_step = replace(step, **{mutated_field: f"changed {mutated_field}"})
    changed_record = replace(
        scoring_case.record,
        steps=(*scoring_case.record.steps[: step_index - 1], changed_step),
    )

    with pytest.raises(ValueError) as captured:
        edge_logprob_mean(
            scoring_backbone,
            changed_record,
            scoring_case.initial_text,
            step_index,
            direction,
        )

    location = str(captured.value).lower()
    assert scoring_case.record.trajectory_id in location
    assert f"step {step_index}" in location
    assert direction.value in location


def test_legacy_template_prefix_hash_is_rejected_with_edge_location(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    rendered = render_hindsight_prefix(
        scoring_case.initial_text,
        scoring_case.record.steps,
        1,
    )
    legacy_hash = stable_hash(
        {
            "kind": "hindsight",
            "template_version": "ttb-render@0",
            "text": rendered.text,
        }
    )
    changed_step = replace(
        scoring_case.record.steps[0],
        hindsight_prefix_hash=legacy_hash,
    )
    legacy_record = replace(
        scoring_case.record,
        steps=(changed_step, *scoring_case.record.steps[1:]),
    )

    with pytest.raises(ValueError) as captured:
        edge_logprob_mean(
            scoring_backbone,
            legacy_record,
            scoring_case.initial_text,
            1,
            ScoringDirection.HINDSIGHT,
        )

    location = str(captured.value).lower()
    assert scoring_case.record.trajectory_id in location
    assert "step 1" in location
    assert "hindsight" in location


def test_scoring_is_deterministic_preserves_record_and_does_not_advance_global_rng(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    record_before = scoring_case.record.to_value()
    content_hash_before = scoring_case.record.content_hash
    rng_before = torch.random.get_rng_state().clone()

    first = score_trajectory(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        ScoringConfig(),
    )
    rng_after_first = torch.random.get_rng_state().clone()
    second = score_trajectory(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        ScoringConfig(),
    )
    rng_after_second = torch.random.get_rng_state().clone()

    assert torch.equal(first.delta, second.delta)
    assert torch.equal(first.loss, second.loss)
    assert first.log_z == second.log_z
    assert first.forward_per_token_means == second.forward_per_token_means
    assert first.backward_per_token_means == second.backward_per_token_means
    assert first.log_shifted_reward == second.log_shifted_reward
    assert scoring_case.record.to_value() == record_before
    assert scoring_case.record.content_hash == content_hash_before
    assert torch.equal(rng_before, rng_after_first)
    assert torch.equal(rng_before, rng_after_second)


def test_materialized_edges_and_residual_pass_contracts_with_one_reward_shift(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    score = score_trajectory(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        ScoringConfig(temperature_beta=1.25),
    )
    forward_version = scoring_backbone.adapter_version(AdapterRole.FORWARD_POLICY)
    backward_version = scoring_backbone.adapter_version(AdapterRole.BACKWARD_POLICY)
    edges = materialize_edge_records(
        score,
        forward_adapter_version=forward_version,
        backward_adapter_version=backward_version,
    )
    residual = materialize_residual(
        score,
        raw_reward=scoring_case.record.reward.value,
    )

    assert len(edges) == scoring_case.record.horizon
    assert all(isinstance(edge, EdgeLogprobRecord) for edge in edges)
    for index, edge in enumerate(edges):
        assert edge.trajectory_id == scoring_case.record.trajectory_id
        assert edge.step_index == index + 1
        assert edge.forward_logprob_per_token == score.forward_per_token_means[index]
        assert edge.backward_logprob_per_token == score.backward_per_token_means[index]
        assert edge.step_importance == (
            edge.forward_logprob_per_token - edge.backward_logprob_per_token
        )
        assert edge.forward_adapter_version == forward_version
        assert edge.backward_adapter_version == backward_version
        assert edge.scoring_stack_id == "training-stack"

    assert isinstance(residual, TrajectoryResidual)
    assert residual.trajectory_id == scoring_case.record.trajectory_id
    assert residual.sum_forward == math.fsum(score.forward_per_token_means)
    assert residual.sum_backward == math.fsum(score.backward_per_token_means)
    assert residual.raw_reward == scoring_case.record.reward.value
    assert residual.log_shifted_reward == math.log(scoring_case.record.shifted_reward)
    assert residual.log_shifted_reward != math.log(
        scoring_case.record.shifted_reward + scoring_case.record.epsilon_min
    )
    expected_delta = (
        score.log_z
        + residual.sum_forward
        - score.temperature_beta * score.log_shifted_reward
        - residual.sum_backward
    )
    assert residual.delta == expected_delta


def test_loss_normalizes_by_horizon_not_total_action_tokens(
    scoring_backbone: QwenPolicyBackbone,
    scoring_case: Any,
) -> None:
    score = score_trajectory(
        scoring_backbone,
        scoring_case.record,
        scoring_case.initial_text,
        ScoringConfig(),
    )
    total_action_tokens = sum(step.action_token_count for step in scoring_case.record.steps)

    assert scoring_case.record.horizon == 2
    assert total_action_tokens == 6
    correct = (score.delta / scoring_case.record.horizon) ** 2
    known_wrong = (score.delta / total_action_tokens) ** 2
    torch.testing.assert_close(score.loss, correct, rtol=0.0, atol=0.0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(score.loss, known_wrong, rtol=0.0, atol=0.0)


def test_empty_reasoning_step_scores_without_collapsing_structure(
    scoring_backbone: QwenPolicyBackbone,
    make_scoring_case: Any,
) -> None:
    case = make_scoring_case(reasoning_texts=("", "reasontwo thoughttwo"))

    score = score_trajectory(
        scoring_backbone,
        case.record,
        case.initial_text,
        ScoringConfig(),
    )

    assert score.horizon == 2
    assert len(score.forward_per_token_means) == 2
    assert len(score.backward_per_token_means) == 2
    assert torch.isfinite(score.delta)
    assert torch.isfinite(score.loss)
