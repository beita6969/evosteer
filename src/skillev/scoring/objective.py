"""Pure composition of policy scores into trajectory-balance objectives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from operator import add
from typing import TYPE_CHECKING

from skillev.contracts import (
    EdgeLogprobRecord,
    TokenizerProtocol,
    TrajectoryRecord,
    TrajectoryResidual,
)
from skillev.policy.interface import (
    AdapterRole,
    ModelInputWindow,
    PolicyScoringMemoryError,
    encode_policy_prompt,
    encode_rollout_prompt,
)

from .edge_plan import PreparedEdgePlan
from .rendering import (
    assembled_context_hash,
    render_forward_prefix,
    render_hindsight_prefix,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from skillev.policy.interface import PolicyBackbone


class ScoringDirection(StrEnum):
    """Closed mapping from trajectory condition to policy adapter."""

    FORWARD = "forward"
    HINDSIGHT = "hindsight"


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Method-level constants for one trajectory score."""

    temperature_beta: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.temperature_beta, bool) or not isinstance(
            self.temperature_beta,
            int | float,
        ):
            raise ValueError("temperature_beta must be a finite positive number")
        normalized = float(self.temperature_beta)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError("temperature_beta must be a finite positive number")
        object.__setattr__(self, "temperature_beta", normalized)


@dataclass(frozen=True, slots=True)
class TrajectoryScore:
    """Differentiable TTB tensors plus detached per-trajectory diagnostics."""

    trajectory_id: str
    horizon: int
    loss: Tensor
    delta: Tensor
    log_z: float
    forward_per_token_means: tuple[float, ...]
    backward_per_token_means: tuple[float, ...]
    log_shifted_reward: float
    temperature_beta: float

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty text")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if self.loss.ndim != 0 or self.delta.ndim != 0:
            raise ValueError("loss and delta must be scalar tensors")
        if not self.loss.requires_grad:
            raise ValueError("loss must retain its gradient graph")
        if not self.delta.requires_grad:
            raise ValueError("delta must retain its gradient graph")
        if (
            not isinstance(self.forward_per_token_means, tuple)
            or len(self.forward_per_token_means) != self.horizon
        ):
            raise ValueError("forward_per_token_means must match horizon")
        if (
            not isinstance(self.backward_per_token_means, tuple)
            or len(self.backward_per_token_means) != self.horizon
        ):
            raise ValueError("backward_per_token_means must match horizon")
        for field_name, value in (
            ("log_z", self.log_z),
            ("log_shifted_reward", self.log_shifted_reward),
            ("temperature_beta", self.temperature_beta),
            *(
                (f"forward_per_token_means[{index}]", value)
                for index, value in enumerate(self.forward_per_token_means)
            ),
            *(
                (f"backward_per_token_means[{index}]", value)
                for index, value in enumerate(self.backward_per_token_means)
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{field_name} must be finite")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.temperature_beta <= 0.0:
            raise ValueError("temperature_beta must be positive")


@dataclass(frozen=True, slots=True)
class StreamingTrajectoryScore:
    """Detached TTB values after exact term-at-a-time ``∇Delta`` accumulation."""

    trajectory_id: str
    horizon: int
    loss: float
    delta: float
    gradient_coefficient: float
    log_z: float
    forward_per_token_means: tuple[float, ...]
    backward_per_token_means: tuple[float, ...]
    log_shifted_reward: float
    temperature_beta: float

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty text")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if len(self.forward_per_token_means) != self.horizon:
            raise ValueError("forward_per_token_means must match horizon")
        if len(self.backward_per_token_means) != self.horizon:
            raise ValueError("backward_per_token_means must match horizon")
        values = (
            self.loss,
            self.delta,
            self.gradient_coefficient,
            self.log_z,
            self.log_shifted_reward,
            self.temperature_beta,
            *self.forward_per_token_means,
            *self.backward_per_token_means,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("streaming trajectory score values must be finite numbers")
        if self.temperature_beta <= 0.0:
            raise ValueError("temperature_beta must be positive")


MaterializedTrajectoryScore = TrajectoryScore | StreamingTrajectoryScore


class TrajectoryScoringMemoryError(RuntimeError):
    """Terminal memory exhaustion with the exact trajectory edge coordinate."""

    def __init__(
        self,
        *,
        trajectory_id: str,
        step_index: int,
        direction: ScoringDirection,
        prefix_token_count: int,
        action_token_count: int,
    ) -> None:
        super().__init__("trajectory teacher-forced scoring exhausted device memory")
        self.trajectory_id = trajectory_id
        self.step_index = step_index
        self.direction = direction
        self.prefix_token_count = prefix_token_count
        self.action_token_count = action_token_count
        self.microbatch_steps: tuple[int, ...] = (step_index,)


def _prefix_ids(
    tokenizer: TokenizerProtocol,
    text: str,
    *,
    trajectory_id: str,
    step_index: int,
    direction: ScoringDirection,
    initial_text: str,
    input_window: ModelInputWindow | None,
) -> tuple[int, ...]:
    try:
        encoded = encode_policy_prompt(
            tokenizer, text, initial_text=initial_text, window=input_window
        ).ids
    except ValueError as error:
        raise ValueError(
            f"trajectory {trajectory_id!r}, step {step_index}, {direction.value}: "
            "rendered prefix encoded to no token ids"
        ) from error
    return tuple(encoded)


def _edge_logprob_mean(
    backbone: PolicyBackbone,
    tokenizer: TokenizerProtocol,
    record: TrajectoryRecord,
    initial_text: str,
    step_index: int,
    direction: ScoringDirection,
    prepared: PreparedEdgePlan | None = None,
) -> Tensor:
    if type(step_index) is not int or not 1 <= step_index <= record.horizon:
        raise ValueError(
            f"trajectory {record.trajectory_id!r}: step_index must be in [1, {record.horizon}]"
        )
    step = record.steps[step_index - 1]

    if prepared is None:
        if direction is ScoringDirection.FORWARD:
            rendered = render_forward_prefix(initial_text, record.steps, step_index)
            expected_hash = step.forward_prefix_hash
            role = AdapterRole.FORWARD_POLICY
        elif direction is ScoringDirection.HINDSIGHT:
            rendered = render_hindsight_prefix(initial_text, record.steps, step_index)
            expected_hash = step.hindsight_prefix_hash
            role = AdapterRole.BACKWARD_POLICY
        else:
            raise ValueError("unsupported scoring direction")

        if rendered.prefix_hash != expected_hash:
            raise ValueError(
                f"trajectory {record.trajectory_id!r}, step {step_index}, "
                f"{direction.value} prefix hash mismatch"
            )
        prefix_ids = _prefix_ids(
            tokenizer,
            rendered.text,
            trajectory_id=record.trajectory_id,
            step_index=step_index,
            direction=direction,
            initial_text=initial_text,
            input_window=ModelInputWindow.from_meta(record.initial_context.meta),
        )
        # The action is the exact generated token sequence admitted by Layer 1.
        # Re-encoding action_text here could silently change a BPE boundary.
        action_ids = step.action_token_ids
    else:
        if prepared.record is not record or prepared.tokenizer_id != tokenizer.tokenizer_id:
            raise ValueError("prepared edges belong to another artifact or tokenizer")
        if prepared.initial_text != initial_text:
            raise ValueError("prepared edges use another initial context")
        role = (
            AdapterRole.FORWARD_POLICY
            if direction is ScoringDirection.FORWARD
            else AdapterRole.BACKWARD_POLICY
        )
        edge = prepared.edge(step_index, role)
        if (
            edge.step_index != step_index
            or edge.role is not role
            or edge.action_ids != step.action_token_ids
        ):
            raise ValueError(
                f"trajectory {record.trajectory_id!r}, step {step_index}: "
                "prepared action span differs"
            )
        prefix_ids, action_ids = edge.prefix_ids, edge.action_ids
    try:
        per_token = backbone.score(prefix_ids, action_ids, role)
    except PolicyScoringMemoryError as error:
        raise TrajectoryScoringMemoryError(
            trajectory_id=record.trajectory_id,
            step_index=step_index,
            direction=direction,
            prefix_token_count=error.prefix_token_count,
            action_token_count=error.action_token_count,
        ) from error
    if per_token.ndim != 1 or len(per_token) != step.action_token_count:
        raise ValueError(
            f"trajectory {record.trajectory_id!r}, step {step_index}, "
            f"{direction.value}: score length does not match action_token_count"
        )
    mean = per_token.sum() / step.action_token_count
    return mean


def edge_logprob_mean(
    backbone: PolicyBackbone,
    record: TrajectoryRecord,
    initial_text: str,
    step_index: int,
    direction: ScoringDirection,
) -> Tensor:
    """Render, verify, and score one recorded action in per-token nats."""

    return _edge_logprob_mean(
        backbone,
        backbone.tokenizer,
        record,
        initial_text,
        step_index,
        direction,
    )


def _encoded_initial_context(
    tokenizer: TokenizerProtocol,
    record: TrajectoryRecord,
    initial_text: str,
) -> tuple[int, ...]:
    if not isinstance(initial_text, str):
        raise ValueError(f"trajectory {record.trajectory_id!r}: initial_text must be text")
    actual_tokenizer_id = tokenizer.tokenizer_id
    if record.tokenizer_id != actual_tokenizer_id:
        raise ValueError(
            f"trajectory {record.trajectory_id!r}: tokenizer_id does not match backbone"
        )
    if assembled_context_hash(initial_text) != record.initial_context.assembled_hash:
        raise ValueError(
            f"trajectory {record.trajectory_id!r}: initial assembled context hash mismatch"
        )
    encoded = encode_rollout_prompt(tokenizer, initial_text)
    if len(encoded) != record.initial_context.assembled_token_count:
        raise ValueError(
            f"trajectory {record.trajectory_id!r}: initial assembled token count mismatch"
        )
    return tuple(encoded)


def _encoded_query(
    tokenizer: TokenizerProtocol,
    record: TrajectoryRecord,
) -> tuple[int, ...]:
    """Encode only the canonical query ``q`` for the ``log Z_theta(q)`` head."""

    encoded = tokenizer.encode(record.initial_context.query)
    if not encoded:
        raise ValueError(f"trajectory {record.trajectory_id!r}: query encoded to invalid token ids")
    return tuple(encoded)


def _detached_float(value: Tensor) -> float:
    return float(value.detach().item())


def _detached_floats(values: list[Tensor]) -> list[float]:
    """One scalar transfer after the edge graphs have already been released."""
    from skillev.policy.teacher_forcing import detached_scalar_values

    return detached_scalar_values(values)


def score_trajectory(
    backbone: PolicyBackbone,
    record: TrajectoryRecord,
    initial_text: str,
    config: ScoringConfig,
) -> TrajectoryScore:
    """Build the differentiable four-term TTB residual and horizon loss."""

    if not isinstance(record, TrajectoryRecord):
        raise ValueError("record must be a TrajectoryRecord")
    if not isinstance(config, ScoringConfig):
        raise ValueError("config must be a ScoringConfig")
    # Validate the complete H0 commitment used by both edge renderers, while
    # conditioning the partition head on q alone as required by equation (7).
    tokenizer = backbone.tokenizer
    _encoded_initial_context(tokenizer, record, initial_text)
    z_value = backbone.z_value(_encoded_query(tokenizer, record))

    forward_scores = tuple(
        _edge_logprob_mean(
            backbone,
            tokenizer,
            record,
            initial_text,
            step_index,
            ScoringDirection.FORWARD,
        )
        for step_index in range(1, record.horizon + 1)
    )
    backward_scores = tuple(
        _edge_logprob_mean(
            backbone,
            tokenizer,
            record,
            initial_text,
            step_index,
            ScoringDirection.HINDSIGHT,
        )
        for step_index in range(1, record.horizon + 1)
    )
    sum_forward = reduce(add, forward_scores)
    sum_backward = reduce(add, backward_scores)
    log_shifted_reward = math.log(record.shifted_reward)
    delta = z_value + sum_forward - config.temperature_beta * log_shifted_reward - sum_backward
    loss = (delta / record.horizon) ** 2
    return TrajectoryScore(
        trajectory_id=record.trajectory_id,
        horizon=record.horizon,
        loss=loss,
        delta=delta,
        log_z=_detached_float(z_value),
        forward_per_token_means=tuple(_detached_float(value) for value in forward_scores),
        backward_per_token_means=tuple(_detached_float(value) for value in backward_scores),
        log_shifted_reward=log_shifted_reward,
        temperature_beta=config.temperature_beta,
    )


def backward_trajectory_delta_streaming(
    backbone: PolicyBackbone,
    record: TrajectoryRecord,
    initial_text: str,
    config: ScoringConfig,
    *,
    prepared: PreparedEdgePlan | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> StreamingTrajectoryScore:
    """Accumulate exact ``∇Delta`` while releasing every edge graph immediately.

    The caller must provide empty temporary parameter gradients.  On return the
    gradients are exactly ``∇Delta``; the caller applies
    ``2 * Delta / T**2`` (and the batch-mean factor) before publishing them to
    the optimizer.
    """

    if not isinstance(record, TrajectoryRecord):
        raise ValueError("record must be a TrajectoryRecord")
    if not isinstance(config, ScoringConfig):
        raise ValueError("config must be a ScoringConfig")
    tokenizer = backbone.tokenizer
    _encoded_initial_context(tokenizer, record, initial_text)

    if progress is not None:
        progress({"stage": "z"})
    z_tensor = backbone.z_value(_encoded_query(tokenizer, record))
    z_scalar = z_tensor.detach()
    z_tensor.backward()  # type: ignore[no-untyped-call]
    del z_tensor

    microbatch_config = getattr(backbone, "teacher_forcing_config", None)
    if microbatch_config is not None and microbatch_config.microbatch_size > 1:
        from .edge_microbatch import backward_edge_groups
        from .edge_plan import prepare_edge_plan

        plan = prepared or prepare_edge_plan(tokenizer, record, initial_text)
        if (
            plan.record is not record
            or plan.initial_text != initial_text
            or plan.tokenizer_id != tokenizer.tokenizer_id
        ):
            raise ValueError("microbatch edge plan belongs to another trajectory")
        forward_values = backward_edge_groups(backbone, plan, AdapterRole.FORWARD_POLICY)
        backward_values = backward_edge_groups(backbone, plan, AdapterRole.BACKWARD_POLICY)
        log_z = _detached_floats([z_scalar])[0]
    else:
        scalars = [z_scalar]
        session = getattr(backbone, "scoring_session", lambda role: nullcontext())
        for direction, role in (
            (ScoringDirection.FORWARD, AdapterRole.FORWARD_POLICY),
            (ScoringDirection.HINDSIGHT, AdapterRole.BACKWARD_POLICY),
        ):
            # Includes backward/checkpoint recomputation, not just forward.
            with session(role):
                for step_index in range(1, record.horizon + 1):
                    if progress is not None:
                        edge = None if prepared is None else prepared.edge(step_index, role)
                        length = 0 if edge is None else edge.full_length
                        checkpointed = bool(
                            microbatch_config is not None
                            and length >= microbatch_config.checkpoint_min_tokens
                            and getattr(backbone, "scoring_checkpoint_enabled", False)
                        )
                        progress(
                            {
                                "stage": "score-and-backward",
                                "direction": direction.value,
                                "step_index": step_index,
                                "prefix_tokens": None if edge is None else len(edge.prefix_ids),
                                "action_tokens": record.steps[step_index - 1].action_token_count,
                                "checkpointed": checkpointed,
                                "offloaded": bool(
                                    checkpointed
                                    and microbatch_config is not None
                                    and length >= microbatch_config.offload_min_tokens
                                ),
                            }
                        )
                    value = _edge_logprob_mean(
                        backbone, tokenizer, record, initial_text, step_index, direction, prepared
                    )
                    scalars.append(value.detach())
                    signed = value if direction is ScoringDirection.FORWARD else -value
                    signed.backward()  # type: ignore[no-untyped-call]
                    del signed, value
                    finish_profile = getattr(backbone, "finish_edge_backward_profile", None)
                    if finish_profile is not None:
                        finish_profile()
                    if progress is not None:
                        progress(
                            {
                                "stage": "edge-backward-returned",
                                "step_index": step_index,
                                "direction": direction.value,
                            }
                        )
        # No per-edge .item(): retain only detached scalars, never edge graphs.
        # Python fsum below retains the original canonical arithmetic order.
        values = _detached_floats(scalars)
        log_z = values[0]
        forward_values = values[1 : record.horizon + 1]
        backward_values = values[record.horizon + 1 :]

    if progress is not None:
        progress({"stage": "materialize"})

    return compose_streaming_score(record, config, log_z, forward_values, backward_values)


def compose_streaming_score(
    record: TrajectoryRecord,
    config: ScoringConfig,
    log_z: float,
    forward_values: Sequence[float],
    backward_values: Sequence[float],
) -> StreamingTrajectoryScore:
    """One terminal arithmetic path for sealed and completed-step execution."""
    log_shifted_reward = math.log(record.shifted_reward)
    delta = math.fsum(
        (
            log_z,
            *forward_values,
            -config.temperature_beta * log_shifted_reward,
            *(-value for value in backward_values),
        )
    )
    loss = (delta / record.horizon) ** 2
    return StreamingTrajectoryScore(
        trajectory_id=record.trajectory_id,
        horizon=record.horizon,
        loss=loss,
        delta=delta,
        gradient_coefficient=2.0 * delta / (record.horizon * record.horizon),
        log_z=log_z,
        forward_per_token_means=tuple(forward_values),
        backward_per_token_means=tuple(backward_values),
        log_shifted_reward=log_shifted_reward,
        temperature_beta=config.temperature_beta,
    )


def materialize_edge_records(
    score: MaterializedTrajectoryScore,
    *,
    forward_adapter_version: str,
    backward_adapter_version: str,
    batch_id: str | None = None,
    policy_snapshot_id: str | None = None,
    library_version: str | None = None,
    action_token_counts: tuple[int, ...] = (),
    action_token_ids: tuple[tuple[int, ...], ...] = (),
) -> tuple[EdgeLogprobRecord, ...]:
    """Translate detached edge diagnostics into the Layer 1 event contract."""

    if not isinstance(score, TrajectoryScore | StreamingTrajectoryScore):
        raise ValueError("score must be a materializable trajectory score")
    from skillev.contracts.ttb_training import EdgeScoreContext

    bound = batch_id is not None
    if bound:
        if (
            policy_snapshot_id is None
            or library_version is None
            or len(action_token_counts) != score.horizon
            or len(action_token_ids) != score.horizon
        ):
            raise ValueError("bound edge scores require a complete scoring context")
    elif (
        policy_snapshot_id is not None
        or library_version is not None
        or action_token_counts
        or action_token_ids
    ):
        raise ValueError("partial edge scoring context")
    return tuple(
        EdgeLogprobRecord(
            context=(
                EdgeScoreContext(
                    batch_id,
                    policy_snapshot_id,
                    library_version,
                    action_token_counts[step_index - 1],
                    action_token_ids[step_index - 1],
                )
                if batch_id is not None
                and policy_snapshot_id is not None
                and library_version is not None
                else None
            ),
            trajectory_id=score.trajectory_id,
            step_index=step_index,
            forward_logprob_per_token=forward_mean,
            backward_logprob_per_token=backward_mean,
            step_importance=forward_mean - backward_mean,
            forward_adapter_version=forward_adapter_version,
            backward_adapter_version=backward_adapter_version,
            scoring_stack_id="training-stack",
        )
        for step_index, (forward_mean, backward_mean) in enumerate(
            zip(
                score.forward_per_token_means,
                score.backward_per_token_means,
                strict=True,
            ),
            start=1,
        )
    )


def materialize_residual(
    score: MaterializedTrajectoryScore,
    *,
    raw_reward: float,
) -> TrajectoryResidual:
    """Build a float64-self-consistent audit residual from detached components."""

    if not isinstance(score, TrajectoryScore | StreamingTrajectoryScore):
        raise ValueError("score must be a materializable trajectory score")
    sum_forward = math.fsum(score.forward_per_token_means)
    sum_backward = math.fsum(score.backward_per_token_means)
    delta = (
        score.log_z + sum_forward - score.temperature_beta * score.log_shifted_reward - sum_backward
    )
    return TrajectoryResidual(
        trajectory_id=score.trajectory_id,
        log_z=score.log_z,
        sum_forward=sum_forward,
        sum_backward=sum_backward,
        log_shifted_reward=score.log_shifted_reward,
        raw_reward=raw_reward,
        temperature_beta=score.temperature_beta,
        delta=delta,
        horizon=score.horizon,
    )
