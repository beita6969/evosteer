"""Step materialization and canonical rollout artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from skillev.contracts import (
    JsonValue,
    ScientificSamplingCoordinate,
    TerminalReward,
    TrajectoryRecord,
    TrajectoryStep,
    build_trajectory_record,
    normalize_json,
    validate_sha256,
)
from skillev.contracts.ttb_common import require_iso_timestamp
from skillev.policy.interface import encode_rollout_prompt
from skillev.runtime.execution import ActionParseResult, EnvironmentObservation
from skillev.scoring import (
    render_forward_prefix_from_parts,
    render_hindsight_prefix_from_parts,
)

from .context import AssembledInitialContext
from .generator import RolloutTokenizerProtocol
from .types import PolicySnapshot, RolloutRequest, RolloutTermination


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _exact_object(
    value: object,
    *,
    label: str,
    expected: set[str],
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or normalized != value:
        raise ValueError(f"{label} must be a normalized JSON object")
    if set(normalized) != expected:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _wire_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class ReasoningDraft:
    step_index: int
    prompt_text: str
    prompt_hash: str
    text: str
    generated_token_ids: tuple[int, ...]
    policy_snapshot_id: str

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("reasoning draft step_index must be positive")
        _text(self.prompt_text, field="reasoning prompt_text")
        validate_sha256(self.prompt_hash)
        if not isinstance(self.text, str):
            raise ValueError("reasoning text must be text")
        if not isinstance(self.generated_token_ids, tuple) or any(
            type(item) is not int or item < 0 for item in self.generated_token_ids
        ):
            raise ValueError("reasoning generated_token_ids must be a token tuple")
        _text(self.policy_snapshot_id, field="reasoning policy_snapshot_id")


@dataclass(frozen=True, slots=True)
class ActionDraft:
    step_index: int
    forward_prefix_text: str
    forward_prefix_hash: str
    text: str
    token_ids: tuple[int, ...]
    parse_result: ActionParseResult
    policy_snapshot_id: str

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("action draft step_index must be positive")
        _text(self.forward_prefix_text, field="action forward_prefix_text")
        validate_sha256(self.forward_prefix_hash)
        _text(self.text, field="action text")
        if not isinstance(self.token_ids, tuple) or not self.token_ids:
            raise ValueError("action token_ids must be a non-empty tuple")
        if any(type(item) is not int or item < 0 for item in self.token_ids):
            raise ValueError("action token_ids must contain non-negative integers")
        if not isinstance(self.parse_result, ActionParseResult):
            raise ValueError("parse_result must be an ActionParseResult")
        _text(self.policy_snapshot_id, field="action policy_snapshot_id")


@dataclass(frozen=True, slots=True)
class CompletedStepDraft:
    reasoning: ReasoningDraft
    action: ActionDraft
    observation: EnvironmentObservation

    def __post_init__(self) -> None:
        if self.reasoning.step_index != self.action.step_index:
            raise ValueError("reasoning and action drafts must have the same step index")
        if self.reasoning.policy_snapshot_id != self.action.policy_snapshot_id:
            raise ValueError("reasoning and action drafts must use the same policy snapshot")
        if not isinstance(self.observation, EnvironmentObservation):
            raise ValueError("observation must be an EnvironmentObservation")
        _text(self.observation.observation_text, field="observation_text")


def materialize_trajectory_step(
    *,
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    draft: CompletedStepDraft,
) -> TrajectoryStep:
    """Commit a complete ``r/a/o`` edge without introducing policy scores."""

    expected_step_index = len(previous_steps) + 1
    if draft.reasoning.step_index != expected_step_index:
        raise ValueError("draft step index does not follow the trajectory prefix")
    forward = render_forward_prefix_from_parts(
        initial_text,
        previous_steps,
        expected_step_index,
        draft.reasoning.text,
    )
    if (
        forward.text != draft.action.forward_prefix_text
        or forward.prefix_hash != draft.action.forward_prefix_hash
    ):
        raise ValueError("action draft does not match the canonical forward prefix")
    hindsight = render_hindsight_prefix_from_parts(
        initial_text,
        previous_steps,
        expected_step_index,
        draft.observation.observation_text,
    )
    return TrajectoryStep(
        index=expected_step_index,
        reasoning_text=draft.reasoning.text,
        action_text=draft.action.text,
        action_token_ids=draft.action.token_ids,
        action_token_count=len(draft.action.token_ids),
        observation_text=draft.observation.observation_text,
        observation_status=draft.observation.observation_status,
        invoked_skill_ids=draft.observation.invoked_skill_ids,
        forward_prefix_hash=forward.prefix_hash,
        hindsight_prefix_hash=hindsight.prefix_hash,
    )


@dataclass(frozen=True, slots=True)
class RolloutManifest:
    """On-policy and engineering identities outside the scientific record."""

    trajectory_id: str
    task_id: str
    policy_snapshot: PolicySnapshot
    library_version: str
    sampling_coordinate: ScientificSamplingCoordinate
    decoding_snapshot_id: str
    assembler_version: str
    action_format_version: str
    generator_backend_id: str
    termination: RolloutTermination
    reasoning_token_counts: tuple[int, ...]
    started_at: str
    completed_at: str
    prompt_encoder_version: str = "legacy-raw-prompt@1"
    action_boundary_version: str = "none"
    reasoning_finish_reasons: tuple[str, ...] = ()
    action_finish_reasons: tuple[str, ...] = ()
    condition_id: str = "trained-skillev@1"
    initial_context_profile: str = "trained-skillev@1"

    def __post_init__(self) -> None:
        for field, value in (
            ("trajectory_id", self.trajectory_id),
            ("task_id", self.task_id),
            ("library_version", self.library_version),
            ("decoding_snapshot_id", self.decoding_snapshot_id),
            ("assembler_version", self.assembler_version),
            ("action_format_version", self.action_format_version),
            ("generator_backend_id", self.generator_backend_id),
            ("prompt_encoder_version", self.prompt_encoder_version),
            ("action_boundary_version", self.action_boundary_version),
            ("condition_id", self.condition_id),
            ("initial_context_profile", self.initial_context_profile),
        ):
            _text(value, field=field)
        if not isinstance(self.policy_snapshot, PolicySnapshot):
            raise ValueError("policy_snapshot must be a PolicySnapshot")
        if not isinstance(self.sampling_coordinate, ScientificSamplingCoordinate):
            raise ValueError("manifest requires a scientific sampling coordinate")
        if self.sampling_coordinate.task_id != self.task_id:
            raise ValueError("manifest sampling coordinate belongs to another task")
        if self.generator_backend_id != self.policy_snapshot.backend_id:
            raise ValueError("manifest backend does not match the policy snapshot")
        if not isinstance(self.termination, RolloutTermination):
            raise ValueError("termination must be a RolloutTermination")
        if not isinstance(self.reasoning_token_counts, tuple) or not self.reasoning_token_counts:
            raise ValueError("reasoning_token_counts must be a non-empty tuple")
        if any(type(count) is not int or count < 0 for count in self.reasoning_token_counts):
            raise ValueError("reasoning token counts must be non-negative integers")
        for label, reasons in (
            ("reasoning_finish_reasons", self.reasoning_finish_reasons),
            ("action_finish_reasons", self.action_finish_reasons),
        ):
            if not isinstance(reasons, tuple) or any(
                not isinstance(reason, str) or not reason for reason in reasons
            ):
                raise ValueError(f"{label} must be a tuple of non-empty text")
            if reasons and len(reasons) != len(self.reasoning_token_counts):
                raise ValueError(f"{label} must align with the rollout horizon")
        require_iso_timestamp(self.started_at, field="started_at", location="rollout manifest")
        require_iso_timestamp(self.completed_at, field="completed_at", location="rollout manifest")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action_format_version": self.action_format_version,
            "action_boundary_version": self.action_boundary_version,
            "action_finish_reasons": list(self.action_finish_reasons),
            "assembler_version": self.assembler_version,
            "completed_at": self.completed_at,
            "decoding_snapshot_id": self.decoding_snapshot_id,
            "generator_backend_id": self.generator_backend_id,
            "library_version": self.library_version,
            "policy_snapshot": self.policy_snapshot.to_value(),
            "prompt_encoder_version": self.prompt_encoder_version,
            "reasoning_finish_reasons": list(self.reasoning_finish_reasons),
            "reasoning_token_counts": list(self.reasoning_token_counts),
            "sampling_coordinate": self.sampling_coordinate.to_value(),
            "started_at": self.started_at,
            "task_id": self.task_id,
            "termination": self.termination.value,
            "trajectory_id": self.trajectory_id,
            "condition_id": self.condition_id,
            "initial_context_profile": self.initial_context_profile,
        }

    @classmethod
    def from_value(cls, value: object) -> RolloutManifest:
        legacy_fields = {
            "action_format_version",
            "assembler_version",
            "completed_at",
            "decoding_snapshot_id",
            "generator_backend_id",
            "library_version",
            "policy_snapshot",
            "reasoning_token_counts",
            "sampling_coordinate",
            "started_at",
            "task_id",
            "termination",
            "trajectory_id",
        }
        current_fields = legacy_fields | {
            "action_boundary_version",
            "action_finish_reasons",
            "prompt_encoder_version",
            "reasoning_finish_reasons",
        }
        profiled_fields = current_fields | {"condition_id", "initial_context_profile"}
        if not isinstance(value, dict):
            raise ValueError("rollout manifest must be a JSON object")
        normalized_value = normalize_json(value)
        if not isinstance(normalized_value, dict) or normalized_value != value:
            raise ValueError("rollout manifest must be a normalized JSON object")
        normalized_fields = set(normalized_value)
        if normalized_fields not in (legacy_fields, current_fields, profiled_fields):
            raise ValueError("rollout manifest has an incompatible field set")
        normalized = normalized_value
        current = normalized_fields in (current_fields, profiled_fields)
        profiled = normalized_fields == profiled_fields
        counts = normalized["reasoning_token_counts"]
        if type(counts) is not list:
            raise ValueError("reasoning_token_counts must be an array")
        reasoning_counts = tuple(
            _wire_int(item, field="reasoning_token_counts item") for item in counts
        )
        reasoning_reasons_value = normalized.get("reasoning_finish_reasons", [])
        action_reasons_value = normalized.get("action_finish_reasons", [])
        if type(reasoning_reasons_value) is not list or type(action_reasons_value) is not list:
            raise ValueError("manifest finish reasons must be arrays")
        reasoning_reasons = tuple(
            _text(item, field="reasoning_finish_reasons item") for item in reasoning_reasons_value
        )
        action_reasons = tuple(
            _text(item, field="action_finish_reasons item") for item in action_reasons_value
        )
        termination = _text(normalized["termination"], field="termination")
        return cls(
            trajectory_id=_text(normalized["trajectory_id"], field="trajectory_id"),
            task_id=_text(normalized["task_id"], field="task_id"),
            policy_snapshot=PolicySnapshot.from_value(normalized["policy_snapshot"]),
            library_version=_text(normalized["library_version"], field="library_version"),
            sampling_coordinate=ScientificSamplingCoordinate.from_value(
                normalized["sampling_coordinate"]
            ),
            decoding_snapshot_id=_text(
                normalized["decoding_snapshot_id"],
                field="decoding_snapshot_id",
            ),
            assembler_version=_text(
                normalized["assembler_version"],
                field="assembler_version",
            ),
            action_format_version=_text(
                normalized["action_format_version"],
                field="action_format_version",
            ),
            generator_backend_id=_text(
                normalized["generator_backend_id"],
                field="generator_backend_id",
            ),
            termination=RolloutTermination(termination),
            reasoning_token_counts=reasoning_counts,
            started_at=_text(normalized["started_at"], field="started_at"),
            completed_at=_text(normalized["completed_at"], field="completed_at"),
            prompt_encoder_version=(
                _text(normalized["prompt_encoder_version"], field="prompt_encoder_version")
                if current
                else "legacy-raw-prompt@1"
            ),
            action_boundary_version=(
                _text(normalized["action_boundary_version"], field="action_boundary_version")
                if current
                else "none"
            ),
            reasoning_finish_reasons=reasoning_reasons,
            action_finish_reasons=action_reasons,
            condition_id=(
                _text(normalized["condition_id"], field="condition_id")
                if profiled
                else "trained-skillev@1"
            ),
            initial_context_profile=(
                _text(normalized["initial_context_profile"], field="initial_context_profile")
                if profiled
                else "trained-skillev@1"
            ),
        )


@dataclass(frozen=True, slots=True)
class RolloutArtifact:
    """A complete on-policy sample consumable by Phase 5."""

    initial_context: AssembledInitialContext
    record: TrajectoryRecord
    manifest: RolloutManifest
    skill_input_evidence: tuple[dict[str, JsonValue], ...] | None = None

    def __post_init__(self) -> None:
        if self.manifest.trajectory_id != self.record.trajectory_id:
            raise ValueError("manifest and record trajectory IDs do not match")
        context_task_id = self.initial_context.contract.meta.get("task_id")
        if not isinstance(context_task_id, str) or self.manifest.task_id != context_task_id:
            raise ValueError("manifest and initial context task IDs do not match")
        context_library_version = self.initial_context.contract.meta.get("library_version")
        if (
            not isinstance(context_library_version, str)
            or self.manifest.library_version != context_library_version
        ):
            raise ValueError("manifest and initial context library versions do not match")
        if self.initial_context.contract != self.record.initial_context:
            raise ValueError("artifact initial-context commitments do not match")
        if self.manifest.decoding_snapshot_id != self.record.decoding_snapshot_id:
            raise ValueError("manifest and record decoding snapshots do not match")
        if self.manifest.policy_snapshot.tokenizer_id != self.record.tokenizer_id:
            raise ValueError("manifest and record tokenizer identities do not match")
        if self.manifest.assembler_version != self.record.initial_context.assembler_version:
            raise ValueError("manifest and initial context assembler versions do not match")
        if len(self.manifest.reasoning_token_counts) != self.record.horizon:
            raise ValueError("reasoning token counts do not align with record horizon")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **(
                {"skill_input_evidence": cast(JsonValue, list(self.skill_input_evidence))}
                if self.skill_input_evidence is not None
                else {}
            ),
            "initial_context": self.initial_context.to_value(),
            "manifest": self.manifest.to_value(),
            "record": self.record.to_value(),
        }

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        tokenizer: RolloutTokenizerProtocol,
    ) -> RolloutArtifact:
        """Load an artifact while decode-admitting its recorded action spans.

        Reload validates ``decode(recorded_ids) == action_text`` through the
        contracts factory; it never reconstructs scoring IDs from action text.
        """

        normalized = _exact_object(
            value,
            label="rollout artifact",
            expected={"initial_context", "manifest", "record"}
            | (
                {"skill_input_evidence"}
                if isinstance(value, dict) and "skill_input_evidence" in value
                else set()
            ),
        )
        raw_record = TrajectoryRecord.from_value(normalized["record"])
        initial_context = AssembledInitialContext.from_value(normalized["initial_context"])
        if (
            len(encode_rollout_prompt(tokenizer, initial_context.text))
            != initial_context.contract.assembled_token_count
        ):
            raise ValueError("artifact initial-context token count is invalid")
        admitted = build_trajectory_record(
            tokenizer=tokenizer,
            trajectory_id=raw_record.trajectory_id,
            environment_id=raw_record.environment_id,
            task_family=raw_record.task_family,
            initial_context=raw_record.initial_context,
            steps=raw_record.steps,
            horizon=raw_record.horizon,
            reward=raw_record.reward,
            shifted_reward=raw_record.shifted_reward,
            epsilon_min=raw_record.epsilon_min,
            tokenizer_id=raw_record.tokenizer_id,
            decoding_snapshot_id=raw_record.decoding_snapshot_id,
            created_at=raw_record.created_at,
        )
        return cls(
            initial_context=initial_context,
            record=admitted,
            manifest=RolloutManifest.from_value(normalized["manifest"]),
            skill_input_evidence=tuple(
                _exact_object(
                    item,
                    label="skill input evidence",
                    expected={
                        "format",
                        "step_index",
                        "phase",
                        "catalog_offered_skill_ids",
                        "catalog_visible_skill_ids",
                        "catalog_not_fully_visible_skill_ids",
                        "catalog_visibility_basis",
                        "visible_skill_body_refs",
                        "not_fully_visible_skill_body_refs",
                        "body_visibility_basis",
                    },
                )
                for item in cast(list[JsonValue], normalized["skill_input_evidence"])
            )
            if isinstance(normalized.get("skill_input_evidence"), list)
            else None,
        )


def finalize_trajectory_record(
    *,
    request: RolloutRequest,
    assembled: AssembledInitialContext,
    steps: tuple[TrajectoryStep, ...],
    reward: TerminalReward,
    tokenizer: RolloutTokenizerProtocol,
    created_at: str,
) -> TrajectoryRecord:
    """Apply the sole reward shift and decode-admit sampled action spans."""

    if reward.environment_id != request.task.environment_id:
        raise ValueError("terminal reward belongs to another environment")
    if assembled.contract.query != request.task.query:
        raise ValueError("assembled context belongs to another task query")
    if (
        len(encode_rollout_prompt(tokenizer, assembled.text))
        != assembled.contract.assembled_token_count
    ):
        raise ValueError("assembled context token count changed before finalization")
    return build_trajectory_record(
        tokenizer=tokenizer,
        trajectory_id=request.trajectory_id,
        environment_id=request.task.environment_id,
        task_family=request.task.task_family,
        initial_context=assembled.contract,
        steps=steps,
        horizon=len(steps),
        reward=reward,
        shifted_reward=reward.value + request.epsilon_min,
        epsilon_min=request.epsilon_min,
        tokenizer_id=tokenizer.tokenizer_id,
        decoding_snapshot_id=request.decoding.snapshot_id,
        created_at=created_at,
    )
