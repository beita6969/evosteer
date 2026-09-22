"""Private served-output values, distinct from actor diagnostics and final claims."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .native_channels import ChannelStatus


class CandidateStatus(StrEnum):
    SUBMITTED = "submitted"
    MODEL_NO_FINAL = "model_no_final"
    MODEL_FORMAT_INVALID = "model_format_invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INFRASTRUCTURE_UNRESOLVED = "infrastructure_unresolved"
    LEGACY_UNVERIFIED = "legacy-unverified"


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredModelOutput:
    run_id: str
    arm_id: str
    episode_id: str
    attempt_id: str
    call_id: str
    policy_id: str
    benchmark: str
    participant: str
    purpose: str
    public_revision: int
    native_thinking: bool
    adapter_name: str | None
    budgets: dict[str, int]
    result: dict[str, Any]
    final_token_ids: tuple[int, ...]
    final_text: str
    channel_status: ChannelStatus
    continuation_of_call_id: str | None = None
    submission_outcome: dict[str, Any] | None = None

    @property
    def scope(self) -> tuple[str, str, str]:
        return self.run_id, self.arm_id, self.episode_id

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> StoredModelOutput:
        return cls(
            **{
                **value,
                "final_token_ids": tuple(value["final_token_ids"]),
                "channel_status": ChannelStatus(value["channel_status"]),
            }
        )


def served_token_stream(outputs: Sequence[StoredModelOutput]) -> tuple[int, ...]:
    """Reassemble only adjacent chunks of the same owner's actual generation."""
    previous = None
    stream: tuple[int, ...] = ()
    for output in outputs:
        if output.continuation_of_call_id is None:
            stream = ()
        elif (
            previous is None
            or output.continuation_of_call_id != previous.call_id
            or output.scope != previous.scope
            or output.attempt_id != previous.attempt_id
            or output.policy_id != previous.policy_id
            or output.adapter_name != previous.adapter_name
            or output.public_revision != previous.public_revision
            or output.purpose != previous.purpose
            or output.participant != "owner"
            or previous.participant != "owner"
            or not output.native_thinking
            or not previous.native_thinking
            or previous.result["finish_reason"] != "length"
            or previous.channel_status
            not in {
                ChannelStatus.REASONING_UNFINISHED,
                ChannelStatus.FINAL_EMPTY,
                ChannelStatus.FINAL_UNFINISHED,
            }
        ):
            raise ValueError("continuation is not the preceding unfinished owner token stream")
        stream += tuple(output.result["content_token_ids"])
        previous = output
    return stream
