"""Facts about a received action, not a second parser or an answer selector."""

from dataclasses import asdict, dataclass
from typing import cast

from skillev.contracts import JsonValue
from skillev.runtime.execution import ActionParseResult, ActionParseStatus


@dataclass(frozen=True, slots=True)
class ActionSubmissionOutcome:
    finish_reason: str
    raw_output_token_count: int
    carrier_status: str
    action_token_cap: int
    controller_turns_remaining: int
    public_error_code: str | None
    response_received: bool = True
    admitted: bool | None = None
    executed: bool | None = None
    execution_status: str | None = None

    @classmethod
    def observe(
        cls,
        text: str,
        parsed: ActionParseResult,
        *,
        finish_reason: str,
        output_tokens: int,
        action_token_cap: int,
        turns_remaining: int,
    ) -> "ActionSubmissionOutcome":
        # The existing codec is the sole authority. In particular, delimiter
        # text inside a valid code/string argument must not create another call.
        if parsed.status is ActionParseStatus.VALID:
            status = "unique-complete"
        elif text.count("<tool_call>") > 1:
            status = "multiple-explicit-calls"
        elif finish_reason == "length" and parsed.status is ActionParseStatus.PARSE_ERROR:
            status = "incomplete-at-budget"
        elif parsed.status is ActionParseStatus.SCHEMA_INVALID:
            status = "unsupported-schema"
        elif not any(marker in text for marker in ("<tool_call>", "<function=", "{")):
            status = "prose-without-call"
        else:
            status = "malformed-carrier"
        return cls(
            finish_reason,
            output_tokens,
            status,
            action_token_cap,
            max(0, turns_remaining),
            parsed.public_error_code,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {"format": "action-submission-outcome@1", **cast(dict[str, JsonValue], asdict(self))}

    def public_feedback(self) -> dict[str, JsonValue] | None:
        if self.carrier_status == "unique-complete":
            return None
        return {
            **self.to_value(),
            "submitted": False,
            "instruction": (
                "No action was admitted. Submit one complete call using the declared tools "
                "and fields within the stated action token cap. The previous partial response "
                "was not executed or automatically completed. Do not send multiple submissions."
            ),
        }
