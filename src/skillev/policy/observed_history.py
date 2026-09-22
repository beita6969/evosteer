"""Public controller facts for history@2; never infer state from model prose."""

import json

from skillev.contracts import JsonValue


def public_feedback_value(text: str) -> dict[str, JsonValue]:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def history_facts(history: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    reads: list[JsonValue] = []
    native_count = 0
    last_command: JsonValue = None
    readonly_streak = repeated_reads = 0
    seen: set[tuple[str, str, str]] = set()
    for row in history:
        feedback = row.get("execution_feedback")
        if not isinstance(feedback, dict):
            continue
        observation = public_feedback_value(str(feedback.get("observation", "")))
        if observation.get("status") == "skill-read" and feedback.get("status") == "success":
            key = (
                str(observation.get("skill_id", "")),
                str(observation.get("version", "")),
                str(observation.get("library_version", "")),
            )
            repeated_reads += int(key in seen)
            seen.add(key)
            readonly_streak += 1
            reads.append(
                {
                    "skill_id": key[0],
                    "version": key[1],
                    "library_version": key[2],
                    "read_turn": row.get("step"),
                    "read_kind": "advisory_document",
                }
            )
        else:
            readonly_streak = 0
        # This counts only observations from the native environment bridge.
        # Failed/advisory reads and owner drafts cannot manufacture commands.
        if isinstance(observation.get("admissible_commands"), list) and isinstance(
            observation.get("text"), str
        ):
            native_count += 1
            last_command = observation.get("command")
    return {
        "native_environment_observation_count": native_count,
        "last_native_command": last_command,
        "skill_reads_used": len(reads),
        "read_documents": reads,
        "repeat_same_version_read_count": repeated_reads,
        "consecutive_advisory_reads": readonly_streak,
        "read_semantics": "A document read does not execute, inspect, or verify the environment.",
    }
