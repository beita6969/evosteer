from __future__ import annotations

import json

import pytest

from skillev.policy import (
    ROLLOUT_SOURCE_MESSAGES_BEGIN,
    ROLLOUT_SOURCE_MESSAGES_END,
    rollout_chat_messages,
)
from skillev.policy.interface import ROLLOUT_CONTROLLER_SYSTEM_MESSAGE


def _prompt(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return (
        "before\n"
        + ROLLOUT_SOURCE_MESSAGES_BEGIN
        + payload
        + ROLLOUT_SOURCE_MESSAGES_END
        + "after\n"
    )


def test_rollout_chat_messages_merges_leading_source_system_block() -> None:
    messages = rollout_chat_messages(
        _prompt(
            [
                {"content": "benchmark system one", "role": "system"},
                {"content": "benchmark system two", "role": "system"},
                {"content": "question", "role": "user"},
            ]
        )
    )

    assert [item["role"] for item in messages] == ["system", "user", "user"]
    assert messages[0]["content"] == (
        ROLLOUT_CONTROLLER_SYSTEM_MESSAGE
        + "\n\nSource system instructions:\nbenchmark system one\n\nbenchmark system two"
    )
    assert messages[1] == {"content": "question", "role": "user"}
    assert messages[2] == {"content": "before\nafter\n", "role": "user"}


def test_rollout_chat_messages_rejects_late_source_system_turn() -> None:
    with pytest.raises(ValueError):
        rollout_chat_messages(
            _prompt(
                [
                    {"content": "question", "role": "user"},
                    {"content": "late instruction", "role": "system"},
                ]
            )
        )
