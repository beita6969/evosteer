from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillev.evaluation.judge_spool import (
    JudgeSpoolClient,
    JudgeSpoolError,
    write_spool_json,
)


def response_for(request: dict[str, Any]) -> dict[str, Any]:
    return {
        **request,
        "status": "completed",
        "api_request_id": "api-request-original",
        "raw_response": {
            "id": "chat-original",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-5.6-luna",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"criteria_met":true}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        },
    }


def test_disconnect_wait_never_creates_a_replacement_operation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    waits = []
    first_request = None

    def reconnect(_: float) -> None:
        nonlocal first_request
        paths = list((tmp_path / "requests").glob("*.json"))
        assert len(paths) == 1
        request = json.loads(paths[0].read_text())
        if first_request is None:
            first_request = request
        assert request == first_request
        waits.append(1)
        if len(waits) == 3:
            write_spool_json(tmp_path / "responses" / paths[0].name, response_for(request))

    monkeypatch.setattr("skillev.evaluation.judge_spool.time.sleep", reconnect)
    result = JudgeSpoolClient(tmp_path).create(model="gpt-5.6-luna", timeout=120, store=False)
    assert len(waits) == 3
    assert result.id == "chat-original"
    assert result._request_id == "api-request-original"
    assert result.usage.total_tokens == 20
    assert first_request["request"]["timeout"] == 120


def test_independent_identical_calls_are_not_merged(tmp_path: Path, monkeypatch: Any) -> None:
    def deliver(_: float) -> None:
        for p in (tmp_path / "requests").glob("*.json"):
            target = tmp_path / "responses" / p.name
            if not target.exists():
                write_spool_json(target, response_for(json.loads(p.read_text())))

    monkeypatch.setattr("skillev.evaluation.judge_spool.time.sleep", deliver)
    client = JudgeSpoolClient(tmp_path)
    client.create(model="gpt-5.6-luna")
    client.create(model="gpt-5.6-luna")
    assert len(list((tmp_path / "responses").glob("*.json"))) == 2


@pytest.mark.parametrize("kind", ["failed", "mismatched"])
def test_missing_or_wrong_result_is_not_a_task_label(
    tmp_path: Path, monkeypatch: Any, kind: str
) -> None:
    def deliver(_: float) -> None:
        p = next((tmp_path / "requests").glob("*.json"))
        response = response_for(json.loads(p.read_text()))
        if kind == "failed":
            response.update(status="unknown", error_type="APITimeoutError")
        else:
            response["request"] = {"model": "another-model"}
        write_spool_json(tmp_path / "responses" / p.name, response)

    monkeypatch.setattr("skillev.evaluation.judge_spool.time.sleep", deliver)
    with pytest.raises(JudgeSpoolError if kind == "failed" else ValueError):
        JudgeSpoolClient(tmp_path).create(model="gpt-5.6-luna")


def test_spool_opt_in_does_not_require_sending_api_key_to_worker(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from skillev_private.evaluation.external_judge_api import make_external_judge_client

    monkeypatch.setenv("SKILLEV_JUDGE_SPOOL_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    assert isinstance(make_external_judge_client(allow_official_fallback=False), JudgeSpoolClient)
    with pytest.raises(ValueError):
        make_external_judge_client()


def test_atomic_record_is_complete_and_private(tmp_path: Path) -> None:
    path = tmp_path / "requests" / "operation.json"
    write_spool_json(path, {"request": {"messages": ["private"]}})
    assert json.loads(path.read_text())["request"]["messages"] == ["private"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob("*.tmp"))
