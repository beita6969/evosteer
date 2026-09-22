"""Durable external-Judge handoff; reconnect transfers results, never resamples.

Only enabled by an explicit private runtime spool path. The external worker owns
API execution and persists each response before delivery. API timeout remains in
the original request; waiting for infrastructure is not another model attempt.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4


def write_spool_json(path: Path, value: Any) -> None:
    """Publish a complete private record atomically, including parent durability."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            os.chmod(temp, 0o600)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temp.unlink(missing_ok=True)


class JudgeSpoolError(RuntimeError):
    """A persisted unsuccessful API outcome must not become a negative label."""

    def __init__(self, request_id: str, error_type: str) -> None:
        self.request_id = request_id
        self.error_type = error_type
        super().__init__(f"External Judge operation {request_id} failed: {error_type}")


class JudgeSpoolClient:
    """OpenAI-shaped terminal-only client; a call creates exactly one durable job."""

    def __init__(self, root: Path, *, poll_seconds: float = 1.0) -> None:
        if poll_seconds <= 0:
            raise ValueError("Spool poll interval must be positive")
        self.root = root
        self.poll_seconds = poll_seconds
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        from openai.types.chat import ChatCompletion

        request_id = str(uuid4())
        request = {"request_id": request_id, "request": kwargs, "created_unix": time.time()}
        write_spool_json(self.root / "requests" / f"{request_id}.json", request)
        response_path = self.root / "responses" / f"{request_id}.json"
        while not response_path.exists():
            time.sleep(self.poll_seconds)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("request_id") != request_id or response.get("request") != kwargs:
            raise ValueError("Judge spool response belongs to another operation")
        # The durable response contains the full request; keep only outstanding
        # operations in the polling directory, rather than scanning run history.
        (self.root / "requests" / f"{request_id}.json").unlink(missing_ok=True)
        if response.get("status") != "completed":
            raise JudgeSpoolError(request_id, response.get("error_type", "UnknownAPIOutcome"))
        result = ChatCompletion.model_validate(response["raw_response"])
        result._request_id = response.get("api_request_id")
        return result

    def close(self) -> None:
        """No owned network client; persisted operations outlive this wrapper."""
