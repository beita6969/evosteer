"""Persist an actual author response before parsing, including malformed responses.

This is a transaction-local cache, not server idempotency. A submitted request
without a received response is ambiguous and MUST NOT be sent again. Public
request/response bodies may contain task material and belong in private storage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev.contracts import JsonValue
from skillev.runtime.sglang_gateway import SGLangControlTransport


@dataclass(slots=True)
class DurableAuthoringTransport:
    delegate: SGLangControlTransport
    path: Path

    def request(
        self,
        *,
        method: str,
        url: str,
        payload: Mapping[str, JsonValue] | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]:
        # Imported lazily: the outer journal also binds this transport.
        from .authoring_journal import UnresolvedAuthoringCallError, _write

        request = {"method": method, "url": url, "payload": payload}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved["request"] != request:
                raise ValueError("author transport request changed during recovery")
            if saved["state"] == "response-received":
                return int(saved["status"]), cast(JsonValue, saved["response"])
            if saved["state"] != "prepared":
                raise UnresolvedAuthoringCallError(
                    "submitted author request has no durable response; server-side "
                    "idempotency or an explicit owner decision is required"
                )
        else:
            _write(self.path, {"request": request, "state": "prepared"})
        _write(self.path, {"request": request, "state": "submitted"})
        status, response = self.delegate.request(
            method=method,
            url=url,
            payload=payload,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )
        # No token decoding, schema check, or usage interpretation before this write.
        _write(
            self.path,
            {
                "request": request,
                "state": "response-received",
                "status": status,
                "response": response,
            },
        )
        return status, response
