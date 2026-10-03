from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from skillev.contracts.canonical import stable_hash

_FENCED = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?```\s*$", re.DOTALL)


class ResponsesApiFrozenText:
    thread_safe_generation = True

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        reference_id: str,
        reasoning_effort: str = "low",
        timeout_seconds: float = 900.0,
        attempts: int = 2,
    ) -> None:
        if not base_url.startswith(("http://", "https://")) or not model or not api_key:
            raise ValueError("Responses API author requires a base URL, model and key")
        self.url = base_url.rstrip("/") + "/responses"
        self.model = model
        self._key = api_key
        self.reference_id = reference_id
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts
        self.configuration_id = str(
            stable_hash({"backend": "responses-api@1", "model": model, "effort": reasoning_effort})
        )

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "student-api-client/1.0",
                "Authorization": f"Bearer {self._key}",
            },
        )
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == self.attempts - 1:
                    raise
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == self.attempts - 1:
                    raise
            time.sleep(5 * (attempt + 1))
        raise AssertionError("unreachable")

    def frozen_text(
        self,
        text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        input_limit: int | None = None,
    ) -> tuple[str, int, int]:
        if not isinstance(text, str) or not text or type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("positive author generation limits required")
        if input_limit is not None and len(text.encode("utf-8")) > input_limit:
            raise ValueError("author prompt exceeds its declared input envelope")
        reply = self._post(
            {
                "model": self.model,
                "input": text,
                "max_output_tokens": max_new_tokens,
                "reasoning": {"effort": self.reasoning_effort},
            }
        )
        output = "".join(
            part.get("text", "")
            for item in reply.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") in {"output_text", "text"}
        )
        fenced = _FENCED.match(output)
        if fenced:
            output = fenced.group(1)
        usage = reply.get("usage") or {}
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if type(input_tokens) is not int or type(output_tokens) is not int:
            raise ValueError("Responses API reply lacks token usage")
        return output.strip(), input_tokens, output_tokens


__all__ = ["ResponsesApiFrozenText"]
