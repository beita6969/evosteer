"""Thread-safe scorer transport accounting, independent of actor inference cost."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any


class IncompleteNativeGradingError(RuntimeError):
    """Keep partial paid-for usage when a rubric invocation has no native score."""

    def __init__(self, scorer_cost: dict[str, float]) -> None:
        super().__init__("Native rubric grading did not finish")
        self.scorer_cost = dict(scorer_cost)


class MeteredCompletions:
    """Observe SDK usage, optionally retaining I/O in the private scorer only."""

    def __init__(
        self,
        delegate: Any,
        *,
        retain_evidence: bool = False,
        record_evidence: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.delegate = delegate
        self._lock = threading.Lock()
        self._attempts = self._completed = self._missing_usage = 0
        self._input_tokens = self._output_tokens = 0
        self._retain_evidence = retain_evidence or record_evidence is not None
        self._record_evidence = record_evidence
        self._criterion_messages: list[list[dict[str, str]]] | None = None
        self._evidence: list[dict[str, Any]] = []

    def create(self, **kwargs: object) -> Any:
        started = time.monotonic()
        with self._lock:
            self._attempts += 1
            evidence: dict[str, Any] = (
                {
                    "attempt": self._attempts,
                    "elapsed_scope": "grader-adapter-call-including-capacity-wait",
                    "criterion_indices": None
                    if self._criterion_messages is None
                    else [
                        index
                        for index, messages in enumerate(self._criterion_messages)
                        if kwargs.get("messages") == messages
                    ],
                    "request": deepcopy(
                        {
                            key: kwargs[key]
                            for key in (
                                "model",
                                "messages",
                                "temperature",
                                "max_tokens",
                                "max_completion_tokens",
                                "reasoning_effort",
                                "top_p",
                                "extra_body",
                                "response_format",
                                "timeout",
                                "stream",
                                "store",
                            )
                            if key in kwargs
                        }
                    ),
                }
                if self._retain_evidence
                else {}
            )
            if self._retain_evidence:
                self._evidence.append(evidence)
                self._publish()
        try:
            response = self.delegate.create(**kwargs)
        except Exception as error:
            if self._retain_evidence:
                with self._lock:
                    evidence["error_type"] = type(error).__name__
                    evidence["provider"] = getattr(error, "judge_provider", None)
                    evidence["endpoint"] = getattr(error, "judge_endpoint", None)
                    evidence["elapsed_seconds"] = time.monotonic() - started
                    rejected_response = getattr(error, "response", None)
                    if getattr(rejected_response, "choices", None):
                        self._response(evidence, rejected_response)
                    elif rejected_response is not None and callable(
                        getattr(rejected_response, "json", None)
                    ):
                        try:
                            body = rejected_response.json()
                        except ValueError:
                            body = getattr(rejected_response, "text", None)
                        evidence["error_response"] = {
                            "status_code": getattr(rejected_response, "status_code", None),
                            "body": body,
                        }  # Private response body only, never authorization headers.
                    self._publish()
            raise
        with self._lock:
            self._response(evidence, response)
            if self._retain_evidence:
                evidence["elapsed_seconds"] = time.monotonic() - started
                self._publish()
        return response

    def bind_criterion_messages(self, messages: list[list[dict[str, str]]] | None) -> None:
        """Private exact association from the official template, not text heuristics."""
        with self._lock:
            if self._attempts:
                raise ValueError("criterion binding must precede every grader request")
            self._criterion_messages = deepcopy(messages)

    def _publish(self) -> None:
        if self._record_evidence is not None:
            self._record_evidence(deepcopy(self._evidence))

    def _response(self, evidence: dict[str, Any], response: Any) -> None:
        usage = getattr(response, "usage", None)
        self._completed += 1
        if usage is None:
            self._missing_usage += 1
        else:
            self._input_tokens += usage.prompt_tokens
            self._output_tokens += usage.completion_tokens
        if self._retain_evidence:
            evidence["response"] = {
                "content": getattr(getattr(response.choices[0], "message", None), "content", None),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "input_tokens": None if usage is None else usage.prompt_tokens,
                "output_tokens": None if usage is None else usage.completion_tokens,
                "request_id": getattr(response, "_request_id", None),
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "provider": getattr(response, "judge_provider", None),
                "endpoint": getattr(response, "judge_endpoint", None),
                "provider_requested_model": getattr(response, "judge_requested_model", None),
                "failover_reason": getattr(response, "judge_failover_reason", None),
                "raw_response": response.model_dump(mode="json")
                if callable(getattr(response, "model_dump", None))
                else None,
            }

    def evidence(self) -> list[dict[str, Any]]:
        """Private grader requests/results, never exposed as actor feedback."""
        with self._lock:
            return deepcopy(self._evidence)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "model_request_attempts": float(self._attempts),
                "model_responses": float(self._completed),
                "unknown_usage_calls": float(
                    self._attempts - self._completed + self._missing_usage
                ),
                "known_input_tokens": float(self._input_tokens),
                "known_output_tokens": float(self._output_tokens),
            }
