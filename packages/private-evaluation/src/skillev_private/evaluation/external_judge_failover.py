"""Provider failover without replaying an uncertain, possibly paid POST.

A failed read-only availability probe can route the first POST to official
OpenAI. Once a POST starts, a gateway failure is preserved and raised; only
subsequent, not-yet-sent requests switch providers. Each response retains its
actual route, independently of the shared Luna rubric protocol.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillev.evaluation.external_judge_policy import (
    DIRECT_JUDGE_API_BASE,
    DIRECT_JUDGE_MODEL,
    EXTERNAL_JUDGE_API_BASE,
    EXTERNAL_JUDGE_MODEL,
    EXTERNAL_JUDGE_PROVIDER,
)


def unavailable(error: BaseException) -> bool:
    from openai import APIConnectionError, APIStatusError

    return isinstance(error, APIConnectionError | TimeoutError | ConnectionError) or (
        isinstance(error, APIStatusError)
        and (error.status_code in {401, 403, 404, 408, 429} or error.status_code >= 500)
    )


@dataclass(slots=True)
class JudgeRouteState:
    """Shared by rubric clients in one process; switching does not resubmit work."""

    official: bool = False
    checked: bool = False
    reason: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def use_official(self, probe: Callable[[], None]) -> bool:
        with self.lock:
            if not self.checked:
                try:
                    probe()
                except Exception as error:
                    if not unavailable(error):
                        raise
                    self.official = True
                    self.reason = "gateway-availability-probe:" + type(error).__name__
                self.checked = True
            return self.official

    def failed_gateway_post(self, error: BaseException) -> None:
        if unavailable(error):
            with self.lock:
                self.checked = self.official = True
                self.reason = "prior-gateway-post-outcome-unresolved:" + type(error).__name__


PROCESS_ROUTE = JudgeRouteState()


def make_official_client() -> Any:
    """Only the explicitly selected failover condition can use this credential."""
    from openai import DefaultHttpxClient, OpenAI

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    filename = os.environ.get("OPENAI_API_KEY_FILE")
    path = (
        Path(filename).expanduser()
        if filename
        else Path.home() / ".config/skillev/official-openai-key"
    )
    if not key and path.is_file():
        key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Official fallback requires a separate private OpenAI credential")
    return OpenAI(
        api_key=key,
        base_url=DIRECT_JUDGE_API_BASE,
        max_retries=0,
        timeout=120.0,
        http_client=DefaultHttpxClient(follow_redirects=False),
    )


class FailoverCompletions:
    """One POST per invocation, with separate credentials and no provider vote."""

    def __init__(
        self,
        primary: Any,
        *,
        probe: Callable[[], None],
        official_factory: Callable[[], Any],
        state: JudgeRouteState | None = None,
    ) -> None:
        self.primary = primary
        self.probe = probe
        self.official_factory = official_factory
        self.state = state if state is not None else PROCESS_ROUTE
        self._official_client: Any = None
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("model") != EXTERNAL_JUDGE_MODEL:
            raise ValueError("failover is only defined for the declared Luna judge")
        if kwargs.get("stream") is not False or "tools" in kwargs:
            raise ValueError("Luna judge requests must be synchronous and tool-free")
        official = self.state.use_official(self.probe)
        provider = "openai-official" if official else EXTERNAL_JUDGE_PROVIDER
        endpoint = DIRECT_JUDGE_API_BASE if official else EXTERNAL_JUDGE_API_BASE
        requested_model = DIRECT_JUDGE_MODEL if official else EXTERNAL_JUDGE_MODEL
        delegate = self.primary
        if official:
            with self._lock:
                if self._official_client is None:
                    self._official_client = self.official_factory()
                delegate = self._official_client.chat.completions
        try:
            result = delegate.create(**{**kwargs, "model": requested_model})
        except Exception as error:
            # Never try a second provider for this POST: its outcome/cost may
            # be unknown. The caller must retain the incomplete rubric record.
            error.judge_provider = provider  # type: ignore[attr-defined]
            error.judge_endpoint = endpoint  # type: ignore[attr-defined]
            if not official:
                self.state.failed_gateway_post(error)
            raise
        result.judge_provider = provider
        result.judge_endpoint = endpoint
        result.judge_requested_model = requested_model
        result.judge_failover_reason = self.state.reason if official else None
        return result

    def close(self) -> None:
        if self._official_client is not None:
            self._official_client.close()
