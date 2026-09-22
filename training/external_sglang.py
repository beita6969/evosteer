"""Transactional adapter publication for an externally managed SGLang server."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _request(
    method: Callable[..., Any],
    url: str,
    *,
    attempts: int,
    timeout: float,
    **kwargs: object,
) -> Any:
    response = None
    for attempt in range(attempts):
        try:
            response = method(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt + 1 >= attempts or (response is not None and response.status_code < 500):
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _model_ids(
    http: Any,
    *,
    api_base: str,
    headers: dict[str, str],
    attempts: int,
    timeout: float,
) -> set[str]:
    response = _request(
        http.get,
        f"{api_base}/v1/models",
        attempts=attempts,
        headers=headers,
        timeout=timeout,
    )
    return {
        str(item["id"])
        for item in response.json().get("data", [])
        if isinstance(item, dict) and item.get("id")
    }


def _unload(
    http: Any,
    *,
    api_base: str,
    headers: dict[str, str],
    adapter_name: str,
    attempts: int,
    timeout: float,
) -> None:
    response = _request(
        http.post,
        f"{api_base}/unload_lora_adapter",
        attempts=attempts,
        headers=headers,
        json={"lora_name": adapter_name},
        timeout=timeout,
    )
    if response.json().get("success") is not True:
        raise RuntimeError("external SGLang rejected an adapter unload")


def publish_external_adapter(
    *,
    api_base: str,
    headers: dict[str, str],
    adapter_name: str,
    adapter_path: Path,
    previous_adapter: str,
    managed_prefix: str,
    request_timeout_seconds: float,
    max_retries: int,
    http: Any = requests,
) -> None:
    """Load and validate ``adapter_name`` without leaving stale managed adapters.

    SGLang limits the number of resident adapters.  A fresh trainer process does
    not know which versions an earlier run left behind, so stale project-owned
    versions are drained before publication.  The currently selected adapter is
    retained until its replacement has answered a validation request.
    """

    attempts = max_retries + 1
    models = _model_ids(
        http,
        api_base=api_base,
        headers=headers,
        attempts=attempts,
        timeout=request_timeout_seconds,
    )
    stale = sorted(
        name
        for name in models
        if name.startswith(managed_prefix) and name not in {adapter_name, previous_adapter}
    )
    for name in stale:
        _unload(
            http,
            api_base=api_base,
            headers=headers,
            adapter_name=name,
            attempts=attempts,
            timeout=request_timeout_seconds,
        )
    if adapter_name in models:
        validation = _request(
            http.post,
            f"{api_base}/v1/chat/completions",
            attempts=attempts,
            headers=headers,
            json={
                "max_tokens": 1,
                "messages": [{"content": "Reply with OK.", "role": "user"}],
                "model": adapter_name,
                "temperature": 0,
            },
            timeout=request_timeout_seconds,
        )
        if not validation.json().get("choices"):
            raise RuntimeError("restored theta adapter failed its validation request")
        if (
            previous_adapter != "theta_uninitialized"
            and previous_adapter != adapter_name
            and previous_adapter in models
        ):
            _unload(
                http,
                api_base=api_base,
                headers=headers,
                adapter_name=previous_adapter,
                attempts=attempts,
                timeout=request_timeout_seconds,
            )
        return

    loaded = False
    try:
        response = _request(
            http.post,
            f"{api_base}/load_lora_adapter",
            attempts=attempts,
            headers=headers,
            json={"lora_name": adapter_name, "lora_path": str(adapter_path)},
            timeout=request_timeout_seconds,
        )
        if response.json().get("success") is not True:
            raise RuntimeError("external SGLang rejected the theta adapter")
        loaded = True
        models = _model_ids(
            http,
            api_base=api_base,
            headers=headers,
            attempts=attempts,
            timeout=request_timeout_seconds,
        )
        if adapter_name not in models:
            raise RuntimeError("new theta adapter is absent from /v1/models")
        validation = _request(
            http.post,
            f"{api_base}/v1/chat/completions",
            attempts=attempts,
            headers=headers,
            json={
                "max_tokens": 1,
                "messages": [{"content": "Reply with OK.", "role": "user"}],
                "model": adapter_name,
                "temperature": 0,
            },
            timeout=request_timeout_seconds,
        )
        if not validation.json().get("choices"):
            raise RuntimeError("new theta adapter failed its validation request")
        if (
            previous_adapter != "theta_uninitialized"
            and previous_adapter != adapter_name
            and previous_adapter in models
        ):
            _unload(
                http,
                api_base=api_base,
                headers=headers,
                adapter_name=previous_adapter,
                attempts=attempts,
                timeout=request_timeout_seconds,
            )
    except Exception:
        if loaded:
            try:
                _unload(
                    http,
                    api_base=api_base,
                    headers=headers,
                    adapter_name=adapter_name,
                    attempts=1,
                    timeout=request_timeout_seconds,
                )
            except Exception:
                logger.exception("failed to unload rejected external adapter")
        raise
