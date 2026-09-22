"""Cancellable SGLang HTTP without a blocking worker surviving an episode."""

from __future__ import annotations

import json
from asyncio import timeout
from collections.abc import Mapping
from typing import Protocol

from skillev.contracts import JsonValue, normalize_json

from .sglang_gateway import SGLangGatewayError


class AsyncSGLangControlTransport(Protocol):
    async def request(
        self,
        *,
        method: str,
        url: str,
        payload: Mapping[str, JsonValue] | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]: ...


class AiohttpSGLangControlTransport:
    """Each long-lived generation owns its socket and closes it on cancellation.

    Request-scoped sessions intentionally avoid a generator-owned connection
    pool: the public generator can be reused across event loops, and its
    synchronous close API must never leave an uninterruptible I/O thread.
    """

    async def request(
        self,
        *,
        method: str,
        url: str,
        payload: Mapping[str, JsonValue] | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]:
        import aiohttp

        try:
            async with (
                timeout(timeout_seconds),
                aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session,
            ):
                async with session.request(
                    method, url, json=payload, allow_redirects=False
                ) as response:
                    status = response.status
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        body.extend(chunk)
                        if len(body) > max_response_bytes:
                            raise SGLangGatewayError("SGLang response exceeded its byte limit")
        except (aiohttp.ClientError, TimeoutError, OSError) as error:
            raise SGLangGatewayError("SGLang asynchronous request failed") from error
        if not body:
            return status, None
        try:
            return status, normalize_json(json.loads(body))
        except (UnicodeDecodeError, ValueError) as error:
            if not 200 <= status < 300:
                return status, None
            raise SGLangGatewayError("SGLang response was not valid JSON") from error
