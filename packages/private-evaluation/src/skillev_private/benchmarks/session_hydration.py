"""Bounded ordered setup which drains live native resets even on cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar, cast

from skillev.training.rollout_workflow import AsyncResourceLimiter

_T = TypeVar("_T")
_R = TypeVar("_R")


async def hydrate_ordered(
    items: Sequence[_T], create: Callable[[_T], Awaitable[_R]], *, limiter: AsyncResourceLimiter
) -> tuple[_R, ...]:
    async def one(item: _T) -> _R:
        async with limiter.lease():
            return await create(item)

    # Shielded draining is deliberate: cancelling to_thread does not stop a
    # native environment reset. Its own finally/close must finish before exit.
    pending = asyncio.gather(*(one(item) for item in items), return_exceptions=True)
    try:
        results = await asyncio.shield(pending)
    except asyncio.CancelledError:
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
        raise
    for value in results:
        if isinstance(value, BaseException):
            raise value
    return tuple(cast(list[_R], results))
