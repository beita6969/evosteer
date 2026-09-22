"""Dependency-light, answer-free ScienceWorld rollout adapter.

The JVM and ScienceWorld Python binding live behind ``ScienceWorldEpisode``.
Only model-visible text observations cross into this public package; the
native 0--100 score remains behind the private evaluator boundary.
"""

from __future__ import annotations

from ._embodied import (
    EmbodiedCommand,
    EmbodiedEpisode,
    EmbodiedPublicItem,
    EmbodiedPublicStep,
    EmbodiedTextEnvironment,
    OrderedEmbodiedTaskProvider,
)

SCIENCEWORLD_BENCHMARK_ID = "scienceworld"
SCIENCEWORLD_RESOURCE_ID = "scienceworld"

ScienceWorldCommand = EmbodiedCommand
ScienceWorldEpisode = EmbodiedEpisode
ScienceWorldPublicStep = EmbodiedPublicStep


class ScienceWorldPublicItem(EmbodiedPublicItem):
    """One public ScienceWorld task/variation with a pinned deployment."""

    __slots__ = ()
    BENCHMARK_ID = SCIENCEWORLD_BENCHMARK_ID
    RESOURCE_ID = SCIENCEWORLD_RESOURCE_ID


class OrderedScienceWorldTaskProvider(OrderedEmbodiedTaskProvider):
    __slots__ = ()
    ITEM_TYPE = ScienceWorldPublicItem


class ScienceWorldEnvironment(EmbodiedTextEnvironment):
    """Public ScienceWorld execution over an injected JVM-backed episode."""

    __slots__ = ()
    ITEM_TYPE = ScienceWorldPublicItem


__all__ = [
    "SCIENCEWORLD_BENCHMARK_ID",
    "SCIENCEWORLD_RESOURCE_ID",
    "OrderedScienceWorldTaskProvider",
    "ScienceWorldCommand",
    "ScienceWorldEnvironment",
    "ScienceWorldEpisode",
    "ScienceWorldPublicItem",
    "ScienceWorldPublicStep",
]
