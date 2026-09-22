"""Deterministic 10 x 512 Protocol 11 training mixture construction."""

from __future__ import annotations

import random
from dataclasses import dataclass

from skillev.experiments.protocol_v11 import ACTIVE_BENCHMARKS_V11, BenchmarkV11

from .protocol_v11_sources import ProtocolV11PrivateRecord


@dataclass(frozen=True, slots=True)
class ProtocolV11Episode:
    episode_id: str
    benchmark: BenchmarkV11
    source_id: str
    cycle: int


def build_protocol_v11_training_mix(
    records: tuple[ProtocolV11PrivateRecord, ...],
) -> tuple[ProtocolV11Episode, ...]:
    blocks: list[ProtocolV11Episode] = []
    for benchmark in ACTIVE_BENCHMARKS_V11:
        source_ids = sorted(
            record.source_id
            for record in records
            if record.benchmark is benchmark and record.role.value == "training"
        )
        if not source_ids:
            raise ValueError(f"{benchmark.value} has no training population")
        generated = 0
        cycle = 0
        while generated < 512:
            shuffled = list(source_ids)
            random.Random(cycle).shuffle(shuffled)  # noqa: S311 - frozen scientific sampling
            for source_id in shuffled:
                if generated == 512:
                    break
                blocks.append(
                    ProtocolV11Episode(
                        episode_id=f"v11/{benchmark.value}/{generated:04d}/cycle-{cycle}",
                        benchmark=benchmark,
                        source_id=source_id,
                        cycle=cycle,
                    )
                )
                generated += 1
            cycle += 1
    random.Random(0).shuffle(blocks)  # noqa: S311 - frozen scientific sampling
    if len(blocks) != 5_120 or len({item.episode_id for item in blocks}) != 5_120:
        raise ValueError("Protocol 11 training mixture does not close")
    return tuple(blocks)


__all__ = ["ProtocolV11Episode", "build_protocol_v11_training_mix"]
