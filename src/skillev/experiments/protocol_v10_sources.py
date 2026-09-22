"""Typed public-source plan for every private Protocol 10 population."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .protocol_v10 import ActiveBenchmarkProtocolV10, BenchmarkV10

SOURCE_PLAN_V10_FORMAT = "skillev-protocol-v10-source-plan@1"


@dataclass(frozen=True, slots=True)
class SourceRouteV10:
    population_id: str
    source: str
    subset: str
    split: str

    def __post_init__(self) -> None:
        values = (self.population_id, self.source, self.subset, self.split)
        if any(not value.strip() for value in values):
            raise ValueError("Protocol 10 source route fields must be non-empty")


@dataclass(frozen=True, slots=True)
class BenchmarkSourceRoutesV10:
    benchmark: BenchmarkV10
    routes: tuple[SourceRouteV10, ...]

    def __post_init__(self) -> None:
        if not self.routes:
            raise ValueError("benchmark source routes cannot be empty")
        names = tuple(route.population_id for route in self.routes)
        if len(names) != len(set(names)):
            raise ValueError("benchmark source plan repeats a population")


@dataclass(frozen=True, slots=True)
class ProtocolV10SourcePlan:
    protocol: ActiveBenchmarkProtocolV10
    benchmarks: tuple[BenchmarkSourceRoutesV10, ...]
    seed: int
    training_episodes_per_benchmark: int
    format: str = SOURCE_PLAN_V10_FORMAT

    def __post_init__(self) -> None:
        if self.format != SOURCE_PLAN_V10_FORMAT or self.seed != self.protocol.seed:
            raise ValueError("Protocol 10 source plan identity differs")
        if (
            self.training_episodes_per_benchmark
            != self.protocol.training_mix.episodes_per_benchmark
        ):
            raise ValueError("Protocol 10 source plan training count differs")
        if tuple(item.benchmark for item in self.benchmarks) != tuple(
            item.benchmark for item in self.protocol.benchmarks
        ):
            raise ValueError("Protocol 10 source plan benchmark order differs")
        expected = {
            population.population_id
            for benchmark in self.protocol.benchmarks
            for population in benchmark.populations
        }
        actual = {
            route.population_id for benchmark in self.benchmarks for route in benchmark.routes
        }
        if actual != expected:
            raise ValueError("Protocol 10 source plan does not cover exact populations")

    def route(self, population_id: str) -> SourceRouteV10:
        matches = tuple(
            route
            for benchmark in self.benchmarks
            for route in benchmark.routes
            if route.population_id == population_id
        )
        if len(matches) != 1:
            raise ValueError("Protocol 10 population has no unique source route")
        return matches[0]


def load_protocol_v10_source_plan(
    path: Path,
    *,
    protocol: ActiveBenchmarkProtocolV10,
) -> ProtocolV10SourcePlan:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Protocol 10 source plan must be an object")
    raw_benchmarks = value.get("benchmarks")
    if not isinstance(raw_benchmarks, dict):
        raise ValueError("Protocol 10 source-plan benchmarks must be an object")
    benchmarks: list[BenchmarkSourceRoutesV10] = []
    for benchmark in tuple(item.benchmark for item in protocol.benchmarks):
        raw_routes = raw_benchmarks.get(benchmark.value)
        if not isinstance(raw_routes, dict):
            raise ValueError("Protocol 10 benchmark source routes are absent")
        routes: list[SourceRouteV10] = []
        for raw in raw_routes.values():
            if not isinstance(raw, dict) or set(raw) != {"population", "source", "split", "subset"}:
                raise ValueError("Protocol 10 source route fields differ")
            if any(type(raw[field]) is not str for field in raw):
                raise TypeError("Protocol 10 source route values must be strings")
            routes.append(
                SourceRouteV10(
                    population_id=raw["population"],
                    source=raw["source"],
                    subset=raw["subset"],
                    split=raw["split"],
                )
            )
        benchmarks.append(BenchmarkSourceRoutesV10(benchmark, tuple(routes)))
    seed = value.get("seed")
    count = value.get("training_episodes_per_benchmark")
    format_value = value.get("format")
    if type(seed) is not int or type(count) is not int or type(format_value) is not str:
        raise TypeError("Protocol 10 source-plan identity fields have wrong types")
    return ProtocolV10SourcePlan(
        protocol=protocol,
        benchmarks=tuple(benchmarks),
        seed=seed,
        training_episodes_per_benchmark=count,
        format=format_value,
    )


__all__ = [
    "SOURCE_PLAN_V10_FORMAT",
    "BenchmarkSourceRoutesV10",
    "ProtocolV10SourcePlan",
    "SourceRouteV10",
    "load_protocol_v10_source_plan",
]
