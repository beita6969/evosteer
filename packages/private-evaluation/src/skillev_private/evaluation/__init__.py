"""Trusted evaluators and private frozen-evaluation result contracts."""

from __future__ import annotations

from importlib import import_module

_MODULES: dict[str, tuple[str, ...]] = {
    "result_contracts": (
        "PRIVATE_EVALUATION_EPISODE_FORMAT",
        "PUBLIC_BENCHMARK_AGGREGATE_FORMAT",
        "EvaluationCostAggregate",
        "EvaluationEpisodeTelemetry",
        "NativeMetricAggregate",
        "NativeMetricValue",
        "ObservationStatusCount",
        "PrivateEvaluationEpisodeResult",
        "PublicBenchmarkAggregateResult",
        "RolloutTerminationCount",
        "aggregate_private_episodes",
        "public_native_metric_values",
    ),
    "frozen_runner": (
        "FrozenEpisodeRequest",
        "FrozenEvaluationExecutionConfig",
        "FrozenEvaluationRunner",
    ),
}
_EXPORTS = {
    name: (f"skillev_private.evaluation.{module}", name)
    for module, names in _MODULES.items()
    for name in names
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
