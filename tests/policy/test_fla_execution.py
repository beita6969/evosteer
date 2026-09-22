from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from skillev.policy.fla_execution import (
    _configuration_value,
    configure_qwen35_fla_kernels,
    stabilize_autotuner,
)


@dataclass
class Configuration:
    kwargs: dict = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 3
    num_ctas: int = 1
    maxnreg: int | None = None


def tuner(configs):
    return SimpleNamespace(
        configs=configs,
        cache={("stale",): configs[-1]},
        cache_results=True,
        prune_configs=lambda kwargs: [c for c in configs if c.num_warps <= kwargs["limit"]],
    )


def test_reference_choices_replace_timing_cache_and_survive_reconfiguration():
    first, reference = Configuration(num_warps=2), Configuration(num_warps=8)
    value = tuner([first, reference])
    choices = [{"key": [32, "torch.bfloat16"], "configuration": _configuration_value(reference)}]
    stabilize_autotuner(value, choices)
    assert value.cache[(32, "torch.bfloat16")] is reference
    assert ("stale",) not in value.cache
    assert not value.cache_results
    value.cache[("new",)] = first
    prune = value.prune_configs
    stabilize_autotuner(value, choices)
    assert value.cache[("new",)] is first
    assert value.prune_configs is prune


def test_unseen_shape_choice_is_rank_order_independent_and_stays_eligible():
    configs = [Configuration(num_warps=i) for i in (8, 4, 2)]
    a, b = tuner(configs), tuner(list(reversed(configs)))
    stabilize_autotuner(a, [])
    stabilize_autotuner(b, [])
    left, right = a.prune_configs({"limit": 4}), b.prune_configs({"limit": 4})
    assert len(left) == len(right) == 1
    assert left[0] == right[0]
    assert left[0].num_warps <= 4


def test_unavailable_reference_kernel_is_not_silently_retuned():
    value = tuner([Configuration()])
    with pytest.raises(RuntimeError):
        stabilize_autotuner(
            value,
            [{"key": [1], "configuration": _configuration_value(Configuration(num_warps=8))}],
        )
    assert value.cache_results


def test_packaged_profile_installs_all_kernels_through_heuristic_wrappers(monkeypatch):
    import json
    from importlib.resources import files

    import skillev.policy.fla_execution as implementation

    profile = json.loads(files("skillev.policy").joinpath("qwen35_fla_execution.json").read_text())

    class Autotuner(SimpleNamespace):
        pass

    installed = {}
    modules = {}
    for kernel, module in implementation._KERNEL_MODULES.items():
        configs = [Configuration(**row["configuration"]) for row in profile[kernel]]
        value = Autotuner(**vars(tuner(configs)))
        installed[kernel] = value
        namespace = modules.setdefault(module, SimpleNamespace())
        setattr(namespace, kernel, SimpleNamespace(fn=value))
    modules["triton.runtime.autotuner"] = SimpleNamespace(Autotuner=Autotuner)
    original_import = implementation.importlib.import_module
    monkeypatch.setattr(
        implementation.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )
    configure_qwen35_fla_kernels()
    for kernel, value in installed.items():
        for row in profile[kernel]:
            assert _configuration_value(value.cache[tuple(row["key"])]) == row["configuration"]


def test_offline_profile_restores_training_choices_and_reports_fallbacks(monkeypatch):
    import skillev.policy.fla_execution as implementation

    a, b = Configuration(num_warps=2), Configuration(num_warps=4)
    value = tuner([a, b])
    stabilize_autotuner(value, [{"key": [1], "configuration": _configuration_value(a)}])
    monkeypatch.setattr(implementation, "_INSTALLED", {"kernel": value})
    value.prune_configs({"limit": 4})
    assert implementation.fla_execution_metrics()["kernel"]["fallbacks"] == 1
    original_cache, original_prune = dict(value.cache), value.prune_configs
    with implementation.offline_fla_profiling() as measured:
        assert value.prune_configs({"limit": 4}) == [a, b]
        value.cache[(2,)] = b  # stand-in for an actual offline measured choice
    assert value.cache == original_cache
    assert value.prune_configs is original_prune
    assert any(
        row["key"] == [2] and row["configuration"] == _configuration_value(b)
        for row in measured["kernel"]
    )
    with pytest.raises(RuntimeError):
        stabilize_autotuner(value, [{"key": [1], "configuration": _configuration_value(b)}])
