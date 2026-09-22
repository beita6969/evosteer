from __future__ import annotations

import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from skillev.policy import sglang_sampling


@pytest.fixture
def upstream(monkeypatch):
    module = ModuleType("sglang.srt.layers.sampler")
    calls = []

    class Sampler:
        def _sample_from_probs(self, probs, sampling_info, positions, simple_sampling_case):
            calls.append("original")
            if getattr(sampling_info, "unsupported", False):
                raise NotImplementedError
            return torch.ones(probs.shape[0], dtype=torch.int32)

    def raw(probs, *, sampling_seed, positions):
        calls.append("raw")
        assert sampling_seed is not None
        assert positions.shape == sampling_seed.shape
        return torch.zeros(probs.shape[0], dtype=torch.int32)

    module.Sampler = Sampler
    module.sampling_from_probs_torch = raw
    module.get_flags = lambda: SimpleNamespace(sampling_backend="pytorch")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module, calls


def _inputs():
    info = SimpleNamespace(
        sampling_seed=torch.zeros(5, dtype=torch.int64),
        top_ks=torch.tensor([1 << 30, 1, 2, 1 << 30, 1 << 30]),
        top_ps=torch.tensor([1.0, 1.0, 1.0, 0.8, 1.0]),
        min_ps=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1]),
    )
    return torch.full((5, 4), 0.25), info, torch.arange(5)


def test_mixed_raw_row_uses_standalone_path_without_changing_filtered_rows(upstream):
    module, calls = upstream
    sglang_sampling.install_raw_sampling_compatibility()
    result = module.Sampler()._sample_from_probs(*_inputs(), False)
    assert result.tolist() == [0, 1, 1, 1, 1]
    assert result.dtype == torch.int32
    assert calls == ["original", "raw"]


@pytest.mark.parametrize("case", ["simple", "unseeded", "other_backend"])
def test_unaffected_paths_remain_upstream(upstream, case):
    module, calls = upstream
    probs, info, positions = _inputs()
    if case == "unseeded":
        info.sampling_seed = None
    if case == "other_backend":
        module.get_flags = lambda: SimpleNamespace(sampling_backend="custom")
    sglang_sampling.install_raw_sampling_compatibility()
    result = module.Sampler()._sample_from_probs(probs, info, positions, case == "simple")
    assert result.tolist() == [1] * 5
    assert calls == ["original"]


def test_upstream_errors_are_not_bypassed(upstream):
    module, calls = upstream
    probs, info, positions = _inputs()
    info.unsupported = True
    sglang_sampling.install_raw_sampling_compatibility()
    with pytest.raises(NotImplementedError):
        module.Sampler()._sample_from_probs(probs, info, positions, False)
    assert calls == ["original"]


def test_installation_is_idempotent(upstream):
    module, _ = upstream
    sglang_sampling.install_raw_sampling_compatibility()
    installed = module.Sampler._sample_from_probs
    sglang_sampling.install_raw_sampling_compatibility()
    assert module.Sampler._sample_from_probs is installed


def test_spawned_entrypoint_installs_without_launching_another_server(monkeypatch):
    from skillev.runtime import sglang_action_boundary, sglang_timing

    calls = []
    monkeypatch.setattr(
        sglang_sampling, "install_raw_sampling_compatibility", lambda: calls.append(1)
    )
    monkeypatch.setattr(
        sglang_sampling, "install_unsigned_seed_compatibility", lambda: calls.append(2)
    )
    monkeypatch.setattr(
        sglang_action_boundary, "install_action_root_boundary", lambda: calls.append(3)
    )
    monkeypatch.setattr(sglang_timing, "install_request_timing_transport", lambda: calls.append(4))
    runpy.run_module("skillev.runtime.sglang_server", run_name="__mp_main__")
    assert calls == [1, 2, 3, 4]


@pytest.mark.parametrize("fail", [False, True])
def test_uint64_seed_bits_and_request_metadata_survive_tensor_construction(monkeypatch, fail):
    module = ModuleType("sglang.srt.sampling.sampling_batch_info")
    seeds = [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=s)) for s in seeds]
    )

    class SamplingBatchInfo:
        @classmethod
        def from_schedule_batch(cls, batch, vocab_size):
            result = torch.tensor(
                [r.sampling_params.sampling_seed for r in batch.reqs], dtype=torch.int64
            )
            if fail:
                raise RuntimeError("downstream construction failure")
            return result

    module.SamplingBatchInfo = SamplingBatchInfo
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sglang_sampling.install_unsigned_seed_compatibility()
    installed = SamplingBatchInfo.from_schedule_batch.__func__
    sglang_sampling.install_unsigned_seed_compatibility()
    assert SamplingBatchInfo.from_schedule_batch.__func__ is installed
    if fail:
        with pytest.raises(RuntimeError):
            SamplingBatchInfo.from_schedule_batch(batch, 4)
    else:
        result = SamplingBatchInfo.from_schedule_batch(batch, 4)
        assert result.to(torch.uint64).tolist() == seeds
    assert [r.sampling_params.sampling_seed for r in batch.reqs] == seeds
