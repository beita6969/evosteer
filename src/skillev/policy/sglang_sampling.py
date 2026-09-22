"""Keep raw-softmax request seeds independent of co-batched judge filters.

SGLang's PyTorch filtered sampler indexes seeded noise by sorted probability
rank, while its unfiltered sampler uses vocabulary order. A greedy judge can
switch the whole batch to the former path. Preserve each unfiltered row's
standalone path; filtered rows and all unseeded execution remain upstream.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from torch import Tensor


class _SamplingInfo(Protocol):
    sampling_seed: Tensor | None
    top_ks: Tensor
    top_ps: Tensor
    min_ps: Tensor


def install_raw_sampling_compatibility() -> None:
    """Install in both the serving parent and spawned scheduler processes."""
    import torch

    upstream = importlib.import_module("sglang.srt.layers.sampler")
    sampler = upstream.Sampler
    original = sampler._sample_from_probs
    if getattr(original, "_skillev_raw_request_seeds", False):
        return

    @wraps(original)
    def sample(
        self: object,
        probs: Tensor,
        sampling_info: _SamplingInfo,
        positions: Tensor,
        simple_sampling_case: bool,
    ) -> Tensor:
        # Keep upstream validation and unsupported-backend errors intact.
        filtered = cast(
            "Tensor", original(self, probs, sampling_info, positions, simple_sampling_case)
        )
        if (
            simple_sampling_case
            or sampling_info.sampling_seed is None
            or upstream.get_flags().sampling_backend != "pytorch"
        ):
            return filtered
        unfiltered_rows = (
            (sampling_info.top_ks >= probs.shape[-1])
            & (sampling_info.top_ps >= 1)
            & (sampling_info.min_ps <= 0)
        ).view(-1)
        raw = upstream.sampling_from_probs_torch(
            probs, sampling_seed=sampling_info.sampling_seed, positions=positions
        )
        # No host synchronization or RNG draw is added: this is the same stateless
        # seeded function used by an unfiltered request running alone.
        return torch.where(unfiltered_rows, raw, filtered)

    sample._skillev_raw_request_seeds = True  # type: ignore[attr-defined]
    sampler._sample_from_probs = sample


class _RequestParameters(Protocol):
    sampling_seed: int | None


class _ScheduledRequest(Protocol):
    sampling_params: _RequestParameters


class _ScheduledBatch(Protocol):
    reqs: Sequence[_ScheduledRequest]


def install_unsigned_seed_compatibility() -> None:
    """Represent uint64 request seeds losslessly in SGLang's int64 tensor.

    The sampler converts that tensor back to uint64 before deriving its noise.
    Only the synchronous tensor-construction call sees signed representatives;
    request metadata, wire seeds and scientific seed coordinates stay unchanged.
    """
    upstream = importlib.import_module("sglang.srt.sampling.sampling_batch_info")
    batch_info = upstream.SamplingBatchInfo
    original = batch_info.from_schedule_batch.__func__
    if getattr(original, "_skillev_unsigned_request_seeds", False):
        return

    @wraps(original)
    def from_schedule_batch(cls: type[object], batch: _ScheduledBatch, vocab_size: int) -> object:
        restored: list[tuple[_RequestParameters, int]] = []
        try:
            for request in batch.reqs:
                parameters = request.sampling_params
                seed = parameters.sampling_seed
                if isinstance(seed, int) and (1 << 63) <= seed < (1 << 64):
                    restored.append((parameters, seed))
                    parameters.sampling_seed = seed - (1 << 64)
            return original(cls, batch, vocab_size)
        finally:
            for parameters, seed in restored:
                parameters.sampling_seed = seed

    from_schedule_batch._skillev_unsigned_request_seeds = True  # type: ignore[attr-defined]
    batch_info.from_schedule_batch = classmethod(from_schedule_batch)
