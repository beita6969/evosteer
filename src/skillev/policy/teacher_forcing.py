"""Independent padded edge batch dimension with per-edge action-token outputs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from .action_logprobs import action_token_logprobs
from .interface import AdapterRole, PolicyScoringMemoryError
from .scoring_execution import ActivationOffload, TeacherForcingConfig


def detached_scalar_values(values: list[torch.Tensor]) -> list[float]:
    """Materialize backend scalars together, preserving their canonical order."""
    return [float(value) for value in torch.stack(values).cpu().tolist()]


def backward_action_logprob_means(
    scores: tuple[torch.Tensor, ...], lengths: tuple[int, ...], *, forward: bool
) -> tuple[float, ...]:
    """Backend tensor reduction; mathematical grouping stays in scoring."""
    means = torch.stack([score.sum() / k for score, k in zip(scores, lengths, strict=True)])
    signed = means.sum() if forward else -means.sum()
    signed.backward()  # type: ignore[no-untyped-call]
    return tuple(float(value) for value in means.detach().cpu().tolist())


class TeacherForcingExecutor:
    def __init__(self, config: TeacherForcingConfig) -> None:
        self.config = config
        self.offload = ActivationOffload(config.pinned_memory_bytes)
        self.metrics: list[dict[str, Any]] = []
        self._profile_start: torch.cuda.Event | None = None
        self._session = False
        self._session_checkpointed: bool | None = None
        self._resident_model: int | None = None
        self._offload_before = (0, 0, 0.0, 0.0, 0)

    @contextmanager
    def session(self, model: Any) -> Iterator[None]:
        if self._session:
            raise RuntimeError("teacher-forcing sessions cannot overlap")
        previous_training = model.training
        self._session = True
        self._session_checkpointed = None
        try:
            yield
        finally:
            self._session = False
            self._session_checkpointed = None
            if model.training != previous_training:
                model.train(previous_training)

    def finish_backward_profile(self) -> None:
        if self.metrics:
            saved, restored, packed, unpacked, resident = self._offload_before
            self.metrics[-1].update(
                offload_saved_bytes=self.offload.saved_bytes - saved,
                offload_restored_bytes=self.offload.restored_bytes - restored,
                offload_pack_host_seconds=self.offload.pack_host_seconds - packed,
                offload_unpack_host_seconds=self.offload.unpack_host_seconds - unpacked,
                offload_resident_parameter_bytes=self.offload.resident_parameter_bytes - resident,
                offload_pinned_pool={
                    "allocated_bytes": self.offload.pool.allocated_bytes,
                    "live_bytes": self.offload.pool.live_bytes,
                    "peak_live_bytes": self.offload.pool.peak_live_bytes,
                    "allocations_cumulative": self.offload.pool.allocations,
                    "reuses_cumulative": self.offload.pool.reuses,
                    "fallbacks_cumulative": self.offload.pool.fallbacks,
                },
            )
        if self._profile_start is None:
            return
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end.record()
        end.synchronize()  # explicit profiling only, never a normal-step synchronization
        self.metrics[-1]["cuda_forward_backward_ms"] = self._profile_start.elapsed_time(end)
        self.metrics[-1]["cuda_owner_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        self._profile_start = None

    def score(
        self,
        *,
        model: Any,
        device: torch.device,
        edges: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        role: AdapterRole,
        checkpoint_enabled: bool,
    ) -> tuple[torch.Tensor, ...]:
        if not edges or any(not p or not a for p, a in edges):
            raise ValueError("teacher forcing requires non-empty prefix/action pairs")
        lengths = [len(p) + len(a) for p, a in edges]
        width = max(lengths)
        if len(edges) > 1 and (
            len(edges) > self.config.microbatch_size
            or width * len(edges) > self.config.microbatch_max_tokens
        ):
            raise ValueError("edge batch exceeds the configured padded-token capacity")
        checkpointed = checkpoint_enabled and width >= self.config.checkpoint_min_tokens
        offloaded = checkpointed and width >= self.config.offload_min_tokens
        if offloaded and self._resident_model != id(model):
            self.offload.bind_resident_parameters(model.parameters())
            self._resident_model = id(model)
        if not self._session or self._session_checkpointed != checkpointed:
            model.train(checkpointed)
            self._session_checkpointed = checkpointed
        # Qwen3.5 multimodal stores text token settings on text_config, unlike
        # the text-only fixture and GPT-2. No synthetic token reaches scoring.
        pad_id = getattr(model.config.get_text_config(), "pad_token_id", None)
        if pad_id is None:
            pad_id = 0  # masked right padding; never scored or used as an action
        inputs = torch.full((len(edges), width), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(inputs)
        for row, ((prefix, action), length) in enumerate(zip(edges, lengths, strict=True)):
            inputs[row, :length] = torch.tensor(prefix + action, dtype=torch.long, device=device)
            mask[row, :length] = 1
        context = self.offload.context() if offloaded else nullcontext()
        start = time.perf_counter()
        if self.config.profile_cuda and device.type == "cuda":
            self._profile_start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            self._profile_start.record()
        saved_before = self.offload.saved_bytes
        resident_before = self.offload.resident_parameter_bytes
        pack_before = self.offload.pack_host_seconds
        self._offload_before = (
            saved_before,
            self.offload.restored_bytes,
            pack_before,
            self.offload.unpack_host_seconds,
            resident_before,
        )
        metric: dict[str, Any] = {
            "role": role.value,
            "edges": len(edges),
            "lengths": lengths,
            "prefix_tokens": sum(len(p) for p, _ in edges),
            "action_tokens": sum(len(a) for _, a in edges),
            "padded_tokens": width * len(edges),
            "checkpointed": checkpointed,
            "offloaded": offloaded,
        }
        try:
            with context:
                kwargs: dict[str, Any] = {
                    "input_ids": inputs,
                    "attention_mask": mask,
                    "use_cache": False,
                    "return_dict": True,
                }
                model_type = model.config.model_type
                if model_type in {"qwen3_5", "qwen3_5_text"}:
                    if len(edges) == 1:
                        kwargs["logits_to_keep"] = len(edges[0][1]) + 1
                        logits = model(**kwargs).logits
                        action_logits = (logits[0, :-1, :],)
                    else:
                        # Union of action prediction positions, not full-prefix vocab
                        # logits. Rows have independent attention/recurrent state.
                        positions = sorted(
                            {j for p, a in edges for j in range(len(p) - 1, len(p) + len(a) - 1)}
                        )
                        kwargs["logits_to_keep"] = torch.tensor(positions, device=device)
                        logits = model(**kwargs).logits
                        lookup = {position: i for i, position in enumerate(positions)}
                        action_logits = tuple(
                            logits[
                                row, [lookup[j] for j in range(len(p) - 1, len(p) + len(a) - 1)], :
                            ]
                            for row, (p, a) in enumerate(edges)
                        )
                elif model_type == "gpt2" and device.type == "cpu":
                    logits = model(**kwargs).logits
                    action_logits = tuple(
                        logits[row, len(p) - 1 : len(p) + len(a) - 1, :]
                        for row, (p, a) in enumerate(edges)
                    )
                else:
                    raise RuntimeError("teacher forcing requires the fixed Qwen3.5 backend")
                return tuple(
                    action_token_logprobs(
                        values,
                        torch.tensor(action, dtype=torch.long, device=device),
                        implementation=self.config.action_logprobs,
                    )
                    for values, (_, action) in zip(action_logits, edges, strict=True)
                )
        except torch.OutOfMemoryError as error:
            raise PolicyScoringMemoryError(
                prefix_token_count=max(len(p) for p, _ in edges),
                action_token_count=max(len(a) for _, a in edges),
                role=role,
            ) from error
        finally:
            metric["forward_host_seconds"] = time.perf_counter() - start
            metric["offload_saved_bytes"] = self.offload.saved_bytes - saved_before
            metric["offload_resident_parameter_bytes"] = (
                self.offload.resident_parameter_bytes - resident_before
            )
            metric["offload_pack_host_seconds"] = self.offload.pack_host_seconds - pack_before
            self.metrics.append(metric)
            if not self._session:
                model.eval()
