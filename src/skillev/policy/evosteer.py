"""One trainable causal-LM actor and an adapter-disabled frozen reference.

No hindsight model or unscored reasoning phase is used. The same finite token
grammar is applied during sampling and both sides of teacher-forced scoring.
Torch/Transformers/PEFT are optional until a model backend is constructed.
"""

from __future__ import annotations

import inspect
import json
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from skillev.contracts.canonical import canonical_json, stable_hash
from skillev.contracts.evosteer import DecisionRecord
from skillev.rollout.evosteer import CONTROLLER_OUTPUT_HEAD_CHARS, CONTROLLER_OUTPUT_TAIL_CHARS

from .action_trie import ActionTrie


def _configuration_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _configuration_value(value.value)
    if isinstance(value, dict):
        return {str(key): _configuration_value(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return sorted((_configuration_value(item) for item in value), key=canonical_json)
    if isinstance(value, tuple | list):
        return [_configuration_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _model_configuration(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is not None and callable(getattr(config, "to_dict", None)):
        return _configuration_value(config.to_dict())
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameters": [(name, list(p.shape), str(p.dtype)) for name, p in model.named_parameters()],
        "structure": str(model),
    }


def _checkpoint_tensor_names(model_path: str) -> tuple[str, ...]:
    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        return tuple(json.loads(index.read_text())["weight_map"])
    from safetensors import safe_open

    names: list[str] = []
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), "pt", device="cpu") as handle:
            names.extend(handle.keys())
    return tuple(names)


def _checkpoint_skip_modules(patterns: Any, tensor_names: tuple[str, ...]) -> Any:
    """Keep only FP8 skip entries that name a module or tensor in the checkpoint.

    Released dense Qwen3.5 FP8 checkpoints list MoE router names such as
    ``layers.N.mlp.gate``. Transformers matches skip entries by prefix, so the
    stale entry silently excludes ``mlp.gate_proj`` from FP8 conversion and its
    block scales are dropped at load time, leaving unscaled FP8 codes as weights.
    """
    if not patterns:
        return patterns
    prefixes = set()
    for name in tensor_names:
        parts = name.split(".")
        prefixes.update(".".join(parts[:end]) for end in range(1, len(parts) + 1))
    return [pattern for pattern in patterns if pattern in prefixes]


def _assert_block_scales_loaded(model: Any, tensor_names: tuple[str, ...]) -> None:
    scaled = {
        name.removesuffix(".weight_scale_inv")
        for name in tensor_names
        if name.endswith(".weight_scale_inv")
    }
    unscaled = [
        name
        for name, module in model.named_modules()
        if name in scaled and getattr(module, "weight_scale_inv", None) is None
    ]
    if unscaled:
        raise RuntimeError(
            f"{len(unscaled)} block-scaled FP8 checkpoint weights were loaded without "
            f"their scales, e.g. {unscaled[:3]}"
        )


_AUTOTUNE_LOCK = threading.RLock()


def _install_triton_autotuner_lock() -> None:
    """Serialize Triton autotuner dispatch across threads.

    ``Autotuner.run`` keeps per-call state on the shared kernel object
    (``self.nargs``, cleared when it returns). In the pipelined loop the actor
    update (worker thread, one GPU) and the rollout replica (event-loop thread,
    another GPU) dispatch the same autotuned linear-attention kernels, so a
    benchmark in one thread can read the other's cleared state. Dispatch is
    host-side (GPU work stays asynchronous), so a process-wide lock costs little.
    Subclasses that override ``run`` (fla's cached autotuner) are wrapped too;
    the lock is re-entrant because they call ``super().run``. Idempotent.
    """
    try:
        from triton.runtime import autotuner
    except ImportError:
        return
    pending = [autotuner.Autotuner]
    while pending:
        kind = pending.pop()
        pending.extend(kind.__subclasses__())
        run = kind.__dict__.get("run")
        if run is None or getattr(run, "_skillev_autotune_lock", False):
            continue

        def locked(self: Any, *args: Any, _run: Any = run, **kwargs: Any) -> Any:
            with _AUTOTUNE_LOCK:
                return _run(self, *args, **kwargs)

        locked._skillev_autotune_lock = True  # type: ignore[attr-defined]
        kind.run = locked


def _install_fp8_input_gradients() -> None:
    """Let actor gradients reach LoRA adapters through frozen block-scaled FP8 linears.

    The finegrained-fp8 matmul kernels register no autograd formula, so a backward
    pass through any FP8Linear downstream of an adapter raises. Base weights are
    frozen, so only the input gradient is needed: grad_x = grad_y @ dequant(W).
    The forward still runs the FP8 kernel; the patch is process-local.
    """
    import torch
    from transformers.integrations import finegrained_fp8

    linear = finegrained_fp8.FP8Linear
    if getattr(linear, "_evosteer_input_gradient", False):
        return
    fp8_forward = linear.forward

    class FrozenFP8Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, inputs: Any, module: Any) -> Any:
            ctx.module = module
            with torch.no_grad():
                return fp8_forward(module, inputs)

        @staticmethod
        def backward(ctx: Any, grad: Any) -> tuple[Any, None]:
            module = ctx.module
            weight = module.weight
            rows, columns = weight.shape
            scale = module.weight_scale_inv.float()
            if module.block_size is not None:
                block_rows, block_columns = module.block_size
                scale = scale.repeat_interleave(block_rows, 0)[:rows]
                scale = scale.repeat_interleave(block_columns, 1)[:, :columns]
            dense = (weight.float() * scale).to(grad.dtype)
            return grad @ dense, None

    def forward(self: Any, inputs: Any) -> Any:
        if torch.is_grad_enabled() and inputs.requires_grad and self.weight.element_size() == 1:
            return FrozenFP8Linear.apply(inputs, self)
        return fp8_forward(self, inputs)

    linear.forward = forward
    linear._evosteer_input_gradient = True


def _declared_context_limits(model: Any, label: str) -> dict[str, int]:
    """Read explicit positional limits; sliding attention windows are not context caps.

    RoPE scaling alone does not prove a larger supported window, so no factor is
    guessed. Backends without positional metadata use the caller's declared cap.
    """
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        config = text_config
    limits = {}
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
        value = getattr(config, name, None)
        if type(value) is int and 0 < value < 1_000_000_000:
            limits[f"{label}.{name}"] = value
    return limits


class ContextWindowExceededError(ValueError):
    """The exact public prefix and reserved continuation do not fit the model."""

    def __init__(
        self,
        *,
        prompt_tokens: int,
        reserved_tokens: int,
        context_window: int,
        operation: str = "actor prompt",
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.reserved_tokens = reserved_tokens
        self.context_window = context_window
        super().__init__(
            f"{operation} needs {prompt_tokens} prefix + {reserved_tokens} continuation tokens, "
            f"exceeding context_window={context_window}; full history was not truncated"
        )


@dataclass(frozen=True, slots=True)
class ActionMenu:
    actions: tuple[str, ...]
    paths: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.actions, tuple)
            or not self.actions
            or any(not isinstance(action, str) or not action for action in self.actions)
            or len(self.actions) != len(self.paths)
            or len(set(self.actions)) != len(self.actions)
        ):
            raise ValueError("action menu requires one unique token path per action")
        ActionTrie(self.paths)


class CausalLMOrchestrator:
    """Adapter-backed actor with a frozen reference sharing base weights.

    A separate frozen reference_model is also accepted for small CPU fixtures.
    Its use is explicit: production from_pretrained uses disable_adapter().
    Full public history is preserved by default; an oversized exact prefix fails
    explicitly. The old head/tail projection requires context_mode='debug_head_tail'
    and has a distinct configuration identity. It is not the paper default.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        reference_id: str,
        encoding_dim: int,
        context_window: int = 8192,
        max_action_tokens: int = 512,
        reference_model: Any = None,
        model_pin: str | None = None,
        context_mode: str = "full",
        activation_checkpointing: bool = False,
    ) -> None:
        import torch

        if (
            not isinstance(reference_id, str)
            or not reference_id.strip()
            or type(encoding_dim) is not int
            or encoding_dim < 1
        ):
            raise ValueError("pinned reference identity and positive encoding width required")
        if (
            type(context_window) is not int
            or type(max_action_tokens) is not int
            or context_window <= max_action_tokens + 32
            or max_action_tokens < 1
        ):
            raise ValueError("context window must leave room for a prompt and complete action")
        if not isinstance(context_mode, str) or context_mode not in {"full", "debug_head_tail"}:
            raise ValueError("context_mode must be full or debug_head_tail")
        context_limits = _declared_context_limits(model, "actor")
        if reference_model is not None:
            context_limits.update(_declared_context_limits(reference_model, "reference"))
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        # Hugging Face uses enormous sentinel integers when the tokenizer has
        # no configured limit; those are not claims about positional support.
        if type(tokenizer_limit) is int and 0 < tokenizer_limit < 1_000_000_000:
            context_limits["tokenizer.model_max_length"] = tokenizer_limit
        if context_limits and context_window > min(context_limits.values()):
            raise ValueError(
                f"context_window={context_window} exceeds declared model/tokenizer limit "
                f"{min(context_limits.values())}"
            )
        if reference_model is None and not callable(getattr(model, "disable_adapter", None)):
            raise TypeError("production actor must expose an adapter-disabled reference")
        eos = getattr(tokenizer, "eos_token_id", None)
        if type(eos) is not int or eos < 0:
            raise ValueError("tokenizer must have an explicit action terminator token")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.reference_model = reference_model
        self.reference_id = reference_id
        self.encoding_dim = encoding_dim
        self.context_window = context_window
        self.max_action_tokens = max_action_tokens
        self.context_mode = context_mode
        self._context_limits = context_limits
        self.model_context_window = min(context_limits.values()) if context_limits else None
        self.eos_token_id = eos
        self.version = 0
        if reference_model is not None:
            actor_storage = {
                parameter.untyped_storage().data_ptr()
                for parameter in model.parameters()
                if parameter.requires_grad
            }
            if any(
                parameter.untyped_storage().data_ptr() in actor_storage
                for parameter in reference_model.parameters()
            ):
                raise ValueError(
                    "a frozen reference cannot share trainable actor parameter storage"
                )
            reference_model.eval()
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)
            if next(reference_model.parameters()).device != next(model.parameters()).device:
                raise ValueError("separate actor/reference models must be on the same device")
            if (
                reference_model.get_input_embeddings().num_embeddings
                != model.get_input_embeddings().num_embeddings
            ):
                raise ValueError("actor/reference models must share the same token vocabulary")
        elif getattr(model, "peft_config", None) is not None:
            # The reference is the adapter-disabled base, so no base tensor may
            # enter the actor optimizer even if the caller enabled it by mistake.
            for name, parameter in model.named_parameters():
                if "lora_" not in name:
                    parameter.requires_grad_(False)
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("actor has no trainable parameters")
        self._torch = torch
        if model_pin is not None and (not isinstance(model_pin, str) or not model_pin.strip()):
            raise ValueError("model_pin must be a nonempty explicit model artifact identity")
        self._model_pin = model_pin or reference_id
        self._base_config_id = stable_hash(_model_configuration(model))
        self._reference_config_id = (
            self._base_config_id
            if reference_model is None
            else stable_hash(_model_configuration(reference_model))
        )
        self._adapter_config_id = stable_hash(
            {
                name: _configuration_value(config.to_dict())
                for name, config in getattr(model, "peft_config", {}).items()
            }
        )
        backend = getattr(tokenizer, "backend_tokenizer", None)
        try:
            tokenizer_source = inspect.getsource(type(tokenizer))
        except (TypeError, OSError):
            tokenizer_source = None
        self._tokenizer_config_id = stable_hash(
            {
                "class": type(tokenizer).__qualname__,
                "vocabulary": tokenizer.get_vocab()
                if callable(getattr(tokenizer, "get_vocab", None))
                else None,
                "implementation": tokenizer_source,
                "backend": None if backend is None else backend.to_str(),
                "chat_template": getattr(tokenizer, "chat_template", None),
                "special_tokens": _configuration_value(
                    getattr(tokenizer, "special_tokens_map", {})
                ),
                "eos_token_id": self.eos_token_id,
                "init_kwargs": _configuration_value(getattr(tokenizer, "init_kwargs", {})),
            }
        )
        # Recompute decoder-layer activations in the actor's gradient pass only.
        # Layers switch to checkpointing inside score(); sampling keeps its cache
        # and every submodule stays in eval mode, so the forward is unchanged.
        self._checkpoint_layers: tuple[Any, ...] = ()
        if activation_checkpointing:
            from functools import partial

            from torch.utils.checkpoint import checkpoint
            from transformers.modeling_layers import GradientCheckpointingLayer

            self._checkpoint_layers = tuple(
                module
                for name, module in model.named_modules()
                if isinstance(module, GradientCheckpointingLayer) and "visual" not in name
            )
            for layer in self._checkpoint_layers:
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        self._logit_window = {
            id(active): "logits_to_keep"
            in inspect.signature(
                active.get_base_model().forward
                if callable(getattr(active, "get_base_model", None))
                else active.forward
            ).parameters
            for active in (model, reference_model)
            if active is not None
        }

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        revision: str | None = None,
        reference_id: str,
        device: str = "cpu",
        dtype: str = "float32",
        lora_rank: int = 64,
        lora_alpha: int = 128,
        target_modules: tuple[str, ...] = ("o_proj",),
        context_window: int = 8192,
        max_action_tokens: int = 512,
        context_mode: str = "full",
    ) -> CausalLMOrchestrator:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported model dtype")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, revision=revision, local_files_only=True, trust_remote_code=False
        )
        # Qwen3.5 is published as a multimodal checkpoint even when the
        # policy supplies text only.  Its state dict stores the language
        # model below ``model.language_model``; the generic causal-LM auto
        # class instead selects Qwen3_5ForCausalLM, whose constructor expects
        # a text-only config and cannot load those keys.  Keep the multimodal
        # wrapper and use its text path (pixel inputs remain omitted).
        config = AutoConfig.from_pretrained(
            model_path, revision=revision, local_files_only=True, trust_remote_code=False
        )
        if getattr(config, "model_type", None) == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration

            model_class: Any = Qwen3_5ForConditionalGeneration
        else:
            model_class = AutoModelForCausalLM
        model_kwargs = {
            "revision": revision,
            "torch_dtype": getattr(torch, dtype),
            "local_files_only": True,
            "trust_remote_code": False,
        }
        # Transformers 5 loads Qwen3.5's fine-grained FP8 linear layers from
        # the kernels-community repositories.  The checkpoint is local and
        # explicitly selected by the caller; allow that already-inspected
        # kernel implementation to load while keeping ordinary model loading
        # on the conservative default path.
        if getattr(config, "model_type", None) == "qwen3_5":
            model_kwargs["allow_all_kernels"] = True
            # Qwen3.5's local checkpoint is block-scaled FP8.  The matching
            # kernels package is installed in the training environment; keep
            # the native FP8 representation so the 27B actor and author fit
            # alongside the existing serving processes on the two H200s.
            # ``trust_remote_code`` here is scoped to the explicitly selected
            # Qwen3.5 kernel repository; ordinary model/tokenizer loading
            # remains local-only and does not enable arbitrary model code.
            model_kwargs["trust_remote_code"] = True
            # Place native FP8 modules directly on the requested CUDA device.
            # Calling ``Module.to`` after loading can re-enter the kernel
            # loader without this trust flag and can create a transient full
            # size copy during the move.
            requested = torch.device(device)
            model_kwargs["device_map"] = {"": int(requested.index or 0)}
        # Only a checkpoint that ships FP8 weights is loaded natively as FP8; a
        # BF16 checkpoint must never be quantized on load.
        fp8_checkpoint = getattr(config, "model_type", None) == "qwen3_5" and bool(
            getattr(config, "quantization_config", None)
        )
        tensor_names: tuple[str, ...] = ()
        if fp8_checkpoint:
            try:
                from transformers import FineGrainedFP8Config

                # Transformers lazily resolves this kernel on the first
                # forward, after its ``allow_all_kernels`` loading context has
                # ended.  Preload it with explicit trust for this local,
                # user-selected checkpoint and register the module in the
                # Transformers cache so the first training step is
                # deterministic and does not fail the publisher check.
                from kernels import get_kernel
                from transformers.integrations.hub_kernels import _KERNEL_MODULE_MAPPING

                # Pin the exact v4 commit so the Pod can run fully offline
                # after the small kernel snapshot is staged in its HF cache.
                _KERNEL_MODULE_MAPPING["finegrained-fp8"] = get_kernel(
                    "kernels-community/finegrained-fp8",
                    revision="8c178950e8710e26a2210e4e909cd02dc16c8715",
                    trust_remote_code=True,
                )

                quant = getattr(config, "quantization_config", {}) or {}
                tensor_names = _checkpoint_tensor_names(model_path)
                model_kwargs["quantization_config"] = FineGrainedFP8Config(
                    activation_scheme=quant.get("activation_scheme", "dynamic"),
                    weight_block_size=tuple(quant.get("weight_block_size", (128, 128))),
                    dequantize=False,
                    modules_to_not_convert=_checkpoint_skip_modules(
                        quant.get("modules_to_not_convert"), tensor_names
                    ),
                    scale_fmt=quant.get("scale_fmt", "float"),
                )
            except ImportError:
                tensor_names = ()
        base: Any = model_class.from_pretrained(model_path, **model_kwargs)
        # After loading, so autotuner subclasses of kernels the model imported exist.
        _install_triton_autotuner_lock()
        # Every orchestrator prompt has a new length, and the cuDNN SDPA backend
        # builds an execution plan per shape: measured 1.0 s vs 0.1 s per 9B
        # forward over distinct lengths. PyTorch's flash backend has no per-shape
        # plan. Process-wide; affects speed and rounding only.
        torch.backends.cuda.enable_cudnn_sdp(False)
        if fp8_checkpoint:
            _assert_block_scales_loaded(base, tensor_names)
            if tensor_names:
                _install_fp8_input_gradients()
        if not hasattr(base, "hf_device_map"):
            base = base.to(device)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        actor = get_peft_model(
            base,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                target_modules=list(target_modules),
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
        config = getattr(base.config, "text_config", base.config)
        width = getattr(config, "hidden_size", getattr(config, "n_embd", None))
        if type(width) is not int:
            raise ValueError("model does not expose its text hidden-state width")
        return cls(
            actor,
            tokenizer,
            reference_id=reference_id,
            encoding_dim=width,
            context_window=context_window,
            max_action_tokens=max_action_tokens,
            context_mode=context_mode,
            activation_checkpointing=getattr(base.config, "model_type", None) == "qwen3_5",
            model_pin=stable_hash(
                {
                    "local_path": str(Path(model_path).expanduser().resolve()),
                    "revision": revision,
                    "commit_hash": getattr(base.config, "_commit_hash", None),
                    "reference_id": reference_id,
                }
            ),
        )

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    @property
    def actor_id(self) -> str:
        return f"{self.reference_id}/evosteer-actor/{self.version}"

    def trainable_parameters(self) -> tuple[Any, ...]:
        return tuple(p for p in self.model.parameters() if p.requires_grad)

    @contextmanager
    def _checkpointed(self) -> Iterator[None]:
        # GradientCheckpointingLayer checks only the layer's own training flag.
        for layer in self._checkpoint_layers:
            layer.training = True
        try:
            yield
        finally:
            for layer in self._checkpoint_layers:
                layer.training = False

    @contextmanager
    def _active_model(self, reference: bool) -> Iterator[Any]:
        if reference and self.reference_model is not None:
            yield self.reference_model
        else:
            context = self.model.disable_adapter() if reference else nullcontext()
            with context:
                yield self.model

    def encode_prompt(self, text: str, *, reserve_tokens: int | None = None) -> tuple[int, ...]:
        if not isinstance(text, str) or not text:
            raise ValueError("prompt text must be nonempty")
        reserve = self.max_action_tokens if reserve_tokens is None else reserve_tokens
        if type(reserve) is not int or reserve < 1:
            raise ValueError("reserved output tokens must be a positive integer")
        limit = self.context_window - reserve
        messages = [
            {
                "role": "system",
                # Interface only: what each action does, never when to use it. How to
                # compose a team is what training must learn, so the prompt carries no
                # strategy (the old "select the output node, then stop" wording biased
                # the actor and the reference toward the minimal ADD/SET_OUTPUT/STOP
                # chain). Kept short: it is re-sent with every decision's prompt.
                "content": (
                    "You orchestrate a team of agents by choosing one legal graph action. "
                    "Action effects: ADD_AGENT adds a node with a role and optional skill and "
                    "executes it; a node whose role is not primary_role_id also receives the "
                    "current draft. RERUN_AGENT executes a node again with its previous output "
                    "and inbound messages. ADD_EDGE feedback delivers the source output at the "
                    "target's next execution; ADD_EDGE revise executes the target at once. "
                    "BIND_SKILL adds a skill to a node and executes it again. DROP_AGENT removes "
                    "a node. SET_OUTPUT selects the node whose output is scored. STOP ends the "
                    "episode and scores that output. reference_value: the frozen reference's "
                    "estimated final score from this state; reference_value_change: its change "
                    "since the previous decision. Each output is shown once (see output_ref, "
                    "body_ref); omission markers are display-only. "
                    "Return only the action; no separate reasoning text."
                ),
            },
            {"role": "user", "content": text},
        ]
        if self._has_chat_template():
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        else:
            ids = self.tokenizer.encode(canonical_json(messages), add_special_tokens=False)
        ids = tuple(ids)
        self._validate_ids(ids)
        if len(ids) > limit:
            if self.context_mode == "full" or limit < 32:
                raise ContextWindowExceededError(
                    prompt_tokens=len(ids),
                    reserved_tokens=reserve,
                    context_window=self.context_window,
                )
            # Explicit synthetic/debug ablation. Exact projected IDs remain
            # recorded, and are shared by pi/rho during sampling and scoring.
            head = max(16, limit // 4)
            ids = ids[:head] + ids[-(limit - head) :]
        if not ids:
            raise ValueError("empty model prompt")
        return ids

    def _has_chat_template(self) -> bool:
        return callable(getattr(self.tokenizer, "apply_chat_template", None)) and bool(
            getattr(self.tokenizer, "chat_template", None)
        )

    def _validate_ids(self, ids: tuple[int, ...]) -> None:
        if not isinstance(ids, tuple) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("model input must be a nonempty tuple of token IDs")
        vocabulary = self.model.get_input_embeddings().num_embeddings
        if any(token >= vocabulary for token in ids):
            raise ValueError("token ID exceeds the model vocabulary")

    def menu(self, action_values: tuple[dict[str, Any], ...]) -> ActionMenu:
        actions = tuple(canonical_json(action) for action in action_values)
        paths: list[tuple[int, ...]] = []
        for text in actions:
            tokens = tuple(self.tokenizer.encode(text, add_special_tokens=False))
            if self.tokenizer.decode(tokens, skip_special_tokens=False) != text:
                raise ValueError("action tokenizer must preserve the canonical executed text")
            if self.eos_token_id in tokens:
                raise ValueError("action content contains the reserved terminator")
            path = (*tokens, self.eos_token_id)
            self._validate_ids(path)
            if len(path) > self.max_action_tokens:
                raise ValueError("legal action exceeds the declared action-token envelope")
            paths.append(path)
        return ActionMenu(actions, tuple(paths))

    def _forward(
        self, model: Any, ids: tuple[int, ...], *, hidden: bool = False, logits_to_keep: int = 0
    ) -> Any:
        torch = self._torch
        self._validate_ids(ids)
        if len(ids) > self.context_window:
            raise ContextWindowExceededError(
                prompt_tokens=len(ids),
                reserved_tokens=0,
                context_window=self.context_window,
                operation="model forward",
            )
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        optional = {}
        if self._logit_window.get(id(model), False):
            optional["logits_to_keep"] = 1 if hidden else logits_to_keep
        # A single unpadded sequence needs no attention mask; causal masking
        # is implied. An explicit all-ones mask only forces slower paths.
        return model(
            input_ids=inputs,
            output_hidden_states=hidden,
            use_cache=False,
            **optional,
        )

    def _next_logits(
        self, model: Any, ids: tuple[int, ...], *, cache: Any, cached_length: int
    ) -> tuple[Any, Any, int]:
        """Advance a local inference cache by all tokens since the last branch.

        Forced grammar tokens are batched into the next update. Cache ownership
        never crosses a sampling call or the actor/reference adapter boundary.
        Models returning no cache transparently continue with full prefills.
        """
        torch = self._torch
        self._validate_ids(ids)
        if len(ids) > self.context_window:
            raise ContextWindowExceededError(
                prompt_tokens=len(ids),
                reserved_tokens=0,
                context_window=self.context_window,
                operation="cached generation",
            )
        start = cached_length if cache is not None else 0
        if start >= len(ids):
            raise ValueError("a generation cache update must consume new tokens")
        input_ids = torch.tensor([ids[start:]], dtype=torch.long, device=self.device)
        # No attention mask: one unpadded sequence, so the cache position alone
        # defines causal masking. An explicit all-ones mask made each cached
        # single-token step of Qwen3.5 about 13x slower (0.45 s vs 0.035 s).
        kwargs = {"input_ids": input_ids, "use_cache": True}
        if cache is not None:
            kwargs["past_key_values"] = cache
        if self._logit_window.get(id(model), False):
            kwargs["logits_to_keep"] = 1
        output = model(**kwargs)
        next_cache = getattr(output, "past_key_values", None)
        return output.logits[0, -1], next_cache, len(ids) if next_cache is not None else 0

    def sample(
        self, prompt_ids: tuple[int, ...], menu: ActionMenu, *, reference: bool, seed: int
    ) -> tuple[int, ...]:
        torch = self._torch
        self._validate_ids(prompt_ids)
        if type(reference) is not bool or type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("sampling requires a boolean reference flag and uint64 seed")
        if len(prompt_ids) + max(map(len, menu.paths)) > self.context_window:
            raise ContextWindowExceededError(
                prompt_tokens=len(prompt_ids),
                reserved_tokens=max(map(len, menu.paths)),
                context_window=self.context_window,
                operation="masked sampling",
            )
        for path in menu.paths:
            self._validate_ids(path)
            if path[-1] != self.eos_token_id or self.eos_token_id in path[:-1]:
                raise ValueError("legal action paths require one final reserved terminator")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        trie = ActionTrie(menu.paths)
        generated: tuple[int, ...] = ()
        cache, cached_length = None, 0
        with torch.no_grad(), self._active_model(reference) as model:
            while not trie.complete(generated):
                allowed = trie.allowed(generated)
                if len(allowed) == 1:
                    token = allowed[0]
                else:
                    logits, cache, cached_length = self._next_logits(
                        model, prompt_ids + generated, cache=cache, cached_length=cached_length
                    )
                    scores = logits[list(allowed)].float()
                    if not bool(torch.isfinite(scores).all()):
                        raise FloatingPointError("nonfinite masked sampling logits")
                    index = torch.multinomial(scores.softmax(-1), 1, generator=generator).item()
                    token = allowed[int(index)]
                generated += (token,)
        return generated

    def score(self, record: DecisionRecord, *, reference: bool = False) -> Any:
        """Sum masked token log probabilities; singleton legal tokens contribute zero."""
        torch = self._torch
        if not isinstance(record, DecisionRecord) or type(reference) is not bool:
            raise ValueError("score requires a validated decision and boolean reference flag")
        for path in record.legal_token_paths:
            self._validate_ids(path)
            if path[-1] != self.eos_token_id or self.eos_token_id in path[:-1]:
                raise ValueError("recorded action support uses an incompatible terminator")
        if (
            self.tokenizer.decode(record.action_token_ids[:-1], skip_special_tokens=False)
            != record.action_json
        ):
            raise ValueError("recorded action text differs from its exact token sequence")
        masks = ActionTrie(record.legal_token_paths).masks(record.action_token_ids)
        self._validate_ids(record.prompt_ids)
        if len(record.prompt_ids) + max(map(len, record.legal_token_paths)) > self.context_window:
            raise ContextWindowExceededError(
                prompt_tokens=len(record.prompt_ids),
                reserved_tokens=max(map(len, record.legal_token_paths)),
                context_window=self.context_window,
                operation="masked teacher forcing",
            )
        if all(len(allowed) == 1 for allowed in masks):
            if reference:
                return torch.zeros((), dtype=torch.float32, device=self.device)
            # A deterministic grammar has probability one independently of the
            # model. Retain a zero derivative for standalone backward calls,
            # without allocating an entire model forward graph for STOP.
            return self.trainable_parameters()[0].reshape(-1)[0].float() * 0.0
        context = torch.no_grad() if reference else self._checkpointed()
        with context, self._active_model(reference) as model:
            logits = self._forward(
                model,
                record.prompt_ids + record.action_token_ids[:-1],
                logits_to_keep=len(record.action_token_ids),
            ).logits
            offset = logits.shape[1] - len(record.action_token_ids)
            if offset < 0:
                raise ValueError("teacher forcing did not return all action-token logits")
            terms = []
            for index, (token, allowed) in enumerate(
                zip(record.action_token_ids, masks, strict=True)
            ):
                row = logits[0, offset + index]
                scores = row[list(allowed)].float()
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("nonfinite teacher-forced logits")
                terms.append(scores.log_softmax(-1)[allowed.index(token)])
            return torch.stack(terms).sum()

    def reference_score_and_encoding(self, record: DecisionRecord) -> tuple[Any, tuple[float, ...]]:
        """Frozen-reference score and state encoding from one adapter-free forward.

        Equal to (score(record, reference=True), encode_state(record.prompt_ids)):
        in a causal model the hidden state at the last prompt position does not
        depend on the action tokens that follow it.
        """
        torch = self._torch
        masks = ActionTrie(record.legal_token_paths).masks(record.action_token_ids)
        if all(len(allowed) == 1 for allowed in masks):
            # A forced action has reference probability one (as in score()).
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, self.encode_state(record.prompt_ids)
        self._validate_ids(record.prompt_ids)
        if len(record.prompt_ids) + max(map(len, record.legal_token_paths)) > self.context_window:
            raise ContextWindowExceededError(
                prompt_tokens=len(record.prompt_ids),
                reserved_tokens=max(map(len, record.legal_token_paths)),
                context_window=self.context_window,
                operation="reference scoring and encoding",
            )
        ids = record.prompt_ids + record.action_token_ids[:-1]
        with torch.no_grad(), self._active_model(True) as model:
            inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
            optional = {}
            if self._logit_window.get(id(model), False):
                optional["logits_to_keep"] = len(record.action_token_ids)
            # Same unmasked call as _forward, so the two scores agree exactly.
            output = model(
                input_ids=inputs,
                output_hidden_states=True,
                use_cache=False,
                **optional,
            )
            logits = output.logits
            offset = logits.shape[1] - len(record.action_token_ids)
            if offset < 0:
                raise ValueError("teacher forcing did not return all action-token logits")
            terms = []
            for index, (token, allowed) in enumerate(
                zip(record.action_token_ids, masks, strict=True)
            ):
                scores = logits[0, offset + index][list(allowed)].float()
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("nonfinite teacher-forced logits")
                terms.append(scores.log_softmax(-1)[allowed.index(token)])
            hidden = output.hidden_states[-1][0, len(record.prompt_ids) - 1].detach().float().cpu()
        if hidden.numel() != self.encoding_dim or not bool(torch.isfinite(hidden).all()):
            raise ValueError("reference state encoding has an invalid width/value")
        return torch.stack(terms).sum(), tuple(hidden.tolist())

    def encode_state(self, prompt_ids: tuple[int, ...]) -> tuple[float, ...]:
        with self._torch.no_grad(), self._active_model(True) as model:
            output = self._forward(model, prompt_ids, hidden=True)
            hidden = output.hidden_states[-1][0, -1].detach().float().cpu()
            if hidden.numel() != self.encoding_dim or not bool(self._torch.isfinite(hidden).all()):
                raise ValueError("reference state encoding has an invalid width/value")
            return tuple(hidden.tolist())

    def _frozen_prompt_ids(self, text: str) -> tuple[int, ...]:
        # Executor instructions differ from the orchestrator's JSON-action system.
        messages = [{"role": "user", "content": text}]
        if self._has_chat_template():
            raw = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        else:
            raw = self.tokenizer.encode(text, add_special_tokens=False)
        prompt = tuple(raw)
        self._validate_ids(prompt)
        return prompt

    def frozen_prompt_tokens(self, text: str) -> int:
        """Token length of a node/author prompt exactly as ``frozen_text`` encodes it."""
        return len(self._frozen_prompt_ids(text))

    def frozen_text(
        self,
        text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        input_limit: int | None = None,
    ) -> tuple[str, int, int]:
        """Frozen executor/author generation; no actor gradients or serving logprobs."""
        torch = self._torch
        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 1
            or isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or not math.isfinite(temperature)
            or temperature <= 0
            or type(seed) is not int
            or not 0 <= seed < 2**64
            or (input_limit is not None and (type(input_limit) is not int or input_limit < 1))
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError("positive executor generation limits required")
        prompt = self._frozen_prompt_ids(text)
        if len(prompt) + max_new_tokens > self.context_window:
            raise ContextWindowExceededError(
                prompt_tokens=len(prompt),
                reserved_tokens=max_new_tokens,
                context_window=self.context_window,
                operation="frozen node/author generation",
            )
        if input_limit is not None and len(prompt) > input_limit:
            raise ValueError("node/author prompt exceeds its declared input envelope")
        generated: list[int] = []
        generator = torch.Generator(device=self.device).manual_seed(seed)
        cache, cached_length = None, 0
        with torch.no_grad(), self._active_model(True) as model:
            for _ in range(max_new_tokens):
                logits, cache, cached_length = self._next_logits(
                    model, prompt + tuple(generated), cache=cache, cached_length=cached_length
                )
                logits = logits.float()
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("nonfinite executor logits")
                token = int(
                    torch.multinomial(
                        (logits / temperature).softmax(-1), 1, generator=generator
                    ).item()
                )
                generated.append(token)
                if token == self.eos_token_id:
                    break
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        return result, len(prompt), len(generated)

    def adapter_state(self) -> dict[str, Any]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.model.named_parameters()
            if value.requires_grad
        }

    def load_adapter_state(self, state: dict[str, Any]) -> None:
        expected = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("checkpoint actor parameter identity differs")
        # Validate all tensors before the first write; malformed later entries
        # cannot leave a partially loaded adapter behind.
        for name, value in state.items():
            if (
                not isinstance(value, self._torch.Tensor)
                or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
                or not bool(self._torch.isfinite(value).all())
            ):
                raise ValueError("invalid checkpoint actor tensor")
        with self._torch.no_grad():
            for name, value in state.items():
                expected[name].copy_(value.to(expected[name]))

    @property
    def configuration_id(self) -> str:
        return str(
            stable_hash(
                {
                    "reference": self.reference_id,
                    "model_pin": self._model_pin,
                    "base_config": self._base_config_id,
                    "reference_config": self._reference_config_id,
                    "adapter_config": self._adapter_config_id,
                    "tokenizer_config": self._tokenizer_config_id,
                    "encoding_dim": self.encoding_dim,
                    "context_window": self.context_window,
                    "max_action_tokens": self.max_action_tokens,
                    "declared_context_limits": self._context_limits,
                    "context_mode": self.context_mode,
                    # @2: controller_projection renders each output version once,
                    # capped at CONTROLLER_OUTPUT_HEAD_CHARS + CONTROLLER_OUTPUT_TAIL_CHARS.
                    "projection": (
                        f"controller-single-output-head{CONTROLLER_OUTPUT_HEAD_CHARS}"
                        f"-tail{CONTROLLER_OUTPUT_TAIL_CHARS}@2"
                    )
                    if self.context_mode == "full"
                    else "public-history-head-tail@2",
                    "prompt_instruction": "action-effects-single-output-view@2",
                    "grammar": "finite-action-trie@1",
                }
            )
        )
