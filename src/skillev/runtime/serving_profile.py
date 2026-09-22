"""Read actual SGLang execution settings, not the desired launch YAML.

These are private deployment facts, not model-content attestations. A missing
measurement stays missing; hostnames, worker PIDs and throughput counters do
not participate in replica compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping

from skillev.contracts import JsonValue, normalize_json

# Differences here have changed token execution in deployed cache/chunk tests.
# Logging and capacity counters are deliberately not numerical identity fields.
_FIELDS = (
    "model_path",
    "tokenizer_path",
    "revision",
    "served_model_name",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "context_length",
    "enable_lora",
    "max_lora_rank",
    "lora_target_modules",
    "enable_deterministic_inference",
    "sampling_backend",
    "attention_backend",
    "linear_attn_backend",
    "disable_radix_cache",
    "mamba_radix_cache_strategy",
    "page_size",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "disable_overlap_schedule",
    "mamba_ssm_dtype",
    "enable_int8_mamba_checkpoint",
    "enable_tf32_matmul",
    "disable_decode_cuda_graph",
    "disable_prefill_cuda_graph",
)
_REQUIRED = (
    "model_path",
    "tokenizer_path",
    "served_model_name",
    "dtype",
    "context_length",
    "enable_lora",
    "enable_deterministic_inference",
    "sampling_backend",
)


def serving_profile(value: object) -> dict[str, JsonValue]:
    data = normalize_json(value)
    if not isinstance(data, dict):
        raise ValueError("SGLang server info must be an object")
    # Deployed 0.5.15 exposes flattened args; other releases nest server_args.
    args = data.get("server_args", data)
    if not isinstance(args, dict) or any(k not in args for k in _REQUIRED):
        raise ValueError("SGLang server info omits required execution settings")
    version = data.get("version", args.get("version"))
    if not isinstance(version, str) or not version:
        raise ValueError("SGLang server info omits its actual version")
    for name in ("model_path", "tokenizer_path", "served_model_name", "dtype"):
        if not isinstance(args[name], str) or not args[name]:
            raise ValueError(f"SGLang server info has invalid {name}")
    if type(args["context_length"]) is not int or args["context_length"] < 1:
        raise ValueError("SGLang server context length is invalid")
    return {"version": version, **{name: args.get(name) for name in _FIELDS}}


def require_same_profile(
    expected: Mapping[str, JsonValue], actual: Mapping[str, JsonValue]
) -> None:
    changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
    if changed:
        # Do not disclose private paths in a public exception/progress event.
        raise ValueError("serving execution differs in: " + ", ".join(changed))


def require_training_service(
    profile: Mapping[str, JsonValue],
    *,
    model_path: str,
    tokenizer_path: str,
    base_model: str,
    minimum_context: int,
    actor: bool,
) -> None:
    require_same_profile(
        {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "served_model_name": base_model,
        },
        {k: profile[k] for k in ("model_path", "tokenizer_path", "served_model_name")},
    )
    context = profile["context_length"]
    if type(context) is not int or context < minimum_context:
        raise ValueError("serving context cannot admit the declared input/output budget")
    if actor and (
        profile["enable_lora"] is not True
        or profile["enable_deterministic_inference"] is not True
        or profile["sampling_backend"] != "pytorch"
    ):
        raise ValueError("actor service does not support the declared LoRA/raw sampling path")
