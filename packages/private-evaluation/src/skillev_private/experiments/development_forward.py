"""Read-only binding to an already published fresh/warmup forward initialization.

No model load, adapter publication, optimizer, or training state is constructed.
The owner publishes the exact adapter first; the existing gateway checks its
presence and pins the declared revision. This is not a weights attestation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skillev.policy.checkpoint import read_policy_checkpoint_state
from skillev.rollout import PolicySnapshot
from skillev.runtime.sglang_gateway import SGLangGateway, SGLangGatewayConfig

FORMAT = "development-forward-initialization@1"


def resolve_forward(request: dict[str, Any]) -> tuple[PolicySnapshot, dict[str, Any]]:
    """Read saved initialization metadata before service inspection or hydration."""
    binding = request.get("forward_initialization")
    if binding is None:
        policy = PolicySnapshot.from_value(json.loads(Path(request["step0_policy"]).read_text()))
        if policy.forward_adapter_version != "adapter-free":
            raise ValueError("non-base development requires explicit forward_initialization")
        return policy, {}
    from .development_ttb import FORMAT as TTB_FORMAT
    from .development_ttb import resolve_ttb

    if isinstance(binding, dict) and binding.get("format") == TTB_FORMAT:
        if "actor_endpoints" in request or "step0_policy" in request:
            raise ValueError("TTB checkpoint validation uses one explicit versioned actor route")
        return resolve_ttb(binding)
    if not isinstance(binding, dict) or binding.get("format") != FORMAT:
        raise ValueError("unsupported development forward initialization")
    if "actor_endpoints" in request:
        raise ValueError("forward-initialized development currently requires a single endpoint")
    if "step0_policy" in request:
        raise ValueError("forward initialization must not also declare an adapter-free Step-0")
    if binding.get("kind") not in {"fresh-forward", "skill-use-warmup"} or not all(
        isinstance(binding.get(k), str) and binding[k].strip()
        for k in ("preparation", "policy", "published_forward_adapter")
    ):
        raise ValueError("explicit initialization kind, files and published adapter are required")

    from .bayesian_training_setup import _read_preparation
    from .warmup_initialization import initialization_condition

    preparation = Path(binding["preparation"])
    config, checkpoint = _read_preparation(preparation)
    state = read_policy_checkpoint_state(Path(checkpoint.directory))
    initialization = initialization_condition(preparation)
    warmup = bool(initialization)
    if warmup != (binding["kind"] == "skill-use-warmup"):
        raise ValueError("declared starting kind differs from the actual preparation")
    if state.optimizer_step != 0 or state.trainable_state != checkpoint.trainable_state:
        raise ValueError("development requires the preparation's initial policy partition")
    if state.forward_version == "adapter-free" or (
        state.forward_version.startswith("warmup-initialization/") != warmup
    ):
        raise ValueError("forward revision differs from the declared initialization kind")
    policy = PolicySnapshot.from_value(json.loads(Path(binding["policy"]).read_text()))
    expected = PolicySnapshot.create(
        backbone_id=state.backbone_id,
        forward_adapter_version=state.forward_version,
        tokenizer_id=config.tokenizer_id,
        backend_id="sglang-native-exact-token",
        initial_trainable_state_hash=state.trainable_state.content_hash,
    )
    if policy != expected:
        raise ValueError("saved development policy differs from the actual preparation snapshot")
    return policy, {
        **binding,
        "preparation_value": json.loads(preparation.read_text()),
        "checkpoint_state": state.to_value(),
        "backbone": config.to_value(),
        **initialization,
        "original_step_zero": not warmup,
        "model_loaded_by_collector": False,
        "adapter_publication": "owner-prepublished-existing-gateway-binding",
        "training_state_writes": False,
    }


def require_forward_candidate(
    declaration: dict[str, Any],
    *,
    backbone: dict[str, Any],
    sampling: dict[str, Any],
    library: dict[str, Any],
) -> None:
    if not declaration:
        return
    expected = dict(declaration["backbone"])
    expected["device"] = backbone["device"]  # CPU initialization may serve on CUDA.
    if expected != backbone:
        raise ValueError("readonly actor backbone differs from the initialization")
    if declaration["kind"] == "skill-use-warmup":
        candidate = declaration["initialization"]["candidate"]
        if candidate["sampling"] != sampling or candidate["library"] != library:
            raise ValueError("readonly phase/sampling/library differs from the warmup candidate")
    elif declaration["kind"] == "ttb-checkpoint":
        if declaration["sampling"] != sampling:
            raise ValueError("readonly phase/sampling differs from the source TTB condition")
        if library not in (declaration["initial_library"], declaration["checkpoint_library"]):
            raise ValueError("declare an actual frozen initial/current library for the policy axis")


def bind_forward_gateway(
    declaration: dict[str, Any], *, endpoint: str, base_model: str, policy: PolicySnapshot
) -> SGLangGateway | None:
    if not declaration:
        return None
    gateway = SGLangGateway(
        SGLangGatewayConfig(
            endpoint_base=endpoint,
            base_model=base_model,
            supervisor_adapter=declaration["published_forward_adapter"],
            control_retries=0,
        )
    )
    gateway.bind_existing_supervisor_adapter(adapter_revision=policy.forward_adapter_version)
    fixed_snapshot(gateway, policy)
    return gateway


def fixed_snapshot(gateway: SGLangGateway | None, policy: PolicySnapshot) -> PolicySnapshot:
    if gateway is not None and (
        gateway.adapter_generation.adapter_revision != policy.forward_adapter_version
        or gateway.adapter_generation.adapter_name != gateway.config.supervisor_adapter
    ):
        raise ValueError("readonly forward route changed from the fixed policy")
    return policy
