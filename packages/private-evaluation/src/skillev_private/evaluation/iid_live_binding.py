"""Forward-only live dependencies for a frozen architecture-matched IID run.

No application, trainable backbone, optimizer or training checkpoint restoration
is constructed. The selected forward adapter must already be published; this
reader does not load, replace, choose or roll back service weights.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from skillev.policy import QwenMultimodalBackboneConfig, QwenTokenizerAdapter
from skillev.rollout.external_sglang import (
    ExternalSGLangRolloutGenerator,
)
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.runtime.serving_profile import (
    require_same_profile,
    require_training_service,
    serving_profile,
)
from skillev.runtime.sglang_gateway import (
    SGLangGateway,
    SGLangGatewayConfig,
    UrllibSGLangControlTransport,
)
from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources
from skillev_private.benchmarks.mbpp_scoring import resolve_mbpp_profile
from skillev_private.benchmarks.protocol_v13_training_sessions import (
    build_protocol13_training_sessions,
    native_scorer_contracts,
)
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig
from skillev_private.experiments.formal_episode_config import formal_actor_transport

from .iid_architecture import IIDSnapshotSelection, decode_record, expanded_controls
from .iid_episode_runtime import copy_continuation_journal, run_iid_episodes
from .integrity_policies import TrainedPolicyBinding


@dataclass(frozen=True)
class NativeIIDDeployment:
    endpoint: str
    deployments: Path
    mbpp_interpreter: Path
    evalplus_source_root: Path


def observe_service(endpoint: str) -> dict[str, Any]:
    transport = UrllibSGLangControlTransport()
    observed = {}
    for name, route in (
        ("server_info", "/get_server_info"),
        ("model_info", "/get_model_info"),
        ("models", "/v1/models"),
    ):
        status, value = transport.request(
            method="GET",
            url=endpoint.rstrip("/") + route,
            payload=None,
            timeout_seconds=30.0,
            max_response_bytes=16 * 1024 * 1024,
        )
        if status != 200 or not isinstance(value, dict):
            raise RuntimeError("read-only IID service inspection failed")
        observed[name] = value
    return observed


async def run_live_iid(
    *,
    architecture: dict[str, Any],
    selection: IIDSnapshotSelection,
    deployment: NativeIIDDeployment,
    output: Path,
    chunk_size: int,
    reference: dict[str, Any] | None = None,
    allow_library_change: bool = False,
    trained_policy: dict[str, Any] | None = None,
    continuation_from: Path | None = None,
    continuation_original_request: Path | None = None,
    dependency_recovery_from: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("read-only output must be new")
    if continuation_from is not None and (
        output.resolve() == continuation_from.resolve()
        or continuation_from.resolve() in output.resolve().parents
        or output.resolve() in continuation_from.resolve().parents
    ):
        raise ValueError("continuation output must not alter the source tree")
    if dependency_recovery_from is not None and continuation_from is not None:
        raise ValueError("choose one explicit recovery mode")
    controls = copy.deepcopy(architecture["controls"])
    formal = BayesianFormalConfig(**controls["formal"])
    backbone = QwenMultimodalBackboneConfig.from_value(controls["model"]["backbone"])
    formal.require_backbone(backbone)
    observed = observe_service(deployment.endpoint)
    actual_profile = serving_profile(observed["server_info"])
    require_same_profile(controls["serving"], actual_profile)
    require_training_service(
        actual_profile,
        model_path=backbone.base_model_path,
        tokenizer_path=backbone.tokenizer_path or backbone.base_model_path,
        base_model=cast(str, actual_profile["served_model_name"]),
        minimum_context=formal.max_input_tokens
        + max(formal.maximum_reasoning_tokens, formal.max_action_tokens),
        actor=True,
    )
    if (
        actual_profile["dtype"] not in {"bfloat16", "torch.bfloat16"}
        or actual_profile["quantization"] is not None
    ):
        raise ValueError("IID actor must use the same unquantized BF16 base as formal training")
    if observed["model_info"]["model_path"] != backbone.base_model_path:
        raise ValueError("actual model differs from the frozen base")
    # The original base is a legitimate Step-0 snapshot; no fabricated trained
    # adapter or optimizer-step metadata is accepted in its place.
    gateway = None
    binding = None
    if trained_policy is not None:
        binding = TrainedPolicyBinding.read(
            selection.policy.snapshot_id,
            trained_policy,
            observed=[observed],
            tokenizer_id=backbone.tokenizer_id,
        )
        if (
            binding.metadata["optimizer_steps"] != selection.optimizer_step
            or binding.metadata["forward_version"] != selection.policy.forward_adapter_version
            or binding.metadata["backbone_id"] != selection.policy.backbone_id
        ):
            raise ValueError("selected forward checkpoint and policy snapshot disagree")
        gateway = SGLangGateway(
            SGLangGatewayConfig(
                endpoint_base=deployment.endpoint,
                base_model=cast(str, actual_profile["served_model_name"]),
                supervisor_adapter=binding.adapter_name,
                control_retries=0,
            )
        )
        gateway.bind_serving_profile(actual_profile)
        gateway.bind_existing_supervisor_adapter(
            adapter_revision=selection.policy.forward_adapter_version
        )
    elif selection.optimizer_step or selection.policy.forward_adapter_version != "adapter-free":
        raise ValueError("a non-base policy must be bound to its real published forward checkpoint")
    tokenizer = QwenTokenizerAdapter.from_config(backbone)
    if selection.policy.tokenizer_id != tokenizer.tokenizer_id:
        raise ValueError("selected policy uses another tokenizer")
    workflow = RolloutWorkflowBinding.from_value(controls["workflow"])
    resources = RolloutWorkflowResources(workflow)
    mbpp = resolve_mbpp_profile(
        {
            "source_root": str(deployment.evalplus_source_root),
            "profile": controls["scorers"]["mbpp-plus"],
        }
    )
    if (
        native_scorer_contracts(mbpp, formal.healthbench_judge, domains=formal.domains)
        != controls["scorers"]
    ):
        raise ValueError("actual native scorer catalog differs from frozen IID controls")
    environments = controls["environments"]
    if json.loads(deployment.deployments.read_text()) != environments["deployments"]:
        raise ValueError("native evaluator deployment controls changed")
    alf_config = Path(environments["deployments"]["alfworld"]["config_path"])
    if alf_config.read_text() != environments["alfworld_config_text"]:
        raise ValueError("ALFWorld execution configuration changed")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    journal = output.parent / f"{output.name}-evaluation-requests.sqlite3"
    if continuation_from is not None:
        copy_continuation_journal(continuation_from, journal)
    request_journal: DurableRequestJournal
    if dependency_recovery_from is not None:
        from .iid_dependency_recovery import prepare_dependency_recovery

        request_journal = prepare_dependency_recovery(
            dependency_recovery_from,
            output,
            expanded=expanded_controls(architecture, selection),
            chunk_size=chunk_size,
        )
    else:
        request_journal = DurableRequestJournal(journal)
    _, sessions = await build_protocol13_training_sessions(
        tuple(decode_record(row) for row in architecture["panel"]["records"]),
        deployments_path=deployment.deployments,
        endpoint_base=deployment.endpoint,
        base_model=cast(str, actual_profile["served_model_name"]),
        resources=resources,
        mbpp_interpreter=deployment.mbpp_interpreter,
        mbpp_source_root=deployment.evalplus_source_root,
        mbpp_profile=mbpp,
        hotpot_deliberation=formal.hotpot_deliberation,
        rollout_budget=formal.task_budget,
        static_rollout_budget=formal.static_task_budget,
        domain_rollout_budgets=formal.domain_task_budgets,
        lazy_environments=True,
        request_journal_path=journal,
        healthbench_judge=formal.healthbench_judge,
    )
    transport = formal_actor_transport(
        deployment.endpoint, worker_threads=workflow.transport_worker_threads
    )
    if {k: v for k, v in transport.to_value().items() if k != "endpoint_base"} != controls[
        "actor_transport"
    ]:
        raise ValueError("actual actor transport differs from frozen formal execution")
    generator = ExternalSGLangRolloutGenerator(
        config=transport,
        tokenizer=tokenizer,
        gateway=gateway,
        snapshot_provider=lambda: selection.policy,
        request_journal=request_journal,
    )

    def validate_execution() -> None:
        after = observe_service(deployment.endpoint)
        require_same_profile(actual_profile, serving_profile(after["server_info"]))
        if binding is not None:
            binding.validate([after])
        if after["model_info"]["model_path"] != backbone.base_model_path:
            raise ValueError("IID actor base changed during the read-only collection")

    try:
        return await run_iid_episodes(
            architecture=architecture,
            selection=selection,
            generator=generator,
            base_sessions=sessions,
            actual_controls=controls,
            output=output,
            chunk_size=chunk_size,
            resources=resources,
            reference=reference,
            allow_library_change=allow_library_change,
            validate_execution=validate_execution,
            continuation_from=continuation_from,
            continuation_original_request=continuation_original_request,
        )
    finally:
        generator.close()
