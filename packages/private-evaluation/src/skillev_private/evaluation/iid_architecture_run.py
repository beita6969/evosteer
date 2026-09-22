"""Private CLI: freeze native IID architecture, then execute a snapshot read-only.

All source files, targets, traces and per-question results remain under private
paths. This command never launches a GPU service or a training process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

from skillev.contracts import normalize_json
from skillev.policy import QwenMultimodalBackboneConfig
from skillev.rollout import PolicySnapshot
from skillev.runtime import SkillLibraryState
from skillev.runtime.serving_profile import require_training_service, serving_profile
from skillev.training.inflight import durable_json
from skillev.training.performance_config import TrainingPerformanceConfig
from skillev_private.benchmarks.mbpp_scoring import resolve_mbpp_profile
from skillev_private.benchmarks.protocol_v13_training_sessions import native_scorer_contracts
from skillev_private.experiments.fresh_restart import load_fresh_config

from .iid_architecture import IIDSnapshotSelection, freeze_iid_architecture
from .iid_episode_sources import source_identities_from_config
from .iid_live_binding import NativeIIDDeployment, observe_service, run_live_iid
from .integrity_sources import load_source_panel, select_source_panel


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("IID request/control file must be a JSON object")
    return value


def freeze(request: dict[str, Any], destination: Path) -> Path:
    formal = load_fresh_config(Path(request["formal_config"]))
    source_config = _read(request["source_config"])
    if source_config.get("canary"):
        raise ValueError("a canary cannot replace the frozen IID Step-0 population")
    source = load_source_panel(source_config)
    source = select_source_panel(source, source_config["evaluation_sample_counts"])
    identities = source_identities_from_config(source_config, source)
    excluded = {}
    for purpose, path in request["excluded_source_files"].items():
        rows = json.loads(Path(path).read_text())
        excluded[purpose] = frozenset(
            (row["benchmark_id"], row["source_question_id"]) for row in rows
        )
    # Explicit even when empty: the earlier released IID panels were repeatedly
    # evaluated. A fresh optimizer does not erase this project exposure.
    prior_evaluation = frozenset(
        (row["benchmark_id"], row["source_question_id"])
        for row in json.loads(Path(request["prior_evaluation_sources_file"]).read_text())
    )
    library = SkillLibraryState.from_value(_read(request["initial_library"]))
    backbone = QwenMultimodalBackboneConfig.from_value(_read(request["backbone"]))
    formal.require_backbone(backbone)
    deployment = NativeIIDDeployment(
        request["endpoint"],
        Path(request["deployments"]),
        Path(request["mbpp_interpreter"]),
        Path(request["evalplus_source_root"]),
    )
    native_deployments = _read(deployment.deployments)
    observed = observe_service(deployment.endpoint)
    profile = serving_profile(observed["server_info"])
    if observed["model_info"]["model_path"] != backbone.base_model_path:
        raise ValueError("frozen model configuration is not the actual deployed base")
    require_training_service(
        profile,
        model_path=backbone.base_model_path,
        tokenizer_path=backbone.tokenizer_path or backbone.base_model_path,
        base_model=cast(str, profile["served_model_name"]),
        minimum_context=formal.max_input_tokens
        + max(formal.maximum_reasoning_tokens, formal.max_action_tokens),
        actor=True,
    )
    if (
        profile["dtype"] not in {"bfloat16", "torch.bfloat16"}
        or profile["quantization"] is not None
    ):
        raise ValueError("freeze requires the actual unquantized BF16 training-compatible service")
    mbpp = resolve_mbpp_profile(
        {
            "source_root": str(deployment.evalplus_source_root),
            "profile": request["mbpp_profile"],
        }
    )
    performance = TrainingPerformanceConfig.load(formal.performance_profile)
    policy = PolicySnapshot.from_value(_read(request["step0_policy"]))
    if (
        policy.forward_adapter_version != "adapter-free"
        or policy.tokenizer_id != backbone.tokenizer_id
    ):
        raise ValueError(
            "this baseline requires the actual adapter-free Step-0 and frozen tokenizer"
        )
    path = freeze_iid_architecture(
        destination=destination,
        architecture_id=request["architecture_id"],
        source=source,
        identities=identities,
        excluded_sources=excluded,
        formal=formal,
        initial_library=library,
        model_controls={"backbone": backbone.to_value()},
        scorer_controls=native_scorer_contracts(
            mbpp, formal.healthbench_judge, domains=formal.domains
        ),
        environment_controls={
            "deployments": native_deployments,
            "alfworld_config_text": Path(native_deployments["alfworld"]["config_path"]).read_text(),
        },
        serving_controls=cast(dict[str, Any], profile),
        workflow=performance.workflow(),
        acceptance_rules=_read(request["acceptance_rules"]),
        prior_evaluation_sources=prior_evaluation,
        source_aliases=_read(request["source_aliases_file"]),
    )
    durable_json(destination / "step0-policy.json", policy.to_value())
    durable_json(
        destination / "deployment-private.json",
        normalize_json(
            {
                "endpoint": deployment.endpoint,
                "deployments": str(deployment.deployments),
                "mbpp_interpreter": str(deployment.mbpp_interpreter),
                "evalplus_source_root": str(deployment.evalplus_source_root),
            }
        ),
    )
    return path


async def run(request: dict[str, Any], destination: Path) -> dict[str, Any]:
    permitted = {
        "architecture_file",
        "arm",
        "policy_file",
        "optimizer_step",
        "library_file",
        "reference_controls",
        "allow_library_change",
        "trained_policy",
        "chunk_size",
        "continuation_from",
        "continuation_original_request",
        "dependency_recovery_from",
    }
    if request.keys() - permitted:
        raise ValueError("snapshot execution cannot override architecture controls")
    architecture_path = Path(request["architecture_file"])
    architecture = _read(architecture_path)
    location = _read(architecture_path.parent / "deployment-private.json")
    deployment = NativeIIDDeployment(
        location["endpoint"],
        Path(location["deployments"]),
        Path(location["mbpp_interpreter"]),
        Path(location["evalplus_source_root"]),
    )
    selection = IIDSnapshotSelection(
        request["arm"],
        PolicySnapshot.from_value(
            _read(request.get("policy_file", architecture_path.parent / "step0-policy.json"))
        ),
        request.get("optimizer_step", 0),
        None
        if "library_file" not in request
        else SkillLibraryState.from_value(_read(request["library_file"])),
    )
    return await run_live_iid(
        architecture=architecture,
        selection=selection,
        deployment=deployment,
        output=destination,
        chunk_size=request["chunk_size"],
        reference=None
        if "reference_controls" not in request
        else _read(request["reference_controls"]),
        allow_library_change=request.get("allow_library_change", False),
        trained_policy=request.get("trained_policy"),
        continuation_from=None
        if "continuation_from" not in request
        else Path(request["continuation_from"]).resolve(),
        continuation_original_request=None
        if "continuation_original_request" not in request
        else Path(request["continuation_original_request"]).resolve(),
        dependency_recovery_from=None
        if "dependency_recovery_from" not in request
        else Path(request["dependency_recovery_from"]).resolve(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "run"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = _read(args.request)
    if args.operation == "freeze":
        print(freeze(request, args.output))
    else:
        summary = asyncio.run(run(request, args.output))
        print(
            json.dumps(summary, allow_nan=False)
        )  # Aggregate only; private examples stay on disk.


if __name__ == "__main__":
    main()
