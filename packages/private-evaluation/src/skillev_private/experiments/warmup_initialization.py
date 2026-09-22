"""Explicit supervised F initialization for a NEW TTB run, never original Step-0.

The existing application builder loads the exported policy partition and creates
fresh Adam/projection/detector state. No warmup optimizer is exported to TTB.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from skillev.contracts import JsonValue
from skillev.policy import PolicyBackbone, PrivateInitialCheckpointBinding
from skillev.policy.checkpoint import read_policy_checkpoint_state
from skillev.policy.versions import TrainableVersions
from skillev.runtime import SkillLibraryState
from skillev.training.fresh_state import (
    FreshNamespaces,
    inspect_fresh_state,
    save_fresh_state_report,
)
from skillev.training.inflight import durable_json

if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication

WARMUP_PREPARATION_FORMAT = "skillev-private-warmup-ttb-preparation@2"
WARMUP_START_FORMAT = "warmup-initialized-application-start@1"


def initialization_condition(preparation: Path) -> dict[str, JsonValue]:
    """Add this explicit partition to scientific/diagnostic controls; old inputs add nothing."""
    value = json.loads(preparation.read_text())
    if value.get("format") != WARMUP_PREPARATION_FORMAT:
        return {}
    declaration = value.get("initialization")
    if not isinstance(declaration, dict) or declaration.get("kind") != "skill-use-warmup@1":
        raise ValueError("warmup preparation needs its explicit initialization declaration")
    return {"initialization": declaration}


def require_initialization_candidate(preparation: Path, *, sampling: dict[str, JsonValue]) -> None:
    condition = initialization_condition(preparation)
    if (
        condition
        and cast(dict[str, Any], condition["initialization"])["candidate"]["sampling"] != sampling
    ):
        raise ValueError("TTB phase/sampling condition differs from the declared warmup candidate")


def load_named(path: Path) -> dict[str, torch.Tensor]:
    values = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(values, dict)
        or not values
        or not all(
            isinstance(name, str)
            and isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and torch.isfinite(value).all()
            for name, value in values.items()
        )
    ):
        raise ValueError("independent initial parameters must be finite named CPU tensors")
    return values


def require_parameters(
    backbone: PolicyBackbone, reference: dict[str, torch.Tensor], *, forward: bool
) -> None:
    groups = backbone.parameter_groups()
    selected = (
        (*groups.forward, *groups.backward, *groups.z_head)
        if forward
        else (*groups.backward, *groups.z_head)
    )
    ids = {id(p) for p in selected}
    named = backbone.named_trainable_parameters()
    if set(named) != set(reference):
        raise ValueError("initial parameter names differ")
    for name, parameter in named.items():
        expected = reference[name]
        if id(parameter) in ids and (
            parameter.shape != expected.shape
            or parameter.dtype != expected.dtype
            or not torch.equal(parameter.detach().cpu(), expected)
        ):
            raise ValueError("initialization parameters differ from the independent preparation")


def _save_tensors(path: Path, values: dict[str, torch.Tensor]) -> None:
    with path.open("xb") as stream:
        torch.save(values, stream)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def export_initialization(
    *,
    backbone: PolicyBackbone,
    warmup_checkpoint: Path,
    source_preparation: dict[str, Any],
    base_parameters: dict[str, torch.Tensor],
    candidate: dict[str, Any],
    output_root: Path,
) -> Path:
    """Export an isolated owner loaded from a complete warmup checkpoint.

    This changes its F version/initial partition binding for NEW TTB use. Do not
    reuse this owner for further warmup; resume the saved warmup checkpoint.
    """
    if not (warmup_checkpoint / "COMPLETE").is_file():
        raise ValueError("warmup export requires a completed checkpoint")
    state = json.loads((warmup_checkpoint / "warmup.json").read_text())
    if state["updates"] < 1 or state["updates"] != len(state["history"]):
        raise ValueError("no completed supervised updates to export")
    backbone.load_checkpoint(str(warmup_checkpoint / "policy"))
    require_parameters(backbone, base_parameters, forward=False)
    current = TrainableVersions.from_backbone(backbone)
    original = TrainableVersions(**state["original_versions"])
    if current != TrainableVersions(
        f"skill-use-warmup/{state['run_id']}@{state['updates']}", original.backward, original.z
    ):
        raise ValueError("loaded owner does not match this warmup boundary")
    checkpoint_state = read_policy_checkpoint_state(warmup_checkpoint / "policy")
    if checkpoint_state.forward_version != current.forward:
        raise ValueError("warmup saved model and update counter differ")
    output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    backbone.synchronize_trainable_versions(
        TrainableVersions(
            f"warmup-initialization/{state['run_id']}@0", original.backward, original.z
        )
    )
    initial = backbone.trainable_state_identity
    backbone.bind_initial_trainable_state(initial)
    backbone.save_checkpoint(str(output_root / "initial-policy"))
    _save_tensors(
        output_root / "initial_named_parameters.pt",
        {
            name: p.detach().to(device="cpu", copy=True)
            for name, p in backbone.named_trainable_parameters().items()
        },
    )
    _save_tensors(output_root / "base_initial_named_parameters.pt", base_parameters)
    declaration = {
        "kind": "skill-use-warmup@1",
        "original_step_zero": False,
        "source_preparation": source_preparation,
        "warmup_checkpoint": str(warmup_checkpoint),
        "warmup_run_id": state["run_id"],
        "warmup_updates": state["updates"],
        "candidate": candidate,
        "application_annotations": [
            {
                "trajectory_id": d["artifact"]["record"]["trajectory_id"],
                "annotation_ref": d["annotation_ref"],
                "decision": d["decision"],
                "source": d["source"],
                **({"annotation": d["annotation"]} if "annotation" in d else {}),
            }
            for d in state["corpus"]["demonstrations"]
        ],
        "supervision_report": state.get("supervision_report"),
        "ttb_initialization": {
            "forward": "supervised",
            "backward": "original-preparation",
            "z": "original-preparation",
            "optimizer": "new-empty",
            "posterior": "new-empty",
            "evolution": "new",
            "task_cursor": 0,
            "ttb_optimizer_step": 0,
        },
    }
    preparation = output_root / "preparation.json"
    durable_json(
        preparation,
        {
            "format": WARMUP_PREPARATION_FORMAT,
            "backbone": source_preparation["backbone"],
            "initial_checkpoint": PrivateInitialCheckpointBinding(
                str(output_root / "initial-policy"), initial
            ).to_value(),
            "initialization": declaration,
        },
    )
    return preparation


def require_initial_state_report(report: dict[str, JsonValue]) -> None:
    from skillev.training.fresh_state import require_fresh_state

    if report.get("format") != WARMUP_START_FORMAT:
        require_fresh_state(report)
        return
    checks = cast(dict[str, JsonValue], report["checks"])
    if (
        report.get("state") != "verified"
        or report.get("failures")
        or report.get("unverified")
        or any(
            value is not True
            for name, value in checks.items()
            if name != "forward_default_lora_b_zero"
        )
    ):
        raise ValueError("warmup-initialized application is not a clean new TTB start")


def require_warmup_initial_application(
    application: SKILLEVApplication,
    *,
    preparation: Path,
    checkpoint_directory: Path,
    root: Path,
    initial_library: SkillLibraryState,
) -> None:
    declaration = cast(dict[str, Any], initialization_condition(preparation)["initialization"])
    if declaration["candidate"]["library"] != initial_library.to_value():
        raise ValueError("warmup initialization targets a different frozen candidate library")
    parameters = load_named(preparation.parent / "initial_named_parameters.pt")
    base = load_named(preparation.parent / "base_initial_named_parameters.pt")
    require_parameters(application.backbone, base, forward=False)
    report = inspect_fresh_state(
        application,
        initial_library=initial_library,
        namespaces=FreshNamespaces(
            request_journals=(root / "requests.sqlite3",),
            evidence_directories=(root / "inflight", root / "evidence"),
        ),
        preparation_state=read_policy_checkpoint_state(checkpoint_directory),
        initial_parameters=parameters,
    )
    # Retain the observed B-zero result, including False. It is not a requirement
    # for this explicitly supervised initialization; all NEW TTB state checks are.
    checks = cast(dict[str, JsonValue], report["checks"])
    checks["backward_z_match_original_preparation"] = True
    checks["forward_is_declared_warmup_initialization"] = str(
        cast(dict[str, Any], report["versions"])["forward"]
    ).startswith("warmup-initialization/")
    failures = [
        name
        for name, result in checks.items()
        if result is False and name != "forward_default_lora_b_zero"
    ]
    unknown = [
        name
        for name, result in checks.items()
        if result is None and name != "forward_default_lora_b_zero"
    ]
    report.update(
        format=WARMUP_START_FORMAT,
        initialization=declaration,
        failures=list(failures),
        unverified=list(unknown),
        state="rejected" if failures else "unverified" if unknown else "verified",
    )
    save_fresh_state_report(root / "fresh-application-start.json", report)
    require_initial_state_report(report)
