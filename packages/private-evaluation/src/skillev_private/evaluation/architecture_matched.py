"""Read-only policy/library substitution in a frozen OOD evaluator, never a trainer.

Architecture bundles and expanded controls contain private benchmark material.
They belong beside evaluation artifacts, not in Git or a training evidence store.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from skillev.evaluation.integrity_pipeline import FrozenPanel
from skillev.evaluation.integrity_results import ExecutionControls, require_paired_controls
from skillev.evaluation.step0_integrity import InferenceArm, SkillMode, decode_integrity_arm

FORMAT = "ood-architecture-matched@1"

if TYPE_CHECKING:
    from .integrity_runtime import PrivateIntegrityRuntime


def evaluation_isolation() -> dict[str, object]:
    """The forward-only evaluator has no optimizer, posterior or evidence writer."""
    return {
        "mode": "read-only-policy-snapshot-substitution",
        "training_updates": 0,
        "posterior_updates": 0,
        "skill_evolution": 0,
        "training_evidence_writes": 0,
        "training_batch_membership": False,
        "ttb_loss_contribution": False,
        "backward_or_z_execution": False,
    }


def freeze_architecture(base: dict[str, Any], destination: Path, architecture_id: str) -> Path:
    """Copy the selected input/scorer assets once; do not sample or read any results."""
    if base.get("catalog") != "ood" or len(base["arms"]) != 1 or not architecture_id.strip():
        raise ValueError("an OOD architecture needs one explicit single-owner arm")
    arm = decode_integrity_arm(base["arms"][0])
    arm.validate_live_topology()
    if arm.optimizer_steps or arm.skill_mode not in {SkillMode.OFF, SkillMode.LIBRARY}:
        raise ValueError("freeze Step-0 with skills off or an explicit read-only library")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    config = copy.deepcopy(base)
    for name, source in config["ood_sources"].items():
        target = destination / f"{name}-private.jsonl"
        target.write_bytes(Path(source).read_bytes())
        config["ood_sources"][name] = str(target.resolve())
    for name, settings in config["scorers"].items():
        if "judgements_path" in settings:
            raise ValueError("fresh owner generation cannot reuse historical judge outcomes")
        settings.pop("cache_directory", None)
        for field in ("checker_path", "template_path"):
            if field in settings:
                target = destination / f"{name}-{field}{Path(settings[field]).suffix}"
                target.write_bytes(Path(settings[field]).read_bytes())
                settings[field] = str(target.resolve())
    config["architecture_id"] = architecture_id
    config["declared_condition"] = FORMAT
    path = destination / "architecture-private.json"
    path.write_text(json.dumps({"format": FORMAT, "runtime_config": config}, indent=2) + "\n")
    return path


def substitute_snapshot(
    architecture: dict[str, Any], selection: dict[str, Any], private_output: Path
) -> dict[str, Any]:
    """Only policy and library state may vary; no runtime override dictionary exists."""
    if architecture["format"] != FORMAT:
        raise ValueError("unsupported OOD architecture bundle")
    config = cast(dict[str, Any], copy.deepcopy(architecture["runtime_config"]))
    arm = decode_integrity_arm(config["arms"][0])
    unknown = selection.keys() - {
        "arm_id",
        "policy_id",
        "optimizer_steps",
        "policy",
        "library_id",
        "library",
    }
    if unknown:
        raise ValueError("snapshot selection cannot override the evaluation architecture")
    changes: dict[str, Any] = {"arm_id": selection["arm_id"]}
    config["trained_policies"] = {}
    if selection.get("policy_id") is not None:
        changes.update(
            policy_id=selection["policy_id"], optimizer_steps=selection["optimizer_steps"]
        )
        config["trained_policies"] = {selection["policy_id"]: copy.deepcopy(selection["policy"])}
    elif selection.get("optimizer_steps", 0) or selection.get("policy"):
        raise ValueError("Step-0 cannot carry a trained checkpoint or nonzero update count")
    if "library_id" in selection or "library" in selection:
        if arm.skill_mode is not SkillMode.LIBRARY:
            raise ValueError("skills-off to library-on is a different architecture condition")
        settings = copy.deepcopy(selection["library"])
        if settings["retrieval_rule"] != arm.skill_retrieval_rule:
            raise ValueError("library state substitution cannot change its discovery interface")
        changes["skill_library_id"] = selection["library_id"]
        config["skill_libraries"] = {selection["library_id"]: settings}
    config["arms"] = [replace(arm, **changes).to_value()]
    for name in ("omni-math", "livemedbench"):
        if name in config["scorers"]:
            config["scorers"][name]["cache_directory"] = str(private_output / f"judge-{name}")
    return config


def _comparison_controls(controls: ExecutionControls) -> ExecutionControls:
    """Storage paths / registered adapter state are not decoding or scoring parameters."""
    service = cast(dict[str, Any], copy.deepcopy(controls.service))
    service.pop("registered_models", None)
    for arguments in service.get("actual_server_arguments", []):
        arguments.pop("lora_paths", None)
    evaluator = cast(dict[str, Any], copy.deepcopy(controls.evaluator))
    for settings in evaluator.get("settings", {}).values():
        settings.pop("cache_directory", None)
    return replace(controls, service=service, evaluator=evaluator)


def require_architecture_match(
    reference: ExecutionControls,
    selected: ExecutionControls,
    reference_arm: InferenceArm,
    selected_arm: InferenceArm,
) -> tuple[str, ...]:
    """Compare actual components, including ordered public input and private scorer targets.

    For a combined weight/library contrast, validate the two axes separately. Its
    result is labelled combined, never attributed solely to weight learning.
    """
    left, right = _comparison_controls(reference), _comparison_controls(selected)
    middle_arm = replace(
        selected_arm,
        arm_id="architecture-intermediate-not-executed",
        skill_library_id=reference_arm.skill_library_id,
    )
    middle = replace(right, skills=left.skills)
    axes = []
    if (reference_arm.policy_id, reference_arm.optimizer_steps) != (
        selected_arm.policy_id,
        selected_arm.optimizer_steps,
    ):
        axes.append(require_paired_controls(left, middle, reference_arm, middle_arm))
    elif left != middle or replace(reference_arm, arm_id=middle_arm.arm_id) != middle_arm:
        raise ValueError("non-snapshot evaluation controls changed")
    if reference_arm.skill_library_id != selected_arm.skill_library_id:
        axes.append(require_paired_controls(middle, right, middle_arm, selected_arm))
    elif middle.skills != right.skills:
        raise ValueError("a library snapshot changed without a new explicit identity")
    return tuple(axes)


def create_runtime(
    path: Path, *, private_output: Path
) -> tuple[PrivateIntegrityRuntime, FrozenPanel, tuple[InferenceArm, ...]]:
    # Import only the forward evaluation runtime. No trainer, RuntimeSnapshot.restore,
    # posterior updater, W&B callback or training evidence store is constructed.
    from .integrity_runtime import PrivateIntegrityRuntime
    from .integrity_sources import order_source_panel
    from .ood_sources import load_ood_panel

    request = json.loads(path.read_text())
    architecture = json.loads(Path(request["architecture_file"]).read_text())
    config = substitute_snapshot(architecture, request["snapshot"], private_output)
    source = load_ood_panel(config)
    if "benchmark_order" in config:
        source = order_source_panel(source, tuple(config["benchmark_order"]))
    arms = tuple(decode_integrity_arm(value) for value in config["arms"])
    runtime = PrivateIntegrityRuntime(config, source, private_output)
    try:
        controls = runtime.controls(arms[0], source.panel.entries)
        reference_path = request.get("reference_controls")
        if reference_path:
            reference = json.loads(Path(reference_path).read_text())
            raw_controls = reference["controls"]
            raw_controls["public_inputs"] = tuple(
                tuple(row) for row in raw_controls["public_inputs"]
            )
            # Round-trip JSON converts tuple-valued tokenizer/serving fields too.
            current = asdict(controls)
            current = json.loads(json.dumps(current))
            current["public_inputs"] = tuple(tuple(row) for row in current["public_inputs"])
            axes = require_architecture_match(
                ExecutionControls(**raw_controls),
                ExecutionControls(**current),
                decode_integrity_arm(reference["arm"]),
                arms[0],
            )
        elif arms[0].optimizer_steps or request["snapshot"].get("library_id"):
            raise ValueError("snapshot substitution needs the Step-0 expanded reference controls")
        else:
            axes = ()
        record = {
            "format": FORMAT,
            "architecture_id": config["architecture_id"],
            "isolation": evaluation_isolation(),
            "snapshot_axes_changed": axes,
            "arm": arms[0].to_value(),
            "controls": asdict(controls),
        }
        output = private_output / "architecture-controls-private.json"
        if output.exists():
            if json.loads(output.read_text()) != json.loads(json.dumps(record)):
                raise ValueError("same-run evaluation cannot change its architecture or snapshot")
        else:
            output.write_text(json.dumps(record, indent=2) + "\n")
    except BaseException:
        runtime.journal.close()
        raise
    return runtime, source.panel, arms
