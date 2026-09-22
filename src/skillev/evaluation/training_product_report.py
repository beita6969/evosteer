"""Describe independent weight/library axes without claiming an evolution gain."""

from .integrity_results import ExecutionControls
from .step0_integrity import InferenceArm, SkillMode


def training_product_report(arm: InferenceArm, controls: ExecutionControls) -> dict[str, object]:
    trained = arm.optimizer_steps > 0
    skills = controls.skills
    raw = skills.get("training_provenance")
    provenance = raw if isinstance(raw, dict) else {}
    raw_checkpoint = controls.model.get("checkpoint")
    checkpoint = raw_checkpoint if isinstance(raw_checkpoint, dict) else {}
    raw_policy = controls.model.get("policy")
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    library_checkpoint = provenance.get("source_checkpoint")
    policy_checkpoint = checkpoint.get(
        "runtime_checkpoint_directory", checkpoint.get("checkpoint_directory")
    )
    same_checkpoint = (
        library_checkpoint == policy_checkpoint
        if isinstance(library_checkpoint, str) and isinstance(policy_checkpoint, str)
        else None
    )
    if arm.skill_mode is SkillMode.OFF:
        condition = "skills-off-separate-control"
    elif skills.get("kind") == "initial":
        condition = "C1" if trained else "C0"
    elif skills.get("kind") == "evolved":
        condition = (
            "C2" if not trained else "C3" if same_checkpoint else "cross-or-unresolved-checkpoint"
        )
    else:
        condition = "unresolved-library-origin"
    return {
        "condition": condition,
        "weight_axis": "trained-forward" if trained else "step-0",
        "library_axis": skills.get("kind", "off"),
        "policy_id": policy.get("snapshot_id", arm.policy_id),
        "library_id": skills.get("library_id"),
        "same_training_checkpoint": same_checkpoint,
        "source_optimizer_step": skills.get("source_optimizer_step"),
        "actual_library_mutation_count": provenance.get("actual_library_mutation_count"),
        "posterior_update_count": provenance.get("posterior_update_count"),
        "claim_boundary": "provenance-not-proof-of-natural-evolution-use-or-benefit",
        "contrasts": {
            "C0:C1": "weights-at-fixed-initial-library",
            "C0:C2": "library-at-step-0",
            "C1:C3": "library-at-fixed-trained-weights",
            "C0:C3": "combined-not-single-axis",
        },
    }
