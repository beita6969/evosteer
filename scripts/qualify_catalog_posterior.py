"""Isolated development posterior from an actual saved catalog-read episode.

No generation, optimizer, IID input or formal counter is created. The original
controlled-submission-presence-only label is NOT task correctness or usefulness.
Run `score` once, then `restore` in another process; failed evidence is retained.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

from scripts.qualify_catalog_authoring import PURPOSE, clock, development_task, preserve, read
from skillev.calibration import CalibrationConfig, CalibrationEngine
from skillev.contracts import TTBBatchStats
from skillev.diagnostics import DiagnosticsConfig
from skillev.policy import AdapterRole, build_qwen_policy_backbone
from skillev.policy.checkpoint import read_policy_checkpoint_state
from skillev.rollout import PolicySnapshot, RolloutArtifact
from skillev.runtime import SkillLibrary, SkillLibraryState
from skillev.scoring import (
    ScoringConfig,
    materialize_edge_records,
    materialize_residual,
    score_trajectory,
)
from skillev.training.projections import (
    FullProjectionRuntimeState,
    MethodProjectionPipeline,
    TrainingStepSource,
)

FORMAT = "controlled-catalog-development-posterior@1"
LABEL_SCOPE = "controlled-submission-presence-only; NOT task correctness or skill usefulness"


class ReadOnlyScores:
    """Drop each model graph, retaining only small leaves for the existing objective.

    The shared objective requires differentiable scalars even for materialization.
    These detached leaves are never backpropagated. No backbone gradient, optimizer
    state, scoring formula or encoded model input is modified.
    """

    def __init__(self, backbone, output):
        self.backbone, self.output = backbone, output
        self.tokenizer = backbone.tokenizer
        self.calls = []

    def score(self, prefix_ids, action_ids, role):
        started = time.monotonic()
        with torch.no_grad():
            values = self.backbone.score(prefix_ids, action_ids, role)
        self.calls.append(
            {
                "role": role.value,
                "prefix_tokens": len(prefix_ids),
                "action_tokens": len(action_ids),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        preserve(self.output / f"score-call-{len(self.calls):03d}.json", self.calls[-1])
        return values.detach().requires_grad_(True)

    def z_value(self, query_ids):
        with torch.no_grad():
            value = self.backbone.z_value(query_ids)
        return value.detach().requires_grad_(True)


def score_source(
    backbone, artifact, output, *, temperature_beta, forward_version, backward_version
):
    require_scoring_identity(artifact, forward_version)
    wrapper = ReadOnlyScores(backbone, output)
    score = score_trajectory(
        wrapper, artifact.record, artifact.initial_context.text, ScoringConfig(temperature_beta)
    )
    residual = materialize_residual(score, raw_reward=artifact.record.reward.value)
    snapshot = artifact.manifest.policy_snapshot
    batch = "controlled-development-posterior-1"
    edges = materialize_edge_records(
        score,
        forward_adapter_version=forward_version,
        backward_adapter_version=backward_version,
        batch_id=batch,
        policy_snapshot_id=snapshot.snapshot_id,
        library_version=artifact.manifest.library_version,
        action_token_counts=tuple(s.action_token_count for s in artifact.record.steps),
        action_token_ids=tuple(s.action_token_ids for s in artifact.record.steps),
    )
    stats = TTBBatchStats(
        batch_id=batch,
        optimizer_step=1,
        library_version=artifact.manifest.library_version,
        residuals=(residual,),
        batch_loss=(residual.delta / residual.horizon) ** 2,
        mean_reward=artifact.record.reward.value,
        created_at=clock(),
    )
    return TrainingStepSource(batch, 1, (artifact,), stats, edges)


def project(source, skill_id, output, *, diagnostics, calibration):
    pipeline = MethodProjectionPipeline.fresh(
        diagnostics_config=diagnostics,
        calibration=CalibrationEngine(calibration),
        library_version=source.stats.library_version,
    )
    if calibration.alpha_0 != 1 or calibration.beta_0 != 1:
        raise ValueError("this declared first-event qualification requires Beta(1,1)")
    preserve(output / "projection-before.json", pipeline.runtime_state().to_value())
    transition = pipeline.preview(source)
    events = [u for u in transition.posterior_batch.updates if u.skill_id == skill_id]
    if not events:
        raise ValueError("saved actual episode provides no new-skill posterior event")
    first_by_cell = {}
    for event in events:
        first_by_cell.setdefault(event.z.cell_key(event.skill_id), event)
    if any(u.alpha_before != 1 or u.beta_count_before != 1 for u in first_by_cell.values()):
        raise ValueError("new skill does not start from the declared unobserved prior")
    pipeline.commit(transition)
    preserve(output / "training-step-source.json", source.to_value())
    preserve(output / "posterior-batch.json", transition.posterior_batch.to_value())
    preserve(output / "projection-after.json", pipeline.runtime_state().to_value())
    result = {
        "format": FORMAT,
        "driver_pid": os.getpid(),
        "label_scope": LABEL_SCOPE,
        "scoring_scope": "controlled offline development; NOT formal on-policy training",
        "teacher_forced_edge_direction_calls": 2 * len(source.edge_records),
        "formal_optimizer_steps": 0,
        "actual_optimizer_steps": 0,
        "formal_posterior_events": 0,
        "development_projection_ordinal": 1,
        "source_optimizer_step_field_scope": "local projection ordinal, NOT an optimizer step",
        "development_new_skill_event_count": len(events),
        "new_skill_id": skill_id,
        "library_version": source.stats.library_version,
        "new_skill_cells": [
            c.to_value() for c in pipeline.calibration_cells if c.skill_id == skill_id
        ],
        "diagnostics": diagnostics.to_value(),
        "calibration": calibration.to_value(),
        "uncovered": [
            "natural trigger",
            "task correctness",
            "causal usefulness",
            "formal training",
        ],
    }
    preserve(output / "result.json", result)
    return result


def require_development(root, artifact):
    plan, mutation = read(root / "plan.json"), read(root / "mutation.json")
    result = read(root / "read-result.json")
    library = SkillLibrary(SkillLibraryState.from_value(read(root / "mutated-library.json")))
    task = development_task()
    if plan.get("purpose") != PURPOSE or artifact.manifest.task_id != task.task_id:
        raise ValueError("only the frozen public development episode is accepted")
    if artifact.record.initial_context.query != task.query:
        raise ValueError("development task text differs from the frozen public task")
    if (
        not result.get("qualified_catalog_read_visibility")
        or artifact.manifest.library_version != library.current_version
        or mutation["library_version_after"] != library.current_version
    ):
        raise ValueError("actual read qualification or mutated library is inconsistent")
    skills = mutation["new_skill_ids"]
    if len(skills) != 1 or skills[0] not in library.active_skill_ids:
        raise ValueError("expected exactly the accepted Generate document")
    if not any(skills[0] in s.invoked_skill_ids for s in artifact.record.steps):
        raise ValueError("no actual recorded new-skill invocation")
    reward = artifact.record.reward
    if (
        reward.native_metric_name != "controlled-submission-presence-only"
        or reward.verifier_version != PURPOSE
        or reward.native_payload.get("purpose") != PURPOSE
    ):
        raise ValueError("unknown development terminal label provenance")
    return plan, skills[0]


def require_scoring_identity(artifact, forward_version):
    if artifact.manifest.policy_snapshot.forward_adapter_version != forward_version:
        raise ValueError(
            "original rollout and actual scorer forward versions differ; "
            "collect a controlled read under the fresh F0 identity, never relabel adapter-free"
        )


def require_initial_parameters(backbone, reference):
    actual = backbone.named_trainable_parameters()
    if set(actual) != set(reference) or not actual:
        raise ValueError("fresh independent parameter mapping is incomplete")
    for name, value in actual.items():
        value = value.detach().cpu()
        if value.dtype != reference[name].dtype or not torch.equal(value, reference[name]):
            raise ValueError("loaded preparation differs from independent initial parameters")
        if not torch.isfinite(value).all():
            raise ValueError("initial parameter is not finite")
    for role in (AdapterRole.FORWARD_POLICY, AdapterRole.BACKWARD_POLICY):
        a = [v for k, v in reference.items() if f".{role.value}." in k and ".lora_A." in k]
        b = [v for k, v in reference.items() if f".{role.value}." in k and ".lora_B." in k]
        if not a or len(a) != len(b) or any(torch.count_nonzero(v) for v in b):
            raise ValueError("fresh adapter is not verified base-equivalent LoRA")
        if any(not torch.count_nonzero(v) for v in a):
            raise ValueError("fresh LoRA A is unexpectedly zero")


def score_run(root, output, preparation, temperature_beta):
    from skillev_private.experiments.bayesian_training_setup import _read_preparation

    if output == root or root in output.parents or output in root.parents:
        raise ValueError("posterior evidence requires a separate development directory")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    preserve(
        output / "started.json", {"format": FORMAT, "driver_pid": os.getpid(), "source": str(root)}
    )
    try:
        config, binding = _read_preparation(preparation)
        plan = read(root / "plan.json")
        expected = dict(plan["backbone"])
        expected["device"] = config.device
        if config.to_value() != expected:
            raise ValueError("scoring and collected base declarations differ")
        if config.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("explicit selected GPU visibility is required")
        state = read_policy_checkpoint_state(Path(binding.directory))
        if state.optimizer_step != 0:
            raise ValueError("development requires genuinely fresh Step-0 preparation")
        raw_artifact = read(root / "read-artifact.json")
        original_policy = PolicySnapshot.from_value(raw_artifact["manifest"]["policy_snapshot"])
        if original_policy.forward_adapter_version != state.forward_version:
            raise ValueError(
                "original rollout/scorer forward identities differ; no model loaded; "
                "a fresh-F0 controlled read is required, not relabelled adapter-free evidence"
            )
        torch.manual_seed(0)
        backbone = build_qwen_policy_backbone(config)
        backbone.load_checkpoint(binding.directory)
        reference = torch.load(
            preparation.parent / "initial_named_parameters.pt",
            map_location="cpu",
            weights_only=True,
        )
        require_initial_parameters(backbone, reference)
        artifact = RolloutArtifact.from_value(raw_artifact, tokenizer=backbone.tokenizer)
        plan, skill_id = require_development(root, artifact)
        require_scoring_identity(artifact, state.forward_version)
        for name in (
            "read-artifact.json",
            "mutated-library.json",
            "mutation.json",
            "plan.json",
            "read-result.json",
        ):
            shutil.copyfile(root / name, output / name)
        preserve(
            output / "scoring-provenance.json",
            {
                "preparation": str(preparation),
                "checkpoint": binding.to_value(),
                "scoring_device": config.device,
                "initial_parameters_exactly_equal": True,
                "forward_base_equivalence": "all original fresh LoRA B tensors are zero",
                "scoring_forward_version": state.forward_version,
                "scoring_backward_version": state.backward_version,
                "scoring_z_version": state.z_version,
                "original_rollout_policy": artifact.manifest.policy_snapshot.to_value(),
                "temperature_beta": temperature_beta,
                "seed": 0,
            },
        )
        source = score_source(
            backbone,
            artifact,
            output,
            temperature_beta=temperature_beta,
            forward_version=state.forward_version,
            backward_version=state.backward_version,
        )
        require_initial_parameters(backbone, reference)
        if any(p.grad is not None for p in backbone.named_trainable_parameters().values()):
            raise ValueError("read-only scoring unexpectedly accumulated gradients")
        return project(
            source,
            skill_id,
            output,
            diagnostics=DiagnosticsConfig(),
            calibration=CalibrationConfig(),
        )
    except BaseException as error:
        preserve(
            output / "failure.json", {"error_type": type(error).__name__, "message": str(error)}
        )
        raise


def restore_run(output):
    result = read(output / "result.json")
    if result["driver_pid"] == os.getpid():
        raise ValueError("qualification restoration requires a new driver process")
    state_value = read(output / "projection-after.json")
    state = FullProjectionRuntimeState.from_value(state_value)
    pipeline = MethodProjectionPipeline.from_runtime_state(
        diagnostics_config=DiagnosticsConfig.from_value(result["diagnostics"]),
        calibration_config=CalibrationConfig.from_value(result["calibration"]),
        state=state,
    )
    cells = [
        c.to_value() for c in pipeline.calibration_cells if c.skill_id == result["new_skill_id"]
    ]
    if pipeline.runtime_state().to_value() != state_value or cells != result["new_skill_cells"]:
        raise ValueError("restored projection/cells differ from saved development evidence")
    restored = {
        "format": FORMAT,
        "driver_pid": os.getpid(),
        "source_driver_pid": result["driver_pid"],
        "label_scope": LABEL_SCOPE,
        "new_skill_cells": cells,
        "projection_exactly_restored": True,
        "model_calls": 0,
        "teacher_forced_calls": 0,
        "optimizer_steps": 0,
        "new_posterior_updates": 0,
    }
    preserve(output / "restored.json", restored)
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    score = commands.add_parser("score")
    score.add_argument("--development-root", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--preparation", type=Path, required=True)
    score.add_argument("--temperature-beta", type=float, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "score":
        result = score_run(
            args.development_root.resolve(),
            args.output.resolve(),
            args.preparation.resolve(),
            args.temperature_beta,
        )
    else:
        result = restore_run(args.output.resolve())
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
