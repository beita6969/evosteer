"""Synthetic saved TTB metadata, not Qwen training or empirical skill gains."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.experiments import autonomous_skill_validation as validation
from skillev_private.experiments import development_forward as forward
from skillev_private.experiments.development_ttb import FORMAT, resolve_ttb

from skillev.contracts import canonical_json
from skillev.policy.checkpoint import POLICY_STATE_FILE, read_policy_checkpoint_state
from skillev.rollout import PolicySnapshot
from skillev.runtime import AttemptRunCursorState, EventEnvelope, EventType
from skillev.runtime.step_transaction import StepTransactionRecord, StepTransactionState
from skillev.training.run_condition import EffectiveRunCondition
from tests.runtime.test_runtime_snapshot_v4 import _snapshot
from tests.training.test_autonomous_skill_validation import frozen_fixture
from tests.training.test_autonomous_ttb import CONFIG, BayesianFormalConfig, source_fixture
from tests.training.test_development_collection import request_fixture, write
from tests.training.test_development_forward import initialization_inputs


@pytest.fixture
def ttb_candidate(tmp_path, training_backbone, training_backbone_config):
    request, _, _ = request_fixture(tmp_path)
    config = BayesianFormalConfig.load(CONFIG)
    before, backbone = initialization_inputs(
        tmp_path,
        request,
        training_backbone,
        training_backbone_config,
        "fresh-forward",
        config.sampling_config.to_value(),
    )
    preparation = request["forward_initialization"]["preparation"]
    raw = json.loads(Path(preparation).read_text())
    original = read_policy_checkpoint_state(Path(raw["initial_checkpoint"]["directory"]))
    root = tmp_path / "actual-ttb-run"
    root.mkdir()
    write(root / "formal-config.json", config.to_value())
    snapshot = _snapshot()
    run_cursor = AttemptRunCursorState.fresh(config.run_plan)
    for _ in range(3):
        run_cursor = run_cursor.after_training_step(config.run_plan)
    snapshot = replace(
        snapshot,
        experiment_id=root.name,
        identity=replace(
            snapshot.identity,
            initial_trainable_state_hash=original.trainable_state.content_hash,
            run_plan_hash=config.run_plan.content_hash,
        ),
        execution_state=replace(
            snapshot.execution_state,
            run_cursor=run_cursor,
            task_cursor=replace(snapshot.execution_state.task_cursor, cursor=84),
        ),
    )
    checkpoint = root / "checkpoints" / "step-00000003"
    (checkpoint / "policy").mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("complete\n")
    (checkpoint / "optimizer.pt").write_bytes(b"not-loaded-by-read-only-evaluator")
    write(checkpoint / "runtime_state.json", snapshot.to_value())
    policy_state = replace(original, optimizer_step=3, forward_version="synthetic-ttb-forward@3")
    (checkpoint / "policy" / POLICY_STATE_FILE).write_text(canonical_json(policy_state.to_value()))
    after = PolicySnapshot.create(
        backbone_id=original.backbone_id,
        forward_adapter_version=policy_state.forward_version,
        tokenizer_id=backbone.tokenizer_id,
        backend_id=before.backend_id,
        initial_trainable_state_hash=before.initial_trainable_state_hash,
    )
    event = EventEnvelope.create(
        event_type=EventType.TRAINING_STEP_COMMITTED,
        run_id=root.name,
        attempt_id="synthetic",
        producer_id="training",
        producer_seq=3,
        occurred_at="2026-09-14T00:00:00Z",
        payload={"optimizer_step": 3},
    )
    transaction = StepTransactionRecord(
        optimizer_step=3,
        batch_id="batch-3",
        policy_snapshot_before=before.snapshot_id,
        policy_snapshot_after=after.snapshot_id,
        checkpoint_name=checkpoint.name,
        adapter_revision=policy_state.forward_version,
        source_events=(event,),
        state=StepTransactionState.COMMITTED,
    )
    (checkpoint.parent / "step-transactions").mkdir()
    write(checkpoint.parent / "step-transactions/step-00000003.json", transaction.to_value())
    _, data_condition = source_fixture()
    source_rows = data_condition["autonomous_ttb_sources"]["ordered_sources"]
    rows = [
        {
            "benchmark_id": source_rows[i % len(source_rows)]["benchmark"],
            "source_question_id": source_rows[i % len(source_rows)]["source_id"],
            "occurrence_id": f"synthetic-occurrence-{i}",
        }
        for i in range(7000)
    ]
    library = snapshot.execution_state.library.to_value()
    condition = EffectiveRunCondition.create(
        condition_id=config.condition,
        scientific={
            "formal": config.expanded_value(),
            "initial_partition": original.trainable_state.to_value(),
            "initial_library": library,
            "data_condition": {"declaration": data_condition, "ordered_selected_sources": rows},
        },
        execution={},
    )
    effective = write(root / "effective-condition-process-123.json", condition.to_value())
    inventory = write(
        root / "skill-sources.json",
        {
            "format": "skill-construction-sources@1",
            "documents": [
                {
                    "document": doc,
                    "kind": "public-procedure",
                    "public_basis": "Synthetic answer-free test method.",
                    "source_files": [],
                }
                for doc in library["documents"]
            ],
        },
    )
    binding = {
        "format": FORMAT,
        "kind": "ttb-checkpoint",
        "checkpoint": str(checkpoint),
        "run_root": str(root),
        "preparation": preparation,
        "effective_condition": effective,
        "skill_sources": inventory,
        "published_forward_adapter": "synthetic-ttb-v3",
    }
    return request, binding, before, after


def test_ttb_binding_reads_real_saved_state_without_loading_optimizer(ttb_candidate):
    request, binding, before, after = ttb_candidate
    policy, declaration = resolve_ttb(binding)
    assert policy == after
    assert declaration["before_policy"] == before.to_value()
    assert declaration["source_optimizer_step"] == 3
    assert declaration["training_sources"]
    assert json.loads(json.dumps(declaration)) == declaration
    assert not declaration["model_loaded_by_collector"]
    assert not declaration["training_state_writes"]
    request["forward_initialization"] = binding
    assert forward.resolve_forward(request) == (policy, declaration)
    forward.require_forward_candidate(
        declaration,
        backbone=declaration["backbone"],
        sampling=declaration["sampling"],
        library=declaration["initial_library"],
    )
    with pytest.raises(ValueError):
        forward.require_forward_candidate(
            declaration,
            backbone=declaration["backbone"],
            sampling={"changed": True},
            library=declaration["initial_library"],
        )


def test_skill_construction_sources_are_retained_separately_from_training(ttb_candidate):
    _, binding, _, _ = ttb_candidate
    path = Path(binding["skill_sources"])
    raw = json.loads(path.read_text())
    sources = write(
        path.parent / "previous-construction-development.json",
        [{"benchmark": "mbpp-plus", "source_id": "previously-inspected-source"}],
    )
    raw["documents"][0].update(kind="source-derived", source_files=[sources])
    write(path, raw)
    _, resolved = resolve_ttb(binding)
    key = ["mbpp-plus", "previously-inspected-source"]
    assert key in resolved["skill_construction_sources"]
    assert key not in resolved["training_sources"]
    assert json.loads(json.dumps(resolved)) == resolved


@pytest.mark.parametrize(
    "fault", ["incomplete", "orphan", "partial-sources", "library-sources", "warmup", "step"]
)
def test_real_ttb_binding_rejects_mismatches_before_dispatch(ttb_candidate, fault):
    _, binding, _, _ = ttb_candidate
    checkpoint = Path(binding["checkpoint"])
    if fault == "incomplete":
        (checkpoint / "COMPLETE").unlink()
    elif fault == "orphan":
        path = checkpoint.parent / "step-transactions/step-00000003.json"
        raw = json.loads(path.read_text())
        raw["state"] = "adapter-committed"
        write(path, raw)
    elif fault == "partial-sources":
        path = Path(binding["effective_condition"])
        raw = json.loads(path.read_text())
        raw["scientific"]["data_condition"]["ordered_selected_sources"] = raw["scientific"][
            "data_condition"
        ]["ordered_selected_sources"][:28]
        write(path, raw)
    elif fault == "library-sources":
        path = Path(binding["skill_sources"])
        raw = json.loads(path.read_text())
        raw["documents"] = []
        write(path, raw)
    elif fault == "warmup":
        path = Path(binding["preparation"])
        raw = json.loads(path.read_text())
        raw["initialization"] = {"kind": "skill-use-warmup"}
        write(path, raw)
    else:
        path = checkpoint / "policy" / POLICY_STATE_FILE
        raw = json.loads(path.read_text())
        raw["optimizer_step"] = 2
        path.write_text(canonical_json(raw))
    with pytest.raises(ValueError):
        resolve_ttb(binding)


def test_four_arm_plan_excludes_all_training_sources_not_only_application_sources(
    tmp_path, ttb_candidate
):
    request, binding, before, after = ttb_candidate
    _, reference = resolve_ttb(binding)
    directory = tmp_path / "four-arm"
    directory.mkdir()
    frozen, plan = frozen_fixture(directory)
    plan.update(
        format=validation.TTB_FORMAT,
        comparison_kind="ttb-checkpoint",
        ttb_binding=binding,
        policies={"before": before.to_value(), "after": after.to_value()},
        candidate_library=reference["initial_library"],
    )
    frozen["initial_library"] = reference["initial_library"]
    plan["excluded_development_sources"] += [
        {"benchmark": d, "source_id": s} for d, s in reference["training_sources"]
    ]
    for arm in validation.ARMS:
        value = copy.deepcopy(frozen)
        value["policy"] = plan["policies"][arm.split("-")[0]]
        value["arm"] = "skills-off" if arm.endswith("-off") else "initial-library"
        value["forward_initialization"] = (
            reference if arm.startswith("after") else forward.resolve_forward(request)[1]
        )
        # The comparison also reads these controls back from the private JSON
        # archive; source tuples must not break a restored after binding.
        value = json.loads(json.dumps(value))
        result = validation.freeze_validation({"plan": plan, "arm": arm}, value)
        assert result["arm"] == arm
    plan["excluded_development_sources"] = plan["excluded_development_sources"][:1]
    with pytest.raises(ValueError):
        validation.freeze_validation({"plan": plan, "arm": "after-on"}, value)
