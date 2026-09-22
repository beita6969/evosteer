"""Synthetic frozen-plan/report behavior, not real skill generalization evidence."""

import copy
import json
from dataclasses import asdict

import pytest
from skillev_private.experiments import autonomous_skill_validation as validation
from skillev_private.experiments import development_collection as collection

from skillev.rollout import PolicySnapshot
from tests.training.test_development_collection import request_fixture, write
from tests.v3_helpers import make_artifact


def frozen_fixture(tmp_path):
    request, _, _ = request_fixture(tmp_path)
    _, frozen = collection.selection(request)
    original = frozen["source_manifest"]["ordered_sources"][0]
    frozen["source_manifest"]["ordered_sources"] = [
        dict(original, source_id=f"source-{i}", task_id=f"task-{i}") for i in range(4)
    ]
    frozen["records"] = [copy.deepcopy(frozen["records"][0]) for _ in range(4)]
    policy = make_artifact("synthetic").manifest.policy_snapshot.to_value()
    after_policy = PolicySnapshot.create(
        **{k: v for k, v in policy.items() if k not in {"snapshot_id", "forward_adapter_version"}},
        forward_adapter_version="warmup-initialization/synthetic@0",
    ).to_value()
    plan = {
        "format": validation.FORMAT,
        "comparison_id": frozen["comparison_id"],
        "architecture": {key: {"synthetic": True} for key in validation.ARCHITECTURE_FIELDS},
        "candidate_library": frozen["initial_library"],
        "policies": {"before": policy, "after": after_policy},
        "excluded_development_sources": [{"benchmark": "healthbench", "source_id": "teacher"}],
        "source_roles": [
            {
                "benchmark": "healthbench",
                "source_id": f"source-{i}",
                "role": validation.ROLES[i // 2],
                "public_basis": "synthetic pre-outcome public task description",
            }
            for i in range(4)
        ],
        "thresholds": asdict(
            validation.AutonomousThresholds(2, 1, 0.5, 0.5, 0, 0, 0.25, 0.15, 1.5)
        ),
    }
    frozen["policy"] = policy
    return frozen, plan


def four_arms(tmp_path):
    frozen, plan = frozen_fixture(tmp_path)
    roots = {}
    for arm in validation.ARMS:
        root = roots[arm] = tmp_path / arm
        episodes = root / "collection/episodes"
        episodes.mkdir(parents=True)
        value = copy.deepcopy(frozen)
        value["policy"] = plan["policies"][arm.split("-")[0]]
        value["arm"] = "skills-off" if arm.endswith("-off") else "initial-library"
        if arm.startswith("after-"):
            value["forward_initialization"] = {
                "kind": "skill-use-warmup",
                "initialization": {
                    "application_annotations": [{"source": ["healthbench", "teacher"]}]
                },
            }
        value["autonomous_validation"] = {"plan": plan, "arm": arm}
        write(root / "selection-private.json", value)
        write(
            root / "controls-private.json",
            {
                "selection": value,
                "policy": value["policy"],
                "library": plan["candidate_library"],
                **plan["architecture"],
            },
        )
        write(root / "collection/collection.json", {"synthetic": True})
        for i, source in enumerate(value["source_manifest"]["ordered_sources"]):
            artifact = make_artifact(f"synthetic-{i}", skills_by_step=((),)).to_value()
            artifact["manifest"].update(
                task_id=source["task_id"],
                policy_snapshot=value["policy"],
                reasoning_token_counts=[5] * len(artifact["record"]["steps"]),
            )
            artifact["record"]["reward"].update(
                value=int(i > 1 or arm == "after-on"), success=i > 1 or arm == "after-on"
            )
            write(
                episodes / f"episode-{i:06d}-private.json",
                {
                    "position": i,
                    "task_id": source["task_id"],
                    "artifact": artifact,
                    "execution_status": "completed",
                    "infrastructure_error": None,
                },
            )
        write(root / "summary.json", collection.aggregate(root, value, elapsed_seconds=10))
    return roots


def test_no_reads_cannot_pass_despite_improved_reward(tmp_path):
    roots = four_arms(tmp_path)
    result = validation.compare_autonomous_use(roots, {"before-on": [], "after-on": []})
    assert result["effects"]["after_library"]["reward_gain"] == 0.5
    assert result["successful_application_sources"] == 0
    assert not result["autonomous_skill_use_qualified"]
    assert not result["five_step_diagnostic_eligible"]
    assert result["training_evidence_writes"] == 0
    assert len(result["arms"]["after-on"]["rows"]) == 4


def test_rejected_read_is_not_a_no_read_control(tmp_path, monkeypatch, training_backbone_config):
    from skillev.policy import QwenTokenizerAdapter
    from tests.rollout.engine_fakes import ByteTokenizer
    from tests.training.test_skill_use_warmup import example

    demo = example(tmp_path, "applied")
    artifact = demo.artifact.to_value()
    artifact["record"]["steps"][0].update(
        invoked_skill_ids=[], observation_status="schema_invalid", observation_text="read rejected"
    )
    write(tmp_path / "episode.json", {"artifact": artifact})
    # Use the actual tokenizer fixture, but force the lazy construction path.
    write(tmp_path / "controls-private.json", {"backbone": training_backbone_config.to_value()})
    monkeypatch.setattr(QwenTokenizerAdapter, "from_config", lambda _: ByteTokenizer())
    usage, complete = validation._review_usage(
        tmp_path,
        [{"position": 0, "artifact_path": "episode.json", "success": False}],
        [],
        None,
    )
    assert len(usage[0]["reads"]) == 1
    assert not usage[0]["reads"][0]["body_returned"]
    assert not usage[0]["successful_application"]
    assert not complete


def test_predeclared_selectivity_benefit_and_full_reviews_all_required(tmp_path, monkeypatch):
    roots = four_arms(tmp_path)

    def reviewed(root, rows, notes, tokenizer):
        return [
            {
                "position": i,
                "reads": [
                    {"repeat_same_version": False, "review": {"selection_relevance": "applicable"}}
                ]
                if root.name == "after-on" and i < 2
                else [],
                "successful_application": root.name == "after-on" and i < 2,
            }
            for i in range(4)
        ], True

    monkeypatch.setattr(validation, "_review_usage", reviewed)
    result = validation.compare_autonomous_use(roots, {"before-on": [], "after-on": []})
    assert result["autonomous_skill_use_qualified"]
    assert not result["formal_iid_admission"]
    assert result["applicable_application_rate"] == 1
    assert result["direct_no_read_rate"] == 1
    # The whole four-arm counterfactual table is kept; no successful subset selection.
    assert all(len(arm["rows"]) == 4 for arm in result["arms"].values())


@pytest.mark.parametrize(
    "change",
    ["teaching", "policy", "source", "threshold", "architecture", "missing", "updates", "sampling"],
)
def test_no_post_hoc_or_mixed_condition_acceptance(tmp_path, change):
    roots = four_arms(tmp_path)
    root = roots["after-on"]
    selection_path = root / "selection-private.json"
    value = json.loads(selection_path.read_text())
    if change == "teaching":
        value["development_practice"] = {"profile": "training-development-skill-practice@3"}
    elif change == "policy":
        value["policy"]["forward_adapter_version"] = "different"
    elif change == "source":
        value["source_manifest"]["ordered_sources"][0]["source_id"] = "teacher"
    elif change == "threshold":
        value["autonomous_validation"]["plan"]["thresholds"]["minimum_direct_no_read_rate"] = 0.1
    elif change == "architecture":
        path = root / "controls-private.json"
        controls = json.loads(path.read_text())
        controls["scorers"] = {"changed_judge": True}
        write(path, controls)
    elif change == "missing":
        (root / "collection/episodes/episode-000003-private.json").unlink()
    elif change == "updates":
        path = root / "summary.json"
        summary = json.loads(path.read_text())
        summary["posterior_updates"] = 1
        write(path, summary)
    else:
        path = root / "collection/episodes/episode-000000-private.json"
        outcome = json.loads(path.read_text())
        outcome["artifact"]["manifest"]["sampling_coordinate"] = {"different_seed": 1}
        write(path, outcome)
    write(selection_path, value)
    with pytest.raises(ValueError):
        validation.compare_autonomous_use(roots, {"before-on": [], "after-on": []})


def test_collection_rejects_teacher_metadata_and_real_warmup_overlap_before_calls(tmp_path):
    frozen, plan = frozen_fixture(tmp_path)
    frozen.update(
        arm="initial-library",
        forward_initialization={
            "kind": "skill-use-warmup",
            "initialization": {"application_annotations": [{"source": ["healthbench", "teacher"]}]},
        },
    )
    declaration = {"plan": plan, "arm": "after-on"}
    frozen["policy"] = plan["policies"]["after"]
    assert validation.freeze_validation(declaration, frozen) == declaration
    frozen["records"][0]["public_task"]["public_context"]["development_skill_practice"] = {
        "instruction": "read first"
    }
    with pytest.raises(ValueError):
        validation.freeze_validation(declaration, frozen)
    del frozen["records"][0]["public_task"]["public_context"]["development_skill_practice"]
    frozen["forward_initialization"]["initialization"]["application_annotations"][0]["source"] = [
        "healthbench",
        "source-0",
    ]
    with pytest.raises(ValueError):
        validation.freeze_validation(declaration, frozen)


def test_rules_cannot_be_added_only_after_collection(tmp_path):
    roots = four_arms(tmp_path)
    selection = json.loads((roots["before-off"] / "selection-private.json").read_text())
    del selection["autonomous_validation"]
    write(roots["before-off"] / "selection-private.json", selection)
    with pytest.raises((ValueError, KeyError)):
        validation.compare_autonomous_use(roots, {"before-on": [], "after-on": []})
