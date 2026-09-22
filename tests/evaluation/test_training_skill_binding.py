"""Frozen train/IID features and checkpoint provenance, without opening a trainer."""

import json
from dataclasses import replace

import pytest
from skillev_private.evaluation.integrity_skills import read_skill_library

from skillev.evolution.retriever import TaskRetrievalFeatures
from skillev.evolution.skill_access import ReadOnlySkillAccess
from skillev.evolution.task_features import configured_public_task_features, public_task_features
from skillev.experiments._evolution_preflight_seed import _seed_document
from skillev.runtime import SkillLibraryState
from tests.evaluation.test_integrity_skill_libraries import checkpoint, snapshot


@pytest.mark.parametrize(
    "benchmark",
    [
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "mbpp-plus",
        "humaneval",
        "math-hard",
    ],
)
def test_public_mapping_matches_explicit_training_family_without_split_or_future_features(
    benchmark,
):
    evaluation = public_task_features(benchmark)
    training = public_task_features(benchmark, task_family=evaluation.task_family)
    assert training == evaluation
    assert "training" not in evaluation.context_id
    assert "evaluation" not in evaluation.context_id
    assert configured_public_task_features(benchmark, evaluation.to_value()) == evaluation
    with pytest.raises(ValueError):
        configured_public_task_features(benchmark, {"final_horizon": 8})


def test_missing_public_subtype_is_not_inferred_from_private_alfworld_routes():
    assert public_task_features("alfworld").task_family == "alfworld/unspecified"
    assert (
        public_task_features("alfworld", task_family="pick_and_place_simple").task_family
        == "alfworld/pick_and_place_simple"
    )


def test_scope_and_missing_tools_are_reported_without_broadening_or_faking_them():
    library = snapshot("scope", "A synthetic public method.")
    document = library.state.documents["synthetic-skill"]
    features = public_task_features("mbpp-plus")
    document = _seed_document(
        skill_id=document.manifest.skill_id,
        title=document.title,
        summary=document.summary,
        instructions=document.instructions,
        requirements=document.requirements,
        task_families=(features.task_family,),
        contexts=(features.context_id,),
        required_tools=("actual-python",),
    )
    state = SkillLibraryState.from_seed_documents((document,))
    access = ReadOnlySkillAccess(state)
    query = TaskRetrievalFeatures("task", features.task_family, features.context_id, ())
    assert not access.retrieve(query)
    assert "required-tools-missing" in access.applicability_report(query)[0]["mismatch_reasons"]
    visible = replace(query, available_tools=("actual-python",))
    assert [skill.metadata.skill_id for skill in access.retrieve(visible)] == ["synthetic-skill"]
    assert not access.retrieve(replace(visible, context="legacy:evaluation"))
    assert access.applicability_report(replace(visible, context="legacy:evaluation"))[0][
        "mismatch_reasons"
    ] == ["context-mismatch"]
    assert state == access._library.state


def test_nonzero_step_does_not_imply_mutation_and_provenance_stays_out_of_actor(tmp_path):
    library = snapshot("unmutated", "Public synthetic body.")
    settings = checkpoint(tmp_path, library)
    path = tmp_path / library.library_id / "runtime_state.json"
    metadata = json.loads(path.read_text())
    metadata["execution_state"]["run_cursor"] = {"committed_cycles": 0, "committed_actions": 0}
    metadata["identity"] = {"initial_library_version": library.state.current_version}
    path.write_text(json.dumps(metadata))
    loaded = read_skill_library(library.library_id, settings)
    assert loaded.source_optimizer_step == 16
    assert loaded.training_provenance()["actual_library_mutation_count"] == 0
    assert loaded.training_provenance()["posterior_update_count"] is None
    assert str(tmp_path) not in repr(loaded.to_value())
    assert "committed_cycles" not in repr(loaded.to_value())


@pytest.mark.parametrize(
    ("trained", "kind", "expected"),
    [
        (False, "initial", "C0"),
        (True, "initial", "C1"),
        (False, "evolved", "C2"),
        (True, "evolved", "C3"),
    ],
)
def test_report_separates_weight_library_and_combined_conditions(trained, kind, expected):
    from skillev.evaluation.step0_integrity import InferenceArm, SkillMode
    from skillev.evaluation.training_product_report import training_product_report
    from skillev.evolution.skill_access import ACTIVE_APPLICABILITY_RULE
    from tests.evaluation.test_step0_integrity_results import controls

    arm = InferenceArm(
        "synthetic-axis",
        optimizer_steps=16 if trained else 0,
        policy_id="synthetic-trained" if trained else None,
        skill_mode=SkillMode.LIBRARY,
        skill_library_id="synthetic",
        skill_retrieval_rule=ACTIVE_APPLICABILITY_RULE,
    )
    expanded = replace(
        controls(),
        model={"checkpoint": {"checkpoint_directory": "/private/same-checkpoint"}},
        skills={
            "kind": kind,
            "library_id": "synthetic",
            "training_provenance": {
                "source_checkpoint": "/private/same-checkpoint",
                "actual_library_mutation_count": 0,
            },
        },
    )
    report = training_product_report(arm, expanded)
    assert report["condition"] == expected
    assert report["actual_library_mutation_count"] == 0
    assert "/private/" not in repr(report)
    if expected == "C3":
        wrong = replace(
            expanded, model={"checkpoint": {"checkpoint_directory": "/private/other-checkpoint"}}
        )
        assert training_product_report(arm, wrong)["condition"] != "C3"
