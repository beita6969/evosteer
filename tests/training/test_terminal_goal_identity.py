"""Actual session/identity serialization and CPU assembly, no benchmark execution."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.alfworld_public_goal import LEGACY_GOAL_BINDING, RESET_GOAL_BINDING
from skillev_private.benchmarks.mbpp_scoring import MBPPScorerProfile
from skillev_private.benchmarks.protocol_v13_training_sessions import (
    Protocol13TrainingSessionFactory,
)
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig
from skillev_private.experiments.bayesian_training_setup import _public_identity

from skillev.evaluation.healthbench_luna_profile import LEGACY_TRAINING_JUDGE, PROFILE_ID
from tests.application import test_full_vertical_loop as vertical
from tests.application.test_reasoning_continuation import with_config
from tests.training.test_formal_training_observation import CONFIG


def identity(initial, goal=LEGACY_GOAL_BINDING, judge=PROFILE_ID):
    config = BayesianFormalConfig.load(CONFIG)
    return _public_identity(
        application=config.application_config("synthetic-terminal-identity"),
        run_plan=config.run_plan,
        task_ids=("synthetic-task",),
        checkpoint=SimpleNamespace(trainable_state=initial),
        mbpp_profile=MBPPScorerProfile(),
        healthbench_judge=judge,
        alfworld_goal_binding=goal,
    )


@pytest.mark.parametrize("goal", [LEGACY_GOAL_BINDING, RESET_GOAL_BINDING])
@pytest.mark.parametrize("judge", [LEGACY_TRAINING_JUDGE, PROFILE_ID])
def test_identity_matches_actual_native_session_contract(training_backbone, goal, judge):
    initial = training_backbone.trainable_state_identity
    sessions = Protocol13TrainingSessionFactory(
        {}, MBPPScorerProfile(), healthbench_judge=judge, alfworld_goal_binding=goal
    )
    actual = identity(initial, goal, judge).snapshot_identity
    assert (
        actual.terminal_evaluation_conditions_json == sessions.terminal_evaluation_conditions_json
    )
    declared = json.loads(actual.terminal_evaluation_conditions_json)
    assert ("alfworld_goal_binding" in declared) is (goal == RESET_GOAL_BINDING)
    assert declared["mbpp-plus"] == MBPPScorerProfile().to_value()
    assert (
        actual
        != identity(
            initial,
            RESET_GOAL_BINDING if goal == LEGACY_GOAL_BINDING else LEGACY_GOAL_BINDING,
            judge,
        ).snapshot_identity
    )


@pytest.mark.parametrize("correct", [False, True])
def test_real_core_assembly_retains_guard_and_accepts_only_matching_goal(
    tmp_path, training_backbone, training_backbone_config, monkeypatch, correct
):
    # Reproduce the failing boundary without loading Qwen9B or invoking a scorer.
    # The real small CPU application reaches the unchanged _assemble check.
    actual = Protocol13TrainingSessionFactory(
        {},
        MBPPScorerProfile(),
        healthbench_judge=PROFILE_ID,
        alfworld_goal_binding=RESET_GOAL_BINDING,
    ).terminal_evaluation_conditions_json
    declared = identity(
        training_backbone.trainable_state_identity,
        RESET_GOAL_BINDING if correct else LEGACY_GOAL_BINDING,
    ).snapshot_identity
    original = vertical.make_public_identity

    def public(**kwargs):
        source = original(**kwargs)
        source = with_config(source, source.application_config)
        return replace(
            source,
            snapshot_identity=replace(
                source.snapshot_identity,
                terminal_evaluation_conditions_json=declared.terminal_evaluation_conditions_json,
            ),
        )

    monkeypatch.setattr(vertical, "make_public_identity", public)
    monkeypatch.setattr(
        vertical._BaseSessionFactory, "terminal_evaluation_conditions_json", actual, raising=False
    )
    if not correct:
        with pytest.raises(ValueError):
            vertical.build_application_fixture(tmp_path, training_backbone_config)
    else:
        fixture = vertical.build_application_fixture(tmp_path, training_backbone_config)
        assert fixture.application.training_loop.optimizer_step == 0
        assert fixture.application.snapshot_identity.terminal_evaluation_conditions_json == actual
