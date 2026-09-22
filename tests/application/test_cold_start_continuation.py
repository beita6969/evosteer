import asyncio
import json
from dataclasses import replace

import pytest
from skillev_private.experiments.bayesian_improve_training import _bind_run_directory
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.application_continuation import SkillColdStartContinuation
from skillev.contracts import canonical_json
from skillev.evolution.cold_start_config import ColdStartConfig
from skillev.runtime import ActionKind, StructuredAction
from skillev.training.config import INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT, PolicyRolloutConfig
from tests.application.test_bayesian_recovery import restore_fixture
from tests.application.test_full_vertical_loop import build_application_fixture
from tests.application.test_reasoning_continuation import with_config


def test_formal_extension_is_explicit_and_cannot_change_other_conditions(tmp_path):
    before = BayesianFormalConfig(
        format="skillev-bayesian-formal-training@6", skill_exposure="catalog-then-read@2"
    )
    after = replace(before, skill_exposure="catalog-then-read@1", cold_start=ColdStartConfig())
    root = tmp_path / "run"
    _bind_run_directory(root, before, None)
    checkpoint = root / "checkpoints/step-20"
    assert _bind_run_directory(root, after, checkpoint, allow_skill_cold_start=True) == before
    assert before.condition != after.condition
    assert before.application_config("run").evolution.cold_start is None
    assert after.application_config("run").evolution.cold_start == ColdStartConfig()
    assert BayesianFormalConfig(**after.to_value()) == after
    for kwargs in ({}, {"allow_catalog_read": True}):
        with pytest.raises(ValueError):
            _bind_run_directory(root, after, checkpoint, **kwargs)
    with pytest.raises(ValueError):
        _bind_run_directory(
            root, replace(after, static_max_turns=2), checkpoint, allow_skill_cold_start=True
        )
    with pytest.raises(ValueError):
        replace(after, skill_exposure="catalog-then-read@2")
    assert json.loads((root / "formal-config.json").read_text()) == before.to_value()


def test_declaration_preserves_live_budgets_and_thinking_without_overwriting(tmp_path):
    from scripts.configure_skill_cold_start import declare

    before = BayesianFormalConfig(
        format="skillev-bayesian-formal-training@6",
        skill_exposure="catalog-then-read@2",
        max_turns=25,
        static_max_turns=8,
        thinking_off_domains=("hotpotqa", "triviaqa", "mbpp-plus", "humaneval", "alfworld"),
        reasoning_tokens_by_domain=(("aime-2026", 16384),),
    )
    source, destination = tmp_path / "condition.json", tmp_path / "cold.json"
    source.write_text(json.dumps({"config": before.to_value()}))
    after = declare(source, destination)
    assert replace(after, cold_start=None, skill_exposure=before.skill_exposure) == before
    assert BayesianFormalConfig.load(destination) == after
    with pytest.raises(FileExistsError):
        declare(source, destination)


def test_complete_checkpoint_extension_preserves_all_learned_state(
    tmp_path, training_backbone_config, monkeypatch
):
    monkeypatch.setattr(
        "tests.application.test_full_vertical_loop.PolicyRolloutConfig",
        lambda **kw: PolicyRolloutConfig(
            format=INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT,
            skill_exposure="catalog-then-read@1",
            **kw,
        ),
    )
    monkeypatch.setattr(
        "tests.application.test_full_vertical_loop._skill_action_text",
        lambda skill_id: canonical_json(
            StructuredAction(ActionKind.SKILL, "invoke", {}, "skill-runtime", skill_id).to_value()
        ),
    )
    fixture = build_application_fixture(tmp_path, training_backbone_config, cycles=2)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1)
    )
    source = fixture.public_identity
    target = with_config(
        source,
        replace(
            source.application_config,
            evolution=replace(source.application_config.evolution, cold_start=ColdStartConfig()),
        ),
    )
    continuation = SkillColdStartContinuation(source, 1)
    assert continuation.require_target(target) == source.runtime_snapshot_identity()
    with pytest.raises(ValueError):
        continuation.require_target(source)
    original = tmp_path / "snapshots/step-00000001"
    restored, _ = restore_fixture(
        fixture,
        original,
        training_backbone_config,
        public_identity=target,
        skill_exposure_continuation=continuation,
    )
    boundary = restored.evolution_loop.save_condition_boundary("cold-start-from-step-one")
    saved = restored.snapshot_store.load_exact(boundary)
    old = fixture.application.snapshot_store.load_exact(original)
    assert saved.execution_state == old.execution_state
    again, _ = restore_fixture(fixture, boundary, training_backbone_config, public_identity=target)
    assert again.training_loop.optimizer_step == 1
    assert again.detector.state == restored.detector.state
