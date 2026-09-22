"""Method integration: full histories, reference channels, author/pair loop, risk gate."""

import asyncio
import json
from dataclasses import replace

import pytest
import torch

from skillev.contracts.canonical import canonical_json
from skillev.contracts.evosteer import EvoTrajectory
from skillev.contracts.evosteer_risk import TrajectoryRiskAssessment
from skillev.evolution.evosteer_author import FrozenSkillAuthor
from skillev.evosteer_application import EvoSteerApplication, EvoSteerConfig
from skillev.rollout.evosteer import controller_projection
from skillev.rollout.evosteer_risk import EvoSteerRiskError, ExecutionRiskPolicy, require_assessed
from tests.evosteer.test_application import application, binding
from tests.evosteer.test_evosteer_author import Model, budget
from tests.evosteer.test_history_contracts import trajectory


def method_application(*, publish_interval=1, validation_interval=1):
    source = application(candidate=False)
    config = replace(
        source.config,
        max_nodes=None,
        max_actions=None,
        current_rollouts=2,
        reference_rollouts=2,
        value_refresh_interval=publish_interval,
        validation_interval=validation_interval,
    )
    app = EvoSteerApplication(
        source.policy, config, author=FrozenSkillAuthor(Model(), budget=budget())
    )

    def deterministic_fixture(prompt, menu, **kwargs):
        # Keep synthetic episodes short while retaining the full legal grammar
        # for real current/reference LM re-scoring and exact gradient checks.
        actions = [json.loads(action) for action in menu.actions]
        for kind in ("STOP", "SET_OUTPUT", "ADD_AGENT"):
            choices = [i for i, action in enumerate(actions) if action["kind"] == kind]
            if choices:
                index = min(
                    choices,
                    key=lambda i: (
                        actions[i].get("role_id") != "solver",
                        "skill_id" in actions[i],
                        menu.actions[i],
                    ),
                )
                return menu.paths[index]
        raise AssertionError("fixture expected a complete legal continuation")

    app.policy.sample = deterministic_fixture
    return app


def test_author_to_candidate_pair_and_all_heads_complete_in_full_history_mode(tmp_path):
    app = method_application()
    assert app.policy.context_mode == "full"
    assert app.config.max_nodes is app.config.max_actions is None
    before = {name: value.clone() for name, value in app.heads.state_dict().items()}
    first = asyncio.run(app.train_batch((binding("first"),)))
    assert first.metrics["source_counts"] == {"current": 2, "natural_reference": 2}
    assert first.metrics["author"]["status"] == "candidate_proposed"
    second = asyncio.run(app.train_batch((binding("second"),)))
    assert second.metrics["source_counts"] == {
        "current": 2,
        "natural_reference": 2,
        "paired_treatment": 1,
        "paired_control": 1,
    }
    assert len(app.author.reports) == 1  # occupied candidate slot: no extra author proposal
    assert second.metrics["paired_reference_states"] == 4  # exclude each forced root
    assert app.admission.comparisons[-1].observation_index == 1
    assert app.admission.comparisons[-1].wins == 1
    assert all(require_assessed(item).side_effect_free for item in second.trajectories)
    snapshots = [store.snapshot("inspect") for store in app.statistics.values()]
    assert sum(len(snapshot.observations) for snapshot in snapshots) == 4
    structural = [
        row
        for store in app.statistics.values()
        for row in store.to_value()["structure_observations"]
    ]
    assert len(structural) == 6
    paired = [row for row in structural if row["source"].startswith("paired_")]
    assert len(paired) == 2
    assert all(tuple(row["forced_prefix_indices"]) == (0,) for row in paired)
    for prefix in ("residual_head", "runtime_head", "outcome_heads.0", "outcome_heads.1"):
        assert any(
            not torch.equal(before[name], value)
            for name, value in app.heads.state_dict().items()
            if name.startswith(prefix)
        ), prefix
    for item in second.trajectories:
        assert EvoTrajectory.from_value(item.to_value()) == item
        for step in item.decisions:
            state = json.loads(step.state_json)
            assert tuple(state["execution_features"]) == step.features
            assert len(state["feature_names"]) == 30
            text = app.policy.tokenizer.decode(step.prompt_ids)
            rendered = json.loads(json.loads(text)[-1]["content"])
            assert tuple(rendered["execution"]["execution_features"]) == step.features
            # The prompt shows each node output once (history refers to graph.nodes);
            # the recorded state remains the full public history it is projected from.
            assert rendered["execution"]["history"] == controller_projection(state)["history"]
    checkpoint = app.save_checkpoint(tmp_path / "method")
    restored = method_application()
    restored.load_checkpoint(checkpoint)
    assert restored.batch_index == restored.value_version == 2
    assert restored.author.state_dict() == app.author.state_dict()
    assert restored.admission.to_value() == app.admission.to_value()
    assert all(
        torch.equal(value, restored.heads.state_dict()[name])
        for name, value in app.heads.state_dict().items()
    )


def test_repeat_record_still_gets_pair_but_does_not_supply_independent_sign_evidence():
    app = method_application(publish_interval=10)
    asyncio.run(app.train_batch((binding("same-task"),)))
    asyncio.run(app.train_batch((binding("same-task"),)))
    spent = app.admission.alpha_spent
    result = asyncio.run(app.train_batch((binding("same-task"),)))
    assert result.metrics["pair_count"] == 1
    assert result.metrics["paired_reference_states"] == 4
    assert result.metrics["skipped_pair_evidence"]["repeated_task"] == 1
    assert app.admission.alpha_spent == spent


def test_distinct_environment_strata_each_receive_their_validation_round():
    app = method_application(publish_interval=10)
    asyncio.run(app.train_batch((binding("author-material"),)))
    first, second = binding("stratum-a"), binding("stratum-b")
    second = replace(second, task=replace(second.task, environment_config_id="another-world@1"))
    batch = asyncio.run(app.train_batch((first, second)))
    assert batch.metrics["pair_count"] == 2
    assert len(app.admission.comparisons) == 2
    assert all(c.wins == 1 and c.observation_index == 1 for c in app.admission.comparisons)
    assert all(not c.closed for c in app.admission.comparisons)
    spent = app.admission.alpha_spent
    reverse = asyncio.run(app.train_batch((second, first)))
    assert reverse.metrics["pair_count"] == 2
    assert reverse.metrics["skipped_pair_evidence"]["repeated_task"] == 2
    assert len(app.admission.comparisons) == 2
    assert app.admission.alpha_spent == spent


def test_every_batch_records_its_pairs_but_only_a_validation_batch_spends_a_look():
    app = method_application(publish_interval=10, validation_interval=3)
    asyncio.run(app.train_batch((binding("author-material"),)))
    between = asyncio.run(app.train_batch((binding("pair-a"),)))
    assert between.metrics["pair_count"] == 1
    assert not between.admission_decisions
    comparison = app.admission.comparisons[-1]
    # The evidence is in the ledger; no look and no alpha were spent on it.
    assert comparison.wins + comparison.losses + comparison.ties == 1
    assert comparison.observation_index == 0
    assert app.admission.alpha_spent == 0.0
    scheduled = asyncio.run(app.train_batch((binding("pair-b"),)))
    assert scheduled.metrics["pair_count"] == 1
    assert len(scheduled.admission_decisions) == 1
    comparison = app.admission.comparisons[-1]
    # One look tests both batches' pairs together, so the level is spent once.
    assert comparison.wins + comparison.losses + comparison.ties == 2
    assert comparison.observation_index == 1
    assert app.admission.alpha_spent > 0.0


def test_rejected_last_trajectory_blocks_all_optimizer_statistics_and_author_writes():
    app = method_application()
    task = binding()
    original = task.session_factory
    calls = 0
    before = {name: value.clone() for name, value in app.heads.state_dict().items()}

    def session(request):
        nonlocal calls
        calls += 1
        opened = original(request)
        if calls == 4:
            opened.risk_assessor = lambda item: TrajectoryRiskAssessment(
                "test-rejected", item.risk_evidence_id, False, False, ("unresolved world effect",)
            )
        return opened

    with pytest.raises(EvoSteerRiskError, match="unresolved world effect"):
        asyncio.run(app.train_batch((replace(task, session_factory=session),)))
    assert calls == 4
    assert app.batch_index == app.policy.version == 0
    assert not app.statistics
    assert not app.author.reports
    assert not app.admission.skills
    assert all(torch.equal(before[name], value) for name, value in app.heads.state_dict().items())
    with pytest.raises(RuntimeError, match="healthy"):
        app.save_checkpoint("unused-after-failure")


def test_missing_risk_assessor_is_not_inferred_from_a_completed_executor():
    app = method_application()
    task = binding()

    def session(request):
        result = task.session_factory(request)
        result.risk_assessor = None
        return result

    with pytest.raises(EvoSteerRiskError, match="explicit trajectory risk assessor"):
        asyncio.run(app.train_batch((replace(task, session_factory=session),)))
    assert app.batch_index == 0
    assert not app.total_usage.model_calls


def test_risk_evidence_is_bound_to_exact_history_and_reward_not_a_default_boolean():
    raw = trajectory(reward=0.0)
    assessor = ExecutionRiskPolicy("executor-1", "static@1", "text_only", "no-tools@1")
    assessed = replace(raw, risk=assessor(raw))
    assert require_assessed(assessed).accepted  # ordinary task failure remains useful data
    assert EvoTrajectory.from_value(assessed.to_value()) == assessed
    with pytest.raises(ValueError, match="evidence"):
        replace(assessed, reward=1.0)
    with pytest.raises(EvoSteerRiskError):
        require_assessed(raw)
    hazardous = replace(
        raw,
        terminal_state_json=canonical_json(
            {
                "stopped": True,
                "risk_events": [{"severity": "unresolved"}],
            }
        ),
    )
    assert not assessor(hazardous).accepted
    malformed = replace(raw, terminal_state_json='{"risk_events":null,"stopped":true}')
    assert not assessor(malformed).accepted
    world_dispatch = replace(raw, usage_json='{"tool_calls":1}')
    assert not assessor(world_dispatch).accepted
    isolated = ExecutionRiskPolicy("executor-1", "static@1", "isolated", "adapter-sandbox@1")
    assert isolated(world_dispatch).accepted


def test_paper_default_has_no_extra_size_action_model_call_or_turn_limits():
    source = application(candidate=False)
    config = EvoSteerConfig(task_families=("synthetic",), roles=source.config.roles)
    assert config.max_nodes is config.max_actions is None
    assert config.current_rollouts == config.reference_rollouts == 2
    assert config.value_refresh_interval == 1
    assert config.total_token_cap == 16_384
    assert config.resource_cap.model_calls >= config.total_token_cap
    assert config.resource_cap.agent_turns >= 2 * config.total_token_cap + 50
    assert config.resource_cap.tool_calls == 50
    assert config.resource_cap.wall_time_milliseconds == 600_000
