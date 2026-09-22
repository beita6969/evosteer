"""CPU integration: actual tensor update, graph execution, paired data and restore."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import uuid
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from skillev.contracts.evosteer import EvoTask, EvoTrajectory
from skillev.evolution.evosteer_author import FrozenSkillAuthor
from skillev.evolution.validated_admission import AdmissionLedger, SkillEntry
from skillev.evosteer_application import (
    EvoSteerApplication,
    EvoSteerConfig,
    ResetReceipt,
    TaskBinding,
    TaskSession,
)
from skillev.orchestration.graph import NodeExecutionResult, RoleSpec
from skillev.policy.evosteer import CausalLMOrchestrator
from skillev.rollout.evosteer_risk import ExecutionRiskPolicy
from skillev.runtime import BudgetLedger, BudgetVector
from skillev.training.evosteer import EvoOptimizerConfig


class BytesTokenizer:
    eos_token_id = 2
    special_tokens_map: ClassVar = {"eos_token": "<eos>"}
    init_kwargs: ClassVar = {}

    def get_vocab(self):
        return {str(i): i for i in range(259)}

    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode()]

    def decode(self, tokens, skip_special_tokens=False):
        return bytes(t - 3 for t in tokens if t >= 3).decode(errors="replace")


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(259, 8)
        self.output = nn.Linear(8, 259)
        self.embedding.requires_grad_(False)
        self.output.requires_grad_(False)
        self.actor_bias = nn.Parameter(torch.zeros(259))
        self.config = SimpleNamespace(to_dict=lambda: {"hidden_size": 8, "vocab_size": 259})

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, **kwargs
    ):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(
            logits=self.output(hidden) + self.actor_bias,
            hidden_states=(hidden,),
            past_key_values=None,
        )


class RepairExecutor:
    frozen_identity = "synthetic-repair-executor@1"

    async def execute(self, request):
        if request.role.role_id == "verifier":
            output = "correction"
        else:
            output = (
                "fixed"
                if request.skills or any(m["body"] == "correction" for m in request.messages)
                else "draft"
            )
        return NodeExecutionResult(
            output,
            BudgetVector(
                input_tokens=10,
                output_tokens=2,
                model_calls=1,
                agent_turns=1,
            ),
        )


def application(*, candidate=True):
    torch.manual_seed(14)
    model = TinyCausalModel()
    policy = CausalLMOrchestrator(
        model,
        BytesTokenizer(),
        reference_id="tiny-base@1",
        encoding_dim=8,
        reference_model=copy.deepcopy(model),
        context_window=65_536,
        max_action_tokens=256,
    )
    maximum = BudgetVector(
        input_tokens=4096,
        output_tokens=128,
        model_calls=1,
        agent_turns=1,
        wall_time_milliseconds=1000,
    )
    config = EvoSteerConfig(
        task_families=("synthetic",),
        roles=(
            RoleSpec("solver", "Solve the synthetic task.", maximum),
            RoleSpec("verifier", "Verify the synthetic result.", maximum),
        ),
        optimizer=EvoOptimizerConfig(actor_learning_rate=0.01),
        max_nodes=2,
        max_actions=6,
        current_rollouts=1,
        reference_rollouts=1,
        total_token_cap=1_000_000,
    )
    ledger = AdmissionLedger()
    if candidate:
        body = "Check the draft against feedback before returning the result."
        ledger.propose(
            SkillEntry(
                "candidate-one",
                "synthetic",
                "sha256:" + hashlib.sha256(body.encode()).hexdigest(),
                body=body,
            ),
            author_window_id="initial",
        )
    return EvoSteerApplication(policy, config, admission=ledger)


def binding(task_id="task-1"):
    executor = RepairExecutor()

    def session(request):
        return TaskSession(
            RepairExecutor(),
            lambda output: float(output == "fixed"),
            reset_receipt=ResetReceipt(
                request.task.identity,
                request.task.reset_id,
                request.task.environment_config_id,
                "initial-repair-state",
                str(uuid.uuid4()),
                request.seed,
            ),
            risk_assessor=ExecutionRiskPolicy(
                RepairExecutor.frozen_identity,
                request.task.environment_config_id,
                scope="text_only",
                capability_id="synthetic-test-world@1",
            ),
        )

    return TaskBinding(
        EvoTask(task_id, "synthetic", "Produce a checked result."),
        executor.frozen_identity,
        session,
        replay_safe=True,
    )


def test_complete_update_uses_reference_only_statistics_and_preserves_frozen_model():
    app = application()
    before = {k: v.clone() for k, v in app.policy.reference_model.state_dict().items()}
    actor_before = app.policy.model.actor_bias.detach().clone()
    result = asyncio.run(app.train_batch((binding(),)))
    assert result.metrics["source_counts"] == {
        "current": 1,
        "natural_reference": 1,
        "paired_treatment": 1,
        "paired_control": 1,
    }
    assert app.batch_index == app.policy.version == 1
    assert torch.isfinite(app.policy.model.actor_bias).all()
    assert not torch.equal(actor_before, app.policy.model.actor_bias)
    assert all(
        torch.equal(before[k], v) for k, v in app.policy.reference_model.state_dict().items()
    )
    reference_ids = [
        observation.observation_id
        for store in app.statistics.values()
        for observation in store.snapshot("inspect").observations
    ]
    assert len(reference_ids) == 1
    assert "natural_reference" in reference_ids[0]
    positive = next(x for x in result.trajectories if x.source == "paired_treatment")
    negative = next(x for x in result.trajectories if x.source == "paired_control")
    assert positive.decisions[0].forced
    assert negative.decisions[0].forced
    assert json.loads(positive.decisions[0].action_json)["skill_id"] == "candidate-one"
    assert json.loads(negative.decisions[0].action_json).get("skill_id") is None
    assert all(
        "candidate-one" not in json.loads(step.action_json).values() for step in negative.decisions
    )
    for trajectory in result.trajectories:
        assert EvoTrajectory.from_value(trajectory.to_value()) == trajectory


def test_checkpoint_roundtrip_keeps_weights_statistics_budget_and_can_continue(tmp_path):
    app = application()
    asyncio.run(app.train_batch((binding(),)))
    path = app.save_checkpoint(tmp_path / "step-one")
    restored = application()
    restored.load_checkpoint(path)
    assert restored.admission.to_value() == app.admission.to_value()
    assert restored.batch_index == restored.policy.version == 1
    assert restored.value_snapshot_id == app.value_snapshot_id
    assert restored.total_usage == app.total_usage
    assert all(
        torch.equal(app.policy.adapter_state()[k], v)
        for k, v in restored.policy.adapter_state().items()
    )
    result = asyncio.run(restored.train_batch((binding("task-2"),)))
    assert result.metrics["batch_index"] == 2
    assert restored.admission.alpha_spent >= app.admission.alpha_spent
    with pytest.raises(FileExistsError):
        app.save_checkpoint(path)


def test_failed_session_aborts_without_admitting_partial_pairs_or_statistics():
    app = application()

    class FailedExecutor(RepairExecutor):
        async def execute(self, request):
            raise OSError("synthetic infrastructure failure")

    broken = TaskBinding(
        binding().task,
        RepairExecutor.frozen_identity,
        lambda request: TaskSession(
            FailedExecutor(),
            lambda _: 0.0,
            risk_assessor=ExecutionRiskPolicy(
                RepairExecutor.frozen_identity,
                request.task.environment_config_id,
                scope="text_only",
                capability_id="synthetic-test-world@1",
            ),
        ),
        replay_safe=True,
    )
    with pytest.raises(OSError):
        asyncio.run(app.train_batch((broken,)))
    assert app.batch_index == app.policy.version == 0
    assert not app.statistics
    assert not app.admission.comparisons
    with pytest.raises(RuntimeError, match="healthy"):
        asyncio.run(app.train_batch((binding(),)))


def test_tampered_checkpoint_is_rejected_before_actor_mutation(tmp_path):
    app = application(candidate=False)
    path = app.save_checkpoint(tmp_path / "initial")
    state = path / "state.json"
    state.write_text(state.read_text().replace('"batch_index":0', '"batch_index":999'))
    with pytest.raises(ValueError, match="checksum"):
        app.load_checkpoint(path)
    assert app.batch_index == 0


def test_unknown_or_duplicate_tasks_rejected_before_training():
    app = application()
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(app.train_batch((binding(), binding())))
    assert app.batch_index == 0


def test_paired_tasks_require_observed_reset_receipt():
    app = application()
    task = binding()
    without_receipt = replace(
        task,
        session_factory=lambda request: TaskSession(
            RepairExecutor(),
            lambda output: 0.0,
            risk_assessor=ExecutionRiskPolicy(
                RepairExecutor.frozen_identity,
                request.task.environment_config_id,
                scope="text_only",
                capability_id="synthetic-test-world@1",
            ),
        ),
    )
    with pytest.raises(ValueError, match="observed reset receipt"):
        asyncio.run(app.train_batch((without_receipt,)))
    assert not app.admission.comparisons
    assert app.batch_index == 0


def test_mismatched_pair_initial_worlds_cannot_enter_evidence():
    app = application()
    task = binding()
    count = 0

    def mismatched(request):
        nonlocal count
        count += 1
        session = task.session_factory(request)
        if count == 4:
            session.reset_receipt = replace(session.reset_receipt, initial_state_id="other-world")
        return session

    with pytest.raises(ValueError, match="same initial state"):
        asyncio.run(app.train_batch((replace(task, session_factory=mismatched),)))
    assert not app.admission.comparisons
    assert not app.statistics


def test_reused_world_receipt_is_rejected_before_second_episode():
    app = application()
    task = binding()

    def reused(request):
        session = task.session_factory(request)
        session.reset_receipt = replace(session.reset_receipt, session_id="one-reused-world")
        return session

    with pytest.raises(ValueError, match="distinct isolated"):
        asyncio.run(app.train_batch((replace(task, session_factory=reused),)))


class NullAuthorModel:
    reference_id = "independent-frozen-author-test@1"

    def frozen_text(self, text, **kwargs):
        return "null", 32, 1


def with_author():
    app = application(candidate=False)
    app.author = FrozenSkillAuthor(
        NullAuthorModel(),
        budget=BudgetLedger(
            run_id="author-test",
            attempt_id="one-run",
            cap=BudgetVector(input_tokens=100_000, output_tokens=10_000, model_calls=4),
        ),
    )
    return app


def test_checkpoint_restores_author_spend_and_window_history(tmp_path):
    app = with_author()
    asyncio.run(app.train_batch((binding(),)))
    path = app.save_checkpoint(tmp_path / "authored")
    restored = with_author()
    restored.load_checkpoint(path)
    assert restored.author.state_dict() == app.author.state_dict()
    asyncio.run(restored.train_batch((binding("task-2"),)))
    assert len(restored.author.reports) == 2
    assert restored.author.state_dict()["budget"]["settled"]["model_calls"] == 2


def test_value_snapshot_identity_binds_published_weights():
    app = application(candidate=False)
    other = application(candidate=False)
    heads = copy.deepcopy(other.heads)
    with torch.no_grad():
        next(heads.runtime_head.parameters()).add_(0.25)
    changed = EvoSteerApplication(other.policy, other.config, heads=heads)
    assert app.value_snapshot_id != changed.value_snapshot_id


def _replica(policy):
    return CausalLMOrchestrator(
        copy.deepcopy(policy.model),
        copy.deepcopy(policy.tokenizer),
        reference_id=policy.reference_id,
        encoding_dim=policy.encoding_dim,
        reference_model=copy.deepcopy(policy.reference_model),
        context_window=policy.context_window,
        max_action_tokens=policy.max_action_tokens,
    )


def test_prefetched_batch_overlaps_update_with_sequential_first_batch_and_one_update_lag():
    sequential = application()
    expected = asyncio.run(sequential.train_batch((binding("task-1"),)))
    app = application()
    app.set_rollout_policy(_replica(app.policy))

    async def pipeline():
        first = app.prepare_batch(await app.collect_batch((binding("task-1"),), batch_number=1))
        prefetch = asyncio.create_task(app.collect_batch((binding("task-2"),), batch_number=2))
        finished = await asyncio.to_thread(app.finish_batch, first)
        second = await prefetch
        app.sync_rollout_policy()
        return finished, second, app.finish_batch(app.prepare_batch(second))

    finished, second, last = asyncio.run(pipeline())

    def decisions(result):
        # Wall-clock features and the budget text they feed vary run to run.
        return [
            (t.sample_id, t.source, t.reward, [d.action_json for d in t.decisions])
            for t in result.trajectories
        ]

    assert decisions(finished) == decisions(expected)
    assert finished.metrics["anchor_tb_loss"] == pytest.approx(expected.metrics["anchor_tb_loss"])
    assert (finished.batch_id, last.batch_id) == ("batch-000001", "batch-000002")
    # The replica lags until the next publish; the prefetched batch was sampled
    # by the adapter published before update 1.
    assert app.batch_index == app.policy.version == 2
    assert app.rollout_policy.version == 1
    current = [t for t in second.trajectories if t.source == "current"]
    assert current and all(t.behavior_policy_id.endswith("/0") for t in current)
    app.sync_rollout_policy()
    assert app.rollout_policy.version == 2
    assert torch.equal(app.rollout_policy.model.actor_bias, app.policy.model.actor_bias)


def test_collection_rejects_batches_more_than_one_ahead_or_stale_preparation():
    app = application()
    with pytest.raises(ValueError, match="one batch ahead"):
        asyncio.run(app.collect_batch((binding(),), batch_number=3))
    collected = asyncio.run(app.collect_batch((binding(),), batch_number=1))
    app.admission = AdmissionLedger.from_value(app.admission.to_value())
    with pytest.raises(ValueError, match="stale"):
        app.prepare_batch(collected)


def test_candidate_comparison_accumulates_across_value_head_refreshes():
    app = application()
    asyncio.run(app.train_batch((binding("task-1"),)))
    first_snapshot = app.value_snapshot_id
    asyncio.run(app.train_batch((binding("task-2"),)))
    assert app.value_snapshot_id != first_snapshot
    comparisons = [c for c in app.admission.comparisons if c.skill_id == "candidate-one"]
    assert len(comparisons) == 1
    assert not comparisons[0].closed
    # One pair from each batch, pooled into the same sequential test.
    assert comparisons[0].wins + comparisons[0].losses + comparisons[0].ties == 2
