"""The formal collector and isolated evaluator execute one shared episode machine."""

import asyncio
import json
from dataclasses import replace

import pytest
import torch

from skillev.rollout.action_surface import (
    ACTION_SURFACE_FORMAT_V3,
    ActionSurface,
    CompletionSpec,
    PublicActionInstructions,
    TerminalMode,
)
from skillev.rollout.readonly_collection import collect_readonly_panel
from skillev.runtime import BudgetLedger, LiveAttemptEventLog, RuntimeEventEmitter
from skillev.training.config import (
    SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
    conservative_rollout_maximum,
)
from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources
from tests.training.fakes import FakeISOClock, OrderedSessionFactory, make_public_tasks


@pytest.mark.parametrize("thinking", [False, True])
def test_same_public_episode_has_same_tokens_wire_seed_and_no_evaluation_writes(
    make_training_harness, tmp_path, thinking
):
    template = make_training_harness()
    rollout = replace(
        template.config.rollout,
        format=SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
        phase_context=True,
        action_wire="native-single-tool-call@3",
        skill_exposure="catalog-then-read@1",
        reasoning_tool_catalog=True,
        token_budget_notice=True,
        public_action_semantics=True,
        task_semantic_guidance="public-task-semantics@5",
        reasoning_native_thinking=thinking,
        max_action_tokens=256,
        per_rollout_maximum=conservative_rollout_maximum(
            max_turns=1,
            max_reasoning_tokens=64,
            max_action_tokens=256,
            max_model_input_tokens=8192,
            max_tool_wall_time_milliseconds=1000,
        ),
    )
    tasks = tuple(
        replace(
            t,
            task_family="triviaqa/qa",
            context_id="triviaqa/public-task",
            public_context={"benchmark_id": "triviaqa", "input_profile": "released-iid-source@1"},
            action_surface=ActionSurface(
                format=ACTION_SURFACE_FORMAT_V3,
                terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
                completion=CompletionSpec({"answer": "string"}, {"answer": "example"}),
                instructions=PublicActionInstructions(semantic=("Answer the public question.",)),
            ),
        )
        for t in make_public_tasks(2)
    )
    harness = make_training_harness(
        tasks=tasks,
        config=replace(template.config, rollout=rollout),
        action_text='<tool_call>{"name":"submit_answer","arguments":{"answer":"blue"}}</tool_call>',
    )
    # The general-purpose tiny fixture defaults to a legacy assembler. Install
    # the same pure-config projection used by SKILLEVApplication and IID.
    assembler = rollout.context_assembler(maximum_h0_tokens=8192)
    harness.loop._collector._context_assembler = assembler
    batch = asyncio.run(harness.loop.collect_batch())
    training_requests = tuple(harness.generator.calls)
    harness.generator.calls.clear()
    before_parameters = {
        k: v.detach().clone() for k, v in harness.backbone.named_trainable_parameters().items()
    }
    before_posterior = harness.projections.runtime_state()
    before_optimizer = harness.loop.optimizer.state_dict()
    before_library = harness.library.state
    before_cursor = harness.task_provider.cursor
    original_events = harness.event_log.path.read_bytes()
    log = LiveAttemptEventLog(
        tmp_path / "evaluation-events.jsonl", run_id="eval", attempt_id="eval"
    )
    outcome = asyncio.run(
        collect_readonly_panel(
            root=tmp_path / "evaluation",
            tasks=tasks,
            generator=harness.generator,
            sessions=OrderedSessionFactory((0.0, 1.0)),
            library=before_library,
            rollout=rollout,
            assembler=assembler,
            epsilon_min=harness.config.method.epsilon_min,
            condition_id=harness.loop._collector._condition_id,
            sampling_schedule_id=harness.loop._collector._sampling_schedule_hash,
            ordered_sequence_id=harness.loop._collector._ordered_task_sequence_hash,
            # Deliberately equal coordinates only for this same-input component test.
            # Real IID freezes its own evaluation schedule, never a training cursor.
            schedule_purpose="iid-training",
            anchor_ordinal=1,
            resources=RolloutWorkflowResources(RolloutWorkflowBinding()),
            ledger=BudgetLedger(
                run_id="eval", attempt_id="eval", cap=rollout.per_rollout_maximum.scale(2)
            ),
            emitter=RuntimeEventEmitter(log, "evaluation"),
            clock=FakeISOClock(),
            chunk_size=1,
        )
    )
    for train, evaluate in zip(training_requests, harness.generator.calls, strict=True):
        assert (
            train.phase,
            train.input_ids,
            train.seed,
            train.max_new_tokens,
            train.decoding_snapshot_id,
            train.action_boundary_version,
        ) == (
            evaluate.phase,
            evaluate.input_ids,
            evaluate.seed,
            evaluate.max_new_tokens,
            evaluate.decoding_snapshot_id,
            evaluate.action_boundary_version,
        )
    for train, evaluate in zip(batch.artifacts, outcome.artifacts, strict=True):
        assert train.record.steps == evaluate.record.steps
        assert train.record.reward == evaluate.record.reward
        assert (
            train.record.initial_context.meta["task_semantic_input_profile"]
            == "released-iid-source@1"
        )
    assert harness.task_provider.cursor == before_cursor
    assert harness.loop.optimizer_step == 0
    assert harness.loop.optimizer.state_dict() == before_optimizer
    assert harness.projections.runtime_state() == before_posterior
    assert harness.library.state == before_library
    assert harness.event_log.path.read_bytes() == original_events
    assert all(
        torch.equal(v, before_parameters[k])
        for k, v in harness.backbone.named_trainable_parameters().items()
    )
    assert not any(
        json.loads(line)["event_type"] == "training_step_committed"
        for line in log.path.read_text().splitlines()
    )
