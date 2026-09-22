"""One public catalog, explicit candidate conditions, separate submission wires."""

import asyncio
import json
from dataclasses import asdict, replace

import pytest
import yaml
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.benchmarks.protocol_v10_action import COMPLETION_WIRE_INSTRUCTION
from skillev.benchmarks.reasoning_guidance import HOTPOT_DELIBERATION
from skillev.evaluation import integrity_actor
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.step0_integrity import InferenceArm, decode_integrity_arm
from skillev.policy.phase_context import phase_chat_messages
from skillev.rollout import AssembledInitialContext, GenerationPhase, ModelVisibleMessage
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V1,
    PUBLIC_TASK_SEMANTICS_V4,
    RELEASED_PUBLIC_INPUT,
    TRAINING_PUBLIC_INPUT,
    TRAINING_SUBMISSION_INSTRUCTION,
    public_task_semantics,
)
from skillev.training.config import PolicyRolloutConfig
from tests.evaluation.test_integrity_actor_reset import ScriptedGenerator, initial_task
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer
from tests.rollout.test_native_phase_context import PhaseTokenizer
from tests.rollout.test_public_action_semantics import PUBLIC_MEANINGS, public_native_context


def shared_formal(**changes):
    return BayesianFormalConfig(
        format="skillev-bayesian-formal-training@6",
        phase_context=True,
        action_wire="native-single-tool-call@1",
        public_action_semantics=True,
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V1,
        **changes,
    )


def shared_context(domain, *, hotpot_deliberation=True):
    old_context, task = public_native_context(domain, deliberation=hotpot_deliberation)
    config = shared_formal(hotpot_deliberation=hotpot_deliberation)
    assembler = config.application_config("synthetic-shared").trainer.rollout.context_assembler(
        maximum_h0_tokens=20000
    )
    context = assembler.assemble(
        task=task,
        retrieved_skills=(),
        active_skill_ids=(),
        library_version="synthetic-library",
        tokenizer=PhaseTokenizer(),
    )
    return context, task, old_context


@pytest.mark.parametrize(("domain", "_old_meaning"), PUBLIC_MEANINGS)
def test_seven_domains_share_task_meaning_without_sharing_submission_wire(domain, _old_meaning):
    context, task, old_context = shared_context(domain)
    view = PublicTaskView.from_training_task(task)
    arm = InferenceArm(
        "synthetic-shared",
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V1,
        hotpot_deliberation=True,
    )
    semantic = public_task_semantics(
        domain.value, input_profile=TRAINING_PUBLIC_INPUT, hotpot_deliberation=True
    )
    evaluation_instruction = integrity_actor.shared_task_instruction(view, arm)
    assert semantic in evaluation_instruction
    for phase in ("Reasoning:\n", "Action:\n"):
        messages, tools = phase_chat_messages(context.text + phase)
        assert semantic in json.dumps(messages)
        if tools is not None:
            assert semantic in json.dumps(tools)
        assert COMPLETION_WIRE_INSTRUCTION not in str((messages, tools))
    if domain.value != "alfworld":
        assert TRAINING_SUBMISSION_INSTRUCTION in context.text
        assert TRAINING_SUBMISSION_INSTRUCTION not in evaluation_instruction
    if domain.value in {"hotpotqa", "triviaqa"}:
        assert "Final answer:" in evaluation_instruction
        assert "Final answer:" not in context.text
    assert task.query == context.contract.query == old_context.contract.query
    assert task.public_context == {"benchmark_id": domain.value}
    assert context.contract.meta["task_semantic_guidance"] == PUBLIC_TASK_SEMANTICS_V1
    assert context.contract.meta["task_semantic_input_profile"] == TRAINING_PUBLIC_INPUT
    assert PUBLIC_TASK_SEMANTICS_V1 in context.contract.assembler_version
    for saved in (old_context, context):
        assert AssembledInitialContext.from_value(saved.to_value()) == saved
    assert "task_semantic_guidance" not in old_context.contract.meta


def test_trivia_context_contract_and_hotpot_deliberation_are_explicit():
    closed = public_task_semantics("triviaqa", input_profile=TRAINING_PUBLIC_INPUT)
    reading = public_task_semantics("triviaqa", input_profile=RELEASED_PUBLIC_INPUT)
    assert "reading context" not in closed
    assert "evidence" not in closed
    assert "reading context" in reading
    for profile in (TRAINING_PUBLIC_INPUT, RELEASED_PUBLIC_INPUT):
        assert "passage's wording" in public_task_semantics("hotpotqa", input_profile=profile)
    with pytest.raises(ValueError):
        public_task_semantics("triviaqa", input_profile="unspecified")
    from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark

    on, _, _ = shared_context(Protocol13Benchmark.HOTPOT_QA)
    off, _, _ = shared_context(Protocol13Benchmark.HOTPOT_QA, hotpot_deliberation=False)
    assert HOTPOT_DELIBERATION in on.text
    assert HOTPOT_DELIBERATION not in off.text


def test_candidates_roundtrip_without_reinterpreting_prior_conditions(tmp_path):
    config = shared_formal()
    path = tmp_path / "synthetic-condition.yaml"
    path.write_text(yaml.safe_dump(config.to_value()))
    assert BayesianFormalConfig.load(path) == config
    rollout = config.application_config("synthetic-shared").trainer.rollout
    assert rollout.format == "skillev-policy-rollout@9"
    assert rollout.hotpot_deliberation is config.hotpot_deliberation
    assert PolicyRolloutConfig.from_value(rollout.to_value()) == rollout
    assert PUBLIC_TASK_SEMANTICS_V1 in config.condition
    assert PUBLIC_TASK_SEMANTICS_V1 in rollout.condition_id
    for version in ("@4", "@5"):
        old = BayesianFormalConfig(format="skillev-bayesian-formal-training" + version)
        assert "task_semantic_guidance" not in old.to_value()
        path.write_text(yaml.safe_dump(old.to_value()))
        assert BayesianFormalConfig.load(path) == old
        with pytest.raises(ValueError):
            replace(old, task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V1)
    old_rollout = replace(
        rollout,
        format="skillev-policy-rollout@8",
        task_semantic_guidance="legacy",
        hotpot_deliberation=False,
    )
    assert "task_semantic_guidance" not in old_rollout.to_value()
    assert PolicyRolloutConfig.from_value(old_rollout.to_value()) == old_rollout
    arm = InferenceArm(
        "synthetic-shared",
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V1,
        hotpot_deliberation=True,
    )
    assert arm.to_value()["format"] == "skillev-policy-integrity@6"
    assert decode_integrity_arm(arm.to_value()) == arm
    old_arm = InferenceArm("synthetic-old")
    assert "task_semantic_guidance" not in old_arm.to_value()
    assert decode_integrity_arm(old_arm.to_value()) == old_arm


@pytest.mark.parametrize(
    "domain", ["hotpotqa", "triviaqa", "mbpp-plus", "humaneval", "aime-2026", "healthbench"]
)
def test_selected_actor_really_receives_shared_instruction_and_original_public_input(
    monkeypatch, domain
):
    from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark

    _, task, _ = shared_context(Protocol13Benchmark(domain))
    if domain == "healthbench":
        task = replace(
            task, model_visible_messages=(ModelVisibleMessage("user", "Synthetic dialogue."),)
        )
    view = PublicTaskView.from_training_task(task)
    tokenizer = CleanTokenizer()
    result = {
        "hotpotqa": "Final answer: synthetic",
        "triviaqa": "Final answer: synthetic",
        "mbpp-plus": "def fictional(x):\n    return x\n",
        "humaneval": "def fictional(x):\n    return x\n",
        "aime-2026": "123",
        "healthbench": "Synthetic conversation response.",
    }[domain]
    generator = ScriptedGenerator([(GenerationPhase.ACTION, result)], tokenizer=tokenizer)
    monkeypatch.setattr(integrity_actor, "BrokerGenerator", lambda descriptor: generator)
    monkeypatch.setattr(integrity_actor, "rpc", lambda *args, **kwargs: None)
    initial = initial_task("alfworld", "unused synthetic environment")
    initial["public_task"] = asdict(view)
    arm = InferenceArm("synthetic-shared", task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V1)
    initial["arm"] = arm.to_value()
    candidate = asyncio.run(integrity_actor.run_episode(initial))
    expected = (
        "synthetic"
        if domain in {"hotpotqa", "triviaqa"}
        else r"\boxed{123}"
        if domain == "aime-2026"
        else result.rstrip()
    )
    assert candidate.text == expected
    assert candidate.intervention_counts["model_calls"] == 1
    assert candidate.intervention_counts["peer_model_calls"] == 0
    rendered = json.dumps(tokenizer.messages[0][0])
    assert public_task_semantics(domain, input_profile=TRAINING_PUBLIC_INPUT) in rendered
    assert ("Synthetic dialogue." if domain == "healthbench" else task.query) in rendered


@pytest.mark.parametrize("version", [PUBLIC_TASK_SEMANTICS_V1, PUBLIC_TASK_SEMANTICS_V4])
def test_alf_actor_shared_guidance_preserves_live_task_and_action_wire(monkeypatch, version):
    tokenizer = CleanTokenizer()
    generator = ScriptedGenerator([(GenerationPhase.ACTION, "Action: look")], tokenizer=tokenizer)
    monkeypatch.setattr(integrity_actor, "BrokerGenerator", lambda descriptor: generator)
    executed = []

    def broker(operation, **arguments):
        if operation == "environment-reset":
            return {
                "observation_text": "Synthetic room.\nYour task is to: Find the amber marker.\n",
                "available_actions": ["look"],
            }
        if operation == "environment-step":
            executed.append(arguments["action"])
            return {
                "observation": "Synthetic terminal state.",
                "terminal": True,
                "action_valid": True,
                "available_actions": ["look"],
            }
        assert operation == "trace"

    monkeypatch.setattr(integrity_actor, "rpc", broker)
    initial = initial_task("alfworld", "Stale fictional task.")
    initial["arm"] = InferenceArm("synthetic-shared", task_semantic_guidance=version).to_value()
    candidate = asyncio.run(integrity_actor.run_episode(initial))
    assert json.loads(candidate.text) == executed == ["look"]
    rendered = json.dumps(tokenizer.messages[0][0])
    assert (
        public_task_semantics("alfworld", input_profile=RELEASED_PUBLIC_INPUT, version=version)
        in rendered
    )
    assert "Find the amber marker." in rendered
    assert "Stale fictional task." not in rendered


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("library", [False, True])
def test_shared_evaluation_arm_preserves_policy_and_skill_bindings(trained, library):
    from skillev.evaluation.step0_integrity import SkillMode
    from skillev.evolution.skill_access import ACTIVE_APPLICABILITY_RULE

    arm = InferenceArm(
        "synthetic-shared-library",
        optimizer_steps=1 if trained else 0,
        policy_id="synthetic-policy" if trained else None,
        skill_mode=SkillMode.LIBRARY if library else SkillMode.OFF,
        skill_library_id="synthetic-library" if library else None,
        skill_retrieval_rule=ACTIVE_APPLICABILITY_RULE if library else None,
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V1,
    )
    assert decode_integrity_arm(arm.to_value()) == arm
