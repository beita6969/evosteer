"""Native execution may read training public inputs without pretending they are IID RC."""

import asyncio
from dataclasses import asdict

import pytest

from skillev.evaluation import integrity_actor
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.rollout import GenerationPhase, ModelVisibleMessage, RolloutTask
from tests.evaluation.test_integrity_actor_reset import ScriptedGenerator, initial_task
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer


def task(domain, messages=()):
    return RolloutTask(
        task_id="synthetic",
        environment_id="synthetic-env",
        task_family=domain,
        context_id="synthetic-context",
        query='Public scaffold: def f(x="a\\\\b"):\n    """Keep café."""',
        available_tools=(),
        public_context={"benchmark_id": domain, "payload": {"released": "public only"}},
        model_visible_messages=messages,
    )


@pytest.mark.parametrize(
    "domain", ["hotpotqa", "triviaqa", "aime-2026", "mbpp-plus", "humaneval", "alfworld"]
)
def test_training_view_preserves_verbatim_public_material_and_declares_its_condition(domain):
    source = task(domain)
    view = PublicTaskView.from_training_task(source)
    assert source.query in view.render()
    assert "public only" in view.render()
    assert view.input_profile == "training-public-source-bridge@1"
    rebuilt = PublicTaskView(**{**asdict(view), "fields": tuple(view.fields)})
    assert rebuilt == view
    source.public_context["payload"]["released"] = "changed later"
    assert "changed later" not in view.render()


def test_diagnostic_closed_book_does_not_weaken_released_trivia_rc_requirement():
    assert PublicTaskView.from_training_task(task("triviaqa"))
    with pytest.raises(ValueError):
        PublicTaskView.from_record("synthetic", "triviaqa", {"question": "No released context"})
    with pytest.raises(ValueError):
        PublicTaskView.from_training_task(task("webshop"))


def test_original_dialogue_roles_reach_the_native_actor(monkeypatch):
    messages = (
        ModelVisibleMessage("system", "Original public system."),
        ModelVisibleMessage("user", "Original public question."),
        ModelVisibleMessage("assistant", "Original clarification?"),
        ModelVisibleMessage("user", "Original final user turn."),
    )
    view = PublicTaskView.from_training_task(task("healthbench", messages))
    assert view.conversation() == tuple({"role": m.role, "content": m.content} for m in messages)
    tokenizer = CleanTokenizer()
    generator = ScriptedGenerator(
        [(GenerationPhase.ACTION, "A public response.")], tokenizer=tokenizer
    )
    monkeypatch.setattr(integrity_actor, "BrokerGenerator", lambda descriptor: generator)
    monkeypatch.setattr(integrity_actor, "rpc", lambda *args, **kwargs: None)
    initial = initial_task("alfworld", "unused synthetic environment")
    initial["public_task"] = asdict(view)
    initial["arm"] = InferenceArm("synthetic-dialogue").to_value()
    candidate = asyncio.run(integrity_actor.run_episode(initial))
    assert candidate.text == "A public response."
    rendered = tokenizer.messages[0][0]
    for source in view.conversation():
        assert any(
            source["content"] in m["content"] and source["role"] == m["role"] for m in rendered
        )
    assert candidate.intervention_counts["peer_model_calls"] == 0
