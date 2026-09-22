import asyncio
import json
from types import SimpleNamespace

import pytest
from skillev_private.experiments.phase_cross_diagnostic import phase_cross

from skillev.rollout import GenerationPhase, RolloutGenerationResult
from skillev.runtime import BudgetVector
from tests.v3_helpers import CharacterTokenizer


class Tokenizer(CharacterTokenizer):
    def encode_rollout_prompt_with_thinking(self, text):
        return self.encode("NATIVE_THINKING\n" + text)


class Actor:
    def __init__(self, name, *, fail=False, action_text=None):
        self.tokenizer = Tokenizer()
        self.pinned = SimpleNamespace(snapshot_id=name)
        self.calls, self.opened, self.closed = [], [], []
        self.fail = fail
        self.action_text = action_text

    def snapshot(self):
        return self.pinned

    def begin_episode(self, name, policy):
        assert policy == self.pinned.snapshot_id
        self.opened.append(name)

    def end_episode(self, name):
        self.closed.append(name)

    async def generate(self, request):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("synthetic unavailable service")
        text = (
            "reasoned-by-" + self.pinned.snapshot_id
            if request.phase is GenerationPhase.REASONING
            else self.action_text
            or '{"kind":"complete","name":"complete","arguments":{"value":{"answer":"x"}}}'
        )
        ids = tuple(self.tokenizer.encode(text))
        return RolloutGenerationResult(
            ids,
            (),
            "length",
            self.pinned.snapshot_id,
            "synthetic",
            BudgetVector(
                input_tokens=len(request.input_ids), output_tokens=len(ids), model_calls=1
            ),
        )


def run(initial, trained, initial_text="PUBLIC TASK\n"):
    return asyncio.run(
        phase_cross(
            initial=initial,
            trained=trained,
            diagnostic_id="synthetic-cross",
            initial_text=initial_text,
            previous_steps=(),
            library_version="library-one",
            decoding_snapshot_id="same-decoding",
            native_thinking=True,
            reasoning_seed=0,
            action_seed=0,
            max_reasoning_tokens=1024,
            max_action_tokens=2048,
            input_window=None,
        )
    )


def test_cross_uses_both_native_reasonings_and_all_four_actions_without_selection():
    initial, trained = Actor("initial"), Actor("trained")
    result = run(initial, trained)
    assert len(result["actions"]) == 4
    for name, actor in (("initial", initial), ("trained", trained)):
        assert len(actor.calls) == 3
        assert actor.opened == actor.closed
        reasoning = next(r for r in actor.calls if r.phase is GenerationPhase.REASONING)
        assert actor.tokenizer.decode(reasoning.input_ids).startswith("NATIVE_THINKING")
        action_calls = [r for r in actor.calls if r.phase is GenerationPhase.ACTION]
        for origin, call in zip(("initial", "trained"), action_calls, strict=True):
            text = actor.tokenizer.decode(call.input_ids)
            assert "reasoned-by-" + origin in text
            assert not text.startswith("NATIVE_THINKING")
            row = result["actions"][f"R-{origin}/A-{name}"]
            assert row["reasoning_policy_snapshot_id"] == origin
            assert row["action_policy_snapshot_id"] == name
            assert row["terminal_task_success"] is None
            assert row["input_ids"] == list(call.input_ids)
            assert call.seed == 0
            assert call.max_new_tokens == 2048
    assert initial.calls[0].input_ids == trained.calls[1].input_ids


def test_cross_failure_releases_episode_and_does_not_resample():
    initial, trained = Actor("initial", fail=True), Actor("trained")
    with pytest.raises(RuntimeError):
        run(initial, trained)
    assert len(initial.calls) == 1
    assert initial.opened == initial.closed
    assert trained.calls == []


@pytest.mark.parametrize(
    "version",
    ["native-single-tool-call@1", "native-single-tool-call@2", "native-single-tool-call@3"],
)
def test_native_cross_uses_declared_codec_and_stop_instead_of_json_diagnostic(version):
    from skillev.policy.phase_context import PhaseContextSpec
    from tests.rollout.test_native_tool_wire import call, contract

    public_contract = contract()
    initial_text = PhaseContextSpec(
        action_wire=version,
        tools_json=json.dumps(public_contract.to_native_tools()),
        action_contract_json=json.dumps(public_contract.to_scoring_metadata()),
    ).wrap("PUBLIC TASK\n")
    action = call("submit_answer", answer="x")
    actors = Actor("initial", action_text=action), Actor("trained", action_text=action)
    result = run(*actors, initial_text=initial_text)
    assert all(row["parse_status"] == "valid" for row in result["actions"].values())
    for actor in actors:
        assert all(c.action_boundary_version == "native-model-stop@1" for c in actor.calls)
    assert len(result["actions"]) == 4
