"""Synthetic CPU-only transports. No benchmark data or actual HTTP."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import qualify_catalog_authoring as run
from skillev.evolution import (
    AuthoredSkillDraft,
    AuthoringActionKind,
    AuthoringFailedError,
    AuthoringResult,
    AuthoringSamplingConfig,
    EvolutionConfig,
)
from skillev.evolution.authoring import uncovered_edge_requirement_id
from skillev.evolution.execution import authoring_request_for_proposal
from skillev.rollout import GenerationPhase, RolloutGenerationResult
from skillev.runtime import (
    BudgetVector,
    SkillApplicability,
    SkillLibrary,
    SkillLibraryState,
    SkillRequirement,
)
from skillev.runtime.request_journal import DurableRequestJournal, UnknownRequestOutcomeError
from skillev.training.config import conservative_rollout_maximum
from tests.v3_helpers import CharacterTokenizer, make_skill_document


def test_controlled_example_uses_the_actual_public_completion_not_an_invented_tool():
    initial = SkillLibraryState.from_seed_documents((make_skill_document("seed"),))
    _, decision, authority = run.declared_control(SkillLibrary(initial))
    edge = decision.proposals[0].edge_exemplars[0]
    assert edge.action_kind is AuthoringActionKind.COMPLETE
    assert not edge.available_tools
    assert run.development_task().action_surface.completion is not None


class Transport:
    def __init__(self, tokenizer, result=None, fail=False):
        self.tokenizer, self.result, self.fail, self.calls = tokenizer, result, fail, 0

    def request(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise TimeoutError("synthetic unknown remote response")
        assert "lora_path" not in kwargs["payload"]
        text = json.dumps(self.result.to_value())
        tokens = self.tokenizer.encode(text)
        return 200, {
            "text": text,
            "output_ids": [*tokens, 0],
            "meta_info": {
                "prompt_tokens": len(kwargs["payload"]["input_ids"]),
                "completion_tokens": len(tokens) + 1,
                "finish_reason": {"type": "stop", "matched": 0},
            },
        }


def setup(root, tokenizer):
    initial = SkillLibraryState.from_seed_documents((make_skill_document("seed"),))
    phase, decision, authority = run.declared_control(SkillLibrary(initial))
    request = authoring_request_for_proposal(
        decision.proposals[0],
        proposal_index=0,
        phase_event=phase,
        library=SkillLibrary(initial),
        authority=authority,
        base_seed=0,
        cycle_ordinal=1,
    )
    result = AuthoringResult(
        (
            AuthoredSkillDraft(
                title="Public documentation procedure",
                summary="Read and distinguish observed from unknown.",
                instructions=(
                    "Read the public documentation. Report actual observations; "
                    "do not invent usefulness."
                ),
                applicability=SkillApplicability((run.FAMILY,), (run.CONTEXT,), (), ()),
                requirements=(
                    SkillRequirement(
                        uncovered_edge_requirement_id(request.edge_exemplars[0].edge_id),
                        "Read public material and identify uncertainty.",
                        "evolvable-strategy",
                    ),
                ),
            ),
        )
    )
    plan = {
        "endpoint": "http://127.0.0.1:9",
        "base_model": "synthetic-base",
        "authoring_sampling": AuthoringSamplingConfig(1.0, 1.0).to_value(),
        "evolution": EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            max_authoring_prompt_tokens=100000,
            max_authoring_completion_tokens=10000,
        ).to_value(),
    }
    run.initialize(root, plan, initial)
    return plan, result


def test_new_process_recovers_exact_author_and_mutation_without_send(tmp_path):
    root = tmp_path / "development"
    tokenizer = CharacterTokenizer()
    plan, result = setup(root, tokenizer)
    transport = Transport(tokenizer, result)
    first, skill = run.recover_mutation(root, plan, tokenizer, transport=transport)
    assert transport.calls == 1
    assert skill in first.active_skill_ids
    original = (root / "accepted-author-result.json").read_bytes()
    command = (
        "import pathlib; from scripts import qualify_catalog_authoring as r; "
        "from tests.v3_helpers import CharacterTokenizer; "
        f"p=pathlib.Path({str(root)!r}); "
        'r.recover_mutation(p,r.read(p/"plan.json"),CharacterTokenizer(),resumed=True)'
    )
    subprocess.run(  # noqa: S603 -- fixed CPU child interpreter/code and pytest-owned path
        [sys.executable, "-c", command],
        check=True,
        cwd=Path(run.__file__).resolve().parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=30,
    )
    restored = run.read(root / "resume-author-summary.json")
    assert restored["physical_author_calls"] == 0
    assert restored["driver_pid"] != os.getpid()
    assert restored["library_version"] == first.current_version
    assert (root / "accepted-author-result.json").read_bytes() == original
    assert len(list(root.glob("author-http-*.json"))) == 1


def test_unknown_dispatch_is_not_replaced(tmp_path):
    root = tmp_path / "unknown"
    tokenizer = CharacterTokenizer()
    plan, _ = setup(root, tokenizer)
    transport = Transport(tokenizer, fail=True)
    with pytest.raises(TimeoutError):
        run.recover_mutation(root, plan, tokenizer, transport=transport)
    with pytest.raises(UnknownRequestOutcomeError):
        run.recover_mutation(root, plan, tokenizer, resumed=True)
    assert transport.calls == 1
    assert not (root / "mutated-library.json").exists()


def test_invalid_author_does_not_install_or_resample(tmp_path):
    root = tmp_path / "invalid"
    tokenizer = CharacterTokenizer()
    plan, result = setup(root, tokenizer)
    invalid = replace(
        result,
        drafts=(
            replace(
                result.drafts[0],
                applicability=SkillApplicability(("wrong-family",), (run.CONTEXT,), (), ()),
            ),
        ),
    )
    transport = Transport(tokenizer, invalid)
    with pytest.raises(AuthoringFailedError):
        run.recover_mutation(root, plan, tokenizer, transport=transport)
    with pytest.raises(AuthoringFailedError):
        run.recover_mutation(root, plan, tokenizer, resumed=True)
    assert transport.calls == 1
    assert not (root / "mutated-library.json").exists()


@pytest.mark.parametrize("do_read", [True, False])
def test_actual_native_execute_episode_returns_body_then_visible_input(
    tmp_path, make_training_harness, do_read
):
    harness = make_training_harness()
    tokenizer = harness.generator.tokenizer
    root = tmp_path / "reading"
    plan, result = setup(root, tokenizer)
    library, skill = run.recover_mutation(
        root, plan, tokenizer, transport=Transport(tokenizer, result)
    )
    rollout = replace(
        harness.config.rollout,
        format="skillev-policy-rollout@9",
        phase_context=True,
        action_wire="native-single-tool-call@3",
        skill_exposure="catalog-then-read@1",
        public_action_semantics=True,
        task_semantic_guidance="legacy",
        reasoning_tool_catalog=True,
        token_budget_notice=True,
        max_turns=3,
        max_reasoning_tokens=128,
        max_action_tokens=1024,
        per_rollout_maximum=conservative_rollout_maximum(
            max_turns=3,
            max_reasoning_tokens=128,
            max_action_tokens=1024,
            max_model_input_tokens=16000,
            max_tool_wall_time_milliseconds=1000,
        ),
    )
    plan.update(rollout=rollout.to_value(), max_input_tokens=16000)

    class Generator:
        def __init__(self):
            self.calls = []
            self.actions = 0
            self.tokenizer = tokenizer

        def snapshot(self):
            return harness.generator.snapshot()

        def begin_episode(self, *args):
            pass

        def end_episode(self, *args):
            pass

        async def generate(self, request):
            self.calls.append(request)
            if request.phase is GenerationPhase.REASONING:
                text = "Inspect available public documentation without an unsupported claim."
            else:
                self.actions += 1
                name, arguments = (
                    ("read_skill", {"skill_id": skill})
                    if do_read and self.actions == 1
                    else ("submit_answer", {"answer": "Read-only integration status."})
                )
                text = (
                    "<tool_call>"
                    + json.dumps({"name": name, "arguments": arguments})
                    + "</tool_call>"
                )
            ids = tuple(tokenizer.encode(text))
            return RolloutGenerationResult(
                ids,
                (),
                "stop",
                self.snapshot().snapshot_id,
                self.snapshot().backend_id,
                BudgetVector(
                    input_tokens=len(request.input_ids), output_tokens=len(ids), model_calls=1
                ),
            )

    generator = Generator()
    if do_read:
        report = asyncio.run(run.read_episode(root, plan, library, skill, generator))
        assert report["catalog_visible"]
        assert report["catalog_visible_before_read"]
        assert report["returned_read_step_indices"] == [1]
        assert report["body_visible_actual_input_evidence"]
        assert report["followed_actions"][0]["index"] == 2
        content = result.drafts[0].instructions
        assert any(content in tokenizer.decode(r.input_ids) for r in generator.calls[2:])
    else:
        with pytest.raises(RuntimeError):
            asyncio.run(run.read_episode(root, plan, library, skill, generator))
        report = run.read(root / "read-result.json")
        assert not report["qualified_catalog_read_visibility"]
    assert report["followed_instructions_or_usefulness"] is None
    assert report["posterior_updates"] == 0
    assert not report["natural_phi_trigger"]


def test_resume_cli_rejects_condition_override_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["qualify", "resume-read", "--root", str(tmp_path), "--endpoint", "http://127.0.0.1:9"],
    )
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert not (tmp_path / "author-requests.sqlite3").exists()


def test_resume_cli_cannot_replace_already_started_read(tmp_path, monkeypatch):
    run.preserve(tmp_path / "plan.json", {})
    run.preserve(tmp_path / "read-started.json", {"driver_pid": 123})
    monkeypatch.setattr(sys, "argv", ["qualify", "resume-read", "--root", str(tmp_path)])
    with pytest.raises(RuntimeError):
        run.main()
    assert not (tmp_path / "actor-requests.sqlite3").exists()


def test_real_generator_constructor_explicitly_binds_adapter_free_route(
    tmp_path, make_training_harness
):
    harness = make_training_harness()
    plan = {
        "endpoint": "http://127.0.0.1:9",
        "policy": harness.generator.snapshot().to_value(),
    }
    generator, transport = run.read_generator(tmp_path, plan, harness.generator.tokenizer)
    try:
        assert generator.gateway is None
        assert generator.snapshot() == harness.generator.snapshot()
        assert generator.request_journal is not None
        assert transport.calls == 0
    finally:
        generator.close()


def test_preflight_recovery_preserves_failed_attempt_without_any_dispatch(tmp_path, monkeypatch):
    def stopped(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", stopped)
    before = {"driver_pid": 456, "started_at": "synthetic-preflight"}
    run.preserve(tmp_path / "read-started.json", before)
    DurableRequestJournal(tmp_path / "actor-requests.sqlite3")
    run.resume_before_dispatch(tmp_path)
    assert not (tmp_path / "read-started.json").exists()
    assert run.read(tmp_path / "read-preflight-failed-456.json") == before


@pytest.mark.parametrize("prior", ["live", "input", "unknown", "completed", "artifact"])
def test_preflight_recovery_never_replaces_active_or_prepared_or_sampled_read(
    tmp_path, monkeypatch, prior
):
    def status(pid, signal):
        if prior != "live":
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", status)
    before = {"driver_pid": 456, "started_at": "synthetic-preflight"}
    run.preserve(tmp_path / "read-started.json", before)
    if prior == "input":
        run.preserve(tmp_path / "actor-input-001.json", {"input_ids": [1]})
    elif prior == "artifact":
        run.preserve(tmp_path / "read-artifact.json", {"synthetic": True})
    elif prior in {"unknown", "completed"}:
        journal = DurableRequestJournal(tmp_path / "actor-requests.sqlite3")

        def send():
            if prior == "unknown":
                raise TimeoutError
            return 200, {"output_ids": [2]}

        try:
            journal.request(
                identity=("synthetic-episode", "1", "reasoning"),
                endpoint="http://127.0.0.1:9/generate",
                payload={"input_ids": [1]},
                send=send,
            )
        except TimeoutError:
            assert prior == "unknown"
    with pytest.raises(RuntimeError):
        run.resume_before_dispatch(tmp_path)
    assert run.read(tmp_path / "read-started.json") == before
