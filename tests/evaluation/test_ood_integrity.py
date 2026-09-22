"""Synthetic OOD coverage for the existing single-owner evaluation boundary."""

import asyncio
import base64
import json
import pickle
import shutil
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldStepResult
from skillev_private.evaluation.ood_export import (
    decode_lcb_tests,
    export_sources,
    project_row,
    sample_rows,
)
from skillev_private.evaluation.ood_luna_judge import (
    JUDGE_EFFORT,
    JUDGE_MODEL,
    JUDGE_PROFILE,
    read_judgement,
)
from skillev_private.evaluation.ood_scienceworld import OODScienceWorldEnvironment
from skillev_private.evaluation.ood_scoring import grade_code, score_ood
from skillev_private.evaluation.ood_sources import load_ood_panel

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.decision_transport import (
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
)
from skillev.evaluation.direct_baseline.interactive_tasks import (
    NativeEnvironmentStep,
    NativePublicState,
)
from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS, OOD_BENCHMARKS, PublicTaskView
from skillev.evaluation.integrity_pipeline import FrozenPanel
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.evaluation.thinking_policy import ThinkingPolicy
from tests.evaluation.test_integrity_broker_boundary import runtime


def test_result_blind_sampling_keeps_source_order_and_not_answers():
    rows = [{"question": str(i), "answer": ["private"]} for i in range(300)]
    selected, population = sample_rows(rows, 64, 0)
    changed, _ = sample_rows([{**row, "answer": ["changed"]} for row in rows], 64, 0)
    assert population == 300
    indices = [i for i, _ in selected]
    assert indices == sorted(indices)
    assert indices == [i for i, _ in changed]
    assert len(set(indices)) == 64
    assert indices != list(range(64))


def test_ood_export_freezes_64_and_keeps_answers_out_of_public_view(tmp_path):
    raw = tmp_path / "source.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps({"question": f"Synthetic {i}", "answer": ["PRIVATE"]}) for i in range(80)
        )
    )
    exported = export_sources(
        {
            "sample_count": 64,
            "seed": 0,
            "sources": {"nq-open": {"paths": [str(raw)], "provenance": {"split": "synthetic"}}},
        },
        tmp_path / "frozen",
    )
    source = load_ood_panel(exported)
    source.panel.validate(canary=False)
    assert len(source.panel.entries) == 64
    assert source.panel.catalog == OOD_BENCHMARKS
    assert all("PRIVATE" not in entry.render() for entry in source.panel.entries)
    assert all(value["accepted_answers"] == ["PRIVATE"] for value in source.targets.values())
    assert "webshop" not in IID_BENCHMARKS
    with pytest.raises(ValueError):
        FrozenPanel(source.panel.entries, "synthetic", "synthetic", (("nq-open", 64),)).validate(
            canary=False
        )


def test_musique_public_context_excludes_support_flags_and_decomposition():
    projected = project_row(
        "musique",
        0,
        {
            "id": "synthetic",
            "question": "Where?",
            "answer": "PRIVATE",
            "answer_aliases": ["SECRET"],
            "question_decomposition": ["GOLD PLAN"],
            "paragraphs": [
                {"title": "Public", "paragraph_text": "Released passage", "is_supporting": True}
            ],
        },
    )
    view = PublicTaskView.from_record(projected["task_id"], "musique", projected["public"])
    assert "Released passage" in view.render()
    assert not any(
        text in view.render() for text in ["PRIVATE", "SECRET", "GOLD PLAN", "is_supporting"]
    )


def test_lcb_decodes_both_official_test_encodings_without_executing_objects():
    value = json.dumps([{"input": "1", "output": "2"}])
    compressed = base64.b64encode(zlib.compress(pickle.dumps(value))).decode()
    assert decode_lcb_tests(value) == decode_lcb_tests(compressed)
    object_pickle = base64.b64encode(zlib.compress(pickle.dumps(Path("/unavailable")))).decode()
    with pytest.raises(ValueError):
        decode_lcb_tests(object_pickle)


def test_code_public_interface_never_contains_hidden_tests_or_reference_solutions():
    projected = project_row(
        "apps-introductory",
        0,
        {
            "difficulty": "introductory",
            "question": "Implement double.",
            "starter_code": "def double(x):",
            "solutions": "PRIVATE SOLUTION",
            "input_output": json.dumps({"inputs": [[9]], "outputs": [18], "fn_name": "double"}),
        },
    )
    assert "def double(x):" in projected["public"]["prompt"]
    assert "18" not in projected["public"]["prompt"]
    assert "PRIVATE SOLUTION" not in str(projected)
    assert projected["target"]["input_output"]["outputs"] == [18]


def test_all_ood_thinking_is_off_without_changing_iid_authority():
    policy = ThinkingPolicy("ood-thinking-off", tuple((b, False) for b in OOD_BENCHMARKS))
    for benchmark in OOD_BENCHMARKS:
        arm = policy.resolve(benchmark, InferenceArm("ood"))
        assert not arm.native_thinking
        arm.validate_live_topology()


def test_scienceworld_exposes_native_command_not_an_oracle_action_list():
    surface = PublicSurface("scienceworld", 1, ("act",))
    response = (
        "<tool_call><function=act><parameter=command>open blue door"
        "</parameter></function></tool_call>"
    )
    assert normalize_decision(ExplicitDecision(response, 1), surface).action == "open blue door"
    names = [tool["function"]["name"] for tool in native_tool_definitions("scienceworld")]
    assert "act" in names
    assert "solver" not in names

    class Environment:
        task_description = "Synthetic science goal"

        def reset(self, **kwargs):
            return "A public room."

        def step(self, action):
            assert action == "open blue door"
            return OfficialScienceWorldStepResult("Cannot open that.", 25.0, False)

    async def run():
        environment = OODScienceWorldEnvironment(Environment(), "synthetic", 0, 0, 200)
        state = await environment.reset()
        assert "Synthetic science goal" in state.observation_text
        assert state.available_actions == ("act",)
        result = await environment.step("open blue door")
        assert result.reward == 0.25
        assert result.available_actions == ("act",)
        assert (await environment.outcome()).reward == 0.25

    asyncio.run(run())


@pytest.mark.parametrize(
    "text", ["The experiment is finished.", "go kitchen", "act", "`go kitchen`"]
)
def test_scienceworld_chat_text_requires_an_explicit_action_carrier(text):
    result = normalize_decision(
        ExplicitDecision(text, 1), PublicSurface("scienceworld", 1, ("act",))
    )
    assert result.action is None
    assert result.error


@pytest.mark.parametrize(
    "carrier",
    [
        'act("{command}")',
        'scienceworld.act(command="{command}")',
        'act command: "{command}"',
        "Action: {command}",
        "<tool_call><function=act><parameter=command>{command}</parameter></function></tool_call>",
    ],
)
@pytest.mark.parametrize("command", ["go kitchen", "The experiment is finished."])
def test_scienceworld_explicit_carriers_do_not_filter_command_meaning(carrier, command):
    result = normalize_decision(
        ExplicitDecision(carrier.format(command=command), 1),
        PublicSurface("scienceworld", 1, ("act",)),
    )
    assert result.action == command
    assert result.error is None


@pytest.mark.parametrize(
    ("observation", "valid"),
    [
        ("No known action matches that input.", False),
        ("Unknown action.  Type 'help' for a list of actions, and 'objects' for referents.", False),
        ("Ambiguous request: Please enter the number for the action you intended.", None),
        ("Cannot open that.", None),
    ],
)
def test_scienceworld_preserves_explicit_native_rejection_without_oracle(observation, valid):
    actions = []

    class Environment:
        def step(self, action):
            actions.append(action)
            return OfficialScienceWorldStepResult(observation, 25.0, False)

    async def run():
        environment = OODScienceWorldEnvironment(Environment(), "synthetic", 0, 0, 200)
        result = await environment.step("arbitrary owner command")
        assert actions == ["arbitrary owner command"]
        assert result.observation == observation
        assert result.action_valid is valid
        assert result.reward == 0.25
        assert not result.terminal

    asyncio.run(run())


@pytest.mark.parametrize("command", ["", "   ", "\x00", "look\x00around"])
@pytest.mark.parametrize("wrapper", ["xml", "json", "python"])
def test_scienceworld_rejects_unexecutable_command_before_worker(command, wrapper):
    if wrapper == "xml":
        response = (
            "<tool_call><function=act><parameter=command>"
            + command
            + "</parameter></function></tool_call>"
        )
    elif wrapper == "json":
        response = json.dumps(
            {
                "kind": "tool",
                "resource_id": "scienceworld",
                "name": "act",
                "arguments": {"command": command},
            }
        )
    else:
        response = f"act({command!r})"
    result = normalize_decision(
        ExplicitDecision(response, 1), PublicSurface("scienceworld", 1, ("act",))
    )
    assert result.action is None
    assert result.error


@pytest.mark.parametrize("benchmark", ["musique", "nq-open"])
def test_ood_qa_uses_alias_aware_native_em_f1(benchmark):
    reader = SimpleNamespace(
        get=lambda *args: SimpleNamespace(text="The blue bird", episode_id="e")
    )
    score = asyncio.run(
        score_ood(
            reader,
            ("r", "a", "e"),
            benchmark,
            {"accepted_answers": ["blue bird", "azure bird"]},
            settings={},
            sandbox=None,
            diagnostics=lambda value: None,
        )
    )
    assert score.value == score.secondary_metrics["answer-exact-match"] == 1.0


def test_official_code_checker_runs_only_in_the_separate_scorer_sandbox(tmp_path, monkeypatch):
    checker = tmp_path / "checker.py"
    checker.write_text("# synthetic checker")
    sandbox = ActorSandbox.current(Path("src"))
    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"results": [true, false]}', stderr="")

    monkeypatch.setattr("skillev_private.evaluation.ood_scoring.subprocess.run", run)
    result = grade_code(
        "livecodebench",
        "print(2)",
        {"input_output": {"inputs": ["1", "2"], "outputs": ["2", "4"]}},
        {"checker_path": str(checker), "wall_timeout_seconds": 20},
        sandbox,
    )
    assert not result["passed"]
    command, kwargs = seen[0]
    assert "--unshare-all" in command
    assert str(checker) in command
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert json.loads(kwargs["input"])["candidate"] == "print(2)"
    assert kwargs["timeout"] == 20
    assert json.loads(kwargs["input"])["test_timeout"] == 6


def test_checker_keeps_real_stderr_for_native_faulthandler(tmp_path):
    if shutil.which("bwrap") is None:
        pytest.skip("the native checker requires bubblewrap")
    checker = tmp_path / "checker.py"
    checker.write_text(
        "import faulthandler\n"
        "def run_test(sample, test=None, debug=False, timeout=6):\n"
        "    faulthandler.enable()\n"
        "    return [True], {}\n"
    )
    sandbox = ActorSandbox.current(Path(__file__).resolve().parents[2] / "src")
    result = grade_code(
        "livecodebench",
        "print(2)",
        {"input_output": {"inputs": ["1"], "outputs": ["2"]}},
        {"checker_path": str(checker)},
        sandbox,
    )
    assert result["passed"]


@pytest.mark.parametrize(
    ("benchmark", "public", "response", "expected"),
    [
        (
            "musique",
            {"question": "Where?", "context": "Synthetic passage."},
            "The supplied passage identifies the requested color.\nFinal answer: blue",
            "blue",
        ),
        (
            "nq-open",
            {"question": "What color?"},
            "The question asks for a color rather than an explanation.\nFinal answer: blue",
            "blue",
        ),
        (
            "musique",
            {"question": "Where?", "context": "Synthetic passage."},
            "I considered green.\n**Answer: blue**\nAn explanation follows.",
            "blue",
        ),
        (
            "nq-open",
            {"question": "What color?"},
            "Answer: green\n**Short Answer**: blue\nAn explanation follows.",
            "blue",
        ),
        ("omni-math", {"problem": "Synthetic arithmetic."}, "Final answer: The result is 4.", "4"),
        (
            "livecodebench",
            {"prompt": "Synthetic Python task."},
            "```python\nprint(2)\n```",
            "print(2)",
        ),
        (
            "apps-introductory",
            {"prompt": "Synthetic Python task."},
            "```python\nprint(2)\n```",
            "print(2)",
        ),
    ],
)
def test_ood_static_task_crosses_real_actor_broker_with_one_owner(
    tmp_path,
    benchmark,
    public,
    response,
    expected,
):
    entry = PublicTaskView.from_record("synthetic-ood", benchmark, public)
    instance = runtime(tmp_path, entry, [response])
    arm = InferenceArm("ood")
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert expected in final.text
        if benchmark in {"musique", "nq-open"}:
            assert final.text == expected
            assert final.submission["raw_response"] == response
        assert final.intervention_counts["model_calls"] == 1
        instructions = repr(instance.tokenizer.messages)
        if benchmark in {"musique", "nq-open"}:
            assert "short phrase" in instructions
            assert "answer alone" in instructions
        elif benchmark in {"livecodebench", "apps-introductory"}:
            assert "Bare Python source" in instructions
            assert "last is your submitted program" in instructions
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()


@pytest.mark.parametrize("reason_before_action", [False, True])
@pytest.mark.parametrize(
    "invalid_first",
    [
        None,
        "<tool_call><function=act><parameter=command></parameter></function></tool_call>",
        "The experiment is finished.",
    ],
)
def test_scienceworld_episode_uses_the_same_single_owner_broker(
    tmp_path, monkeypatch, invalid_first, reason_before_action
):
    entry = PublicTaskView.from_record(
        "science-fixture", "scienceworld", {"task": "Synthetic goal"}
    )
    responses = [
        ("I need a public observation before choosing an object.\n" if reason_before_action else "")
        + "Action: look around"
    ]
    if invalid_first is not None:
        responses.insert(0, invalid_first)
    instance = runtime(tmp_path, entry, responses)
    instance.source.interactive = {entry.task_id: {"case": {"max_steps": 200}}}

    class Environment:
        async def reset(self):
            return NativePublicState("Synthetic science observation.", ("act",))

        async def step(self, action):
            assert action == "look around"
            return NativeEnvironmentStep(
                "Native terminal observation.", True, 0.5, False, None, available_actions=("act",)
            )

        async def outcome(self):
            from skillev_private.evaluation.ood_scienceworld import OODScienceWorldOutcome

            return OODScienceWorldOutcome(
                0.5,
                False,
                True,
                native_final_score=50,
                native_steps=1,
                termination_reason="environment-terminal-unspecified",
            )

        async def close(self):
            pass

    monkeypatch.setattr(
        "skillev_private.evaluation.integrity_runtime.create_native_environment",
        lambda *args, **kwargs: Environment(),
    )
    arm = InferenceArm("ood")
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert json.loads(final.text) == ["look around"]
        assert final.intervention_counts["model_calls"] == len(responses)
        assert final.completion_tokens == sum(map(len, responses))
        assert len(instance.journal.model_outputs(("synthetic", "ood", entry.task_id))) == len(
            responses
        )
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
        score = asyncio.run(
            score_ood(
                instance.journal,
                ("synthetic", "ood", entry.task_id),
                "scienceworld",
                {},
                settings={},
                sandbox=instance.sandbox,
                diagnostics=lambda value: None,
            )
        )
        assert score.value == 0.5
        assert score.secondary_metrics["success"] == 0.0
    finally:
        instance.journal.close()


@pytest.mark.parametrize("verdict", [True, False, "uncertain"])
def test_luna_high_judge_requires_a_resolved_verdict_for_the_frozen_candidate(tmp_path, verdict):
    candidate = SimpleNamespace(run_id="run", arm_id="arm", episode_id="task", owner_call_id="call")
    path = tmp_path / "judgements.json"
    settings = {"judgements_path": str(path)}
    with pytest.raises(FileNotFoundError):
        read_judgement(candidate, settings)
    record = {
        "run_id": "run",
        "arm_id": "arm",
        "task_id": "task",
        "owner_call_id": "call",
        "equivalent": verdict,
        "reason": "Synthetic mathematical equivalence explanation.",
    }
    result = {
        "profile": JUDGE_PROFILE,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_EFFORT,
        "records": [record],
    }
    path.write_text(json.dumps(result))
    if isinstance(verdict, bool):
        assert read_judgement(candidate, settings)["passed"] is verdict
    else:
        with pytest.raises(ValueError):
            read_judgement(candidate, settings)
    record["owner_call_id"] = "another-final"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        read_judgement(candidate, settings)


def test_nq_primary_is_em_not_partial_token_f1():
    reader = SimpleNamespace(get=lambda *args: SimpleNamespace(text="blue", episode_id="e"))
    score = asyncio.run(
        score_ood(
            reader,
            ("r", "a", "e"),
            "nq-open",
            {"accepted_answers": ["blue bird"]},
            settings={},
            sandbox=None,
            diagnostics=lambda value: None,
        )
    )
    assert score.metric == "answer-exact-match"
    assert score.value == 0.0
    assert score.secondary_metrics["answer-f1"] > 0.0
