"""Checking roles see the draft, feedback reaches reruns, and truncation is observable."""

import asyncio
from dataclasses import replace

import pytest

from skillev.contracts.canonical import stable_hash
from skillev.orchestration.actions import GraphAction
from skillev.orchestration.actions import GraphActionKind as K
from skillev.orchestration.evosteer_features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    answer_present,
    extract_features,
    normalized_answer,
)
from skillev.orchestration.execution import DRAFT_MESSAGE_PROTOCOL, GraphRuntime
from skillev.orchestration.graph import NodeExecutionRequest, NodeExecutionResult, RoleSpec
from skillev.orchestration.model_executor import FrozenTextExecutor
from skillev.policy.sglang_text import SGLangFrozenText
from skillev.runtime.budget_ledger import BudgetLedger
from skillev.runtime.contracts import BudgetVector

MAXIMUM = BudgetVector(
    input_tokens=64,
    output_tokens=8,
    model_calls=1,
    agent_turns=1,
    wall_time_milliseconds=10_000,
)
USAGE = BudgetVector(input_tokens=2, output_tokens=3, model_calls=1, agent_turns=1)


class ScriptedExecutor:
    """Solvers answer 'wrong' until they receive a verdict; checkers judge the draft."""

    frozen_identity = "scripted-draft-executor@1"

    def __init__(self):
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        bodies = [message["body"] for message in request.messages]
        if request.role.role_id in {"verifier", "alpha-checker"}:
            output = "verdict: incorrect" if "wrong" in bodies else "verdict: correct"
        else:
            output = "right" if any(b.startswith("verdict:") for b in bodies) else "wrong"
        return NodeExecutionResult(output, USAGE)


def build(roles=("solver", "verifier"), *, executor=None):
    executor = executor or ScriptedExecutor()
    ledger = BudgetLedger(run_id="v7", attempt_id="episode", cap=MAXIMUM.scale(40))
    runtime = GraphRuntime(
        task_prompt="Produce the right result.",
        roles=tuple(RoleSpec(role, f"Act as {role}.", MAXIMUM) for role in roles),
        skills={"repair": "Repair the draft."},
        executor=executor,
        ledger=ledger,
        evaluator=lambda output: float(output == "right"),
        max_nodes=None,
        max_actions=None,
        runtime_id="v7-episode",
        seed=5,
    )
    return runtime, executor, ledger


def add(node_id, role_id="solver", skill_id=None):
    return GraphAction(K.ADD_AGENT, node_id=node_id, role_id=role_id, skill_id=skill_id)


def run(coroutine):
    return asyncio.run(coroutine)


def feature(runtime, name):
    return dict(zip(FEATURE_NAMES, runtime.features(), strict=True))[name]


def test_added_verifier_receives_the_draft_and_history_names_its_source():
    async def scenario():
        runtime, executor, _ = build()
        await runtime.apply(add("n0"))
        assert executor.requests[0].messages == ()
        await runtime.apply(add("n1", "verifier"))
        draft = {
            "source_id": "n0",
            "target_id": "n1",
            "protocol": DRAFT_MESSAGE_PROTOCOL,
            "source_output_version": 1,
            "body": "wrong",
        }
        assert executor.requests[1].messages == (draft,)
        assert executor.requests[1].previous_output is None
        assert runtime.graph.nodes[1].output == "verdict: incorrect"
        recorded = runtime.history[1]["observation"]["node_request"]["messages"]
        assert recorded == [draft]
        assert feature(runtime, "communication_delivered") == 1.0

    run(scenario())


def test_solver_role_and_an_empty_graph_never_receive_a_draft():
    async def scenario():
        runtime, executor, _ = build()
        await runtime.apply(add("n0", "verifier"))
        await runtime.apply(add("n1"))
        await runtime.apply(add("n2"))
        await runtime.apply(GraphAction(K.RERUN_AGENT, node_id="n2"))
        assert [request.messages for request in executor.requests[:1]] == [()]
        # Solvers stay independent samples even after a checker has spoken.
        assert all(request.messages == () for request in executor.requests[1:])

    run(scenario())


def test_draft_is_the_selected_output_else_the_latest_solver_node():
    async def scenario():
        runtime, executor, _ = build()
        await runtime.apply(add("n0"))
        await runtime.apply(add("n1"))
        await runtime.apply(add("n2", "verifier"))
        assert executor.requests[-1].messages[0]["source_id"] == "n1"
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        await runtime.apply(add("n3", "verifier"))
        assert executor.requests[-1].messages[0]["source_id"] == "n0"
        # A checker that is itself the output checks the latest solver draft,
        # not another checker's verdict.
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n3"))
        await runtime.apply(GraphAction(K.RERUN_AGENT, node_id="n3"))
        assert executor.requests[-1].messages[0]["source_id"] == "n1"
        assert executor.requests[-1].previous_output == "verdict: incorrect"

    run(scenario())


def test_a_second_checker_checks_the_solution_not_the_first_verdict():
    async def scenario():
        runtime, executor, _ = build()
        await runtime.apply(add("n0"))
        await runtime.apply(add("n1", "verifier"))
        await runtime.apply(add("n2", "verifier"))
        assert executor.requests[-1].messages[0]["source_id"] == "n0"
        await runtime.apply(GraphAction(K.RERUN_AGENT, node_id="n1"))
        assert executor.requests[-1].messages[0]["source_id"] == "n0"

    run(scenario())


def test_a_checker_without_any_solver_checks_the_latest_other_node():
    async def scenario():
        runtime, executor, _ = build()
        await runtime.apply(add("n0", "verifier"))
        await runtime.apply(add("n1", "verifier"))
        assert executor.requests[-1].messages[0]["source_id"] == "n0"

    run(scenario())


def test_primary_role_is_first_declared_not_alphabetical_and_survives_restore():
    async def scenario():
        runtime, executor, ledger = build(roles=("zeta-solver", "alpha-checker"))
        await runtime.apply(add("n0", "zeta-solver"))
        await runtime.apply(add("n1", "alpha-checker"))
        assert executor.requests[0].messages == ()
        assert executor.requests[1].messages[0]["source_id"] == "n0"
        snapshot = runtime.snapshot()
        assert [role["role_id"] for role in snapshot["config"]["roles"]] == [
            "zeta-solver",
            "alpha-checker",
        ]
        restored = GraphRuntime.from_snapshot(
            snapshot, executor=executor, ledger=ledger, evaluator=lambda output: 0.0
        )
        assert restored.public_state() == runtime.public_state()
        await restored.apply(add("n2", "zeta-solver"))
        assert executor.requests[-1].messages == ()

    run(scenario())


def test_feedback_edge_is_delivered_on_rerun_and_rechecked_against_the_revision():
    async def scenario():
        runtime, executor, _ = build()
        await runtime.apply(add("n0"))
        await runtime.apply(add("n1", "verifier"))
        await runtime.apply(
            GraphAction(K.ADD_EDGE, source_id="n1", target_id="n0", protocol="feedback")
        )
        assert len(executor.requests) == 2  # Feedback is queued, not executed.
        await runtime.apply(GraphAction(K.RERUN_AGENT, node_id="n0"))
        rerun = executor.requests[-1]
        assert rerun.previous_output == "wrong"
        assert rerun.messages == (
            {
                "source_id": "n1",
                "target_id": "n0",
                "protocol": "feedback",
                "source_output_version": 1,
                "body": "verdict: incorrect",
            },
        )
        assert runtime.graph.nodes[0].output == "right"
        # Solve -> Check -> feedback -> rerun -> check the revised draft.
        await runtime.apply(GraphAction(K.RERUN_AGENT, node_id="n1"))
        check = executor.requests[-1]
        assert check.messages[0]["source_id"] == "n0"
        assert check.messages[0]["source_output_version"] == 2
        assert check.messages[0]["body"] == "right"
        assert runtime.graph.nodes[1].output == "verdict: correct"
        # A later bind of the target reads the checker's newest verdict.
        await runtime.apply(GraphAction(K.BIND_SKILL, node_id="n0", skill_id="repair"))
        assert executor.requests[-1].messages[0]["source_output_version"] == 2
        assert executor.requests[-1].messages[0]["body"] == "verdict: correct"
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        receipt = await runtime.apply(GraphAction(K.STOP))
        assert receipt.reward == 1.0

    run(scenario())


def test_an_edge_from_the_draft_source_is_not_duplicated_by_the_draft():
    async def scenario():
        runtime, executor, _ = build()
        await runtime.apply(add("n0"))
        await runtime.apply(add("n1", "verifier"))
        await runtime.apply(
            GraphAction(K.ADD_EDGE, source_id="n0", target_id="n1", protocol="feedback")
        )
        await runtime.apply(GraphAction(K.RERUN_AGENT, node_id="n1"))
        assert [m["protocol"] for m in executor.requests[-1].messages] == ["feedback"]
        await runtime.apply(
            GraphAction(K.ADD_EDGE, source_id="n1", target_id="n0", protocol="revise")
        )
        assert [m["protocol"] for m in executor.requests[-1].messages] == ["revise"]

    run(scenario())


def test_snapshot_replays_draft_messages_and_rejects_a_removed_draft():
    async def scenario():
        runtime, executor, ledger = build()
        await runtime.apply(add("n0"))
        await runtime.apply(add("n1", "verifier"))
        await runtime.apply(GraphAction(K.DROP_AGENT, node_id="n1"))
        await runtime.apply(add("n2", "verifier"))
        snapshot = runtime.snapshot()
        restored = GraphRuntime.from_snapshot(
            snapshot, executor=executor, ledger=ledger, evaluator=lambda output: 0.0
        )
        assert restored.state_id == runtime.state_id
        tampered = runtime.snapshot()
        tampered["history"][3]["observation"]["node_request"]["messages"] = []
        tampered["snapshot_hash"] = stable_hash(
            {key: item for key, item in tampered.items() if key != "snapshot_hash"}
        )
        with pytest.raises(ValueError, match="declared graph effect"):
            GraphRuntime.from_snapshot(
                tampered, executor=executor, ledger=ledger, evaluator=lambda output: 0.0
            )

    run(scenario())


class TextPolicy:
    """frozen_text 3-tuple backend; the reply length decides truncation."""

    configuration_id = "text-policy@1"

    def __init__(self, text, output_tokens):
        self.text, self.output_tokens = text, output_tokens
        self.calls = []

    def frozen_text(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return self.text, 7, self.output_tokens


class FinishPolicy(TextPolicy):
    """A backend that knows its finish reason; frozen_text must not be used."""

    def __init__(self, text, output_tokens, truncated):
        super().__init__(text, output_tokens)
        self.truncated = truncated

    def frozen_text(self, prompt, **kwargs):
        raise AssertionError("the detailed generation path must be preferred")

    def frozen_generation(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return self.text, 7, self.output_tokens, self.truncated


def request(maximum=MAXIMUM):
    return NodeExecutionRequest(
        runtime_id="v7",
        node_id="n0",
        role=RoleSpec("solver", "Solve.", maximum),
        task_prompt="Task.",
        skills=(),
        messages=(),
        previous_output=None,
        execution_index=1,
        seed=3,
    )


@pytest.mark.parametrize(
    ("policy", "outcome"),
    [
        (
            TextPolicy("partial reasoning", MAXIMUM.output_tokens),
            {"status": "failed", "answer_present": False, "failure_kind": "truncated"},
        ),
        (
            TextPolicy("42", 2),
            {"status": "answered", "answer_present": True, "failure_kind": None},
        ),
        (
            TextPolicy("  ", 1),
            {"status": "answered", "answer_present": False, "failure_kind": None},
        ),
        (
            # A finished completion that only plans states no answer.
            TextPolicy("Plan:\n1. gather the facts\n2. solve the task", 4),
            {"status": "answered", "answer_present": False, "failure_kind": None},
        ),
        (
            FinishPolicy("stop in the last slot", MAXIMUM.output_tokens, False),
            {"status": "answered", "answer_present": True, "failure_kind": None},
        ),
        (
            FinishPolicy("cut", 3, True),
            {"status": "failed", "answer_present": False, "failure_kind": "truncated"},
        ),
    ],
)
def test_frozen_text_executor_reports_output_cap_truncation(policy, outcome):
    result = run(FrozenTextExecutor(policy).execute(request()))
    assert result.output == policy.text  # Never replaced or trimmed.
    assert result.metadata == {
        "execution_outcome": outcome,
        "prompt_fit": {"omitted_characters": 0},
    }
    assert result.usage.output_tokens == policy.output_tokens
    assert policy.calls[0]["max_new_tokens"] == MAXIMUM.output_tokens


def test_executor_rejects_malformed_detailed_generation():
    class Broken(FinishPolicy):
        def frozen_generation(self, prompt, **kwargs):
            return self.text, 7, self.output_tokens, "length"

    with pytest.raises(TypeError, match="boolean"):
        run(FrozenTextExecutor(Broken("x", 1, True)).execute(request()))


def test_truncated_node_sets_last_execution_failed_and_is_not_an_answer():
    names = ("last_execution_failed", "answered_fraction")

    async def scenario(policy):
        runtime, _, _ = build(executor=FrozenTextExecutor(policy))
        await runtime.apply(add("n0"))
        executed = tuple(feature(runtime, name) for name in names)
        await runtime.apply(GraphAction(K.SET_OUTPUT, node_id="n0"))
        return executed, tuple(feature(runtime, name) for name in names)

    executed, selected = run(scenario(TextPolicy("long unfinished", MAXIMUM.output_tokens)))
    assert executed == (1.0, 0.0)
    # The event flag is per action; the missing answer persists after selection.
    assert selected == (0.0, 0.0)
    executed, selected = run(scenario(TextPolicy("42", 2)))
    assert executed == (0.0, 1.0)
    assert selected == (0.0, 1.0)


class ScriptedSGLang(SGLangFrozenText):
    class Tokenizer:
        eos_token_id = 0
        chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [1] * len(text)

    def __init__(self, meta, text="out"):
        super().__init__(
            "http://sglang.invalid",
            self.Tokenizer(),
            reference_id="scripted",
            context_window=1024,
        )
        self.reply = {"text": text, "meta_info": meta}
        self.bodies = []

    def _post(self, body, attempts=4):
        self.bodies.append(body)
        return self.reply


@pytest.mark.parametrize(
    ("meta", "truncated"),
    [
        ({"completion_tokens": 8, "finish_reason": {"type": "length", "length": 8}}, True),
        # A stop token sampled in the final slot is a natural finish.
        ({"completion_tokens": 8, "finish_reason": {"type": "stop", "matched": 0}}, False),
        ({"completion_tokens": 3, "finish_reason": {"type": "stop", "matched": 0}}, False),
        ({"completion_tokens": 8}, True),
        ({"completion_tokens": 3, "finish_reason": None}, False),
    ],
)
def test_sglang_reports_length_finish_and_keeps_the_three_tuple(meta, truncated):
    backend = ScriptedSGLang(meta)
    kwargs = {"max_new_tokens": 8, "temperature": 0.3, "seed": 9}
    detailed = backend.frozen_generation("prompt", **kwargs)
    assert detailed == ("out", len("prompt"), meta["completion_tokens"], truncated)
    assert backend.frozen_text("prompt", **kwargs) == detailed[:3]
    assert backend.bodies[0] == backend.bodies[1]
    assert backend.bodies[0]["sampling_params"]["max_new_tokens"] == 8


def test_sglang_truncation_reaches_the_node_outcome():
    backend = ScriptedSGLang({"completion_tokens": 8, "finish_reason": {"type": "length"}})
    # One scripted token per character: the rendered node prompt needs room.
    maximum = BudgetVector(input_tokens=1000, output_tokens=8, model_calls=1, agent_turns=1)
    result = run(FrozenTextExecutor(backend).execute(request(maximum)))
    assert result.metadata["execution_outcome"]["failure_kind"] == "truncated"


@pytest.mark.parametrize(("eos_bias", "failure_kind"), [(-1e4, "truncated"), (1e4, None)])
def test_in_process_policy_truncation_is_inferred_from_its_token_count(eos_bias, failure_kind):
    torch = pytest.importorskip("torch")
    import copy

    from skillev.policy.evosteer import CausalLMOrchestrator
    from tests.evosteer.test_application import BytesTokenizer, TinyCausalModel

    torch.manual_seed(3)
    model = TinyCausalModel()
    reference = copy.deepcopy(model)
    with torch.no_grad():
        # Frozen generation runs the reference copy; pin EOS to never/always.
        reference.actor_bias[BytesTokenizer.eos_token_id] = eos_bias
    policy = CausalLMOrchestrator(
        model,
        BytesTokenizer(),
        reference_id="tiny-base@1",
        encoding_dim=8,
        reference_model=reference,
        context_window=65_536,
        max_action_tokens=256,
    )
    maximum = BudgetVector(input_tokens=4096, output_tokens=4, model_calls=1, agent_turns=1)
    result = run(FrozenTextExecutor(policy).execute(replace(request(maximum), seed=1)))
    # The in-process loop stops early only at EOS, so a reply that used the
    # whole allowance is the cut one.
    assert result.usage.output_tokens == (4 if failure_kind else 1)
    assert result.metadata["execution_outcome"]["failure_kind"] == failure_kind
    assert result.metadata["execution_outcome"]["answer_present"] is False


def test_normalized_answer_ignores_presentation_only():
    assert normalized_answer("```python\ndef f():\n    return 1\n```") == normalized_answer(
        "def f():\n  return 1"
    )
    assert normalized_answer("  Paris\n") == normalized_answer("paris")
    assert normalized_answer("`B`") == "b"
    assert normalized_answer("so the answer is \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert normalized_answer("\\boxed{3} then \\boxed{ 42 }") == "42"
    assert normalized_answer("cut off at \\boxed{4") == "cut off at \\boxed{4"
    assert normalized_answer("Paris") != normalized_answer("London")
    # The last complete box counts, even after a later unclosed one.
    assert normalized_answer("\\boxed{42} ... recheck \\boxed{4") == "42"
    # Short-answer cue and code-block conventions, as the graders read them.
    assert normalized_answer("Reasoning...\nThe answer is: C.") == normalized_answer("C")
    assert normalized_answer("Final answer: **Paris**") == "paris"
    assert normalized_answer("Here:\n```python\ndef g(): pass\n```\nDone.") == "def g(): pass"
    # Backticks are stripped only when they wrap the whole answer.
    assert normalized_answer("`a` or `b`") == "`a` or `b`"


def test_answer_agreement_compares_normalized_answers():
    def agreement(*outputs):
        graph = {
            "nodes": [
                {"node_id": f"n{i}", "role_id": "solver", "skill_ids": [], "output": output}
                for i, output in enumerate(outputs)
            ],
            "edges": [],
            "output_node_id": None,
        }
        values = extract_features(
            graph=graph,
            history=[],
            max_nodes=None,
            max_actions=None,
            role_count=1,
            skill_count=0,
            used=BudgetVector(),
            cap=MAXIMUM,
        )
        assert len(values) == 30
        return dict(zip(FEATURE_NAMES, values, strict=True))["answer_agreement"]

    assert FEATURE_VERSION == "evosteer-public-features@4"
    assert agreement("```\nx = 1\n```", "x = 1") == 1.0
    assert agreement("Work... \\boxed{42}", "\\boxed{ 42 }", "\\boxed{41}") == pytest.approx(1 / 3)
    assert agreement("Paris", "London") == 0.0
    # Outputs that are pure formatting present no comparable answer.
    assert agreement("```\n```", "```\n```") == 0.0


PLAN = (
    "### Step-by-Step Plan\n\n"
    "1. **Analyze the Condition $f(n) = n$**: find the fixed points.\n"
    "2. **Enumerate the candidates**: test each three-digit $n$.\n"
    "3. **Sum the survivors**: add what remains.\n"
)
FACTS = (
    "Based on the task and the provided test case, here are the necessary facts "
    "and their sources:\n\n"
    "**Facts Needed:**\n\n"
    "1. **Function Name**: `find_adverb_position`\n"
    "2. **Input**: one sentence, as text.\n"
    "3. **Output**: a tuple of the position and the adverb.\n"
)


def test_answer_present_reads_the_output_not_the_node_role():
    # Outputs that only say how an answer would be reached state none.
    assert answer_present(PLAN) is False
    assert answer_present(FACTS) is False
    # Nothing consults the role, so the same plan with an answer added counts,
    # and any node that states an answer counts whatever its role instruction.
    assert answer_present(PLAN + "\nTherefore the sum is \\boxed{279}.") is True
    # The graders' three forms, and ordinary prose, are answers.
    assert answer_present("Lawrence County") is True
    assert answer_present("Mahatma Gandhi") is True
    assert answer_present("```python\ndef find_adverb_position(text):\n    return 0\n```") is True
    assert answer_present("Work it out.\nThe answer is: 279") is True
    assert answer_present("Two of them are fixed, so the total is 279.") is True
    assert answer_present("   ") is False
    # A list that is itself the answer is not a plan: it has no title, and one
    # item is a bare answer written as a bullet.
    assert answer_present("1. Tokyo\n2. Delhi\n3. Shanghai") is True
    assert answer_present("- Lawrence County") is True


def test_answered_fraction_counts_only_nodes_that_state_an_answer():
    def answered(*outputs):
        graph = {
            "nodes": [
                {"node_id": f"n{i}", "role_id": "solver", "skill_ids": [], "output": output}
                for i, output in enumerate(outputs)
            ],
            "edges": [],
            "output_node_id": None,
        }
        values = extract_features(
            graph=graph,
            history=[],
            max_nodes=None,
            max_actions=None,
            role_count=1,
            skill_count=0,
            used=BudgetVector(),
            cap=MAXIMUM,
        )
        return dict(zip(FEATURE_NAMES, values, strict=True))["answered_fraction"]

    # Schema 4: a plan or a fact list no longer reads as an answered node.
    assert FEATURE_VERSION == "evosteer-public-features@4"
    assert answered(PLAN, FACTS) == 0.0
    assert answered(PLAN, "Lawrence County") == 0.5
    assert answered("\\boxed{279}", "```python\ndef f():\n    return 1\n```") == 1.0


class CharPolicy(TextPolicy):
    """One token per prompt character, so the envelope is checked in characters."""

    context_window = 100_000

    def __init__(self, text="ok", output_tokens=2):
        super().__init__(text, output_tokens)
        self.prompts = []

    def frozen_prompt_tokens(self, prompt):
        return len(prompt)

    def frozen_text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return super().frozen_text(prompt, **kwargs)


def checking_request(input_tokens, *, draft, previous):
    maximum = BudgetVector(
        input_tokens=input_tokens,
        output_tokens=MAXIMUM.output_tokens,
        model_calls=1,
        agent_turns=1,
        wall_time_milliseconds=MAXIMUM.wall_time_milliseconds,
    )
    return NodeExecutionRequest(
        runtime_id="v7",
        node_id="n1",
        role=RoleSpec("verifier", "Check the draft.", maximum),
        task_prompt="Task: add 2 and 2.",
        skills=(("s1", 'Step 1: recompute.\nStep 2: say "VERDICT".'),),
        messages=(
            {
                "source_id": "n0",
                "target_id": "n1",
                "protocol": "draft",
                "source_output_version": 1,
                "body": draft,
            },
        ),
        previous_output=previous,
        execution_index=2,
        seed=5,
    )


def test_node_prompt_is_plain_text_with_verbatim_procedures():
    policy = CharPolicy()
    result = run(
        FrozenTextExecutor(policy).execute(
            checking_request(100_000, draft="The answer is 4.", previous="Earlier check.")
        )
    )
    prompt = policy.prompts[0]
    assert prompt.startswith("Role: Check the draft.\n\nTask:\nTask: add 2 and 2.")
    # The procedure is shown as written, not as an escaped JSON string.
    assert 'Procedure to follow (skill s1):\nStep 1: recompute.\nStep 2: say "VERDICT".' in prompt
    assert "Input from agent n0 (draft):\nThe answer is 4." in prompt
    assert "Your previous output:\nEarlier check." in prompt
    assert prompt.endswith(
        "Perform your role using the task, the procedure, the inputs from other agents "
        "and your previous output. Return your result."
    )
    assert result.metadata["prompt_fit"] == {"omitted_characters": 0}


def test_oversized_inputs_are_shortened_to_the_envelope_keeping_their_endings():
    draft = "D" * 6000 + " so the answer is \\boxed{4}"
    previous = "P" * 3000 + " previous verdict: correct"
    policy = CharPolicy()
    limit = 4000
    result = run(
        FrozenTextExecutor(policy).execute(
            checking_request(limit, draft=draft, previous=previous)
        )
    )
    prompt = policy.prompts[0]
    assert len(prompt) <= limit
    assert policy.calls[0]["input_limit"] == limit
    # Task and procedure are never shortened; each input keeps its ending.
    assert "Task:\nTask: add 2 and 2." in prompt
    assert 'Step 2: say "VERDICT".' in prompt
    assert prompt.count("characters omitted to fit the input limit") >= 1
    assert "so the answer is \\boxed{4}" in prompt
    omitted = result.metadata["prompt_fit"]["omitted_characters"]
    assert omitted > 0
    # Deterministic: the same request renders the same fitted prompt.
    again = CharPolicy()
    run(FrozenTextExecutor(again).execute(checking_request(limit, draft=draft, previous=previous)))
    assert again.prompts == policy.prompts


def test_inputs_are_not_shortened_below_the_floor():
    policy = CharPolicy()
    # Far too small for the fixed task and procedure: the prompt is passed on
    # (and the backend's own envelope check reports it) rather than emptied.
    run(FrozenTextExecutor(policy).execute(checking_request(50, draft="x" * 900, previous=None)))
    assert "characters omitted" in policy.prompts[0]
    assert len(policy.prompts[0]) > 800


def test_structured_skill_is_rendered_as_readable_sections_without_losing_fields():
    import json

    from skillev.orchestration.model_executor import _procedure_text

    skill = {
        "name": "Concise multi-hop",
        "description": "Resolve the entity chain.",
        "trigger": "Use for multi-passage short answers.",
        "plan": ["Identify the subject.", "Output only the answer phrase."],
        "pitfall": ["Do not select verbose text."],
        "constraint": "Use only the supplied passages.",
        "extra": {"k": 1},
    }
    text = _procedure_text(json.dumps(skill, sort_keys=True))
    assert text.splitlines() == [
        "Name: Concise multi-hop",
        "Purpose: Resolve the entity chain.",
        "Use when: Use for multi-passage short answers.",
        "Steps:",
        "1. Identify the subject.",
        "2. Output only the answer phrase.",
        "Pitfalls to avoid:",
        "- Do not select verbose text.",
        "Constraint: Use only the supplied passages.",
        'extra: {"k": 1}',
    ]
    # Plain-text procedures and JSON that is not an object are shown verbatim.
    assert _procedure_text("Step 1: recompute.") == "Step 1: recompute."
    assert _procedure_text("[1, 2]") == "[1, 2]"


def test_skill_menu_shows_name_and_trigger_but_never_the_procedure():
    import json as _json

    from skillev.orchestration.execution import _skill_label

    body = _json.dumps(
        {
            "name": "Concise multi-hop passage answering",
            "trigger": "Use when a question must be answered from provided passages.",
            "plan": ["Never shown in the menu."],
            "constraint": "Never shown in the menu.",
        },
        sort_keys=True,
    )
    label = _skill_label(body)
    assert label == {
        "name": "Concise multi-hop passage answering",
        "trigger": "Use when a question must be answered from provided passages.",
    }
    assert "Never shown" not in _json.dumps(label)
    # Any body that is not an authored skill object stays identifier-only.
    assert _skill_label("Step 1: recompute.\nStep 2: check.") == {}
    assert _skill_label("   ") == {}
    assert _skill_label(_json.dumps({"plan": ["x"]})) == {}
