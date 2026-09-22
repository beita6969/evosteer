from __future__ import annotations

import asyncio
import io
import json
import sys
import types
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.benchmarks import protocol_v13_mbpp_worker
from skillev_private.benchmarks.code_math import CodeExecutionResult, CodeExecutionStatus
from skillev_private.benchmarks.mbpp_scoring import (
    MBPP_REQUEST_FORMAT,
    MBPP_VERDICT_FORMAT,
    MBPPScorerProfile,
)
from skillev_private.benchmarks.protocol_v10_official import HealthBenchGrade
from skillev_private.benchmarks.protocol_v10_workers import (
    InMemoryWorkerTransport,
    PrivateJSONWorker,
)
from skillev_private.benchmarks.protocol_v13_training import (
    Protocol13TrainingEpisode,
    Protocol13TrainingOutput,
    Protocol13TrainingRecord,
)
from skillev_private.benchmarks.protocol_v13_training_sessions import (
    _action_contract,
    _completion_environment,
    _HealthEvaluator,
    _HumanEvalEvaluator,
    _MBPPPlusEvaluator,
    _StaticEvaluator,
)
from skillev_private.evaluation.result_contracts import public_native_metric_values
from skillev_private.experiments.protocol_v13_training_debug import (
    _application_config,
    _phi_per_cycle,
    _run,
    _validated_worker_interpreter,
)

from skillev.contracts import stable_hash
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.owner_final import project_owner_final
from skillev.evaluation.step0_completion import StepZeroTerminalMode
from skillev.rollout import (
    RolloutTask,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ExactAttemptRunPlan


def test_invalid_run_identifier_fails_before_runtime_initialization(tmp_path: Path) -> None:
    root = tmp_path / "mini-20260905T071345Z"
    # No performance or model arguments are supplied: they must not be accessed
    # before this operational identifier is rejected.
    with pytest.raises(ValueError):
        _run(Namespace(steps=8, run_root=root))
    assert not root.exists()


def _record(
    benchmark: Protocol13Benchmark = Protocol13Benchmark.HOTPOT_QA,
) -> Protocol13TrainingRecord:
    task_id = f"protocol13/{benchmark.value}/training/0000"
    surface, profile = _action_contract(benchmark, None)
    task = RolloutTask(
        task_id=task_id,
        environment_id=f"benchmark:{benchmark.value}@fixture:completion",
        task_family=f"{benchmark.value}/completion",
        context_id=f"protocol13:{benchmark.value}:iid",
        query="Which public answer follows from the supplied context?",
        available_tools=(),
        public_context={
            "benchmark_id": benchmark.value,
            "dataset_revision": "fixture",
            "payload": {"context": "Public context only."},
            "split": "training",
        },
        action_surface=surface,
        budget_profile=profile,
    )
    return Protocol13TrainingRecord(
        episode=Protocol13TrainingEpisode(
            benchmark=benchmark,
            population_id="fixture-population",
            episode_id=task_id,
            source_id="fixture-source",
            repeat_ordinal=0,
            block_position=0,
            optimizer_step=1,
            global_position=0,
        ),
        input=task,
        output=Protocol13TrainingOutput(
            evaluator_kind={
                Protocol13Benchmark.HOTPOT_QA: "hotpotqa-official-em-f1",
                Protocol13Benchmark.TRIVIA_QA: "triviaqa-official-alias-em-f1",
                Protocol13Benchmark.AIME_2026: "integer-exact",
            }[benchmark],
            target={
                "accepted_answers": [
                    "42" if benchmark is Protocol13Benchmark.AIME_2026 else "private answer"
                ]
            },
        ),
    )


def _request(task_id: str, answer: str) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory",
        task_id=task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue({"answer": answer}),
        public_transcript_hash=stable_hash({"transcript": "public"}),
    )


def _mbpp_record() -> Protocol13TrainingRecord:
    task_id = "protocol13/mbpp-plus/training/0000"
    surface, profile = _action_contract(Protocol13Benchmark.MBPP_PLUS, None)
    task = RolloutTask(
        task_id=task_id,
        environment_id="benchmark:mbpp-plus@fixture:completion",
        task_family="mbpp-plus/code-generation",
        context_id="protocol13:mbpp-plus:iid",
        query="def solve():\n    pass\n",
        available_tools=(),
        public_context={
            "benchmark_id": "mbpp-plus",
            "dataset_revision": "fixture",
            "payload": {"language": "python"},
            "split": "training",
        },
        action_surface=surface,
        budget_profile=profile,
    )
    return Protocol13TrainingRecord(
        episode=Protocol13TrainingEpisode(
            benchmark=Protocol13Benchmark.MBPP_PLUS,
            population_id="fixture-population",
            episode_id=task_id,
            source_id="fixture-source",
            repeat_ordinal=0,
            block_position=0,
            optimizer_step=1,
            global_position=0,
        ),
        input=task,
        output=Protocol13TrainingOutput(
            evaluator_kind="evalplus-base-plus",
            target={"private": "fixture"},
        ),
    )


def test_protocol13_static_reward_keeps_private_answer_in_evaluator() -> None:
    record = _record()
    reward = asyncio.run(
        _StaticEvaluator(record, record.input).evaluate(
            _request(record.input.task_id, "private answer")
        )
    )

    assert "private answer" not in record.input.query
    assert reward.value == 1.0
    assert reward.success is True


def test_protocol13_static_reward_rejects_cross_task_request() -> None:
    record = _record()

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(
            _StaticEvaluator(record, record.input).evaluate(
                _request("protocol13/hotpotqa/training/another", "private answer")
            )
        )


def test_protocol13_qa_exports_both_metrics_without_changing_the_ttb_projection() -> None:
    record = _record()
    reward = asyncio.run(
        _StaticEvaluator(record, record.input).evaluate(_request(record.input.task_id, "answer"))
    )
    metrics = {item.metric_name: item.value for item in public_native_metric_values(reward)}
    assert metrics["answer-f1"] == reward.value == pytest.approx(2 / 3)
    assert metrics["answer-exact-match"] == 0
    assert not reward.success


@pytest.mark.parametrize(
    "benchmark", [Protocol13Benchmark.HOTPOT_QA, Protocol13Benchmark.TRIVIA_QA]
)
def test_training_qa_scores_the_owners_explicit_final_not_the_discussion(benchmark) -> None:
    record = _record(benchmark)
    answer = "Some discussion of the supplied context.\nFinal answer: private answer"
    reward = asyncio.run(
        _StaticEvaluator(record, record.input).evaluate(_request(record.input.task_id, answer))
    )
    assert reward.success
    assert reward.value == 1


@pytest.mark.parametrize(
    ("answer", "success"),
    [
        ("42", True),
        ("042", True),
        (r"\boxed{42}", True),
        ("Final answer: 42", True),
        ("Discussion contains 17.\nFinal answer: 42", True),
        ("Final answer: 42\nFinal answer: 43", False),
        (r"\boxed{42} or \boxed{43}", False),
        ("1000", False),
    ],
)
def test_training_aime_uses_the_same_answer_blind_projection_as_clean_evaluation(
    answer, success
) -> None:
    record = _record(Protocol13Benchmark.AIME_2026)
    projected = project_owner_final(
        StepZeroTerminalMode.AIME_INTEGER, answer, owner_id="owner", message_id="message"
    )
    reward = asyncio.run(
        _StaticEvaluator(record, record.input).evaluate(_request(record.input.task_id, answer))
    )
    assert reward.success is success
    assert reward.value == float(success)
    assert success is (projected is not None and projected.payload == r"\boxed{42}")
    # A different private answer cannot affect which payload is submitted.
    different = replace(record, output=replace(record.output, target={"accepted_answers": ["43"]}))
    other_reward = asyncio.run(
        _StaticEvaluator(different, different.input).evaluate(
            _request(different.input.task_id, answer)
        )
    )
    assert not other_reward.success


@pytest.mark.parametrize("wrapper", ["{}", "```python\n{}\n```", "Final code:\n```python\n{}\n```"])
def test_training_humaneval_keeps_native_completion_indentation(wrapper) -> None:
    original = _mbpp_record()
    task = replace(
        original.input,
        query="def solve():\n",
        task_family="humaneval/code-generation",
        public_context={**original.input.public_context, "benchmark_id": "humaneval"},
    )
    record = replace(
        original,
        episode=replace(original.episode, benchmark=Protocol13Benchmark.HUMAN_EVAL),
        input=task,
        output=Protocol13TrainingOutput(
            evaluator_kind="humaneval-native",
            target={
                "entry_point": "solve",
                "test": "def check(candidate): assert candidate() == 1",
            },
        ),
    )
    calls = []

    class Executor:
        async def run(self, request):
            calls.append(request)
            compile(request.prompt + request.completion, "<synthetic-candidate>", "exec")
            return CodeExecutionResult(CodeExecutionStatus.PASSED)

    source = "    return 1"
    reward = asyncio.run(
        _HumanEvalEvaluator(record, task, Executor()).evaluate(
            _request(task.task_id, wrapper.format(source))
        )
    )
    assert reward.success
    assert len(calls) == 1
    assert calls[0].completion == source
    assert calls[0].prompt == task.query


def test_training_mbpp_does_not_submit_a_program_from_nested_unclosed_fences() -> None:
    answer = "```python\ndef solve(): return 1\n```python\ndef solve(): return 2\n```"
    record = _mbpp_record()
    transport = InMemoryWorkerTransport(lambda _request: pytest.fail("ambiguous program executed"))
    worker = PrivateJSONWorker(
        command=("fixture-worker",),
        working_directory=Path.cwd(),
        timeout_seconds=1.0,
        transport=transport,
    )
    reward = asyncio.run(
        _MBPPPlusEvaluator(record, record.input, worker, MBPPScorerProfile()).evaluate(
            _request(record.input.task_id, answer)
        )
    )
    assert not reward.success
    assert not transport.requests


@pytest.mark.parametrize("passed", [False, True])
def test_training_mbpp_uses_the_same_final_block_independent_of_the_verdict(passed) -> None:
    record = _mbpp_record()
    transport = InMemoryWorkerTransport(
        lambda _request: json.dumps(_mbpp_verdict(passed, passed)).encode()
    )
    worker = PrivateJSONWorker(
        command=("fixture-worker",),
        working_directory=Path.cwd(),
        timeout_seconds=1.0,
        transport=transport,
    )
    response = "```python\ndef solve(): return 1\n```\n```python\ndef solve(): return 2\n```"
    reward = asyncio.run(
        _MBPPPlusEvaluator(record, record.input, worker, MBPPScorerProfile()).evaluate(
            _request(record.input.task_id, response)
        )
    )
    assert reward.success is passed
    assert len(transport.requests) == 1
    assert json.loads(transport.requests[0])["submission"] == "def solve(): return 2"


@pytest.mark.parametrize(
    ("raw", "negative", "success"), [(-0.4, 1, False), (0.8, 1, False), (0.8, 0, True)]
)
def test_protocol13_health_keeps_raw_metric_separate_from_reward_and_success(
    raw, negative, success
) -> None:
    task = replace(_record().input, public_context={"benchmark_id": "healthbench"})
    calls = []

    class Grader:
        verifier_version = "synthetic-health"

        async def grade(self, task_id, answer):
            calls.append((task_id, answer))
            return HealthBenchGrade(raw, negative)

    reward = asyncio.run(_HealthEvaluator(task, Grader()).evaluate(_request(task.task_id, "reply")))
    metrics = {item.metric_name: item.value for item in public_native_metric_values(reward)}
    assert metrics["qwen-local-rubric-score"] == raw
    assert reward.value == max(0, min(1, raw))
    assert reward.success is success
    assert calls == [(task.task_id, "reply")]


@pytest.mark.parametrize(
    ("base", "plus"), [(True, True), (True, False), (False, True), (False, False)]
)
def test_protocol13_mbpp_preserves_both_verdicts_and_scores_their_conjunction(base, plus) -> None:
    transport = InMemoryWorkerTransport(
        lambda _request: json.dumps(_mbpp_verdict(base, plus)).encode()
    )
    worker = PrivateJSONWorker(
        command=("fixture-worker",),
        working_directory=Path.cwd(),
        timeout_seconds=1.0,
        transport=transport,
    )
    record = _mbpp_record()
    reward = asyncio.run(
        _MBPPPlusEvaluator(record, record.input, worker, MBPPScorerProfile()).evaluate(
            _request(record.input.task_id, "def solve():\n    return 1\n")
        )
    )
    metrics = {item.metric_name: item.value for item in public_native_metric_values(reward)}
    assert reward.value == float(base and plus)
    assert reward.success is (base and plus)
    assert metrics["base-pass"] == float(base)
    assert metrics["plus-pass"] == float(plus)
    assert reward.native_payload["plus-passed"] is plus
    assert len(transport.requests) == 1  # A valid failure does not get retried.


def test_protocol13_mbpp_malformed_verdict_is_infrastructure_not_reward_zero() -> None:
    transport = InMemoryWorkerTransport(
        lambda _request: b'{"base_passed": "true", "plus_passed": true, "status": "done"}'
    )
    worker = PrivateJSONWorker(
        command=("fixture-worker",),
        working_directory=Path.cwd(),
        timeout_seconds=1.0,
        transport=transport,
    )
    record = _mbpp_record()
    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(
            _MBPPPlusEvaluator(record, record.input, worker, MBPPScorerProfile()).evaluate(
                _request(record.input.task_id, "def solve():\n    return 1\n")
            )
        )
    assert len(transport.requests) == 1


def test_protocol13_completion_session_is_fresh_and_debug_plan_is_exact() -> None:
    record = _record()
    first = _completion_environment(record.input)
    second = _completion_environment(record.input)
    application, plan = _application_config(run_id="protocol13-debug-test")

    assert first is not second
    assert first.delegate is not second.delegate
    assert plan.total_training_steps == 8
    assert application.trainer.execution.batch_size == 28
    assert application.trainer.optimizer.adapter_learning_rate == 1e-4
    assert application.trainer.optimizer.z_learning_rate == 1e-4
    assert application.trainer.checkpoint.every_n_steps == 10
    assert application.diagnostics.window_size == 50

    formal_application, formal_plan = _application_config(
        run_id="protocol13-formal-shape-test",
        steps=250,
        run_plan=ExactAttemptRunPlan(
            phase_search_steps=249,
            closure_steps=1,
            maximum_cycles=2,
        ),
    )
    assert formal_plan.total_training_steps == 250
    assert formal_application.trainer.execution.batch_size == 28
    assert formal_application.trainer.checkpoint.every_n_steps == 10
    assert formal_plan.maximum_cycles == 2
    with pytest.raises(ValueError):
        _application_config(run_id="explicit-plan-required", steps=250)


def test_protocol13_preserves_virtualenv_interpreter_symlink(tmp_path: Path) -> None:
    executable = tmp_path / "python-real"
    executable.write_text("", encoding="utf-8")
    virtualenv_entrypoint = tmp_path / "python"
    virtualenv_entrypoint.symlink_to(executable)

    assert _validated_worker_interpreter(virtualenv_entrypoint) == virtualenv_entrypoint


def test_declared_phi_cycle_budget_is_not_hardcoded_or_a_training_cycle_limit() -> None:
    config, _ = _application_config(run_id="explicit-phi-envelope")
    budget = _phi_per_cycle(config, 3)
    assert budget.model_calls == 3
    assert budget.input_tokens == 3 * config.evolution.max_authoring_prompt_tokens
    assert budget.output_tokens == 3 * config.evolution.max_authoring_completion_tokens
    with pytest.raises(ValueError):
        _phi_per_cycle(config, 0)


def _mbpp_verdict(base: bool, plus: bool) -> dict[str, object]:
    return {
        "format": MBPP_VERDICT_FORMAT,
        "scorer_profile": MBPPScorerProfile().to_value(),
        "base_passed": base,
        "plus_passed": plus,
        "status": "passed" if base and plus else "failed",
        "syntax": {"status": "valid"},
        "lanes": {
            name: {"native_status": "pass" if passed else "fail"}
            for name, passed in (("base", base), ("plus", plus))
        },
    }


def test_protocol13_mbpp_retries_only_infrastructure_envelope() -> None:
    responses = iter(
        (
            {
                "error_module": None,
                "error_stage": "base-candidate",
                "error_type": "OSError",
                "infrastructure_error": True,
            },
            _mbpp_verdict(True, True),
        )
    )
    transport = InMemoryWorkerTransport(
        lambda _request: json.dumps(next(responses)).encode("utf-8")
    )
    worker = PrivateJSONWorker(
        command=("fixture-worker",),
        working_directory=Path.cwd(),
        timeout_seconds=1.0,
        transport=transport,
    )
    record = _mbpp_record()

    reward = asyncio.run(
        _MBPPPlusEvaluator(record, record.input, worker, MBPPScorerProfile()).evaluate(
            _request(record.input.task_id, "def solve():\n    return 1\n")
        )
    )

    assert reward.success is True
    assert reward.native_payload["scorer-attempts"] == 2
    assert len(transport.requests) == 2
    payload = json.loads(transport.requests[0])
    assert payload["source_task_id"] == record.episode.source_id
    assert payload["task_id"] == record.input.task_id


def test_protocol13_mbpp_worker_restores_official_input_types(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    decoded = [["official-runtime-value"]]

    def deserialize(task_id: str, inputs: object) -> object:
        calls.append((task_id, inputs))
        return decoded

    def trusted_exec(*_args: object, **_kwargs: object) -> tuple[list[int], list[float]]:
        assert _args[1] is decoded
        return [1], [0.01]

    def untrusted_check(*_args: object, **_kwargs: object) -> tuple[str, list[bool]]:
        assert _args[2] is decoded
        return "pass", [True]

    modules = {
        "evalplus": types.ModuleType("evalplus"),
        "evalplus.data": types.ModuleType("evalplus.data"),
        "evalplus.data.mbpp": types.ModuleType("evalplus.data.mbpp"),
        "evalplus.eval": types.ModuleType("evalplus.eval"),
        "evalplus.eval._special_oracle": types.ModuleType("evalplus.eval._special_oracle"),
        "evalplus.gen": types.ModuleType("evalplus.gen"),
        "evalplus.gen.util": types.ModuleType("evalplus.gen.util"),
    }
    modules["evalplus.data.mbpp"].mbpp_deserialize_inputs = deserialize
    modules["evalplus.eval"].PASS = "pass"
    modules["evalplus.eval"].untrusted_check = untrusted_check
    modules["evalplus.eval._special_oracle"].MBPP_OUTPUT_NOT_NONE_TASKS = set()
    modules["evalplus.gen.util"].trusted_exec = trusted_exec
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    request = {
        "format": MBPP_REQUEST_FORMAT,
        "scorer_profile": MBPPScorerProfile().to_value(),
        "operation": "evaluate-mbpp-plus",
        "private_target": {
            "assertion": [],
            "atol": 0,
            "base_input": [["serialized-base"]],
            "canonical_solution": "def solve(value): return 1\n",
            "contract": [],
            "entry_point": "solve",
            "plus_input": [["serialized-plus"]],
        },
        "prompt": "",
        "source_task_id": "private-mbpp-source",
        "submission": "def solve(value): return 1\n",
        "task_id": "protocol13/mbpp-plus/training/0000/rollout-00",
    }
    stdin = types.SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode()))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)

    protocol_v13_mbpp_worker.main()

    assert calls == [
        ("private-mbpp-source", [["serialized-base"]]),
        ("private-mbpp-source", [["serialized-plus"]]),
    ]
    assert json.loads(stdout.getvalue())["status"] == "passed"
