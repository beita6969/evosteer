"""A transport failure retains cost but never fabricates a rubric verdict."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from skillev_private.evaluation import integrity_native_scoring, integrity_runtime
from skillev_private.evaluation.integrity_grader_usage import IncompleteNativeGradingError
from skillev_private.evaluation.integrity_replicas import ReplicaLoadBalancer

from skillev.evaluation import healthbench_official
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.integrity_pipeline import scorer_cost_summary
from skillev.evaluation.integrity_results import NativeScore
from skillev.evaluation.sealed_candidates import CandidateJournal, EventOrigin


def judge_profile():
    return healthbench_official.HealthBenchQwenJudgeProfile(
        profile_id="synthetic-local-rubric",
        backend="qwen-sglang",
        model_route="fixture-base",
        rubric_source_repository="https://github.com/openai/simple-evals",
        rubric_source_revision="synthetic",
        rubric_source_path="healthbench_eval.py",
        call_mode="per-rubric",
        system_message="You are a helpful assistant.",
        response_format="plain-json-then-schema-once",
        max_tokens=2048,
        temperature=0.5,
        top_p=None,
        top_k=None,
        seed=None,
        enable_thinking=False,
        request_timeout_seconds=120,
        maximum_attempts=1,
        total_deadline_seconds=300,
        semantic_repair_attempts=1,
        official_gpt_comparable=False,
    )


def test_frozen_health_profile_does_not_reopen_a_mutable_yaml():
    profile = judge_profile()
    settings = {"profile": "nonexistent.yaml", "effective_profile": asdict(profile)}
    assert integrity_native_scoring.resolve_healthbench_profile(settings) == profile


@pytest.mark.parametrize("fail", [False, True])
def test_native_grader_preserves_success_and_failed_attempt_usage(monkeypatch, fail):
    requests = []
    closed = []

    def create(**kwargs):
        requests.append(kwargs)
        if fail and len(requests) == 2:
            raise OSError("synthetic network failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"criteria_met":true}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        healthbench_official,
        "load_qwen_judge_profile",
        lambda _: judge_profile(),
    )

    class OfficialFixture:
        def grade_sample(self, **kwargs):
            assert kwargs["prompt"] == target["prompt"]
            assert kwargs["response_text"] == "fixed answer"
            self.grader_model([{"role": "user", "content": "synthetic criterion one"}])
            self.grader_model([{"role": "user", "content": "synthetic criterion two"}])
            return (
                {},
                "",
                [
                    {"points": 2, "criteria_met": True},
                    {"points": -1, "criteria_met": False},
                ],
            )

    monkeypatch.setattr(
        "skillev_private.benchmarks.protocol_v10_healthbench_worker._load_official",
        lambda _: (OfficialFixture, SimpleNamespace(from_dict=lambda row: row), SimpleNamespace),
    )
    target = {
        "prompt": [{"role": "user", "content": "fictional question"}],
        "rubrics": [{"points": 2}, {"points": -1}],
    }
    settings = {
        "effective_profile": asdict(judge_profile()),
        "official_source": "unused",
        "endpoint_base": "http://fixture",
        "model_route": "fixture-base",
    }
    diagnostics = {}
    if fail:
        with pytest.raises(IncompleteNativeGradingError) as caught:
            integrity_native_scoring._grade_health(
                "fixed answer", target, settings, diagnostics=diagnostics
            )
        cost = caught.value.scorer_cost
        assert isinstance(
            caught.value.__cause__, healthbench_official.HealthBenchTransportExhausted
        )
        assert cost["unknown_usage_calls"] == 1
        assert cost["wall_seconds"] >= 0
    else:
        value, negative, cost = integrity_native_scoring._grade_health(
            "fixed answer", target, settings, diagnostics=diagnostics
        )
        assert value == 1
        assert negative == 0
        assert cost["unknown_usage_calls"] == 0
    assert cost["model_request_attempts"] == 2
    assert cost["model_responses"] == (1 if fail else 2)
    assert cost["known_input_tokens"] == (10 if fail else 20)
    assert cost["known_output_tokens"] == (4 if fail else 8)
    assert cost["semantic_repairs"] == 0
    assert diagnostics["effective_profile"] == asdict(judge_profile())
    assert len(diagnostics["requests"]) == 2
    for evidence, request in zip(diagnostics["requests"], requests, strict=True):
        assert evidence["request"] == request
        assert request["temperature"] == judge_profile().temperature
        assert request["max_tokens"] == judge_profile().max_tokens
        assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    if fail:
        assert diagnostics["requests"][1]["error_type"] == "OSError"
        assert diagnostics["status"] == "incomplete-grading"
        assert not diagnostics["rubric_grades"]
    else:
        assert diagnostics["status"] == "completed"
        assert [row["criteria_met"] for row in diagnostics["rubric_grades"]] == [True, False]
    assert closed == [True]


def test_failed_native_grader_records_private_cost_without_a_score(monkeypatch):
    import asyncio

    runtime = object.__new__(integrity_runtime.PrivateIntegrityRuntime)
    endpoint = "http://fixture-judge"
    runtime.config = {
        "endpoints": [endpoint],
        "scorers": {"healthbench": {"endpoint_base": endpoint, "model_route": "fixture-base"}},
    }
    runtime.replicas = ReplicaLoadBalancer(1)
    runtime.observed = [
        {
            "model_info": {"model_path": "fixture"},
            "server_info": {"served_model_name": "fixture-base"},
        }
    ]
    runtime.source = SimpleNamespace(targets={})
    runtime.sandbox = runtime.mbpp_sandbox = None
    events = []
    runtime.journal = SimpleNamespace(record=lambda *args, **kwargs: events.append((args, kwargs)))
    cost = {"model_request_attempts": 2.0, "model_responses": 1.0, "unknown_usage_calls": 1.0}

    async def fail(*args, **kwargs):
        raise IncompleteNativeGradingError(cost) from TimeoutError("synthetic timeout")

    monkeypatch.setattr(integrity_runtime, "score_native", fail)
    scope = ("fixture-run", "A2", "fixture-episode")
    with pytest.raises(IncompleteNativeGradingError):
        asyncio.run(runtime.score(None, scope, "healthbench"))
    failures = [
        (actual_scope, payload)
        for (actual_scope, stage, payload), _ in events
        if stage == "scorer-failure"
    ]
    assert len(failures) == 1
    actual_scope, payload = failures[0]
    assert actual_scope == scope
    assert payload["failure_type"] == "TimeoutError"
    assert payload["scorer_cost"] == cost
    assert all(metadata["origin"] is EventOrigin.SCORER for _, metadata in events)
    assert runtime.replicas.inflight == [0]


def test_report_counts_failed_grading_cost_once_and_only_in_its_scope(tmp_path):
    journal = CandidateJournal(tmp_path / "private.sqlite")
    scope = ("run", "A2", "episode")
    failure = {"scorer_cost": {"model_request_attempts": 3.0, "unknown_usage_calls": 1.0}}
    journal.record(scope, "scorer-failure", failure, origin=EventOrigin.SCORER)
    journal.record(
        ("another-run", "A2", "episode"), "scorer-failure", failure, origin=EventOrigin.SCORER
    )
    journal.record(("run", "A1", "episode"), "scorer-failure", failure, origin=EventOrigin.SCORER)
    score = NativeScore(
        "episode",
        "healthbench",
        "qwen-local-rubric-score",
        0.5,
        secondary_metrics={"triggered-negative-rubric-count": 0.0},
        verifier_version=NATIVE_VERIFIER_VERSIONS["healthbench"],
        scorer_cost={"model_request_attempts": 2.0, "unknown_usage_calls": 0.0},
        grader_used=True,
    )
    summary = scorer_cost_summary(journal, [score], (scope,))
    assert summary["records_with_cost"] == 1
    assert summary["failed_invocations"] == 1
    assert summary["completed_score_totals"]["model_request_attempts"] == 2
    assert summary["failed_invocation_totals"]["model_request_attempts"] == 3
    assert summary["totals"]["model_request_attempts"] == 5
    assert summary["totals"]["unknown_usage_calls"] == 1
    assert scorer_cost_summary(journal, [score], (scope,)) == summary
    journal.close()
