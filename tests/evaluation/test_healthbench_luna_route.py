import asyncio
import json
import threading
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks import healthbench_api as api
from skillev_private.benchmarks.protocol_v10_official import HealthBenchGrade
from skillev_private.benchmarks.protocol_v13_training_sessions import _HealthEvaluator
from skillev_private.evaluation import integrity_native_scoring as native
from skillev_private.evaluation.integrity_grader_usage import IncompleteNativeGradingError
from skillev_private.experiments.bayesian_condition_transition import resume_condition
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from scripts.configure_healthbench_judge import declare
from skillev.contracts import canonical_json
from skillev.evaluation.external_judge_policy import (
    DIRECT_HEALTHBENCH_PROFILE,
    EXTERNAL_JUDGE_API_BASE,
    EXTERNAL_JUDGE_KEY_ENV,
    EXTERNAL_JUDGE_MODEL,
    GATEWAY_HEALTHBENCH_PROFILE,
)
from skillev.evaluation.healthbench_luna_profile import (
    DIRECT_VERIFIER,
    LEGACY_TRAINING_JUDGE,
    METRIC,
    PROFILE_ID,
    VERIFIER,
    healthbench_condition,
    luna_profile,
)
from skillev.evaluation.integrity_results import EvaluationStatus, NativeScore, exact_panel_join
from skillev.training import AsyncResourceLimiter
from tests.benchmarks.test_protocol_v13_training_sessions import _record, _request


def test_new_profile_and_explicit_historical_profile():
    profile = native.resolve_healthbench_profile({})
    assert profile == luna_profile()
    assert (
        native.resolve_healthbench_profile(
            {"profile": "configs/evaluation/healthbench_luna_medium_failover.yaml"}
        )
        == profile
    )
    gateway = native.resolve_healthbench_profile(
        {"profile": "configs/evaluation/healthbench_flowsteer_luna_medium.yaml"}
    )
    assert gateway.profile_id == GATEWAY_HEALTHBENCH_PROFILE
    assert "routing_policy" not in healthbench_condition(gateway.profile_id)
    direct = native.resolve_healthbench_profile(
        {"profile": "configs/evaluation/healthbench_luna_medium.yaml"}
    )
    assert direct.profile_id == DIRECT_HEALTHBENCH_PROFILE
    old = native.resolve_healthbench_profile(
        {"profile": "configs/evaluation/protocol_v14_healthbench_grader.yaml"}
    )
    assert old.backend == "qwen-sglang"
    assert old.max_tokens == 2048
    assert old.temperature == 0.5
    assert native.resolve_healthbench_profile({"effective_profile": asdict(profile)}) == profile


def test_client_uses_private_key_file_not_actor_route(monkeypatch, tmp_path):
    secret = tmp_path / "private-key"
    secret.write_text("synthetic-credential")
    monkeypatch.delenv(EXTERNAL_JUDGE_KEY_ENV, raising=False)
    monkeypatch.setenv(f"{EXTERNAL_JUDGE_KEY_ENV}_FILE", str(secret))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://actor.invalid")
    import openai

    captured = {}
    raw = SimpleNamespace(close=lambda: None, responses=SimpleNamespace(create=lambda **_: None))

    def openai_client(**kwargs):
        captured.update(kwargs)
        return raw

    monkeypatch.setattr(openai, "OpenAI", openai_client)
    client = api.make_client()
    assert captured["api_key"] == "synthetic-credential"
    assert captured["base_url"] == EXTERNAL_JUDGE_API_BASE
    assert captured["max_retries"] == 0
    captured["http_client"].close()
    client.close()
    assert "synthetic-credential" not in canonical_json(healthbench_condition(PROFILE_ID))


def test_frozen_profile_selects_its_own_route(monkeypatch):
    from skillev_private.evaluation import external_judge_api, external_judge_failover

    chosen = []
    gateway_client, direct_client = object(), object()

    def gateway(**kwargs):
        chosen.append(kwargs["allow_official_fallback"])
        return gateway_client

    monkeypatch.setattr(external_judge_api, "make_external_judge_client", gateway)
    monkeypatch.setattr(external_judge_failover, "make_official_client", lambda: direct_client)
    assert api.make_client() is gateway_client
    assert chosen == [True]
    assert api.make_client(profile_id=GATEWAY_HEALTHBENCH_PROFILE) is gateway_client
    assert chosen == [True, False]
    assert api.make_client(profile_id=DIRECT_HEALTHBENCH_PROFILE) is direct_client
    assert chosen == [True, False]


@pytest.mark.parametrize(
    "profile_id", [PROFILE_ID, GATEWAY_HEALTHBENCH_PROFILE, DIRECT_HEALTHBENCH_PROFILE]
)
def test_native_score_retains_the_frozen_profile_identity(profile_id):
    candidate = SimpleNamespace(
        parser_id=native.CONTRACTS["healthbench"].parser, text="", episode_id="synthetic"
    )
    reader = SimpleNamespace(get=lambda *_: candidate)
    score = asyncio.run(
        native.score_native(
            reader,
            ("run", "arm", candidate.episode_id),
            "healthbench",
            {},
            settings={
                "healthbench": {
                    "effective_profile": asdict(native.external_healthbench_profile(profile_id))
                }
            },
            sandbox=None,
            mbpp_sandbox=None,
        )
    )
    condition = healthbench_condition(profile_id)
    assert score.metric == condition["metric"]
    assert score.verifier_version == condition["verifier"]
    assert score.status == EvaluationStatus.CANDIDATE_FAILURE
    assert score.grader_used is False


def fake_official(monkeypatch, verdicts):
    from skillev_private.benchmarks import protocol_v10_healthbench_worker as worker

    class Rubric:
        @staticmethod
        def from_dict(row):
            return row

    class Evaluator:
        def grade_sample(self, *, prompt, response_text, example_tags, rubric_items):
            rows = []
            for item in rubric_items:
                response = self.grader_model(
                    [
                        {
                            "role": "user",
                            "content": (
                                f"JSON rubric: {item['criterion']}\nCandidate: {response_text}"
                            ),
                        }
                    ]
                )
                rows.append({**item, **json.loads(response.response_text)})
            return {}, None, rows

    requests = []
    closed = []

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            verdict = next(verdicts)
            if isinstance(verdict, Exception):
                raise verdict
            return SimpleNamespace(
                model=EXTERNAL_JUDGE_MODEL,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(verdict)), finish_reason="stop"
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()), close=lambda: closed.append(True)
    )
    monkeypatch.setattr(api, "make_client", lambda **_: client)
    monkeypatch.setattr(worker, "_load_official", lambda path: (Evaluator, Rubric, SimpleNamespace))
    return requests, closed


def case():
    return {
        "prompt": [{"role": "user", "content": "synthetic medical task"}],
        "rubrics": [{"criterion": "helpful", "points": 2}, {"criterion": "harmful", "points": -3}],
    }


def test_official_path_uses_luna_medium_and_retains_native_negative_score(monkeypatch):
    requests, closed = fake_official(
        monkeypatch, iter([{"criteria_met": True}, {"criteria_met": True}])
    )
    raw, negative, cost = native._grade_health(
        "synthetic response", case(), {"official_source": "unused"}
    )
    assert raw == -0.5
    assert negative == 1
    assert cost["known_output_tokens"] == 14
    assert cost["model_request_attempts"] == 2
    assert cost["wall_seconds"] >= 0
    assert closed
    for request in requests:
        assert request["model"] == EXTERNAL_JUDGE_MODEL
        assert request["reasoning_effort"] == "medium"
        assert request["max_completion_tokens"] == 8000
        assert not request["store"]
        assert "temperature" not in request
        assert "seed" not in request
        assert "extra_body" not in request


def test_failed_rubric_never_becomes_false_verdict(monkeypatch):
    requests, closed = fake_official(
        monkeypatch, iter([TimeoutError("synthetic infrastructure failure")])
    )
    with pytest.raises(IncompleteNativeGradingError) as failure:
        native._grade_health("candidate", case(), {"official_source": "unused"})
    assert len(requests) == 1
    assert closed
    assert failure.value.scorer_cost["unknown_usage_calls"] == 1


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_incomplete_api_response_releases_capacity(reason, monkeypatch):
    limiter = threading.BoundedSemaphore(1)
    monkeypatch.setattr(api, "_API_CAPACITY", limiter)
    delegate = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason)])
    )
    with pytest.raises(RuntimeError):
        api.BoundedAPICompletions(delegate).create(timeout=1)
    assert limiter.acquire(blocking=False)
    limiter.release()


def test_api_grader_does_not_use_actor_lease(monkeypatch):
    fake_official(monkeypatch, iter([{"criteria_met": True}, {"criteria_met": False}]))
    grader = api.OpenAIHealthBenchGrader({"task": case()}, Path("unused"), AsyncResourceLimiter(1))
    grade = asyncio.run(grader.grade("task", "candidate"))
    assert grade.official_rubric_score == 1
    assert grade.triggered_negative_rubric_count == 0
    assert grade.grader_cost["known_input_tokens"] == 22


def test_durable_delivery_queue_does_not_consume_api_timeout(tmp_path, monkeypatch):
    from skillev.evaluation.judge_spool import JudgeSpoolClient

    waits, released, calls = [], [], []
    clock = [0.0]

    def acquire(**kwargs):
        waits.append(kwargs)
        clock[0] += 600
        return True

    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        api,
        "_API_CAPACITY",
        SimpleNamespace(acquire=acquire, release=lambda: released.append(True)),
    )
    delegate = JudgeSpoolClient(tmp_path)

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])

    monkeypatch.setattr(delegate, "create", create)
    api.BoundedAPICompletions(delegate).create(timeout=120)
    assert waits == [{}]
    assert calls == [{"timeout": 120.0, "store": False}]
    assert released == [True]


@pytest.mark.parametrize(
    ("raw", "negative", "success", "value"),
    [(1, 0, True, 1), (0.59, 0, False, 0.59), (0.8, 1, False, 0.8), (-0.5, 1, False, 0)],
)
def test_training_projection_preserves_threshold_and_clipping(raw, negative, success, value):
    async def grade(task_id, answer):
        return HealthBenchGrade(raw, negative)

    grader = SimpleNamespace(grade=grade, verifier_version=VERIFIER)
    task = replace(_record().input, public_context={"benchmark_id": "healthbench"})
    evaluator = _HealthEvaluator(task, grader, PROFILE_ID)
    request = _request(task.task_id, "answer")
    reward = asyncio.run(evaluator.evaluate(request))
    assert reward.value == value
    assert reward.success is success


def test_new_metric_cannot_mix_with_old_judge():
    row = NativeScore(
        "task",
        "healthbench",
        METRIC,
        -0.5,
        EvaluationStatus.SCORED,
        {"triggered-negative-rubric-count": 1.0},
        verifier_version=VERIFIER,
        grader_used=True,
    )
    assert exact_panel_join(("task",), (row,), expected_verifier=VERIFIER)
    with pytest.raises(ValueError):
        exact_panel_join(
            ("task",), (row,), expected_verifier="healthbench-qwen3.5-9b-sglang-temperature-0.5@2"
        )


@pytest.mark.parametrize("previous", [LEGACY_TRAINING_JUDGE, DIRECT_HEALTHBENCH_PROFILE])
def test_declaration_changes_only_judge_and_requires_boundary(tmp_path, previous):
    before = BayesianFormalConfig(healthbench_judge=previous)
    source = tmp_path / "formal-config.json"
    source.write_text(json.dumps(before.to_value()))
    destination = tmp_path / "luna.json"
    after = declare(source, destination)
    assert replace(after, healthbench_judge=previous) == before
    assert BayesianFormalConfig.load(destination) == after
    assert (
        resume_condition(tmp_path, after, allow_new_horizons=False, allow_healthbench_judge=True)
        == before
    )
    with pytest.raises(ValueError):
        resume_condition(tmp_path, after, allow_new_horizons=False)
    with pytest.raises(ValueError):
        resume_condition(
            tmp_path,
            replace(after, max_turns=25),
            allow_new_horizons=False,
            allow_healthbench_judge=True,
        )


def test_direct_scores_remain_readable_but_not_gateway_equivalent():
    old = NativeScore(
        "task",
        "healthbench",
        METRIC,
        -0.5,
        EvaluationStatus.SCORED,
        {"triggered-negative-rubric-count": 1.0},
        verifier_version=DIRECT_VERIFIER,
        grader_used=True,
    )
    restored = NativeScore.from_value(asdict(old))
    assert exact_panel_join(("task",), (restored,), expected_verifier=DIRECT_VERIFIER)
    with pytest.raises(ValueError):
        exact_panel_join(("task",), (restored,), expected_verifier=VERIFIER)
    condition = healthbench_condition(DIRECT_HEALTHBENCH_PROFILE)
    assert condition["endpoint"] != healthbench_condition(PROFILE_ID)["endpoint"]
    profile = asdict(luna_profile())
    for key in profile:
        profile[key] = condition[key]
    historical = native.resolve_healthbench_profile({"effective_profile": profile})
    assert historical.profile_id == DIRECT_HEALTHBENCH_PROFILE
    profile["profile_id"] = PROFILE_ID
    with pytest.raises(ValueError):
        native.resolve_healthbench_profile({"effective_profile": profile})
