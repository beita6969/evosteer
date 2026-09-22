"""Synthetic terminal judges only; these tests never call OpenAI or read real keys."""

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from skillev_private.evaluation.external_judge_api import (
    make_external_judge_client,
    make_healthbench_external_sampler,
)
from skillev_private.evaluation.ood_api_judge import grade_candidate, judge_batch, parse_report
from skillev_private.evaluation.ood_luna_judge import (
    JUDGE_EFFORT,
    JUDGE_MODEL,
    JUDGE_PROFILE,
    read_judgement,
)
from skillev_private.evaluation.ood_scoring import score_ood

from skillev.evaluation.external_judge_policy import (
    DIRECT_JUDGE_MODEL,
    DIRECT_OMNI_JUDGE_PROFILE,
    DIRECT_OMNI_JUDGE_VERIFIER,
    EXTERNAL_JUDGE_API_BASE,
    EXTERNAL_JUDGE_KEY_ENV,
    EXTERNAL_JUDGE_MAX_TOKENS,
    GATEWAY_HEALTHBENCH_PROFILE,
    LEGACY_OMNI_JUDGE_METRIC,
    LEGACY_OMNI_JUDGE_PROFILE,
    LEGACY_OMNI_JUDGE_VERIFIER,
    OMNI_JUDGE_METRIC,
    OMNI_JUDGE_VERIFIER,
)
from skillev.evaluation.healthbench_transport import HealthBenchTransportError
from skillev.evaluation.integrity_results import NativeScore, exact_panel_join


@pytest.fixture(autouse=True)
def isolated_route(monkeypatch):
    from skillev_private.evaluation import external_judge_failover as failover

    monkeypatch.setattr(failover, "PROCESS_ROUTE", failover.JudgeRouteState())


def _report(verdict="TRUE"):
    return (
        "## Student Final Answer\nSynthetic answer\n"
        f"## Equivalence Judgement\n{verdict}\n"
        "## Justification\nSynthetic equivalence explanation.\n=== report over ==="
    )


def _response(verdict="TRUE", *, finish="stop"):
    return SimpleNamespace(
        id="response-fixture",
        model=JUDGE_MODEL,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=_report(verdict), refusal=None),
                finish_reason=finish,
            )
        ],
        usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 20}),
    )


def _batch(count=3):
    return {
        "profile": JUDGE_PROFILE,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_EFFORT,
        "backend": "openai-chat-completions",
        "max_completion_tokens": EXTERNAL_JUDGE_MAX_TOKENS,
        "planned_count": count,
        "records": [
            {
                "run_id": "run",
                "arm_id": "arm",
                "task_id": f"task-{index}",
                "owner_call_id": f"call-{index}",
                "judge_prompt": f"Synthetic prompt {index}",
            }
            for index in range(count)
        ],
    }


def _client(responses):
    pending = iter(responses)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        return response

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ), requests


def test_terminal_omni_api_reuses_false_and_never_retries_failed_judging(tmp_path):
    template = tmp_path / "template.txt"
    template.write_text("{{Problem}}\n{{Reference Answer}}\n{{Solution}}")
    settings = {"template_path": str(template), "cache_directory": str(tmp_path / "judge")}
    candidate = SimpleNamespace(
        run_id="r", arm_id="a", episode_id="synthetic", owner_call_id="c", text="final answer"
    )
    target = {"problem": "Synthetic question", "answer": "Private reference"}
    client, requests = _client([_response("FALSE")])
    result = grade_candidate(candidate, target, settings, client=client)
    assert result["passed"] is False
    assert result["cost"]["model_calls"] == 1
    assert grade_candidate(candidate, target, settings, client=client) == result
    assert len(requests) == 1
    candidate.text = "changed answer"
    with pytest.raises(ValueError):
        grade_candidate(candidate, target, settings, client=client)
    candidate.episode_id = "unresolved"
    client, requests = _client([TimeoutError("synthetic")])
    for _ in range(2):
        with pytest.raises(RuntimeError):
            grade_candidate(candidate, target, settings, client=client)
    assert len(requests) == 1


def test_gateway_requires_its_own_key_and_disables_sdk_retries(monkeypatch, tmp_path):
    from pathlib import Path

    import openai

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv(EXTERNAL_JUDGE_KEY_ENV, raising=False)
    monkeypatch.delenv(f"{EXTERNAL_JUDGE_KEY_ENV}_FILE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "official-key-must-not-reach-the-gateway")
    with pytest.raises(RuntimeError):
        make_external_judge_client()
    monkeypatch.setenv(EXTERNAL_JUDGE_KEY_ENV, "synthetic-not-a-real-key")
    settings = {}
    raw = SimpleNamespace(close=lambda: None, responses=SimpleNamespace(create=lambda **_: None))

    def client(**kwargs):
        settings.update(kwargs)
        return raw

    monkeypatch.setattr(openai, "OpenAI", client)
    wrapped = make_external_judge_client()
    assert settings["api_key"] == "synthetic-not-a-real-key"
    assert settings["base_url"] == EXTERNAL_JUDGE_API_BASE
    assert settings["max_retries"] == 0
    assert not settings["http_client"].follow_redirects
    wrapped.close()
    settings["http_client"].close()


def test_gateway_adapter_uses_one_sync_tool_free_responses_request(monkeypatch, tmp_path):
    from pathlib import Path

    import openai

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv(EXTERNAL_JUDGE_KEY_ENV, "synthetic-not-a-real-key")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="lab-gpt-5.6-luna",
            status="completed",
            output_text='{"criteria_met":true}',
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            _request_id="gateway-request-1",
        )

    raw = SimpleNamespace(
        close=lambda: None,
        responses=SimpleNamespace(create=create),
        models=SimpleNamespace(list=lambda **_: None),
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **_: raw)
    client = make_external_judge_client()
    response = client.chat.completions.create(
        model="lab-gpt-5.6-luna",
        messages=[{"role": "user", "content": "grade"}],
        reasoning_effort="medium",
        max_completion_tokens=8000,
        timeout=120,
        stream=False,
        store=False,
    )
    assert calls == [
        {
            "model": "lab-gpt-5.6-luna",
            "input": [{"role": "user", "content": "grade"}],
            "max_output_tokens": 8000,
            "reasoning": {"effort": "medium"},
            "stream": False,
            "store": False,
            "timeout": 120,
        }
    ]
    assert response.choices[0].message.content == '{"criteria_met":true}'
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 7


def test_medium_api_failure_stops_and_explicit_resume_never_rejudges_true_or_false(tmp_path):
    batch = _batch()
    output = tmp_path / "judgements.json"
    client, requests = _client([_response("FALSE"), TimeoutError("private diagnostic")])
    state = judge_batch(batch, output, client=client)
    assert not state["complete"]
    assert state["resolved_count"] == 1
    assert state["planned_count"] == 3
    assert len(requests) == 2
    assert state["records"][0]["equivalent"] is False
    assert "equivalent" not in state["records"][1]
    assert "private diagnostic" not in output.read_text()
    assert state["records"][2]["status"] == "not-requested"
    assert state["records"][1]["attempts"][0]["usage"] is None

    client, requests = _client([_response(), _response()])
    completed = judge_batch(batch, output, client=client, resume=True)
    assert completed["complete"]
    assert completed["attempt_count"] == 4
    assert len(completed["records"][1]["attempts"]) == 2
    assert [r["messages"][0]["content"] for r in requests] == [
        "Synthetic prompt 1",
        "Synthetic prompt 2",
    ]
    for request in requests:
        assert request["model"] == JUDGE_MODEL
        assert request["reasoning_effort"] == "medium"
        assert request["max_completion_tokens"] == 8000
        assert request["store"] is False
        assert "temperature" not in request
        assert "top_p" not in request
        assert "tools" not in request
    assert completed["records"][2]["attempts"][0]["usage"]["completion_tokens"] == 20
    with pytest.raises(FileExistsError):
        judge_batch(batch, output, client=client)
    changed = _batch()
    changed["records"][1]["judge_prompt"] = "changed candidate"
    with pytest.raises(ValueError):
        judge_batch(changed, output, client=client, resume=True)


@pytest.mark.parametrize(("verdict", "finish"), [("uncertain", "stop"), ("TRUE", "length")])
def test_unparseable_or_truncated_judge_never_becomes_a_boolean(tmp_path, verdict, finish):
    client, requests = _client([_response(verdict, finish=finish)])
    state = judge_batch(_batch(1), tmp_path / "judgements.json", client=client)
    assert not state["complete"]
    assert "equivalent" not in state["records"][0]
    assert state["records"][0]["attempts"][0]["raw_output"] == _report(verdict)
    assert len(requests) == 1


def test_report_parser_rejects_duplicate_and_incomplete_reports():
    assert parse_report(_report("FALSE"))[0] is False
    for report in (
        _report() + "\n## Equivalence Judgement\nFALSE",
        _report().replace("=== report over ===", ""),
    ):
        with pytest.raises(ValueError):
            parse_report(report)


@pytest.mark.parametrize(
    ("profile", "effort", "metric", "verifier"),
    [
        (LEGACY_OMNI_JUDGE_PROFILE, "high", LEGACY_OMNI_JUDGE_METRIC, LEGACY_OMNI_JUDGE_VERIFIER),
        (DIRECT_OMNI_JUDGE_PROFILE, "medium", OMNI_JUDGE_METRIC, DIRECT_OMNI_JUDGE_VERIFIER),
    ],
)
def test_legacy_omni_readable_but_never_relabelled_or_mixed(
    tmp_path, profile, effort, metric, verifier
):
    candidate = SimpleNamespace(
        run_id="run", arm_id="arm", episode_id="task-0", owner_call_id="call-0", text="answer"
    )
    batch = _batch(1)
    batch.update(profile=profile, reasoning_effort=effort, judge_model=DIRECT_JUDGE_MODEL)
    batch["records"][0].update(equivalent=True, reason="Synthetic justification")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(batch))
    settings = {"judgements_path": str(path)}
    with pytest.raises(ValueError):
        read_judgement(candidate, settings)
    settings["judge_profile"] = profile
    verdict = read_judgement(candidate, settings)
    assert verdict["metric"] == metric
    with pytest.raises(ValueError):
        judge_batch(batch, tmp_path / "new.json", client=None)

    score = asyncio.run(
        score_ood(
            SimpleNamespace(get=lambda *args: candidate),
            ("run", "arm", "task-0"),
            "omni-math",
            {},
            settings={"omni-math": settings},
            sandbox=None,
            diagnostics=lambda row: None,
        )
    )
    restored = NativeScore.from_value(asdict(score))
    assert restored.verifier_version == verifier
    assert exact_panel_join(("task-0",), (restored,), expected_verifier=verifier) == (restored,)
    with pytest.raises(ValueError):
        exact_panel_join(("task-0",), (restored,), expected_verifier=OMNI_JUDGE_VERIFIER)


def test_medium_judge_import_uses_new_native_metric(tmp_path):
    client, _ = _client([_response()])
    path = tmp_path / "medium.json"
    judge_batch(_batch(1), path, client=client)
    candidate = SimpleNamespace(
        run_id="run", arm_id="arm", episode_id="task-0", owner_call_id="call-0", text="answer"
    )
    score = asyncio.run(
        score_ood(
            SimpleNamespace(get=lambda *args: candidate),
            ("run", "arm", "task-0"),
            "omni-math",
            {},
            settings={"omni-math": {"judgements_path": str(path)}},
            sandbox=None,
            diagnostics=lambda row: None,
        )
    )
    assert score.metric == OMNI_JUDGE_METRIC
    assert score.verifier_version == OMNI_JUDGE_VERIFIER
    assert score.value == 1


def test_future_external_healthbench_sampler_uses_medium_not_legacy_low():
    client, requests = _client([_response()])
    sampler = make_healthbench_external_sampler(response_type=SimpleNamespace, client=client)
    response = sampler([{"role": "user", "content": "Synthetic rubric JSON request"}])
    assert requests[0]["model"] == JUDGE_MODEL
    assert requests[0]["reasoning_effort"] == "medium"
    assert requests[0]["max_completion_tokens"] == EXTERNAL_JUDGE_MAX_TOKENS
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert response.response_metadata["judge_profile"] == GATEWAY_HEALTHBENCH_PROFILE


def test_external_healthbench_truncation_is_not_a_rubric_verdict():
    client, requests = _client([_response(finish="length")])
    sampler = make_healthbench_external_sampler(response_type=SimpleNamespace, client=client)
    with pytest.raises(HealthBenchTransportError):
        sampler([{"role": "user", "content": "Synthetic rubric"}])
    assert len(requests) == 1
