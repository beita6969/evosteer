from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillev.evaluation.healthbench_official import (
    CANDIDATE_MAX_TOKENS,
    CANDIDATE_TEMPERATURE,
    OPENAI_SYSTEM_MESSAGE,
    SAMPLE_COUNT,
    ExternalHealthBenchRubricSampler,
    ExternalJudgeTransport,
    HealthBenchTransportError,
    HealthBenchTransportExhausted,
    JsonlJournal,
    QwenChatTransport,
    QwenHealthBenchRubricSampler,
    QwenOfficialCandidateSampler,
    RoundRobinSampler,
    RubricAttemptRegistry,
    TransportRetryPolicy,
    aggregate_official_scores,
    candidate_usage_dict,
    generate_candidates,
    normalize_healthbench_model_invalid_scores,
)


@dataclass
class _SamplerResponse:
    response_text: str
    response_metadata: dict[str, object]
    actual_queried_message_list: list[dict[str, str]]


class _FakeCompletions:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None
        self.payloads: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.payload = kwargs
        self.payloads.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="medical answer"), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(),
        )


def _sampler() -> tuple[QwenOfficialCandidateSampler, _FakeCompletions]:
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return (
        QwenOfficialCandidateSampler(
            client=client,
            model="qwen35-direct-base",
            sampler_response_type=_SamplerResponse,
        ),
        completions,
    )


def test_candidate_changes_only_backbone_transport_controls() -> None:
    sampler, completions = _sampler()
    response = sampler([{"role": "user", "content": "question"}])
    assert response.response_text == "medical answer"
    assert completions.payload is not None
    for key, value in {
        "model": "qwen35-direct-base",
        "messages": [
            {"role": "system", "content": OPENAI_SYSTEM_MESSAGE},
            {"role": "user", "content": "question"},
        ],
        "temperature": CANDIDATE_TEMPERATURE,
        "max_tokens": CANDIDATE_MAX_TOKENS,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }.items():
        assert completions.payload[key] == value
    assert completions.payload["timeout"] == 120


def test_qwen_transport_maps_disabled_top_k_to_sglang_wire_sentinel() -> None:
    completions = _FakeCompletions()
    transport = QwenChatTransport(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="qwen35-direct-base",
        retry_policy=TransportRetryPolicy(),
        transient_error_types=(),
        top_k=0,
        seed=0,
    )

    transport.create(messages=[{"role": "user", "content": "question"}], response_format=None)

    assert completions.payload is not None
    assert completions.payload["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": -1,
        "seed": 0,
    }


def test_external_gpt51_grader_request_matches_pinned_protocol() -> None:
    completions = _FakeCompletions()
    transport = ExternalJudgeTransport(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="gpt-51-1113-global",
        reasoning_effort="low",
        max_completion_tokens=4096,
        retry_policy=TransportRetryPolicy(),
        transient_error_types=(),
    )
    sampler = ExternalHealthBenchRubricSampler(
        transport=transport,
        response_type=_SamplerResponse,
    )
    response = sampler([{"role": "user", "content": "grade this rubric"}])
    assert response.response_text == "medical answer"
    assert completions.payload is not None
    for key, value in {
        "model": "gpt-51-1113-global",
        "messages": [{"role": "user", "content": "grade this rubric"}],
        "stream": False,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4096,
        "reasoning_effort": "low",
    }.items():
        assert completions.payload[key] == value
    assert completions.payload["timeout"] == 120
    assert "temperature" not in completions.payload
    assert "top_p" not in completions.payload


class _EmptyCompletions(_FakeCompletions):
    def create(self, **kwargs: object) -> object:
        self.payload = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=SimpleNamespace(),
        )


def test_external_grader_missing_response_is_infrastructure_not_false() -> None:
    transport = ExternalJudgeTransport(
        client=SimpleNamespace(chat=SimpleNamespace(completions=_EmptyCompletions())),
        model="gpt-51-1113-global",
        retry_policy=TransportRetryPolicy(),
        transient_error_types=(),
    )
    sampler = ExternalHealthBenchRubricSampler(
        transport=transport,
        response_type=_SamplerResponse,
    )
    with pytest.raises(HealthBenchTransportError):
        sampler([{"role": "user", "content": "grade this rubric"}])


def test_candidate_generation_never_passes_rubrics(tmp_path: Path) -> None:
    sampler, completions = _sampler()
    examples = [
        {
            "prompt_id": f"p-{index}",
            "prompt": [{"role": "user", "content": f"question {index}"}],
            "rubrics": [{"criterion": "SECRET", "points": 1}],
        }
        for index in range(SAMPLE_COUNT)
    ]
    records = generate_candidates(
        examples=examples,
        sampler=sampler,
        journal=JsonlJournal(tmp_path / "candidates.jsonl"),
        usage_parser=lambda _usage: {},
        n_threads=4,
    )
    assert len(records) == SAMPLE_COUNT
    assert completions.payload is not None
    assert "SECRET" not in str(completions.payload["messages"])


def test_aggregate_clips_after_mean_and_has_no_project_sr() -> None:
    records = [
        {"metrics": {"overall_score": -1.0, "example_tag": 0.25}},
        {"metrics": {"overall_score": 1.0, "example_tag": 0.75}},
        *({"metrics": {"overall_score": 0.5}} for _ in range(SAMPLE_COUNT - 2)),
    ]

    def official_stat(values: list[float], stat: str) -> float | int:
        if stat == "mean":
            return min(1.0, max(0.0, sum(values) / len(values)))
        if stat == "n_samples":
            return len(values)
        if stat == "bootstrap_std":
            return 0.01
        raise AssertionError(stat)

    result = aggregate_official_scores(records, clipped_stat=official_stat)
    assert result["overall_score"] == pytest.approx((SAMPLE_COUNT - 2) * 0.5 / SAMPLE_COUNT)
    assert result["overall_score:n_samples"] == SAMPLE_COUNT
    assert result["example_tag"] == pytest.approx(0.5)
    assert result["example_tag:n_samples"] == 2
    assert "success_rate" not in result


def test_empty_candidate_is_a_definitive_metric_zero() -> None:
    scores = {
        "empty": {"metrics": {"overall_score": -0.5, "tag:score": 0.25}},
        "valid": {"metrics": {"overall_score": 0.75, "tag:score": 0.5}},
    }
    candidates = {
        "empty": {"response_text": "  \n"},
        "valid": {"response_text": "candidate answer"},
    }
    normalized, invalid_count = normalize_healthbench_model_invalid_scores(
        scores=scores,
        candidates=candidates,
    )
    assert invalid_count == 1
    assert normalized["empty"]["metrics"] == {
        "overall_score": 0.0,
        "tag:score": 0.0,
    }
    assert normalized["empty"]["model_output_invalid"] is True
    assert normalized["valid"]["metrics"] == scores["valid"]["metrics"]
    assert normalized["valid"]["model_output_invalid"] is False


def test_candidate_usage_accepts_missing_sglang_detail_objects() -> None:
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    assert candidate_usage_dict(usage) == {
        "input_tokens": 11,
        "input_cached_tokens": None,
        "output_tokens": 7,
        "output_reasoning_tokens": None,
        "total_tokens": 18,
    }


def test_candidate_route_rejects_adapter_names() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    with pytest.raises(ValueError, match="adapter-free"):
        QwenOfficialCandidateSampler(
            client=client,
            model="qwen-lora-adapter",
            sampler_response_type=_SamplerResponse,
        )


def test_round_robin_sampler_distributes_calls_across_replicas() -> None:
    calls: list[str] = []

    def first(_messages: object) -> str:
        calls.append("first")
        return "a"

    def second(_messages: object) -> str:
        calls.append("second")
        return "b"

    sampler = RoundRobinSampler((first, second))
    messages = [{"role": "user", "content": "rubric"}]
    assert [sampler(messages), sampler(messages), sampler(messages)] == ["a", "b", "a"]
    assert calls == ["first", "second", "first"]


def test_qwen_grader_adds_rubric_schema_only_after_official_parser_retry() -> None:
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sampler = QwenOfficialCandidateSampler(
        client=client,
        model="qwen35-direct-base",
        sampler_response_type=_SamplerResponse,
        rubric_schema_on_retry=True,
    )
    messages = [{"role": "user", "content": "Return the rubric JSON."}]
    sampler(messages)
    sampler(messages)
    assert "response_format" not in completions.payloads[0]
    response_format = completions.payloads[1]["response_format"]
    assert isinstance(response_format, dict)
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["criteria_met"]
    assert schema["properties"]["criteria_met"] == {"type": "boolean"}


class _TransientError(OSError):
    pass


class _FailingCompletions(_FakeCompletions):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def create(self, **kwargs: object) -> object:
        self.payloads.append(kwargs)
        if self.failures:
            self.failures -= 1
            raise _TransientError("temporary")
        return super().create(**kwargs)


def test_transport_retry_is_bounded_and_does_not_change_semantic_contract() -> None:
    completions = _FailingCompletions(3)
    transport = QwenChatTransport(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="base",
        retry_policy=TransportRetryPolicy(
            maximum_attempts=3,
            request_timeout_seconds=1,
            total_deadline_seconds=2,
            initial_backoff_seconds=0,
        ),
        transient_error_types=(_TransientError,),
        sleeper=lambda _delay: None,
    )
    with pytest.raises(HealthBenchTransportExhausted):
        transport.create(messages=[{"role": "user", "content": "rubric"}], response_format=None)
    assert len(completions.payloads) == 3
    assert all("response_format" not in payload for payload in completions.payloads)


def test_replica_pool_shares_parser_retry_state() -> None:
    first = _FakeCompletions()
    second = _FakeCompletions()
    policy = TransportRetryPolicy()
    sampler = QwenHealthBenchRubricSampler(
        transports=(
            QwenChatTransport(
                SimpleNamespace(chat=SimpleNamespace(completions=first)),
                "base",
                policy,
                (),
            ),
            QwenChatTransport(
                SimpleNamespace(chat=SimpleNamespace(completions=second)),
                "base",
                policy,
                (),
            ),
        ),
        response_type=_SamplerResponse,
        semantic_attempts=RubricAttemptRegistry(),
    )
    messages = [{"role": "user", "content": "rubric"}]
    sampler(messages)
    sampler(messages)
    used = first if first.payloads else second
    unused = second if first.payloads else first
    assert len(used.payloads) == 2
    assert not unused.payloads
    assert "response_format" not in used.payloads[0]
    assert "response_format" in used.payloads[1]
    with pytest.raises(RuntimeError):
        sampler(messages)
    assert len(used.payloads) == 2
    assert sampler.semantic_attempts.semantic_repair_count == 1
