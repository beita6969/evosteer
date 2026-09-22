from skillev_private.benchmarks.mbpp_scoring import mbpp_failure_kind
from skillev_private.benchmarks.qa_diagnostics import qa_answer_diagnostics
from skillev_private.benchmarks.qa_metrics import (
    normalize_hotpotqa_answer,
    normalize_triviaqa_answer,
)


def test_qa_normalization_is_actual_domain_rule_not_an_alias_repair():
    aliases = ("a-b", "different")
    before = aliases
    for domain, normalizer in [
        ("hotpotqa", normalize_hotpotqa_answer),
        ("triviaqa", normalize_triviaqa_answer),
    ]:
        evidence = qa_answer_diagnostics(
            domain,
            original_submission="Final answer: A-B",
            projected_answer="A-B",
            accepted_aliases=aliases,
        )
        assert evidence["normalized_answer"] == normalizer("A-B")
        assert evidence["normalized_aliases"] == [normalizer(a) for a in aliases]
        assert evidence["exact_matching_alias_indices"] == [0]
        assert evidence["original_submission"] == "Final answer: A-B"
        assert evidence["cause"] is None
    assert aliases == before
    missing = qa_answer_diagnostics(
        "hotpotqa", original_submission=None, projected_answer=None, accepted_aliases=aliases
    )
    assert missing["normalized_answer"] is None
    assert missing["exact_matching_alias_indices"] is None


def test_mbpp_lane_diagnostics_reuse_native_syntax_and_lane_verdicts_without_new_score():
    assert mbpp_failure_kind({"base_passed": True, "plus_passed": True}) is None
    assert mbpp_failure_kind({"base_passed": True, "plus_passed": False}) == "plus-test-failure"
    assert (
        mbpp_failure_kind(
            {"base_passed": False, "plus_passed": False, "syntax": {"status": "invalid"}}
        )
        == "syntax-error"
    )
    assert (
        mbpp_failure_kind(
            {
                "base_passed": False,
                "plus_passed": False,
                "lanes": {"base": {"native_status": "timeout"}},
            }
        )
        == "base-timeout"
    )


def test_training_qa_private_alias_evidence_is_not_a_public_metric_or_prompt():
    import asyncio

    from skillev_private.benchmarks.protocol_v13_training_sessions import _StaticEvaluator
    from skillev_private.evaluation.result_contracts import public_native_metric_values

    from tests.benchmarks.test_protocol_v13_training_sessions import _record, _request

    record = _record()
    original = record.input.to_value()
    reward = asyncio.run(
        _StaticEvaluator(record, record.input).evaluate(
            _request(record.input.task_id, "Final answer: private answer")
        )
    )
    assert reward.success
    assert reward.value == 1
    assert reward.native_payload["qa_diagnostics"]["projected_answer"] == "private answer"
    assert reward.native_payload["qa_diagnostics"]["accepted_aliases"] == ["private answer"]
    assert "private answer" not in repr(public_native_metric_values(reward))
    assert record.input.to_value() == original
