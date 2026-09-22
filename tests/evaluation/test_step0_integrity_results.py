from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.integrity_results import (
    EvaluationStatus,
    ExecutionControls,
    NativeScore,
    PublicationStatus,
    compare_paired,
    exact_panel_join,
    native_mean,
    native_score_diagnostics,
)
from skillev.evaluation.public_validation_guard import choose_order_invariant_pairwise_candidate
from skillev.evaluation.step0_integrity import (
    AgentTopology,
    InferenceArm,
    InterventionCounts,
    SkillMode,
    validate_paired_intervention,
)


def native_score(
    task_id: str,
    benchmark: str,
    value: float,
    *,
    status: EvaluationStatus = EvaluationStatus.SCORED,
    secondary_metrics: dict[str, float] | None = None,
    grader_used: bool | None = None,
) -> NativeScore:
    if secondary_metrics is None:
        secondary_metrics = {
            "hotpotqa": {"answer-exact-match": value, "answer-f1": value},
            "triviaqa": {"answer-exact-match": value, "answer-f1": value},
            "webshop": {"success": float(value == 1.0)},
            "alfworld": {"success": value},
            "mbpp-plus": {"base-pass": value, "plus-pass": value},
            "healthbench": {"triggered-negative-rubric-count": 0.0},
        }.get(benchmark, {})
    if benchmark == "healthbench" and grader_used is None:
        grader_used = True
    return NativeScore(
        task_id,
        benchmark,
        {
            "hotpotqa": "answer-f1",
            "triviaqa": "answer-f1",
            "aime-2026": "accuracy",
            "healthbench": "qwen-local-rubric-score",
            "webshop": "native-reward",
            "alfworld": "success",
            "mbpp-plus": "base-plus-pass-at-1",
            "humaneval": "pass-at-1",
        }[benchmark],
        value,
        status,
        secondary_metrics,
        verifier_version=NATIVE_VERIFIER_VERSIONS[benchmark],
        grader_used=grader_used,
    )


def test_text_skill_ablation_cannot_also_change_the_agent_topology() -> None:
    a2 = InferenceArm("A2", agent_topology=AgentTopology.MULTI_AGENT)
    a3 = replace(a2, arm_id="A3", skill_mode=SkillMode.GENERIC_TEXT)
    validate_paired_intervention(a2, a3)
    with pytest.raises(ValueError):
        validate_paired_intervention(a2, replace(a3, agent_topology=AgentTopology.SINGLE))


def test_binary_native_success_is_not_a_fractional_or_truthy_score() -> None:
    with pytest.raises(ValueError):
        native_score("synthetic", "aime-2026", 0.5)


def test_counterfactual_labels_never_change_the_actor_projection() -> None:
    row = {
        "question": "Synthetic public question",
        "public_context": "Public evidence.",
        "answer": "private A",
        "aliases": ["private A"],
        "rubrics": ["private rubric"],
    }
    a = PublicTaskView.from_record("example", "triviaqa", row)
    b = PublicTaskView.from_record(
        "example",
        "triviaqa",
        {**row, "answer": "private B", "aliases": ["private B"], "rubrics": []},
    )
    assert a == b
    assert "private A" not in a.render()


def test_health_projection_does_not_copy_nested_message_metadata() -> None:
    task = PublicTaskView.from_record(
        "example",
        "healthbench",
        {
            "prompt": [{"role": "user", "content": "Public request", "rubric": "hidden"}],
            "rubrics": [{"criterion": "private"}],
        },
    )
    assert "Public request" in task.render()
    assert "hidden" not in task.render()


@pytest.mark.parametrize("observed", [("one",), ("one", "one"), ("one", "two", "extra")])
def test_missing_duplicate_and_extra_rows_do_not_change_the_denominator(
    observed: tuple[str, ...],
) -> None:
    scores = tuple(native_score(item, "hotpotqa", 1.0) for item in observed)
    with pytest.raises(ValueError):
        exact_panel_join(
            ("one", "two"), scores, expected_verifier=NATIVE_VERIFIER_VERSIONS["hotpotqa"]
        )


def test_candidate_failures_count_but_infrastructure_is_incomplete() -> None:
    a = native_score("one", "hotpotqa", 1.0)
    b = native_score("two", "hotpotqa", 0.0, status=EvaluationStatus.CANDIDATE_FAILURE)
    joined = exact_panel_join(
        ("one", "two"), (b, a), expected_verifier=NATIVE_VERIFIER_VERSIONS["hotpotqa"]
    )
    assert native_mean(joined) == 0.5
    with pytest.raises(RuntimeError):
        exact_panel_join(
            ("one", "two"),
            (a, replace(b, status=EvaluationStatus.INFRASTRUCTURE_FAILURE)),
            expected_verifier=NATIVE_VERIFIER_VERSIONS["hotpotqa"],
        )


def test_health_negative_items_are_averaged_before_clipping() -> None:
    scores = (
        native_score("one", "healthbench", -0.5),
        native_score("two", "healthbench", 1.0),
    )
    assert (
        native_mean(
            exact_panel_join(
                ("one", "two"),
                scores,
                expected_verifier=NATIVE_VERIFIER_VERSIONS["healthbench"],
            )
        )
        == 0.25
    )


def test_native_diagnostics_keeps_failed_candidates_and_explicit_alfworld_splits() -> None:
    scores = (
        native_score("seen-success", "alfworld", 1.0),
        native_score("seen-failed", "alfworld", 0.0, status=EvaluationStatus.CANDIDATE_FAILURE),
        native_score("unseen-failed", "alfworld", 0.0),
        native_score("name-looks-valid_seen", "alfworld", 1.0),
    )
    split_labels = ("valid_seen", "valid_seen", "valid_unseen")
    environment = {
        "cases": {
            row.task_id: {"case": {"payload": {"split": split}}}
            for row, split in zip(scores[:3], split_labels, strict=True)
        }
    }
    report = native_score_diagnostics(scores, environment)
    assert report["status_counts"]["candidate-failure"] == 1
    assert report["splits"]["valid_seen"]["count"] == 2
    assert report["splits"]["valid_seen"]["value"] == 0.5
    assert report["splits"]["valid_unseen"]["count"] == 1
    assert report["splits"]["valid_unseen"]["value"] == 0.0
    assert report["split_metadata_missing_count"] == 1
    assert native_mean(scores) == 0.5
    missing = native_score_diagnostics(scores, {})
    assert missing["split_metadata_missing_count"] == 4
    assert missing["splits"]["valid_seen"]["value"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_metrics_cannot_enter_an_aggregate(value: float) -> None:
    with pytest.raises(ValueError):
        native_score("example", "hotpotqa", value)


def controls() -> ExecutionControls:
    return ExecutionControls(
        model={"weights": "synthetic-qwen"},
        tokenizer={"vocabulary": "synthetic"},
        service={"max_context_length": 65536},
        public_inputs=(("one", "public question"),),
        tools={},
        environment={},
        evaluator={
            "implementation": "official synthetic scorer",
            "verifier_versions": dict(NATIVE_VERIFIER_VERSIONS),
        },
        parser={"version": "single-final@1"},
        sampling={"temperature": 0.7, "seed": 0},
        budgets={"output_tokens": 8192, "calls": 1},
    )


def test_actual_descriptor_difference_cannot_be_overridden_by_caller_booleans() -> None:
    a = controls()
    b = replace(a, budgets={"output_tokens": 16384, "calls": 1})
    assert a.differences(b) == ("budgets",)
    scores = (native_score("one", "hotpotqa", 1.0),)
    with pytest.raises(ValueError):
        compare_paired(
            ("one",),
            scores,
            scores,
            left_controls=a,
            right_controls=b,
            left_arm=InferenceArm("a"),
            right_arm=InferenceArm("b"),
            left_counts=InterventionCounts(),
            right_counts=InterventionCounts(),
            left_expected_verifier=NATIVE_VERIFIER_VERSIONS["hotpotqa"],
            right_expected_verifier=NATIVE_VERIFIER_VERSIONS["hotpotqa"],
        )


def test_clean_negative_result_remains_publishable_and_high_contaminated_result_does_not() -> None:
    clean = PublicationStatus("pass", "complete", "pass", "not-improved", "not-matched")
    assert clean.publishable
    assert not replace(
        clean, integrity_status="contaminated", performance_goal_status="improved"
    ).publishable


def test_active_label_swapped_selector_is_disabled() -> None:
    with pytest.raises(RuntimeError):
        choose_order_invariant_pairwise_candidate(())
