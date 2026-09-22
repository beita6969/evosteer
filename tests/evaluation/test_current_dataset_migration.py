"""Synthetic-only catalog, public projection and native scorer integration."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from skillev_private.evaluation.iid_episode_sources import IIDSourceIdentity
from skillev_private.evaluation.integrity_sources import select_source_panel
from skillev_private.evaluation.ood_export import project_row, source_rows
from skillev_private.evaluation.ood_math_hard import final_math_answer, grade_math_hard
from skillev_private.evaluation.ood_scoring import score_ood

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS, OOD_BENCHMARKS, PublicTaskView
from skillev.evaluation.thinking_policy import ThinkingPolicy


def test_current_catalog_and_default_profiles_agree():
    catalog = json.loads(Path("configs/evaluation/current_datasets.json").read_text())
    assert tuple(catalog["iid"]) == IID_BENCHMARKS
    assert tuple(catalog["ood"]) == OOD_BENCHMARKS
    assert not catalog["native_thinking_default"]
    assert not {"humaneval", "webshop"} & set(IID_BENCHMARKS)
    assert not {"omni-math", "livemedbench", "livecodebench"} & set(OOD_BENCHMARKS)
    for filename, ood in [("current_iid_step0.yaml", False), ("current_ood_step0.json", True)]:
        value = yaml.safe_load((Path("configs/evaluation") / filename).read_text())
        assert ThinkingPolicy.from_value(value["thinking_policy"]) == ThinkingPolicy.default(
            ood=ood
        )
        expected_thinking = (
            {"math-hard", "gpqa-diamond-bioorganic"} if ood else {"aime-2026", "healthbench"}
        )
        for benchmark, profile in value.get("decoding", value.get("decoding_overrides")).items():
            assert profile["enable_thinking"] is (benchmark in expected_thinking)
        if not ood:
            assert value["evaluation_sample_counts"] == {
                name: 30 if name == "aime-2026" else 64 for name in IID_BENCHMARKS
            }
            standalone = yaml.safe_load(
                Path("configs/evaluation/step0_thinking_by_benchmark.yaml").read_text()
            )
            assert ThinkingPolicy.from_value(standalone) == ThinkingPolicy.default()
        if ood:
            for benchmark in expected_thinking:
                limits = value["budgets"][benchmark]
                assert limits["native_chunk_tokens"] == 6000
                assert limits["native_final_reserve_tokens"] == 2000
                assert limits["total_output_tokens"] == 8000


def test_upstream_json_array_with_jsonl_extension_is_read_without_dropping_rows(tmp_path):
    rows = [{"problem": "synthetic one"}, {"problem": "synthetic two"}]
    source = tmp_path / "algebra.jsonl"
    source.write_text(json.dumps(rows, indent=2))
    assert list(source_rows([str(source)])) == rows
    source.write_text("\n".join(json.dumps(row) for row in rows))
    assert list(source_rows([str(source)])) == rows


def test_historical_identity_reading_does_not_admit_a_retired_domain_to_new_panels():
    identity = IIDSourceIdentity("old", "humaneval", "source", "old-panel", "v1", "test")
    assert identity.canonical_source == ("humaneval", "source")
    with pytest.raises(ValueError):
        select_source_panel(SimpleNamespace(), {"humaneval": 1})


def test_math_projection_keeps_private_work_and_answer_out_of_actor():
    row = {
        "problem": "Synthetic task",
        "level": "Level 5",
        "type": "Algebra",
        "solution": r"PRIVATE WORK ends with \boxed{PRIVATE ANSWER}",
    }
    projected = project_row("math-hard", 0, row)
    view = PublicTaskView.from_record(projected["task_id"], "math-hard", projected["public"])
    assert view.input_profile == "released-ood-source@1"
    assert "PRIVATE" not in view.render()
    assert projected["target"]["answer"] == "PRIVATE ANSWER"
    assert "math-hard" in OOD_BENCHMARKS
    with pytest.raises(ValueError):
        project_row("math-hard", 0, {**row, "level": "Level 4"})


def test_math_final_box_is_not_selected_by_correctness_or_parse_success():
    assert final_math_answer(r"Draft \boxed{3}. Final \boxed{4}") == "4"
    assert final_math_answer(r"Draft \boxed{3}. Final \boxed{") is None
    assert final_math_answer("7") == "7"


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        (r"\boxed{0.5}", r"\frac{1}{2}", True),
        (r"\boxed{3} then \boxed{4}", "3", False),
        (r"\boxed{(-\infty,2)}", r"(-\infty,2)", True),
    ],
)
def test_math_verify_grades_only_final_answer_in_isolated_worker(prediction, reference, expected):
    sandbox = ActorSandbox.current(Path("src"))
    result = grade_math_hard(prediction, reference, sandbox)
    assert result["passed"] is expected


def test_math_scoring_dispatch_and_infrastructure_failure_do_not_call_external_judge(monkeypatch):
    from skillev_private.evaluation import ood_math_hard

    def grade(candidate, reference, sandbox):
        assert candidate == r"\boxed{7}"
        assert reference == "7"
        return {"passed": True, "prediction_parsed": True}

    monkeypatch.setattr(ood_math_hard, "grade_math_hard", grade)
    reader = SimpleNamespace(get=lambda *args: SimpleNamespace(text=r"\boxed{7}", episode_id="e"))
    kwargs = {"settings": {}, "sandbox": None, "diagnostics": lambda _: None}
    result = asyncio.run(score_ood(reader, ("r", "a", "e"), "math-hard", {"answer": "7"}, **kwargs))
    assert result.value == 1.0
    assert result.metric == "accuracy"

    def unavailable(*args):
        raise RuntimeError("synthetic worker unavailable")

    monkeypatch.setattr(ood_math_hard, "grade_math_hard", unavailable)
    with pytest.raises(RuntimeError):
        asyncio.run(score_ood(reader, ("r", "a", "e"), "math-hard", {"answer": "7"}, **kwargs))
