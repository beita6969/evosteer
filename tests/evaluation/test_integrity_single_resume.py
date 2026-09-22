"""Formal runs never import candidates; historical inspection preserves their source."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from skillev.evaluation.integrity_pipeline import run_paired
from skillev.evaluation.integrity_resume import (
    EvaluationRunMode,
    read_historical_arm,
    require_execution_mode,
)
from skillev.evaluation.sealed_candidates import CandidateJournal
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_step0_integrity_pipeline import Runtime, panel, run


def single(runtime, directory: Path, source: Path | None = None, *, resume=False):
    return asyncio.run(
        run_paired(
            runtime,
            panel(),
            (InferenceArm("A2"),),
            run_id="single",
            directory=directory,
            canary=True,
            reuse_completed_from=source,
            run_mode=EvaluationRunMode.SAME_RUN_RESUME
            if resume
            else EvaluationRunMode.FORMAL_FRESH,
        )
    )


def test_single_arm_generates_and_scores_only_selected_policy(tmp_path):
    runtime = Runtime()
    report = single(runtime, tmp_path)
    assert runtime.generated == runtime.graded == 1
    assert len(report["arms"]) == 1
    assert report["benchmarks"]["aime-2026"]["count"] == 1
    assert report["benchmarks"]["aime-2026"]["value"] == 1.0
    assert "comparisons" not in report
    resumed = Runtime()
    assert single(resumed, tmp_path, resume=True) == report
    assert resumed.generated == resumed.graded == 0


def test_formal_cross_run_import_is_refused_before_any_output_or_model_call(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "single"
    run(Runtime(), source)
    runtime = Runtime()
    with pytest.raises(ValueError):
        single(runtime, destination, source)
    assert runtime.generated == runtime.graded == 0
    assert runtime.closed == 1
    assert not destination.exists()


def test_historical_read_retains_original_run_answers_and_definitive_scores(tmp_path):
    source = tmp_path / "source"
    run(Runtime(), source)
    journal = CandidateJournal(source / "candidates-private.sqlite")
    journal.record(("fixture", "A2", "incomplete"), "fixture-event", {"note": "unfinished"})
    original = journal.get("fixture", "A2", "one")
    score = journal.stored_score(("fixture", "A2", "one"))
    journal.close()
    snapshot = read_historical_arm(source, arm_id="A2")
    assert snapshot.origin_run_id == original.run_id == "fixture"
    assert snapshot.candidates == (original,)
    assert snapshot.definitive_scores == (score,)
    assert "unverified" in snapshot.status
    assert read_historical_arm(source, arm_id="A2") == snapshot


def test_fresh_and_resume_modes_are_explicit_and_cannot_change_controls(tmp_path):
    with pytest.raises(ValueError):
        single(Runtime(), tmp_path, resume=True)
    single(Runtime(), tmp_path)
    with pytest.raises(ValueError):
        single(Runtime(), tmp_path)
    changed = Runtime(budget=20)
    with pytest.raises(ValueError):
        single(changed, tmp_path, resume=True)
    assert changed.generated == changed.graded == 0


def test_missing_plan_does_not_make_an_old_journal_fresh(tmp_path):
    journal = CandidateJournal(tmp_path / "candidates-private.sqlite")
    journal.record(("old-run", "A2", "one"), "fixture-event", {})
    journal.close()
    with pytest.raises(ValueError):
        single(Runtime(), tmp_path)
    assert not (tmp_path / "frozen-plan-private.json").exists()


def test_historical_diagnostic_mode_has_no_generation_entrypoint(tmp_path):
    with pytest.raises(ValueError):
        require_execution_mode(
            EvaluationRunMode.HISTORICAL_DIAGNOSTIC, directory=tmp_path, importing_other_run=False
        )
