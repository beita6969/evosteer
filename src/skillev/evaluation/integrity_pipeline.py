"""Single-arm or paired execution with insert-once answers.

This uses no retired approval, receipt, attestation, or seal_* script. A private
SQLite record merely prevents replacing a submitted answer after seeing a score.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from .input_metric_contracts import IID_BENCHMARKS, OOD_BENCHMARKS, PublicTaskView
from .integrity_communication_report import communication_summary
from .integrity_metric_schema import aggregate_secondary_metrics, require_expected_verifier
from .integrity_results import (
    ExecutionControls,
    NativeScore,
    compare_paired,
    exact_panel_join,
    native_mean,
    native_score_diagnostics,
    require_paired_controls,
    scienceworld_terminal_metrics,
)
from .integrity_resume import EvaluationRunMode, require_execution_mode
from .sealed_candidates import CandidateJournal, CandidateReader, EventOrigin, FinalCandidate
from .step0_integrity import (
    InferenceArm,
    InterventionCounts,
    SkillMode,
    validate_paired_intervention,
)
from .training_product_report import training_product_report
from .trajectory_diagnostics import trajectory_diagnostics


@dataclass(frozen=True, slots=True)
class FrozenPanel:
    entries: tuple[PublicTaskView, ...]
    exposure: str
    source_provenance: str
    sample_counts: tuple[tuple[str, int], ...] = ()
    catalog: tuple[str, ...] = IID_BENCHMARKS

    def validate(self, *, canary: bool) -> None:
        if self.catalog not in (IID_BENCHMARKS, OOD_BENCHMARKS):
            raise ValueError("evaluation requires an approved IID or OOD catalog")
        ids = [entry.task_id for entry in self.entries]
        if not ids or len(set(ids)) != len(ids) or not self.exposure or not self.source_provenance:
            raise ValueError("frozen panel identity or provenance is incomplete")
        for benchmark in self.catalog:
            count = sum(entry.benchmark == benchmark for entry in self.entries)
            expected = (
                dict(self.sample_counts).get(benchmark, 0)
                if self.sample_counts
                else 30
                if benchmark == "aime-2026"
                else 128
            )
            if count != expected and not canary:
                raise ValueError(f"declared panel requires {expected} {benchmark} records")
        if any(entry.benchmark not in self.catalog for entry in self.entries):
            raise ValueError("frozen panel includes a benchmark outside its declared catalog")


class PairedRuntime(Protocol):
    """Trusted source adapter; actor receives public entries, scorer a read-only candidate."""

    def controls(
        self, arm: InferenceArm, entries: tuple[PublicTaskView, ...]
    ) -> ExecutionControls: ...
    def interventions(self, arm: InferenceArm) -> InterventionCounts: ...
    def validate_candidate(
        self, reader: CandidateReader, entry: PublicTaskView, arm: InferenceArm, run_id: str
    ) -> None: ...
    async def generate(
        self, entry: PublicTaskView, arm: InferenceArm, run_id: str
    ) -> FinalCandidate: ...
    async def score(
        self, reader: CandidateReader, scope: tuple[str, str, str], benchmark: str
    ) -> NativeScore: ...
    async def refresh(self) -> None: ...
    async def close(self) -> None: ...


def scorer_cost_summary(
    reader: CandidateReader,
    scores: list[NativeScore],
    scopes: tuple[tuple[str, str, str], ...],
) -> dict[str, object]:
    """Include earlier failed grading invocations after a scoring-only resume."""
    selected = set(scopes)
    # One pass, not a full trace-table scan for each of the 926 candidates.
    events = (
        json.loads(payload)
        for run_id, arm_id, episode_id, payload in reader.connection.execute(
            "SELECT run_id, arm_id, episode_id, payload FROM trace "
            "WHERE stage='scorer-failure' AND origin=?",
            (EventOrigin.SCORER.value,),
        )
        if (run_id, arm_id, episode_id) in selected
    )
    failed = [
        cast(dict[str, float], event["scorer_cost"])
        for event in events
        if isinstance(event, dict) and isinstance(event.get("scorer_cost"), dict)
    ]
    completed = [score.scorer_cost for score in scores]

    def totals(costs: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: sum(cost.get(key, 0.0) for cost in costs)
            for key in sorted({key for cost in costs for key in cost})
        }

    return {
        "records_with_cost": sum(bool(cost) for cost in completed),
        "records_without_cost": sum(not cost for cost in completed),
        "failed_invocations": len(failed),
        "failed_invocation_totals": totals(failed),
        "completed_score_totals": totals(completed),
        "totals": totals([*completed, *failed]),
    }


async def run_paired(
    runtime: PairedRuntime,
    panel: FrozenPanel,
    arms: tuple[InferenceArm, ...],
    *,
    run_id: str,
    directory: Path,
    concurrency: int = 4,
    scoring_concurrency: int = 8,
    canary: bool = False,
    minimum_records_per_hour: float = 0.0,
    reuse_completed_from: Path | None = None,
    run_mode: EvaluationRunMode = EvaluationRunMode.FORMAL_FRESH,
) -> dict[str, object]:
    try:
        require_execution_mode(
            run_mode, directory=directory, importing_other_run=reuse_completed_from is not None
        )
    except (ValueError, TypeError):
        await runtime.close()
        raise
    panel.validate(canary=canary)
    if (
        min(concurrency, scoring_concurrency) < 1
        or len(arms) not in {1, 2}
        or len({arm.arm_id for arm in arms}) != len(arms)
    ):
        raise ValueError("execution needs positive concurrency and one or two distinct arms")
    if len(arms) == 2:
        validate_paired_intervention(arms[0], arms[1])
    elif arms[0].legacy:
        raise ValueError("single-arm evaluation must use the direct evaluated policy")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    frozen = {
        "run_id": run_id,
        "panel": asdict(panel),
        "arms": [arm.to_value() for arm in arms],
        "canary": canary,
        "schema": "step0-integrity-single@2" if len(arms) == 1 else "step0-integrity-paired@2",
    }
    trained = any(arm.optimizer_steps for arm in arms)
    if trained:
        frozen["schema"] = (
            "policy-integrity-single@1" if len(arms) == 1 else "policy-integrity-paired@1"
        )
    plan_path = directory / "frozen-plan-private.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("cannot resume with changed public input or evaluation arm")
    else:
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(frozen, stream, ensure_ascii=False)
    journal = CandidateJournal(directory / "candidates-private.sqlite")
    started, completed, newly_generated = time.monotonic(), 0, 0
    total = len(panel.entries) * len(arms)
    controls = {arm.arm_id: runtime.controls(arm, panel.entries) for arm in arms}

    def expected_verifier(arm_id: str, benchmark: str) -> str:
        versions = controls[arm_id].evaluator.get("verifier_versions")
        if not isinstance(versions, dict) or not isinstance(versions.get(benchmark), str):
            raise ValueError("frozen evaluator controls have no expected native verifier")
        return require_expected_verifier(versions[benchmark])

    counts = {arm.arm_id: InterventionCounts() for arm in arms}
    generation_error: BaseException | None = None

    async def generate_one(entry: PublicTaskView, arm: InferenceArm) -> tuple[FinalCandidate, bool]:
        scope = (run_id, arm.arm_id, entry.task_id)
        try:
            final = journal.get(*scope)
        except KeyError:
            final = await runtime.generate(entry, arm, run_id)
            if (final.run_id, final.arm_id, final.episode_id) != scope:
                raise ValueError("actor final belongs to another episode") from None
            try:
                committed = journal.get(*scope)
            except KeyError:
                journal.seal(final)
            else:
                if committed != final:
                    raise ValueError(
                        "runtime returned a different final than it committed"
                    ) from None
            return final, True
        runtime.validate_candidate(journal, entry, arm, run_id)
        return final, False

    def progress(*, failure: BaseException | None = None) -> None:
        rate = 3600 * newly_generated / max(time.monotonic() - started, 0.001)
        value = {
            "stage": "generation",
            "completed": completed,
            "planned": total,
            "newly_generated": newly_generated,
            "resumed": completed - newly_generated,
            "records_per_hour": rate,
            "eta_seconds": (total - completed) * 3600 / rate
            if rate and generation_error is None
            else None,
            "status": "generation-failure-draining"
            if generation_error is not None
            else "needs-human-decision"
            if minimum_records_per_hour and newly_generated >= 8 and rate < minimum_records_per_hour
            else "running",
        }
        if failure is not None:
            value["failure_type"] = type(failure).__name__
        print(json.dumps(value), flush=True)

    try:
        if any(
            value.service.get("coordinator_concurrency", concurrency) != concurrency
            for value in controls.values()
        ):
            raise ValueError("actual coordinator concurrency differs from its declared controls")
        journal.freeze_run(
            run_id, {**frozen, "controls": {key: asdict(value) for key, value in controls.items()}}
        )
        for arm in arms:
            for benchmark in {entry.benchmark for entry in panel.entries}:
                expected_verifier(arm.arm_id, benchmark)
        if len(arms) == 2:
            require_paired_controls(
                controls[arms[0].arm_id], controls[arms[1].arm_id], arms[0], arms[1]
            )
        remaining = iter((entry, arm) for entry in panel.entries for arm in arms)
        jobs: set[asyncio.Task[tuple[FinalCandidate, bool]]] = set()
        try:
            while True:
                while generation_error is None and len(jobs) < concurrency:
                    item = next(remaining, None)
                    if item is None:
                        break
                    jobs.add(asyncio.create_task(generate_one(*item)))
                if not jobs:
                    break
                finished, jobs = await asyncio.wait(jobs, return_when=asyncio.FIRST_COMPLETED)
                # Inspect failures before admitting more work. Do not cancel
                # unrelated active actors: their already-paid-for answers must
                # survive, and failure must be visible throughout HTTP cleanup.
                for job in finished:
                    error = asyncio.CancelledError() if job.cancelled() else job.exception()
                    if error is not None:
                        if generation_error is None:
                            generation_error = error
                        progress(failure=error)
                for job in finished:
                    if job.cancelled() or job.exception() is not None:
                        continue
                    final, generated = job.result()
                    newly_generated += int(generated)
                    completed += 1
                    for name, value in final.intervention_counts.items():
                        total_counts = counts[final.arm_id]
                        setattr(total_counts, name, getattr(total_counts, name) + value)
                    progress()
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
        if generation_error is not None:
            raise generation_error
        await runtime.refresh()
        for arm in arms:
            if controls[arm.arm_id] != runtime.controls(arm, panel.entries):
                raise RuntimeError("actual runtime controls changed during generation")
        reader = CandidateReader(directory / "candidates-private.sqlite")
        scores: dict[str, list[NativeScore]] = {arm.arm_id: [] for arm in arms}
        scoring_started, scored, newly_scored = time.monotonic(), 0, 0
        scoring_error: BaseException | None = None

        async def score_one(
            entry: PublicTaskView, arm: InferenceArm
        ) -> tuple[str, NativeScore, bool]:
            scope = (run_id, arm.arm_id, entry.task_id)
            runtime.validate_candidate(reader, entry, arm, run_id)
            stored = journal.stored_score(scope)
            score = (
                NativeScore.from_value(stored)
                if stored is not None
                else await runtime.score(reader, scope, entry.benchmark)
            )
            if score.task_id != entry.task_id or score.benchmark != entry.benchmark:
                raise ValueError("native score belongs to another panel entry")
            if stored is None:
                # SQLite access stays on this event-loop thread. Persist each
                # completed score even if another active grader has failed.
                journal.save_score(scope, asdict(score))
            return arm.arm_id, score, stored is None

        remaining = iter((entry, arm) for entry in panel.entries for arm in arms)
        scoring_jobs: set[asyncio.Task[tuple[str, NativeScore, bool]]] = set()
        try:
            while True:
                while scoring_error is None and len(scoring_jobs) < scoring_concurrency:
                    item = next(remaining, None)
                    if item is None:
                        break
                    scoring_jobs.add(asyncio.create_task(score_one(*item)))
                if not scoring_jobs:
                    break
                graded, scoring_jobs = await asyncio.wait(
                    scoring_jobs, return_when=asyncio.FIRST_COMPLETED
                )
                for scoring_job in graded:
                    error = (
                        asyncio.CancelledError()
                        if scoring_job.cancelled()
                        else scoring_job.exception()
                    )
                    if error is not None and scoring_error is None:
                        scoring_error = error
                for scoring_job in graded:
                    if scoring_job.cancelled() or scoring_job.exception() is not None:
                        continue
                    arm_id, score, generated_score = scoring_job.result()
                    scores[arm_id].append(score)
                    scored += 1
                    newly_scored += int(generated_score)
                    rate = 3600 * newly_scored / max(time.monotonic() - scoring_started, 0.001)
                    print(
                        json.dumps(
                            {
                                "stage": "scoring",
                                "completed": scored,
                                "planned": total,
                                "newly_scored": newly_scored,
                                "resumed": scored - newly_scored,
                                "records_per_hour": rate,
                                "eta_seconds": (total - scored) * 3600 / rate
                                if rate and scoring_error is None
                                else None,
                                "status": "scoring-failure-draining"
                                if scoring_error is not None
                                else "needs-human-decision"
                                if minimum_records_per_hour
                                and newly_scored >= 8
                                and rate < minimum_records_per_hour
                                else "running",
                            }
                        ),
                        flush=True,
                    )
                if scoring_error is not None:
                    print(
                        json.dumps(
                            {
                                "stage": "scoring",
                                "completed": scored,
                                "planned": total,
                                "status": "scoring-failure-draining",
                                "failure_type": type(scoring_error).__name__,
                                "eta_seconds": None,
                            }
                        ),
                        flush=True,
                    )
        finally:
            for scoring_job in scoring_jobs:
                if not scoring_job.done():
                    scoring_job.cancel()
            await asyncio.gather(*scoring_jobs, return_exceptions=True)
            reader.close()
        if scoring_error is not None:
            raise scoring_error
        panel_order = {entry.task_id: index for index, entry in enumerate(panel.entries)}
        for native_scores in scores.values():
            native_scores.sort(key=lambda score: panel_order[score.task_id])
        comparisons: dict[str, dict[str, object]] = {}
        for benchmark in panel.catalog:
            entries = tuple(entry for entry in panel.entries if entry.benchmark == benchmark)
            if not entries:
                continue
            if len(arms) == 1:
                counts[arms[0].arm_id].require_arm(arms[0])
                native = exact_panel_join(
                    tuple(entry.task_id for entry in entries),
                    tuple(
                        score for score in scores[arms[0].arm_id] if score.benchmark == benchmark
                    ),
                    expected_verifier=expected_verifier(arms[0].arm_id, benchmark),
                )
                comparisons[benchmark] = {
                    "count": len(native),
                    "arm_id": arms[0].arm_id,
                    "metric": native[0].metric,
                    "value": native_mean(native),
                    "secondary_metrics": aggregate_secondary_metrics(native),
                    "integrity_status": "pass",
                    "execution_status": "complete",
                    "performance_goal_status": "not-assessed-without-a-control-arm",
                    "external_reference_status": "not-a-matched-control",
                }
            else:
                comparisons[benchmark] = compare_paired(
                    tuple(entry.task_id for entry in entries),
                    tuple(
                        score for score in scores[arms[0].arm_id] if score.benchmark == benchmark
                    ),
                    tuple(
                        score for score in scores[arms[1].arm_id] if score.benchmark == benchmark
                    ),
                    left_controls=runtime.controls(arms[0], entries),
                    right_controls=runtime.controls(arms[1], entries),
                    left_arm=arms[0],
                    right_arm=arms[1],
                    left_counts=counts[arms[0].arm_id],
                    right_counts=counts[arms[1].arm_id],
                    left_expected_verifier=expected_verifier(arms[0].arm_id, benchmark),
                    right_expected_verifier=expected_verifier(arms[1].arm_id, benchmark),
                )
            comparisons[benchmark]["native_diagnostics"] = {
                arm.arm_id: native_score_diagnostics(
                    tuple(row for row in scores[arm.arm_id] if row.benchmark == benchmark),
                    controls[arm.arm_id].environment,
                )
                for arm in arms
            }
            communication = {
                arm.arm_id: communication_summary(
                    journal,
                    tuple((run_id, arm.arm_id, entry.task_id) for entry in entries),
                    benchmark_by_scope={
                        (run_id, arm.arm_id, entry.task_id): benchmark for entry in entries
                    },
                )
                for arm in arms
            }
            comparisons[benchmark]["communication"] = communication
            comparisons[benchmark]["trajectory_diagnostics"] = {
                arm.arm_id: trajectory_diagnostics(
                    journal,
                    tuple((run_id, arm.arm_id, entry.task_id) for entry in entries),
                    benchmark=benchmark,
                )
                for arm in arms
            }
            if benchmark == "scienceworld":
                comparisons[benchmark]["native_terminal_metrics"] = {
                    arm.arm_id: scienceworld_terminal_metrics(
                        [
                            cast(dict[str, object], rows[0]) if len(rows) == 1 else {}
                            for entry in entries
                            for rows in [
                                journal.traces(
                                    (run_id, arm.arm_id, entry.task_id),
                                    "native-outcome",
                                    origin=EventOrigin.ENVIRONMENT,
                                )
                            ]
                        ]
                    )
                    for arm in arms
                }
            statuses = {item["status"] for item in communication.values()}
            state_order = (
                "incomplete-evidence",
                "transport-defect",
                "unresolved-delivery-failure",
                "unresolved-output-failure",
                "not-observed",
                "complete-with-repairs",
                "complete",
            )
            comparisons[benchmark]["communication_status"] = next(
                state for state in state_order if state in statuses
            )
            comparisons[benchmark]["native_metric_contract_status"] = "pass"
            if not statuses <= {"complete", "complete-with-repairs"}:
                comparisons[benchmark]["integrity_status"] = "communication-not-complete"
        report = {
            "evaluation_isolation": {
                arm.arm_id: controls[arm.arm_id].parser.get("evaluation_isolation") for arm in arms
            },
            "architecture_id": controls[arms[0].arm_id].parser.get("architecture_id"),
            "training_product_axes": {
                arm.arm_id: training_product_report(arm, controls[arm.arm_id]) for arm in arms
            },
            "schema": "step0-integrity-single-summary@3"
            if len(arms) == 1
            else "step0-integrity-paired-summary@3",
            "canary": canary,
            "run_id": run_id,
            "population_exposure": panel.exposure,
            "count_per_arm": len(panel.entries),
            "planned_per_arm": len(panel.entries),
            "scored_per_arm": {arm.arm_id: len(scores[arm.arm_id]) for arm in arms},
            "missing_per_arm": {arm.arm_id: 0 for arm in arms},
            "benchmarks" if len(arms) == 1 else "comparisons": comparisons,
            "candidate_authority": "evaluated-policy-single-final-no-selection",
            "scoring_concurrency": scoring_concurrency,
            "implementation_revision": controls[arms[0].arm_id].parser.get(
                "implementation_revision", "synthetic-test-runtime"
            ),
            "interventions": {arm.arm_id: asdict(counts[arm.arm_id]) for arm in arms},
            "skill_usage": {
                arm.arm_id: {
                    "allowed": arm.skill_mode is not SkillMode.OFF,
                    "library_id": controls[arm.arm_id].skills.get("library_id"),
                    "source_kind": controls[arm.arm_id].skills.get("kind", "none"),
                    "retrieval_matches": counts[arm.arm_id].skill_retrieved,
                    "body_exposures": counts[arm.arm_id].skill_blocks_injected,
                    "explicit_invocations": counts[arm.arm_id].skill_calls,
                    "no_match_requests": counts[arm.arm_id].skill_no_match,
                    "skill_budget_skips": counts[arm.arm_id].skill_budget_skips,
                    "context_omissions": counts[arm.arm_id].skill_context_omissions,
                }
                for arm in arms
            },
            "arms": [arm.to_value() for arm in arms],
            "effective_benchmark_configuration": {
                arm.arm_id: controls[arm.arm_id].parser.get("effective_benchmark_configuration", {})
                for arm in arms
            },
            "actual_cost": {
                arm.arm_id: {
                    "input_tokens": sum(
                        journal.get(run_id, arm.arm_id, entry.task_id).prompt_tokens
                        for entry in panel.entries
                    ),
                    "output_tokens": sum(
                        journal.get(run_id, arm.arm_id, entry.task_id).completion_tokens
                        for entry in panel.entries
                    ),
                    "model_calls": counts[arm.arm_id].model_calls,
                    "tool_calls": counts[arm.arm_id].tool_calls,
                    "peer_model_calls": counts[arm.arm_id].peer_model_calls,
                }
                for arm in arms
            },
            "scorer_cost": {
                arm.arm_id: scorer_cost_summary(
                    journal,
                    scores[arm.arm_id],
                    tuple((run_id, arm.arm_id, entry.task_id) for entry in panel.entries),
                )
                for arm in arms
            },
        }
        if trained:
            report["schema"] = (
                "policy-integrity-single-summary@1"
                if len(arms) == 1
                else "policy-integrity-paired-summary@1"
            )
        (directory / "summary.json").write_text(json.dumps(report, default=asdict, indent=2) + "\n")
        return report
    finally:
        journal.close()
        await runtime.close()
