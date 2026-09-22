"""Scorer-side adapters read the submitted candidate, never invoke an actor."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import JsonValue, SuccessRule, TerminalReward
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.sealed_candidates import CandidateReader
from skillev.rollout import TerminalEvaluatorError
from skillev_private.benchmarks.protocol_v10_official import HealthBenchOfficialGrader
from skillev_private.benchmarks.qa_diagnostics import qa_answer_diagnostics
from skillev_private.benchmarks.qa_metrics import (
    best_alias_metrics,
    normalize_triviaqa_answer,
    score_hotpotqa_answers,
)


@dataclass(frozen=True, slots=True)
class SealedQAScorer:
    reader: CandidateReader
    benchmark: str
    accepted_answers: tuple[str, ...]
    environment_id: str

    async def score(self, run_id: str, arm_id: str, episode_id: str) -> TerminalReward:
        candidate = self.reader.get(run_id, arm_id, episode_id)
        if self.benchmark == "hotpotqa":
            metrics = score_hotpotqa_answers(candidate.text, self.accepted_answers)
        elif self.benchmark == "triviaqa":
            metrics = best_alias_metrics(
                candidate.text, self.accepted_answers, normalize=normalize_triviaqa_answer
            )
        else:
            raise ValueError("QA scorer supports only the two IID QA contracts")
        em, f1 = (metrics.em, metrics.f1) if candidate.text.strip() else (0.0, 0.0)
        payload: dict[str, JsonValue] = {
            "benchmark_id": self.benchmark,
            "qa_diagnostics": qa_answer_diagnostics(
                self.benchmark,
                original_submission=None
                if candidate.submission is None
                else candidate.submission["raw_response"],
                projected_answer=candidate.text,
                accepted_aliases=self.accepted_answers,
            ),
            "public_metrics": {"answer-exact-match": em, "answer-f1": f1},
        }
        return TerminalReward(
            value=f1,
            success=bool(em),
            success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
            success_threshold=None,
            native_metric_name="answer-f1",
            native_payload=payload,
            environment_id=self.environment_id,
            verifier_version=NATIVE_VERIFIER_VERSIONS[self.benchmark],
        )


@dataclass(frozen=True, slots=True)
class SealedHealthScorer:
    reader: CandidateReader
    grader: HealthBenchOfficialGrader
    environment_id: str

    async def score(self, run_id: str, arm_id: str, episode_id: str) -> TerminalReward:
        candidate = self.reader.get(run_id, arm_id, episode_id)
        # No run/arm identity is forwarded to the arm-blind rubric judge.
        # Empty/invalid candidates remain zero; grader outages remain incomplete.
        raw, negative = 0.0, 0
        if candidate.text.strip():
            try:
                grade = await self.grader.grade(episode_id, candidate.text)
            except Exception as error:
                raise TerminalEvaluatorError("HealthBench judge infrastructure failed") from error
            raw = grade.official_rubric_score
            negative = grade.triggered_negative_rubric_count
        payload: dict[str, JsonValue] = {
            "benchmark_id": "healthbench",
            "public_metrics": {"qwen-local-rubric-score": raw},
            "triggered-negative-rubric-count": negative,
            "native_raw_score": raw if candidate.text.strip() else None,
            "learning_reward": min(1.0, max(0.0, raw)),
            "binary_success": raw >= 0.60 and negative == 0,
            "negative_criterion_count": negative if candidate.text.strip() else None,
            "criterion_ledger": None if not candidate.text.strip() else grade.criterion_ledger,
            "grader-used": bool(candidate.text.strip()),
        }
        return TerminalReward(
            value=max(0.0, min(1.0, raw)),
            success=raw >= 0.60 and negative == 0,
            success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
            success_threshold=None,
            native_metric_name="qwen-local-rubric-score",
            native_payload=payload,
            environment_id=self.environment_id,
            verifier_version=self.grader.verifier_version,
        )
