"""Private HealthBench rubric grading with a frozen Qwen SGLang base model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from skillev.contracts import JsonValue, normalize_json
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.training import AsyncResourceLimiter

from .protocol_v10_official import HealthBenchGrade
from .protocol_v10_population import PrivateBenchmarkPopulation
from .protocol_v10_sessions import ProtocolV10PrivateRecord
from .protocol_v10_workers import (
    _HEALTHBENCH_WORKER,
    PrivateJSONWorker,
    ProtocolV10WorkerError,
)
from .trusted_model_request_broker import TrustedModelRequestBroker

HEALTHBENCH_QWEN_VERIFIER = "healthbench-qwen3.5-9b-sglang-temperature-0@1"


@dataclass(frozen=True, slots=True)
class HealthBenchQwenVerifierIdentity:
    model_revision: str
    tokenizer_revision: str
    sglang_version: str
    prompt_template_version: str = "openai-simple-evals-healthbench-rubric@1"
    parser_version: str = "strict-json-one-repair@1"
    scoring_version: str = "openai-simple-evals-healthbench-score@1"

    def __post_init__(self) -> None:
        values = (
            self.model_revision,
            self.tokenizer_revision,
            self.sglang_version,
            self.prompt_template_version,
            self.parser_version,
            self.scoring_version,
        )
        if any(type(value) is not str or not value.strip() for value in values):
            raise ValueError("HealthBench Qwen verifier identity is incomplete")


@dataclass(frozen=True, slots=True)
class HealthBenchQwenGraderConfig:
    endpoint_base: str
    base_model: str
    identity: HealthBenchQwenVerifierIdentity
    request_timeout_seconds: float = 90.0
    worker_timeout_seconds: float = 900.0
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_output_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.endpoint_base.startswith(("http://", "https://")):
            raise ValueError("HealthBench Qwen endpoint must be HTTP(S)")
        if not self.base_model.strip():
            raise ValueError("HealthBench Qwen base-model route is absent")
        if self.base_model != self.identity.model_revision:
            raise ValueError("HealthBench Qwen route and frozen model revision differ")
        if (
            self.temperature != 0.0
            or self.top_p != 1.0
            or self.seed != 0
            or self.max_output_tokens != 1024
        ):
            raise ValueError("HealthBench Qwen decoding differs from the frozen profile")
        if self.request_timeout_seconds <= 0:
            raise ValueError("HealthBench Qwen request timeout must be positive")
        if self.worker_timeout_seconds <= self.request_timeout_seconds:
            raise ValueError("HealthBench Qwen worker timeout must exceed request timeout")

    @property
    def openai_base(self) -> str:
        return self.endpoint_base.rstrip("/").removesuffix("/v1") + "/v1"


@dataclass(frozen=True, slots=True)
class QwenSGLangHealthBenchGrader:
    """Keep rubrics in a subprocess and reserve both global and grader capacity."""

    worker: PrivateJSONWorker
    private_cases: Mapping[str, JsonValue]
    model_request_limiter: AsyncResourceLimiter
    verifier_version: str = HEALTHBENCH_QWEN_VERIFIER
    broker_config: HealthBenchQwenGraderConfig | None = None
    request_journal: DurableRequestJournal | None = None

    def __post_init__(self) -> None:
        cases = normalize_json(dict(self.private_cases))
        if not isinstance(cases, dict) or not cases:
            raise ValueError("HealthBench private cases are unavailable")
        object.__setattr__(self, "private_cases", cases)

    async def grade(self, task_id: str, candidate_answer: str) -> HealthBenchGrade:
        try:
            private_case = self.private_cases[task_id]
        except KeyError as error:
            raise ProtocolV10WorkerError("HealthBench task is not privately routed") from error
        payload: dict[str, JsonValue] = {
            "candidate_answer": candidate_answer,
            "operation": "grade",
            "private_case": private_case,
            "task_id": task_id,
        }
        if self.broker_config is None:
            async with self.model_request_limiter.lease():
                result = await self.worker.request(payload)
        else:
            config = self.broker_config
            async with TrustedModelRequestBroker(
                endpoint=config.openai_base,
                model=config.base_model,
                limiter=self.model_request_limiter,
                timeout=config.request_timeout_seconds,
                request_journal=self.request_journal,
                task_id=task_id,
            ) as broker:
                worker = replace(
                    self.worker,
                    command=(*self.worker.command, "--broker-socket", broker.socket_path),
                )
                result = await worker.request(payload)
        if set(result) != {"official_rubric_score", "triggered_negative_rubric_count"}:
            raise ProtocolV10WorkerError("HealthBench worker response fields differ")
        score = result["official_rubric_score"]
        negative = result["triggered_negative_rubric_count"]
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or type(negative) is not int
        ):
            raise ProtocolV10WorkerError("HealthBench worker response types differ")
        return HealthBenchGrade(float(score), negative)


@dataclass(frozen=True, slots=True)
class QwenSGLangHealthBenchDeployment:
    interpreter_path: Path
    official_source_root: Path
    config: HealthBenchQwenGraderConfig

    def __post_init__(self) -> None:
        if not self.interpreter_path.is_absolute() or not self.interpreter_path.is_file():
            raise ValueError("HealthBench interpreter must be an absolute file")
        if not self.official_source_root.is_absolute() or not self.official_source_root.is_dir():
            raise ValueError("HealthBench simple-evals root must be an absolute directory")

    def build(
        self,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
        *,
        model_request_limiter: AsyncResourceLimiter,
        health_grader_limiter: AsyncResourceLimiter,
    ) -> QwenSGLangHealthBenchGrader:
        by_source = {record.source_id: record.private_payload for record in records}
        if set(by_source) != {item.source_id for item in population.items}:
            raise ValueError("HealthBench public and private routes differ")
        command = (
            str(self.interpreter_path),
            str(_HEALTHBENCH_WORKER),
            "--official-source-root",
            str(self.official_source_root),
            "--grader-model",
            self.config.base_model,
            "--api-base-url",
            self.config.openai_base,
            "--request-timeout-seconds",
            str(self.config.request_timeout_seconds),
        )
        return QwenSGLangHealthBenchGrader(
            worker=PrivateJSONWorker(
                command=command,
                working_directory=self.official_source_root.parent,
                timeout_seconds=self.config.worker_timeout_seconds,
                process_limiter=health_grader_limiter,
            ),
            private_cases=by_source,
            broker_config=self.config,
            model_request_limiter=model_request_limiter,
        )


__all__ = [
    "HEALTHBENCH_QWEN_VERIFIER",
    "HealthBenchQwenGraderConfig",
    "HealthBenchQwenVerifierIdentity",
    "QwenSGLangHealthBenchDeployment",
    "QwenSGLangHealthBenchGrader",
]
