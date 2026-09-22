"""Official-parity HealthBench Full diagnostics for a local Qwen backbone.

The benchmark data and per-example artifacts are deliberately supplied at
runtime and written outside Git.  This module changes only the candidate model
transport: sampling, prompts, grading, and aggregation remain owned by the
pinned OpenAI simple-evals implementation.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import yaml

from .external_judge_policy import (
    DIRECT_HEALTHBENCH_PROFILE,
    DIRECT_JUDGE_MODEL,
    EXTERNAL_JUDGE_MODEL,
    GATEWAY_HEALTHBENCH_PROFILE,
    HEALTHBENCH_EXTERNAL_PROFILE,
)

# Re-export the existing import surface for diagnostic callers.
from .healthbench_transport import (
    CANDIDATE_MAX_TOKENS as CANDIDATE_MAX_TOKENS,
)
from .healthbench_transport import (
    CANDIDATE_TEMPERATURE as CANDIDATE_TEMPERATURE,
)
from .healthbench_transport import (
    ExternalJudgeTransport as ExternalJudgeTransport,
)
from .healthbench_transport import (
    HealthBenchTransportError as HealthBenchTransportError,
)
from .healthbench_transport import (
    HealthBenchTransportExhausted as HealthBenchTransportExhausted,
)
from .healthbench_transport import (
    QwenChatTransport as QwenChatTransport,
)
from .healthbench_transport import (
    TransportRetryPolicy as TransportRetryPolicy,
)
from .healthbench_transport import _Client

OPENAI_SYSTEM_MESSAGE = "You are a helpful assistant."
GRADER_MODEL = "gpt-4.1-2025-04-14"
SAMPLE_COUNT = 128

_HEALTHBENCH_GRADER_INVOCATION: ContextVar[str | None] = ContextVar(
    "healthbench_grader_invocation", default=None
)


class HealthBenchFailureKind(StrEnum):
    CANDIDATE_GENERATION_INFRASTRUCTURE = "candidate-generation-infrastructure"
    GRADER_TRANSPORT_INFRASTRUCTURE = "grader-transport-infrastructure"
    GRADER_RESPONSE_MALFORMED = "grader-response-malformed"
    LOCAL_IMPLEMENTATION_ERROR = "local-implementation-error"


@dataclass(frozen=True, slots=True)
class HealthBenchExternalJudgeProfile:
    profile_id: str
    backend: str
    model: str
    rubric_source_repository: str
    rubric_source_revision: str
    rubric_source_path: str
    endpoint_environment: str
    api_key_environment: str
    call_mode: str
    response_format: str
    max_completion_tokens: int
    reasoning_effort: str
    temperature: None
    top_p: None
    request_timeout_seconds: float
    maximum_attempts: int

    def __post_init__(self) -> None:
        required = (
            self.profile_id,
            self.model,
            self.rubric_source_repository,
            self.rubric_source_revision,
            self.rubric_source_path,
            self.endpoint_environment,
            self.api_key_environment,
        )
        if any(not value.strip() for value in required):
            raise ValueError("external HealthBench judge identity is incomplete")
        gateway = self.profile_id in {HEALTHBENCH_EXTERNAL_PROFILE, GATEWAY_HEALTHBENCH_PROFILE}
        declared = gateway or self.profile_id == DIRECT_HEALTHBENCH_PROFILE
        if (
            self.backend != "openai-chat-completions"
            or self.call_mode != "per-rubric"
            or self.response_format != "json-object"
            or self.max_completion_tokens != (8000 if declared else 4096)
            or self.reasoning_effort != ("medium" if declared else "low")
            or (
                declared and self.model != (EXTERNAL_JUDGE_MODEL if gateway else DIRECT_JUDGE_MODEL)
            )
            or (gateway and self.maximum_attempts != 1)
            or self.temperature is not None
            or self.top_p is not None
        ):
            raise ValueError("external HealthBench judge differs from its declared profile")
        if self.request_timeout_seconds <= 0 or self.maximum_attempts < 1:
            raise ValueError("external HealthBench judge retry policy is invalid")


@dataclass(frozen=True, slots=True)
class HealthBenchQwenJudgeProfile:
    profile_id: str
    backend: str
    model_route: str
    rubric_source_repository: str
    rubric_source_revision: str
    rubric_source_path: str
    call_mode: str
    system_message: str
    response_format: str
    max_tokens: int
    temperature: float
    top_p: None
    top_k: None
    seed: None
    enable_thinking: bool
    request_timeout_seconds: float
    maximum_attempts: int
    total_deadline_seconds: float
    semantic_repair_attempts: int
    official_gpt_comparable: bool

    def __post_init__(self) -> None:
        required = (
            self.profile_id,
            self.model_route,
            self.rubric_source_repository,
            self.rubric_source_revision,
            self.rubric_source_path,
            self.system_message,
        )
        if any(not value.strip() for value in required):
            raise ValueError("Qwen HealthBench judge identity is incomplete")
        if (
            self.backend != "qwen-sglang"
            or self.call_mode != "per-rubric"
            or self.system_message != OPENAI_SYSTEM_MESSAGE
            or self.response_format != "plain-json-then-schema-once"
            or self.max_tokens != CANDIDATE_MAX_TOKENS
            or self.temperature != CANDIDATE_TEMPERATURE
            or self.top_p is not None
            or self.top_k is not None
            or self.seed is not None
            or self.enable_thinking
            or self.semantic_repair_attempts != 1
            or self.official_gpt_comparable
            or any(marker in self.model_route.lower() for marker in ("adapter", "lora", "skill"))
        ):
            raise ValueError("Qwen HealthBench judge protocol differs from the owner contract")
        if (
            self.request_timeout_seconds <= 0
            or self.maximum_attempts < 1
            or self.total_deadline_seconds <= self.request_timeout_seconds
        ):
            raise ValueError("Qwen HealthBench judge retry policy is invalid")


def load_external_judge_profile(path: Path) -> HealthBenchExternalJudgeProfile:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError("external HealthBench judge config must be a mapping")
    expected = {
        "format",
        "profile_id",
        "backend",
        "model",
        "rubric_source_repository",
        "rubric_source_revision",
        "rubric_source_path",
        "endpoint_environment",
        "api_key_environment",
        "call_mode",
        "response_format",
        "max_completion_tokens",
        "reasoning_effort",
        "temperature",
        "top_p",
        "request_timeout_seconds",
        "maximum_attempts",
    }
    if set(value) != expected or value["format"] != ("skillev-protocol14-healthbench-grader@1"):
        raise ValueError("external HealthBench judge config fields differ")
    _required_none(value["temperature"], "temperature")
    _required_none(value["top_p"], "top p")
    return HealthBenchExternalJudgeProfile(
        profile_id=_required_text(value["profile_id"], "profile ID"),
        backend=_required_text(value["backend"], "backend"),
        model=_required_text(value["model"], "model"),
        rubric_source_repository=_required_text(
            value["rubric_source_repository"], "rubric source repository"
        ),
        rubric_source_revision=_required_text(
            value["rubric_source_revision"], "rubric source revision"
        ),
        rubric_source_path=_required_text(value["rubric_source_path"], "rubric source path"),
        endpoint_environment=_required_text(value["endpoint_environment"], "endpoint environment"),
        api_key_environment=_required_text(value["api_key_environment"], "API key environment"),
        call_mode=_required_text(value["call_mode"], "call mode"),
        response_format=_required_text(value["response_format"], "response format"),
        max_completion_tokens=_required_integer(
            value["max_completion_tokens"], "max completion tokens"
        ),
        reasoning_effort=_required_text(value["reasoning_effort"], "reasoning effort"),
        temperature=None,
        top_p=None,
        request_timeout_seconds=_required_number(
            value["request_timeout_seconds"], "request timeout"
        ),
        maximum_attempts=_required_integer(value["maximum_attempts"], "maximum attempts"),
    )


def load_qwen_judge_profile(path: Path) -> HealthBenchQwenJudgeProfile:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError("Qwen HealthBench judge config must be a mapping")
    expected = {
        "format",
        "profile_id",
        "backend",
        "model_route",
        "rubric_source_repository",
        "rubric_source_revision",
        "rubric_source_path",
        "call_mode",
        "system_message",
        "response_format",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "enable_thinking",
        "request_timeout_seconds",
        "maximum_attempts",
        "total_deadline_seconds",
        "semantic_repair_attempts",
        "official_gpt_comparable",
    }
    if set(value) != expected or value["format"] != (
        "skillev-protocol14-healthbench-qwen-grader@1"
    ):
        raise ValueError("Qwen HealthBench judge config fields differ")
    _required_none(value["top_p"], "top p")
    _required_none(value["top_k"], "top k")
    _required_none(value["seed"], "seed")
    return HealthBenchQwenJudgeProfile(
        profile_id=_required_text(value["profile_id"], "profile ID"),
        backend=_required_text(value["backend"], "backend"),
        model_route=_required_text(value["model_route"], "model route"),
        rubric_source_repository=_required_text(
            value["rubric_source_repository"], "rubric source repository"
        ),
        rubric_source_revision=_required_text(
            value["rubric_source_revision"], "rubric source revision"
        ),
        rubric_source_path=_required_text(value["rubric_source_path"], "rubric source path"),
        call_mode=_required_text(value["call_mode"], "call mode"),
        system_message=_required_text(value["system_message"], "system message"),
        response_format=_required_text(value["response_format"], "response format"),
        max_tokens=_required_integer(value["max_tokens"], "max tokens"),
        temperature=_required_number(value["temperature"], "temperature"),
        top_p=None,
        top_k=None,
        seed=None,
        enable_thinking=_required_boolean(value["enable_thinking"], "thinking mode"),
        request_timeout_seconds=_required_number(
            value["request_timeout_seconds"], "request timeout"
        ),
        maximum_attempts=_required_integer(value["maximum_attempts"], "maximum attempts"),
        total_deadline_seconds=_required_number(value["total_deadline_seconds"], "total deadline"),
        semantic_repair_attempts=_required_integer(
            value["semantic_repair_attempts"], "semantic repair attempts"
        ),
        official_gpt_comparable=_required_boolean(
            value["official_gpt_comparable"], "official GPT comparability"
        ),
    )


@dataclass(slots=True)
class RubricAttemptRegistry:
    _attempts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def begin_semantic_call(
        self,
        invocation_id: str,
        messages: list[dict[str, str]],
        *,
        maximum_calls: int = 2,
    ) -> int:
        if not invocation_id.strip():
            raise ValueError("HealthBench semantic invocation ID must be non-empty")
        key = invocation_id + "\n" + json.dumps(messages, ensure_ascii=False, sort_keys=True)
        with self._lock:
            attempt = self._attempts.get(key, 0) + 1
            if attempt > maximum_calls:
                raise RuntimeError("HealthBench rubric exceeded its declared request allowance")
            self._attempts[key] = attempt
        return attempt

    @property
    def semantic_repair_count(self) -> int:
        with self._lock:
            return sum(max(0, value - 1) for value in self._attempts.values())


def _rubric_response_schema() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "healthbench_rubric_grade",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "criteria_met": {"type": "boolean"},
                },
                "required": ["criteria_met"],
                "additionalProperties": False,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    prompt_id: str
    response_text: str
    actual_queried_message_list: tuple[dict[str, str], ...]
    usage: Mapping[str, int | None]


class QwenOfficialCandidateSampler:
    """Qwen transport with the official GPT-4.1 sampler's public contract.

    The only additional request field disables Qwen's model-specific thinking
    channel.  All benchmark-level controls match ChatCompletionSampler:
    helpful-assistant system message, temperature 0.5, 2,048 output tokens,
    one direct completion, and exponential infrastructure retry.
    """

    def __init__(
        self,
        *,
        client: _Client,
        model: str,
        sampler_response_type: type[Any],
        retryable_errors: tuple[type[Exception], ...] = (
            OSError,
            TimeoutError,
        ),
        rubric_schema_on_retry: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
        retry_policy: TransportRetryPolicy | None = None,
        semantic_attempts: RubricAttemptRegistry | None = None,
        temperature: float = CANDIDATE_TEMPERATURE,
        max_tokens: int = CANDIDATE_MAX_TOKENS,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        enable_thinking: bool = False,
    ) -> None:
        if not model.strip():
            raise ValueError("candidate model route must be non-empty")
        if any(marker in model.lower() for marker in ("adapter", "lora", "skill")):
            raise ValueError("candidate route must be adapter-free")
        self.model = model
        self.sampler_response_type = sampler_response_type
        self.rubric_schema_on_retry = rubric_schema_on_retry
        self.semantic_attempts = semantic_attempts or RubricAttemptRegistry()
        self.transport = QwenChatTransport(
            client=client,
            model=model,
            retry_policy=retry_policy or TransportRetryPolicy(),
            transient_error_types=retryable_errors,
            sleeper=sleeper,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            enable_thinking=enable_thinking,
        )

    def __call__(self, message_list: Sequence[Mapping[str, str]]) -> Any:
        messages = [
            {"role": "system", "content": OPENAI_SYSTEM_MESSAGE},
            *({"role": item["role"], "content": item["content"]} for item in message_list),
        ]
        semantic_attempt = (
            self.semantic_attempts.begin_semantic_call("candidate", messages)
            if self.rubric_schema_on_retry
            else 1
        )
        response = self.transport.create(
            messages=messages,
            response_format=(
                _rubric_response_schema()
                if self.rubric_schema_on_retry and semantic_attempt > 1
                else None
            ),
        )
        content = response.choices[0].message.content
        # A successful transport with no assistant content is a model outcome,
        # not retryable infrastructure.  Preserve it as an empty candidate so
        # it remains in the fixed panel denominator.
        if content is None:
            content = ""
        return self.sampler_response_type(
            response_text=content,
            response_metadata={
                "usage": response.usage,
                "model": getattr(response, "model", None),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
            },
            actual_queried_message_list=messages,
        )


@dataclass(slots=True)
class QwenHealthBenchRubricSampler:
    transports: tuple[QwenChatTransport, ...]
    response_type: type[Any]
    semantic_attempts: RubricAttemptRegistry

    def __post_init__(self) -> None:
        if not self.transports:
            raise ValueError("at least one grader transport is required")

    def _select_transport(self, messages: list[dict[str, str]]) -> QwenChatTransport:
        """Keep a rubric and its semantic repair on one deterministic replica."""

        visible_characters = sum(len(item["role"]) + len(item["content"]) for item in messages)
        return self.transports[visible_characters % len(self.transports)]

    def __call__(self, message_list: Sequence[Mapping[str, str]]) -> Any:
        messages = [
            {"role": "system", "content": OPENAI_SYSTEM_MESSAGE},
            *({"role": item["role"], "content": item["content"]} for item in message_list),
        ]
        invocation_id = _HEALTHBENCH_GRADER_INVOCATION.get()
        if invocation_id is None:
            # simple-evals grades rubric items in its own thread pool, which
            # does not inherit the outer context variable.  The full grader
            # message is already part of the registry key, so this stable lane
            # identity still keeps semantic retries distinct per rubric item.
            invocation_id = "official-grader-worker"
        semantic_attempt = self.semantic_attempts.begin_semantic_call(invocation_id, messages)
        transport = self._select_transport(messages)
        response = transport.create(
            messages=messages,
            response_format=None if semantic_attempt == 1 else _rubric_response_schema(),
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("grader endpoint returned an empty response")
        return self.response_type(
            response_text=content,
            response_metadata={"usage": response.usage},
            actual_queried_message_list=messages,
        )


@dataclass(slots=True)
class ExternalHealthBenchRubricSampler:
    """simple-evals sampler contract, retaining the selected external judge identity."""

    transport: ExternalJudgeTransport
    response_type: type[Any]
    semantic_attempts: RubricAttemptRegistry = field(default_factory=RubricAttemptRegistry)
    profile_id: str | None = None

    def __call__(self, message_list: Sequence[Mapping[str, str]]) -> Any:
        messages = [{"role": item["role"], "content": item["content"]} for item in message_list]
        invocation_id = _HEALTHBENCH_GRADER_INVOCATION.get() or "official-grader-worker"
        self.semantic_attempts.begin_semantic_call(
            invocation_id,
            messages,
            maximum_calls=1
            if self.profile_id
            in {
                HEALTHBENCH_EXTERNAL_PROFILE,
                GATEWAY_HEALTHBENCH_PROFILE,
            }
            else 2,
        )
        response = self.transport.create(messages=messages)
        content = response.choices[0].message.content
        return self.response_type(
            response_text=content,
            response_metadata={
                "usage": response.usage,
                "judge_profile": self.profile_id,
                "judge_model": self.transport.model,
                "reasoning_effort": self.transport.reasoning_effort,
                "returned_model": getattr(response, "model", None),
                "provider": getattr(response, "judge_provider", None),
                "endpoint": getattr(response, "judge_endpoint", None),
                "provider_requested_model": getattr(response, "judge_requested_model", None),
                "failover_reason": getattr(response, "judge_failover_reason", None),
            },
            actual_queried_message_list=messages,
        )


@dataclass(frozen=True, slots=True)
class HealthBenchGraderReceipt:
    grader_backend: str
    grader_model: str
    grader_replicas: int
    unique_grader_models: int
    official_gpt4_1_comparable: bool
    semantic_repair_count: int
    transport_retry_count: int
    malformed_final_count: int
    rubric_item_count: int

    def __post_init__(self) -> None:
        if self.grader_replicas < 1 or self.unique_grader_models < 1:
            raise ValueError("grader identity counts must be positive")
        if self.unique_grader_models > self.grader_replicas:
            raise ValueError("unique grader models cannot exceed replicas")
        if (
            min(
                self.semantic_repair_count,
                self.transport_retry_count,
                self.malformed_final_count,
                self.rubric_item_count,
            )
            < 0
        ):
            raise ValueError("grader receipt counts cannot be negative")


class RoundRobinSampler:
    """Thread-safe request distribution across equivalent sampler replicas."""

    def __init__(self, samplers: Sequence[Callable[[Sequence[Mapping[str, str]]], Any]]) -> None:
        if not samplers:
            raise ValueError("at least one sampler replica is required")
        self.samplers = tuple(samplers)
        self._lock = threading.Lock()
        self._next = 0

    def __call__(self, messages: Sequence[Mapping[str, str]]) -> Any:
        with self._lock:
            sampler = self.samplers[self._next]
            self._next = (self._next + 1) % len(self.samplers)
        return sampler(messages)


class JsonlJournal:
    """Append-only per-example journal used to resume costly official evaluation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> dict[str, dict[str, object]]:
        if not self.path.exists():
            return {}
        records: dict[str, dict[str, object]] = {}
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict) or type(value.get("prompt_id")) is not str:
                    raise ValueError(f"invalid journal record at line {line_number}")
                prompt_id = cast(str, value["prompt_id"])
                if prompt_id in records:
                    raise ValueError(f"duplicate journal record for {prompt_id}")
                records[prompt_id] = cast(dict[str, object], value)
        return records

    def append(self, record: Mapping[str, object]) -> None:
        prompt_id = record.get("prompt_id")
        if type(prompt_id) is not str or not prompt_id:
            raise ValueError("journal record requires prompt_id")
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(payload + "\n")
                stream.flush()


def validate_official_panel(
    examples: Sequence[Mapping[str, object]], *, expected_count: int = SAMPLE_COUNT
) -> None:
    if expected_count <= 0:
        raise ValueError("official HealthBench expected count must be positive")
    if len(examples) != expected_count:
        raise ValueError(f"official panel must contain {expected_count} examples")
    prompt_ids = [item.get("prompt_id") for item in examples]
    if any(type(value) is not str or not value for value in prompt_ids):
        raise ValueError("official HealthBench prompt IDs are incomplete")
    if len(set(cast(list[str], prompt_ids))) != expected_count:
        raise ValueError("official HealthBench panel contains duplicate prompt IDs")


def candidate_usage_dict(usage: object) -> Mapping[str, int | None]:
    """Normalize SGLang Chat Completions usage to official report field names."""

    if usage is None:
        return {
            "input_tokens": None,
            "input_cached_tokens": None,
            "output_tokens": None,
            "output_reasoning_tokens": None,
            "total_tokens": None,
        }
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "input_cached_tokens": getattr(prompt_details, "cached_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "output_reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def generate_candidates(
    *,
    examples: Sequence[Mapping[str, object]],
    sampler: QwenOfficialCandidateSampler,
    journal: JsonlJournal,
    usage_parser: Callable[[object], Mapping[str, int | None]],
    n_threads: int,
    failure_journal: JsonlJournal | None = None,
    expected_count: int = SAMPLE_COUNT,
) -> dict[str, dict[str, object]]:
    """Generate only missing direct answers; rubrics never enter the sampler call."""

    if n_threads <= 0:
        raise ValueError("n_threads must be positive")
    validate_official_panel(examples, expected_count=expected_count)
    completed = journal.load()
    expected = {cast(str, row["prompt_id"]) for row in examples}
    if not set(completed).issubset(expected):
        raise ValueError("candidate journal belongs to another HealthBench panel")

    pending = [row for row in examples if row["prompt_id"] not in completed]

    def generate(row: Mapping[str, object]) -> dict[str, object]:
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("HealthBench prompt must be a non-empty message list")
        # Deliberately pass only the official prompt. Rubrics remain in row and are
        # used exclusively by the later grader phase.
        response = sampler(cast(list[dict[str, str]], prompt))
        record: dict[str, object] = {
            "format": "skillev-healthbench-official-candidate@1",
            "prompt_id": cast(str, row["prompt_id"]),
            "response_text": response.response_text,
            "actual_queried_message_list": response.actual_queried_message_list,
            "response_model": response.response_metadata.get("model"),
            "finish_reason": response.response_metadata.get("finish_reason"),
            "usage": dict(usage_parser(response.response_metadata.get("usage"))),
        }
        journal.append(record)
        return record

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(generate, row): cast(str, row["prompt_id"]) for row in pending}
        errors: list[Exception] = []
        for future in as_completed(futures):
            prompt_id = futures[future]
            try:
                record = future.result()
            except (HealthBenchTransportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(exc)
                if failure_journal is not None:
                    failure_journal.append(
                        {
                            "format": "skillev-healthbench-failure@1",
                            "prompt_id": f"{prompt_id}:candidate:{time.time_ns()}",
                            "failure_of_prompt_id": prompt_id,
                            "phase": "candidate",
                            "failure_kind": (
                                HealthBenchFailureKind.CANDIDATE_GENERATION_INFRASTRUCTURE.value
                            ),
                            "error_type": type(exc).__name__,
                        }
                    )
            else:
                completed[cast(str, record["prompt_id"])] = record
        if errors:
            raise HealthBenchTransportError(
                f"{len(errors)} HealthBench candidate rows require infrastructure retry"
            ) from errors[0]
    return completed


def grade_candidates(
    *,
    examples: Sequence[Mapping[str, object]],
    candidates: Mapping[str, Mapping[str, object]],
    evaluator: Any,
    journal: JsonlJournal,
    n_threads: int,
    failure_journal: JsonlJournal | None = None,
    expected_count: int = SAMPLE_COUNT,
    invocation_namespace: str = "",
) -> dict[str, dict[str, object]]:
    """Run the pinned simple-evals grade_sample implementation for missing rows."""

    if n_threads <= 0:
        raise ValueError("n_threads must be positive")
    validate_official_panel(examples, expected_count=expected_count)
    expected = {cast(str, row["prompt_id"]) for row in examples}
    if set(candidates) != expected:
        raise ValueError("all official-panel candidates are required before grading")
    completed = journal.load()
    if not set(completed).issubset(expected):
        raise ValueError("score journal belongs to another HealthBench panel")
    pending = [row for row in examples if row["prompt_id"] not in completed]

    def grade(row: Mapping[str, object]) -> dict[str, object]:
        prompt_id = cast(str, row["prompt_id"])
        candidate = candidates[prompt_id]
        messages = candidate.get("actual_queried_message_list")
        response_text = candidate.get("response_text")
        rubrics = row.get("rubrics")
        tags = row.get("example_tags", [])
        if not isinstance(messages, list) or type(response_text) is not str:
            raise ValueError("candidate journal record is incomplete")
        if not isinstance(rubrics, list) or not isinstance(tags, list):
            raise ValueError("official HealthBench grading inputs are incomplete")
        invocation_id = f"{invocation_namespace}:{prompt_id}" if invocation_namespace else prompt_id
        token = _HEALTHBENCH_GRADER_INVOCATION.set(invocation_id)
        try:
            metrics, _readable, rubric_grades = evaluator.grade_sample(
                prompt=messages,
                response_text=response_text,
                example_tags=tags,
                rubric_items=rubrics,
            )
        finally:
            _HEALTHBENCH_GRADER_INVOCATION.reset(token)
        record: dict[str, object] = {
            "format": "skillev-healthbench-official-score@1",
            "prompt_id": prompt_id,
            "metrics": metrics,
            "rubric_items": rubric_grades,
        }
        journal.append(record)
        return record

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(grade, row): cast(str, row["prompt_id"]) for row in pending}
        errors: list[Exception] = []
        for future in as_completed(futures):
            prompt_id = futures[future]
            try:
                record = future.result()
            except (HealthBenchTransportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(exc)
                if failure_journal is not None:
                    failure_journal.append(
                        {
                            "format": "skillev-healthbench-failure@1",
                            "prompt_id": f"{prompt_id}:grader:{time.time_ns()}",
                            "failure_of_prompt_id": prompt_id,
                            "phase": "grader",
                            "failure_kind": (
                                HealthBenchFailureKind.GRADER_TRANSPORT_INFRASTRUCTURE.value
                                if isinstance(exc, HealthBenchTransportError)
                                else HealthBenchFailureKind.LOCAL_IMPLEMENTATION_ERROR.value
                            ),
                            "error_type": type(exc).__name__,
                        }
                    )
            else:
                completed[cast(str, record["prompt_id"])] = record
        if errors:
            raise HealthBenchTransportError(
                f"{len(errors)} HealthBench grader rows require infrastructure retry"
            ) from errors[0]
    return completed


def aggregate_official_scores(
    scores: Iterable[Mapping[str, object]],
    *,
    clipped_stat: Callable[[list[float], str], float | int],
    expected_count: int = SAMPLE_COUNT,
) -> dict[str, object]:
    """Reproduce simple-evals' mean-then-clip and bootstrap aggregation order."""

    by_metric: dict[str, list[float]] = defaultdict(list)
    count = 0
    for record in scores:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("score record metrics are absent")
        keys = {str(name) for name in metrics}
        if "overall_score" not in keys:
            raise ValueError("HealthBench score record lacks overall_score")
        count += 1
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("HealthBench metric values must be numeric")
            by_metric[str(name)].append(float(value))
    if count != expected_count:
        raise ValueError(f"official aggregate requires {expected_count} scores")
    aggregated: dict[str, object] = {}
    for name, values in by_metric.items():
        aggregated[name] = clipped_stat(values, "mean")
        aggregated[f"{name}:n_samples"] = clipped_stat(values, "n_samples")
        aggregated[f"{name}:bootstrap_std"] = clipped_stat(values, "bootstrap_std")
    return aggregated


def normalize_healthbench_model_invalid_scores(
    *,
    scores: Mapping[str, Mapping[str, object]],
    candidates: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], int]:
    """Force a normally returned empty candidate to a definitive metric zero."""

    if set(scores) != set(candidates):
        raise ValueError("HealthBench scores and candidates differ")
    normalized: dict[str, dict[str, object]] = {}
    invalid_count = 0
    for prompt_id, score in scores.items():
        metrics = score.get("metrics")
        response_text = candidates[prompt_id].get("response_text")
        if not isinstance(metrics, dict) or type(response_text) is not str:
            raise ValueError("HealthBench score or candidate record is incomplete")
        if "overall_score" not in metrics:
            raise ValueError("HealthBench score record lacks overall_score")
        for value in metrics.values():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("HealthBench metric values must be numeric")
        row = dict(score)
        if not response_text.strip():
            row["metrics"] = dict.fromkeys(metrics, 0.0)
            row["model_output_invalid"] = True
            invalid_count += 1
        else:
            row["metrics"] = dict(metrics)
            row["model_output_invalid"] = False
        normalized[prompt_id] = row
    return normalized, invalid_count


def aggregate_healthbench_seed_metrics(
    *,
    metrics_by_seed: Mapping[int, Mapping[str, object]],
    terminal_count_by_seed: Mapping[int, int],
    infrastructure_failures_by_seed: Mapping[int, int],
    expected_seeds: tuple[int, ...] = (40,),
    expected_count: int = SAMPLE_COUNT,
) -> dict[str, object]:
    """Aggregate only complete, infrastructure-clear seed runs.

    Official per-seed metrics remain on their native unit interval.  This
    answer-free aggregate publishes percentage values.  A single frozen seed
    uses identity reduction; multi-seed diagnostic callers additionally
    receive population standard deviation without selecting a favorable seed.
    """

    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("HealthBench aggregation requires unique expected seeds")
    expected = set(expected_seeds)
    if (
        set(metrics_by_seed) != expected
        or set(terminal_count_by_seed) != expected
        or set(infrastructure_failures_by_seed) != expected
    ):
        raise ValueError("HealthBench seed records differ from the frozen aggregation")
    if any(terminal_count_by_seed[seed] != expected_count for seed in expected_seeds):
        raise ValueError("HealthBench seed run is incomplete")
    if any(infrastructure_failures_by_seed[seed] != 0 for seed in expected_seeds):
        raise HealthBenchTransportError("HealthBench seed run has infrastructure failures")
    metric_keys = tuple(metrics_by_seed[expected_seeds[0]])
    if not metric_keys or any(
        tuple(metrics_by_seed[seed]) != metric_keys for seed in expected_seeds[1:]
    ):
        raise ValueError("HealthBench metric identities differ across seeds")
    output_metrics: dict[str, object] = {}
    for metric_id in metric_keys:
        values: list[float] = []
        for seed in expected_seeds:
            value = metrics_by_seed[seed][metric_id]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("HealthBench seed metrics must be numeric")
            values.append(float(value) * 100.0)
        output_metrics[metric_id] = {
            "mean_percent": statistics.fmean(values),
            "population_std_percent": (
                statistics.pstdev(values) if len(expected_seeds) > 1 else None
            ),
            "per_seed_percent": {
                str(seed): value for seed, value in zip(expected_seeds, values, strict=True)
            },
        }
    return {
        "format": "skillev-healthbench-seed-aggregate@1",
        "seeds": list(expected_seeds),
        "run_count": len(expected_seeds),
        "reducer": "arithmetic-mean" if len(expected_seeds) > 1 else "identity",
        "dispersion": ("population-standard-deviation" if len(expected_seeds) > 1 else None),
        "sample_count_per_seed": expected_count,
        "metrics": output_metrics,
    }


def _required_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _required_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _required_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _required_none(value: object, label: str) -> None:
    if value is not None:
        raise ValueError(f"{label} must be omitted by using null")
    return None


def _required_boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


__all__ = [
    "CANDIDATE_MAX_TOKENS",
    "CANDIDATE_TEMPERATURE",
    "GRADER_MODEL",
    "OPENAI_SYSTEM_MESSAGE",
    "SAMPLE_COUNT",
    "ExternalHealthBenchRubricSampler",
    "ExternalJudgeTransport",
    "HealthBenchExternalJudgeProfile",
    "HealthBenchFailureKind",
    "HealthBenchGraderReceipt",
    "HealthBenchQwenJudgeProfile",
    "HealthBenchTransportError",
    "HealthBenchTransportExhausted",
    "JsonlJournal",
    "QwenChatTransport",
    "QwenHealthBenchRubricSampler",
    "QwenOfficialCandidateSampler",
    "RoundRobinSampler",
    "RubricAttemptRegistry",
    "TransportRetryPolicy",
    "aggregate_healthbench_seed_metrics",
    "aggregate_official_scores",
    "candidate_usage_dict",
    "generate_candidates",
    "grade_candidates",
    "load_external_judge_profile",
    "load_qwen_judge_profile",
    "normalize_healthbench_model_invalid_scores",
    "validate_official_panel",
]
