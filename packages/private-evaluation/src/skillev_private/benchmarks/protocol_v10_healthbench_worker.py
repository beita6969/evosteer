"""Private one-shot HealthBench grader using the pinned simple-evals code.

The worker receives the candidate response and the owner-only case payload over
stdin.  Rubrics and independent reference responses therefore remain outside
the rollout task and never enter model-visible state.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


def _absolute_directory(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{label} must be an absolute directory")
    return path.resolve()


def _object(value: object, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return value


def _messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("HealthBench prompt must be a non-empty message list")
    messages: list[dict[str, str]] = []
    for raw in value:
        message = _object(raw, fields={"content", "role"}, label="HealthBench message")
        if type(message["role"]) is not str or type(message["content"]) is not str:
            raise TypeError("HealthBench message fields must be text")
        messages.append({"content": message["content"], "role": message["role"]})
    return messages


@dataclass(slots=True)
class _BoundedOpenAISampler:
    client: Any
    model: str
    sampler_response_type: type[Any]
    max_calls: int
    repair_max_output_tokens: int = 1024
    calls: int = 0
    attempts_by_prompt: dict[str, int] | None = None
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        if self.repair_max_output_tokens not in (1024, 4096):
            raise ValueError("unsupported HealthBench repair output budget")
        self.attempts_by_prompt = {}

    def __call__(self, messages: list[dict[str, str]]) -> Any:
        with self._lock:
            self.calls += 1
            if self.calls > self.max_calls:
                raise RuntimeError("HealthBench grader exhausted its parse attempts")
            prompt_key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            assert self.attempts_by_prompt is not None
            attempt = self.attempts_by_prompt.get(prompt_key, 0) + 1
            self.attempts_by_prompt[prompt_key] = attempt
        if attempt > 2:
            raise RuntimeError("HealthBench grader returned malformed JSON twice")
        request_messages = list(messages)
        if attempt == 2:
            request_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Repair the prior response. Return only one JSON object with "
                        "string explanation and boolean criteria_met fields."
                    ),
                }
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=request_messages,
            temperature=0,
            top_p=1,
            seed=0,
            max_tokens=1024 if attempt == 1 else self.repair_max_output_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            response_format={
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
                        "required": ["explanation", "criteria_met"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        content = response.choices[0].message.content
        if type(content) is not str or not content.strip():
            raise RuntimeError("HealthBench grader returned an empty response")
        return self.sampler_response_type(
            response_text=content,
            response_metadata={"usage": response.usage},
            actual_queried_message_list=request_messages,
        )


def _load_official(source_root: Path) -> tuple[Any, Any, type[Any]]:
    if not (source_root / "healthbench_eval.py").is_file():
        raise ValueError("pinned simple-evals HealthBench source is absent")
    sys.path.insert(0, str(source_root.parent))
    package = source_root.name
    module = importlib.import_module(f"{package}.healthbench_eval")
    types_module = importlib.import_module(f"{package}.types")
    return module.HealthBenchEval, module.RubricItem, types_module.SamplerResponse


def _rubrics(case: dict[str, Any], rubric_type: type[Any]) -> list[Any]:
    kind = case.get("grader_kind")
    if kind in {"official-healthbench", "healthbench-qwen35-local-simple-evals"}:
        raw = case.get("rubrics")
        if not isinstance(raw, list) or not raw:
            raise ValueError("HealthBench official rubrics are absent")
        return [rubric_type.from_dict(item) for item in raw]
    if kind == "reference-health-dialogue":
        reference = case.get("reference_response")
        if type(reference) is not str or not reference.strip():
            raise ValueError("health-dialogue reference response is absent")
        criterion = (
            "The assistant response is medically consistent with the private reference "
            "response, covers its clinically important information, and introduces no "
            f"contradictory medical advice. Private reference response: {reference}"
        )
        return [rubric_type(criterion=criterion, points=1.0, tags=["reference-consistency"])]
    raise ValueError("HealthBench grader kind is unsupported")


def main(arguments: argparse.Namespace) -> None:
    source_root = _absolute_directory(arguments.official_source_root, label="simple-evals root")
    request = _object(
        json.loads(sys.stdin.buffer.readline()),
        fields={"candidate_answer", "operation", "private_case", "task_id"},
        label="HealthBench worker request",
    )
    if request["operation"] != "grade" or type(request["task_id"]) is not str:
        raise ValueError("HealthBench worker operation is unsupported")
    if type(request["candidate_answer"]) is not str:
        raise TypeError("HealthBench candidate answer must be text")
    case = request["private_case"]
    if not isinstance(case, dict):
        raise TypeError("HealthBench private case must be an object")
    prompt = _messages(case.get("prompt"))
    health_eval_type, rubric_type, sampler_response_type = _load_official(source_root)
    rubric_items = _rubrics(case, rubric_type)

    from openai import OpenAI

    client_options: dict[str, Any] = {
        "api_key": "EMPTY",
        "max_retries": 0,
        "timeout": arguments.request_timeout_seconds,
    }
    if arguments.api_base_url is not None:
        client_options["base_url"] = arguments.api_base_url
    if getattr(arguments, "broker_socket", None) is not None:
        import httpx

        client_options["http_client"] = httpx.Client(
            transport=httpx.HTTPTransport(uds=arguments.broker_socket)
        )
        client_options["base_url"] = "http://localhost/v1"
    sampler = _BoundedOpenAISampler(
        OpenAI(**client_options),
        arguments.grader_model,
        sampler_response_type,
        len(rubric_items) * 2,
        repair_max_output_tokens=getattr(arguments, "repair_max_output_tokens", 1024),
    )
    evaluator = object.__new__(health_eval_type)
    evaluator.grader_model = sampler
    evaluator.length_adjustment_center = None
    evaluator.length_adjustment_penalty_per_500_chars = None
    with contextlib.redirect_stdout(io.StringIO()):
        metrics, _, rubric_grades = evaluator.grade_sample(
            prompt=prompt,
            response_text=request["candidate_answer"],
            example_tags=[],
            rubric_items=rubric_items,
        )
    score = metrics.get("overall_score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise RuntimeError("HealthBench official score is unavailable")
    negative = sum(
        1
        for item in rubric_grades
        if item.get("points", 0) < 0 and item.get("criteria_met") is True
    )
    json.dump(
        {
            "official_rubric_score": float(score),
            "triggered_negative_rubric_count": negative,
        },
        sys.stdout,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source-root", required=True)
    parser.add_argument("--grader-model", default="Qwen3.5-9B")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--broker-socket")
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--repair-max-output-tokens", type=int, choices=(1024, 4096), default=1024)
    main(parser.parse_args())
