from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from functools import partial
from typing import Any

from skillev.contracts.canonical import JsonValue, stable_hash
from skillev.runtime.contracts import BudgetVector

from .evosteer_features import _final_boxed, answer_present
from .graph import NodeExecutionRequest, NodeExecutionResult


def _generation(
    response: Any, *, max_new_tokens: int, detailed: bool
) -> tuple[str, int, int, bool]:
    width = 4 if detailed else 3
    if not isinstance(response, tuple) or len(response) != width:
        raise TypeError("frozen policy returned an incompatible generation reply")
    output, inputs, outputs = response[:3]
    if type(output) is not str or type(inputs) is not int or type(outputs) is not int:
        raise TypeError("invalid frozen policy response fields")
    if detailed:
        truncated = response[3]
        if type(truncated) is not bool:
            raise TypeError("frozen_generation must report truncation as a boolean")
    else:
        truncated = outputs >= max_new_tokens
    return output, inputs, outputs, truncated


_ANSWER_LINE = re.compile(
    r"^[ \t>#*_-]*(?:final[ \t]+)?answer[ \t*_]*[:：][ \t*_]*(?:\n[ \t]*)?[^\s*_]",
    re.IGNORECASE | re.MULTILINE,
)


def states_final_answer(text: str) -> bool:
    boxed = _final_boxed(text)
    return bool(boxed and boxed.strip()) or _ANSWER_LINE.search(text) is not None


def cut_outcome(output: str, *, answered_at_cap: bool) -> dict[str, JsonValue]:
    failed: dict[str, JsonValue] = {
        "status": "failed",
        "answer_present": False,
        "failure_kind": "truncated",
    }
    if not answered_at_cap:
        return failed
    if not states_final_answer(output):
        return {**failed, "truncated": True}
    return {"status": "answered", "answer_present": True, "failure_kind": None, "truncated": True}


_FIT_MARKER = "\n[... {count} characters omitted to fit the input limit ...]\n"
_FIT_FLOOR_CHARS = 800


def _elide(text: str, keep: int) -> str:
    omitted = len(text) - keep
    if omitted <= 0:
        return text
    head = (keep * 2) // 5
    return text[:head] + _FIT_MARKER.format(count=omitted) + text[len(text) - (keep - head):]


_SKILL_FIELDS = (
    ("name", "Name"),
    ("description", "Purpose"),
    ("trigger", "Use when"),
    ("plan", "Steps"),
    ("pitfall", "Pitfalls to avoid"),
    ("constraint", "Constraint"),
)


def _procedure_text(body: str) -> str:
    try:
        value = json.loads(body)
    except ValueError:
        return body
    if not isinstance(value, dict) or not value:
        return body

    def lines(label: str, item: Any, numbered: bool) -> list[str]:
        if isinstance(item, list):
            marks = [f"{i}." if numbered else "-" for i in range(1, len(item) + 1)]
            return [f"{label}:"] + [
                f"{mark} {x if isinstance(x, str) else json.dumps(x, sort_keys=True)}"
                for mark, x in zip(marks, item, strict=True)
            ]
        text = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
        return [f"{label}: {text}"]

    known = [key for key, _ in _SKILL_FIELDS]
    out: list[str] = []
    for key, label in _SKILL_FIELDS:
        if key in value:
            out += lines(label, value[key], numbered=key == "plan")
    for key in sorted(k for k in value if k not in known):
        out += lines(key, value[key], numbered=False)
    return "\n".join(out)


def _render(
    request: NodeExecutionRequest,
    messages: list[dict[str, Any]],
    previous_output: str | None,
) -> str:
    parts = [f"Role: {request.role.instruction}", f"Task:\n{request.task_prompt}"]
    for skill_id, body in request.skills:
        parts.append(f"Procedure to follow (skill {skill_id}):\n{_procedure_text(body)}")
    for message in messages:
        parts.append(
            f"Input from agent {message.get('source_id')} "
            f"({message.get('protocol')}):\n{message.get('body')}"
        )
    if previous_output is not None:
        parts.append(f"Your previous output:\n{previous_output}")
    inputs = ["the task"]
    if request.skills:
        inputs.append("the procedure")
    if messages:
        inputs.append("the inputs from other agents")
    if previous_output is not None:
        inputs.append("your previous output")
    listed = inputs[0] if len(inputs) == 1 else ", ".join(inputs[:-1]) + " and " + inputs[-1]
    parts.append(f"Perform your role using {listed}. Return your result.")
    return "\n\n".join(parts)


def _fitted_prompt(
    request: NodeExecutionRequest,
    count: Any,
    budget: int | None,
    *,
    renderer: Callable[[NodeExecutionRequest, list[dict[str, Any]], str | None], str] = _render,
) -> tuple[str, int]:
    originals: dict[int, str] = {
        index: message["body"]
        for index, message in enumerate(request.messages)
        if isinstance(message.get("body"), str)
    }
    if request.previous_output is not None:
        originals[-1] = request.previous_output
    keep = {index: len(text) for index, text in originals.items()}

    def render() -> str:
        messages = [dict(message) for message in request.messages]
        for index, message in enumerate(messages):
            if index in originals:
                message["body"] = _elide(originals[index], keep[index])
        previous = None if -1 not in originals else _elide(originals[-1], keep[-1])
        return renderer(request, messages, previous)

    prompt = render()
    if budget is None or not callable(count):
        return prompt, 0
    while count(prompt) > budget:
        shrinkable = [(kept, index) for index, kept in keep.items() if kept > _FIT_FLOOR_CHARS]
        if not shrinkable:
            break
        kept, index = max(shrinkable)
        keep[index] = max(_FIT_FLOOR_CHARS, (kept * 7) // 10)
        prompt = render()
    return prompt, sum(len(originals[index]) - kept for index, kept in keep.items())


class FrozenTextExecutor:
    def __init__(
        self,
        policy: Any,
        *,
        temperature: float = 0.3,
        answered_at_cap: bool = False,
        stop_regex: tuple[str, ...] = (),
    ) -> None:
        if temperature <= 0:
            raise ValueError("executor temperature must be positive")
        if type(answered_at_cap) is not bool:
            raise TypeError("answered_at_cap must be a boolean")
        if type(stop_regex) is not tuple or not all(type(p) is str and p for p in stop_regex):
            raise ValueError("stop_regex must be a tuple of nonempty patterns")
        if stop_regex and not getattr(policy, "supports_stop_regex", False):
            raise ValueError("stop_regex needs a backend that stops at a regex (SGLang)")
        self.policy = policy
        self.temperature = temperature
        self.answered_at_cap = answered_at_cap
        self.stop_regex = stop_regex
        identity: dict[str, Any] = {
            "reference": policy.configuration_id,
            "temperature": temperature,
            "executor": "frozen-completion-node@4",
        }
        if answered_at_cap:
            identity["answered_at_cap"] = True
        if stop_regex:
            identity["stop_regex"] = list(stop_regex)
        self._identity = stable_hash(identity)

    @property
    def frozen_identity(self) -> str:
        return str(self._identity)

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        max_new_tokens = request.role.model_maximum.output_tokens
        input_limit = request.role.model_maximum.input_tokens
        window = getattr(self.policy, "context_window", None)
        budget = input_limit
        if type(window) is int:
            budget = min(input_limit, window - max_new_tokens)
        prompt, omitted = _fitted_prompt(
            request, getattr(self.policy, "frozen_prompt_tokens", None), budget
        )
        detailed = getattr(self.policy, "frozen_generation", None)
        generate = partial(
            detailed if callable(detailed) else self.policy.frozen_text,
            prompt,
            max_new_tokens=max_new_tokens,
            input_limit=input_limit,
            temperature=self.temperature,
            seed=request.seed,
            **({"stop_regex": self.stop_regex} if self.stop_regex else {}),
        )

        def timed() -> tuple[Any, float]:
            started = time.monotonic()
            result = generate()
            return result, time.monotonic() - started

        if getattr(self.policy, "thread_safe_generation", False):
            response, seconds = await asyncio.to_thread(timed)
        else:
            response, seconds = timed()
        output, inputs, outputs, truncated = _generation(
            response, max_new_tokens=max_new_tokens, detailed=callable(detailed)
        )
        return NodeExecutionResult(
            output,
            BudgetVector(
                input_tokens=inputs,
                output_tokens=outputs,
                model_calls=1,
                agent_turns=1,
                wall_time_milliseconds=min(
                    max(0, int(seconds * 1000)), request.role.model_maximum.wall_time_milliseconds
                ),
            ),
            {
                "execution_outcome": (
                    cut_outcome(output, answered_at_cap=self.answered_at_cap)
                    if truncated
                    else {
                        "status": "answered",
                        "answer_present": answer_present(output),
                        "failure_kind": None,
                    }
                ),
                "prompt_fit": {"omitted_characters": omitted},
            },
        )
