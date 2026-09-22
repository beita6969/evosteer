"""Frozen model nodes using the same base-model facilities as the orchestrator."""

from __future__ import annotations

import asyncio
import json
import time
from functools import partial
from typing import Any

from skillev.contracts.canonical import stable_hash
from skillev.runtime.contracts import BudgetVector

from .evosteer_features import answer_present
from .graph import NodeExecutionRequest, NodeExecutionResult


def _generation(
    response: Any, *, max_new_tokens: int, detailed: bool
) -> tuple[str, int, int, bool]:
    """Validate a backend reply and resolve whether it stopped at the output cap.

    A backend with ``frozen_generation`` reports truncation itself (SGLang reads
    its finish reason). The plain ``frozen_text`` 3-tuple cannot distinguish a
    stop token sampled in the last slot from a cut, so a reply that used the
    whole allowance is counted as truncated: its answer is at best unfinished.
    """
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


_FIT_MARKER = "\n[... {count} characters omitted to fit the input limit ...]\n"
# Below this many characters a text is not shortened further; the prompt is then
# genuinely too large for the role and the backend's envelope check reports it.
_FIT_FLOOR_CHARS = 800


def _elide(text: str, keep: int) -> str:
    """Keep ``keep`` characters: 40% from the start, 60% from the end.

    A draft states its answer at the end, so the end is kept preferentially.
    """
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
    """A structured skill (a JSON object of named fields) as readable sections.

    Author skills are JSON objects; as raw JSON their steps arrive escaped and
    inline. Every field is kept: the known ones in a fixed order, then any other
    keys sorted. Lists become numbered steps or bullets. A body that is not a
    JSON object is shown verbatim.
    """
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
    """Plain-text node prompt: role, task, bound procedures, inputs, previous output.

    Skill procedures are shown verbatim under a heading instead of as escaped
    JSON strings, so the executor can read and follow them.
    """
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
    request: NodeExecutionRequest, count: Any, budget: int | None
) -> tuple[str, int]:
    """Render the prompt, shortening the longest input texts until it fits.

    Drafts, messages and the previous output can each be as long as a role's
    output cap, so their sum can exceed the input envelope; a checking node then
    sees the start and end of the longest texts instead of the batch aborting.
    The task and skill procedures are never shortened. Every shortening is cut
    from the original text, so the omitted count is exact. Deterministic.
    """
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
        return _render(request, messages, previous)

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
    """Completion-task node executor. Interactive environments use an injected executor.

    This adapter performs no hidden grading or reference-answer access. Bound
    skill bodies enter this node request only. An empty completion is preserved
    for the final evaluator to judge; there is no fabricated fallback response.
    A completion cut at the role's output cap is still returned unchanged, but
    reported as a failed execution (``failure_kind='truncated'``) that did not
    present an answer, so features and the orchestrator can see it.
    """

    def __init__(self, policy: Any, *, temperature: float = 0.3) -> None:
        if temperature <= 0:
            raise ValueError("executor temperature must be positive")
        self.policy = policy
        self.temperature = temperature
        self._identity = stable_hash(
            {
                "reference": policy.configuration_id,
                "temperature": temperature,
                # @2: output-cap truncation is reported in execution_outcome.
                # @3: plain-text prompt with procedures; inputs fitted to the envelope.
                # @4: structured skill fields rendered as readable sections.
                "executor": "frozen-completion-node@4",
            }
        )

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
        # The 3-tuple frozen_text stays the contract for authors and tool nodes;
        # a backend may additionally expose the exact finish reason.
        detailed = getattr(self.policy, "frozen_generation", None)
        generate = partial(
            detailed if callable(detailed) else self.policy.frozen_text,
            prompt,
            max_new_tokens=max_new_tokens,
            input_limit=input_limit,
            temperature=self.temperature,
            seed=request.seed,
        )

        def timed() -> tuple[Any, float]:
            # Charged time is the generation itself, measured where it runs;
            # waiting for the event loop behind other episodes is not usage.
            started = time.monotonic()
            result = generate()
            return result, time.monotonic() - started

        # A remote server may overlap with other episodes; the in-process model
        # toggles its adapter and must stay on the event-loop thread.
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
                wall_time_milliseconds=max(0, int(seconds * 1000)),
            ),
            {
                "execution_outcome": {
                    "status": "failed" if truncated else "answered",
                    # A cut completion has not finished stating its answer, and
                    # a finished one only states an answer if its text does.
                    "answer_present": not truncated and answer_present(output),
                    "failure_kind": "truncated" if truncated else None,
                },
                # Characters of drafts/messages/previous output the node did not see.
                "prompt_fit": {"omitted_characters": omitted},
            },
        )
