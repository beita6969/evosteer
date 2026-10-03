from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from skillev.contracts.canonical import JsonValue, normalize_json, stable_hash
from skillev.runtime.budget_ledger import BudgetLedger
from skillev.runtime.contracts import BudgetReservation, BudgetSettlement, BudgetVector

from .evosteer_features import answer_present
from .graph import NodeExecutionRequest, NodeExecutionResult
from .model_executor import (
    FrozenTextExecutor,
    _fitted_prompt,
    _generation,
    _render,
    cut_outcome,
)
from .text_tools import (
    KNOWN_TEXT_TOOLS,
    MIN_TURN_OUTPUT,
    RESULT_BLOCK,
    STOP_STRINGS,
    TextTool,
    TextToolsConfig,
    ToolRun,
    first_call,
    head_tail,
    normalize_final,
)

FORMAT = "frozen-tool-text-node@1"
_WRAPPER_TOKENS = 256
_CLOSING = "\n\nPerform your role using "
_CUT_OFF = "[cut off; continue]"
_FINAL = (
    "No tool calls are left. Reply now with your final result. Only this reply is "
    "passed on, so it must be complete: include everything the task asks for "
    "(the complete code or the final answer)."
)
_FINAL_WITHOUT_TURNS = (
    "No tool calls are available for this reply. Reply now with your final result; "
    "it must be complete: include everything the task asks for."
)
_ELIDED_RESULT = "[result omitted to fit the input limit]"
CodeOutput = Callable[[str, str, tuple[str, ...]], str | None]


def tool_block(tools: tuple[str, ...], settings: TextToolsConfig, calls: int) -> str:
    lines = [
        "Tools: you may end a reply with ONE tool call. It is run, its result is shown to "
        "you as <result>...</result>, and you continue.",
    ]
    plural = "s" if calls != 1 else ""
    if "python" in tools:
        lines.append(
            "<python>\ncode\n</python> runs the code in a fresh Python process (no network, "
            f"nothing kept between calls, {settings.python_wall_seconds} s limit) and "
            "returns what it prints."
        )
    if "search" in tools:
        lines.append(
            "<search>query</search> searches Wikipedia and returns the top "
            f"{settings.search_top_k} passages."
        )
    lines.append(
        f"You can make at most {calls} tool call{plural}. Code in ``` fences "
        "is never run. Do not write <result> yourself."
    )
    lines.append(
        "Your final reply is the reply without a tool call; only it is passed on, so it "
        "must be complete."
    )
    return "\n".join(lines)


def _with_tool_block(
    block: str,
) -> Callable[[NodeExecutionRequest, list[dict[str, Any]], str | None], str]:
    def render(
        request: NodeExecutionRequest, messages: list[dict[str, Any]], previous: str | None
    ) -> str:
        text = _render(request, messages, previous)
        head, separator, tail = text.rpartition(_CLOSING)
        if not separator:
            raise AssertionError("node prompt lost its closing instruction")
        return f"{head}\n\n{block}{separator}{tail}"

    return render


def _turn_seed(seed: int, turn: int) -> int:
    if turn == 1:
        return seed
    return int(stable_hash({"node_seed": seed, "turn": turn}).split(":", 1)[-1][:16], 16)


def _byte_count(text: str) -> int:
    return len(text.encode("utf-8")) + 32


@dataclass(frozen=True, slots=True)
class _Turn:
    reply: str
    result: str | None

    def render(self, *, elide_result: bool = False) -> str:
        if self.result is None:
            return f"{self.reply}\n{_CUT_OFF}"
        result = _ELIDED_RESULT if elide_result else self.result
        return f"{self.reply}\n<result>\n{result}\n</result>"


def _work(base: str, turns: list[_Turn], closing: str, *, elide_before: int = 0) -> str:
    if not turns:
        return f"{base}\n\n{closing}"
    shown = [
        turn.render(elide_result=index < elide_before) for index, turn in enumerate(turns)
    ]
    return "\n\n".join([base, "Your work so far:", *shown, closing])


class FrozenToolTextExecutor:
    supports_role_tools = True

    def __init__(
        self,
        policy: Any,
        tools: Mapping[str, TextTool],
        settings: TextToolsConfig,
        *,
        temperature: float = 0.3,
        answered_at_cap: bool = False,
        stop_regex: tuple[str, ...] = (),
        code_output: CodeOutput | None = None,
        code_output_id: str | None = None,
    ) -> None:
        if not isinstance(settings, TextToolsConfig):
            raise TypeError("settings must be TextToolsConfig")
        if code_output is not None and (
            not callable(code_output)
            or type(code_output_id) is not str
            or not code_output_id.strip()
        ):
            raise ValueError("a code_output hook requires a pinned code_output_id")
        if code_output is None and code_output_id is not None:
            raise ValueError("code_output_id supplied without its code_output hook")
        ordered = dict(sorted(tools.items()))
        if not ordered or any(
            name not in KNOWN_TEXT_TOOLS or getattr(tool, "name", None) != name
            for name, tool in ordered.items()
        ):
            raise ValueError("tool backends must be named known text tools")
        self.policy = policy
        self._plain = FrozenTextExecutor(
            policy, temperature=temperature, answered_at_cap=answered_at_cap, stop_regex=stop_regex
        )
        self._temperature = temperature
        self._answered_at_cap = answered_at_cap
        self._code_output = code_output
        self._tools = ordered
        self._settings = settings
        self._server_stop = bool(getattr(policy, "supports_stop_strings", False))
        self._completed = 0
        self._last_outcome: dict[str, JsonValue] | None = None
        identity: dict[str, JsonValue] = {
            "executor": FORMAT,
            "text_node": self._plain.frozen_identity,
            "tools": {name: tool.identity for name, tool in ordered.items()},
            "call_syntax": "xml-tags@1",
            "result_tag": "result@1",
            "call_stop": "server" if self._server_stop else "post-hoc",
            "turn": [
                settings.tool_turn_output_tokens,
                settings.final_output_reserve,
                settings.result_chars,
            ],
        }
        if code_output_id is not None:
            identity["code_output"] = code_output_id
        self._identity = str(stable_hash(identity))

    @property
    def frozen_identity(self) -> str:
        return self._identity

    def public_environment_state(self) -> dict[str, JsonValue]:
        return {
            "enabled": False,
            "environment_terminal": False,
            "tool_calls_completed": self._completed,
            "last_outcome": None if self._last_outcome is None else dict(self._last_outcome),
        }

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        role_tools = tuple(getattr(request.role, "tools", ()) or ())
        if not role_tools:
            result = await self._plain.execute(request)
            outcome = result.metadata.get("execution_outcome")
            if (
                self._code_output is None
                or not isinstance(outcome, dict)
                or not outcome.get("answer_present")
                or answer_present(result.output, code=True)
            ):
                return result
            outcome = {**outcome, "answer_present": False}
            return NodeExecutionResult(
                result.output, result.usage, {**result.metadata, "execution_outcome": outcome}
            )
        missing = sorted(set(role_tools) - set(self._tools))
        if missing:
            raise ValueError(
                f"role {request.role.role_id!r} has tools without a backend: {missing}"
            )
        return await self._run(request, role_tools)

    @staticmethod
    def _reserve(ledger: BudgetLedger, invocation: str, label: str, maximum: BudgetVector) -> str:
        identifier = f"{invocation}:{label}"
        ledger.reserve(
            BudgetReservation(identifier, ledger.run_id, ledger.attempt_id, invocation, maximum)
        )
        return identifier

    async def _generate(
        self,
        ledger: BudgetLedger,
        invocation: str,
        turn: int,
        prompt: str,
        max_new: int,
        *,
        seed: int,
        stop: tuple[str, ...],
    ) -> tuple[str, bool, dict[str, JsonValue]]:
        remaining = ledger.available
        maximum = BudgetVector(
            input_tokens=remaining.input_tokens,
            output_tokens=max_new,
            model_calls=1,
            wall_time_milliseconds=remaining.wall_time_milliseconds,
        )
        reservation = self._reserve(ledger, invocation, f"model:{turn}", maximum)
        detailed = getattr(self.policy, "frozen_generation", None)
        options: dict[str, Any] = {
            "max_new_tokens": max_new,
            "input_limit": remaining.input_tokens,
            "temperature": self._temperature,
            "seed": seed,
        }
        if stop and self._server_stop:
            options["stop"] = stop
        generate = partial(
            detailed if callable(detailed) else self.policy.frozen_text, prompt, **options
        )

        def timed() -> tuple[Any, float]:
            started = time.monotonic()
            result = generate()
            return result, time.monotonic() - started

        if getattr(self.policy, "thread_safe_generation", False):
            response, seconds = await asyncio.to_thread(timed)
        else:
            response, seconds = timed()
        text, inputs, outputs, truncated = _generation(
            response, max_new_tokens=max_new, detailed=callable(detailed)
        )
        measured_ms = max(0, math.ceil(seconds * 1000))
        wall_ms = min(measured_ms, maximum.wall_time_milliseconds)
        ledger.settle(
            BudgetSettlement(
                reservation,
                BudgetVector(
                    input_tokens=inputs,
                    output_tokens=outputs,
                    model_calls=1,
                    wall_time_milliseconds=wall_ms,
                ),
            )
        )
        usage: dict[str, JsonValue] = {
            "input_tokens": inputs,
            "output_tokens": outputs,
            "wall_ms": wall_ms,
            "truncated": truncated,
        }
        if measured_ms > wall_ms:
            usage["wall_overrun_ms"] = measured_ms - wall_ms
        return text, truncated, usage

    def _final_prompt(
        self,
        base: str,
        turns: list[_Turn],
        count: Callable[[str], int],
        remaining: BudgetVector,
        window: int | None,
    ) -> tuple[str, int, dict[str, JsonValue]]:
        want = min(remaining.output_tokens, self._settings.final_output_reserve)
        options: list[tuple[str, dict[str, JsonValue]]] = []
        if not turns:
            options.append((_work(base, [], _FINAL_WITHOUT_TURNS), {}))
        else:
            for elide in range(len(turns)):
                options.append(
                    (_work(base, turns, _FINAL, elide_before=elide), {"elided_results": elide})
                )
            for drop in range(1, len(turns) + 1):
                kept = turns[drop:]
                options.append(
                    (
                        _work(base, kept, _FINAL, elide_before=max(0, len(kept) - 1)),
                        {"elided_results": max(0, len(kept) - 1), "dropped_turns": drop},
                    )
                )
        for prompt, left_out in options:
            tokens = count(prompt)
            if tokens <= remaining.input_tokens and (window is None or tokens + want <= window):
                return prompt, tokens, left_out
        raise ValueError("the final tool-node prompt cannot fit the role's remaining envelope")

    async def _run(
        self, request: NodeExecutionRequest, role_tools: tuple[str, ...]
    ) -> NodeExecutionResult:
        settings = self._settings
        cap = request.role.model_maximum
        if min(cap.input_tokens, cap.output_tokens, cap.model_calls, cap.agent_turns) < 1 or (
            cap.wall_time_milliseconds < 1
        ):
            raise ValueError("node envelope must admit at least one frozen model turn")
        allowed_calls = min(cap.tool_calls, cap.model_calls - 1)
        if allowed_calls < 1:
            raise ValueError("a tool role's envelope must admit a tool call and a final reply")
        tools = {name: self._tools[name] for name in role_tools}
        tool_reserve = max(tool.wall_reserve_ms for tool in tools.values())
        window = getattr(self.policy, "context_window", None)
        window = window if type(window) is int else None
        counter = getattr(self.policy, "frozen_prompt_tokens", None)
        count: Callable[[str], int] = counter if callable(counter) else _byte_count
        ledger = BudgetLedger(
            run_id=request.runtime_id, attempt_id=f"node-{request.execution_index}", cap=cap
        )
        invocation = f"{request.runtime_id}:{request.node_id}:{request.execution_index}"
        agent_turn = self._reserve(ledger, invocation, "agent-turn", BudgetVector(agent_turns=1))
        budget = cap.input_tokens // 2
        if window is not None:
            budget = min(budget, window - settings.tool_turn_output_tokens)
        base, omitted = _fitted_prompt(
            request,
            count,
            max(1, budget),
            renderer=_with_tool_block(
                tool_block(role_tools, settings, allowed_calls)
            ),
        )
        turns: list[_Turn] = []
        transcript: list[dict[str, JsonValue]] = []
        by_tool: dict[str, int] = {}
        statuses: dict[str, int] = {}
        executed = model_calls = turn = 0
        slowest_ms = 0
        hook = self._code_output
        coding = hook is not None
        programs: list[str] = []
        output: str | None = None
        outcome: dict[str, JsonValue] | None = None
        termination = "final-reply"
        while True:
            remaining = ledger.available
            calls_left = min(remaining.tool_calls, remaining.model_calls - 1)
            if calls_left < 1:
                termination = "no-calls-left"
                break
            closing = f"Continue. Tool calls left: {calls_left}."
            prompt = base if not turns else _work(base, turns, closing)
            tokens = count(prompt)
            max_new = min(
                settings.tool_turn_output_tokens,
                remaining.output_tokens - settings.final_output_reserve,
            )
            if window is not None:
                max_new = min(max_new, window - tokens)
            if (
                max_new < MIN_TURN_OUTPUT
                or 2 * tokens + max_new + settings.result_chars + _WRAPPER_TOKENS
                > remaining.input_tokens
                or ledger.settled.wall_time_milliseconds + 2 * slowest_ms + tool_reserve
                > cap.wall_time_milliseconds
            ):
                termination = "budget-limit"
                break
            turn += 1
            text, truncated, usage = await self._generate(
                ledger,
                invocation,
                turn,
                prompt,
                max_new,
                seed=_turn_seed(request.seed, turn),
                stop=STOP_STRINGS,
            )
            model_calls += 1
            slowest_ms = max(slowest_ms, int(usage["wall_ms"]))
            call = first_call(text)
            if call is None:
                if not truncated:
                    output = normalize_final(text)
                    transcript.append(
                        {
                            "turn": turn,
                            "kind": "final",
                            "forced": False,
                            "normalized": output != text,
                            **({"reply": text} if output != text else {}),
                            "model": usage,
                        }
                    )
                    break
                turns.append(_Turn(text, None))
                transcript.append({"turn": turn, "kind": "cut-off", "reply": text, "model": usage})
                continue
            reply = RESULT_BLOCK.sub("", text[: call.start]) + text[call.start : call.end]
            entry: dict[str, JsonValue] = {
                "turn": turn,
                "kind": "call",
                "tool": call.tool,
                "reply": reply,
                "model": usage,
            }
            if call.tool not in tools:
                result = f"The {call.tool} tool is not available to this role."
                statuses["forbidden"] = statuses.get("forbidden", 0) + 1
                turns.append(_Turn(reply, result))
                transcript.append(
                    {**entry, "executed": False, "status": "forbidden", "result": result}
                )
                continue
            tool = tools[call.tool]
            maximum = BudgetVector(tool_calls=1, wall_time_milliseconds=tool.wall_reserve_ms)
            if not maximum.fits_within(ledger.available):
                result = "Not run: no time is left for tool calls."
                statuses["not-run"] = statuses.get("not-run", 0) + 1
                turns.append(_Turn(reply, result))
                transcript.append(
                    {**entry, "executed": False, "status": "not-run", "result": result}
                )
                termination = "budget-limit"
                break
            reservation = self._reserve(ledger, invocation, f"tool:{turn}", maximum)
            run = await tool.run(call.argument)
            if not isinstance(run, ToolRun):
                raise TypeError("tool backend returned an incompatible result")
            charged_ms = min(run.wall_ms, maximum.wall_time_milliseconds)
            ledger.settle(
                BudgetSettlement(
                    reservation, BudgetVector(tool_calls=1, wall_time_milliseconds=charged_ms)
                )
            )
            executed += 1
            by_tool[call.tool] = by_tool.get(call.tool, 0) + 1
            statuses[run.status] = statuses.get(run.status, 0) + 1
            self._completed += 1
            self._last_outcome = {"tool": call.tool, "status": run.status}
            if call.tool == "python" and run.status == "success":
                programs.append(call.argument)
            result = head_tail(run.text, settings.result_chars)
            turns.append(_Turn(reply, result))
            transcript.append(
                {
                    **entry,
                    "executed": True,
                    "status": run.status,
                    "wall_ms": run.wall_ms,
                    "result": result,
                    "record": run.record,
                    "sandbox": run.sandbox,
                }
            )
        if output is None:
            remaining = ledger.available
            if model_calls and remaining.wall_time_milliseconds < slowest_ms:
                output = normalize_final(turns[-1].reply) if turns else ""
                termination = "wall-limit"
                outcome = {"status": "failed", "answer_present": False, "failure_kind": "timeout"}
                transcript.append({"turn": turn + 1, "kind": "final-skipped"})
            else:
                prompt, tokens, left_out = self._final_prompt(base, turns, count, remaining, window)
                max_new = remaining.output_tokens
                if window is not None:
                    max_new = min(max_new, window - tokens)
                turn += 1
                text, truncated, usage = await self._generate(
                    ledger,
                    invocation,
                    turn,
                    prompt,
                    max_new,
                    seed=_turn_seed(request.seed, turn),
                    stop=(),
                )
                model_calls += 1
                output = normalize_final(text)
                transcript.append(
                    {
                        "turn": turn,
                        "kind": "final",
                        "forced": True,
                        "normalized": output != text,
                        **({"reply": text} if output != text else {}),
                        "left_out": left_out,
                        "model": usage,
                    }
                )
                if truncated:
                    outcome = cut_outcome(output, answered_at_cap=self._answered_at_cap)
        ledger.settle(BudgetSettlement(agent_turn, BudgetVector(agent_turns=1)))
        ledger.assert_fully_settled()
        program = hook(request.task_prompt, output, tuple(programs)) if hook and programs else None
        if program:
            block = f"```python\n{program.strip()}\n```"
            output = f"{output.rstrip()}\n\n{block}" if output.strip() else block
            transcript[-1]["program_appended"] = True
        if outcome is None:
            outcome = {
                "status": "answered",
                "answer_present": answer_present(output, code=coding),
                "failure_kind": None,
            }
        elif coding and outcome["answer_present"] and not answer_present(output, code=True):
            outcome = {**outcome, "answer_present": False}
        metadata = normalize_json(
            {
                "execution_outcome": outcome,
                "prompt_fit": {"omitted_characters": omitted},
                "tool_use": {
                    "calls": executed,
                    "by_tool": by_tool,
                    "statuses": statuses,
                    "model_calls": model_calls,
                    "termination": termination,
                },
                "tool_transcript": transcript,
            }
        )
        return NodeExecutionResult(output, ledger.settled, metadata)


__all__ = ["FORMAT", "FrozenToolTextExecutor", "tool_block"]
