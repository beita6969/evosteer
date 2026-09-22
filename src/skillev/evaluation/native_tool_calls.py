"""Literal Qwen3.5 tool-call envelopes; no execution, voting or argument inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from .response_syntax import protocol_lines
from .scienceworld_commands import TYPED_COMMANDS

NativeTools = tuple[dict[str, Any], ...]

# Literal zero-argument simulator commands, not framework-selected actions.
ALFWORLD_OBSERVATION_CALLS = ("look", "inventory", "help")

_BRACKET_COMMAND = r"(click|search)\[([^\[\]\r\n<]+)\]"
_LITERAL_FUNCTION = r"(?:[A-Za-z_][A-Za-z0-9_]* [^<>\r\n]+|(?:click|search)\[[^\[\]\r\n<]+\])"
# A real length-stopped response included Qwen's message-end token after its
# complete tool envelope. Accept only that outer boundary, not a prose suffix,
# second message/call, or missing closing syntax. Arguments remain literal.
_ENVELOPE_END = r"</function>\s*</tool_call>\s*(?:<\|im_end\|>\s*)?"


@dataclass(frozen=True, slots=True)
class NativeToolCall:
    name: str
    arguments: dict[str, str]


def _unframe(value: str) -> str:
    # The official template frames each literal with one newline on each side.
    # Preserve all other whitespace, backslashes, quotes and code indentation.
    value = value[2:] if value.startswith("\r\n") else value.removeprefix("\n")
    return value[:-2] if value.endswith("\r\n") else value.removesuffix("\n")


def _agrees_with_final(payload: str, repeated: list[str]) -> bool:
    """Compare only literal declared finals, never infer an answer from explanation."""
    declared = [
        match[1].strip()
        for _, line in protocol_lines(payload)
        if (match := re.fullmatch(r"Final(?: answer)?:[ \t]*(.+?)\s*", line, re.I))
    ]
    values = declared or [payload.strip()]
    return all(value == repeated[0] for value in (*values, *repeated))


def _tool_starts(text: str) -> list[int]:
    return [
        offset + len(line) - len(line.lstrip())
        for offset, line in protocol_lines(text)
        if line.lstrip().startswith("<tool_call>")
    ]


def native_tool_call(text: str) -> NativeToolCall | None:
    """Accept one native envelope with optional outer Qwen message-end framing.

    Mentions in code fences or inline prose are not submissions. Multiple calls,
    ambiguous envelopes and duplicate parameters need an owner repair, not a
    framework choice. A closed, argument-free literal command can omit its
    function header's closing angle bracket. Values are literal, not XML entities.
    """
    starts = _tool_starts(text)
    if not starts:
        return None
    if len(starts) != 1:
        raise ValueError("one native tool call is required per owner decision")
    prefix = text[: starts[0]].strip().splitlines()
    if prefix and re.match(r"(?:(?:for\s+)?examples?\b|do\s+not\b|don't\b)", prefix[-1], re.I):
        raise ValueError("a quoted or negated tool call is not a submission")
    repeated_finals: list[str] = []
    for _, line in protocol_lines(text[: starts[0]]):
        if final := re.fullmatch(r"Final(?: answer)?:[ \t]*(.+?)\s*", line, re.I):
            repeated_finals.append(final[1].strip())
            continue
        if re.match(
            r"(?:Action|Final(?: answer| response| code)?|Terminal payload|Message to \w+|Review):"
            r"|(?:\w+\.)?(?:click|search|act|点击|搜索)\s*[\[(]"
            r'|\{\s*"kind"\s*:\s*"(?:tool|message|history)"',
            line,
            re.I,
        ):
            raise ValueError("multiple explicit submission channels")
    envelope = re.fullmatch(
        rf"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_]*|点击|搜索|{_LITERAL_FUNCTION})>"
        r"(.*?)" + _ENVELOPE_END,
        text[starts[0] :],
        re.S,
    )
    if envelope is None:
        # A real ScienceWorld owner repeatedly copied act(command into the
        # function header, then emitted one literal argument and its closing
        # parameter tag. The published operation/parameter are explicit; no
        # action or object is inferred from its surrounding discussion.
        signature = re.fullmatch(
            r"<tool_call>\s*<function=act\(command\)?>\r?\n"
            r"(.*?)</parameter>\s*" + _ENVELOPE_END,
            text[starts[0] :],
            re.S,
        )
        # Real ALFWorld and WebShop trajectories put complete commands here
        # with a missing ">" and an empty parameter closing tag. Keep its literal
        # command, never infer arguments. The action boundary still validates
        # the environment, exact current command menu and state revision.
        literal = re.fullmatch(
            rf"<tool_call>\s*<function=({_LITERAL_FUNCTION})\r?\n"
            r"\s*(?:</parameter>\s*)?" + _ENVELOPE_END,
            text[starts[0] :],
        )
        if signature is not None:
            function_name, rest = "act", "<parameter=command>" + signature[1] + "</parameter>"
        elif literal is None:
            raise ValueError("malformed native tool envelope")
        else:
            function_name, rest = literal[1], ""
    else:
        function_name, rest = envelope[1], envelope[2]
    arguments: dict[str, str] = {}
    while rest.strip():
        parameter = re.match(
            r"\s*<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>", rest, re.S
        )
        if parameter is None or parameter[1] in arguments:
            raise ValueError("malformed or duplicate native parameter")
        value = _unframe(parameter[2])
        if re.search(r"</?(?:tool_call|function|parameter)(?:[=>])", value):
            raise ValueError("ambiguous nested native envelope")
        arguments[parameter[1]] = value
        rest = rest[parameter.end() :]
    if repeated_finals and not (
        function_name == "submit_answer"
        and set(arguments) == {"answer"}
        and _agrees_with_final(arguments["answer"], repeated_finals)
    ):
        raise ValueError("explicit final and native submission disagree")
    bracket_command = re.fullmatch(_BRACKET_COMMAND, function_name)
    if bracket_command is not None:
        if arguments:
            raise ValueError("a literal command cannot also supply parameters")
        name, argument = bracket_command.groups()
        return NativeToolCall(name, {"target" if name == "click" else "query": argument})
    name = {"点击": "click", "搜索": "search"}.get(function_name, function_name)
    return NativeToolCall(name, arguments)


def native_corpus_queries(text: str) -> tuple[str, ...] | None:
    """Preserve an owner's ordered read-only query batch, never select one call.

    A real owner requested a river's source and mouth in two tool envelopes.
    Stateful environment actions and final submissions still require one carrier.
    """
    starts = _tool_starts(text)
    if not starts:
        return None
    boundaries = [0, *starts[1:], len(text)]
    envelopes = [text[start:end] for start, end in pairwise(boundaries)]
    if any(envelope.rstrip().endswith("<|im_end|>") for envelope in envelopes[:-1]):
        raise ValueError("corpus queries cannot continue past the owner's message end")
    calls = [native_tool_call(envelope) for envelope in envelopes]
    if len(calls) == 1 and calls[0] is not None and calls[0].name != "corpus_search":
        return None
    if any(
        call is None
        or call.name != "corpus_search"
        or set(call.arguments) != {"query"}
        or not call.arguments["query"].strip()
        for call in calls
    ):
        raise ValueError("only literal read-only corpus queries may share a tool batch")
    return tuple(call.arguments["query"] for call in calls if call is not None)


def native_control_payload(text: str) -> dict[str, Any] | None:
    queries = native_corpus_queries(text)
    if queries is not None:
        return (
            {"kind": "corpus_search", "query": queries[0]}
            if len(queries) == 1
            else {"kind": "corpus_search", "queries": list(queries)}
        )
    call = native_tool_call(text)
    if call is None:
        return None
    args = call.arguments
    if call.name in {"list_skills", "retrieve_skills", "read_skill", "invoke_skill"}:
        allowed = {
            "list_skills": {"cursor", "limit"},
            "retrieve_skills": set(),
            "read_skill": {"skill_id"},
            "invoke_skill": {"skill_id"},
        }[call.name]
        if args.keys() - allowed:
            raise ValueError("unsupported native skill parameter")
        values: dict[str, Any] = dict(args)
        for key in ("cursor", "limit"):
            if key in values:
                if re.fullmatch(r"[0-9]+", values[key]) is None:
                    raise ValueError("skill pagination requires literal integer parameters")
                values[key] = int(values[key])
        return {"kind": "skill", "operation": call.name, **values}
    if call.name in {"solver", "researcher"} and set(args) == {"body"}:
        return {"kind": "message", "recipient": call.name, "body": args["body"]}
    if (
        call.name == "history"
        and "archive" in args
        and args.keys() <= {"archive", "cursor", "limit"}
    ):
        cursor, limit = args.get("cursor", "0"), args.get("limit", "4")
        if not all(re.fullmatch(r"[0-9]+", value) for value in (cursor, limit)):
            raise ValueError("history requires literal integer parameters")
        return {
            "kind": "history",
            "archive": args["archive"],
            "cursor": int(cursor),
            "limit": int(limit),
        }
    if call.name in {
        "search",
        "click",
        "act",
        "wait",
        "submit_answer",
        *ALFWORLD_OBSERVATION_CALLS,
        *TYPED_COMMANDS,
    }:
        return None  # The environment or final-owner boundary validates these.
    if " " in call.name and not args:
        # Observed Qwen outputs put a COMPLETE native command in the function
        # slot. Do not infer arguments: the action boundary must match that
        # literal command to the current ALFWorld menu before execution.
        return None
    raise ValueError("unknown native control or unsupported parameters")
