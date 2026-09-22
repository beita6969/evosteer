"""Syntax-only decisions bound to the public state on which they were made.

Free-form reasoning and peer messages are never searched for executable verbs.
The caller supplies the channel and revision, not the language model.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from enum import StrEnum

from .native_tool_calls import ALFWORLD_OBSERVATION_CALLS, native_tool_call
from .scienceworld_commands import TYPED_COMMANDS


class DecisionChannel(StrEnum):
    ACTION = "action"
    MESSAGE = "message"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class PublicSurface:
    mode: str
    revision: int
    native_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExplicitDecision:
    text: str
    state_revision: int
    channel: DecisionChannel = DecisionChannel.ACTION


@dataclass(frozen=True, slots=True)
class TransportResult:
    action: str | None
    error: str | None = None


_ARGUMENT_NAMES = {
    "search": "query",
    "搜索": "query",
    "click": "target",
    "点击": "target",
    "act": "command",
    **dict.fromkeys(TYPED_COMMANDS, "target"),
}


def _unfence(text: str) -> str:
    fenced = re.fullmatch(
        r"```(?:json|text|python|tool)?[ \t]*\r?\n(.*?)\r?\n?```", text, re.I | re.S
    )
    return fenced[1].strip() if fenced is not None else text


def _tool_text(name: str, arguments: object, surface: PublicSurface) -> TransportResult:
    if surface.mode == "alfworld" and " " in name and arguments == {}:
        return TransportResult(name)  # Exact current-menu validation follows below.
    name = {"点击": "click", "搜索": "search"}.get(name, name.lower())
    if surface.mode == "scienceworld" and name == "wait":
        # Observed owners put the complete zero-argument native command in the
        # function slot. This is exactly act("wait"), not a chosen duration or
        # an action inferred from discussion; normal surface/revision checks apply.
        return (
            TransportResult("wait")
            if arguments == {}
            else TransportResult(None, "invalid-tool-argument")
        )
    if surface.mode == "scienceworld" and name in TYPED_COMMANDS:
        if name not in surface.native_actions:
            return TransportResult(None, "unavailable-tool")
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"target"}
            or not isinstance(arguments["target"], str)
            or not arguments["target"].strip()
        ):
            return TransportResult(None, "invalid-tool-argument")
        return TransportResult(TYPED_COMMANDS[name] + arguments["target"])
    if surface.mode == "alfworld" and name in ALFWORLD_OBSERVATION_CALLS:
        return (
            TransportResult(name)
            if arguments == {}
            else TransportResult(None, "invalid-tool-argument")
        )
    if name not in {
        "webshop": ("click", "search"),
        "alfworld": ("act",),
        "scienceworld": ("act",),
        "trivia-search": ("search",),
    }.get(surface.mode, ()):
        return TransportResult(None, "unavailable-tool")
    key = {"click": "target", "search": "query", "act": "command"}[name]
    if (
        not isinstance(arguments, dict)
        or len(arguments) != 1
        or not isinstance(arguments.get(key), str)
    ):
        return TransportResult(None, "invalid-tool-argument")
    argument = arguments[key]
    if name == "act":
        return TransportResult(argument)
    # The public menu contains complete calls. A real owner copied click[x]
    # into click's target argument; wrapping it again lost that explicit intent.
    # Accept one same-operation wrapper, without choosing or replacing a target.
    wrapped = re.fullmatch(r"(click|search|点击|搜索)\[([^\[\]\r\n]+)\]", argument, re.I)
    if wrapped is not None:
        verb = {"点击": "click", "搜索": "search"}.get(wrapped[1].lower(), wrapped[1].lower())
        if verb == name:
            return TransportResult(f"{name}[{wrapped[2]}]")
    return TransportResult(f"{name}[{argument}]")


def _string_literal(text: str) -> str | None:
    # Sentence punctuation outside the quoted argument is not part of its value.
    # A period inside the quotes is preserved; only one outside period is allowed.
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    return None


def _native_payload(
    text: str, surface: PublicSurface, *, require_call: bool = False
) -> TransportResult:
    text = _unfence(text.strip())
    inline = re.fullmatch(r"`([^`\r\n]+)`", text)
    if inline is not None:
        text = inline[1]
    bracket = re.fullmatch(r"(click|search|点击|搜索)\[([^\[\]\r\n]+)\]", text, re.I)
    if bracket is not None and surface.mode in {"webshop", "trivia-search"}:
        verb = {"点击": "click", "搜索": "search"}.get(bracket[1].lower(), bracket[1].lower())
        return TransportResult(f"{verb}[{bracket[2]}]")
    parameterized = re.fullmatch(
        r"(click|search|点击|搜索)\[(query|target)\][ \t]+with[ \t]+"
        r"(query|target)[ \t]+(.+)",
        text,
        re.I | re.S,
    )
    if parameterized is not None:
        # The model repeated the public signature and supplied its literal
        # named argument. Both names must agree; no query is inferred from prose.
        if parameterized[2].lower() != parameterized[3].lower():
            return TransportResult(None, "invalid-tool-argument")
        return _tool_text(
            parameterized[1],
            {parameterized[3].lower(): _string_literal(parameterized[4])},
            surface,
        )
    resource = {
        "webshop": "webshop",
        "alfworld": "alfworld",
        "scienceworld": "scienceworld",
        "trivia-search": "triviaqa",
    }
    if text.startswith("{"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return TransportResult(None, "invalid-json")
        if not isinstance(value, dict) or value.get("kind") != "tool":
            return TransportResult(None, "not-a-tool-call")
        if value.get("resource_id") != resource.get(surface.mode):
            return TransportResult(None, "wrong-tool-resource")
        if not isinstance(value.get("name"), str):
            return TransportResult(None, "invalid-tool-call")
        return _tool_text(value["name"], value.get("arguments"), surface)
    named = re.fullmatch(
        r"(\w+)[ \t]+(?:with[ \t]+)?(query|target|command)"
        r"(?:[ \t]*[:=][ \t]*|[ \t]+)(.+)",
        text,
        re.I | re.S,
    )
    if named is not None:
        # An explicit parameter followed by one quoted literal is a call, not
        # a request to infer an argument from surrounding natural language.
        literal = _string_literal(named[3])
        return _tool_text(named[1], {named[2].lower(): literal}, surface)
    positional = re.fullmatch(r"(search|click|act)[ \t]+(?:(for|on)[ \t]+)?(.+)", text, re.I | re.S)
    if positional is not None:
        name = positional[1].lower()
        connector = positional[2].lower() if positional[2] else None
        literal = _string_literal(positional[3])
        if literal is not None and (
            connector is None or (name, connector) in {("search", "for"), ("click", "on")}
        ):
            positional_key = {"search": "query", "click": "target", "act": "command"}[name]
            return _tool_text(name, {positional_key: literal}, surface)
    if not re.match(r"[\w.]+\s*\(", text):
        return (
            TransportResult(None, "expected-one-native-call")
            if require_call
            else TransportResult(text)
        )
    # Real model responses use the public function signature, not just bracket
    # calls. Parse its literal argument without executing Python or any tool.
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return TransportResult(None, "invalid-tool-call")
    if not isinstance(expression, ast.Call):
        return TransportResult(None, "invalid-tool-call")
    function = expression.func
    if isinstance(function, ast.Name):
        name = function.id
    elif isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        if function.value.id != resource.get(surface.mode):
            return TransportResult(None, "wrong-tool-resource")
        name = function.attr
    else:
        return TransportResult(None, "invalid-tool-call")
    if not expression.args and not expression.keywords:
        return _tool_text(name, {}, surface)
    if len(expression.args) + len(expression.keywords) != 1:
        return TransportResult(None, "invalid-tool-argument")
    if expression.args:
        key = _ARGUMENT_NAMES.get(name.lower())
        argument = expression.args[0]
    else:
        keyword = expression.keywords[0]
        key, argument = keyword.arg, keyword.value
    if key is None or not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return TransportResult(None, "invalid-tool-argument")
    return _tool_text(name, {key: argument.value}, surface)


def _named_argument(text: str) -> dict[str, str] | None:
    field = re.fullmatch(r"(query|target|command)[ \t]*([:=])[ \t]*(.+)", text, re.I | re.S)
    if field is None:
        return None
    value = field[3].strip()
    if value.startswith(('"', "'")):
        literal = _string_literal(value)
        if literal is None:
            return None
        value = literal
    elif field[2] == "=" or "\n" in value or "\r" in value:
        return None
    return {field[1].lower(): value}


def _bare_native_call(text: str, surface: PublicSurface) -> bool:
    if text.lstrip().startswith("{"):
        return False  # A trailing JSON representation is compared explicitly below.
    payload = _native_payload(text, surface)
    action = payload.action
    return action is not None and (
        (action in surface.native_actions and action != "search")
        or re.fullmatch(r"(?:click|search|点击|搜索)\[[^\[\]]+\]", action) is not None
    )


def _bind_explicit_arguments(
    header: str, arguments: object, surface: PublicSurface
) -> TransportResult:
    # Real calls repeated the published search[query] signature in Action:
    # and supplied query in Arguments:. Bind only an exact named placeholder;
    # a concrete target in the header must never be replaced by another value.
    signature = re.fullmatch(r"(\w+)\[(query|target|command)\]", header, re.I)
    if signature is not None:
        header = signature[1]
        if _ARGUMENT_NAMES.get(header.lower()) != signature[2].lower():
            return TransportResult(None, "invalid-tool-argument")
    if isinstance(arguments, str):
        key = _ARGUMENT_NAMES.get(header.lower())
        arguments = {key: arguments} if key is not None else None
    return _tool_text(header, arguments, surface)


def _explicit_payload(text: str, surface: PublicSurface) -> TransportResult:
    """Separate one submitted call from discussion, never mine mentions of verbs."""
    text = _unfence(text)
    try:
        native = native_tool_call(text)
    except ValueError:
        return TransportResult(None, "invalid-tool-call")
    if native is not None:
        return _tool_text(native.name, native.arguments, surface)
    if text.startswith("{"):
        return TransportResult(text)
    lines = text.splitlines()
    fields: list[tuple[int, int, str]] = []
    bare_calls: list[tuple[int, int, str]] = []
    in_fence = False
    consumed_until = -1
    for index, line in enumerate(lines):
        if index <= consumed_until:
            continue
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        match = re.fullmatch(r"Action:[ \t]*(.*)", line, re.I)
        if not in_fence and match is not None:
            fields.append((index, index, match[1]))
        elif not in_fence and _bare_native_call(line, surface):
            bare_calls.append((index, index, line))
        elif not in_fence and line.lstrip().startswith("{"):
            # A whole, standalone JSON call is also an explicit boundary. Decode
            # it as an object, never search nested arguments for executable text.
            rest = "\n".join(lines[index:]).lstrip()
            try:
                value, end = json.JSONDecoder().raw_decode(rest)
            except json.JSONDecodeError:
                continue
            consumed_until = index + rest[:end].count("\n")
            if (
                not fields
                and isinstance(value, dict)
                and value.get("kind") == "tool"
                and not rest[end:].split("\n", 1)[0].strip()
            ):
                bare_calls.append((index, consumed_until, rest[:end]))
    if not fields and (
        surface.mode in {"webshop", "trivia-search"}
        or any(call.lstrip().startswith("{") for _, _, call in bare_calls)
    ):
        # A standalone native API call is already a submission boundary; prose
        # before it does not require an extra Action: label. Count all literal
        # calls, including unavailable targets, before validating the surface.
        # ALFWorld's ordinary command words can also be discussion, so this
        # does not start extracting natural-language verbs from paragraphs.
        fields, bare_calls = bare_calls, []
    if not fields:
        if surface.mode == "scienceworld":
            # Unlike ALFWorld, this surface has no finite native-command menu.
            # Treating arbitrary prose as a command sent completion narratives
            # to the simulator repeatedly. Require an explicit act/Action/JSON
            # carrier, without filtering the command argument's meaning.
            call = _native_payload(text, surface, require_call=True)
            if call.action is None:
                return call
        return TransportResult(text)
    if len(fields) != 1 or bare_calls:
        return TransportResult(None, "ambiguous-action-fields")
    index, end_index, action = fields[0]
    action = action.strip()
    prefix = [line.strip() for line in lines[:index] if line.strip()]
    if prefix and re.match(r"(?:(?:for\s+)?examples?\b|do\s+not\b|don't\b)", prefix[-1], re.I):
        return TransportResult(None, "quoted-or-negated-action")
    remainder = "\n".join(lines[end_index + 1 :]).strip()
    if not remainder:
        return TransportResult(action.strip())
    argument = re.fullmatch(r"Arguments?:[ \t]*(.+)", remainder, re.I | re.S)
    if argument is None:
        named = _named_argument(remainder)
        if named is not None:
            return _bind_explicit_arguments(action, named, surface)
        # A native Action field and its identical JSON serialization still
        # designate one decision. Conflicting or prose-only fields do not.
        if _unfence(remainder).startswith("{"):
            first, second = _native_payload(action, surface), _native_payload(remainder, surface)
            if first.action is not None and first.action == second.action:
                return first
            if second.action is not None:
                # A bare operation header can leave its argument to the JSON
                # below, provided both representations name the same tool.
                value = json.loads(_unfence(remainder))
                header = _bind_explicit_arguments(action, value.get("arguments"), surface)
                if header.action == second.action:
                    return second
        return TransportResult(None, "ambiguous-action-section")
    try:
        arguments = json.loads(argument[1])
    except json.JSONDecodeError:
        arguments = _named_argument(argument[1])
        raw = argument[1].strip()
        if arguments is None and not (
            re.match(r"(?:query|target|command)\s*[:=]", raw, re.I)
            or raw.startswith(('"', "'", "{", "["))
            or "\n" in raw
            or "\r" in raw
        ):
            # The explicit Argument field names the single parameter's literal
            # value. No object, query, or subsequent action is inferred.
            arguments = raw
    return _bind_explicit_arguments(action, arguments, surface)


def normalize_decision(decision: ExplicitDecision, surface: PublicSurface) -> TransportResult:
    if decision.channel is not DecisionChannel.ACTION:
        return TransportResult(None, "not-an-action-channel")
    if decision.state_revision != surface.revision:
        return TransportResult(None, "stale-public-state")
    payload = _explicit_payload(decision.text.strip(), surface)
    if payload.action is None:
        return payload
    native = _native_payload(payload.action, surface)
    if native.action is None:
        return native
    text = native.action
    if surface.mode in {"webshop", "trivia-search"}:
        match = re.fullmatch(r"(click|search|点击|搜索)\[([^\[\]\r\n]+)\]", text, re.I)
        if match is None:
            return TransportResult(None, "expected-one-native-call")
        verb = {"点击": "click", "搜索": "search"}.get(match[1].lower(), match[1].lower())
        if surface.mode == "trivia-search" and verb != "search":
            return TransportResult(None, "unavailable-tool")
        text = f"{verb}[{match[2]}]"
        allowed = text in surface.native_actions or (
            verb == "search" and "search" in surface.native_actions
        )
        if surface.mode == "webshop" and verb == "click":
            # The deployed native step lowercases its argument before looking up
            # the clickable key. Accept exactly that equivalence, not fuzzy labels.
            # Keep the owner's spelling on the wire; the native engine normalizes it.
            allowed = allowed or f"click[{match[2].lower()}]" in surface.native_actions
    elif surface.mode in {"alfworld", "scienceworld"}:
        # ScienceWorld's free-form act surface must not admit an empty/NUL
        # command: the official worker rejects it before stepping, which used
        # to abort the whole evaluation instead of asking the owner to repair.
        if not text.strip() or "\x00" in text:
            return TransportResult(None, "invalid-tool-argument")
        if "\n" in text or "\r" in text:
            # A paragraph ending in a mentioned command is not a submitted call.
            # Report the missing call boundary, not that a listed tool is absent.
            return TransportResult(None, "expected-one-native-call")
        allowed = text in surface.native_actions or (
            surface.mode == "scienceworld" and "act" in surface.native_actions
        )
    else:
        return TransportResult(None, "unknown-tool-surface")
    if not allowed:
        return TransportResult(None, "not-on-current-public-surface")
    return TransportResult(text)


def public_repair_feedback(
    result: TransportResult, surface: PublicSurface, *, finish_reason: str = ""
) -> str:
    """No evaluator fields, semantic hints, target substitutions, or hidden retries."""
    representation_help = ""
    if result.error in {
        "expected-one-native-call",
        "ambiguous-action-fields",
        "ambiguous-action-section",
        "invalid-json",
        "invalid-tool-call",
        "invalid-tool-argument",
        "not-a-tool-call",
        "wrong-tool-resource",
        "unavailable-tool",
    }:
        representation_help = (
            " Submit one native call by itself, or put it in a single final Action: field "
            "after any discussion. Discussion alone is not an executable call."
        )
        if surface.mode == "alfworld":
            representation_help += (
                " For a function call, the function is act and its command parameter "
                "contains your complete household command."
            )
        elif surface.mode == "scienceworld":
            representation_help += (
                " Use act(command) with your complete command as a quoted string, "
                "or Action: followed by that command. A chat completion claim does "
                "not end the simulator episode."
            )
        elif surface.mode == "webshop":
            representation_help += (
                " The shopping episode is still active; a chat recommendation "
                "does not submit a purchase."
            )
        if result.error == "wrong-tool-resource":
            resource = "triviaqa" if surface.mode == "trivia-search" else surface.mode
            representation_help += f" This tool's JSON resource_id is {json.dumps(resource)}."
    # A real ALFWorld owner interpreted the internal "surface" error code as
    # an object's physical location. Explain command availability, not world facts.
    reason = {
        "not-on-current-public-surface": (
            "The proposed command is not in the environment's current available-action list."
        ),
        "stale-public-state": "The command refers to an older environment state.",
        "not-an-action-channel": "The response was sent on a non-action channel.",
        "expected-one-native-call": "The response did not contain one executable native call.",
        "ambiguous-action-fields": "The response contains conflicting action fields.",
        "ambiguous-action-section": "The response does not identify one action to execute.",
        "invalid-json": "The tool-call JSON could not be decoded.",
        "invalid-tool-call": "The tool call could not be decoded.",
        "invalid-tool-argument": "The call does not supply the tool's required literal argument.",
        "not-a-tool-call": "The submitted object is not a tool call.",
        "wrong-tool-resource": "The call names a different environment interface.",
        "unavailable-tool": "That tool is not available through this environment interface.",
    }.get(result.error or "", "The response could not be executed by this environment interface.")
    limit_notice = (
        " This generation ended at its output-token limit." if finish_reason == "length" else ""
    )
    search_help = " Search accepts your query." if "search" in surface.native_actions else ""
    return (
        f"The action was not sent to the environment and caused no state change. {reason}"
        f"{limit_notice} Current public state revision: "
        f"{surface.revision}. Available actions: "
        f"{json.dumps(surface.native_actions, ensure_ascii=False)}. "
        "Use a command from this list." + search_help + representation_help
    )
