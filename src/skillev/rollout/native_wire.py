"""Explicit, single-call tool carriers; never repairs or selects answer content.

Independent implementation inspired by tools-based interfaces, not vendored upstream
code. The complete sampled carrier remains the scored span, including its delimiters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.contracts.action_text import action_payload
from skillev.contracts.action_wire import (
    NATIVE_TOOL_HANDOFF_WIRE,
    NATIVE_TOOL_WIRES,
    STRICT_NATIVE_TOOL_WIRE,
)
from skillev.runtime import ActionKind, ActionParseResult, ActionParseStatus, StructuredAction

from .action_surface import TerminalMode, ToolActionSpecV2

if TYPE_CHECKING:
    from .action_contract import ActionContract

NATIVE_TOOL_WIRE_VERSION = STRICT_NATIVE_TOOL_WIRE
NATIVE_TOOL_BOUNDARY_VERSION = "native-model-stop@1"


def _schema(value: JsonValue) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        if "type" in value:
            return dict(value)
        return _object_schema({key: _schema(child) for key, child in value.items()})
    if isinstance(value, str):
        kind = next(
            (kind for kind in ("string", "integer", "boolean", "number") if kind in value), None
        )
        return {"type": kind, "description": value} if kind else {"description": value}
    return {"const": value}


def _object_schema(
    properties: dict[str, JsonValue], required: list[str] | None = None
) -> dict[str, JsonValue]:
    return {
        "type": "object",
        "properties": properties,
        "required": cast(list[JsonValue], list(properties) if required is None else required),
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class NativeToolBinding:
    name: str
    kind: ActionKind
    resource_id: str | None
    internal_name: str
    parameters_json: str
    description: str

    def to_tool(self) -> dict[str, JsonValue]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": json.loads(self.parameters_json),
            },
        }


def native_bindings(
    contract: ActionContract, *, public_action_semantics: bool = False
) -> tuple[NativeToolBinding, ...]:
    surface = contract.surface
    if surface is None:
        raise ValueError("native wire requires an explicit public action surface")
    semantics = " ".join(contract.render_native_semantics()) if public_action_semantics else ""
    rows: list[NativeToolBinding] = []
    names: set[str] = {"submit_answer", "read_skill"}
    for index, tool in enumerate(surface.tools):
        # Names are frozen per surface; collisions use a deterministic position,
        # never fuzzy matching or selection based on an execution outcome.
        name = tool.name
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) or name in names:
            name = f"task_tool_{index}"
        while name in names:
            name += "_"
        names.add(name)
        if isinstance(tool, ToolActionSpecV2):
            properties: dict[str, JsonValue] = {}
            required = []
            for key, field in tool.arguments.items():
                entry: dict[str, JsonValue] = {
                    "type": [field.value_type.value, "null"]
                    if field.nullable
                    else field.value_type.value
                }
                for bound in ("minimum", "maximum"):
                    value = getattr(field, bound)
                    if value is not None:
                        entry[bound] = value
                if not field.required:
                    entry["default"] = field.default
                else:
                    required.append(key)
                properties[key] = entry
            parameters = _object_schema(properties, required)
        else:
            parameters = _object_schema(
                {key: _schema(value) for key, value in tool.arguments_schema.items()}
            )
        rows.append(
            NativeToolBinding(
                name,
                ActionKind.TOOL,
                tool.resource_id,
                tool.name,
                canonical_json(parameters),
                f"Execute {tool.resource_id}.{tool.name} with the public task arguments."
                + (" " + semantics if semantics else ""),
            )
        )
    if surface.terminal_mode is TerminalMode.EXPLICIT_COMPLETION and surface.completion is not None:
        value_schema = surface.completion.value_schema
        # Explicit answer object becomes submit_answer(answer=...). Other native
        # completion shapes retain their declared value rather than guessing one.
        parameters = _schema(value_schema)
        if not isinstance(value_schema, dict) or "type" in value_schema:
            parameters = _object_schema({"value": parameters})
        rows.append(
            NativeToolBinding(
                "submit_answer",
                ActionKind.COMPLETE,
                None,
                "complete",
                canonical_json(parameters),
                "Submit the final response using exactly the declared fields."
                + (" " + semantics if semantics else ""),
            )
        )
    visible = tuple(sid for sid in contract.retrieved_skill_ids if sid in contract.active_skill_ids)
    if visible:
        parameters = _object_schema({"skill_id": {"type": "string", "enum": list(visible)}})
        rows.append(
            NativeToolBinding(
                "read_skill",
                ActionKind.SKILL,
                "skill-runtime",
                "invoke",
                canonical_json(parameters),
                "Invoke one visible skill by exact ID; optional, not a required step.",
            )
        )
    return tuple(rows)


class NativeToolWire:
    format_version = NATIVE_TOOL_WIRE_VERSION

    def __init__(
        self, contract: ActionContract, *, format_version: str = NATIVE_TOOL_WIRE_VERSION
    ) -> None:
        if format_version not in NATIVE_TOOL_WIRES:
            raise ValueError("unsupported native tool carrier")
        self.format_version = format_version
        self.contract = contract
        self.bindings = native_bindings(contract)

    def parse(self, action_text: str) -> ActionParseResult:
        if not isinstance(action_text, str):
            raise TypeError("action text must be text")
        text = action_text.strip()
        if self.format_version != STRICT_NATIVE_TOOL_WIRE:
            try:
                text = _single_carrier(
                    text, braced_commentary=self.format_version == NATIVE_TOOL_HANDOFF_WIRE
                )
            except ValueError:
                return ActionParseResult(
                    ActionParseStatus.PARSE_ERROR, None, "native_call_carrier_invalid"
                )
            if not text.startswith("<tool_call>"):
                try:
                    binding, arguments = self._json_call(text)
                    return self._action(binding, arguments)
                except json.JSONDecodeError:
                    return ActionParseResult(
                        ActionParseStatus.PARSE_ERROR, None, "native_call_carrier_invalid"
                    )
                except (ValueError, TypeError, KeyError):
                    return ActionParseResult(
                        ActionParseStatus.SCHEMA_INVALID, None, "native_call_schema_invalid"
                    )
        # V1 retains its original strict whole-response contract. Old evidence
        # must not acquire a different interpretation when the runtime upgrades.
        if not (text.startswith("<tool_call>") and text.endswith("</tool_call>")):
            return ActionParseResult(
                ActionParseStatus.PARSE_ERROR, None, "native_call_carrier_invalid"
            )
        try:
            inner = text[len("<tool_call>") : -len("</tool_call>")].strip()
            if self.format_version != STRICT_NATIVE_TOOL_WIRE and inner.startswith("{"):
                binding, arguments = self._json_call(inner)
                return self._action(binding, arguments)
            match = re.fullmatch(
                r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*)</function>", inner, flags=re.DOTALL
            )
            if match is None:
                raise ValueError("not one explicit Qwen function")
            matched_binding = next((row for row in self.bindings if row.name == match[1]), None)
            if matched_binding is None:
                raise ValueError("unknown function")
            binding = matched_binding
            parameters = json.loads(binding.parameters_json)["properties"]
            arguments_raw: dict[str, JsonValue] = {}
            rest = match[2].strip()
            while rest:
                parameter = re.match(r"<parameter=([^>]+)>(.*?)</parameter>", rest, flags=re.DOTALL)
                if parameter is None:
                    raise ValueError("incomplete or multiple calls")
                key, value = parameter[1], parameter[2]
                if key not in parameters or key in arguments_raw:
                    raise ValueError("unknown or duplicated parameter")
                # Remove exactly the framing newline emitted by Qwen's template,
                # not arbitrary code indentation or meaningful string whitespace.
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                kind = parameters[key].get("type")
                if kind == "string" or (
                    isinstance(kind, list) and "string" in kind and value != "null"
                ):
                    arguments_raw[key] = value
                else:
                    arguments_raw[key] = normalize_json(json.loads(value))
                rest = rest[parameter.end() :].strip()
            return self._action(binding, arguments_raw)
        except (ValueError, TypeError, KeyError):
            return ActionParseResult(
                ActionParseStatus.SCHEMA_INVALID, None, "native_call_schema_invalid"
            )

    def _json_call(self, text: str) -> tuple[NativeToolBinding, dict[str, JsonValue]]:
        value = normalize_json(json.loads(text, object_pairs_hook=_unique_object))
        if not isinstance(value, dict):
            raise ValueError("tool carrier must be an object")
        if set(value) in ({"name", "arguments"}, {"name", "parameters"}):
            binding = next((row for row in self.bindings if row.name == value["name"]), None)
            arguments = value.get("arguments", value.get("parameters"))
            if binding is None or not isinstance(arguments, dict):
                raise ValueError("explicit call has no declared function/arguments")
        else:
            # Bare arguments are accepted only when the PUBLIC signatures alone
            # identify one function. No model, scorer or answer-value selection.
            matches = [row for row in self.bindings if _argument_fields_match(row, value)]
            if len(matches) != 1:
                raise ValueError("arguments do not identify one declared function")
            binding, arguments = matches[0], value
        if not _argument_fields_match(binding, arguments):
            raise ValueError("arguments differ from the declared function")
        return binding, arguments

    def _action(
        self, binding: NativeToolBinding, arguments_raw: dict[str, JsonValue]
    ) -> ActionParseResult:
        arguments: JsonValue = normalize_json(arguments_raw)
        skill_id: str | None = None
        if binding.kind is ActionKind.SKILL:
            if set(arguments_raw) != {"skill_id"}:
                raise ValueError("skill arguments differ")
            candidate_id = arguments_raw["skill_id"]
            if not isinstance(candidate_id, str):
                raise ValueError("skill ID must be text")
            skill_id = candidate_id
            if (
                skill_id not in self.contract.retrieved_skill_ids
                or skill_id not in self.contract.active_skill_ids
            ):
                raise ValueError("skill is not in the visible catalog")
            arguments = {}
        elif binding.kind is ActionKind.COMPLETE:
            surface = self.contract.surface
            assert surface is not None
            assert surface.completion is not None
            schema = surface.completion.value_schema
            if isinstance(schema, dict) and "type" not in schema:
                if set(arguments_raw) != set(schema):
                    raise ValueError("completion fields differ")
                arguments = {"value": arguments}
            elif set(arguments_raw) != {"value"}:
                raise ValueError("completion requires its native value")
        action = StructuredAction(
            binding.kind, binding.internal_name, arguments, binding.resource_id, skill_id
        )
        return ActionParseResult(ActionParseStatus.VALID, action, None)


def _single_carrier(text: str, *, braced_commentary: bool = False) -> str:
    """Separate commentary from one explicit call, not from a candidate answer.

    A call inside reasoning is not a submission. Multiple/incomplete calls stay
    errors. The caller retains the entire response and sampled IDs for scoring.
    """
    payload = action_payload(text)
    if payload.startswith("{") and (not braced_commentary or _json_object_start(payload)):
        # Delimiter strings inside JSON arguments are data, not tool/channel
        # boundaries. Malformed JSON still fails in the one JSON decoder.
        return payload
    fences = list(
        re.finditer(
            r"^[ \t]*```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```[ \t]*(?=\r?$)",
            text,
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
    )
    native_start = text.find("<tool_call>")
    if fences and (native_start < 0 or fences[0].start() < native_start):
        if len(fences) != 1:
            raise ValueError("multiple explicit JSON carriers are not one action")
        fence = fences[0]
        before, after = text[: fence.start()], text[fence.end() :]
        for outside in (before, after):
            if any(marker in outside for marker in ("```", "<tool_call>", "</tool_call>")):
                raise ValueError("additional or incomplete explicit carrier")
        _validate_commentary(before, after, reject_incomplete_json=braced_commentary)
        # Decode the owner's one explicit object, not an answer found in prose.
        # Signature admission and duplicate-key rejection still happen below.
        return fence[1]
    opens, closes = text.count("<tool_call>"), text.count("</tool_call>")
    if opens or closes:
        if opens != 1 or closes != 1:
            raise ValueError("expected one complete explicit tool call")
        start, end = text.index("<tool_call>"), text.index("</tool_call>")
        if end < start:
            raise ValueError("tool delimiters are out of order")
        before, after = text[:start], text[end + len("</tool_call>") :]
        _validate_commentary(before, after, reject_incomplete_json=braced_commentary)
        return text[start : end + len("</tool_call>")]
    if "<think>" in text or "</think>" in text:
        raise ValueError("reasoning must not be interpreted as an action carrier")
    return action_payload(text)


def _json_object_start(text: str) -> bool:
    # An object key starts with a quote (or } for an empty object). In V3,
    # ordinary braced annotations no longer hide a later explicit native call.
    # An unfinished actual JSON object still cannot be bypassed this way.
    return re.match(r'^\{\s*(?:"|})', text) is not None


def _validate_commentary(before: str, after: str, *, reject_incomplete_json: bool = False) -> None:
    if before.rfind("<think>") > before.rfind("</think>") or "</think>" in after:
        raise ValueError("a tool call in a reasoning channel is not a submission")
    for outside in (before, after):
        carrier = action_payload(outside)
        if carrier.startswith("{"):
            try:
                additional = json.loads(carrier)
            except ValueError:
                if reject_incomplete_json and _json_object_start(carrier):
                    raise ValueError("additional incomplete JSON carrier") from None
                continue
            if isinstance(additional, dict):
                raise ValueError("multiple explicit carriers are not one action")


def _argument_fields_match(binding: NativeToolBinding, arguments: dict[str, JsonValue]) -> bool:
    schema = json.loads(binding.parameters_json)
    return set(schema["required"]) <= set(arguments) <= set(schema["properties"])


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate argument field")
        result[key] = value
    return result
