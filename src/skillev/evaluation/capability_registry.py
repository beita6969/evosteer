"""Public capabilities and optional text advice; no benchmark routing or action policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .native_tool_calls import ALFWORLD_OBSERVATION_CALLS, NativeTools
from .scienceworld_commands import (
    COMMAND_PROFILE,
    LEGACY_COMMAND_PROFILE,
    TYPED_COMMAND_PROFILES,
    TYPED_COMMANDS,
    act_description,
    typed_description,
)


class CapabilityKind(StrEnum):
    TOOL = "tool"
    TEXT_SKILL = "text-skill"
    PEER = "peer"


class CapabilityEffect(StrEnum):
    ADVICE = "advice"
    READ_ONLY = "read-only"
    ENVIRONMENT_WRITE = "environment-write"


@dataclass(frozen=True, slots=True)
class Capability:
    capability_id: str
    kind: CapabilityKind
    description: str
    public_input_contract: str
    provenance: str
    body: str = ""
    cost: str = "charged to the shared episode budget"
    effect: CapabilityEffect = CapabilityEffect.ADVICE
    available_to: tuple[str, ...] = ("owner",)
    description_version: str = "public-capability@1"


@dataclass(frozen=True, slots=True)
class AdviceQuery:
    root_task: str
    explicit_request: str | None = None

    def text(self) -> str:
        return self.root_task if self.explicit_request is None else self.explicit_request


_STOPWORDS = frozenset(
    "a an and are as at be by can complete do for from in is it of on or task the "
    "this to use with you your".split()
)


def _terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold())) - _STOPWORDS


class CapabilityRegistry:
    def __init__(self, entries: tuple[Capability, ...]) -> None:
        if len({entry.capability_id for entry in entries}) != len(entries):
            raise ValueError("capability identities must be unique")
        self.entries = entries

    def visible(self, participant: str) -> tuple[Capability, ...]:
        return tuple(entry for entry in self.entries if participant in entry.available_to)

    def render(self, participant: str = "owner") -> str:
        return "Available capabilities (only these interfaces are executable):\n" + "\n".join(
            f"- {entry.capability_id}: {entry.description} "
            f"Input: {entry.public_input_contract}. Effect: {entry.effect.value}; {entry.cost}."
            for entry in self.visible(participant)
        )

    def text_advice(self, public_task: str, *, retrieve: bool) -> tuple[Capability, ...]:
        """Historical advice helper; clean execution uses the production skill library.

        Task/content overlap, not task-family labels, panel index, or evaluator data.

        Returning no advice is valid. Advisory text never adds/removes a legal action.
        """
        terms = _terms(public_task)
        return tuple(
            entry
            for entry in self.entries
            if entry.kind is CapabilityKind.TEXT_SKILL
            and (not retrieve or terms.intersection(_terms(entry.description)))
        )


def runtime_capabilities(
    mode: str,
    *,
    peers: tuple[str, ...] = (),
    skills: bool = False,
    scienceworld_profile: str = LEGACY_COMMAND_PROFILE,
) -> CapabilityRegistry:
    """Discover only implemented interfaces, never a benchmark expert policy."""
    entries = [
        Capability(
            "history",
            CapabilityKind.TOOL,
            "Read a page of this episode's public history: conversation contains the "
            "initial dialogue already supplied in the prompt; discussion contains newly "
            "generated owner messages; environment and tool_result contain execution "
            "observations. This tool does not search external sources or records",
            '{"kind":"history","archive":"conversation|environment|discussion|tool_result",'
            '"cursor":0,"limit":4} (limit 1 through 8)',
            "episode public archive",
            cost="one public tool read; the requesting model call uses the shared budget",
            effect=CapabilityEffect.READ_ONLY,
        )
    ]
    if mode == "webshop":
        for name, argument, description in (
            ("search", "query", "Submit a product query from the current page"),
            ("click", "target", "Activate an exact displayed link, option or control"),
        ):
            entries.append(
                Capability(
                    name,
                    CapabilityKind.TOOL,
                    description,
                    f"{name}[{argument}]",
                    "native simulator API",
                    effect=CapabilityEffect.ENVIRONMENT_WRITE,
                )
            )
    elif mode == "alfworld":
        entries.append(
            Capability(
                "act",
                CapabilityKind.TOOL,
                "Execute one command from the current public menu",
                "Action: concrete native command",
                "native simulator API",
                effect=CapabilityEffect.ENVIRONMENT_WRITE,
            )
        )
        descriptions = {
            "look": "Display the current surroundings using the native look command",
            "inventory": "List held objects using the native inventory command",
            "help": "Display the simulator's native command help",
        }
        entries.extend(
            Capability(
                name,
                CapabilityKind.TOOL,
                descriptions[name],
                f"{name}() with no arguments; equivalent to act with command {name}",
                "native simulator API",
                effect=CapabilityEffect.ENVIRONMENT_WRITE,
            )
            for name in ALFWORLD_OBSERVATION_CALLS
        )
    elif mode == "scienceworld":
        if scienceworld_profile not in {
            LEGACY_COMMAND_PROFILE,
            COMMAND_PROFILE,
            *TYPED_COMMAND_PROFILES,
        }:
            raise ValueError("unknown ScienceWorld command reference")
        entries.append(
            Capability(
                "act",
                CapabilityKind.TOOL,
                act_description(scienceworld_profile)
                if scienceworld_profile != LEGACY_COMMAND_PROFILE
                else "Send one textual command to the native ScienceWorld environment",
                "Action: concrete native command",
                "native simulator API",
                effect=CapabilityEffect.ENVIRONMENT_WRITE,
            )
        )
        if scienceworld_profile in TYPED_COMMAND_PROFILES:
            entries.extend(
                Capability(
                    name,
                    CapabilityKind.TOOL,
                    typed_description(name, scienceworld_profile),
                    f'{name}(target="literal object name")',
                    "literal native simulator command binding",
                    effect=CapabilityEffect.ENVIRONMENT_WRITE,
                    description_version=scienceworld_profile,
                )
                for name in TYPED_COMMANDS
            )
    for peer in peers:
        entries.append(
            Capability(
                peer,
                CapabilityKind.PEER,
                "Ask an optional advice-only peer; it cannot execute tools "
                "or submit on your behalf",
                f"Message to {peer}: followed by your verbatim request",
                "isolated peer conversation",
                cost="one peer model call from the shared budget",
            )
        )
    if skills:
        for name, description, arguments in (
            (
                "list_skills",
                "Discover the complete active skill catalog, including nonmatching skills",
                '"cursor":0,"limit":8',
            ),
            (
                "retrieve_skills",
                "Retrieve skills applicable to this public task using the declared library rule",
                "",
            ),
            (
                "read_skill",
                "View a complete active skill by ID; prioritize it within the skill context budget",
                '"skill_id":"visible ID"',
            ),
            (
                "invoke_skill",
                "Explicitly adopt a text skill by ID; view its instructions, "
                "without generating an answer or changing the environment",
                '"skill_id":"visible ID"',
            ),
        ):
            entries.append(
                Capability(
                    name,
                    CapabilityKind.TOOL,
                    description,
                    '{"kind":"skill","operation":"'
                    + name
                    + '"'
                    + ("," + arguments if arguments else "")
                    + "}",
                    "read-only active production skill library",
                    effect=CapabilityEffect.READ_ONLY,
                )
            )
    return CapabilityRegistry(tuple(entries))


def native_tool_definitions(
    mode: str,
    *,
    peers: tuple[str, ...] = (),
    skills: bool = False,
    natural_language: bool = False,
    owner_finish: bool = False,
    scienceworld_profile: str = LEGACY_COMMAND_PROFILE,
    corpus_search: bool = False,
) -> NativeTools:
    """Expose the existing public runtime interfaces through the model template.

    Peer access remains advice-only. Static tasks gain a submission envelope,
    not a search tool, external solver, answer chooser or extra model call.
    """
    definitions = []
    if corpus_search:
        if mode != "completion":
            raise ValueError("corpus search is a separate completion condition")
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": "corpus_search",
                    "description": "Search the frozen public Wikipedia corpus with your query. "
                    "Returns ranked passages, not an answer or another agent's advice.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    for capability in runtime_capabilities(
        mode, peers=peers, skills=skills, scienceworld_profile=scienceworld_profile
    ).visible("owner"):
        name = capability.capability_id
        if name == "history":
            properties = {
                "archive": {
                    "type": "string",
                    "enum": ["conversation", "environment", "discussion", "tool_result"],
                },
                "cursor": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
            }
        elif name == "list_skills":
            properties = {
                "cursor": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 32, "default": 8},
            }
        elif name in {"read_skill", "invoke_skill"}:
            properties = {"skill_id": {"type": "string"}}
        elif mode == "scienceworld" and name in TYPED_COMMANDS:
            properties = {"target": {"type": "string", "description": "Your literal object name."}}
        elif name == "retrieve_skills" or (
            mode == "alfworld" and name in ALFWORLD_OBSERVATION_CALLS
        ):
            properties = {}
        else:
            key = {"search": "query", "click": "target", "act": "command"}.get(name, "body")
            argument_description = {
                "query": "Your query text. A complete search[query] call is also accepted.",
                "target": "The exact displayed link, button or option label inside click[target], "
                "or its complete click[target] entry from the public menu.",
                "command": (
                    "One complete textual ScienceWorld command."
                    if mode == "scienceworld"
                    else "One complete native command from the current public menu, unchanged."
                ),
                "body": "Your verbatim request for advice, including any relevant public context.",
            }[key]
            properties = {key: {"type": "string", "description": argument_description}}
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": capability.description + ". " + capability.cost + ".",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": ["archive"]
                        if name == "history"
                        else []
                        if name == "list_skills"
                        else list(properties),
                        "additionalProperties": False,
                    },
                },
            }
        )
    if mode == "completion":
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": "submit_answer",
                    "description": (
                        "Send your next reply to the user. Your reply may include an "
                        "explanation or a clarification question as appropriate. "
                        "A direct chat reply is also accepted."
                        if natural_language
                        else "Submit your own final answer or complete source code, "
                        "not discussion. A direct chat answer is also accepted."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    if owner_finish:
        if mode != "scienceworld":
            raise ValueError("owner finish is a separate ScienceWorld interface condition")
        from .interactive_termination import OWNER_FINISH_INSTRUCTION

        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": OWNER_FINISH_INSTRUCTION,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tuple(definitions)


def generic_advisory_capabilities() -> CapabilityRegistry:
    return CapabilityRegistry(
        (
            Capability(
                "evidence-and-uncertainty@1",
                CapabilityKind.TEXT_SKILL,
                "Evidence, sources, questions, facts, uncertainty and conflicting observations",
                "public task and observations only",
                "project generic advice, 2026-09-05",
                "Use the available evidence to answer the task. Check uncertain claims "
                "when useful; distinguish observations from assumptions and use your "
                "judgment when evidence conflicts.",
            ),
            Capability(
                "problem-solving@1",
                CapabilityKind.TEXT_SKILL,
                "Solve a problem, calculate, reason, implement a function or complete a task",
                "public task and observations only",
                "project generic advice, 2026-09-05",
                "Break down the problem if useful, work through it, and check your proposed result "
                "against the original request. You may revise your approach "
                "or disregard this advice.",
            ),
        )
    )
