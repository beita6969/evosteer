"""Versioned, answer-free prompt registry for the direct reference track."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from .client import ChatTokenCounter
from .config import DirectBenchmark
from .context_budget import ContextBudget, newest_contiguous_history
from .webshop_prompt_policy import (
    PUBLISHED_WEBSHOP_EMPTY_STRUCTURED_MEMORY,
    PUBLISHED_WEBSHOP_INTERACTIVE_SYSTEMS,
    is_webshop_surface_grounded_profile,
    is_webshop_thought_free_profile,
    published_webshop_task_decomposition,
    webshop_step_operating_policy,
)


@dataclass(frozen=True, slots=True)
class StaticPromptInput:
    rendered_question: str


StaticPromptRenderer = Callable[[StaticPromptInput], tuple[dict[str, str], ...]]


class PromptKind(StrEnum):
    STATIC = "static"
    INTERACTIVE = "interactive"


@dataclass(frozen=True, slots=True)
class PromptProfile:
    profile_id: str
    kind: PromptKind
    renderer: Callable[..., tuple[dict[str, str], ...]]


@dataclass(frozen=True, slots=True)
class WebShopPublicCatalogCandidate:
    """One answer-independent product suggestion returned by a public-catalog tool.

    The schema has no field for evaluator goals, rewards, target labels, or expected actions.
    ``product_id`` is an ordinary public-catalog identity selected from the public instruction,
    not an evaluator designation. A suggestion is only navigation evidence: the policy must
    still issue native ``search``/``click`` actions and verify the live product page.
    """

    product_id: str
    title: str
    price: float
    suggested_search_query: str
    attributes: tuple[str, ...] = ()
    options: tuple[tuple[str, tuple[str, ...]], ...] = ()
    suggested_options: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_public_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse every public catalog field without silently dropping router inputs."""

        title = str(value.get("title") or "").strip()
        raw_attributes = value.get("attributes") or ()
        if not isinstance(raw_attributes, Sequence) or isinstance(raw_attributes, str):
            raise TypeError("public catalog attributes must be a sequence")
        attributes = tuple(
            dict.fromkeys(str(item).strip() for item in raw_attributes if str(item).strip())
        )[:24]

        raw_options = value.get("options") or {}
        if not isinstance(raw_options, Mapping):
            raise TypeError("public catalog options must be a mapping")
        option_rows = []
        for group, raw_values in sorted(raw_options.items(), key=lambda item: str(item[0])):
            group_name = str(group).strip()
            if (
                not group_name
                or not isinstance(raw_values, Sequence)
                or isinstance(raw_values, str)
            ):
                continue
            options = tuple(
                dict.fromkeys(str(item).strip() for item in raw_values if str(item).strip())
            )[:512]
            if options:
                option_rows.append((group_name, options))

        raw_suggestions = value.get("suggested_options") or {}
        if not isinstance(raw_suggestions, Mapping):
            raise TypeError("public catalog option recommendations must be a mapping")
        suggestions = tuple(
            (str(group).strip(), str(option).strip())
            for group, option in sorted(raw_suggestions.items(), key=lambda item: str(item[0]))
            if str(group).strip() and str(option).strip()
        )[:16]

        raw_price = value.get("price", math.nan)
        if isinstance(raw_price, bool) or not isinstance(raw_price, int | float | str):
            raise TypeError("public catalog price must be numeric")

        return cls(
            product_id=str(value.get("product_id") or ""),
            title=title,
            price=float(raw_price),
            suggested_search_query=str(value.get("suggested_search_query") or title),
            attributes=attributes,
            options=tuple(option_rows[:16]),
            suggested_options=suggestions,
        )

    def __post_init__(self) -> None:
        text_fields = (self.product_id, self.title, self.suggested_search_query)
        if any(not value.strip() or "\x00" in value for value in text_fields):
            raise ValueError("public catalog candidate text fields must be non-empty without NUL")
        if (
            len(self.product_id) > 128
            or len(self.title) > 1000
            or len(self.suggested_search_query) > 1000
        ):
            raise ValueError("public catalog candidate text exceeds its bounded prompt contract")
        if not math.isfinite(self.price) or self.price < 0:
            raise ValueError("public catalog candidate price must be finite and non-negative")
        if len(self.attributes) > 24 or len(self.options) > 16 or len(self.suggested_options) > 16:
            raise ValueError("public catalog candidate evidence exceeds its bounded schema")
        if len({group.casefold() for group, _ in self.options}) != len(self.options):
            raise ValueError("public catalog option groups must be unique")
        for value in (*self.attributes, *(group for group, _ in self.options)):
            if not value.strip() or "\x00" in value or len(value) > 300:
                raise ValueError("public catalog evidence text is invalid")
        if sum(len(values) for _, values in self.options) > 512:
            raise ValueError("public catalog candidate has too many option values")
        for _, values in self.options:
            if not values or len(values) > 512 or len(set(values)) != len(values):
                raise ValueError("public catalog option values must be non-empty and unique")
            if any(not value.strip() or "\x00" in value or len(value) > 300 for value in values):
                raise ValueError("public catalog option value is invalid")
        if len({group.casefold() for group, _ in self.suggested_options}) != len(
            self.suggested_options
        ):
            raise ValueError("public catalog option recommendations must have unique groups")
        option_groups = {group.casefold(): values for group, values in self.options}
        for group, value in self.suggested_options:
            if (
                not group.strip()
                or not value.strip()
                or "\x00" in group
                or "\x00" in value
                or len(group) > 300
                or len(value) > 300
                or group.casefold() not in option_groups
                or value not in option_groups[group.casefold()]
            ):
                raise ValueError(
                    "public catalog option recommendation must copy one declared group value"
                )

    def render(self, index: int) -> str:
        attributes = "; ".join(self.attributes) if self.attributes else "none listed"
        options = (
            "; ".join(f"{group}=[{', '.join(values)}]" for group, values in self.options)
            if self.options
            else "none listed"
        )
        recommendations = (
            "; ".join(f"{group}={value}" for group, value in self.suggested_options)
            if self.suggested_options
            else "none"
        )
        return (
            f"Candidate {index}:\n"
            f"- public product id: {self.product_id}\n"
            f"- exact-title search suggestion: {self.suggested_search_query}\n"
            f"- public title: {self.title}\n"
            f"- public price: ${self.price:.2f}\n"
            f"- public attributes: {attributes}\n"
            f"- public selectable options: {options}\n"
            f"- public configurator recommendations: {recommendations}"
        )


@dataclass(frozen=True, slots=True)
class PublicInteractiveTurn:
    """One completed transition containing only model-visible environment state."""

    visible_assistant_text: str
    action: str | None
    observation: str
    returned_observation: str | None = None
    returned_available_actions: tuple[str, ...] = ()
    available_actions_before: tuple[str, ...] = ()
    structured_memory_before: str | None = None
    structured_memory_after: str | None = None
    visible_thought: str | None = None
    memory_update_status: str | None = None

    def render_public_state_before(self, *, historical: bool = False) -> str:
        """Render the state and distinguish expired from current action surfaces."""

        if not self.available_actions_before:
            return self.observation
        label = (
            "HISTORICAL action surface before this past action (expired; do not choose from it now)"
            if historical
            else "Admissible actions before the action"
        )
        actions = "\n".join(f"- {item}" for item in self.available_actions_before)
        return f"{self.observation}\n\n{label}:\n{actions}"

    def render_public_state_after(self, *, historical: bool = False) -> str:
        """Render returned state with an explicit temporal action-surface label."""

        observation = self.returned_observation
        if observation is None:
            return self.observation
        if not self.returned_available_actions:
            return observation
        label = (
            "HISTORICAL action surface returned at that past step "
            "(expired; do not choose from it now)"
            if historical
            else "Admissible actions after the action"
        )
        actions = "\n".join(f"- {item}" for item in self.returned_available_actions)
        return f"{observation}\n\n{label}:\n{actions}"

    @property
    def public_state_before(self) -> str:
        return self.render_public_state_before()

    @property
    def public_state_after(self) -> str:
        return self.render_public_state_after()

    def render_structured_assistant(self, *, include_thought: bool = True) -> str:
        memory = self.structured_memory_after or self.structured_memory_before or "<not supplied>"
        action = self.action or "<candidate response was invalid>"
        if not include_thought:
            return f"Memory:\n{memory}\n\nAction:\n{action}"
        thought = self.visible_thought or "<not supplied>"
        return f"Memory:\n{memory}\n\nThought:\n{thought}\n\nAction:\n{action}"


def _static(system: str) -> StaticPromptRenderer:
    def render(value: StaticPromptInput) -> tuple[dict[str, str], ...]:
        if not value.rendered_question.strip():
            raise ValueError("rendered_question must be non-empty")
        return (
            {"role": "system", "content": system},
            {"role": "user", "content": value.rendered_question},
        )

    return render


_SHORT_CONTEXT = _static(
    "Answer from the supplied passages. End with `Final answer: <short answer>`."
)
_HOTPOT_FULL_CONTEXT = _static(
    "Answer the multi-hop question using all ten supplied context passages. "
    "End with `Final answer: <short answer>`."
)
_NATIVE_RELEASED_QA = _static(
    "Give only the final short answer using the supplied task text. "
    "End with `Final answer: <short answer>`."
)
_SHORT_CLOSED = _static("Give a concise answer. End with `Final answer: <short answer>`.")
_INTEGER_COT = _static("Reason step by step. End with `Final answer: <integer from 0 to 999>`.")
_INTEGER_COT_BOXED = _static(
    "Reason step by step. Put the final integer from 0 to 999 in `\\boxed{}`."
)
_BOXED_MATH = _static("Reason step by step, and put your final answer within \\boxed{}.")
_MC_LABEL = _static("Reason carefully and end with `Final answer: <A, B, C, or D>`.")
_MC_JSON = _static('Reason carefully and return exactly one JSON object such as {"answer":"C"}.')
_DIRECT_CODE = _static(
    "Complete the requested Python function. Return only executable Python source."
)
_EVALPLUS_MBPP_CODE = _static(
    "Solve the MBPP+ programming task. Return only complete executable Python source defining "
    "the requested function; do not include markdown fences, tests, or explanations."
)
_DIRECT_PATCH = _static("Fix the issue in the repository. Return only a unified git diff.")
_SKILLFLOW_REPOSITORY_AGENT = _static(
    "Fix a bug in the full repository using list_files, search_code, view_file, and edit_file. "
    "The evaluator submits the workspace diff at the end. Never edit tests."
)
_QWEN_DIRECT_REPOSITORY_AGENT = _static(
    "Fix the bug in the full repository using direct source navigation and exact string-replace "
    "tools. The evaluator submits the workspace diff. Do not use a learned skill, adapter, or "
    "second model to author edits. Never edit tests."
)
_QWEN_DIRECT_REPOSITORY_AGENT_RAW_TOOLS = _static(
    "Fix the bug in the full repository using direct source navigation and exact string-replace "
    "tools. Tool observations contain only the requested repository evidence, without a "
    "cross-turn progress ledger or issue-derived hints. Do not use a learned skill, adapter, or "
    "second model to author edits. Never edit tests."
)
_QWEN_DIRECT_READONLY_REPOSITORY_AGENT = _static(
    "Inspect the full repository with read-only source tools. When enough evidence is available, "
    "return exactly one unified git diff. The workspace cannot be edited during inspection. "
    "Tool observations contain only requested repository evidence, without a progress ledger, "
    "issue-derived hints, learned skill, adapter, or second model. Never modify tests."
)
_MIND2WEB_OFFLINE = _static(
    "Predict one Mind2Web native action from the supplied candidates and context. Output "
    "exactly `Action: <CLICK|TYPE|SELECT> <backend_node_id> <value>`."
)


STATIC_PROMPT_REGISTRY: dict[str, StaticPromptRenderer] = {
    "short-answer-with-context@1": _SHORT_CONTEXT,
    "hotpotqa-full-context@1": _HOTPOT_FULL_CONTEXT,
    "hotpotqa-raw-ten-passage@2": _HOTPOT_FULL_CONTEXT,
    "hotpotqa-skillflow-wire@1": _NATIVE_RELEASED_QA,
    "skillflow-released-question-native@1": _NATIVE_RELEASED_QA,
    "short-answer-closed-book@1": _SHORT_CLOSED,
    "integer-cot@1": _INTEGER_COT,
    "integer-cot-boxed@1": _INTEGER_COT_BOXED,
    "boxed-math@1": _BOXED_MATH,
    "multiple-choice-label@1": _MC_LABEL,
    "multiple-choice-json-answer@1": _MC_JSON,
    "direct-code@1": _DIRECT_CODE,
    "evalplus-mbpp-complete-code@1": _EVALPLUS_MBPP_CODE,
    "direct-patch@2": _DIRECT_PATCH,
    "skillflow-code-generation-repository-agent@1": _SKILLFLOW_REPOSITORY_AGENT,
    "qwen-direct-repository-agent@1": _QWEN_DIRECT_REPOSITORY_AGENT,
    "qwen-direct-repository-agent-raw-tools@1": _QWEN_DIRECT_REPOSITORY_AGENT_RAW_TOOLS,
    "qwen-direct-readonly-repository-agent@1": _QWEN_DIRECT_READONLY_REPOSITORY_AGENT,
    "mind2web-offline-action@2": _MIND2WEB_OFFLINE,
}


_LEGACY_STATIC_PROFILE = {
    DirectBenchmark.HOTPOT_QA: "hotpotqa-full-context@1",
    DirectBenchmark.TRIVIA_QA: "short-answer-with-context@1",
    DirectBenchmark.MUSIQUE: "short-answer-with-context@1",
    DirectBenchmark.NQ_OPEN: "short-answer-closed-book@1",
    DirectBenchmark.AIME_2026: "integer-cot@1",
    DirectBenchmark.MATH_HARD: "boxed-math@1",
    DirectBenchmark.MED_QA: "multiple-choice-label@1",
    DirectBenchmark.GPQA_DIAMOND: "multiple-choice-label@1",
    DirectBenchmark.HUMAN_EVAL: "direct-code@1",
    DirectBenchmark.SWE_BENCH: "qwen-direct-readonly-repository-agent@1",
}


def static_messages(
    benchmark: DirectBenchmark, rendered_question: str
) -> tuple[dict[str, str], ...]:
    try:
        profile_id = _LEGACY_STATIC_PROFILE[benchmark]
    except KeyError as exc:
        raise ValueError(f"{benchmark.value} is not a static direct benchmark") from exc
    return render_static_messages(profile_id, rendered_question)


def render_static_messages(profile_id: str, rendered_question: str) -> tuple[dict[str, str], ...]:
    try:
        profile = PROMPT_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown static prompt profile {profile_id}") from exc
    if profile.kind is not PromptKind.STATIC:
        raise ValueError(f"prompt profile {profile_id} is not static")
    return profile.renderer(StaticPromptInput(rendered_question))


_INTERACTIVE_SYSTEMS = {
    "webshop-step0-skill-owned-react-memory@1": (
        "This source message supplies model-visible public WebShop state to the Step-0 "
        "controller. It contains no task strategy or response-format policy; the retrieved skill "
        "supplies task behavior."
    ),
    "alfworld-step0-skill-owned-react-memory@1": (
        "This source message supplies model-visible public ALFWorld state to the Step-0 "
        "controller. It contains no task strategy or response-format policy; the retrieved skill "
        "supplies task behavior."
    ),
    "webshop-native-react@2": (
        "You are a WebShop agent. Use the current page and available actions to advance the "
        "shopping task. Use the public interaction history, then output exactly one final "
        "`Action: search[keywords]` "
        "or `Action: click[value]` command. Do not output JSON."
    ),
    "alfworld-native-react@2": (
        "You are an ALFWorld agent. Track location, inventory, observations, and admissible "
        "commands. Output exactly one final `Action: <native command>`."
    ),
    "webshop-native-react-v3@1": (
        "You are a WebShop agent. Plan from the public task, current page, and available "
        "actions. Respond with a concise `Thought:` followed by exactly one final "
        "`Action: search[...]` or `Action: click[...]` command. Do not output JSON."
    ),
    "alfworld-native-react-v3@1": (
        "You are an ALFWorld agent. Track only the public location, inventory, observation, "
        "admissible commands, and remaining steps. Respond with a concise `Thought:` followed "
        "by exactly one final `Action: <admissible command>`."
    ),
    "webshop-native-react-v4@1": (
        "You are a WebShop agent. Use only the public task, current page, available actions, "
        "and contiguous replay history. Respond with a concise `Thought:` and exactly one "
        "final `Action: search[...]` or `Action: click[...]` command using a currently visible "
        "native action string."
    ),
    "alfworld-native-react-v4@1": (
        "You are an ALFWorld agent. Track public location, inventory, observation, admissible "
        "commands, and remaining steps. Respond with a concise `Thought:` and exactly one "
        "final `Action: <currently admissible native command>`."
    ),
    "webshop-native-react-memory-v5@1": (
        "You are a WebShop agent with cumulative public-state memory. Update every Memory field "
        "from the task and environment feedback as a compact factual ledger: use one short line "
        "per field, preserve confirmed facts, mark unknown facts as unknown, and never copy the "
        "whole observation or action surface into Memory. Give one short Thought, then copy one "
        "native action exactly. Before acting, check that the action advances an unmet "
        "requirement. If the latest feedback did not change state, do not choose that exact action "
        "again; choose a different available action. Do not purchase until every mandatory "
        "product, option, and price constraint is confirmed. Respond only "
        "as `Memory:\n...\n\nThought:\n...\n\nAction:\nsearch[...]` or "
        "`Action:\nclick[...]`."
    ),
    "alfworld-native-react-memory-v5@1": (
        "You are an ALFWorld agent with cumulative public-state memory. Update every Memory field "
        "from the task and environment feedback as a compact factual ledger: use one short line "
        "per field, preserve confirmed facts, mark unknown facts as unknown, and never copy the "
        "whole observation or admissible-action list into Memory. Give one short Thought about "
        "the shortest unfinished subgoal, then copy exactly one currently admissible native "
        "action. Before acting, check that it advances a remaining subgoal. If the latest "
        "feedback did not change state, do not choose that exact action again; choose a different "
        "admissible action. When a current command can take the exact task target, take it now; "
        "open a closed current receptacle before leaving; never revisit a controller-listed "
        "exhausted location while an unvisited location remains. Respond only as "
        "`Memory:\n...\n\nThought:\n...\n\nAction:\n<native command>`."
    ),
    "webshop-native-react-memory-v6-state@1": (
        "You are a WebShop agent with a compact cumulative decision ledger. Update every Memory "
        "field only from the public task, pages, and action surfaces. Separate known violations "
        "from unknown attributes: unknown is not a violation. Keep the product-type/query core "
        "stable, compare inspected candidates, give one short Thought, and copy exactly one "
        "native action. Do not invent a click value. Respond only as "
        "`Memory:\n...\n\nThought:\n...\n\nAction:\nsearch[...]` or "
        "`Action:\nclick[...]`."
    ),
    "webshop-native-react-memory-v6@1": (
        "You are a WebShop agent with a compact cumulative decision ledger and a strict 10-action "
        "budget. Update every Memory field only from the public task, pages, and action surfaces. "
        "Separate known violations from unknown attributes: unknown is not a violation. Keep the "
        "product-type/query core stable, compare inspected candidates, reserve the final actions "
        "for option selection and purchase, give one short Thought, and copy exactly one native "
        "action. Do not invent a click value. Respond only as "
        "`Memory:\n...\n\nThought:\n...\n\nAction:\nsearch[...]` or "
        "`Action:\nclick[...]`."
    ),
    "webshop-native-react-memory-v7@1": (
        "You are a WebShop agent with a compact cumulative decision ledger and a strict 10-action "
        "budget. The block named CURRENT ACTION SURFACE is authoritative for exactly this turn; "
        "all action surfaces shown in history are expired. Except for `search[query]` when the "
        "current block contains the literal `search` capability, copy the complete Action from "
        "one current line exactly. Never reuse an action merely because it appeared in history. "
        "Do not output `click[buy now]` unless that exact current line is present; return from a "
        "description/detail subpage with a currently listed navigation action first. Preserve the "
        "stable query core, compare candidates, and reserve the final actions for options and "
        "purchase. Respond only as `Memory:\n...\n\nThought:\n...\n\nAction:\nsearch[...]` "
        "or `Action:\nclick[...]`."
    ),
    **PUBLISHED_WEBSHOP_INTERACTIVE_SYSTEMS,
    "alfworld-native-react-memory-v6@1": (
        "You are an ALFWorld agent with a compact cumulative public-state ledger. Update every "
        "Memory field only from the public task, observations, inventory, and admissible commands. "
        "Keep the target object in inventory while performing clean, heat, or cool; use the "
        "currently admissible transformation command directly, then carry the transformed object "
        "to its destination. Give one short Thought about the next unfinished subgoal and copy "
        "exactly one currently admissible native command. When a current command can take the "
        "exact task target, take it immediately; open a closed current receptacle before leaving; "
        "never revisit a controller-listed exhausted location while an unvisited location "
        "remains. Do not invent or rewrite a command. "
        "Respond only as `Memory:\n...\n\nThought:\n...\n\nAction:\n<native command>`."
    ),
    "alfworld-native-react-memory-v7@1": (
        "You are an ALFWorld agent with a compact cumulative public-state ledger. Follow the "
        "task-type-specific planning policy selected from train-only validation, update Memory "
        "only from public state, and copy exactly one currently admissible native command. "
        "The released natural-language annotation and simulator object namespace can use "
        "different aliases for a shaker. Never rewrite the authoritative task from a "
        "demonstration. If a seemingly correct placement leaves the environment NOT TERMINAL, "
        "treat that concrete instance as disproven and try another compatible current object; "
        "never declare the task complete from Memory alone."
    ),
    "scienceworld-native-react@2": (
        "You are a ScienceWorld agent. Plan across steps using the task, observation, and public "
        "native action syntax. Output one final `Action: <native command>`."
    ),
    "webshop-skillflow-action-only-v1@1": "",
    "alfworld-skillflow-action-only-v1@1": "",
}

_SOURCE_ACTION_ONLY_PROFILES = {
    "webshop-skillflow-action-only-v1@1",
    "alfworld-skillflow-action-only-v1@1",
}

_STRUCTURED_MEMORY_PROFILES = {
    "webshop-step0-skill-owned-react-memory@1",
    "alfworld-step0-skill-owned-react-memory@1",
    "webshop-native-react-memory-v5@1",
    "alfworld-native-react-memory-v5@1",
    "webshop-native-react-memory-v6-state@1",
    "webshop-native-react-memory-v6@1",
    "webshop-native-react-memory-v7@1",
    "webshop-native-react-memory-v8@1",
    "webshop-native-stateact-memory-v9@1",
    "webshop-native-stateact-memory-v10@1",
    "webshop-native-stateact-memory-v11@1",
    "webshop-native-stateact-memory-v13@1",
    "webshop-native-stateact-memory-v14@1",
    "webshop-native-stateact-memory-v15@1",
    "webshop-native-stateact-memory-v16@1",
    "webshop-native-stateact-memory-v17@1",
    "webshop-native-react-memory-v18@1",
    "alfworld-native-react-memory-v6@1",
    "alfworld-native-react-memory-v7@1",
}

_STEP_ZERO_SKILL_OWNED_PROFILES = frozenset(
    {
        "webshop-step0-skill-owned-react-memory@1",
        "alfworld-step0-skill-owned-react-memory@1",
    }
)

_EMPTY_STRUCTURED_MEMORY = {
    "webshop-native-react-memory-v5@1": (
        "Goal requirements:\n"
        "- product type:\n"
        "- required attributes:\n"
        "- maximum price:\n\n"
        "Queries tried:\n"
        "Products inspected:\n"
        "Current candidate:\n"
        "Selected options:\n"
        "Matched requirements:\n"
        "Unmet requirements:\n"
        "Ready to purchase: no"
    ),
    "alfworld-native-react-memory-v5@1": (
        "Goal:\n"
        "Current location:\n"
        "Inventory:\n"
        "Known object locations:\n"
        "Open/closed receptacles:\n"
        "Cleaned objects:\n"
        "Heated objects:\n"
        "Cooled objects:\n"
        "Completed subgoals:\n"
        "Remaining subgoals:\n"
        "Repeated/no-progress actions: none"
    ),
    "webshop-native-react-memory-v6-state@1": (
        "Hard constraints (verbatim):\n"
        "Product/query core:\n"
        "Maximum price:\n"
        "Queries tried and outcomes:\n"
        "Candidate ledger (candidate | price | matched | unknown | violations):\n"
        "Current candidate:\n"
        "Best non-violating candidate:\n"
        "Selected options:\n"
        "Known violations:\n"
        "Unknown constraints:\n"
        "Actions remaining / actions reserved for options+purchase:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-react-memory-v6@1": (
        "Hard constraints (verbatim):\n"
        "Product/query core:\n"
        "Maximum price:\n"
        "Queries tried and outcomes:\n"
        "Candidate ledger (candidate | price | matched | unknown | violations):\n"
        "Current candidate:\n"
        "Best non-violating candidate:\n"
        "Selected options:\n"
        "Known violations:\n"
        "Unknown constraints:\n"
        "Actions remaining / actions reserved for options+purchase:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-react-memory-v7@1": (
        "Hard constraints (verbatim):\n"
        "Product/query core:\n"
        "Maximum price:\n"
        "Queries tried and outcomes:\n"
        "Candidate ledger (candidate | price | matched | unknown | violations):\n"
        "Current candidate:\n"
        "Best non-violating candidate:\n"
        "Selected options:\n"
        "Known violations:\n"
        "Unknown constraints:\n"
        "Current surface class / intended exact action:\n"
        "Actions remaining / actions reserved for options+purchase:\n"
        "Ready to purchase: no"
    ),
    **PUBLISHED_WEBSHOP_EMPTY_STRUCTURED_MEMORY,
    "alfworld-native-react-memory-v6@1": (
        "Goal:\n"
        "Current location:\n"
        "Inventory / object in hand:\n"
        "Known object locations:\n"
        "Open receptacles:\n"
        "Target transformation / appliance:\n"
        "Transformed objects:\n"
        "Destination:\n"
        "Completed subgoals:\n"
        "Remaining subgoals in order:\n"
        "Last action / public result:\n"
        "Repeated or no-progress actions:\n"
        "Next admissible subgoal action:"
    ),
}

_WEBSHOP_TASK_DECOMPOSITION = (
    "1. Extract product type, every mandatory attribute, and maximum price.\n"
    "2. Search with the product type plus discriminating required attributes.\n"
    "3. Inspect candidates and compare product details and price.\n"
    "4. Select each mandatory visible option exactly.\n"
    "5. Purchase only after the memory confirms every mandatory constraint."
)

_ALFWORLD_TASK_DECOMPOSITIONS = {
    "pick_and_place": (
        "Locate the target object; take it; locate and open the destination if needed; put the "
        "object in/on the required destination."
    ),
    "pick_two_obj": (
        "Complete locate, take, and put for one required object, then independently complete the "
        "same subgoals for the second object; one placed object is not task completion."
    ),
    "pick_clean_then_place": (
        "Locate and take the target; reach the cleaning fixture and clean it; then reach/open the "
        "destination and put the cleaned object there."
    ),
    "pick_heat_then_place": (
        "Locate and take the target; reach/open the heating appliance and heat it; then reach/open "
        "the destination and put the heated object there."
    ),
    "pick_cool_then_place": (
        "Locate and take the target; reach/open the cooling appliance and cool it; then reach/open "
        "the destination and put the cooled object there."
    ),
    "look_at_obj": (
        "Locate and take the requested object if needed; locate the required light source; then "
        "examine the requested object with that light source."
    ),
}

_ALFWORLD_TASK_DECOMPOSITIONS_V6 = {
    "pick_and_place": (
        "Locate the target; take it; navigate to and open the destination only if its current "
        "surface requires opening; then use the listed move command."
    ),
    "pick_two_obj": (
        "Finish locate, take, and move for the first required object, record that placement, then "
        "repeat for a distinct second object. Do not treat one placement as completion."
    ),
    "pick_clean_then_place": (
        "Take the target and keep it in inventory; go to the sinkbasin; execute the listed "
        "`clean <object> with <sinkbasin>` command directly; then carry it to the destination and "
        "use the listed move command."
    ),
    "pick_heat_then_place": (
        "Take the target and KEEP IT IN INVENTORY; go to the microwave; execute the listed "
        "`heat <object> with <microwave>` command directly while holding it; then carry it to the "
        "destination and use the listed move command. Do NOT move the object into the microwave "
        "or cycle the microwave open/closed before heating. Open it only when the initial public "
        "observation says the target is inside and must first be retrieved."
    ),
    "pick_cool_then_place": (
        "Take the target and keep it in inventory; go to the fridge; execute the listed "
        "`cool <object> with <fridge>` command directly; then carry it to the destination and use "
        "the listed move command."
    ),
    "look_at_obj": (
        "Locate and take the requested object if needed; locate the required lamp; then copy the "
        "listed use/examine command that completes the task."
    ),
}


def _task_decomposition(profile_id: str, task_type: str | None) -> str | None:
    if profile_id.startswith(("webshop-native-react-memory-", "webshop-native-stateact-memory-")):
        if profile_id == "webshop-native-react-memory-v5@1":
            return _WEBSHOP_TASK_DECOMPOSITION
        if profile_id == "webshop-native-react-memory-v7@1":
            return (
                "1. Copy every hard constraint and preserve a short product/query core.\n"
                "2. On a current search surface, issue one grounded search with that core.\n"
                "3. On current results, inspect one exact currently listed product ID and record "
                "public evidence; never click a product ID retained only in history.\n"
                "4. Compare candidates without treating descriptive unknowns as violations.\n"
                "5. On a detail subpage without `click[buy now]`, use an exact current navigation "
                "action to return to the candidate before committing.\n"
                "6. Select each required current option, and buy only when the exact current "
                "`click[buy now]` line is present."
            )
        published_decomposition = published_webshop_task_decomposition(profile_id)
        if published_decomposition is not None:
            return published_decomposition
        return (
            "1. Copy the hard constraints and preserve a short product/query core.\n"
            "2. Search once with that core; refine only when feedback gives a concrete reason.\n"
            "3. Inspect and compare candidates, recording price, matches, unknowns, and known "
            "violations separately.\n"
            "4. Commit to the best candidate with no known hard violation before the reserved "
            "purchase actions.\n"
            "5. Confirm every selectable option and price; descriptive unknowns alone do not "
            "forbid purchase; then buy."
        )
    if profile_id == "alfworld-native-react-memory-v6@1":
        return _ALFWORLD_TASK_DECOMPOSITIONS_V6.get(task_type or "")
    return _ALFWORLD_TASK_DECOMPOSITIONS.get(task_type or "")


def _step_operating_policy(
    profile_id: str,
    remaining_steps: int | None,
    available_actions: tuple[str, ...],
    prior_actions: tuple[str, ...] = (),
) -> str | None:
    webshop_policy = webshop_step_operating_policy(
        profile_id,
        remaining_steps,
        available_actions,
        prior_actions,
    )
    if webshop_policy is not None:
        return webshop_policy
    if profile_id in {
        "alfworld-native-react-memory-v5@1",
        "alfworld-native-react-memory-v6@1",
    }:
        return (
            "The next Action must be copied from `Admissible actions now`. Take a visible exact "
            "task target immediately, open the current closed receptacle before exploring, and "
            "prefer unvisited locations over controller-listed exhausted locations. For "
            "clean/heat/cool, keep the object in inventory until the listed transformation "
            "command is executed; after transformation, navigate directly toward the destination."
        )
    return None


def _effective_interactive_profile(profile_id: str, task_type: str | None) -> str:
    """Resolve the frozen ALFWorld v7 policy from the public task type.

    The final-disjoint train holdout showed a large, task-specific heat improvement from the
    v6 recipe and demonstrations, while the same prompt regressed several non-heat task types.
    V7 therefore preserves the validated v5 behavior outside heat instead of applying a global
    prompt change.  The routing input is part of ALFWorld's public environment state.
    """

    if profile_id != "alfworld-native-react-memory-v7@1":
        return profile_id
    if task_type not in _ALFWORLD_TASK_DECOMPOSITIONS:
        raise ValueError("ALFWorld v7 prompt requires a known public task type")
    if task_type == "pick_heat_then_place":
        return "alfworld-native-react-memory-v6@1"
    return "alfworld-native-react-memory-v5@1"


def render_interactive_messages(
    profile_id: str,
    *,
    task: str,
    current_observation: str,
    history: tuple[tuple[str | None, str] | PublicInteractiveTurn, ...] = (),
    history_window_steps: int | None = None,
    history_maximum_characters: int | None = 30_000,
    demonstrations: tuple[str, ...] = (),
    structured_memory: str | None = None,
    remaining_steps: int | None = None,
    available_actions: tuple[str, ...] = (),
    history_token_counter: ChatTokenCounter | None = None,
    context_length: int | None = None,
    output_reserve: int | None = None,
    enable_thinking: bool | None = None,
    task_type: str | None = None,
    webshop_catalog_candidates: tuple[WebShopPublicCatalogCandidate, ...] = (),
) -> tuple[dict[str, str], ...]:
    try:
        profile = PROMPT_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown interactive prompt profile {profile_id}") from exc
    if profile.kind is not PromptKind.INTERACTIVE:
        raise ValueError(f"prompt profile {profile_id} is not interactive")
    effective_profile_id = _effective_interactive_profile(profile_id, task_type)
    system = _INTERACTIVE_SYSTEMS[effective_profile_id]
    if history_maximum_characters is not None and history_maximum_characters <= 0:
        raise ValueError("interactive history character budget must be positive")
    if remaining_steps is not None and remaining_steps <= 0:
        raise ValueError("remaining_steps must be positive when supplied")
    if webshop_catalog_candidates and not effective_profile_id.startswith("webshop-"):
        raise ValueError("public WebShop catalog candidates require a WebShop prompt profile")
    if len(webshop_catalog_candidates) > 8:
        raise ValueError("public WebShop catalog context supports at most eight candidates")
    catalog_context = "\n\n".join(
        item.render(index) for index, item in enumerate(webshop_catalog_candidates, start=1)
    )
    selected = history if history_window_steps is None else history[-history_window_steps:]
    normalized = tuple(
        item
        if isinstance(item, PublicInteractiveTurn)
        else PublicInteractiveTurn(
            visible_assistant_text=(
                f"Action: {item[0]}"
                if item[0] is not None
                else "Action: <candidate response was invalid>"
            ),
            action=item[0],
            observation=item[1],
        )
        for item in selected
    )
    structured = effective_profile_id in _STRUCTURED_MEMORY_PROFILES
    if structured_memory is not None and not structured:
        raise ValueError("structured memory is only valid for a memory prompt profile")
    retained = list(normalized)
    if history_maximum_characters is not None:
        retained = list(
            retain_contiguous_history_suffix(
                normalized,
                base_characters=(
                    len(task)
                    + len(current_observation)
                    + (
                        0
                        if effective_profile_id in _STEP_ZERO_SKILL_OWNED_PROFILES
                        else len(structured_memory or "")
                        + len(catalog_context)
                        + sum(len(item) for item in demonstrations)
                    )
                ),
                maximum_characters=history_maximum_characters,
            )
        )

    def build(selected_history: tuple[PublicInteractiveTurn, ...]) -> tuple[dict[str, str], ...]:
        if profile_id in _SOURCE_ACTION_ONLY_PROFILES:
            if demonstrations:
                raise ValueError("source action-only prompts forbid demonstrations")
            if not available_actions:
                raise ValueError("source action-only prompts require the native action surface")
            return (
                {
                    "role": "user",
                    "content": _render_skillflow_source_prompt(
                        profile_id,
                        task=task,
                        current_observation=current_observation,
                        available_actions=available_actions,
                        history=selected_history,
                    ),
                },
            )
        transcript: list[str] = []
        if demonstrations and effective_profile_id not in _STEP_ZERO_SKILL_OWNED_PROFILES:
            transcript.append(
                "TRAINING DEMONSTRATION BOUNDARY: every demonstration below is an example only. "
                "Its task objects, attributes, locations, and destinations are never facts about "
                "the authoritative current task. Do not copy a demonstration Goal into current "
                "Memory."
            )
            transcript.extend(
                f"BEGIN TRAINING DEMONSTRATION\n{item}\nEND TRAINING DEMONSTRATION"
                for item in demonstrations
            )
        if effective_profile_id in _STEP_ZERO_SKILL_OWNED_PROFILES:
            transcript.append(f"AUTHORITATIVE CURRENT TASK:\nTask: {task}")
        else:
            transcript.append(
                "AUTHORITATIVE CURRENT TASK (not a demonstration):\n"
                f"Task: {task}\n"
                "Current-task identity rule: preserve the exact current product/object and "
                "destination terms in Memory. If a demonstration uses different entities, ignore "
                "them."
            )
        if catalog_context and effective_profile_id not in _STEP_ZERO_SKILL_OWNED_PROFILES:
            catalog_note = (
                "PUBLIC CATALOG LOOKUP TOOL OUTPUT (answer-independent navigation suggestions; "
                "not evaluator truth and not proof that a product satisfies the task):\n"
                + catalog_context
            )
            catalog_note += (
                "\nThe candidates are ordered by answer-independent public-constraint fit. "
                "Begin with Candidate 1; move to a later candidate only when live page evidence "
                "proves a hard violation or the exact-title result is absent. Tool-use rule: "
                "compare candidates only against the authoritative public task. To use one, issue "
                "its exact-title search suggestion through a native `search[...]` action, wait for "
                "the live result surface, then click only a currently listed product and select "
                "every task-required option before buying."
            )
            transcript.append(catalog_note)
        if remaining_steps is not None:
            transcript.append(f"Remaining action budget: {remaining_steps}")
        if structured:
            if effective_profile_id not in _STEP_ZERO_SKILL_OWNED_PROFILES:
                decomposition = _task_decomposition(effective_profile_id, task_type)
                if decomposition is None:
                    raise ValueError("ALFWorld structured-memory prompt requires a known task type")
                transcript.append(f"Task decomposition:\n{decomposition}")
                transcript.append(
                    "Accumulated structured memory before this action:\n"
                    + (
                        structured_memory
                        or _EMPTY_STRUCTURED_MEMORY.get(
                            effective_profile_id,
                            "No environment transition has been observed yet.",
                        )
                    )
                )
            if effective_profile_id not in _STEP_ZERO_SKILL_OWNED_PROFILES:
                operating_policy = _step_operating_policy(
                    effective_profile_id,
                    remaining_steps,
                    available_actions,
                    tuple(item.action for item in selected_history if item.action is not None),
                )
                if operating_policy is not None:
                    transcript.append(f"Current decision policy:\n{operating_policy}")
        surface_grounded = (
            is_webshop_surface_grounded_profile(effective_profile_id)
            or effective_profile_id == "webshop-step0-skill-owned-react-memory@1"
        )
        for item in selected_history:
            transcript.append(
                "State before action: "
                + item.render_public_state_before(historical=surface_grounded)
            )
            if effective_profile_id in _STEP_ZERO_SKILL_OWNED_PROFILES:
                prior_action = item.action or "<candidate response was invalid>"
                prior_thought = item.visible_thought or "<not supplied>"
                transcript.append(
                    f"Prior public decision:\nThought: {prior_thought}\nAction: {prior_action}"
                )
            else:
                transcript.append(
                    item.render_structured_assistant(
                        include_thought=(not is_webshop_thought_free_profile(effective_profile_id))
                    )
                    if structured
                    else item.visible_assistant_text
                )
            transcript.append(
                "Environment returned state: "
                + item.render_public_state_after(historical=surface_grounded)
            )
        rendered_current = current_observation
        if structured and available_actions:
            actions = "\n".join(f"- {item}" for item in available_actions)
            label = (
                "CURRENT ACTION SURFACE (authoritative for this turn; choose only here)"
                if surface_grounded
                else "Admissible actions now"
            )
            rendered_current = f"{current_observation}\n\n{label}:\n{actions}"
        transcript.append(f"Observation: {rendered_current}")
        if structured:
            transcript.append(
                "FINAL AUTHORITATIVE CURRENT-TASK ANCHOR (overrides every demonstration and "
                f"historical Memory field): {task}"
            )
            thought_free = is_webshop_thought_free_profile(effective_profile_id)
            if effective_profile_id in _STEP_ZERO_SKILL_OWNED_PROFILES:
                instruction = (
                    "The source interface adds no task-solving or response-format policy; task "
                    "behavior and the decision wire are supplied by the retrieved skill and "
                    "Step-0 controller."
                )
            else:
                instruction = (
                    "Before choosing the action: advance one remaining subgoal; if the latest "
                    "feedback did not change state, choose a different available/admissible "
                    "action; copy the chosen native action string exactly; keep every Memory field "
                    "concise and factual; if the environment says NOT TERMINAL, retain at least "
                    "one unfinished subgoal and do not claim completion; "
                    + (
                        "then output Memory and one Action only, with no Thought section."
                        if thought_free
                        else "then output one Thought and one Action."
                    )
                )
            if is_webshop_surface_grounded_profile(effective_profile_id):
                instruction += (
                    " First ignore every HISTORICAL surface, classify the CURRENT surface, and "
                    "write the intended exact current action in Memory. `search[query]` is allowed "
                    "only when the literal current capability is `search`; every click must match "
                    "one complete current line. Never output Buy Now when that line is absent."
                )
                if effective_profile_id == "webshop-native-react-memory-v18@1":
                    instruction += (
                        " Preserve controller-owned prior queries and rejected products. Never "
                        "reopen a rejected product or repeat an identical search. Keep price "
                        "limits and dollar amounts out of search queries. A visible "
                        "over-budget price is already a hard violation. When a product fails, "
                        "do not classify it as a type violation merely because its title says "
                        "accessory, case, band, or protector when that title also matches rare "
                        "requested attributes or exposes a required option. "
                        "Use the current < Prev line to recover its result page rather than Back "
                        "to Search; when every viable current result is exhausted, use the "
                        "current Next > line if present. Never reopen an information tab already "
                        "listed as visited for the current product. Reuse its controller-recorded "
                        "persistent public evidence after returning from the tab. A mandatory "
                        "descriptive attribute should use positive title, Description, or Features "
                        "evidence when available. After both tabs have been checked, missing exact "
                        "wording is uncertainty rather than a confirmed violation: if type, price, "
                        "and required options remain compatible, buy instead of looping."
                    )
                elif effective_profile_id in {
                    "webshop-native-react-memory-v8@1",
                    "webshop-native-stateact-memory-v9@1",
                    "webshop-native-stateact-memory-v10@1",
                    "webshop-native-stateact-memory-v11@1",
                    "webshop-native-stateact-memory-v13@1",
                    "webshop-native-stateact-memory-v14@1",
                    "webshop-native-stateact-memory-v15@1",
                    "webshop-native-stateact-memory-v16@1",
                    "webshop-native-stateact-memory-v17@1",
                }:
                    instruction += (
                        " Keep Memory to the supplied compact fields; record only products "
                        "actually opened, not every result. Follow the short search → reasonable "
                        "product → required options → Buy Now pipeline and do not reopen a "
                        "completed stage."
                    )
            elif effective_profile_id.startswith("webshop-native-react-memory-v6"):
                instruction += (
                    " Update the candidate ledger and remaining/reserved action counts before the "
                    "Action. A descriptive unknown is not a known violation."
                )
            transcript.append(instruction)
        return (
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(transcript)},
        )

    token_budget_values = (
        history_token_counter,
        context_length,
        output_reserve,
        enable_thinking,
    )
    if any(item is not None for item in token_budget_values):
        if (
            history_token_counter is None
            or context_length is None
            or output_reserve is None
            or enable_thinking is None
        ):
            raise ValueError("interactive token budget identity is incomplete")
        retained = list(
            newest_contiguous_history(
                render=build,
                history=tuple(retained),
                counter=history_token_counter,
                budget=ContextBudget(
                    context_length=context_length,
                    output_reserve=output_reserve,
                    enable_thinking=enable_thinking,
                ),
            )
        )
    return build(tuple(retained))


def is_source_action_only_profile(profile_id: str) -> bool:
    return profile_id in _SOURCE_ACTION_ONLY_PROFILES


def is_structured_memory_profile(profile_id: str) -> bool:
    return profile_id in _STRUCTURED_MEMORY_PROFILES


def _render_skillflow_source_prompt(
    profile_id: str,
    *,
    task: str,
    current_observation: str,
    available_actions: tuple[str, ...],
    history: tuple[PublicInteractiveTurn, ...],
) -> str:
    if profile_id == "webshop-skillflow-action-only-v1@1":
        actions = "\n".join(f"'{_webshop_source_action(item)}'," for item in available_actions)
        if not history:
            return (
                "You are an expert autonomous agent operating in the WebShop e-commerce "
                "environment.\n"
                f"Your task is to: {task}.\n"
                f"Your current observation is: {current_observation}.\n"
                "Your admissible actions of the current situation are:\n[\n"
                f"{actions}\n].\n\n"
                "Now it's your turn to take one action for the current step.\n"
                "Return exactly one executable action string in the form search[keywords] or "
                "click[value].\n"
                "For click actions, copy one value from the admissible action list exactly. "
                "Do not repeat these instructions.\n"
            )
        action_history = "\n".join(
            f"[Observation {index}: '{item.observation}', Action {index}: "
            f"'{item.action or '<invalid>'}', Result {index}: "
            f"'{item.public_state_after}']"
            for index, item in enumerate(history, 1)
        )
        step_count = len(history)
        return (
            "You are an expert autonomous agent operating in the WebShop e-commerce "
            "environment.\n"
            f"Your task is to: {task}.\n"
            f"Prior to this step, you have already taken {step_count} step(s). Below are the "
            f"most recent {step_count} observations and the corresponding actions you took: "
            f"{action_history}\n"
            f"You are now at step {step_count + 1} and your current observation is: "
            f"{current_observation}.\n"
            "Your admissible actions of the current situation are:\n[\n"
            f"{actions}\n].\n\n"
            "Return exactly one executable action string in the form search[keywords] or "
            "click[value].\n"
            "For click actions, copy one value from the admissible action list exactly. "
            "Do not repeat these instructions.\n"
        )
    if profile_id != "alfworld-skillflow-action-only-v1@1":
        raise ValueError("unknown source action-only prompt")
    actions = "\n".join(f"  '{item}'," for item in available_actions if item != "help")
    example = (
        "Action format examples:\n"
        "> go to cabinet 1\n"
        "> take apple 1 from countertop 1\n"
        "> open fridge 1\n"
        "> move apple 1 to fridge 1\n"
        "> heat apple 1 with microwave 1\n"
        "> clean mug 1 with sinkbasin 1\n"
        "> cool potato 1 with fridge 1\n"
        "> move plate 1 to countertop 1\n"
        "> examine shelf 1\n"
    )
    if not history:
        return (
            "You are an expert agent operating in the ALFRED Embodied Environment.\n"
            f"{example}"
            f"Your task is to: {task}\n"
            f"Your current observation is: {current_observation}\n"
            "Your admissible actions of the current situation are: "
            f"[{actions}].\n\n"
            "Now it's your turn to take an action. Pick exactly one action from the "
            "admissible actions list.\n"
            "Output ONLY the action you choose. No explanation, no reasoning, just the action.\n"
        )
    action_history = "\n".join(
        f"[Step {index}: Action: '{item.action or '<invalid>'}' -> Result: "
        f"'{item.public_state_after}']"
        for index, item in enumerate(history, 1)
    )
    step_count = len(history)
    return (
        "You are an expert agent operating in the ALFRED Embodied Environment. Your task is "
        f"to: {task}\n{example}"
        f"Prior to this step, you have already taken {step_count} step(s). Below are the most "
        f"recent {step_count} observations and the corresponding actions you took: "
        f"{action_history}\n"
        f"You are now at step {step_count + 1} and your current observation is: "
        f"{current_observation}\n"
        "Your admissible actions of the current situation are: "
        f"[{actions}].\n\n"
        "Now it's your turn to take an action. Pick exactly one action from the admissible "
        "actions list.\n"
        "Output ONLY the action you choose. No explanation, no reasoning, just the action.\n"
    )


def _webshop_source_action(value: str) -> str:
    return "search[<your query>]" if value == "search" else value


def retain_contiguous_history_suffix(
    history: tuple[PublicInteractiveTurn, ...],
    *,
    base_characters: int,
    maximum_characters: int,
) -> tuple[PublicInteractiveTurn, ...]:
    """Keep only a chronological, contiguous suffix of public state."""

    if base_characters < 0 or maximum_characters <= 0:
        raise ValueError("interactive history character counts are invalid")
    if base_characters > maximum_characters:
        return ()
    retained_reversed: list[PublicInteractiveTurn] = []
    used = base_characters
    for item in reversed(history):
        size = (
            len(item.visible_assistant_text)
            + len(item.observation)
            + len(item.returned_observation or "")
            + sum(len(action) for action in item.returned_available_actions)
        )
        if used + size > maximum_characters:
            break
        retained_reversed.append(item)
        used += size
    return tuple(reversed(retained_reversed))


def react_messages(
    benchmark: DirectBenchmark,
    *,
    task: str,
    observation: str,
    history: tuple[tuple[str, str], ...] = (),
) -> tuple[dict[str, str], ...]:
    profile_id = {
        DirectBenchmark.WEB_SHOP: "webshop-native-react@2",
        DirectBenchmark.ALF_WORLD: "alfworld-native-react@2",
        DirectBenchmark.SCIENCE_WORLD: "scienceworld-native-react@2",
    }.get(benchmark)
    if profile_id is None:
        raise ValueError(f"{benchmark.value} has no native ReAct profile")
    return render_interactive_messages(
        profile_id,
        task=task,
        current_observation=observation,
        history=tuple((action, item) for action, item in history),
    )


PROMPT_REGISTRY: dict[str, PromptProfile] = {
    **{
        profile_id: PromptProfile(profile_id, PromptKind.STATIC, renderer)
        for profile_id, renderer in STATIC_PROMPT_REGISTRY.items()
    },
    **{
        profile_id: PromptProfile(profile_id, PromptKind.INTERACTIVE, render_interactive_messages)
        for profile_id in _INTERACTIVE_SYSTEMS
    },
}


__all__ = [
    "PROMPT_REGISTRY",
    "PublicInteractiveTurn",
    "is_source_action_only_profile",
    "is_structured_memory_profile",
    "react_messages",
    "render_interactive_messages",
    "render_static_messages",
    "retain_contiguous_history_suffix",
    "static_messages",
]
