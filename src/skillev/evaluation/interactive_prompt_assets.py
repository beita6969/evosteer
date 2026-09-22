"""Train-only public demonstrations for benchmark-native interactive agents."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path

import yaml

from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.interactive_tasks import (
    action_matches_public_surface,
)

_MINIMUM_EXAMPLES = {"alfworld": 2, "webshop": 4}
_CONSTRAINT_TAG_SELECTION = "constraint-tags@1"
_LEXICAL_BM25_SELECTION = "lexical-bm25@1"
_LEXICAL_BM25_TOP1_SELECTION = "lexical-bm25-top1@1"


@dataclass(frozen=True, slots=True)
class InteractiveReplayStep:
    observation_before: str
    available_actions_before: tuple[str, ...]
    action: str
    observation_after: str
    available_actions_after: tuple[str, ...]
    memory: str | None = None
    thought: str | None = None


@dataclass(frozen=True, slots=True)
class InteractivePromptExample:
    example_id: str
    source_entry_id: str
    source_split: str
    task: str
    turns: tuple[str, ...]
    task_type: str | None = None
    replay_steps: tuple[InteractiveReplayStep, ...] = ()


@dataclass(frozen=True, slots=True)
class InteractivePromptAsset:
    benchmark: str
    asset_id: str
    source_kind: str
    source_split: str
    source_revision: str
    reasoning_mode: str
    examples: tuple[InteractivePromptExample, ...]

    def render_demonstrations(
        self,
        *,
        task: str | None = None,
        task_type: str | None = None,
        structured_memory: bool = False,
        include_thought: bool = True,
        selection_profile: str = _CONSTRAINT_TAG_SELECTION,
        compact_action_surfaces: bool | None = None,
    ) -> tuple[str, ...]:
        examples = self._select_examples(
            task=task,
            task_type=task_type,
            selection_profile=selection_profile,
        )
        return tuple(
            f"Demonstration (training example):\nTask: {example.task}\n"
            + (
                _render_structured_memory_replay(
                    self.benchmark,
                    example,
                    compact_action_surfaces=(
                        self.asset_id.endswith("memory-v6")
                        if compact_action_surfaces is None
                        else compact_action_surfaces
                    ),
                    include_thought=include_thought,
                )
                if structured_memory
                else "\n".join(example.turns)
            )
            for example in examples
        )

    def _select_examples(
        self,
        *,
        task: str | None,
        task_type: str | None,
        selection_profile: str,
    ) -> tuple[InteractivePromptExample, ...]:
        if task_type is not None:
            selected = tuple(item for item in self.examples if item.task_type == task_type)
            if not selected:
                raise ValueError("no train demonstration matches the ALFWorld task type")
            return selected
        if task is None or self.benchmark != "webshop":
            return self.examples
        if selection_profile in {
            _LEXICAL_BM25_SELECTION,
            _LEXICAL_BM25_TOP1_SELECTION,
        }:
            return _select_webshop_examples_bm25(
                task,
                self.examples,
                limit=1 if selection_profile == _LEXICAL_BM25_TOP1_SELECTION else 2,
            )
        if selection_profile != _CONSTRAINT_TAG_SELECTION:
            raise ValueError("unknown interactive demonstration selection profile")
        target_tags = _webshop_constraint_tags(task)
        ranked = sorted(
            enumerate(self.examples),
            key=lambda item: (
                -len(target_tags.intersection(_webshop_constraint_tags(item[1].task))),
                item[0],
            ),
        )
        return tuple(item for _index, item in ranked[:2])

    def validate_final_isolation(self, final_ids: frozenset[str]) -> None:
        overlap = final_ids.intersection(
            identity
            for item in self.examples
            for identity in (item.example_id, item.source_entry_id)
        )
        if overlap:
            raise ValueError("interactive prompt examples overlap final population")


def load_interactive_prompt_asset(path: Path) -> InteractivePromptAsset:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format") not in {
        "skillev-public-interactive-prompt@1",
        "skillev-public-interactive-prompt@2",
    }:
        raise ValueError("invalid interactive prompt asset")
    if raw["format"] == "skillev-public-interactive-prompt@2":
        return _load_source_proven_asset(raw)
    examples = raw.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("interactive prompt asset requires examples")
    parsed: list[InteractivePromptExample] = []
    for item in examples:
        if not isinstance(item, dict) or not isinstance(item.get("turns"), list):
            raise ValueError("interactive prompt example is invalid")
        if not item["turns"] or any(not isinstance(turn, dict) for turn in item["turns"]):
            raise ValueError("interactive prompt example turns are invalid")
        turns = tuple(
            f"Observation: {turn['observation']}\n{turn['assistant']}" for turn in item["turns"]
        )
        parsed.append(
            InteractivePromptExample(
                str(item["example_id"]),
                str(item["example_id"]),
                str(raw["source_split"]),
                str(item["task"]),
                turns,
            )
        )
    benchmark = str(raw["benchmark"])
    if len(parsed) < _MINIMUM_EXAMPLES.get(benchmark, 1):
        raise ValueError("interactive prompt asset has insufficient train-only examples")
    return InteractivePromptAsset(
        benchmark=benchmark,
        asset_id=f"legacy-{benchmark}-synthetic",
        source_kind="synthetic-unverified",
        source_split=str(raw["source_split"]),
        source_revision="unverified",
        reasoning_mode=str(raw["reasoning_mode"]),
        examples=tuple(parsed),
    )


def _load_source_proven_asset(raw: dict[str, object]) -> InteractivePromptAsset:
    benchmark = _text(raw.get("benchmark"), "benchmark")
    asset_id = _text(raw.get("asset_id"), "asset ID")
    source_kind = _text(raw.get("source_kind"), "source kind")
    if source_kind != "official-train-replay":
        raise ValueError("formal prompt asset must be an official train replay")
    if raw.get("source_split") != "train":
        raise ValueError("formal prompt asset must be train-only")
    examples = raw.get("examples")
    if not isinstance(examples, list):
        raise ValueError("formal prompt examples must be an array")
    parsed: list[InteractivePromptExample] = []
    task_types: set[str] = set()
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("formal prompt example must be an object")
        steps = example.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("formal prompt example must contain replay steps")
        turns: list[str] = []
        replay_steps: list[InteractiveReplayStep] = []
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("formal prompt step must be an object")
            before = _text(step.get("observation_before"), "observation before")
            after = _text(step.get("observation_after"), "observation after")
            action = _text(step.get("action"), "action")
            available = step.get(
                "available_actions_before",
                step.get("admissible_commands_before"),
            )
            if not isinstance(available, list) or any(type(item) is not str for item in available):
                raise ValueError("formal prompt action surface is invalid")
            if not action_matches_public_surface(
                DirectBenchmark(benchmark),
                action,
                tuple(available),
            ):
                raise ValueError("formal prompt action was not available in its replay")
            turns.append(
                f"Observation: {before}\nThought: Use a currently available native action.\n"
                f"Action: {action}\nNext observation: {after}"
            )
            after_available = step.get(
                "available_actions_after",
                step.get("admissible_commands_after", []),
            )
            if not isinstance(after_available, list) or any(
                type(item) is not str for item in after_available
            ):
                raise ValueError("formal prompt returned action surface is invalid")
            replay_steps.append(
                InteractiveReplayStep(
                    before,
                    tuple(available),
                    action,
                    after,
                    tuple(after_available),
                    (
                        _text(step.get("memory"), "formal prompt step memory")
                        if step.get("memory") is not None
                        else None
                    ),
                    (
                        _text(step.get("thought"), "formal prompt step thought")
                        if step.get("thought") is not None
                        else None
                    ),
                )
            )
        task_type = example.get("task_type")
        if type(task_type) is str:
            task_types.add(task_type)
        parsed.append(
            InteractivePromptExample(
                _text(example.get("example_id"), "example ID"),
                _text(
                    example.get(
                        "source_entry_id",
                        example.get("goal_id", example.get("game_id")),
                    ),
                    "source entry ID",
                ),
                _text(example.get("source_split"), "example source split"),
                _text(example.get("task"), "task"),
                tuple(turns),
                task_type if type(task_type) is str else None,
                tuple(replay_steps),
            )
        )
    if any(item.source_split != "train" for item in parsed):
        raise ValueError("formal prompt examples must be train-only")
    minimum_examples = (
        2
        if asset_id == "webshop-official-train-replay-memory-v5"
        else _MINIMUM_EXAMPLES.get(benchmark, 1)
    )
    if len(parsed) < minimum_examples:
        raise ValueError("formal prompt asset has insufficient train replays")
    if asset_id in {
        "webshop-official-train-replay-memory-v5",
        "webshop-official-train-replay-memory-v6",
        "webshop-official-train-efficient-memory-v7",
    }:
        for item in parsed:
            prefix, separator, index = item.source_entry_id.rpartition("goal-")
            if prefix or separator != "goal-" or not index.isdigit() or int(index) < 1500:
                raise ValueError("WebShop memory demonstration is outside the official train range")
    if asset_id.endswith(("memory-v6", "memory-v7")) and any(
        step.memory is None or step.thought is None
        for example in parsed
        for step in example.replay_steps
    ):
        raise ValueError("v6 prompt demonstrations require concrete memory and thought annotations")
    if benchmark == "alfworld" and task_types != {
        "pick_and_place",
        "pick_two_obj",
        "pick_clean_then_place",
        "pick_heat_then_place",
        "pick_cool_then_place",
        "look_at_obj",
    }:
        raise ValueError("ALFWorld prompt asset does not cover all six task types")
    return InteractivePromptAsset(
        benchmark=benchmark,
        asset_id=asset_id,
        source_kind=source_kind,
        source_split="train",
        source_revision=_text(raw.get("source_revision"), "source revision"),
        reasoning_mode=_text(raw.get("reasoning_mode"), "reasoning mode"),
        examples=tuple(parsed),
    )


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


_PRICE = re.compile(
    r"(?i)(?:under|below|less than|maximum(?: price)?(?: of)?)\s*\$?([0-9]+(?:\.[0-9]+)?)"
)


def _webshop_constraint_tags(task: str) -> frozenset[str]:
    lowered = task.casefold()
    groups = {
        "price": ("price", "dollar", "under ", "below ", "less than"),
        "color": ("color", "black", "white", "blue", "red", "green", "pink", "gray"),
        "size": ("size", "small", "medium", "large", "inch", "ounce", "pack"),
        "material": ("material", "cotton", "leather", "steel", "wood", "plastic"),
        "brand": ("brand", "made by", "from "),
        "use": ("for ", "compatible", "intended", "use"),
    }
    return frozenset(name for name, markers in groups.items() if any(x in lowered for x in markers))


_RETRIEVAL_WORD = re.compile(r"[a-z0-9]+")
_RETRIEVAL_STOP_WORDS = frozenset(
    "a an and are be buy find for from i in is it me my of on or please that the to want with "
    "price dollar dollars under below less than maximum".split()
)


def _retrieval_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _RETRIEVAL_WORD.findall(text.casefold())
        if len(token) > 1 and token not in _RETRIEVAL_STOP_WORDS
    )


def _select_webshop_examples_bm25(
    task: str,
    examples: tuple[InteractivePromptExample, ...],
    *,
    limit: int,
) -> tuple[InteractivePromptExample, ...]:
    """Retrieve the requested train replays by public-task lexical similarity.

    This independently adapts ExpeL's task-similarity experience retrieval while
    avoiding an embedding model in the backbone-only lane.  The corpus is the
    already validated train-only prompt asset; no environment state or answer is
    consulted.
    """

    if limit not in {1, 2}:
        raise ValueError("WebShop BM25 demonstration limit must be one or two")
    documents = tuple(_retrieval_tokens(example.task) for example in examples)
    query = frozenset(_retrieval_tokens(task))
    if not query or not any(documents):
        return examples[:limit]
    document_frequency = Counter(token for document in documents for token in set(document))
    average_length = sum(len(document) for document in documents) / len(documents)
    corpus_size = len(documents)

    def score(index: int) -> float:
        document = documents[index]
        frequencies = Counter(document)
        length_normalizer = 0.25 + 0.75 * len(document) / average_length
        total = 0.0
        for token in query:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_document_frequency = log(
                1
                + (corpus_size - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            total += (
                inverse_document_frequency * frequency * 2.2 / (frequency + 1.2 * length_normalizer)
            )
        return total

    target_tags = _webshop_constraint_tags(task)
    ranked = sorted(
        range(corpus_size),
        key=lambda index: (
            -score(index),
            -len(target_tags.intersection(_webshop_constraint_tags(examples[index].task))),
            index,
        ),
    )
    return tuple(examples[index] for index in ranked[:limit])


def _render_structured_memory_replay(
    benchmark: str,
    example: InteractivePromptExample,
    *,
    compact_action_surfaces: bool = False,
    include_thought: bool = True,
) -> str:
    if not example.replay_steps:
        raise ValueError("structured-memory demonstration lacks replay steps")
    for previous, current in zip(example.replay_steps, example.replay_steps[1:], strict=False):
        if previous.observation_after != current.observation_before:
            raise ValueError("structured-memory demonstration is not a contiguous replay")
    rendered = (
        _render_webshop_memory_replay(
            example,
            include_action_surfaces=not compact_action_surfaces,
            include_thought=include_thought,
        )
        if benchmark == "webshop"
        else _render_alfworld_memory_replay(example)
    )
    if benchmark == "webshop" and not compact_action_surfaces:
        return rendered
    return (
        "Every Action below was replay-verified against the native public action surface "
        "before that step. Unchosen action lists are omitted so the example teaches the plan "
        "rather than a location-specific menu.\n" + rendered
    )


def _render_webshop_memory_replay(
    example: InteractivePromptExample,
    *,
    include_action_surfaces: bool,
    include_thought: bool,
) -> str:
    queries: list[str] = []
    products: list[str] = []
    selected: list[str] = []
    parts: list[str] = []
    price = _PRICE.search(example.task)
    for index, step in enumerate(example.replay_steps, 1):
        lowered = step.action.casefold()
        action_value = step.action.partition("[")[2].removesuffix("]")
        returned = step.observation_after.casefold()
        if lowered.startswith("search["):
            queries.append(action_value)
            thought = (
                "Search with the product type and mandatory attributes from the task."
                if len(queries) == 1
                else "Refine the query rather than repeating a query that did not help."
            )
        elif lowered == "click[buy now]":
            thought = "All mandatory constraints are confirmed; purchase the verified candidate."
        elif any(marker in returned for marker in ("price:", "buy now", "description")):
            products.append(action_value)
            thought = "Inspect this candidate's product details, options, and price before buying."
        else:
            selected.append(action_value)
            thought = "Select the required visible option exactly as exposed by the page."
        ready = "yes" if lowered == "click[buy now]" else "no"
        unmet = "none" if ready == "yes" else "verify remaining task constraints"
        memory = step.memory or (
            "Goal requirements:\n"
            f"- product type: {example.task}\n"
            "- required attributes: all attributes stated in the task\n"
            f"- maximum price: {price.group(1) if price else 'not specified'}\n\n"
            f"Queries tried: {', '.join(queries) or 'none'}\n"
            f"Products inspected: {', '.join(products) or 'none'}\n"
            f"Current candidate: {products[-1] if products else 'none'}\n"
            f"Selected options: {', '.join(selected) or 'none'}\n"
            f"Matched requirements: {'all confirmed' if ready == 'yes' else 'check current page'}\n"
            f"Unmet requirements: {unmet}\n"
            f"Ready to purchase: {ready}"
        )
        parts.append(
            _render_demo_step(
                index,
                step,
                memory,
                step.thought or thought,
                include_action_surfaces=include_action_surfaces,
                include_thought=include_thought,
            )
        )
    return "\n\n".join(parts)


def _render_alfworld_memory_replay(example: InteractivePromptExample) -> str:
    location = "unknown"
    inventory: list[str] = []
    opened: list[str] = []
    cleaned: list[str] = []
    heated: list[str] = []
    cooled: list[str] = []
    completed: list[str] = []
    parts: list[str] = []
    for index, step in enumerate(example.replay_steps, 1):
        action = step.action.strip()
        lowered = action.casefold()
        if lowered.startswith("go to "):
            location = action[6:]
            thought = "Navigate to the location needed for the shortest unfinished subgoal."
        elif lowered.startswith("open "):
            opened.append(action[5:])
            completed.append("opened required receptacle")
            thought = "Open the receptacle before trying to access its contents."
        elif lowered.startswith("take "):
            inventory.append(action[5:].split(" from ", 1)[0])
            completed.append("acquired target object")
            thought = "Take the required object so the transformation or placement can proceed."
        elif lowered.startswith("clean "):
            cleaned.append(action[6:].split(" with ", 1)[0])
            completed.append("cleaned target object")
            thought = "Complete the required cleaning transformation before placement."
        elif lowered.startswith("heat "):
            heated.append(action[5:].split(" with ", 1)[0])
            completed.append("heated target object")
            thought = "Complete the required heating transformation before placement."
        elif lowered.startswith("cool "):
            cooled.append(action[5:].split(" with ", 1)[0])
            completed.append("cooled target object")
            thought = "Complete the required cooling transformation before placement."
        elif lowered.startswith(("move ", "put ")):
            moved = action.split(" to ", 1)[0].split(" in/on ", 1)[0]
            inventory = [item for item in inventory if item not in moved]
            completed.append("placed one required object")
            thought = "Place the prepared object in the task's required destination."
        elif lowered.startswith("examine "):
            completed.append("examined target with required object")
            thought = "Examine the requested object now that the prerequisite is satisfied."
        else:
            thought = "Use this public-state action to advance the shortest unfinished subgoal."
        terminal = index == len(example.replay_steps)
        memory = (
            f"Goal: {example.task}\n"
            f"Current location: {location}\n"
            f"Inventory: {', '.join(inventory) or 'empty'}\n"
            "Known object locations: retain locations stated by public observations\n"
            f"Open/closed receptacles: opened {', '.join(opened) or 'none'}\n"
            f"Cleaned objects: {', '.join(cleaned) or 'none'}\n"
            f"Heated objects: {', '.join(heated) or 'none'}\n"
            f"Cooled objects: {', '.join(cooled) or 'none'}\n"
            f"Completed subgoals: {', '.join(dict.fromkeys(completed)) or 'none'}\n"
            "Remaining subgoals: "
            f"{'none' if terminal else 'continue the shortest unmet goal step'}\n"
            "Repeated/no-progress actions: none"
        )
        parts.append(
            _render_demo_step(
                index,
                step,
                step.memory or memory,
                step.thought or thought,
                include_action_surfaces=False,
                include_thought=True,
            )
        )
    return "\n\n".join(parts)


def _render_demo_step(
    index: int,
    step: InteractiveReplayStep,
    memory: str,
    thought: str,
    *,
    include_action_surfaces: bool,
    include_thought: bool,
) -> str:
    assistant = f"Memory:\n{memory}\n\n"
    if include_thought:
        assistant += f"Thought:\n{thought}\n\n"
    assistant += f"Action:\n{step.action}"
    if include_action_surfaces:
        before_actions = "\n".join(f"- {item}" for item in step.available_actions_before)
        after_actions = "\n".join(f"- {item}" for item in step.available_actions_after)
        return (
            f"State before action {index}:\n{step.observation_before}\n\n"
            f"Admissible actions before action {index}:\n{before_actions}\n\n"
            f"{assistant}\n\n"
            f"Environment returned state:\n{step.observation_after}\n\n"
            f"Returned admissible actions:\n{after_actions or 'none (terminal)'}"
        )
    initial_state = f"Initial public state:\n{step.observation_before}\n\n" if index == 1 else ""
    return (
        f"{initial_state}Step {index}:\n"
        f"{assistant}\n\n"
        f"Environment feedback:\n{step.observation_after}"
    )


__all__ = [
    "InteractivePromptAsset",
    "InteractivePromptExample",
    "InteractiveReplayStep",
    "load_interactive_prompt_asset",
]
