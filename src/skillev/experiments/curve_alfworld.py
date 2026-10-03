from __future__ import annotations

import asyncio
import difflib
import hashlib
import importlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillev.contracts.evosteer import EvoTask
from skillev.evosteer_application import ResetReceipt, SessionRequest, TaskBinding, TaskSession
from skillev.orchestration.graph import NodeExecutionRequest, NodeExecutionResult
from skillev.rollout.evosteer_risk import ExecutionRiskPolicy
from skillev.runtime.contracts import BudgetVector

ALFWORLD_FAMILY = "alfworld"
ALFWORLD_CONFIG_ENV = "ALFWORLD_CONFIG_FILE"
ALFWORLD_DATA_ENV = "ALFWORLD_DATA"


def _patch_textworld_python313() -> None:
    try:
        from textworld.envs.pddl import textgen
    except ImportError:
        return
    if getattr(textgen.EvalSymbol, "_skillev_python313_compatible", False):
        return

    def derive(symbol: Any, context: Any = None) -> list[Any]:
        active = context or symbol.context
        value = eval(symbol.expression, vars(textgen), dict(active["variables"]))
        return [textgen.TerminalSymbol(value)]

    textgen.EvalSymbol.derive = derive
    textgen.EvalSymbol._skillev_python313_compatible = True


def _load_alfworld_env() -> Any:
    _patch_textworld_python313()
    from skillev.alfworld_env import ALFWorldEnv, AlfredEnvConfig

    return ALFWorldEnv, AlfredEnvConfig


def _configured_path() -> str:
    path = os.environ.get(ALFWORLD_CONFIG_ENV, "").strip()
    if path:
        return path
    repo_config = Path(__file__).resolve().parents[3] / "configs/alfworld/base_config.yaml"
    if repo_config.exists():
        return str(repo_config)
    try:
        package = importlib.import_module("alfworld")
        candidate = Path(str(getattr(package, "__file__", ""))).parent / "configs/base_config.yaml"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return ""


def _max_episode_steps() -> int:
    return max(1, int(os.environ.get("ALFWORLD_MAX_STEPS", "50")))


def _steps_per_node() -> int:
    return max(1, int(os.environ.get("ALFWORLD_STEPS_PER_NODE", "25")))


_REPORT_INSTRUCTION = "The episode has ended. Summarise the final observation in one line."
_TASK_CUE, _NO_EFFECT = "Your task is to:", "Nothing happens."
_OFF_PROTOCOL = re.compile(
    r"<\s*/?\s*[a-z_]+\s*>|```|\b(?:tools?|python|search\w*|code|answers?)\b", re.IGNORECASE
)
_ROOM = re.compile(r"Looking quickly around you, you see (.+?)\.(?:\s|$)", re.IGNORECASE | re.DOTALL)
_ARRIVAL = re.compile(r"^You arrive at [^.]*\.\s*")
_REASK_SEED_OFFSET = 1 << 20
_REASK_EXAMPLES = 4
_GAME_PROTOCOL = (
    "You play a text game: reply 'Thought: <one or two short sentences>' then 'Action: <one command "
    "copied from admissible_commands>'; the game runs it. This overrides the role text: no tool "
    "calls, code, search queries or 'Answer:' lines."
)
_STEP_INSTRUCTION = (
    "Whatever your role says, reply with one 'Thought:' line and then one 'Action:' line whose "
    "command is copied from admissible_commands."
)
_RULES = (
    "You hold at most one object; 'take' needs empty hands. Open a closed receptacle to see inside. "
    "Receptacles at your current spot need no 'go to'. Moving an object into a sinkbasin, microwave "
    "or fridge does not clean, heat or cool it."
)
_PROCEDURES = {
    "pick_and_place": (
        "1. Find the object: go to likely receptacles you have not visited, opening closed ones. "
        "2. Take it. 3. Go to the destination. 4. Move the object to it."
    ),
    "pick_two": (
        "1. Find one of the objects and take it. 2. Go to the destination and move it there. "
        "3. Find a second one (another instance, often elsewhere) and take it. 4. Move it to the "
        "destination too."
    ),
    "clean": (
        "'a clean X' is an X you must clean. 1. Find an X and take it. 2. Go to a sinkbasin and run "
        "'clean <object> with sinkbasin <n>'. 3. Go to the destination and move the object to it."
    ),
    "heat": (
        "'a hot X' is an X you must heat (a 'hot plate' is a plate). 1. Find an X and take it. 2. Go to "
        "the microwave and run 'heat <object> with microwave <n>' (no need to open it or put the object "
        "in). 3. Go to the destination and move the object to it."
    ),
    "cool": (
        "'a cool X' is an X you must cool. 1. Find an X and take it. 2. Go to the fridge and run 'cool "
        "<object> with fridge <n>' (no need to open it or put the object in). 3. Go to the destination "
        "and move the object to it."
    ),
    "look_at_obj_in_light": (
        "1. Find the object and take it. 2. Find a desklamp (often on a desk, sidetable or dresser) "
        "and go there. 3. Holding the object, run 'use desklamp <n>'."
    ),
}
_REASK_ERROR = (
    "{reason} The game did not run it. Reply again with one 'Thought:' line and one 'Action:' line "
    "whose command is copied from admissible_commands, for example: {examples}."
)
_ACTION_LINE = re.compile(r"^action\s*\d*\s*[:：]\s*(.*)$", re.IGNORECASE)
_THOUGHT_LINE = re.compile(r"^thought\s*\d*\s*[:：]\s*", re.IGNORECASE)
_INLINE_ACTION = re.compile(r"(?<!\w)(action\s*\d*\s*[:：])\s*(.*)$", re.IGNORECASE)
_MARKUP = re.compile(r"[*_`#>]")
_LEVELS = (
    (1200, 6, 20, 800, True),
    (480, 6, 10, 800, True),
    (0, 6, 0, 800, True),
    (0, 4, 0, 640, True),
    (0, 3, 0, 480, True),
    (0, 2, 0, 400, True),
    (0, 2, 0, 320, False),
    (0, 1, 0, 240, False),
)
_THOUGHT_CHARS = 200
_INVALID_CHARS = 200


PROTOCOL_VERSION = 4


@dataclass(frozen=True)
class ALFWorldProtocol:
    steps_per_node: int = 25
    max_episode_steps: int = 50
    history_pairs: int = 6
    observation_chars: int = 240
    earlier_actions: int = 40
    memory_chars: int = 800
    memory_observation_chars: int = 120
    trace_chars: int = 240
    step_output_tokens: int = 160

    def __post_init__(self) -> None:
        if self.max_episode_steps % self.steps_per_node:
            raise ValueError("ALFWORLD_STEPS_PER_NODE must divide ALFWORLD_MAX_STEPS")
        if min(self.memory_chars, self.memory_observation_chars, self.trace_chars) < 16:
            raise ValueError("memory and trace limits must be at least 16 characters")
        if self.step_output_tokens < 16:
            raise ValueError("step_output_tokens must be at least 16")

    @property
    def max_phases(self) -> int:
        return -(-self.max_episode_steps // self.steps_per_node)

    @classmethod
    def from_env(cls) -> "ALFWorldProtocol":
        return cls(steps_per_node=_steps_per_node(), max_episode_steps=_max_episode_steps())

    def to_value(self) -> dict[str, int | bool]:
        return {
            "version": PROTOCOL_VERSION,
            "steps_per_node": self.steps_per_node,
            "max_episode_steps": self.max_episode_steps,
            "history_pairs": self.history_pairs,
            "observation_chars": self.observation_chars,
            "earlier_actions": self.earlier_actions,
            "reask_invalid": True,
            "location_memory": True,
            "memory_chars": self.memory_chars,
            "memory_observation_chars": self.memory_observation_chars,
            "trace_chars": self.trace_chars,
            "step_output_tokens": self.step_output_tokens,
            "loop_note": True,
        }


def _executor_identity(policy: Any, protocol: "ALFWorldProtocol") -> str:
    base = str(getattr(policy, "configuration_id", "policy")) + os.environ.get(ALFWORLD_DATA_ENV, "")
    pinned = base + json.dumps(protocol.to_value(), sort_keys=True)
    return f"alfworld-text-executor@{PROTOCOL_VERSION}:" + hashlib.sha256(pinned.encode()).hexdigest()[:16]


def _catalog(*, mode: str = "train") -> tuple[tuple[str, ...], str]:
    ALFWorldEnv, AlfredEnvConfig = _load_alfworld_env()
    config = AlfredEnvConfig(config_file=_configured_path(), max_episode_steps=_max_episode_steps())
    catalog = ALFWorldEnv(config=config, mode=mode)
    files = tuple(str(item) for item in catalog.game_files)
    if not files:
        raise RuntimeError("ALFWorld has no game files; set ALFWORLD_DATA and run scripts/provision_alfworld.sh")
    return files, str(config.config_file)


def training_split_manifest(*, mode: str = "train") -> dict[str, Any]:
    files, config = _catalog(mode=mode)
    return {
        "benchmark": ALFWORLD_FAMILY,
        "split": mode,
        "count": len(files),
        "config_file": config,
        "data_root": os.environ.get(ALFWORLD_DATA_ENV, ""),
        "game_files": list(files),
    }


class _ALFWorldEpisode:
    def __init__(
        self,
        *,
        request: SessionRequest,
        mode: str,
        max_steps: int,
        selection_seed: int | None = None,
        protocol: ALFWorldProtocol | None = None,
    ) -> None:
        self.protocol = protocol or ALFWorldProtocol()
        self.last_outcome: dict[str, Any] | None = None
        ALFWorldEnv, AlfredEnvConfig = _load_alfworld_env()
        config = AlfredEnvConfig(config_file=_configured_path(), max_episode_steps=max(1, int(max_steps)))
        self.env = ALFWorldEnv(config=config, mode=mode)
        self.history: list[tuple[str, str]] = []
        self.request = request
        self.max_steps = max(1, int(max_steps))
        self.steps = 0
        self.done = False
        self.won = False
        self.last_observation = ""
        self.admissible: tuple[str, ...] = ()
        self.game_file = ""
        self.reset(selection_seed=selection_seed)

    def reset(self, *, selection_seed: int | None = None) -> None:
        self.last_observation = str(self.env.reset(seed=self.request.seed if selection_seed is None else selection_seed))
        self.initial_observation = self.last_observation
        self.history = []
        self.steps = 0
        self.done = False
        self.won = False
        self.admissible = tuple(str(x) for x in getattr(self.env, "_admissible_commands", ()))
        self.initial_admissible = self.admissible
        self.game_file = str(getattr(self.env, "current_game_file", ""))
        self.last_outcome = None

    def step(self, action: str, *, match: str = "exact") -> tuple[str, bool]:
        if self.done:
            return self.last_observation, self.won
        obs, _reward, done, info = self.env.step(action)
        self.steps += 1
        self.last_observation = str(obs)
        self.history.append((action, self.last_observation))
        self.admissible = tuple(str(x) for x in info.get("available_actions", ()))
        self.won = bool(info.get("won", False))
        self.done = bool(done) or self.won or self.steps >= self.max_steps
        status = (
            "parse_error"
            if match in ("fallback", "invalid")
            else "failed"
            if self.last_observation.strip() == _NO_EFFECT
            else "success"
        )
        self.last_outcome = {"status": status, "action": action, "terminal": self.done}
        return self.last_observation, self.won

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
            return
        inner = getattr(self.env, "alfred_env", None)
        if inner is not None and hasattr(inner, "close"):
            inner.close()

    def public_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "environment": "alfworld-textworld",
            "steps": self.steps,
            "done": self.done,
            "admissible_commands": list(self.admissible),
            "observation": self.last_observation,
        }
        k, phases = self.protocol.steps_per_node, self.protocol.max_phases
        state.update(
            {
                "enabled": True,
                "steps_per_phase": k,
                "max_phases": phases,
                "steps_completed": self.steps,
                "phase_index": min(phases, self.steps // k),
                "steps_in_phase": self.steps % k,
                "remaining_steps": max(0, self.max_steps - self.steps),
                "environment_terminal": self.done,
                "last_outcome": self.last_outcome,
            }
        )
        return state


def _normalise(text: str) -> str:
    return " ".join(str(text).strip().strip("`'\".").lower().split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _game_role(role: Any) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(getattr(role, "instruction", "")).strip())
    kept = " ".join(s.strip() for s in sentences if s.strip() and not _OFF_PROTOCOL.search(s))
    role_id = str(getattr(role, "role_id", "") or "")
    return f"{role_id}: {kept}" if role_id and kept else kept or role_id


def _room_locations(observation: str) -> list[str]:
    match = _ROOM.search(observation)
    if match is None:
        return []
    names = []
    for item in re.split(r",\s*(?:and\s+)?|\s+and\s+", match.group(1)):
        name = re.sub(r"^(?:an?|the)\s+", "", item.strip().lower())
        if re.fullmatch(r"[a-z][a-z ]*\d+", name):
            names.append(name)
    return names


def _mentions(command: str, location: str) -> bool:
    return re.search(r"(?<!\S)" + re.escape(location) + r"(?!\S)", command) is not None


def _task_type(task: str) -> str | None:
    text = task.rpartition(_TASK_CUE)[2].lower()
    if re.search(r"\b(?:look at|examine)\b", text) and re.search(r"\b(?:desk|floor)?lamp\b", text):
        return "look_at_obj_in_light"
    if re.search(r"\btwo\b", text):
        return "pick_two"
    for kind, words in (("clean", r"clean"), ("heat", r"heat|hot"), ("cool", r"cool|cold")):
        if re.search(rf"\b(?:{words})\b", text):
            return kind
    if re.search(r"\b(?:put|place)\b", text):
        return "pick_and_place"
    return None


def _parse_reply(text: str) -> tuple[str | None, str]:
    lines = [_MARKUP.sub("", line).strip() for line in str(text).splitlines()]
    for index, line in enumerate(lines):
        match = _ACTION_LINE.match(line)
        if match is None:
            continue
        command = match.group(1).strip() or next((rest for rest in lines[index + 1 :] if rest), "")
        thought = " ".join(_THOUGHT_LINE.sub("", before) for before in lines[:index] if before)
        return command or None, _clip(thought, _THOUGHT_CHARS)
    for index, line in enumerate(lines):
        match = _INLINE_ACTION.search(line)
        if match is None or not match.group(2).strip():
            continue
        before = [*lines[:index], line[: match.start(1)]]
        thought = " ".join(_THOUGHT_LINE.sub("", part).strip() for part in before if part.strip())
        return match.group(2).strip(), _clip(thought, _THOUGHT_CHARS)
    return None, ""


def _last_line(text: str) -> str:
    lines = [_MARKUP.sub("", line).strip() for line in str(text).splitlines()]
    return next((line for line in reversed(lines) if line), "")


def _select_action(text: str, admissible: tuple[str, ...]) -> tuple[str | None, str]:
    attempt, _thought = _parse_reply(text)
    if attempt is None:
        return None, "no_action"
    if not admissible:
        return attempt, "unchecked"
    norm = _normalise(attempt)
    by_norm = {_normalise(command): command for command in admissible}
    if norm in by_norm:
        return by_norm[norm], "exact"
    found = [
        command
        for key, command in by_norm.items()
        if key and re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", norm)
    ]
    if found:
        return max(found, key=len), "word"
    return attempt, "invalid"


def _known_locations(episode: Any) -> list[str]:
    known: list[str] = []
    for name in _room_locations(episode.initial_observation) + [
        command[len("go to ") :].strip()
        for command in (*episode.initial_admissible, *episode.admissible)
        if command.startswith("go to ")
    ]:
        if name and name not in known:
            known.append(name)
    return known


def _current_location(episode: Any) -> str | None:
    current = None
    for action, observation in episode.history:
        if action.startswith("go to ") and observation.strip() != _NO_EFFECT:
            current = action[len("go to ") :].strip()
    return current


def _spot(episode: Any, current: str | None) -> list[str]:
    if current is None:
        return []
    offered = {command[len("go to ") :].strip() for command in episode.admissible if command.startswith("go to ")}
    if not offered:
        return [current]
    initial = [command[len("go to ") :].strip() for command in episode.initial_admissible if command.startswith("go to ")]
    return [current] + [name for name in initial if name != current and name not in offered]


_PICKED = re.compile(r"^You pick up the (.+?) from the ")
_CHANGED = re.compile(r"^You (heat|cool|clean) the (.+?) using the ")
_DONE = {"heat": "heated", "cool": "cooled", "clean": "cleaned"}
_PLACED = re.compile(r"^You (?:move|put) the (.+?) (?:to|in|on|into|onto) the ")
_CARRYING = re.compile(r"^You are carrying:\s*(?:an?\s+|the\s+)?(.+?)\.?$")


def _holding(episode: Any, *, changes: bool = False) -> str | None:
    holding, done = None, []
    for _action, observation in episode.history:
        text = observation.strip()
        picked, carrying, changed = _PICKED.match(text), _CARRYING.match(text), _CHANGED.match(text)
        if picked:
            holding, done = picked.group(1), []
        elif carrying:
            if carrying.group(1) != holding:
                done = []
            holding = carrying.group(1)
        elif changed and changed.group(2) == holding:
            done = [*[d for d in done if d != _DONE[changed.group(1)]], _DONE[changed.group(1)]]
        elif _PLACED.match(text) or text.startswith("You are not carrying anything"):
            holding, done = None, []
    if holding is not None and changes and done:
        return f"{holding} ({', '.join(done)})"
    return holding


def _named_location(action: str, known: list[str]) -> str | None:
    named = [(action.rfind(name), name) for name in known if _mentions(action, name)]
    return max(named)[1] if named else None


def _change_note(action: str, observation: str) -> str:
    verb, _, rest = action.partition(" ")
    if verb == "take":
        return "took " + rest.partition(" from ")[0]
    if verb in ("move", "put"):
        return "put " + re.split(r" (?:to|in|on|into|onto|in/on) ", rest, maxsplit=1)[0] + " here"
    if verb in ("clean", "heat", "cool"):
        return f"{verb}ed " + rest.partition(" with ")[0]
    if verb == "close":
        return "closed"
    if verb == "use":
        return "turned on " + rest
    return _clip(observation, 60)


def _memory_entry(seen: str, notes: list[str], limit: int) -> str:
    later = f"Later: {', '.join(notes[-3:])}." if notes else ""
    if not later:
        return _clip(seen, limit)
    room = limit - len(later) - 1
    return f"{_clip(seen, room)} {later}" if seen and room >= 16 else _clip(later, limit)


def _location_memory(episode: Any, *, observation_chars: int, total_chars: int) -> dict[str, Any]:
    known = _known_locations(episode)
    if not known:
        return {}
    current: str | None = None
    seen: dict[str, str] = {}
    notes: dict[str, list[str]] = {}
    order: dict[str, None] = {}
    for action, observation in episode.history:
        text = observation.strip()
        if text == _NO_EFFECT:
            continue
        place = action[len("go to ") :].strip() if action.startswith("go to ") else _named_location(action, known)
        verb = action.partition(" ")[0]
        if verb == "go":
            current = place
            seen[place], notes[place] = _ARRIVAL.sub("", text).strip() or text, []
        elif place is not None and verb in ("open", "examine") and "you see" in text:
            seen[place], notes[place] = re.sub(r"^You open the [^.]*\.\s*", "", text), []
        elif place is not None or (verb == "use" and current is not None):
            place = place or current
            notes.setdefault(place, []).append(_change_note(action, text))
        else:
            continue
        order.pop(place, None)
        order[place] = None
    here = _spot(episode, current)
    value: dict[str, Any] = {
        "visited_last_seen": {name: "" for name in order},
        "not_visited": [name for name in known if name not in order and name not in here],
    }
    size = len(json.dumps(value))
    for name in reversed(list(order)):
        text = _memory_entry(seen.get(name, ""), notes.get(name, []), observation_chars)
        grown = len(json.dumps(text)) - 2
        if size + grown > total_chars:
            break
        value["visited_last_seen"][name] = text
        size += grown
    return value


def _loop_note(episode: Any, not_visited: list[str]) -> str:
    if not episode.history:
        return ""
    action, observation = last = episode.history[-1]
    times = episode.history.count(last)
    if times < 2:
        return ""
    options = f"e.g. go to one you have not visited: {', '.join(not_visited[:3])}" if not_visited else (
        "e.g. open or take something here, or do the next step of the procedure"
    )
    return (
        f"'{_clip(action, 80)}' has now given the same result {times} times; repeating it or going back "
        f"and forth will not change anything. Do something different ({options})."
    )


def _invalid_reason(attempt: str | None, episode: Any) -> tuple[str, str | None]:
    if attempt is None:
        return "Your reply has no 'Action:' line.", None
    command = _normalise(attempt)
    known = _known_locations(episode)
    current = _current_location(episode)
    here = _spot(episode, current)
    holding = _holding(episode)
    where = f"you are at {current}" if current else "you are in the middle of the room"

    def not_here(place: str) -> tuple[str, str | None] | None:
        if place not in known or place in here:
            return None
        go = f"go to {place}"
        return f"You are not at {place} ({where}).", go if go in episode.admissible else None

    match = re.fullmatch(r"go to (.+)", command)
    if match:
        target = match.group(1)
        if target == current:
            return f"You are already at {target}.", None
        if target in here:
            return f"{target} is at your current spot ({where}), so act on it without 'go to'.", None
        if target not in known:
            return f"There is no {target} in this room.", None
        return f"'{command}' is not possible now.", None
    match = re.fullmatch(r"take (.+?) from (.+)", command)
    if match:
        item, place = match.groups()
        if holding:
            return f"You are already holding {holding}; put it somewhere before taking another object.", None
        return not_here(place) or (f"There is no {item} at {place} now.", None)
    match = re.fullmatch(r"(?:move|put) (.+?) (?:to|in|on|into|onto|in/on) (.+)", command)
    if match:
        item, place = match.groups()
        if holding is None:
            return f"You are not holding anything; take {item} first.", None
        if holding != item:
            return f"You are holding {holding}, not {item}.", None
        return not_here(place) or (f"You cannot put {item} in {place} now; open it first if it is closed.", None)
    match = re.fullmatch(r"(clean|heat|cool) (.+?) with (.+)", command)
    if match:
        verb, item, place = match.groups()
        if holding != item:
            return f"You are not holding {item}.", None
        return not_here(place) or (f"You cannot {verb} {item} with {place}.", None)
    match = re.fullmatch(r"(open|close|examine|use) (.+)", command)
    if match:
        verb, target = match.groups()
        found = not_here(target)
        if found:
            return found
        if verb == "use":
            return f"There is no {target} at your spot ({where}).", None
    return f"'{_clip(command, 80)}' is not one of admissible_commands here.", None


def _reask_examples(probe: str, episode: Any, suggested: str | None, count: int = _REASK_EXAMPLES) -> list[str]:
    others = [command for command in episode.admissible if not command.startswith("go to ") and command != "help"]
    probe = _normalise(probe)[:80]
    ranked = difflib.get_close_matches(probe, others, n=count, cutoff=0.0) if probe else others[:count]
    picked = ([suggested] if suggested else []) + [command for command in ranked if command != suggested]
    return picked[:count] or list(episode.admissible[:count])


def _compact_skill(body: str, chars: int) -> str:
    try:
        value = json.loads(body)
    except ValueError:
        value = None
    if not isinstance(value, dict) or not value:
        return _clip(str(body).strip(), max(chars, 60))
    name = str(value.get("name") or value.get("description") or "skill")
    if chars <= 0:
        return _clip(name, 80)

    def items(key: str) -> list[str]:
        raw = value.get(key) or []
        return [str(item) for item in (raw if isinstance(raw, list) else [raw])]

    lines = [_clip(name, 80)]
    lines += [_clip(f"{i}. {step}", 160) for i, step in enumerate(items("plan"), 1)]
    lines += [_clip(f"Avoid: {pitfall}", 160) for pitfall in items("pitfall")]
    kept, size = [], 0
    for line in lines:
        if kept and size + len(line) + 1 > chars:
            break
        kept.append(line)
        size += len(line) + 1
    return "\n".join(kept)


class ALFWorldTextExecutor:
    def __init__(
        self,
        policy: Any,
        episode: _ALFWorldEpisode,
        *,
        temperature: float = 0.3,
        protocol: ALFWorldProtocol | None = None,
    ) -> None:
        self.policy = policy
        self.episode = episode
        self.temperature = float(temperature)
        self.protocol = protocol or getattr(episode, "protocol", None) or ALFWorldProtocol()
        self._identity = _executor_identity(policy, self.protocol)
        self._thoughts: dict[int, str] = {}

    @property
    def frozen_identity(self) -> str:
        return self._identity

    @property
    def environment_done(self) -> bool:
        return bool(self.episode.done or self.episode.steps >= self.episode.max_steps)

    def public_environment_state(self) -> dict[str, Any]:
        return self.episode.public_state()

    def _prompt_tokens(self, prompt: str) -> int:
        count = getattr(self.policy, "frozen_prompt_tokens", None)
        return int(count(prompt)) if callable(count) else len(prompt) // 2

    def _prompt(
        self,
        request: NodeExecutionRequest,
        node_steps: int,
        node_limit: int,
        level: int,
        *,
        rejected: str | None = None,
        reason: str = "",
        examples: tuple[str, ...] = (),
    ) -> str:
        protocol, episode = self.protocol, self.episode
        skill_chars, pairs, earlier_count, memory_chars, with_thought = _LEVELS[level]
        pairs = min(pairs, protocol.history_pairs)
        earlier_count = min(earlier_count, protocol.earlier_actions)
        history = episode.history
        recent = history[-pairs:] if pairs else []
        earlier: list[str] = []
        for action, _ in history[: len(history) - len(recent)]:
            if action not in earlier:
                earlier.append(action)
        _head, cue, tail = episode.initial_observation.rpartition(_TASK_CUE)
        task = (cue + tail).strip() if cue else episode.initial_observation
        kind = _task_type(task)
        recent_history = []
        for offset, (action, observation) in enumerate(recent, len(history) - len(recent) + 1):
            entry = {"action": action, "observation": _clip(observation, protocol.observation_chars)}
            if with_thought and offset == len(history) and self._thoughts.get(offset):
                entry = {"thought": self._thoughts[offset], **entry}
            recent_history.append(entry)
        current = _current_location(episode)
        here = _spot(episode, current)
        state: dict[str, Any] = {"location": current or "the middle of the room"}
        if len(here) > 1:
            state["also_here"] = here[1:]
        state["holding"] = _holding(episode, changes=True) or "nothing"
        memory = _location_memory(
            episode,
            observation_chars=protocol.memory_observation_chars,
            total_chars=min(memory_chars, protocol.memory_chars),
        )
        not_visited = memory.get("not_visited") if memory else None
        if not_visited is None:
            not_visited = _location_memory(episode, observation_chars=16, total_chars=0).get("not_visited", [])
        note = _loop_note(episode, not_visited)
        if note:
            state["warning"] = note
        value: dict[str, Any] = {
            "protocol": _GAME_PROTOCOL,
            "role": _game_role(request.role),
            "task": task,
            "procedure": _RULES + (" Steps: " + _PROCEDURES[kind] if kind else ""),
            "skills": [_compact_skill(body, skill_chars) for _, body in request.skills],
            "progress": {
                "episode_steps": f"{episode.steps}/{episode.max_steps}",
                "node_steps": f"{node_steps}/{node_limit}",
            },
            "earlier_actions": earlier[-earlier_count:] if earlier_count else [],
            "locations": memory,
            "recent_history": recent_history,
            "state": state,
            "observation": episode.last_observation,
            "admissible_commands": list(episode.admissible),
        }
        if rejected is not None:
            value["rejected_reply"] = _clip(rejected.strip(), protocol.trace_chars)
            value["error"] = _REASK_ERROR.format(
                reason=reason, examples=", ".join(f"'{command}'" for command in examples)
            )
        value["instruction"] = _STEP_INSTRUCTION
        return json.dumps({key: item for key, item in value.items() if item not in ([], {}, "")})

    def _fitted_prompt(
        self,
        request: NodeExecutionRequest,
        node_steps: int,
        node_limit: int,
        input_budget: int,
        share: int,
    ) -> tuple[str, int] | None:
        last = len(_LEVELS) - 1
        previous = None
        for level in range(len(_LEVELS)):
            prompt = self._prompt(request, node_steps, node_limit, level)
            if prompt == previous and level < last:
                continue
            previous = prompt
            if self._prompt_tokens(prompt) <= (input_budget if level == last else min(input_budget, share)):
                return prompt, level
        return None

    async def _generate(self, prompt: str, *, max_new_tokens: int, input_limit: int, seed: int) -> tuple[str, int, int, float]:
        def timed() -> tuple[str, int, int, float]:
            began = time.monotonic()
            text, inputs, outputs = self.policy.frozen_text(
                prompt,
                max_new_tokens=max_new_tokens,
                input_limit=input_limit,
                temperature=self.temperature,
                seed=seed % 2**63,
            )
            return str(text), int(inputs), int(outputs), time.monotonic() - began

        if getattr(self.policy, "thread_safe_generation", False):
            return await asyncio.to_thread(timed)
        return timed()

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        cap = request.role.model_maximum
        protocol = self.protocol
        per_step_output = max(16, min(int(protocol.step_output_tokens), int(cap.output_tokens)))
        step_limit = min(protocol.steps_per_node, int(cap.model_calls), int(cap.tool_calls))
        wall_ms = int(cap.wall_time_milliseconds or 0)
        inputs = outputs = calls = 0
        charged = slowest = 0.0
        taken: list[str] = []
        matches: list[str] = []
        invalid = reasks = recovered = 0
        trace: list[dict[str, Any]] = []
        if self.episode.done or step_limit < 1:
            _text, inputs, outputs, charged = await self._generate(
                json.dumps(
                    {
                        "role": _game_role(request.role),
                        "final_observation": self.episode.last_observation,
                        "instruction": _REPORT_INSTRUCTION,
                    },
                    sort_keys=True,
                ),
                max_new_tokens=per_step_output,
                input_limit=int(cap.input_tokens),
                seed=int(request.seed),
            )
            calls = 1
        else:
            for offset in range(step_limit):
                if self.episode.done or calls >= int(cap.model_calls):
                    break
                if calls and wall_ms and (charged + 2 * slowest) * 1000 > wall_ms:
                    break
                output_budget = min(per_step_output, int(cap.output_tokens) - outputs)
                input_budget = int(cap.input_tokens) - inputs
                share = input_budget // (step_limit - len(taken))
                fitted = self._fitted_prompt(request, len(taken), step_limit, input_budget, share)
                if output_budget < 1 or fitted is None:
                    if calls:
                        break
                    raise ValueError("ALFWorld role envelope cannot hold one environment step")
                prompt = fitted[0]
                text, used_in, used_out, seconds = await self._generate(
                    prompt, max_new_tokens=output_budget, input_limit=input_budget, seed=int(request.seed) + offset
                )
                inputs, outputs, calls = inputs + used_in, outputs + used_out, calls + 1
                failed = ("no_action", "invalid")
                action, match = _select_action(text, self.episode.admissible)
                reply = text
                entry: dict[str, Any] = {
                    "reply": _clip(text, protocol.trace_chars),
                    "match": match,
                    "level": fitted[1],
                }
                if match in failed:
                    invalid += 1
                    retry_output = min(per_step_output, int(cap.output_tokens) - outputs)
                    retry_input = int(cap.input_tokens) - inputs
                    steps_left = min(step_limit - len(taken), self.episode.max_steps - self.episode.steps)
                    spare = (
                        calls + steps_left <= int(cap.model_calls)
                        and retry_output >= 1
                        and not (wall_ms and (charged + seconds + 2 * max(slowest, seconds)) * 1000 > wall_ms)
                    )
                    if not spare:
                        retry = None
                    else:
                        reason, suggested = _invalid_reason(action, self.episode)
                        examples = _reask_examples(action or _last_line(text), self.episode, suggested)
                        entry["reask_reason"] = reason
                        retry = self._prompt(
                            request, len(taken), step_limit, fitted[1],
                            rejected=text, reason=reason, examples=tuple(examples),
                        )
                    if retry is not None and self._prompt_tokens(retry) <= retry_input:
                        again, used_in, used_out, extra = await self._generate(
                            retry,
                            max_new_tokens=retry_output,
                            input_limit=retry_input,
                            seed=int(request.seed) + _REASK_SEED_OFFSET + offset,
                        )
                        inputs, outputs, calls = inputs + used_in, outputs + used_out, calls + 1
                        seconds += extra
                        reasks += 1
                        action, match = _select_action(again, self.episode.admissible)
                        reply = again
                        recovered += int(match not in failed)
                        entry.update(reask_reply=_clip(again, protocol.trace_chars), reask_match=match)
                    if match in failed:
                        action = _clip(action if action is not None else _last_line(reply), _INVALID_CHARS)
                        match = "invalid"
                began = time.monotonic()
                self.episode.step(action, match=match)
                seconds += time.monotonic() - began
                charged, slowest = charged + seconds, max(slowest, seconds)
                taken.append(action)
                matches.append(match)
                if match != "invalid":
                    self._thoughts[self.episode.steps] = _parse_reply(reply)[1]
                trace.append(
                    {
                        "step": self.episode.steps,
                        **entry,
                        "action": action,
                        "observation": _clip(self.episode.last_observation, protocol.trace_chars),
                    }
                )
        public = json.dumps(
            {
                "actions": taken,
                "observation": self.episode.last_observation,
                "episode_done": self.episode.done,
                "environment_steps_total": self.episode.steps,
            },
            sort_keys=True,
        )
        step_record: dict[str, Any] = {
            "actions": taken,
            "match": matches,
            "done": self.episode.done,
            "steps": self.episode.steps,
            "tool_calls": len(taken),
        }
        step_record.update(invalid_replies=invalid, reasks=reasks, reask_recovered=recovered)
        metadata: dict[str, Any] = {"environment_step": step_record, "tool_transcript": trace}
        return NodeExecutionResult(
            public,
            BudgetVector(
                input_tokens=inputs,
                output_tokens=outputs,
                model_calls=calls,
                agent_turns=1,
                tool_calls=len(taken),
                wall_time_milliseconds=min(math.ceil(charged * 1000), wall_ms) if wall_ms else math.ceil(charged * 1000),
            ),
            metadata,
        )

def _check_envelope(protocol: ALFWorldProtocol, config: Any) -> None:
    roles_for = getattr(config, "roles_for", None)
    if not callable(roles_for):
        return
    if ALFWORLD_FAMILY not in tuple(getattr(config, "task_families", ()) or ()):
        return
    for role in roles_for(ALFWORLD_FAMILY):
        cap = role.model_maximum
        if cap.output_tokens < cap.model_calls * protocol.step_output_tokens:
            raise ValueError(
                f"ALFWorld role {role.role_id!r}: {cap.output_tokens} output tokens for {cap.model_calls} "
                f"model calls cannot hold {protocol.step_output_tokens}-token replies; "
                "raise the role's output-token envelope"
            )


def make_bindings(
    *,
    policy: Any,
    config: Any,
    mode: str = "train",
    count: int | None = None,
    games: dict[int, str] | None = None,
    temperature: float = 0.3,
) -> tuple[TaskBinding, ...]:
    protocol = ALFWorldProtocol.from_env()
    _check_envelope(protocol, config)
    files, config_file = _catalog(mode=mode)
    if games is not None:
        chosen = []
        for index, suffix in sorted(games.items()):
            if not 0 <= index < len(files) or not files[index].replace("\\", "/").endswith(suffix):
                raise ValueError(f"ALFWorld {mode} catalog differs from the manifest at index {index}: {suffix}")
            chosen.append((index, files[index]))
    else:
        chosen = list(enumerate(files[: int(count)] if count is not None else files))
    bindings: list[TaskBinding] = []
    for index, _game_file in chosen:
        task = EvoTask(
            task_id=f"alfworld/{mode}/{index:06d}",
            family=ALFWORLD_FAMILY,
            prompt="Complete the ALFWorld instruction shown by the reset observation.",
            reset_id=f"reset-{index}",
            environment_config_id=f"alfworld-textworld:{mode}:{config_file}",
        )
        executor_id = _executor_identity(policy, protocol)

        def session(request: SessionRequest, *, _task=task, _index=index) -> TaskSession:
            episode = _ALFWorldEpisode(
                request=request,
                mode=mode,
                max_steps=protocol.max_episode_steps,
                selection_seed=_index,
                protocol=protocol,
            )
            executor = ALFWorldTextExecutor(
                policy, episode, temperature=temperature, protocol=protocol
            )

            def evaluate(_output: str) -> float:
                return 1.0 if episode.done and episode.won else 0.0

            receipt = ResetReceipt(
                request.task.identity,
                request.task.reset_id,
                request.task.environment_config_id,
                str(hashlib.sha256(episode.last_observation.encode()).hexdigest()),
                str(uuid.uuid4()),
                request.seed,
            )
            risk = ExecutionRiskPolicy(
                executor.frozen_identity,
                request.task.environment_config_id,
                scope="isolated",
                capability_id="alfworld-textworld@1",
            )
            return TaskSession(executor, evaluate, close=episode.close, reset_receipt=receipt, risk_assessor=risk)

        bindings.append(TaskBinding(task, executor_id, session, replay_safe=True))
    return tuple(bindings)


__all__ = ["ALFWORLD_FAMILY", "make_bindings", "training_split_manifest"]
