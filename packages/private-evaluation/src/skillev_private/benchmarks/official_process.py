"""Pinned cross-process adapters for the three official text environments.

The benchmark packages have mutually incompatible Python requirements.  They
therefore run in explicitly selected interpreters while this module implements
the dependency-free factory protocols consumed by the private benchmark
bridges.  The child protocol is intentionally tiny: answer-free task inventory,
reset/step projections, and the terminal reward signal are the only values
allowed back across the pipe.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping
from contextlib import ExitStack, nullcontext, suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias, cast

from skillev.contracts import JsonValue, normalize_json
from skillev.diagnostics.rollout_progress import current_progress
from skillev.rollout.errors import EpisodeInfrastructureError

from .alfworld_official import (
    OfficialALFWorldResetResult,
    OfficialALFWorldStepResult,
    OfficialALFWorldTask,
    OfficialALFWorldTextEnv,
)
from .alfworld_public_goal import LEGACY_GOAL_BINDING, require_goal_binding
from .deadline_pipe import DeadlinePipe
from .scienceworld_official import (
    OfficialScienceWorldStepResult,
    OfficialScienceWorldTask,
    OfficialScienceWorldTextEnv,
)
from .webshop_official import (
    OfficialWebShopGoal,
    OfficialWebShopStepResult,
    OfficialWebShopTextEnv,
)

_PROTOCOL_VERSION = "skillev-official-environment-worker@1"
_MAX_MESSAGE_BYTES = 2 * 1024 * 1024
_REVISION_LENGTH = 40
_WORKER_SCRIPT = Path(__file__).with_name("official_environment_worker.py")
WEBSHOP_JSON_STORAGE_FORMAT = "official-json-array@1"
WEBSHOP_SQLITE_STORAGE_FORMAT = "streaming-sqlite@1"


class OfficialEnvironmentInfrastructureError(EpisodeInfrastructureError):
    """The pinned official environment process did not complete its request."""


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty text without NUL")
    return value


def _seed(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("official environment seed must be a non-negative integer")
    return value


def _positive_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("official environment timeout must be numeric")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("official environment timeout must be positive and finite")
    return timeout


def _path(value: Path | str, *, field_name: str, directory: bool) -> Path:
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{field_name} must identify an existing {kind}")
    return path


def _executable_path(value: Path | str, *, field_name: str) -> Path:
    """Validate an executable path without erasing a virtualenv symlink."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = path.absolute()
    if not path.is_file():
        raise ValueError(f"{field_name} must identify an existing file")
    return path


def _revision(value: object) -> str:
    revision = _text(value, field_name="source_revision").lower()
    if len(revision) != _REVISION_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("source_revision must be a full 40-character Git commit")
    return revision


def _verify_checkout(*, source_root: Path, source_revision: str, timeout: float) -> None:
    try:
        completed = subprocess.run(  # noqa: S603 - the executable is the fixed Git CLI
            ("git", "-C", str(source_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        subprocess.run(  # noqa: S603 - the executable is the fixed Git CLI
            ("git", "-C", str(source_root), "diff", "--quiet", "HEAD", "--"),
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise OfficialEnvironmentInfrastructureError(
            "official source revision could not be resolved"
        ) from error
    if completed.stdout.strip().lower() != source_revision:
        raise ValueError("official source checkout does not match its pinned revision")


def _object(
    value: object, *, fields: set[str], label: str, optional_fields: frozenset[str] = frozenset()
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if (
        not isinstance(normalized, dict)
        or not fields.issubset(normalized)
        or set(normalized) - fields - optional_fields
    ):
        raise OfficialEnvironmentInfrastructureError(f"{label} has an incompatible shape")
    return normalized


def _string_array(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise OfficialEnvironmentInfrastructureError(f"{label} must be an array")
    items = tuple(_text(item, field_name=label) for item in value)
    if len(set(items)) != len(items):
        raise OfficialEnvironmentInfrastructureError(f"{label} must be unique")
    return items


@dataclass(frozen=True, slots=True)
class PinnedOfficialProcess:
    """Interpreter and source checkout identity shared by one official package."""

    interpreter_path: Path
    source_root: Path
    source_revision: str
    request_timeout_seconds: float = 60.0
    worker_script_path: Path = _WORKER_SCRIPT
    worker_stderr_path: Path | None = None

    def __post_init__(self) -> None:
        interpreter = _executable_path(
            self.interpreter_path,
            field_name="interpreter_path",
        )
        source_root = _path(self.source_root, field_name="source_root", directory=True)
        worker = _path(
            self.worker_script_path,
            field_name="worker_script_path",
            directory=False,
        )
        revision = _revision(self.source_revision)
        timeout = _positive_timeout(self.request_timeout_seconds)
        _verify_checkout(source_root=source_root, source_revision=revision, timeout=timeout)
        object.__setattr__(self, "interpreter_path", interpreter)
        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "request_timeout_seconds", timeout)
        object.__setattr__(self, "worker_script_path", worker)


class WorkerLifecycleState(StrEnum):
    OPEN = "open"
    CLOSED_SUCCESSFULLY = "closed-successfully"
    DISCARDED_AFTER_INITIALIZATION_FAILURE = "discarded-after-initialization-failure"
    DISCARDED_AFTER_REQUEST_FAILURE = "discarded-after-request-failure"


@dataclass(slots=True)
class OfficialWorkerClient:
    runtime: PinnedOfficialProcess
    _process: subprocess.Popen[bytes] = field(init=False, repr=False)
    _response_fd: int = field(init=False, repr=False)
    _channel: DeadlinePipe = field(init=False, repr=False)
    _request_id: int = field(default=0, init=False, repr=False)
    _state: WorkerLifecycleState = field(
        default=WorkerLifecycleState.OPEN,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        read_fd, write_fd = os.pipe()
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        with ExitStack() as files:
            error_log = self.runtime.worker_stderr_path
            if error_log is not None:
                error_log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._process = subprocess.Popen(  # noqa: S603 - interpreter is deployment-pinned
                (
                    str(self.runtime.interpreter_path),
                    str(self.runtime.worker_script_path),
                    "--response-fd",
                    str(write_fd),
                ),
                cwd=self.runtime.source_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=files.enter_context(error_log.open("ab"))
                if error_log
                else subprocess.DEVNULL,
                pass_fds=(write_fd,),
                env=environment,
            )
        os.close(write_fd)
        self._response_fd = read_fd
        assert self._process.stdin is not None
        self._channel = DeadlinePipe(self._process.stdin.fileno(), read_fd, _MAX_MESSAGE_BYTES)

    def request(self, operation: str, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        if self._state is not WorkerLifecycleState.OPEN:
            raise RuntimeError("official worker is not open")
        deadline = time.monotonic() + self.runtime.request_timeout_seconds
        try:
            return self._request(operation, payload, deadline=deadline)
        except (OSError, ValueError, TypeError, OfficialEnvironmentInfrastructureError) as error:
            self._discard(WorkerLifecycleState.DISCARDED_AFTER_REQUEST_FAILURE)
            raise OfficialEnvironmentInfrastructureError(
                f"official environment request failed: {type(error).__name__}: {error}"
            ) from error

    def _request(
        self, operation: str, payload: Mapping[str, JsonValue], *, deadline: float
    ) -> dict[str, JsonValue]:
        self._request_id += 1
        request = normalize_json(
            {
                "operation": _text(operation, field_name="worker operation"),
                "payload": dict(payload),
                "protocol_version": _PROTOCOL_VERSION,
                "request_id": self._request_id,
            }
        )
        encoded = json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ValueError("official environment worker request is too large")
        self._channel.write(encoded, deadline=deadline)
        response_bytes = self._channel.read(deadline=deadline)
        response_value: object = json.loads(response_bytes)
        response = _object(
            response_value,
            fields={"protocol_version", "request_id", "result"},
            label="official environment worker response",
        )
        if response["protocol_version"] != _PROTOCOL_VERSION:
            raise OfficialEnvironmentInfrastructureError("official worker protocol version differs")
        if response["request_id"] != self._request_id:
            raise OfficialEnvironmentInfrastructureError(
                "official worker response request ID differs"
            )
        result = response["result"]
        if not isinstance(result, dict):
            raise OfficialEnvironmentInfrastructureError(
                "official environment worker result must be an object"
            )
        return result

    def close_successfully(self) -> None:
        """Perform the sole normal terminal transition exactly once."""

        if self._state is WorkerLifecycleState.DISCARDED_AFTER_REQUEST_FAILURE:
            # Cleanup must not mask the original infrastructure failure or send
            # another request on a stream containing a timed-out partial reply.
            return
        if self._state is not WorkerLifecycleState.OPEN:
            raise RuntimeError("official worker was closed more than once")
        if self._process.poll() is not None:
            raise OfficialEnvironmentInfrastructureError(
                "official worker exited before the explicit close"
            )
        result = self.request("close", {})
        closed = _object(result, fields={"closed"}, label="worker close result")["closed"]
        if closed is not True:
            raise OfficialEnvironmentInfrastructureError("official worker did not confirm close")
        self._process.wait(timeout=self.runtime.request_timeout_seconds)
        if self._process.returncode != 0:
            raise OfficialEnvironmentInfrastructureError(
                "official worker close returned non-zero status"
            )
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("official worker request channel disappeared")
        stdin.close()
        os.close(self._response_fd)
        self._state = WorkerLifecycleState.CLOSED_SUCCESSFULLY

    def _discard(self, state: WorkerLifecycleState) -> None:
        if self._state is WorkerLifecycleState.OPEN:
            if self._process.poll() is None:
                with suppress(OSError):
                    self._process.terminate()
                try:
                    self._process.wait(timeout=min(2.0, self.runtime.request_timeout_seconds))
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2.0)
            if self._process.stdin is not None:
                with suppress(OSError):
                    self._process.stdin.close()
            with suppress(OSError):
                os.close(self._response_fd)
        self._state = state


def _initialization_payload(
    *,
    benchmark: str,
    runtime: PinnedOfficialProcess,
    deployment: Mapping[str, JsonValue],
    task: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "benchmark": benchmark,
        "deployment": dict(deployment),
        "source_revision": runtime.source_revision,
        "source_root": str(runtime.source_root),
        "task": dict(task),
    }


def _initialize(client: OfficialWorkerClient, payload: Mapping[str, JsonValue]) -> None:
    result = client.request("initialize", payload)
    ready = _object(result, fields={"ready"}, label="worker initialize result")["ready"]
    if ready is not True:
        raise OfficialEnvironmentInfrastructureError("official environment worker is not ready")


def _initialize_or_discard(
    client: OfficialWorkerClient,
    payload: Mapping[str, JsonValue],
) -> None:
    """Reap a child that cannot complete the one-time initialization handshake."""

    try:
        _initialize(client, payload)
    except Exception:
        client._discard(WorkerLifecycleState.DISCARDED_AFTER_INITIALIZATION_FAILURE)
        raise


@dataclass(frozen=True, slots=True)
class JsonArrayWebShopDeployment:
    runtime: PinnedOfficialProcess
    products_path: Path
    index_path: Path
    inventory_size: int
    seed: int
    kind: str = "json-array"

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, PinnedOfficialProcess):
            raise TypeError("WebShop JSON deployment requires a pinned runtime")
        products = _path(self.products_path, field_name="products_path", directory=False)
        index = _path(self.index_path, field_name="index_path", directory=True)
        _seed(self.seed)
        if self.inventory_size not in {100, 1_000, 100_000}:
            raise ValueError("WebShop JSON inventory size is unsupported")
        index_name = {
            100: "indexes_100",
            1_000: "indexes_1k",
            100_000: "indexes_100k",
        }[self.inventory_size]
        expected_index = (self.runtime.source_root / "search_engine" / index_name).resolve()
        if index != expected_index:
            raise ValueError("WebShop search index differs from the official pinned layout")
        object.__setattr__(self, "products_path", products)
        object.__setattr__(self, "index_path", index)


@dataclass(frozen=True, slots=True)
class SQLiteWebShopDeployment:
    runtime: PinnedOfficialProcess
    store_path: Path
    goals_path: Path
    index_path: Path
    seed: int
    kind: str = "sqlite"

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, PinnedOfficialProcess):
            raise TypeError("WebShop SQLite deployment requires a pinned runtime")
        object.__setattr__(
            self,
            "store_path",
            _path(self.store_path, field_name="store_path", directory=False),
        )
        object.__setattr__(
            self,
            "goals_path",
            _path(self.goals_path, field_name="goals_path", directory=False),
        )
        object.__setattr__(
            self,
            "index_path",
            _path(self.index_path, field_name="index_path", directory=True),
        )
        _seed(self.seed)


WebShopDeployment: TypeAlias = JsonArrayWebShopDeployment | SQLiteWebShopDeployment


@dataclass(frozen=True, slots=True)
class OfficialWebShopProcessFactory:
    deployment: WebShopDeployment
    observation_mode: str = "text"

    def __post_init__(self) -> None:
        if not isinstance(self.deployment, JsonArrayWebShopDeployment | SQLiteWebShopDeployment):
            raise TypeError("WebShop process factory requires a closed deployment variant")
        if self.observation_mode not in {"text", "text_rich"}:
            raise ValueError("unsupported native WebShop observation mode")

    def deployment_value(self) -> dict[str, JsonValue]:
        """Exact child-process deployment payload for the selected storage mode."""

        match self.deployment:
            case JsonArrayWebShopDeployment() as deployment:
                return {
                    "index_path": str(deployment.index_path),
                    "inventory_size": deployment.inventory_size,
                    "kind": deployment.kind,
                    "products_path": str(deployment.products_path),
                    "seed": deployment.seed,
                }
            case SQLiteWebShopDeployment() as deployment:
                return {
                    "goals_path": str(deployment.goals_path),
                    "index_path": str(deployment.index_path),
                    "kind": deployment.kind,
                    "seed": deployment.seed,
                    "store_path": str(deployment.store_path),
                }
        from typing import assert_never

        assert_never(self.deployment)

    def create(self, goal: OfficialWebShopGoal) -> OfficialWebShopTextEnv:
        if not isinstance(goal, OfficialWebShopGoal):
            raise TypeError("WebShop process factory requires OfficialWebShopGoal")
        if type(goal.payload) is not int or goal.payload < 0:
            raise ValueError("official WebShop goal payload must be a numeric goal index")
        client = OfficialWorkerClient(self.deployment.runtime)
        _initialize_or_discard(
            client,
            _initialization_payload(
                benchmark="webshop",
                runtime=self.deployment.runtime,
                deployment=self.deployment_value(),
                task={
                    "goal_index": goal.payload,
                    **(
                        {"observation_mode": self.observation_mode}
                        if self.observation_mode != "text"
                        else {}
                    ),
                },
            ),
        )
        return _WebShopProcessEnv(goal=goal, client=client)


@dataclass(frozen=True, slots=True)
class UninitializedWebShopEnvironment:
    kind: str = "uninitialized"


@dataclass(frozen=True, slots=True)
class ActiveWebShopEnvironment:
    instruction_text: str
    available_actions: tuple[str, ...]
    kind: str = "active"


@dataclass(frozen=True, slots=True)
class ClosedWebShopEnvironment:
    instruction_text: str
    available_actions: tuple[str, ...]
    kind: str = "closed"


WebShopLifecycle: TypeAlias = (
    UninitializedWebShopEnvironment | ActiveWebShopEnvironment | ClosedWebShopEnvironment
)


@dataclass(slots=True)
class _WebShopProcessEnv:
    goal: OfficialWebShopGoal
    client: OfficialWorkerClient = field(repr=False)
    _state: WebShopLifecycle = field(default_factory=UninitializedWebShopEnvironment, init=False)

    @property
    def goal_id(self) -> str:
        return self.goal.goal_id

    @property
    def session_id(self) -> str:
        return self.goal.session_id

    @property
    def instruction_text(self) -> str:
        if not isinstance(self._state, ActiveWebShopEnvironment | ClosedWebShopEnvironment):
            raise OfficialEnvironmentInfrastructureError("WebShop environment is not reset")
        return self._state.instruction_text

    def reset(self, session_id: str) -> str:
        if not isinstance(self._state, UninitializedWebShopEnvironment):
            raise OfficialEnvironmentInfrastructureError("WebShop environment reset is one-shot")
        if session_id != self.goal.session_id:
            raise ValueError("WebShop reset session differs from its pinned identity")
        result = _object(
            self.client.request("reset", {}),
            fields={"available_actions", "instruction_text", "observation_text"},
            label="WebShop reset result",
        )
        self._state = ActiveWebShopEnvironment(
            instruction_text=_text(result["instruction_text"], field_name="instruction"),
            available_actions=_string_array(
                result["available_actions"], label="WebShop available actions"
            ),
        )
        return _text(result["observation_text"], field_name="WebShop observation")

    def step(self, action: str) -> OfficialWebShopStepResult:
        state = self._state
        if not isinstance(state, ActiveWebShopEnvironment):
            raise OfficialEnvironmentInfrastructureError("WebShop environment is not active")
        result = _object(
            self.client.request("step", {"action": _text(action, field_name="action")}),
            fields={"available_actions", "observation_text", "reward", "terminal"},
            label="WebShop step result",
        )
        available_actions = _string_array(
            result["available_actions"], label="WebShop available actions"
        )
        step = OfficialWebShopStepResult(
            observation_text=cast(str, result["observation_text"]),
            reward=cast(float, result["reward"]),
            terminal=cast(bool, result["terminal"]),
        )
        if step.terminal:
            self.client.close_successfully()
            self._state = ClosedWebShopEnvironment(
                instruction_text=state.instruction_text,
                available_actions=available_actions,
            )
        else:
            self._state = ActiveWebShopEnvironment(
                instruction_text=state.instruction_text,
                available_actions=available_actions,
            )
        return step

    def get_available_actions(self) -> tuple[str, ...]:
        state = self._state
        if not isinstance(state, ActiveWebShopEnvironment | ClosedWebShopEnvironment):
            raise OfficialEnvironmentInfrastructureError("WebShop environment is not reset")
        return state.available_actions

    async def close(self) -> None:
        """Release a live worker when a rollout ends before environment terminal."""

        if self.client._state is WorkerLifecycleState.OPEN:
            await asyncio.to_thread(self.client.close_successfully)


@dataclass(frozen=True, slots=True)
class ALFWorldGameDeployment:
    data_directory: Path
    train_eval: str
    instruction_text: str

    def __post_init__(self) -> None:
        directory = _path(
            self.data_directory,
            field_name="ALFWorld data_directory",
            directory=True,
        )
        if self.train_eval not in {
            "train",
            "eval_in_distribution",
            "eval_out_of_distribution",
        }:
            raise ValueError("ALFWorld train_eval is unsupported")
        instruction = _text(self.instruction_text, field_name="ALFWorld instruction")
        _single_alfworld_game(directory)
        object.__setattr__(self, "data_directory", directory)
        object.__setattr__(self, "instruction_text", instruction)


def _single_alfworld_game(directory: Path) -> Path:
    game_files = tuple(directory.rglob("game.tw-pddl"))
    trajectories = tuple(directory.rglob("traj_data.json"))
    if (
        len(game_files) != 1
        or len(trajectories) != 1
        or game_files[0].parent != trajectories[0].parent
    ):
        raise ValueError("ALFWorld case data directory must contain exactly one complete game")
    return game_files[0]


@dataclass(frozen=True, slots=True)
class OfficialALFWorldProcessFactory:
    runtime: PinnedOfficialProcess
    config_path: Path
    games: Mapping[str, ALFWorldGameDeployment]
    seed: int
    simulator_max_steps: int | None = None
    goal_binding: str = LEGACY_GOAL_BINDING

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, PinnedOfficialProcess):
            raise TypeError("ALFWorld process factory requires a pinned runtime")
        require_goal_binding(self.goal_binding)
        config = _path(self.config_path, field_name="ALFWorld config_path", directory=False)
        _seed(self.seed)
        games = dict(self.games)
        if not games:
            raise ValueError("ALFWorld process factory requires game deployments")
        for game_id, deployment in games.items():
            _text(game_id, field_name="ALFWorld game_id")
            if not isinstance(deployment, ALFWorldGameDeployment):
                raise TypeError("ALFWorld game deployment is incompatible")
        if self.simulator_max_steps is not None and (
            type(self.simulator_max_steps) is not int or self.simulator_max_steps <= 0
        ):
            raise ValueError("ALFWorld simulator_max_steps must be positive or null")
        object.__setattr__(self, "config_path", config)
        object.__setattr__(self, "games", games)

    def create(self, task: OfficialALFWorldTask) -> OfficialALFWorldTextEnv:
        if not isinstance(task, OfficialALFWorldTask):
            raise TypeError("ALFWorld process factory requires OfficialALFWorldTask")
        if task.seed != self.seed:
            raise ValueError("ALFWorld task seed differs from its deployment identity")
        deployment = self.games.get(task.game_id)
        if deployment is None:
            raise ValueError("ALFWorld game has no pinned data deployment")
        client = OfficialWorkerClient(self.runtime)
        _initialize_or_discard(
            client,
            _initialization_payload(
                benchmark="alfworld",
                runtime=self.runtime,
                deployment={
                    "config_path": str(self.config_path),
                    "data_directory": str(deployment.data_directory),
                    "instruction_text": deployment.instruction_text,
                    **(
                        {"goal_binding": self.goal_binding}
                        if self.goal_binding != LEGACY_GOAL_BINDING
                        else {}
                    ),
                    "seed": self.seed,
                    "train_eval": deployment.train_eval,
                },
                task={
                    "game_id": task.game_id,
                    "max_steps": task.max_steps,
                    "simulator_max_steps": self.simulator_max_steps or task.max_steps,
                },
            ),
        )
        return _ALFWorldProcessEnv(task=task, client=client)


@dataclass(slots=True)
class _ALFWorldProcessEnv:
    task: OfficialALFWorldTask
    client: OfficialWorkerClient = field(repr=False)

    @property
    def game_id(self) -> str:
        return self.task.game_id

    @property
    def seed(self) -> int:
        return self.task.seed

    @property
    def max_steps(self) -> int:
        return self.task.max_steps

    def reset(self, seed: int) -> OfficialALFWorldResetResult:
        if seed != self.task.seed:
            raise ValueError("ALFWorld reset seed differs from its pinned identity")
        result = _object(
            self.client.request("reset", {}),
            fields={"admissible_commands", "instruction_text", "observation_text"},
            label="ALFWorld reset result",
        )
        return OfficialALFWorldResetResult(
            observation_text=cast(str, result["observation_text"]),
            instruction_text=cast(str, result["instruction_text"]),
            admissible_commands=_string_array(
                result["admissible_commands"], label="ALFWorld admissible commands"
            ),
        )

    def step(self, action: str) -> OfficialALFWorldStepResult:
        row = current_progress()
        with row.environment_command(action) if row is not None else nullcontext():
            result = _object(
                self.client.request("step", {"action": _text(action, field_name="action")}),
                fields={"admissible_commands", "observation_text", "success", "terminal"},
                label="ALFWorld step result",
            )
        if row is not None:
            row.observation(result["observation_text"])
        step = OfficialALFWorldStepResult(
            observation_text=cast(str, result["observation_text"]),
            admissible_commands=_string_array(
                result["admissible_commands"], label="ALFWorld admissible commands"
            ),
            terminal=cast(bool, result["terminal"]),
            success=cast(bool | None, result["success"]),
        )
        if step.terminal:
            self.client.close_successfully()
        return step

    def close_after_preparation_failure(self) -> None:
        """Release only this owned worker before an episode handle exists."""
        if self.client._state is WorkerLifecycleState.OPEN:
            self.client.close_successfully()

    async def close(self) -> None:
        """Release a live worker when a rollout ends before environment terminal."""

        if self.client._state is WorkerLifecycleState.OPEN:
            await asyncio.to_thread(self.client.close_successfully)


@dataclass(frozen=True, slots=True)
class OfficialScienceWorldProcessFactory:
    runtime: PinnedOfficialProcess
    jar_path: Path
    simplification: str
    seed: int
    private_diagnostics: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, PinnedOfficialProcess):
            raise TypeError("ScienceWorld process factory requires a pinned runtime")
        jar = _path(self.jar_path, field_name="ScienceWorld jar_path", directory=False)
        simplification = self.simplification
        if type(simplification) is not str or "\x00" in simplification:
            raise ValueError("ScienceWorld simplification must be text without NUL")
        _seed(self.seed)
        object.__setattr__(self, "jar_path", jar)

    def create(self, task: OfficialScienceWorldTask) -> OfficialScienceWorldTextEnv:
        if not isinstance(task, OfficialScienceWorldTask):
            raise TypeError("ScienceWorld process factory requires OfficialScienceWorldTask")
        if task.seed != self.seed:
            raise ValueError("ScienceWorld task seed differs from its deployment identity")
        client = OfficialWorkerClient(self.runtime)
        _initialize_or_discard(
            client,
            _initialization_payload(
                benchmark="scienceworld",
                runtime=self.runtime,
                deployment={
                    "jar_path": str(self.jar_path),
                    "seed": self.seed,
                    "simplification": self.simplification,
                    "private_diagnostics": self.private_diagnostics,
                },
                task={
                    "max_steps": task.max_steps,
                    "task_name": task.task_name,
                    "variation_index": task.variation_index,
                },
            ),
        )
        return _ScienceWorldProcessEnv(task=task, client=client)


@dataclass(slots=True)
class _ScienceWorldProcessEnv:
    task: OfficialScienceWorldTask
    client: OfficialWorkerClient = field(repr=False)
    _task_description: str | None = field(default=None, init=False)
    public_state: dict[str, str] = field(default_factory=dict, init=False)
    reset_score: float | None = field(default=None, init=False)
    native_moves: int | None = field(default=None, init=False)

    @property
    def task_name(self) -> str:
        return self.task.task_name

    @property
    def variation_index(self) -> int:
        return self.task.variation_index

    @property
    def seed(self) -> int:
        return self.task.seed

    @property
    def max_steps(self) -> int:
        return self.task.max_steps

    @property
    def task_description(self) -> str:
        if self._task_description is None:
            raise OfficialEnvironmentInfrastructureError("ScienceWorld environment is not reset")
        return self._task_description

    def reset(
        self,
        *,
        task_name: str,
        variation_index: int,
        seed: int,
        max_steps: int,
    ) -> str:
        if (
            task_name != self.task.task_name
            or variation_index != self.task.variation_index
            or seed != self.task.seed
            or max_steps != self.task.max_steps
        ):
            raise ValueError("ScienceWorld reset identity differs from its pinned task")
        result = _object(
            self.client.request("reset", {}),
            fields={"observation_text", "task_description"},
            label="ScienceWorld reset result",
            optional_fields=frozenset({"public_state", "native_score", "native_moves"}),
        )
        self.public_state = cast(dict[str, str], result.get("public_state", {}))
        raw_score = result.get("native_score")
        self.reset_score = (
            OfficialScienceWorldStepResult("reset", cast(float, raw_score), False).raw_score
            if raw_score is not None
            else None
        )
        self.native_moves = cast(int | None, result.get("native_moves"))
        self._task_description = _text(
            result["task_description"], field_name="ScienceWorld task description"
        )
        return _text(result["observation_text"], field_name="ScienceWorld observation")

    def step(self, action: str) -> OfficialScienceWorldStepResult:
        result = _object(
            self.client.request("step", {"action": _text(action, field_name="action")}),
            fields={"observation_text", "score", "terminal"},
            label="ScienceWorld step result",
            optional_fields=frozenset({"public_state", "native_moves", "private_diagnostics"}),
        )
        step = OfficialScienceWorldStepResult(
            observation_text=cast(str, result["observation_text"]),
            score=cast(float, result["score"]),
            terminal=cast(bool, result["terminal"]),
            public_state=cast(dict[str, str], result.get("public_state", {})),
            native_moves=cast(int | None, result.get("native_moves")),
            private_diagnostics=cast(dict[str, object], result.get("private_diagnostics", {})),
        )
        self.public_state, self.native_moves = step.public_state, step.native_moves
        if step.terminal:
            self.client.close_successfully()
        return step

    async def close(self) -> None:
        """Release a live worker when a rollout ends before environment terminal."""

        if self.client._state is WorkerLifecycleState.OPEN:
            await asyncio.to_thread(self.client.close_successfully)


__all__ = [
    "WEBSHOP_JSON_STORAGE_FORMAT",
    "WEBSHOP_SQLITE_STORAGE_FORMAT",
    "ALFWorldGameDeployment",
    "ActiveWebShopEnvironment",
    "ClosedWebShopEnvironment",
    "JsonArrayWebShopDeployment",
    "OfficialALFWorldProcessFactory",
    "OfficialEnvironmentInfrastructureError",
    "OfficialScienceWorldProcessFactory",
    "OfficialWebShopProcessFactory",
    "OfficialWorkerClient",
    "PinnedOfficialProcess",
    "SQLiteWebShopDeployment",
    "UninitializedWebShopEnvironment",
    "WebShopDeployment",
    "WebShopLifecycle",
    "WorkerLifecycleState",
]
