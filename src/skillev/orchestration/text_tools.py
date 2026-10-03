from __future__ import annotations

import asyncio
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import Any, Protocol

from skillev.contracts.canonical import JsonValue, normalize_json, stable_hash
from skillev.runtime.contracts import BudgetVector

from .retrieval_client import (
    SERVICE_FORMAT,
    RetrievalClient,
    RetrievalIdentityError,
    RetrievalQueryError,
    RetrievalServiceError,
    format_search_r1,
)

KNOWN_TEXT_TOOLS = frozenset({"python", "search"})
CALL = re.compile(r"<(python|search)>(.*?)</\1>", re.S)
STOP_STRINGS = ("</python>", "</search>")
TOOL_STATUSES = frozenset({"success", "failed", "timeout", "tool_error"})
_TAG = re.compile(r"</?(?:python|search|result)>")
RESULT_BLOCK = re.compile(r"<result>.*?(?:</result>|$)", re.S)
_FENCED = re.compile(r"\A\s*```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)\n?[ \t]*```\s*\Z", re.S | re.I)
_SEARCH_QUERY_CHARS = 300
MIN_TURN_OUTPUT = 256

ISOLATION_ENV = "EVOSTEER_PYTHON_TOOL_ISOLATION"
FS_ENV = "EVOSTEER_PYTHON_TOOL_FS"
FS_MODES = ("", "landlock")
CONCURRENCY_ENV = "EVOSTEER_PYTHON_TOOL_CONCURRENCY"
ISOLATION_TIERS = ("unshare-net+setpriv-nobody", "setpriv-nobody", "unshare-net", "rlimit-prelude")
_UNPRIVILEGED_TIERS = frozenset({"unshare-net+setpriv-nobody", "setpriv-nobody"})
_NOBODY = 65534
_FILE_BYTES = 8 << 20
_OPEN_FILES = 64
_NOBODY_PROCESSES = 64
_CONTAINMENT_WAIT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: str
    argument: str
    start: int
    end: int


def _python_source(raw: str) -> str:
    fenced = _FENCED.match(raw)
    if fenced:
        raw = fenced.group(1)
    return textwrap.dedent(raw.strip("\r\n")).strip()


def _search_query(raw: str) -> str:
    return " ".join(raw.split())[:_SEARCH_QUERY_CHARS]


def first_call(text: str) -> ToolCall | None:
    match = CALL.search(text)
    if match is None:
        return None
    tool, raw = match.group(1), match.group(2)
    argument = _python_source(raw) if tool == "python" else _search_query(raw)
    return ToolCall(tool, argument, match.start(), match.end())


def normalize_final(text: str) -> str:
    if _TAG.search(text) is None:
        return text

    def replace(match: re.Match[str]) -> str:
        if match.group(1) == "python":
            return "```python\n" + _python_source(match.group(2)) + "\n```"
        return ""

    value = CALL.sub(replace, text)
    value = RESULT_BLOCK.sub("", value)
    return _TAG.sub("", value).strip()


def head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n[... {} characters omitted ...]\n"
    keep = max(0, limit - len(marker.format(len(text))))
    head = keep // 3
    tail = keep - head
    return text[:head] + marker.format(len(text) - keep) + text[len(text) - tail :]


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class TextToolsConfig:
    family_tools: tuple[tuple[str, tuple[str, ...]], ...]
    role_maximum: tuple[tuple[str, BudgetVector], ...] = ()
    python_cpu_seconds: int = 10
    python_wall_seconds: int = 15
    python_memory_mb: int = 4096
    search_top_k: int = 3
    search_passage_chars: int = 600
    search_timeout_seconds: int = 30
    result_chars: int = 2000
    tool_turn_output_tokens: int = 4096
    final_output_reserve: int = 1024
    tool_only_roles: tuple[str, ...] = ()
    role_tool_instructions: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    family_role_instructions: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.family_tools, tuple) or not self.family_tools:
            raise ValueError("text_tools.family_tools must name at least one family")
        for item in self.family_tools:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0].strip()
                or not isinstance(item[1], tuple)
            ):
                raise TypeError("text_tools.family_tools must hold (family, tools) pairs")
            tools = item[1]
            if not tools or list(tools) != sorted(set(tools)) or not set(tools) <= KNOWN_TEXT_TOOLS:
                raise ValueError(
                    f"tools of {item[0]!r} must be a sorted, distinct, nonempty subset of "
                    f"{sorted(KNOWN_TEXT_TOOLS)}"
                )
        families = [family for family, _ in self.family_tools]
        if families != sorted(set(families)):
            raise ValueError("text_tools.family_tools must name distinct families, sorted")
        if not isinstance(self.role_maximum, tuple):
            raise TypeError("text_tools.role_maximum must be a tuple of (role, BudgetVector)")
        for item in self.role_maximum:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0].strip()
                or not isinstance(item[1], BudgetVector)
            ):
                raise TypeError("text_tools.role_maximum must hold (role, BudgetVector) pairs")
            role_id, maximum = item
            if maximum.model_calls < 2 or maximum.tool_calls < 1:
                raise ValueError(
                    f"{role_id} tool envelope needs >= 2 model calls and >= 1 tool call"
                )
            if min(maximum.input_tokens, maximum.output_tokens, maximum.agent_turns) < 1:
                raise ValueError(f"{role_id} tool envelope must admit tokens and an agent turn")
            if maximum.wall_time_milliseconds < 1:
                raise ValueError(f"{role_id} tool envelope must admit wall time")
        roles = [role for role, _ in self.role_maximum]
        if roles != sorted(set(roles)):
            raise ValueError("text_tools.role_maximum must name distinct roles, sorted")
        only = self.tool_only_roles
        if (
            not isinstance(only, tuple)
            or any(type(role) is not str or not role.strip() for role in only)
            or list(only) != sorted(set(only))
        ):
            raise ValueError("text_tools.tool_only_roles must be distinct role ids, sorted")
        sentences = self.role_tool_instructions
        if not isinstance(sentences, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0].strip()
            or not isinstance(item[1], tuple)
            for item in sentences
        ):
            raise TypeError("text_tools.role_tool_instructions must hold (role, pairs) items")
        named = [role for role, _ in sentences]
        if named != sorted(set(named)):
            raise ValueError("text_tools.role_tool_instructions must name distinct roles, sorted")
        for role_id, pairs in sentences:
            if (
                not pairs
                or any(
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or type(pair[0]) is not str
                    or type(pair[1]) is not str
                    for pair in pairs
                )
                or [tool for tool, _ in pairs] != sorted({tool for tool, _ in pairs})
                or not {tool for tool, _ in pairs} <= KNOWN_TEXT_TOOLS
            ):
                raise ValueError(
                    f"text_tools.role_tool_instructions[{role_id!r}] must map distinct tools "
                    f"among {sorted(KNOWN_TEXT_TOOLS)}, sorted, to sentences"
                )
            if any(not text.strip() or text != text.strip() for _, text in pairs):
                raise ValueError(
                    f"text_tools.role_tool_instructions[{role_id!r}] sentences must be "
                    "nonempty and carry no surrounding whitespace"
                )
        texts = self.family_role_instructions
        if not isinstance(texts, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or type(item[0]) is not str
            or not isinstance(item[1], tuple)
            for item in texts
        ):
            raise TypeError("text_tools.family_role_instructions must hold (family, pairs) items")
        named = [family for family, _ in texts]
        if named != sorted(set(named)) or not set(named) <= {name for name, _ in self.family_tools}:
            raise ValueError(
                "text_tools.family_role_instructions must name distinct families with tools, sorted"
            )
        for family, pairs in texts:
            if (
                not pairs
                or any(
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or type(pair[0]) is not str
                    or type(pair[1]) is not str
                    or not pair[1].strip()
                    or pair[1] != pair[1].strip()
                    for pair in pairs
                )
                or [role for role, _ in pairs] != sorted({role for role, _ in pairs})
            ):
                raise ValueError(
                    f"text_tools.family_role_instructions[{family!r}] must map distinct roles, "
                    "sorted, to nonempty instructions without surrounding whitespace"
                )
        nested = {
            "family_tools",
            "role_maximum",
            "tool_only_roles",
            "role_tool_instructions",
            "family_role_instructions",
        }
        for item in fields(self):
            if item.name not in nested:
                _positive_int(getattr(self, item.name), f"text_tools.{item.name}")
        if self.tool_turn_output_tokens < MIN_TURN_OUTPUT:
            raise ValueError(f"text_tools.tool_turn_output_tokens must be >= {MIN_TURN_OUTPUT}")

    def tools_for(self, family: str) -> tuple[str, ...]:
        for name, tools in self.family_tools:
            if name == family:
                return tools
        return ()

    def maximum_for(self, role_id: str) -> BudgetVector | None:
        for name, maximum in self.role_maximum:
            if name == role_id:
                return maximum
        return None

    def tool_instructions_for(self, role_id: str) -> tuple[tuple[str, str], ...]:
        for name, pairs in self.role_tool_instructions:
            if name == role_id:
                return pairs
        return ()

    def family_instruction_for(self, family: str, role_id: str) -> str | None:
        for name, pairs in self.family_role_instructions:
            if name == family:
                return dict(pairs).get(role_id)
        return None

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted({tool for _, tools in self.family_tools for tool in tools}))

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "family_tools": {family: list(tools) for family, tools in self.family_tools},
            "role_maximum": {role: maximum.to_value() for role, maximum in self.role_maximum},
        }
        for item in fields(self):
            if item.name == "tool_only_roles":
                if self.tool_only_roles:
                    value[item.name] = list(self.tool_only_roles)
            elif item.name == "role_tool_instructions":
                if self.role_tool_instructions:
                    value[item.name] = {
                        role: dict(pairs) for role, pairs in self.role_tool_instructions
                    }
            elif item.name == "family_role_instructions":
                if self.family_role_instructions:
                    value[item.name] = {
                        family: dict(pairs) for family, pairs in self.family_role_instructions
                    }
            elif item.name not in value:
                value[item.name] = getattr(self, item.name)
        return value

    @classmethod
    def from_value(cls, value: object) -> TextToolsConfig:
        if not isinstance(value, Mapping):
            raise TypeError("text_tools must be a JSON object")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown text_tools keys: {unknown}")
        raw_families = value.get("family_tools")
        if not isinstance(raw_families, Mapping) or not raw_families:
            raise ValueError("text_tools.family_tools must be a nonempty object")
        family_tools = []
        for family, tools in raw_families.items():
            if (
                not isinstance(tools, list)
                or not tools
                or any(type(tool) is not str for tool in tools)
                or len(set(tools)) != len(tools)
            ):
                raise ValueError(f"text_tools.family_tools[{family!r}] must list distinct tools")
            family_tools.append((family, tuple(sorted(tools))))
        raw_roles = value.get("role_maximum", {})
        if not isinstance(raw_roles, Mapping):
            raise ValueError("text_tools.role_maximum must be an object")
        role_maximum = tuple(
            sorted((role, BudgetVector.from_value(budget)) for role, budget in raw_roles.items())
        )
        raw_only = value.get("tool_only_roles", [])
        if not isinstance(raw_only, list) or len(set(map(str, raw_only))) != len(raw_only):
            raise ValueError("text_tools.tool_only_roles must list distinct role ids")
        raw_sentences = value.get("role_tool_instructions", {})
        if not isinstance(raw_sentences, Mapping) or any(
            not isinstance(pairs, Mapping) for pairs in raw_sentences.values()
        ):
            raise ValueError(
                "text_tools.role_tool_instructions must be an object of role -> {tool: sentence}"
            )
        role_tool_instructions = tuple(
            sorted((role, tuple(sorted(pairs.items()))) for role, pairs in raw_sentences.items())
        )
        raw_texts = value.get("family_role_instructions", {})
        if not isinstance(raw_texts, Mapping) or any(
            not isinstance(pairs, Mapping) for pairs in raw_texts.values()
        ):
            raise ValueError(
                "text_tools.family_role_instructions must be an object of family -> {role: text}"
            )
        family_role_instructions = tuple(
            sorted((family, tuple(sorted(pairs.items()))) for family, pairs in raw_texts.items())
        )
        nested = {
            "family_tools",
            "role_maximum",
            "tool_only_roles",
            "role_tool_instructions",
            "family_role_instructions",
        }
        rest = {key: item for key, item in value.items() if key not in nested}
        return cls(
            tuple(sorted(family_tools)),
            role_maximum,
            **rest,
            tool_only_roles=tuple(sorted(raw_only)),
            role_tool_instructions=role_tool_instructions,
            family_role_instructions=family_role_instructions,
        )


@dataclass(frozen=True, slots=True)
class ToolRun:
    status: str
    text: str
    wall_ms: int
    record: dict[str, JsonValue] = field(default_factory=dict)
    sandbox: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if self.status not in TOOL_STATUSES:
            raise ValueError(f"unknown tool status {self.status!r}")
        if type(self.text) is not str:
            raise TypeError("tool result text must be text")
        if type(self.wall_ms) is not int or self.wall_ms < 0:
            raise ValueError("tool wall time must be a nonnegative integer")
        object.__setattr__(self, "record", normalize_json(self.record))
        if self.sandbox is not None:
            object.__setattr__(self, "sandbox", normalize_json(self.sandbox))


class TextTool(Protocol):
    name: str

    @property
    def identity(self) -> str: ...

    @property
    def wall_reserve_ms(self) -> int: ...

    async def run(self, argument: str) -> ToolRun: ...


class SandboxUnavailableError(RuntimeError):
    pass


def _tier_prefix(tier: str, unshare_flags: tuple[str, ...]) -> tuple[str, ...]:
    setpriv = (
        "setpriv",
        f"--reuid={_NOBODY}",
        f"--regid={_NOBODY}",
        "--clear-groups",
        "--no-new-privs",
        "--",
    )
    if tier == "unshare-net+setpriv-nobody":
        return ("unshare", *unshare_flags, "--", *setpriv)
    if tier == "setpriv-nobody":
        return setpriv
    if tier == "unshare-net":
        return ("unshare", *unshare_flags, "--")
    return ()


_UNSHARE_CANDIDATES = (
    ("--net", "--pid", "--fork", "--mount-proc"),
    ("--net", "--pid", "--fork"),
    ("--net",),
    ("--map-root-user", "--net", "--pid", "--fork", "--mount-proc"),
    ("--map-root-user", "--net", "--pid", "--fork"),
    ("--map-root-user", "--net"),
)
_PROBES: dict[str, dict[str, tuple[str, ...]]] = {}
_PROBE_LOCK = threading.Lock()


def _probe(command: tuple[str, ...], expect_uid: int | None) -> bool:
    try:
        done = subprocess.run(
            [*command, "-c", "import os; print(os.getuid())"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=20,
            check=False,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if done.returncode != 0:
        return False
    return expect_uid is None or done.stdout.strip() == str(expect_uid).encode()


def available_isolation(python: str = sys.executable) -> dict[str, tuple[str, ...]]:
    with _PROBE_LOCK:
        cached = _PROBES.get(python)
        if cached is not None:
            return cached
        found: dict[str, tuple[str, ...]] = {}
        if sys.platform.startswith("linux"):
            unshare = next(
                (
                    flags
                    for flags in _UNSHARE_CANDIDATES
                    if shutil.which("unshare") and _probe(("unshare", *flags, "--", python), None)
                ),
                None,
            )
            for tier in ISOLATION_TIERS[:-1]:
                needs_unshare = tier.startswith("unshare-net")
                if needs_unshare and unshare is None:
                    continue
                if tier == "unshare-net+setpriv-nobody" and "--map-root-user" in unshare:
                    continue
                if "setpriv" in tier and not shutil.which("setpriv"):
                    continue
                prefix = _tier_prefix(tier, unshare or ())
                if _probe((*prefix, python), _NOBODY if tier in _UNPRIVILEGED_TIERS else None):
                    found[tier] = prefix
        found["rlimit-prelude"] = ()
        _PROBES[python] = found
        return found


def select_isolation(requested: str | None, available: Mapping[str, tuple[str, ...]]) -> str:
    if requested:
        name = "rlimit-prelude" if requested == "rlimit" else requested
        if name not in ISOLATION_TIERS:
            raise ValueError(
                f"unknown python tool isolation {requested!r}; use one of {ISOLATION_TIERS}"
            )
        if name not in available:
            raise SandboxUnavailableError(
                f"python tool isolation {name!r} is not available here (have {sorted(available)})"
            )
        return name
    best = next(tier for tier in ISOLATION_TIERS if tier in available)
    if best not in _UNPRIVILEGED_TIERS:
        raise SandboxUnavailableError(
            f"python tools would run as the trainer's user ({best}); install setpriv, run as root, "
            f"or accept it explicitly with {ISOLATION_ENV}=rlimit"
        )
    return best


_POOL_LOCK = threading.Lock()
_PYTHON_POOL: ThreadPoolExecutor | None = None


def _python_pool() -> ThreadPoolExecutor:
    global _PYTHON_POOL
    with _POOL_LOCK:
        if _PYTHON_POOL is None:
            raw = os.environ.get(CONCURRENCY_ENV, "").strip()
            workers = int(raw) if raw else 4
            if workers < 1:
                raise ValueError(f"{CONCURRENCY_ENV} must be a positive integer")
            _PYTHON_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="python-tool")
        return _PYTHON_POOL


def _group_alive(pgid: int) -> bool:
    proc = "/proc"
    if os.path.isdir(proc):
        for entry in os.listdir(proc):
            if not entry.isdigit():
                continue
            try:
                with open(f"{proc}/{entry}/stat", "rb") as handle:
                    stat = handle.read()
            except OSError:
                continue
            rest = stat[stat.rfind(b")") + 2 :].split()
            if len(rest) > 2 and int(rest[2]) == pgid and rest[0] not in {b"Z", b"X"}:
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


_LANDLOCK_READ = ("/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/etc", "/proc", "/sys", "/dev")
LANDLOCK_FAILED_STATUS = 97

_LANDLOCK_SOURCE = """\
def _landlock():
    import ctypes, os, struct, sys
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    create, add_rule, restrict = 444, 445, 446
    abi = libc.syscall(create, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 1:
        raise OSError(ctypes.get_errno(), "Landlock is not available")
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13  # REFER
    if abi >= 3:
        handled |= 1 << 14  # TRUNCATE
    if abi >= 5:
        handled |= 1 << 15  # IOCTL_DEV
    attr = struct.pack("=QQ", handled, 3) if abi >= 4 else struct.pack("=Q", handled)
    ruleset = libc.syscall(create, ctypes.c_char_p(attr), ctypes.c_size_t(len(attr)), ctypes.c_uint32(0))
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    read = (1 << 0) | (1 << 2) | (1 << 3)  # EXECUTE, READ_FILE, READ_DIR
    roots = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)}
    rules = [(path, read) for path in sorted(roots | set(__READ__))]
    rules += [(os.getcwd(), handled), ("/dev/null", (1 << 1) | (1 << 2) | (handled & (1 << 14))), ("/dev/shm", handled)]
    for path, access in rules:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            continue
        if os.path.isfile(path):
            access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14) | (1 << 15)
        rule = struct.pack("=Qi", access & handled, fd)
        done = libc.syscall(add_rule, ctypes.c_int(ruleset), ctypes.c_int(1), ctypes.c_char_p(rule), ctypes.c_uint32(0))
        os.close(fd)
        if done < 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "no_new_privs")
    if libc.syscall(restrict, ctypes.c_int(ruleset), ctypes.c_uint32(0)) != 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(ruleset)


try:
    _landlock()
except BaseException as _error:
    import sys as _early_sys
    print(f"[sandbox setup failed: {_error}]", file=_early_sys.stderr)
    _early_sys.exit(__FAILED__)
del _landlock
"""


def _runner_source(
    cpu_seconds: int, memory_bytes: int, unprivileged: bool, landlock: bool = False
) -> str:
    linux_limits = [f"_limit('RLIMIT_AS', {memory_bytes})"]
    if unprivileged:
        linux_limits.append(f"_limit('RLIMIT_NPROC', {_NOBODY_PROCESSES})")
    else:
        linux_limits.append(f"_limit('RLIMIT_NPROC', _own_processes() + {_NOBODY_PROCESSES})")
    linux = "\n    ".join(linux_limits)
    confine = ""
    if landlock:
        confine = _LANDLOCK_SOURCE.replace("__READ__", repr(_LANDLOCK_READ)).replace(
            "__FAILED__", str(LANDLOCK_FAILED_STATUS)
        )
    return f"""\
{confine}import resource as _resource
import sys as _sys


def _limit(name, value):
    _resource.setrlimit(getattr(_resource, name), (value, value))


def _own_processes():
    # RLIMIT_NPROC counts tasks (threads), not processes, of the real uid.
    import os
    uid, count = os.getuid(), 0
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                if os.stat("/proc/" + entry).st_uid == uid:
                    count += len(os.listdir("/proc/" + entry + "/task"))
            except OSError:
                pass
    return count


if _sys.platform.startswith("linux"):
    {linux}
_limit("RLIMIT_CPU", {cpu_seconds})
_limit("RLIMIT_FSIZE", {_FILE_BYTES})
_limit("RLIMIT_NOFILE", {_OPEN_FILES})
del _limit, _resource, _own_processes

import _socket
import socket as _socket_module


class _NoNetwork(_socket_module.socket):
    def __init__(self, *args, **kwargs):
        raise PermissionError("network access is disabled in this sandbox")


def _no_network(*args, **kwargs):
    raise PermissionError("network access is disabled in this sandbox")


_socket_module.socket = _socket.socket = _NoNetwork
for _name in ("socketpair", "create_connection", "create_server", "fromfd", "getaddrinfo"):
    setattr(_socket_module, _name, _no_network)
_socket.getaddrinfo = _no_network
del _name, _socket, _socket_module

import traceback as _traceback

with open("main.py", encoding="utf-8") as _handle:
    _source = _handle.read()
try:
    _code = compile(_source, "main.py", "exec")
    exec(_code, {{"__name__": "__main__", "__builtins__": __builtins__}})
except SystemExit:
    raise
except BaseException as _error:
    _traceback.print_exception(type(_error), _error, _error.__traceback__.tb_next)
    _sys.exit(1)
"""


class PythonSandbox:
    name = "python"

    def __init__(
        self,
        *,
        cpu_seconds: int = 10,
        wall_seconds: int = 15,
        memory_mb: int = 4096,
        result_chars: int = 2000,
        isolation: str | None = None,
        python: str = sys.executable,
        fs: str | None = None,
    ) -> None:
        for label, value in (
            ("cpu_seconds", cpu_seconds),
            ("wall_seconds", wall_seconds),
            ("memory_mb", memory_mb),
            ("result_chars", result_chars),
        ):
            _positive_int(value, label)
        available = available_isolation(python)
        requested = isolation
        if requested is None:
            requested = os.environ.get(ISOLATION_ENV, "").strip()
        self.isolation = select_isolation(requested or None, available)
        self.fs = (os.environ.get(FS_ENV, "") if fs is None else fs).strip()
        if self.fs not in FS_MODES:
            raise ValueError(f"unknown python tool fs confinement {self.fs!r}; use one of {FS_MODES}")
        self._prefix = available[self.isolation]
        self.cpu_seconds, self.wall_seconds = cpu_seconds, wall_seconds
        self.memory_mb, self.result_chars = memory_mb, result_chars
        self._python = python
        self._unprivileged = self.isolation in _UNPRIVILEGED_TIERS
        self._identity = str(
            stable_hash(
                {
                    "tool": "python",
                    "format": "python-sandbox@1",
                    "isolation": self.isolation,
                    "cpu_seconds": cpu_seconds,
                    "wall_seconds": wall_seconds,
                    "memory_mb": memory_mb,
                    "file_bytes": _FILE_BYTES,
                    "open_files": _OPEN_FILES,
                    "result_chars": result_chars,
                    "python": platform.python_version(),
                    **({"fs": self.fs} if self.fs else {}),
                    **({"prefix": list(self._prefix)} if self.isolation.startswith("unshare") else {}),
                }
            )
        )
        if self.fs:
            check = self.execute("print(6 * 7)")
            if check.status != "success" or check.text.strip() != "42":
                raise SandboxUnavailableError(f"python tool fs confinement {self.fs!r} failed: {check.text[:300]}")

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def wall_reserve_ms(self) -> int:
        return (self.wall_seconds + 5) * 1000

    def _environment(self, home: str) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": home,
            "TMPDIR": home,
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }

    def execute(self, code: str) -> ToolRun:
        began = time.monotonic()
        home = tempfile.mkdtemp(prefix="evosteer-python-")
        try:
            if self._unprivileged:
                os.chown(home, _NOBODY, _NOBODY)
            os.chmod(home, 0o700 if self._unprivileged else 0o755)
            for name, text in (
                ("main.py", code),
                (
                    "runner.py",
                    _runner_source(
                        self.cpu_seconds, self.memory_mb << 20, self._unprivileged, self.fs == "landlock"
                    ),
                ),
            ):
                path = os.path.join(home, name)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text)
                os.chmod(path, 0o444)
            command = [*self._prefix, self._python, "-u", "-s", "-P", "runner.py"]
            with tempfile.TemporaryFile() as output:
                process = subprocess.Popen(
                    command,
                    cwd=home,
                    env=self._environment(home),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    process.wait(timeout=self.wall_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                exit_status = process.wait()
                deadline = time.monotonic() + _CONTAINMENT_WAIT_SECONDS
                contained = not _group_alive(process.pid)
                while not contained and time.monotonic() < deadline:
                    time.sleep(0.02)
                    contained = not _group_alive(process.pid)
                output.seek(0)
                raw = output.read(_FILE_BYTES + 65536)
        finally:
            shutil.rmtree(home, ignore_errors=True)
        wall_ms = math.ceil((time.monotonic() - began) * 1000)
        text = raw.decode("utf-8", "replace")
        cpu_killed = exit_status in (-signal.SIGXCPU, -signal.SIGKILL) and not timed_out
        if timed_out or cpu_killed:
            status = "timeout"
            note = f"[stopped: the {self.wall_seconds} s time limit was reached]"
        elif exit_status == 0:
            status, note = "success", None
        elif exit_status == LANDLOCK_FAILED_STATUS and text.startswith("[sandbox setup failed"):
            status, note = "tool_error", None
        else:
            status, note = "failed", f"[exit status {exit_status}]"
        visible = text if text.strip() else "(no output)"
        if note:
            visible = f"{visible.rstrip()}\n{note}"
        return ToolRun(
            status,
            head_tail(visible, self.result_chars),
            wall_ms,
            {
                "exit_status": exit_status,
                "timed_out": timed_out,
                "cpu_limited": cpu_killed,
                "output_chars": len(text),
                "code_chars": len(code),
            },
            {"isolation": self.isolation, "contained": contained},
        )

    async def run(self, argument: str) -> ToolRun:
        queued = time.monotonic()

        def job() -> ToolRun:
            waited_ms = int((time.monotonic() - queued) * 1000)
            result = self.execute(argument)
            return ToolRun(
                result.status,
                result.text,
                result.wall_ms,
                {**result.record, "queued_ms": waited_ms},
                result.sandbox,
            )

        return await asyncio.get_running_loop().run_in_executor(_python_pool(), job)


class SearchUnavailableError(RuntimeError):
    pass


class SearchBreaker:
    def __init__(self, threshold: int = 20) -> None:
        self.threshold = _positive_int(threshold, "breaker threshold")
        self._failures = 0
        self._lock = threading.Lock()

    @property
    def failures(self) -> int:
        return self._failures

    def success(self) -> None:
        with self._lock:
            self._failures = 0

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            failures = self._failures
        if failures >= self.threshold:
            raise SearchUnavailableError(
                f"retrieval failed {failures} times in a row; stopping instead of training "
                "on a dead search tool"
            )


SEARCH_BREAKER = SearchBreaker()


class SearchTool:
    name = "search"

    def __init__(
        self,
        client: RetrievalClient,
        *,
        passage_chars: int = 600,
        result_chars: int = 2000,
        breaker: SearchBreaker | None = None,
    ) -> None:
        if client.identity is None:
            raise ValueError("pin the retrieval service (RetrievalClient.wait_ready) first")
        self._client = client
        self.passage_chars = _positive_int(passage_chars, "passage_chars")
        self.result_chars = _positive_int(result_chars, "result_chars")
        self._breaker = breaker or SEARCH_BREAKER
        identity: dict[str, JsonValue] = {
            "tool": "search",
            "format": "search-tool@1",
            "service_format": SERVICE_FORMAT,
            "service": client.identity,
            "top_k": client.top_k,
            "passage_chars": passage_chars,
            "result_chars": result_chars,
            "query_chars": _SEARCH_QUERY_CHARS,
            "timeout_seconds": client.timeout_seconds,
        }
        self._identity = str(stable_hash(identity))

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def wall_reserve_ms(self) -> int:
        return math.ceil(self._client.timeout_seconds + 5) * 1000

    async def run(self, argument: str) -> ToolRun:
        query = _search_query(argument)
        if not query:
            return ToolRun(
                "tool_error",
                "Search error: the query is empty.",
                0,
                {"query": query, "error": "empty_query"},
            )

        def timed() -> tuple[Any, BaseException | None, float]:
            began = time.monotonic()
            try:
                return self._client.search(query), None, time.monotonic() - began
            except RetrievalIdentityError:
                raise
            except (RetrievalServiceError, RetrievalQueryError) as error:
                return None, error, time.monotonic() - began

        result, error, seconds = await self._client.in_pool(timed)
        wall_ms = math.ceil(seconds * 1000)
        if isinstance(error, RetrievalQueryError):
            return ToolRun(
                "tool_error",
                f"Search error: {error}",
                wall_ms,
                {"query": query, "error": "query_rejected"},
            )
        if error is not None:
            self._breaker.failure()
            return ToolRun(
                "tool_error",
                "Search error: the search service did not answer; no results.",
                wall_ms,
                {"query": query, "error": "service_unavailable"},
            )
        self._breaker.success()
        text = format_search_r1(result.hits, self.passage_chars)
        if len(text) > self.result_chars:
            omitted = len(text) - self.result_chars
            text = text[: self.result_chars] + f"\n[... {omitted} characters omitted ...]"
        return ToolRun("success", text, wall_ms, result.to_value())


__all__ = [
    "CALL",
    "CONCURRENCY_ENV",
    "ISOLATION_ENV",
    "FS_ENV",
    "FS_MODES",
    "LANDLOCK_FAILED_STATUS",
    "ISOLATION_TIERS",
    "KNOWN_TEXT_TOOLS",
    "MIN_TURN_OUTPUT",
    "SEARCH_BREAKER",
    "STOP_STRINGS",
    "PythonSandbox",
    "SandboxUnavailableError",
    "SearchBreaker",
    "SearchTool",
    "SearchUnavailableError",
    "TextTool",
    "TextToolsConfig",
    "ToolCall",
    "ToolRun",
    "available_isolation",
    "first_call",
    "head_tail",
    "normalize_final",
    "select_isolation",
]
