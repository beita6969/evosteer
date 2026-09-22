#!/usr/bin/env python3
"""Run the released SWE-bench IID panel with an adapter-free direct Qwen agent.

The formal contract gives one Qwen read-only repository navigation followed by
one final unified-diff submission. No MExec delegation, editable workspace
feedback, adapter, or learned skill is available. Only the public issue,
repository identity, and base revision enter an episode. Gold patches and
official tests remain outside generation and are consumed later by the separate
official scorer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from skillev_private.direct_reference.manifests import load_population_manifest
from skillev_private.direct_reference.populations import load_skillflow_iid_cases

from skillev.evaluation.direct_baseline import DirectBenchmark
from skillev.evaluation.direct_baseline.parsing import parse_unified_diff
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol

LOGGER = logging.getLogger("qwen35-swe-repo-agent")
EXPECTED_COUNT = 128
EXPECTED_STEPS = 28
DIRECT_REPOSITORY_AGENT_PROMPTS = frozenset(
    {
        "qwen-direct-repository-agent@1",
        "qwen-direct-repository-agent-raw-tools@1",
        "qwen-direct-readonly-repository-agent@1",
    }
)
DIRECT_RAW_TOOLS_PROMPT = "qwen-direct-repository-agent-raw-tools@1"
DIRECT_READONLY_PROMPT = "qwen-direct-readonly-repository-agent@1"
SUPERVISOR_TEMPERATURE = 0.8
SUPERVISOR_MAX_TOKENS = 512
EXECUTOR_TEMPERATURE = 0.1
EXECUTOR_MAX_TOKENS = 8192
_REPOSITORY_NAME = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, slots=True)
class SWERepositoryTask:
    task_id: str
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    task_id: str
    instance_id: str
    attempt: int
    state: str
    model_patch: str | None
    infrastructure_error: str | None
    endpoint_index: int
    elapsed_seconds: float
    steps: int
    supervisor_calls: int
    transcript: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if self.state not in {"candidate", "candidate_failure", "infrastructure_failure"}:
            raise ValueError("unknown episode outcome state")
        if self.state == "candidate":
            if not self.model_patch or self.infrastructure_error is not None:
                raise ValueError("candidate outcome requires one patch and no infrastructure error")
        elif self.state == "candidate_failure":
            if self.model_patch is not None or self.infrastructure_error is not None:
                raise ValueError("candidate failure cannot carry a patch or infrastructure error")
        elif self.model_patch is not None or not self.infrastructure_error:
            raise ValueError("infrastructure failure requires an error and no patch")


@dataclass(frozen=True, slots=True)
class UpstreamRuntime:
    environment_type: type[Any]
    executor_type: type[Any]
    workspace_type: type[Any]
    supervisor_call: Callable[..., tuple[str, str | None, dict[str, object] | None]]
    supervisor_tools: Sequence[dict[str, object]]
    task_configs: Mapping[str, Mapping[str, object]]


class _SeededCompletions:
    """Bind the declared run seed to every upstream OpenAI request."""

    def __init__(self, delegate: object, seed: int) -> None:
        self._delegate = delegate
        self._seed = seed

    def create(self, *args: object, **kwargs: object) -> object:
        supplied = kwargs.get("seed")
        if supplied is not None and supplied != self._seed:
            raise ValueError("upstream request seed differs from the formal run seed")
        kwargs["seed"] = self._seed
        return self._delegate.create(*args, **kwargs)  # type: ignore[attr-defined]


class _SeededChat:
    def __init__(self, delegate: object, seed: int) -> None:
        self.completions = _SeededCompletions(delegate.completions, seed)  # type: ignore[attr-defined]


class _SeededOpenAIClient:
    def __init__(self, delegate: object, seed: int) -> None:
        self._delegate = delegate
        self.chat = _SeededChat(delegate.chat, seed)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class EpisodeInfrastructureError(RuntimeError):
    """A retryable generation-service or repository-runtime failure."""


class _DisabledExecutor:
    """Prove that the direct agent never delegates an edit to a second model call."""

    def execute(self, *_args: object, **_kwargs: object) -> str:
        raise RuntimeError("direct repository agent cannot invoke MExec")


_DIRECT_SYSTEM_PROMPT = (
    "Fix the bug in the repository using the provided source tools. The evaluator submits the "
    "workspace diff at the end. Inspect the real source before editing. Use str_replace_editor "
    "to make exact, minimal source changes yourself; no second model or learned skill will edit "
    "for you. Never edit tests, fixtures, or evaluator files. Do not submit prose."
)
_DIRECT_READONLY_SYSTEM_PROMPT = (
    "Fix the public issue in the repository. Use only list_files, search_code, and view_file to "
    "inspect source. The workspace is read-only during inspection. When ready, reply with exactly "
    "one unified git diff and no prose. You may not use tests, learned skills, adapters, a second "
    "model, or an environment-supplied progress ledger."
)


def _validated_direct_tool_path(root: Path, path: object, *, mutating: bool) -> Path:
    if type(path) is not str or not path.strip() or "\x00" in path:
        raise ValueError("direct repository tool path is invalid")
    root = root.resolve()
    supplied = Path(path)
    candidate = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("direct repository tool path escapes the workspace") from exc
    if mutating:
        lowered = tuple(part.lower() for part in relative.parts)
        if ".git" in lowered or any(
            part in {"test", "tests", "testing", "fixtures"} or part.startswith("test_")
            for part in lowered
        ):
            raise ValueError("direct repository agent cannot modify tests or fixtures")
    return candidate


def _readonly_patch_candidate(content: str) -> str | None:
    return parse_unified_diff(content).value


class _JsonlJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, outcome: EpisodeOutcome) -> None:
        encoded = json.dumps(asdict(outcome), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iid-population", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--skillflow-source", type=Path, required=True)
    parser.add_argument("--repo-cache", type=Path, required=True)
    parser.add_argument("--endpoint-base", action="append", default=[])
    parser.add_argument("--served-model-name")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    parser.add_argument("--episode-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--prepare-repositories", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _require_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"SWE-bench {field_name} is missing or invalid")
    return value


def _load_tasks(arguments: argparse.Namespace) -> tuple[SWERepositoryTask, ...]:
    protocol = load_direct_reference_protocol(arguments.config)
    manifest = load_population_manifest(arguments.population_manifest)
    cases = load_skillflow_iid_cases(
        arguments.iid_population,
        protocol=protocol,
        include=frozenset({DirectBenchmark.SWE_BENCH}),
        manifests={DirectBenchmark.SWE_BENCH: manifest},
    )
    tasks = tuple(
        SWERepositoryTask(
            task_id=case.public_task.task_id,
            instance_id=_require_text(case.private_metadata.get("instance_id"), "instance_id"),
            repo=_require_text(case.private_metadata.get("repo"), "repo"),
            base_commit=_require_text(case.private_metadata.get("base_commit"), "base_commit"),
            problem_statement=_require_text(
                case.private_metadata.get("problem_statement"), "problem_statement"
            ),
        )
        for case in cases
    )
    if len(tasks) != EXPECTED_COUNT:
        raise ValueError(f"SWE-bench panel has {len(tasks)} records, expected {EXPECTED_COUNT}")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("SWE-bench task IDs must be unique")
    if len({task.instance_id for task in tasks}) != len(tasks):
        raise ValueError("SWE-bench instance IDs must be unique")
    return tasks


def _model_question(task: SWERepositoryTask) -> dict[str, object]:
    """Build the complete model-visible object without a target or gold-derived excerpt."""

    return {
        "question": task.problem_statement,
        "task_type": "code_generation",
        "context": [],
        "extra": {
            "instance_id": task.instance_id,
            "repo": task.repo,
            "base_commit": task.base_commit,
        },
    }


def _repository_path(cache: Path, repo: str) -> Path:
    if _REPOSITORY_NAME.fullmatch(repo) is None or ".." in repo:
        raise ValueError("SWE-bench repository identity is invalid")
    return cache / repo.replace("/", "__")


def _configure_process_local_git_safety(cache: Path, worktree_root: Path) -> None:
    """Trust repositories only inside this short-lived evaluator process.

    ShareStore presents files with a service-account owner, so Git otherwise
    rejects repositories that this process created itself.  The Git available
    on the evaluator host does not implement safe-directory subtree patterns,
    while worktree paths are allocated dynamically.  A process-local wildcard
    therefore covers the controlled cache/worktree roots without changing the
    user's global Git configuration or any repository configuration.
    """

    del cache, worktree_root
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = "safe.directory"
    os.environ["GIT_CONFIG_VALUE_0"] = "*"


def _run_git(command: Sequence[str], *, cwd: Path | None = None, timeout: float = 1800) -> None:
    result = subprocess.run(  # noqa: S603 - argv only; executable is fixed by each caller
        tuple(command), cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = detail[-1][:300] if detail else f"exit {result.returncode}"
        raise RuntimeError(f"git operation failed: {suffix}")


def _validate_repository_origin(path: Path, repo: str) -> None:
    result = subprocess.run(
        ("git", "remote", "get-url", "origin"),
        cwd=path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    normalized = result.stdout.strip().removesuffix(".git")
    allowed = {
        f"https://github.com/{repo}",
        f"git@github.com:{repo}",
        f"ssh://git@github.com/{repo}",
    }
    if result.returncode != 0 or normalized not in allowed:
        raise RuntimeError("repository cache origin differs from the frozen public repository")


def _prepare_repositories(tasks: Sequence[SWERepositoryTask], cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    revisions: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        revisions[task.repo].add(task.base_commit)
    for ordinal, (repo, commits) in enumerate(sorted(revisions.items()), start=1):
        path = _repository_path(cache, repo)
        if not path.exists():
            LOGGER.info("preparing public repository %d/%d", ordinal, len(revisions))
            _run_git(
                (
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--quiet",
                    f"https://github.com/{repo}.git",
                    str(path),
                )
            )
        elif not (path / ".git").is_dir():
            raise RuntimeError("repository cache entry is not a non-bare Git clone")
        _validate_repository_origin(path, repo)
        _run_git(("git", "fetch", "--quiet", "--prune", "origin"), cwd=path)
        for commit in sorted(commits):
            result = subprocess.run(  # noqa: S603 - git argv, no shell
                ("git", "cat-file", "-e", f"{commit}^{{commit}}"),
                cwd=path,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError("repository cache is missing a frozen base revision")


def _validate_repositories(tasks: Sequence[SWERepositoryTask], cache: Path) -> None:
    for task in tasks:
        path = _repository_path(cache, task.repo)
        if not (path / ".git").is_dir():
            raise RuntimeError("repository cache is incomplete; run with --prepare-repositories")
        _validate_repository_origin(path, task.repo)
        result = subprocess.run(  # noqa: S603 - git argv, no shell
            ("git", "cat-file", "-e", f"{task.base_commit}^{{commit}}"),
            cwd=path,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("repository cache is missing a frozen base revision")


def _import_upstream(source: Path, *, run_seed: int) -> UpstreamRuntime:
    required = (
        source / "training" / "environment.py",
        source / "training" / "batch_inference.py",
        source / "training" / "task_prompts.py",
        source / "src" / "executor" / "m_exec.py",
        source / "src" / "skills" / "workspace.py",
    )
    if any(not path.is_file() for path in required):
        raise ValueError("SkillFlow source does not contain the required repository-agent modules")
    sys.path.insert(0, str(source.resolve()))
    importlib.invalidate_caches()
    environment = importlib.import_module("training.environment")
    batch = importlib.import_module("training.batch_inference")
    prompts = importlib.import_module("training.task_prompts")
    executor = importlib.import_module("src.executor.m_exec")
    workspace = importlib.import_module("src.skills.workspace")
    batch_runtime = cast(Any, batch)
    executor_base = cast(Any, executor).MExec
    original_get_client = batch_runtime._get_client

    def seeded_get_client(api_base: str, api_key: str = "") -> object:
        return _SeededOpenAIClient(original_get_client(api_base, api_key), run_seed)

    batch_runtime._get_client = seeded_get_client

    class SeededMExec(executor_base):  # type: ignore[misc]
        @property
        def client(self) -> object:
            delegate = super().client
            cached = getattr(self._thread_local, "seeded_client", None)
            if cached is None or getattr(cached, "_delegate", None) is not delegate:
                cached = _SeededOpenAIClient(delegate, run_seed)
                self._thread_local.seeded_client = cached
            return cached

    runtime = UpstreamRuntime(
        environment_type=cast(type[Any], environment.GenericTaskEnvironment),
        executor_type=cast(type[Any], SeededMExec),
        workspace_type=cast(type[Any], workspace.SkillWorkspace),
        supervisor_call=cast(Callable[..., Any], batch_runtime.supervisor_call),
        supervisor_tools=cast(Sequence[dict[str, object]], batch_runtime.SUPERVISOR_TOOLS),
        task_configs=cast(Mapping[str, Mapping[str, object]], prompts.TASK_CONFIGS),
    )
    code_config = runtime.task_configs.get("code_generation", {})
    if code_config.get("max_episode_steps") != EXPECTED_STEPS:
        raise RuntimeError("upstream code-generation horizon differs from the frozen 28 steps")
    return runtime


def _isolated_environment_type(base: type[Any]) -> type[Any]:
    class IsolatedRepositoryEnvironment(base):  # type: ignore[misc, valid-type]
        def _setup_swe_repo(self) -> str:
            repo = _require_text(self._extra.get("repo"), "repo")
            base_commit = _require_text(self._extra.get("base_commit"), "base_commit")
            repo_path = _repository_path(self._skillev_repo_cache, repo)
            worktree_root = self._skillev_worktree_root
            worktree_root.mkdir(parents=True, exist_ok=True)
            worktree = Path(tempfile.mkdtemp(prefix="episode-", dir=worktree_root))
            worktree.rmdir()
            with self._skillev_repo_lock:
                result = subprocess.run(  # noqa: S603 - git argv, no shell
                    ("git", "worktree", "add", str(worktree), base_commit, "--detach", "--quiet"),
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=90,
                    check=False,
                )
            if result.returncode != 0:
                shutil.rmtree(worktree, ignore_errors=True)
                raise EpisodeInfrastructureError("isolated repository worktree creation failed")
            self._worktree_dir = str(worktree)
            self._worktree_repo = str(repo_path)
            return str(worktree)

        def cleanup(self) -> None:
            with self._skillev_repo_lock:
                super().cleanup()

        def _handle_str_replace_editor(self, args: dict[str, object]) -> str:
            command = str(args.get("command", ""))
            mutating = command in {"create", "str_replace", "insert", "undo_edit"}
            try:
                path = _validated_direct_tool_path(
                    Path(self._repo_path), args.get("path"), mutating=mutating
                )
            except ValueError as exc:
                return f"[str_replace_editor] [ERROR] {exc}"
            normalized = dict(args)
            normalized["path"] = str(path)
            return cast(str, super()._handle_str_replace_editor(normalized))

        def _handle_view_file(self, args: dict[str, object]) -> str:
            try:
                path = _validated_direct_tool_path(
                    Path(self._repo_path), args.get("path"), mutating=False
                )
            except ValueError as exc:
                return f"[view_file] [ERROR] {exc}"
            normalized = dict(args)
            normalized["path"] = str(path)
            return cast(str, super()._handle_view_file(normalized))

        def _swe_memory_summary(self, *args: object, **kwargs: object) -> str:
            if getattr(self, "_skillev_direct_raw_tools", False):
                return ""
            return cast(str, super()._swe_memory_summary(*args, **kwargs))

        def _swe_issue_member_mentions(self, *args: object, **kwargs: object) -> list[object]:
            if getattr(self, "_skillev_direct_raw_tools", False):
                return []
            return cast(list[object], super()._swe_issue_member_mentions(*args, **kwargs))

    return IsolatedRepositoryEnvironment


def _endpoint_models(endpoint: str) -> set[str]:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("generation endpoint must be an HTTP(S) URL")
    url = endpoint.rstrip("/") + "/models"
    request = urllib.request.Request(  # noqa: S310 - HTTP(S) was validated above
        url, headers={"Authorization": "Bearer EMPTY"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - operator URL
        value = json.load(response)
    rows = value.get("data") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("generation endpoint returned an invalid model catalog")
    return {
        row["id"]
        for row in rows
        if isinstance(row, dict) and type(row.get("id")) is str and row["id"].strip()
    }


def _turn_transcript(traj: object) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for turn in getattr(traj, "turns", ()):
        rows.append(
            {
                "action_type": str(getattr(turn, "action_type", "")),
                "instruction": str(getattr(turn, "instruction", "")),
                "observation": str(getattr(turn, "observation", "")),
                "parse_error": bool(getattr(turn, "parse_error", False)),
                "supervisor_output": str(getattr(turn, "supervisor_output", "")),
            }
        )
    return tuple(rows)


def _contains_terminal_executor_error(transcript: Sequence[Mapping[str, object]]) -> bool:
    return any("[EXECUTION_ERROR]" in str(row.get("observation", "")) for row in transcript)


def _run_episode(
    task: SWERepositoryTask,
    *,
    attempt: int,
    endpoint_index: int,
    endpoint: str,
    model: str,
    runtime: UpstreamRuntime,
    environment_type: type[Any],
    repo_cache: Path,
    worktree_root: Path,
    repo_lock: threading.Lock,
    request_retries: int,
    episode_timeout_seconds: float,
    direct_agent: bool,
    raw_tools_agent: bool,
    read_only_agent: bool,
) -> EpisodeOutcome:
    started = time.monotonic()
    supervisor_calls = 0
    env: Any | None = None
    traj: Any | None = None
    patch: str | None = None
    try:
        executor = (
            _DisabledExecutor()
            if direct_agent
            else runtime.executor_type(
                api_base=endpoint,
                model_name=model,
                default_temperature=EXECUTOR_TEMPERATURE,
                default_max_tokens=EXECUTOR_MAX_TOKENS,
                max_retries=request_retries,
            )
        )
        workspace = runtime.workspace_type(skills_dir=None)
        if getattr(workspace, "size", -1) != 0:
            raise RuntimeError("backbone-only evaluation requires an empty skill workspace")
        env = environment_type(
            m_exec=executor,
            max_episode_steps=EXPECTED_STEPS,
            skill_workspace=workspace,
            reward_mode="outcome_only",
            skill_mode="policy_action",
        )
        env._skillev_repo_cache = repo_cache
        env._skillev_worktree_root = worktree_root
        env._skillev_repo_lock = repo_lock
        env._skillev_direct_raw_tools = raw_tools_agent
        messages, traj = env.reset(_model_question(task))
        if direct_agent:
            messages[0]["content"] = (
                _DIRECT_READONLY_SYSTEM_PROMPT if read_only_agent else _DIRECT_SYSTEM_PROMPT
            )
            allowed_tools = (
                {"list_files", "search_code", "view_file"}
                if read_only_agent
                else {"list_files", "search_code", "view_file", "str_replace_editor"}
            )
            env._tools = [
                tool
                for tool in runtime.supervisor_tools
                if cast(dict[str, object], tool.get("function", {})).get("name") in allowed_tools
            ]
            if len(env._tools) != len(allowed_tools):
                raise RuntimeError("upstream direct repository tool surface is incomplete")
        if not getattr(env, "_worktree_dir", None):
            raise EpisodeInfrastructureError("episode did not receive an isolated repository")
        tools = env._tools
        done = False
        timed_out = False
        for _ in range(EXPECTED_STEPS):
            if time.monotonic() - started > episode_timeout_seconds:
                timed_out = True
                break
            error: Exception | None = None
            content, tool_name, tool_args = "", None, None
            for _request_attempt in range(request_retries):
                supervisor_calls += 1
                try:
                    content, tool_name, tool_args = runtime.supervisor_call(
                        messages=messages,
                        api_base=endpoint,
                        model=model,
                        tools=tools,
                        max_tokens=SUPERVISOR_MAX_TOKENS,
                        temperature=SUPERVISOR_TEMPERATURE,
                        enable_thinking=False,
                    )
                    error = None
                    break
                except Exception as exc:  # upstream client exposes several transport types
                    error = exc
            if error is not None:
                raise EpisodeInfrastructureError(
                    "supervisor request retries were exhausted"
                ) from error
            if read_only_agent and tool_name is None:
                candidate = _readonly_patch_candidate(content)
                if candidate is not None:
                    patch = candidate
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "No unambiguous unified diff was submitted. Inspect more source with "
                            "a read-only tool, or return exactly one unified git diff."
                        ),
                    }
                )
                continue
            _, done, _ = env.step(content, tool_name, tool_args, traj)
            if raw_tools_agent:
                env._recent_tool_calls.clear()
                env._repeat_count.clear()
                env._consecutive_repeats = 0
            if done:
                break
        if not read_only_agent:
            patch_value = env._generate_workspace_diff()
            patch = patch_value if type(patch_value) is str and patch_value.strip() else None
        transcript = _turn_transcript(traj)
        if patch is not None:
            state, infrastructure_error = "candidate", None
        elif timed_out or _contains_terminal_executor_error(transcript):
            raise EpisodeInfrastructureError(
                "episode timed out" if timed_out else "executor request failed without a candidate"
            )
        else:
            state, infrastructure_error = "candidate_failure", None
        return EpisodeOutcome(
            task_id=task.task_id,
            instance_id=task.instance_id,
            attempt=attempt,
            state=state,
            model_patch=patch,
            infrastructure_error=infrastructure_error,
            endpoint_index=endpoint_index,
            elapsed_seconds=time.monotonic() - started,
            steps=len(transcript),
            supervisor_calls=supervisor_calls,
            transcript=transcript,
        )
    except Exception as exc:
        return EpisodeOutcome(
            task_id=task.task_id,
            instance_id=task.instance_id,
            attempt=attempt,
            state="infrastructure_failure",
            model_patch=None,
            infrastructure_error=(
                str(exc)
                if isinstance(exc, EpisodeInfrastructureError)
                else "UnexpectedEpisodeInfrastructureError"
            ),
            endpoint_index=endpoint_index,
            elapsed_seconds=time.monotonic() - started,
            steps=len(getattr(traj, "turns", ())),
            supervisor_calls=supervisor_calls,
            transcript=_turn_transcript(traj) if traj is not None else (),
        )
    finally:
        if env is not None:
            env.cleanup()


def _load_journal(path: Path, tasks: Sequence[SWERepositoryTask]) -> dict[str, EpisodeOutcome]:
    allowed = {task.task_id: task for task in tasks}
    latest: dict[str, EpisodeOutcome] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            raw["transcript"] = tuple(raw.get("transcript", ()))
            outcome = EpisodeOutcome(**raw)
            task = allowed.get(outcome.task_id)
            if task is None or task.instance_id != outcome.instance_id:
                raise ValueError("episode journal contains a foreign task")
            prior = latest.get(outcome.task_id)
            if prior is not None and outcome.attempt != prior.attempt + 1:
                raise ValueError("episode journal attempts are not contiguous")
            latest[outcome.task_id] = outcome
    return latest


def _pending_tasks(
    tasks: Sequence[SWERepositoryTask],
    latest: Mapping[str, EpisodeOutcome],
    max_attempts: int,
) -> tuple[SWERepositoryTask, ...]:
    pending: list[SWERepositoryTask] = []
    for task in tasks:
        outcome = latest.get(task.task_id)
        if outcome is None or (
            outcome.state == "infrastructure_failure" and outcome.attempt < max_attempts
        ):
            pending.append(task)
    return tuple(pending)


def _write_outputs(
    output_dir: Path,
    tasks: Sequence[SWERepositoryTask],
    latest: Mapping[str, EpisodeOutcome],
    *,
    elapsed_seconds: float,
    run_seed: int,
    direct_agent: bool,
    raw_tools_agent: bool,
    read_only_agent: bool,
) -> dict[str, object]:
    predictions = output_dir / "predictions.jsonl"
    rows: list[dict[str, object]] = []
    for task in tasks:
        outcome = latest.get(task.task_id)
        rows.append(
            {
                "task_id": task.task_id,
                "instance_id": task.instance_id,
                "model_patch": outcome.model_patch if outcome is not None else None,
                "generation_infrastructure_error": (
                    outcome.infrastructure_error
                    if outcome is not None and outcome.state == "infrastructure_failure"
                    else None
                ),
                "generation_state": outcome.state if outcome is not None else "missing",
            }
        )
    temporary = predictions.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(predictions)
    final = tuple(latest.get(task.task_id) for task in tasks)
    summary: dict[str, object] = {
        "planned_count": len(tasks),
        "final_record_count": sum(outcome is not None for outcome in final),
        "candidate_response_count": sum(
            outcome is not None and outcome.state != "infrastructure_failure" for outcome in final
        ),
        "submission_count": sum(
            outcome is not None and outcome.state == "candidate" for outcome in final
        ),
        "candidate_failure_count": sum(
            outcome is not None and outcome.state == "candidate_failure" for outcome in final
        ),
        "generation_infrastructure_failures": sum(
            outcome is not None and outcome.state == "infrastructure_failure" for outcome in final
        ),
        "supervisor_calls": sum(outcome.supervisor_calls for outcome in final if outcome),
        "elapsed_seconds": elapsed_seconds,
        "protocol": {
            "agent": (
                "Qwen direct repository tools"
                if direct_agent
                else "SkillFlow delegated code_generation repository tools"
            ),
            "skills": "empty",
            "adapter": "none",
            "max_episode_steps": EXPECTED_STEPS,
            "supervisor_temperature": SUPERVISOR_TEMPERATURE,
            "supervisor_max_tokens": SUPERVISOR_MAX_TOKENS,
            "supervisor_thinking": False,
            "executor": "disabled" if direct_agent else "same frozen base route",
            "executor_temperature": None if direct_agent else EXECUTOR_TEMPERATURE,
            "executor_max_tokens": None if direct_agent else EXECUTOR_MAX_TOKENS,
            "executor_thinking": None if direct_agent else False,
            "run_seed": run_seed,
            "observation_contract": (
                "raw-tool-results-no-swe-memory" if raw_tools_agent else "upstream-orchestrated"
            ),
            "candidate_contract": (
                "one-final-unified-diff-after-read-only-inspection"
                if read_only_agent
                else "workspace-git-diff"
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.concurrency <= 0 or arguments.request_retries <= 0:
        raise ValueError("concurrency and request retries must be positive")
    if arguments.max_infrastructure_attempts <= 0 or arguments.episode_timeout_seconds <= 0:
        raise ValueError("attempt and timeout limits must be positive")
    tasks = _load_tasks(arguments)
    worktree_root = arguments.private_output_dir.resolve() / "worktrees"
    _configure_process_local_git_safety(arguments.repo_cache, worktree_root)
    if arguments.prepare_repositories:
        _prepare_repositories(tasks, arguments.repo_cache.resolve())
    _validate_repositories(tasks, arguments.repo_cache.resolve())
    if arguments.prepare_only:
        return {"prepared_repository_count": len({task.repo for task in tasks})}
    protocol = load_direct_reference_protocol(arguments.config)
    spec = protocol.benchmark(DirectBenchmark.SWE_BENCH)
    if len(spec.seed_aggregation.seeds) != 1:
        raise ValueError("SWE repository-agent runner requires exactly one declared run seed")
    run_seed = spec.seed_aggregation.seeds[0]
    direct_agent = spec.prompt_profile in DIRECT_REPOSITORY_AGENT_PROMPTS
    raw_tools_agent = spec.prompt_profile in {DIRECT_RAW_TOOLS_PROMPT, DIRECT_READONLY_PROMPT}
    read_only_agent = spec.prompt_profile == DIRECT_READONLY_PROMPT
    if not direct_agent and spec.prompt_profile != "skillflow-code-generation-repository-agent@1":
        raise ValueError("SWE prompt profile is not a supported repository-agent contract")
    model = arguments.served_model_name or protocol.model.served_model_name
    if model != protocol.model.served_model_name:
        raise ValueError("served model differs from the executable protocol")
    endpoints = tuple(dict.fromkeys(str(value).rstrip("/") for value in arguments.endpoint_base))
    if not endpoints:
        raise ValueError("at least one --endpoint-base is required")
    for endpoint in endpoints:
        if model not in _endpoint_models(endpoint):
            raise RuntimeError("generation endpoint does not expose the frozen base-model route")
    runtime = _import_upstream(arguments.skillflow_source.resolve(), run_seed=run_seed)
    environment_type = _isolated_environment_type(runtime.environment_type)
    output_dir = arguments.private_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "episodes.jsonl"
    if journal_path.exists() and not arguments.resume:
        raise FileExistsError(
            "episode journal exists; pass --resume to preserve completed candidates"
        )
    journal = _JsonlJournal(journal_path)
    latest = _load_journal(journal_path, tasks)
    started = time.monotonic()
    repo_lock = threading.Lock()
    worktree_root = output_dir / "worktrees"
    while pending := _pending_tasks(tasks, latest, arguments.max_infrastructure_attempts):
        with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.concurrency) as executor:
            futures: list[concurrent.futures.Future[EpisodeOutcome]] = []
            for task in pending:
                prior = latest.get(task.task_id)
                attempt = 1 if prior is None else prior.attempt + 1
                endpoint_index = (tasks.index(task) + attempt - 1) % len(endpoints)
                futures.append(
                    executor.submit(
                        _run_episode,
                        task,
                        attempt=attempt,
                        endpoint_index=endpoint_index,
                        endpoint=endpoints[endpoint_index],
                        model=model,
                        runtime=runtime,
                        environment_type=environment_type,
                        repo_cache=arguments.repo_cache.resolve(),
                        worktree_root=worktree_root,
                        repo_lock=repo_lock,
                        request_retries=arguments.request_retries,
                        episode_timeout_seconds=arguments.episode_timeout_seconds,
                        direct_agent=direct_agent,
                        raw_tools_agent=raw_tools_agent,
                        read_only_agent=read_only_agent,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                outcome = future.result()
                journal.append(outcome)
                latest[outcome.task_id] = outcome
                completed = sum(
                    value.state != "infrastructure_failure"
                    or value.attempt >= arguments.max_infrastructure_attempts
                    for value in latest.values()
                )
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = completed / elapsed
                eta = (len(tasks) - completed) / rate if rate > 0 else float("inf")
                LOGGER.info(
                    "terminal=%d/%d rate=%.3f tasks/s eta=%.0fs",
                    completed,
                    len(tasks),
                    rate,
                    eta,
                )
    return _write_outputs(
        output_dir,
        tasks,
        latest,
        elapsed_seconds=time.monotonic() - started,
        run_seed=run_seed,
        direct_agent=direct_agent,
        raw_tools_agent=raw_tools_agent,
        read_only_agent=read_only_agent,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(json.dumps(run(_arguments()), sort_keys=True))


if __name__ == "__main__":
    main()
