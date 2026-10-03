from __future__ import annotations

import ast
import os
from typing import Any

from skillev.evosteer_application import SessionRequest, TaskBinding, TaskSession
from skillev.experiments import curve_benchmarks
from skillev.orchestration.model_executor import FrozenTextExecutor
from skillev.orchestration.retrieval_client import RETRIEVAL_IDENTITY_ENV, RetrievalClient
from skillev.orchestration.text_tools import PythonSandbox, SearchTool, TextTool, TextToolsConfig
from skillev.orchestration.tool_text_executor import FrozenToolTextExecutor
from skillev.rollout.evosteer_risk import ExecutionRiskPolicy

CAPABILITY_ID = "curve-text-tools@1"
CODE_FAMILIES = frozenset({"mbpp_plus"})
CODE_OUTPUT_ID = "curve-tested-program@1"
_TEST_LINE = "Your code should pass this test:\n"
_SEARCH_RETRIES = 3
_BACKENDS: dict[tuple[str, TextToolsConfig], TextTool] = {}


def _backend(name: str, settings: TextToolsConfig) -> TextTool:
    key = (name, settings)
    if key not in _BACKENDS:
        if name == "python":
            _BACKENDS[key] = PythonSandbox(
                cpu_seconds=settings.python_cpu_seconds,
                wall_seconds=settings.python_wall_seconds,
                memory_mb=settings.python_memory_mb,
                result_chars=settings.result_chars,
            )
        elif name == "search":
            client = RetrievalClient(
                top_k=settings.search_top_k,
                timeout_seconds=settings.search_timeout_seconds,
                retries=_SEARCH_RETRIES,
                expected_identity=os.environ.get(RETRIEVAL_IDENTITY_ENV) or None,
            )
            client.wait_ready()
            _BACKENDS[key] = SearchTool(
                client,
                passage_chars=settings.search_passage_chars,
                result_chars=settings.result_chars,
            )
        else:
            raise ValueError(f"unknown text tool {name!r}")
    return _BACKENDS[key]


def prompt_entry(prompt: str) -> str | None:
    _, found, test = prompt.rpartition(_TEST_LINE)
    if not found:
        return None
    entry = curve_benchmarks._entry_point({"test_list": [test.strip()]})
    tree = curve_benchmarks._parse(test.strip()) if entry is None else None
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
    return entry


def tested_program(prompt: str, output: str, programs: tuple[str, ...]) -> str | None:
    entry = prompt_entry(prompt)
    if entry is None:
        return None
    tree = curve_benchmarks._parse(curve_benchmarks._extract_code(output, entry))
    if tree is not None and curve_benchmarks._defines(tree, entry):
        return None
    for program in reversed(programs):
        tree = curve_benchmarks._parse(program)
        if tree is not None and curve_benchmarks._defines(tree, entry):
            return curve_benchmarks._sanitize(program, entry)
    return None


def executor_options(family: str) -> dict[str, Any]:
    if family in CODE_FAMILIES:
        return {"code_output": tested_program, "code_output_id": CODE_OUTPUT_ID}
    return {}


def _wrap(
    binding: TaskBinding,
    model: Any,
    tools: dict[str, TextTool],
    settings: TextToolsConfig,
    options: dict[str, Any] | None = None,
) -> TaskBinding:
    options = {**(options or {}), **executor_options(binding.task.family)}
    executor_id = FrozenToolTextExecutor(model, tools, settings, **options).frozen_identity
    names = tuple(sorted(tools))
    inner_factory = binding.session_factory
    task = binding.task

    def session(request: SessionRequest) -> TaskSession:
        inner = inner_factory(request)
        if type(inner.executor) is not FrozenTextExecutor:
            if inner.close is not None:
                inner.close()
            raise ValueError(f"text tools wrap only one-call text tasks, not {task.family!r}")
        return TaskSession(
            FrozenToolTextExecutor(model, tools, settings, **options),
            inner.evaluator,
            close=inner.close,
            reset_receipt=inner.reset_receipt,
            risk_assessor=ExecutionRiskPolicy(
                executor_id,
                task.environment_config_id,
                scope="sandboxed_tools",
                capability_id=CAPABILITY_ID,
                tools=names,
            ),
        )

    return TaskBinding(task, executor_id, session, replay_safe=binding.replay_safe)


def task_factory(
    *,
    policy: Any,
    config: Any,
    executor_temperature: float = 0.3,
    executor_stop_regex: tuple[str, ...] = (),
) -> tuple[TaskBinding, ...]:
    options = {"temperature": executor_temperature, "stop_regex": executor_stop_regex}
    bindings = curve_benchmarks.task_factory(
        policy=policy,
        config=config,
        executor_temperature=executor_temperature,
        executor_stop_regex=executor_stop_regex,
    )
    settings = getattr(config, "text_tools", None)
    if settings is None:
        return bindings
    if not isinstance(settings, TextToolsConfig):
        raise TypeError("config.text_tools must be TextToolsConfig or None")
    model = curve_benchmarks._executor_model(policy)
    wrapped: list[TaskBinding] = []
    for binding in bindings:
        names = settings.tools_for(binding.task.family)
        if not names:
            wrapped.append(binding)
            continue
        tools = {name: _backend(name, settings) for name in names}
        wrapped.append(_wrap(binding, model, tools, settings, options))
    return tuple(wrapped)


__all__ = [
    "CAPABILITY_ID",
    "CODE_FAMILIES",
    "CODE_OUTPUT_ID",
    "executor_options",
    "prompt_entry",
    "task_factory",
    "tested_program",
]
