"""Runnable offline smoke and explicit local-model EvoSteer training entrypoints.

Task factories are trusted Python integration code: ``factory(policy=...,
config=EvoSteerConfig)`` must return a finite sequence of TaskBinding objects.
JSONL completion tasks keep expected_answers exclusively in evaluator closures.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from skillev.contracts.canonical import stable_hash
from skillev.contracts.evosteer import EvoTask
from skillev.evosteer_application import (
    EvoSteerApplication,
    EvoSteerConfig,
    ResetReceipt,
    SessionRequest,
    TaskBinding,
    TaskSession,
)
from skillev.orchestration.graph import RoleSpec
from skillev.orchestration.model_executor import FrozenTextExecutor
from skillev.policy.evosteer import CausalLMOrchestrator
from skillev.rollout.evosteer_risk import ExecutionRiskPolicy
from skillev.runtime import BudgetLedger, BudgetVector
from skillev.training.evosteer import EvoOptimizerConfig
from skillev.training.task_schedule import CurriculumStage, SourceBalancedTaskSchedule

CONFIG_FORMAT = "evosteer-local-training@2"
RUN_FORMAT = "evosteer-cli-run@2"
CONTEXT_FORMAT = "evosteer-cli-checkpoint@2"


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _object(value: Any, *, allowed: set[str], required: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValueError(f"{label} has missing or unsupported fields")
    return value


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0, inclusive: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < minimum
        or (not inclusive and value == minimum)
    ):
        raise ValueError(f"{name} must be finite and {'>=' if inclusive else '>'} {minimum}")
    return float(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError("JSON must contain only finite numbers")

    return json.loads(text, parse_constant=reject_constant)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _budget(value: Any, name: str) -> BudgetVector:
    values = _object(
        value,
        allowed={item.name for item in fields(BudgetVector)},
        required=set(),
        label=name,
    )
    return BudgetVector(**values)


def _application_config(value: Any) -> EvoSteerConfig:
    values = dict(
        _object(
            value,
            allowed={item.name for item in fields(EvoSteerConfig)},
            required={"task_families", "roles"},
            label="application config",
        )
    )
    families = values["task_families"]
    if not isinstance(families, list) or not families:
        raise ValueError("task_families must be a nonempty array")
    values["task_families"] = tuple(_text(item, "task family") for item in families)
    if not isinstance(values["roles"], list) or not values["roles"]:
        raise ValueError("roles must be a nonempty array")
    roles = []
    for row in values["roles"]:
        row = _object(
            row,
            allowed={"role_id", "instruction", "model_maximum"},
            required={"role_id", "instruction"},
            label="role",
        )
        role_values = dict(row)
        if "model_maximum" in row:
            role_values["model_maximum"] = _budget(row["model_maximum"], "role model_maximum")
        roles.append(RoleSpec(**role_values))
    values["roles"] = tuple(roles)
    if values.get("episode_budget") is not None:
        values["episode_budget"] = _budget(values["episode_budget"], "episode_budget")
    if "optimizer" in values:
        optimizer = _object(
            values["optimizer"],
            allowed={item.name for item in fields(EvoOptimizerConfig)},
            required=set(),
            label="optimizer",
        )
        # Rates, clips and counts must be positive; the decay, the supervised loss
        # weights and AdamW's first-moment decays may be zero. An upper bound is
        # EvoOptimizerConfig's own business and it re-checks every field below.
        zero_allowed = {
            "weight_decay",
            "value_loss_weight",
            "outcome_loss_weight",
            "actor_beta1",
            "head_beta1",
        }
        for key, number in optimizer.items():
            if key == "gradient_mode":
                if number not in ("streaming", "dense"):
                    raise ValueError("gradient_mode must be streaming or dense")
                continue
            _number(number, key, inclusive=key in zero_allowed)
        values["optimizer"] = EvoOptimizerConfig(**optimizer)
    for flag in ("strict_value_context", "allow_repeated_pair_tasks"):
        if flag in values and type(values[flag]) is not bool:
            raise ValueError(f"{flag} must be a boolean")
    for key in {item.name for item in fields(EvoSteerConfig)} - {
        "task_families",
        "roles",
        "optimizer",
        "episode_budget",
        "strict_value_context",
        "allow_repeated_pair_tasks",
        "structure_source_mode",
    }:
        if key in values:
            if key in {"max_nodes", "max_actions"} and values[key] is None:
                continue
            # author_evidence_pool = 0 is the paper's current-batch-only author.
            _integer(
                values[key], key, minimum=0 if key in {"seed", "author_evidence_pool"} else 1
            )
    if values.get("author_start_families", 1) > len(values["task_families"]):
        raise ValueError("author_start_families cannot exceed the number of task families")
    # EvoSteerConfig re-checks every field; a nonzero author_evidence_pool must be
    # >= 2 and needs at least one batch before the first author window.
    return EvoSteerConfig(**values)


def _model_config(value: Any, base: Path) -> dict[str, Any]:
    allowed = {
        "model_path",
        "revision",
        "reference_id",
        "device",
        "dtype",
        "lora_rank",
        "lora_alpha",
        "target_modules",
        "context_window",
        "max_action_tokens",
        "context_mode",
    }
    values = dict(
        _object(
            value,
            allowed=allowed,
            required={"model_path", "reference_id"},
            label="local model config",
        )
    )
    model_path = Path(_text(values["model_path"], "model_path")).expanduser()
    model_path = (
        (base / model_path).resolve() if not model_path.is_absolute() else model_path.resolve()
    )
    if not model_path.is_dir():
        raise ValueError("model_path must point to an existing local model directory")
    values["model_path"] = str(model_path)
    for key in ("reference_id", "revision", "device", "dtype"):
        if key in values:
            _text(values[key], key)
    if values.get("context_mode", "full") not in {"full", "debug_head_tail"}:
        raise ValueError("context_mode must be full or debug_head_tail")
    for key in ("lora_rank", "lora_alpha", "context_window", "max_action_tokens"):
        if key in values:
            _integer(values[key], key)
    if values.get("max_action_tokens", 512) >= values.get("context_window", 8192):
        raise ValueError("max_action_tokens must be smaller than context_window")
    if "target_modules" in values:
        modules = values["target_modules"]
        if not isinstance(modules, list) or not modules:
            raise ValueError("target_modules must be a nonempty array")
        values["target_modules"] = tuple(_text(item, "target module") for item in modules)
    return values


def _configuration(path: Path) -> tuple[dict[str, Any], EvoSteerConfig]:
    value = _object(
        _json(path.read_text(encoding="utf-8")),
        allowed={
            "format",
            "model",
            "application",
            "steps",
            "batch_size",
            "executor_temperature",
            "author",
            "method_mode",
            "sampling",
        },
        required={"format", "model", "application", "steps", "batch_size"},
        label="training config",
    )
    if value["format"] != CONFIG_FORMAT:
        raise ValueError("unsupported training config format")
    values = dict(value)
    mode = values.setdefault("method_mode", "full")
    if mode not in {"full", "debug"}:
        raise ValueError("method_mode must be full or debug")
    if mode == "full" and "author" not in values:
        raise ValueError(
            "full method requires an independent frozen skill author; use debug for ablations"
        )
    _integer(values["steps"], "steps")
    _integer(values["batch_size"], "batch_size")
    values["executor_temperature"] = _number(
        values.get("executor_temperature", 0.3), "executor_temperature"
    )
    application = _application_config(values["application"])
    values["model"] = _model_config(values["model"], path.resolve().parent)
    if mode == "full" and (
        values["model"].get("context_mode", "full") != "full"
        or application.max_nodes is not None
        or application.max_actions is not None
        or application.current_rollouts != 2
        or application.reference_rollouts != 2
        or application.value_refresh_interval != 1
        or application.structure_source_mode != "reference_with_paired"
    ):
        raise ValueError(
            "full method requires full context, resource-only graph limits, 2+2 rollouts, "
            "per-batch value publication and paired structure prefixes"
        )
    if "author" in values:
        from skillev.evolution.evosteer_author import SkillAuthorConfig

        row = _object(
            values["author"],
            allowed={"model", "config", "budget"},
            required={"model", "config", "budget"},
            label="author config",
        )
        model = _model_config(row["model"], path.resolve().parent)
        author_config = SkillAuthorConfig(
            **_object(
                row["config"],
                allowed={item.name for item in fields(SkillAuthorConfig)},
                required=set(),
                label="author generation config",
            )
        )
        budget = _budget(row["budget"], "author budget")
        if not author_config.maximum.fits_within(budget):
            raise ValueError("author budget must fit at least one configured author call")
        if author_config.max_new_tokens >= model.get("context_window", 8192):
            raise ValueError("author output budget must fit its local model context")
        values["author"] = {
            "model": model,
            "config": asdict(author_config),
            "budget": asdict(budget),
        }
    _sampling_schedule(values, application.seed)
    return values, application


def _sampling_schedule(config: dict[str, Any], seed: int) -> SourceBalancedTaskSchedule | None:
    if "sampling" not in config:
        return None
    row = _object(
        config["sampling"],
        allowed={"task_sources", "curriculum"},
        required={"task_sources", "curriculum"},
        label="task sampling",
    )
    if not isinstance(row["curriculum"], list):
        raise ValueError("curriculum must be an explicit array of source stages")
    return SourceBalancedTaskSchedule(
        row["task_sources"],
        curriculum=tuple(CurriculumStage.from_value(stage) for stage in row["curriculum"]),
        batch_size=config["batch_size"],
        seed=seed,
    )


def _read_completion_tasks(path: Path) -> tuple[tuple[tuple[EvoTask, Any], ...], str]:
    """Separate public tasks from private answer closures before constructing a model."""
    payload = path.read_bytes()
    tasks = []
    seen = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = _object(
            _json(line),
            allowed={
                "task_id",
                "family",
                "prompt",
                "reset_id",
                "environment_config_id",
                "expected_answers",
                "prior_rate",
                "prior_count",
            },
            required={"task_id", "family", "prompt", "expected_answers"},
            label=f"task JSONL line {line_number}",
        )
        answers = row["expected_answers"]
        if not isinstance(answers, list) or not answers or any(type(a) is not str for a in answers):
            raise ValueError(
                f"expected_answers on line {line_number} must be a nonempty text array"
            )
        task = EvoTask(**{key: value for key, value in row.items() if key != "expected_answers"})
        if task.task_id in seen:
            raise ValueError("task IDs must be unique across the JSONL dataset")
        seen.add(task.task_id)

        def evaluate(output: str, accepted: frozenset[str] = frozenset(answers)) -> float:
            return float(output in accepted)

        tasks.append((task, evaluate))
    if not tasks:
        raise ValueError("task JSONL dataset is empty")
    return tuple(tasks), "sha256:" + hashlib.sha256(payload).hexdigest()


def _completion_bindings(
    rows: tuple[tuple[EvoTask, Any], ...],
    policy: CausalLMOrchestrator,
    temperature: float,
) -> tuple[TaskBinding, ...]:
    bindings = []
    executor_id = FrozenTextExecutor(policy, temperature=temperature).frozen_identity
    for task, evaluate in rows:

        def session(request: SessionRequest, evaluator: Any = evaluate) -> TaskSession:
            return TaskSession(
                FrozenTextExecutor(policy, temperature=temperature),
                evaluator,
                reset_receipt=ResetReceipt(
                    request.task.identity,
                    request.task.reset_id,
                    request.task.environment_config_id,
                    request.task.identity,
                    str(uuid.uuid4()),
                    request.seed,
                ),
                risk_assessor=ExecutionRiskPolicy(
                    executor_id,
                    request.task.environment_config_id,
                    scope="text_only",
                    capability_id="static-completion-no-tools@1",
                ),
            )

        bindings.append(TaskBinding(task, executor_id, session, replay_safe=True))
    return tuple(bindings)


def _factory_bindings(
    target: str,
    policy: CausalLMOrchestrator,
    config: EvoSteerConfig,
) -> tuple[tuple[TaskBinding, ...], str]:
    module_name, separator, name = target.partition(":")
    if not separator or not module_name or not name or ":" in name:
        raise ValueError("task factory must be module:callable")
    module = importlib.import_module(module_name)
    factory = getattr(module, name)
    if not callable(factory):
        raise ValueError("task factory must be callable")
    bindings = tuple(factory(policy=policy, config=config))
    if not bindings or any(not isinstance(item, TaskBinding) for item in bindings):
        raise ValueError("task factory must return a nonempty sequence of TaskBinding objects")
    module_path = getattr(module, "__file__", None)
    source_hash = (
        hashlib.sha256(Path(module_path).read_bytes()).hexdigest() if module_path else None
    )
    return bindings, str(
        stable_hash(
            {
                "factory": target,
                "module_sha256": source_hash,
                "tasks": [item.task.identity for item in bindings],
                "executors": [item.executor_id for item in bindings],
            }
        )
    )


def _validate_bindings(
    bindings: tuple[TaskBinding, ...], config: EvoSteerConfig, batch_size: int
) -> None:
    if not bindings or batch_size > len(bindings):
        raise ValueError("batch_size must not exceed the number of distinct tasks")
    if len({item.task.task_id for item in bindings}) != len(bindings):
        raise ValueError("task bindings must have unique task IDs")
    if any(item.task.family not in config.task_families for item in bindings):
        raise ValueError("a task family is outside the configured task-family universe")


def _build_training(
    args: argparse.Namespace,
) -> tuple[EvoSteerApplication, tuple[TaskBinding, ...], dict[str, Any], str]:
    import torch

    config, application_config = _configuration(args.config)
    rows, source_id = _read_completion_tasks(args.tasks) if args.tasks else ((), "")
    # Validate the dataset before allocating any model weights.
    if rows:
        if config["batch_size"] > len(rows):
            raise ValueError("batch_size exceeds the number of tasks")
        if any(task.family not in application_config.task_families for task, _ in rows):
            raise ValueError("a task family is outside the configured task-family universe")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(application_config.seed)
        policy = CausalLMOrchestrator.from_pretrained(**config["model"])
        author = None
        if "author" in config:
            from skillev.evolution.evosteer_author import FrozenSkillAuthor, SkillAuthorConfig

            row = config["author"]
            author_url = os.environ.get("EVOSTEER_AUTHOR_URL")
            api_base = os.environ.get("EVOSTEER_AUTHOR_API_BASE")
            if api_base:
                # A hosted, independent author model (the paper's separate author).
                from skillev.policy.api_text import ResponsesApiFrozenText

                author_model = ResponsesApiFrozenText(
                    api_base,
                    os.environ["EVOSTEER_AUTHOR_API_MODEL"],
                    Path(os.environ["EVOSTEER_AUTHOR_API_KEY_FILE"]).read_text().strip(),
                    reference_id=f"{os.environ['EVOSTEER_AUTHOR_API_MODEL']}-author@{row['model']['reference_id']}",
                    reasoning_effort=os.environ.get("EVOSTEER_AUTHOR_API_EFFORT", "low"),
                )
            elif author_url:
                # The author is a frozen base-model completion; serving it from
                # SGLang avoids a second resident copy of large checkpoints.
                from skillev.policy.sglang_text import SGLangFrozenText

                author_model = SGLangFrozenText(
                    author_url,
                    policy.tokenizer,
                    reference_id=row["model"]["reference_id"],
                    context_window=row["model"].get("context_window", 8192),
                )
            else:
                author_model = CausalLMOrchestrator.from_pretrained(**row["model"])
                author_model.model.requires_grad_(False)
            author = FrozenSkillAuthor(
                author_model,
                config=SkillAuthorConfig(**row["config"]),
                budget=BudgetLedger(
                    run_id="evosteer-cli-author",
                    attempt_id="initial",
                    cap=BudgetVector(**row["budget"]),
                ),
            )
    if args.task_factory:
        bindings, source_id = _factory_bindings(args.task_factory, policy, application_config)
    else:
        bindings = _completion_bindings(rows, policy, config["executor_temperature"])
    _validate_bindings(bindings, application_config, config["batch_size"])
    if config["method_mode"] == "full" and any(not item.replay_safe for item in bindings):
        raise ValueError(
            "full method requires isolated resettable tasks for candidate/control trials"
        )
    if "sampling" in config and set(config["sampling"]["task_sources"]) != {
        item.task.task_id for item in bindings
    }:
        raise ValueError("sampling task/source manifest must match the loaded task IDs exactly")
    app = EvoSteerApplication(policy, application_config, author=author)
    sampler_device = os.environ.get("EVOSTEER_SAMPLER_DEVICE")
    if sampler_device:
        # A frozen-base replica on another GPU samples episodes while the actor
        # trains; its adapter is republished after every update.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(application_config.seed)
            sampler = CausalLMOrchestrator.from_pretrained(
                **{**config["model"], "device": sampler_device}
            )
        app.set_rollout_policy(sampler)
    return app, bindings, config, source_id


async def _executor_uses_actor(bindings: tuple[TaskBinding, ...], actor: Any) -> bool:
    """Whether any task's executor generates with the actor model itself."""
    for binding in bindings:
        session = binding.session_factory(SessionRequest(binding.task, 0))
        try:
            if getattr(session.executor, "policy", None) is actor:
                return True
        finally:
            if session.close is not None:
                closed = session.close()
                if inspect.isawaitable(closed):
                    await closed
    return False


async def _heldout(
    app: EvoSteerApplication,
    bindings: tuple[TaskBinding, ...],
    output: Path,
    sources: tuple[str, ...],
    samples: int,
) -> None:
    """Evaluate the current checkpoint on fixed held-out tasks; append one row per source."""
    started = time.monotonic()
    trajectories = await app.evaluate(bindings, sources=sources, samples=samples)
    name = f"step-{app.batch_index:06d}"
    with (output / "evals" / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
        for trajectory in trajectories:
            stream.write(json.dumps(trajectory.to_value(), ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "eval.jsonl").open("a", encoding="utf-8") as stream:
        for source in sources:
            rows = [t for t in trajectories if t.source == source]
            by_family: dict[str, list[float]] = {}
            for t in rows:
                by_family.setdefault(t.task.family, []).append(t.reward)
            row = {
                "step": app.batch_index,
                "actor_version": app.policy.version,
                "source": source,
                "samples_per_task": samples,
                "n": len(rows),
                "reward_mean": sum(t.reward for t in rows) / len(rows),
                "reward_by_family": {k: sum(v) / len(v) for k, v in sorted(by_family.items())},
                "rewards": {t.sample_id: t.reward for t in rows},
                "seconds": round(time.monotonic() - started, 1),
            }
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print(json.dumps({"event": "heldout", **{k: v for k, v in row.items() if k != "rewards"}}), flush=True)


async def _run(
    app: EvoSteerApplication,
    bindings: tuple[TaskBinding, ...],
    *,
    output: Path,
    steps: int,
    batch_size: int,
    source_id: str,
    synthetic: bool,
    resume: Path | None,
    run_config: dict[str, Any],
    eval_bindings: tuple[TaskBinding, ...] = (),
    eval_every: int = 0,
    eval_samples: int = 1,
) -> dict[str, Any]:
    _validate_bindings(bindings, app.config, batch_size)
    # Executor HTTP calls and the concurrent update each hold a worker thread.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=96))
    schedule = _sampling_schedule(run_config, app.config.seed)
    by_task = {binding.task.task_id: binding for binding in bindings}
    if schedule is not None and set(run_config["sampling"]["task_sources"]) != set(by_task):
        raise ValueError("sampling task/source manifest must match the loaded task IDs exactly")
    context = {
        "format": CONTEXT_FORMAT,
        "task_source_id": source_id,
        "batch_size": batch_size,
        "task_order": [item.task.identity for item in bindings],
        "synthetic": synthetic,
        "executor_ids": [item.executor_id for item in bindings],
        "sampling_configuration_id": schedule.configuration_id if schedule is not None else None,
    }
    if resume is not None:
        stored = _json((resume / "cli-context.json").read_text(encoding="utf-8"))
        sampling_state = stored.pop("sampling_state")
        expected = {
            **context,
            "application_state_sha256": (resume / "state.sha256").read_text().strip(),
        }
        if stored != expected:
            raise ValueError(
                "resume checkpoint has a different CLI task source, order or batch size"
            )
        app.load_checkpoint(resume)
        app.sync_rollout_policy()
        if schedule is not None:
            schedule.load_state_dict(sampling_state)
            if schedule.committed_steps != app.batch_index:
                raise ValueError("task schedule and optimizer checkpoint steps differ")
        elif sampling_state is not None:
            raise ValueError("checkpoint has a task schedule but this run does not")
        if steps <= app.batch_index:
            raise ValueError("steps is a total target and must exceed the resumed batch_index")
    start_index = app.batch_index
    _write_json(
        output / "run.json",
        {
            "format": RUN_FORMAT,
            "synthetic": synthetic,
            "method_mode": run_config.get("method_mode", "debug" if synthetic else "full"),
            "purpose": "synthetic integration check; not a paper benchmark"
            if synthetic
            else "training",
            "config": run_config,
            "application": asdict(app.config),
            "task_source_id": source_id,
            "task_order": [item.task.identity for item in bindings],
            "resume": str(resume.resolve()) if resume is not None else None,
            "start_batch_index": start_index,
            "target_steps": steps,
            "policy_configuration_id": app.policy.configuration_id,
            "runtime_environment": {
                name: os.environ.get(name)
                for name in (
                    "EVOSTEER_EXECUTOR_URL",
                    "EVOSTEER_AUTHOR_URL",
                    "EVOSTEER_AUTHOR_API_BASE",
                    "EVOSTEER_AUTHOR_API_MODEL",
                    "EVOSTEER_AUTHOR_API_EFFORT",
                    "EVOSTEER_EXECUTOR_CONTEXT",
                    "EVOSTEER_ROLLOUT_CONCURRENCY",
                    "EVOSTEER_PIPELINE",
                    "EVOSTEER_SAMPLER_DEVICE",
                    "EVOSTEER_GIL_SWITCH_INTERVAL",
                    "EVOSTEER_CURVE_DATA",
                    "EVOSTEER_CURVE_SPLIT",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR",
                )
            },
        },
    )
    for name in ("batches", "trajectories", "checkpoints"):
        (output / name).mkdir()
    if eval_bindings:
        (output / "evals").mkdir()
        if app.batch_index == 0:
            # Step 0: the actor equals rho (LoRA B = 0); both sources are measured.
            await _heldout(app, eval_bindings, output, ("current", "natural_reference"), eval_samples)
    last_checkpoint = None

    async def commit(result: Any, plan: Any) -> None:
        nonlocal last_checkpoint
        metrics = {
            "batch_id": result.batch_id,
            "synthetic": synthetic,
            **result.metrics,
            "admission_decisions": list(result.admission_decisions),
            "task_sampling": plan.to_value() if plan is not None else {"mode": "ordered_records"},
        }
        _write_json(output / "batches" / f"{result.batch_id}.json", metrics)
        with (output / "trajectories" / f"{result.batch_id}.jsonl").open(
            "x", encoding="utf-8"
        ) as stream:
            for trajectory in result.trajectories:
                stream.write(
                    json.dumps(trajectory.to_value(), ensure_ascii=False, allow_nan=False) + "\n"
                )
        checkpoint = app.save_checkpoint(output / "checkpoints" / result.batch_id)
        _write_json(
            checkpoint / "cli-context.json",
            {
                **context,
                "application_state_sha256": (checkpoint / "state.sha256").read_text().strip(),
                "sampling_state": schedule.state_dict() if schedule is not None else None,
            },
        )
        last_checkpoint = str(checkpoint.resolve())
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "batch_index": app.batch_index,
                    "checkpoint": last_checkpoint,
                    "synthetic": synthetic,
                }
            ),
            flush=True,
        )

    async def evaluate_step() -> None:
        if eval_bindings and (app.batch_index % eval_every == 0 or app.batch_index == steps):
            await _heldout(app, eval_bindings, output, ("current",), eval_samples)

    pipelined = os.environ.get("EVOSTEER_PIPELINE") == "1"
    if pipelined and (
        schedule is None
        or app.rollout_policy is app.policy
        or await _executor_uses_actor(bindings, app.policy)
    ):
        # Overlapping collection with the update is safe only when neither the
        # orchestrator nor the executor nodes run on the actor being trained.
        raise ValueError(
            "EVOSTEER_PIPELINE=1 requires a task schedule, a rollout replica "
            "(EVOSTEER_SAMPLER_DEVICE) and executors that do not run on the actor"
        )
    if pipelined and app.batch_index < steps:
        # Algorithm 1's asynchronous prefetch: batch k+1 is collected by the
        # rollout replica while batch k updates the actor, so current rollouts
        # lag the trained adapter by one update (Appendix A, Remark A.3).
        plan = schedule.plan_next()
        collected = await app.collect_batch(
            tuple(by_task[task_id] for task_id in plan.task_ids), batch_number=app.batch_index + 1
        )
        while True:
            prepared = await asyncio.to_thread(app.prepare_batch, collected)
            schedule.commit(plan)
            upcoming = next_plan = None
            if app.batch_index + 2 <= steps:
                next_plan = schedule.plan_next()
                upcoming = asyncio.create_task(
                    app.collect_batch(
                        tuple(by_task[task_id] for task_id in next_plan.task_ids),
                        batch_number=app.batch_index + 2,
                    )
                )
                # Let the collection freeze its value snapshot and admission
                # base before the update thread starts replacing them.
                await asyncio.sleep(0)
            try:
                result = await asyncio.to_thread(app.finish_batch, prepared)
            except BaseException:
                if upcoming is not None:
                    upcoming.cancel()
                    await asyncio.gather(upcoming, return_exceptions=True)
                raise
            # Persist batch k before waiting on the prefetch, so a failed
            # collection never discards a completed update.
            await commit(result, plan)
            if upcoming is None:
                app.sync_rollout_policy()
                await evaluate_step()
                break
            collected = await upcoming
            # Publish only between collections: one batch never mixes adapters.
            app.sync_rollout_policy()
            await evaluate_step()
            plan = next_plan
    while app.batch_index < steps:
        plan = schedule.plan_next() if schedule is not None else None
        if plan is not None:
            batch = tuple(by_task[task_id] for task_id in plan.task_ids)
        else:
            offset = app.batch_index * batch_size
            batch = tuple(bindings[(offset + i) % len(bindings)] for i in range(batch_size))
        result = await app.train_batch(batch)
        if schedule is not None and plan is not None:
            schedule.commit(plan)
        app.sync_rollout_policy()
        await commit(result, plan)
        await evaluate_step()
    summary = {
        "status": "completed",
        "synthetic": synthetic,
        "batch_index": app.batch_index,
        "actor_version": app.policy.version,
        "completed_batches": app.batch_index - start_index,
        "checkpoint": last_checkpoint,
        "output": str(output.resolve()),
    }
    _write_json(output / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="EvoSteer local training and offline synthetic smoke"
    )
    commands = root.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke", help="real tiny HF/PEFT training with a synthetic executor; no downloads"
    )
    smoke.add_argument(
        "--output", required=True, type=Path, help="new output directory; never overwritten"
    )
    smoke.add_argument("--steps", type=_positive, default=2, help="total target optimizer batches")
    smoke.add_argument("--resume", type=Path, help="explicit smoke checkpoint directory")
    train = commands.add_parser("train", help="train from explicitly configured local weights")
    train.add_argument("--config", required=True, type=Path)
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tasks", type=Path, help="JSONL public task fields plus private expected_answers"
    )
    source.add_argument(
        "--task-factory",
        help="trusted module:callable accepting policy= and config=; returns TaskBindings",
    )
    train.add_argument(
        "--output", required=True, type=Path, help="new output directory; never overwritten"
    )
    train.add_argument(
        "--resume", type=Path, help="explicit checkpoint directory; total steps comes from config"
    )
    train.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="evaluate the actor on the task factory's validation split every N steps (0 = off)",
    )
    train.add_argument(
        "--eval-samples", type=_positive, default=1, help="held-out rollouts per task per source"
    )
    return root


def _gil_switch_interval() -> None:
    """Optionally shorten the interpreter's thread switch interval (seconds).

    In the pipelined loop the actor update (worker thread) and the rollout
    loop (event-loop thread) both launch many short CUDA calls. With the default
    5 ms interval each GIL hand-off can stall the waiting thread for up to 5 ms,
    which measured a ~4x slower update under rollout load; 0.2-0.5 ms recovers
    most of it. Scheduling only: no computed value depends on it.
    """
    raw = os.environ.get("EVOSTEER_GIL_SWITCH_INTERVAL")
    if raw is None:
        return
    value = float(raw)
    if not 0.0 < value <= 0.1:
        raise ValueError("EVOSTEER_GIL_SWITCH_INTERVAL must be in (0, 0.1] seconds")
    sys.setswitchinterval(value)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _gil_switch_interval()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"Output directory cannot be created: {exc}", file=sys.stderr)
        return 2
    previous_threads = None
    try:
        if args.resume is not None:
            args.resume = args.resume.expanduser().resolve()
            if not args.resume.is_dir():
                raise ValueError("resume must name an existing checkpoint directory")
            for name in ("state.json", "state.sha256", "trainable.pt", "cli-context.json"):
                if not (args.resume / name).is_file():
                    raise ValueError("resume directory is missing required checkpoint files")
        if args.command == "smoke":
            import torch

            from skillev.evosteer_demo import build_smoke_application

            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            app, bindings = build_smoke_application()
            config = {
                "fixture": "synthetic-repair-tiny-hf-peft@1",
                "steps": args.steps,
                "batch_size": 1,
            }
            source_id = "synthetic-repair-fixture@1"
        else:
            app, bindings, config, source_id = _build_training(args)
        eval_bindings: tuple[TaskBinding, ...] = ()
        if args.command == "train" and args.eval_every > 0:
            if not args.task_factory:
                raise ValueError("held-out evaluation requires a task factory with a validation split")
            previous_split = os.environ.get("EVOSTEER_CURVE_SPLIT")
            os.environ["EVOSTEER_CURVE_SPLIT"] = "validation"
            try:
                eval_bindings, _ = _factory_bindings(args.task_factory, app.policy, app.config)
            finally:
                if previous_split is None:
                    os.environ.pop("EVOSTEER_CURVE_SPLIT")
                else:
                    os.environ["EVOSTEER_CURVE_SPLIT"] = previous_split
            if {b.task.task_id for b in eval_bindings} & {b.task.task_id for b in bindings}:
                raise ValueError("held-out tasks overlap the training tasks")
        summary = asyncio.run(
            _run(
                app,
                bindings,
                output=output,
                steps=config["steps"],
                batch_size=config["batch_size"],
                source_id=source_id,
                synthetic=args.command == "smoke",
                resume=args.resume,
                run_config=config,
                eval_bindings=eval_bindings,
                eval_every=getattr(args, "eval_every", 0),
                eval_samples=getattr(args, "eval_samples", 1),
            )
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        _write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "synthetic": args.command == "smoke",
                "output": str(output),
            },
        )
        print(f"EvoSteer failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    finally:
        if previous_threads is not None:
            import torch

            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    raise SystemExit(main())
