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
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from skillev.contracts.canonical import stable_hash
from skillev.evosteer_application import (
    EvoSteerApplication,
    EvoSteerConfig,
    SessionRequest,
    TaskBinding,
)
from skillev.orchestration.graph import RoleSpec
from skillev.orchestration.text_tools import TextToolsConfig
from skillev.policy.evosteer import CausalLMOrchestrator
from skillev.policy.sglang_text import check_stop_regex
from skillev.runtime.budget_ledger import BudgetLedger
from skillev.runtime.contracts import BudgetVector
from skillev.training.evosteer import EvoOptimizerConfig
from skillev.training.task_schedule import CurriculumStage, SourceBalancedTaskSchedule

CONFIG_FORMAT = "evosteer-local-training@2"
RUN_FORMAT = "evosteer-cli-run@2"
CONTEXT_FORMAT = "evosteer-cli-checkpoint@2"


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
            allowed={"role_id", "instruction", "model_maximum", "tools"},
            required={"role_id", "instruction"},
            label="role",
        )
        role_values = dict(row)
        if "model_maximum" in row:
            role_values["model_maximum"] = _budget(row["model_maximum"], "role model_maximum")
        if "tools" in row:
            tools = row["tools"]
            if not isinstance(tools, list) or not tools:
                raise ValueError("role tools must be a nonempty array of distinct names")
            names = [_text(tool, "role tool") for tool in tools]
            if len(set(names)) != len(names):
                raise ValueError("role tools must be a nonempty array of distinct names")
            role_values["tools"] = tuple(sorted(names))
        roles.append(RoleSpec(**role_values))
    values["roles"] = tuple(roles)
    if values.get("episode_budget") is not None:
        values["episode_budget"] = _budget(values["episode_budget"], "episode_budget")
    if values.get("family_role_maximum") is not None:
        raw = values["family_role_maximum"]
        if not isinstance(raw, dict) or not raw:
            raise ValueError("family_role_maximum must be a nonempty object of family -> budget")
        values["family_role_maximum"] = tuple(
            (_text(family, "family_role_maximum family"), _budget(budget, f"{family} model_maximum"))
            for family, budget in sorted(raw.items())
        )
    if values.get("text_tools") is not None:
        values["text_tools"] = TextToolsConfig.from_value(values["text_tools"])
    if "optimizer" in values:
        optimizer = _object(
            values["optimizer"],
            allowed={item.name for item in fields(EvoOptimizerConfig)},
            required=set(),
            label="optimizer",
        )
        zero_allowed = {
            "weight_decay",
            "value_loss_weight",
            "outcome_loss_weight",
            "actor_beta1",
            "head_beta1",
        }
        for key, number in optimizer.items():
            _number(number, key, inclusive=key in zero_allowed)
        values["optimizer"] = EvoOptimizerConfig(**optimizer)
    for key in ("non_output_roles", "stop_gate_families"):
        if key in values:
            names = values[key]
            if not isinstance(names, list) or len(set(map(str, names))) != len(names):
                raise ValueError(f"{key} must be an array of distinct names")
            values[key] = tuple(sorted(_text(name, key) for name in names))
    if "allow_repeated_pair_tasks" in values and type(values["allow_repeated_pair_tasks"]) is not bool:
        raise ValueError("allow_repeated_pair_tasks must be a boolean")
    for key in {item.name for item in fields(EvoSteerConfig)} - {
        "task_families",
        "roles",
        "optimizer",
        "episode_budget",
        "family_role_maximum",
        "allow_repeated_pair_tasks",
        "structure_source_mode",
        "text_tools",
        "non_output_roles",
        "stop_gate_families",
    }:
        if key in values:
            if key in {"max_nodes", "max_actions"} and values[key] is None:
                continue
            _integer(values[key], key, minimum=0 if key in {"seed", "author_evidence_pool"} else 1)
    if values.get("author_start_families", 1) > len(values["task_families"]):
        raise ValueError("author_start_families cannot exceed the number of task families")
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
            "executor_stop_regex",
            "author",
            "sampling",
        },
        required={"format", "model", "application", "steps", "batch_size"},
        label="training config",
    )
    if value["format"] != CONFIG_FORMAT:
        raise ValueError("unsupported training config format")
    values = dict(value)
    if "author" not in values:
        raise ValueError("training requires an independent frozen skill author")
    _integer(values["steps"], "steps")
    _integer(values["batch_size"], "batch_size")
    values["executor_temperature"] = _number(
        values.get("executor_temperature", 0.3), "executor_temperature"
    )
    if "executor_stop_regex" in values:
        patterns = values["executor_stop_regex"]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError("executor_stop_regex must be a nonempty array of patterns")
        check_stop_regex(tuple(patterns))
    application = _application_config(values["application"])
    values["model"] = _model_config(values["model"], path.resolve().parent)
    if (
        application.max_nodes is not None
        or application.max_actions is not None
        or application.current_rollouts != 2
        or application.reference_rollouts != 2
        or application.value_refresh_interval != 1
        or application.structure_source_mode != "reference_with_paired"
    ):
        raise ValueError(
            "training requires resource-only graph limits, 2+2 rollouts, "
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


def executor_options(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "executor_temperature": config["executor_temperature"],
        "executor_stop_regex": tuple(config.get("executor_stop_regex", ())),
    }


def _factory_bindings(
    target: str,
    policy: CausalLMOrchestrator,
    config: EvoSteerConfig,
    **executor: Any,
) -> tuple[tuple[TaskBinding, ...], str]:
    module_name, separator, name = target.partition(":")
    if not separator or not module_name or not name or ":" in name:
        raise ValueError("task factory must be module:callable")
    module = importlib.import_module(module_name)
    factory = getattr(module, name)
    if not callable(factory):
        raise ValueError("task factory must be callable")
    bindings = tuple(factory(policy=policy, config=config, **executor))
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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(application_config.seed)
        policy = CausalLMOrchestrator.from_pretrained(
            **config["model"]
        )
        author = None
        if "author" in config:
            from skillev.evolution.evosteer_author import FrozenSkillAuthor, SkillAuthorConfig

            row = config["author"]
            author_url = os.environ.get("EVOSTEER_AUTHOR_URL")
            api_base = os.environ.get("EVOSTEER_AUTHOR_API_BASE")
            if api_base:
                from skillev.policy.api_text import ResponsesApiFrozenText

                author_model = ResponsesApiFrozenText(
                    api_base,
                    os.environ["EVOSTEER_AUTHOR_API_MODEL"],
                    Path(os.environ["EVOSTEER_AUTHOR_API_KEY_FILE"]).read_text().strip(),
                    reference_id=f"{os.environ['EVOSTEER_AUTHOR_API_MODEL']}-author@{row['model']['reference_id']}",
                    reasoning_effort=os.environ.get("EVOSTEER_AUTHOR_API_EFFORT", "low"),
                )
            elif author_url:
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
    bindings, source_id = _factory_bindings(
        args.task_factory, policy, application_config, **executor_options(config)
    )
    _validate_bindings(bindings, application_config, config["batch_size"])
    if any(not item.replay_safe for item in bindings):
        raise ValueError(
            "training requires isolated resettable tasks for candidate/control trials"
        )
    if "sampling" in config and set(config["sampling"]["task_sources"]) != {
        item.task.task_id for item in bindings
    }:
        raise ValueError("sampling task/source manifest must match the loaded task IDs exactly")
    app = EvoSteerApplication(policy, application_config, author=author)
    sampler_device = os.environ.get("EVOSTEER_SAMPLER_DEVICE")
    if sampler_device:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(application_config.seed)
            sampler = CausalLMOrchestrator.from_pretrained(
                **{**config["model"], "device": sampler_device}
            )
        app.set_rollout_policy(sampler)
    return app, bindings, config, source_id


async def _executor_uses_actor(bindings: tuple[TaskBinding, ...], actor: Any) -> bool:
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


async def _run(
    app: EvoSteerApplication,
    bindings: tuple[TaskBinding, ...],
    *,
    output: Path,
    steps: int,
    batch_size: int,
    source_id: str,
    resume: Path | None,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    _validate_bindings(bindings, app.config, batch_size)
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
            "purpose": "training",
            "config": run_config,
            "application": app.config.record(),
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
                    "ALFWORLD_DATA",
                    "ALFWORLD_MAX_STEPS",
                    "ALFWORLD_STEPS_PER_NODE",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR",
                )
            }
            | {
                name: os.environ[name]
                for name in (
                    "EVOSTEER_RETRIEVAL_URL",
                    "EVOSTEER_RETRIEVAL_IDENTITY",
                    "EVOSTEER_PYTHON_TOOL_CONCURRENCY",
                    "EVOSTEER_PYTHON_TOOL_ISOLATION",
                )
                if name in os.environ
            },
        },
    )
    for name in ("batches", "trajectories", "checkpoints"):
        (output / name).mkdir()
    last_checkpoint = None

    async def commit(result: Any, plan: Any) -> None:
        nonlocal last_checkpoint
        metrics = {
            "batch_id": result.batch_id,
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
                }
            ),
            flush=True,
        )

    pipelined = os.environ.get("EVOSTEER_PIPELINE") == "1"
    if pipelined and (
        schedule is None
        or app.rollout_policy is app.policy
        or await _executor_uses_actor(bindings, app.policy)
    ):
        raise ValueError(
            "EVOSTEER_PIPELINE=1 requires a task schedule, a rollout replica "
            "(EVOSTEER_SAMPLER_DEVICE) and executors that do not run on the actor"
        )
    if pipelined and app.batch_index < steps:
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
                await asyncio.sleep(0)
            try:
                result = await asyncio.to_thread(app.finish_batch, prepared)
            except BaseException:
                if upcoming is not None:
                    upcoming.cancel()
                    await asyncio.gather(upcoming, return_exceptions=True)
                raise
            await commit(result, plan)
            if upcoming is None:
                app.sync_rollout_policy()
                break
            collected = await upcoming
            app.sync_rollout_policy()
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
    summary = {
        "status": "completed",
        "batch_index": app.batch_index,
        "actor_version": app.policy.version,
        "completed_batches": app.batch_index - start_index,
        "checkpoint": last_checkpoint,
        "output": str(output.resolve()),
    }
    _write_json(output / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="EvoSteer local training")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train from explicitly configured local weights")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument(
        "--task-factory",
        required=True,
        help="trusted module:callable accepting policy= and config=; returns TaskBindings",
    )
    train.add_argument(
        "--output", required=True, type=Path, help="new output directory; never overwritten"
    )
    train.add_argument(
        "--resume", type=Path, help="explicit checkpoint directory; total steps comes from config"
    )
    return root


def _gil_switch_interval() -> None:
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
    try:
        if args.resume is not None:
            args.resume = args.resume.expanduser().resolve()
            if not args.resume.is_dir():
                raise ValueError("resume must name an existing checkpoint directory")
            for name in ("state.json", "state.sha256", "trainable.pt", "cli-context.json"):
                if not (args.resume / name).is_file():
                    raise ValueError("resume directory is missing required checkpoint files")
        app, bindings, config, source_id = _build_training(args)
        summary = asyncio.run(
            _run(
                app,
                bindings,
                output=output,
                steps=config["steps"],
                batch_size=config["batch_size"],
                source_id=source_id,
                resume=args.resume,
                run_config=config,
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
                "output": str(output),
            },
        )
        print(f"EvoSteer failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
