#!/usr/bin/env python3
"""Run the private native-environment portion of the Qwen direct suite."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

from skillev_private.benchmarks.alfworld_official import OfficialALFWorldTask
from skillev_private.benchmarks.official_process import (
    ALFWorldGameDeployment,
    OfficialALFWorldProcessFactory,
    OfficialScienceWorldProcessFactory,
    OfficialWebShopProcessFactory,
    PinnedOfficialProcess,
    SQLiteWebShopDeployment,
)
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldTask
from skillev_private.benchmarks.webshop_official import OfficialWebShopGoal
from skillev_private.direct_reference import (
    DirectALFWorldEnvironment,
    DirectScienceWorldEnvironment,
    DirectWebShopEnvironment,
    InteractiveJournal,
    InteractiveJournalRecord,
    can_resume,
)
from skillev_private.direct_reference.journal import (
    require_fresh_output_directory,
    write_attempt_marker,
)

from skillev.evaluation.direct_baseline import (
    DirectBenchmark,
    InvalidCandidatePolicy,
    InvalidEnvironmentActionPolicy,
    NativeInteractiveAttempt,
    NativeInteractiveEnvironment,
    NativeInteractiveTask,
    NativePublicState,
    OpenAICompatibleDirectClient,
    QwenChatTokenCounter,
    run_native_interactive_task,
)
from skillev.evaluation.direct_baseline.client import DirectGenerationRequest
from skillev.evaluation.direct_baseline.config import (
    DirectDecodingProfile,
    DirectReferenceProtocol,
)
from skillev.evaluation.direct_baseline.interactive_tasks import (
    InteractiveReasoningMode,
    InteractiveStepRecord,
)
from skillev.evaluation.direct_baseline.protocol import (
    load_direct_reference_protocol,
    validate_runtime_registries,
)
from skillev.evaluation.interactive_prompt_assets import (
    InteractivePromptAsset,
    load_interactive_prompt_asset,
)

_FORMATS = {
    "skillev-qwen35-direct-interactive@2",
    "skillev-qwen35-direct-interactive@3",
}
_BENCHMARKS = frozenset(
    {DirectBenchmark.WEB_SHOP, DirectBenchmark.ALF_WORLD, DirectBenchmark.SCIENCE_WORLD}
)


@dataclass(frozen=True, slots=True)
class _Case:
    task: NativeInteractiveTask
    create_environment: Callable[[], NativeInteractiveEnvironment]


def _webshop_environment(
    factory: OfficialWebShopProcessFactory,
    goal: OfficialWebShopGoal,
    task_text: str,
) -> NativeInteractiveEnvironment:
    return DirectWebShopEnvironment(factory.create(goal), goal.session_id, task_text)


def _alfworld_environment(
    factory: OfficialALFWorldProcessFactory,
    task: OfficialALFWorldTask,
    task_text: str,
) -> NativeInteractiveEnvironment:
    return DirectALFWorldEnvironment(factory.create(task), task.seed, task_text)


def _scienceworld_environment(
    factory: OfficialScienceWorldProcessFactory,
    task: OfficialScienceWorldTask,
) -> NativeInteractiveEnvironment:
    return DirectScienceWorldEnvironment(
        factory.create(task),
        task.task_name,
        task.variation_index,
        task.seed,
        task.max_steps,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-base", required=True)
    parser.add_argument("--served-model-name", default="qwen35-direct-base")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-length", type=int, default=98_304)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prompt-asset-dir",
        type=Path,
        help="Directory containing train-only versioned native ReAct prompt assets.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(item.value for item in _BENCHMARKS),
        default=sorted(item.value for item in _BENCHMARKS),
    )
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _runtime(value: object) -> PinnedOfficialProcess:
    raw = _mapping(value, "runtime")
    return PinnedOfficialProcess(
        interpreter_path=Path(raw["interpreter_path"]),
        source_root=Path(raw["source_root"]),
        source_revision=str(raw["source_revision"]),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", 120)),
    )


def _prompt_asset_version(profile_id: str) -> str | None:
    if profile_id in {
        "alfworld-native-react-memory-v7@1",
        "webshop-native-react-memory-v7@1",
        "webshop-native-react-memory-v8@1",
        "webshop-native-stateact-memory-v9@1",
        "webshop-native-stateact-memory-v10@1",
        "webshop-native-stateact-memory-v11@1",
        "webshop-native-stateact-memory-v13@1",
    }:
        # Later public-policy profiles deliberately reuse the already source-proven v6 train
        # replay. Only policy/routing changes, so the prompt-asset evidence identity is stable.
        return "v6"
    if profile_id in {
        "webshop-native-stateact-memory-v14@1",
        "webshop-native-stateact-memory-v15@1",
    }:
        # AgentBoard's released WebShop scaffold and the original ReAct example both teach a
        # short successful purchase trajectory.  The source-proven v5 asset contains the same
        # efficient search -> product -> required options -> buy shape, unlike the deliberately
        # exploratory v6 replay used by the earlier policy ablations.
        return "v5"
    if profile_id in {
        "webshop-native-stateact-memory-v16@1",
        "webshop-native-stateact-memory-v17@1",
        "webshop-native-react-memory-v18@1",
    }:
        return "v7"
    for version in ("v6", "v5", "v4", "v3"):
        if profile_id.endswith(f"-{version}@1"):
            return version
    return None


def _uses_structured_memory(profile_id: str) -> bool:
    return profile_id.endswith(
        (
            "-v5@1",
            "-v6@1",
            "-v7@1",
            "-v8@1",
            "-v9@1",
            "-v10@1",
            "-v11@1",
            "-v13@1",
            "-v14@1",
            "-v15@1",
            "-v16@1",
            "-v17@1",
            "-v18@1",
        )
    )


def _demonstration_selection_profile(profile_id: str) -> str:
    if profile_id in {
        "webshop-native-stateact-memory-v17@1",
        "webshop-native-react-memory-v18@1",
    }:
        return "lexical-bm25-top1@1"
    if profile_id in {
        "webshop-native-stateact-memory-v13@1",
        "webshop-native-stateact-memory-v16@1",
    }:
        return "lexical-bm25@1"
    return "constraint-tags@1"


def _compact_demonstration_action_surfaces(profile_id: str) -> bool | None:
    return True if profile_id == "webshop-native-stateact-memory-v15@1" else None


def _uses_task_specific_demonstrations(profile_id: str, task_type: str | None) -> bool:
    if profile_id != "alfworld-native-react-memory-v7@1":
        return True
    return task_type == "pick_heat_then_place"


def _load_cases(
    path: Path,
    protocol: DirectReferenceProtocol,
    selected: frozenset[DirectBenchmark],
    prompt_asset_dir: Path | None = None,
) -> tuple[_Case, ...]:
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "manifest")
    if raw.get("format") not in _FORMATS:
        raise ValueError("interactive manifest format is incompatible")
    contracts = _mapping(raw.get("benchmarks"), "benchmark contracts")
    runtimes = {
        name: _runtime(value) for name, value in _mapping(raw["runtimes"], "runtimes").items()
    }
    deployments = _mapping(raw["deployments"], "deployments")
    factories: dict[str, object] = {}
    for name, value in deployments.items():
        item = _mapping(value, f"deployment {name}")
        runtime = runtimes[str(item["runtime"])]
        kind = item["kind"]
        if kind == "webshop-sqlite":
            factories[name] = OfficialWebShopProcessFactory(
                SQLiteWebShopDeployment(
                    runtime=runtime,
                    store_path=Path(item["store_path"]),
                    goals_path=Path(item["goals_path"]),
                    index_path=Path(item["index_path"]),
                    seed=int(item["seed"]),
                )
            )
        elif kind == "alfworld":
            games = {
                game_id: ALFWorldGameDeployment(
                    data_directory=Path(game["data_directory"]),
                    train_eval=str(game["train_eval"]),
                    instruction_text=str(game["instruction_text"]),
                )
                for game_id, game in _mapping(item["games"], "ALFWorld games").items()
            }
            factories[name] = OfficialALFWorldProcessFactory(
                runtime=runtime,
                config_path=Path(item["config_path"]),
                games=games,
                seed=int(item["seed"]),
                simulator_max_steps=(
                    int(item["simulator_max_steps"])
                    if item.get("simulator_max_steps") is not None
                    else None
                ),
            )
        elif kind == "scienceworld":
            factories[name] = OfficialScienceWorldProcessFactory(
                runtime=runtime,
                jar_path=Path(item["jar_path"]),
                simplification=str(item.get("simplification", "")),
                seed=int(item["seed"]),
            )
        else:
            raise ValueError(f"unsupported deployment kind {kind}")

    cases: list[_Case] = []
    seen: set[str] = set()
    rows = raw.get("cases")
    if not isinstance(rows, list):
        raise ValueError("interactive manifest cases must be an array")
    assets: dict[DirectBenchmark, InteractivePromptAsset] = {}
    for benchmark in selected:
        spec = protocol.benchmark(benchmark)
        version = _prompt_asset_version(spec.prompt_profile)
        if version is None:
            continue
        if prompt_asset_dir is None:
            raise ValueError("source-proven interactive profiles require --prompt-asset-dir")
        asset = load_interactive_prompt_asset(
            prompt_asset_dir / f"{benchmark.value}_native_react_{version}.yaml"
        )
        if asset.benchmark != benchmark.value or asset.source_split != "train":
            raise ValueError(f"{benchmark.value} prompt asset is not a train-only asset")
        contract = _mapping(contracts.get(benchmark.value), f"{benchmark.value} contract")
        if version in {"v4", "v5", "v6", "v7"} and (
            asset.asset_id != contract.get("prompt_asset_id")
            or asset.source_revision != contract.get("prompt_asset_source_revision")
        ):
            raise ValueError(f"{benchmark.value} prompt asset identity differs")
        assets[benchmark] = asset
    final_ids: dict[DirectBenchmark, set[str]] = {benchmark: set() for benchmark in assets}
    for value in rows:
        item = _mapping(value, "interactive case")
        benchmark = DirectBenchmark(str(item["benchmark"]))
        if benchmark not in assets:
            continue
        payload = _mapping(item["payload"], "case payload")
        final_ids[benchmark].add(str(item["task_id"]))
        source_key = "goal_id" if benchmark is DirectBenchmark.WEB_SHOP else "game_id"
        final_ids[benchmark].add(str(payload[source_key]))
    for benchmark, asset in assets.items():
        asset.validate_final_isolation(frozenset(final_ids[benchmark]))
    for value in rows:
        item = _mapping(value, "interactive case")
        task_id = str(item["task_id"])
        if task_id in seen:
            raise ValueError("interactive task IDs must be unique")
        seen.add(task_id)
        benchmark = DirectBenchmark(str(item["benchmark"]))
        if benchmark not in selected:
            continue
        task_text = str(item["task"])
        max_steps = int(item["max_steps"])
        payload = _mapping(item["payload"], "case payload")
        spec = protocol.benchmark(benchmark)
        contract = _mapping(contracts.get(benchmark.value), f"{benchmark.value} contract")
        expected_contract = {
            "expected_count": spec.sample_count,
            "prompt_profile": spec.prompt_profile,
            "decoding_profile": spec.decoding_profile,
            "parser_profile": spec.parser_profile,
            "environment_contract": spec.environment_contract,
            "invalid_candidate_policy": spec.invalid_candidate_policy,
            "invalid_environment_action_policy": spec.invalid_environment_action_policy,
            "history_window_steps": spec.history_window_steps,
            "horizon_policy": spec.horizon_policy,
            "max_steps_cap": spec.max_steps_cap,
            "required_max_steps": spec.required_max_steps,
            "include_reasoning_in_history": spec.include_reasoning_in_history,
        }
        if any(contract.get(key) != expected for key, expected in expected_contract.items()):
            raise ValueError(f"{benchmark.value} manifest contract differs from protocol")
        if spec.max_steps_cap is None or max_steps > spec.max_steps_cap:
            raise ValueError(f"{benchmark.value} case exceeds the protocol step cap")
        if spec.required_max_steps is not None and max_steps != spec.required_max_steps:
            raise ValueError(f"{benchmark.value} case differs from the required matched horizon")
        demonstrations: tuple[str, ...] = ()
        reasoning_mode = InteractiveReasoningMode.ACTION_ONLY
        if _prompt_asset_version(spec.prompt_profile) is not None:
            asset = assets[benchmark]
            reasoning_mode = InteractiveReasoningMode(asset.reasoning_mode)
            task_type = (
                str(payload["task_type"]) if benchmark is DirectBenchmark.ALF_WORLD else None
            )
            if _uses_task_specific_demonstrations(spec.prompt_profile, task_type):
                demonstrations = asset.render_demonstrations(
                    task=task_text,
                    task_type=task_type,
                    structured_memory=_uses_structured_memory(spec.prompt_profile),
                    include_thought=(
                        spec.prompt_profile
                        not in {
                            "webshop-native-stateact-memory-v9@1",
                            "webshop-native-stateact-memory-v10@1",
                            "webshop-native-stateact-memory-v11@1",
                            "webshop-native-stateact-memory-v13@1",
                            "webshop-native-stateact-memory-v14@1",
                            "webshop-native-stateact-memory-v15@1",
                            "webshop-native-stateact-memory-v16@1",
                            "webshop-native-stateact-memory-v17@1",
                        }
                    ),
                    selection_profile=_demonstration_selection_profile(spec.prompt_profile),
                    compact_action_surfaces=_compact_demonstration_action_surfaces(
                        spec.prompt_profile
                    ),
                )
        history_characters = contract.get("history_maximum_characters", 30_000)
        task = NativeInteractiveTask(
            task_id,
            benchmark,
            task_text,
            protocol.profile(spec.decoding_profile),
            max_steps,
            prompt_profile_id=spec.prompt_profile,
            parser_profile_id=spec.parser_profile,
            population_id=spec.population,
            run_seed=spec.seed_aggregation.seeds[0],
            invalid_candidate_policy=InvalidCandidatePolicy(str(spec.invalid_candidate_policy)),
            invalid_environment_action_policy=InvalidEnvironmentActionPolicy(
                str(spec.invalid_environment_action_policy)
            ),
            history_window_steps=(spec.history_window_steps),
            history_maximum_characters=(
                None if history_characters is None else int(history_characters)
            ),
            include_reasoning_in_history=spec.include_reasoning_in_history,
            reasoning_mode=reasoning_mode,
            demonstrations=demonstrations,
            task_type=(
                str(payload["task_type"]) if benchmark is DirectBenchmark.ALF_WORLD else None
            ),
        )
        factory = factories[str(item["deployment"])]
        if benchmark is DirectBenchmark.WEB_SHOP:
            if not isinstance(factory, OfficialWebShopProcessFactory):
                raise ValueError("WebShop case references an incompatible deployment")
            goal = OfficialWebShopGoal(
                task_id,
                str(payload["environment_id"]),
                str(payload["goal_id"]),
                str(payload["session_id"]),
                int(payload["goal_index"]),
            )
            cases.append(
                _Case(
                    task,
                    partial(_webshop_environment, factory, goal, task_text),
                )
            )
        elif benchmark is DirectBenchmark.ALF_WORLD:
            if not isinstance(factory, OfficialALFWorldProcessFactory):
                raise ValueError("ALFWorld case references an incompatible deployment")
            alfworld_task = OfficialALFWorldTask(
                task_id,
                str(payload["environment_id"]),
                str(payload["game_id"]),
                int(payload["seed"]),
                max_steps,
                dict(payload),
            )
            cases.append(
                _Case(
                    task,
                    partial(_alfworld_environment, factory, alfworld_task, task_text),
                )
            )
        else:
            if not isinstance(factory, OfficialScienceWorldProcessFactory):
                raise ValueError("ScienceWorld case references an incompatible deployment")
            scienceworld_task = OfficialScienceWorldTask(
                task_id,
                str(payload["environment_id"]),
                str(payload["task_name"]),
                int(payload["variation_index"]),
                int(payload["seed"]),
                max_steps,
                dict(payload),
            )
            cases.append(
                _Case(
                    task,
                    partial(_scienceworld_environment, factory, scienceworld_task),
                )
            )
    for benchmark in selected:
        count = sum(case.task.benchmark is benchmark for case in cases)
        if count != protocol.benchmark(benchmark).sample_count:
            raise ValueError(f"{benchmark.value} manifest has {count} cases, expected 128")
    return tuple(cases)


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    protocol = load_direct_reference_protocol(arguments.config)
    validate_runtime_registries(protocol)
    selected = frozenset(DirectBenchmark(value) for value in arguments.benchmarks)
    cases = _load_cases(
        arguments.private_manifest,
        protocol,
        selected,
        prompt_asset_dir=arguments.prompt_asset_dir,
    )
    contract_id = _execution_contract_id(
        cases,
        protocol.model.served_model_name,
        context_length=arguments.context_length,
    )
    start: dict[str, object] = {
        "format": "skillev-direct-attempt@2",
        "status": "running",
        "execution_contract_id": contract_id,
        "planned_task_ids": [case.task.task_id for case in cases],
        "prompt_profiles": sorted({case.task.prompt_profile_id for case in cases}),
        "decoding_profiles": sorted({case.task.profile.profile_id for case in cases}),
        "population_ids": sorted({case.task.population_id for case in cases}),
        "seeds": sorted({case.task.run_seed for case in cases}),
    }
    start_path = arguments.private_output_dir / "attempt-start.json"
    if arguments.resume:
        if not start_path.is_file() or json.loads(start_path.read_text(encoding="utf-8")) != start:
            raise ValueError("interactive resume marker differs from the requested contract")
    else:
        require_fresh_output_directory(arguments.private_output_dir)
        write_attempt_marker(start_path, start)
    journal = InteractiveJournal(
        arguments.private_output_dir / "interactive-journal.jsonl",
        execution_contract_id=contract_id,
    )
    prior = journal.load()
    case_ids = {case.task.task_id for case in cases}
    if not set(prior).issubset(case_ids):
        raise ValueError("interactive journal contains a task outside the frozen panel")
    # Do not probe the final panel before the run. Environment capability checks
    # belong to a separate non-final fixture; the actual panel remains untouched
    # until its recorded attempt.
    served_model_name = arguments.served_model_name
    if served_model_name != protocol.model.served_model_name:
        raise ValueError("served model name differs from the frozen protocol")
    client = OpenAICompatibleDirectClient(
        endpoint_base=arguments.endpoint_base,
        served_model_name=arguments.served_model_name,
        api_key=arguments.api_key,
        timeout_seconds=arguments.request_timeout_seconds,
        context_length=arguments.context_length,
        token_counter=QwenChatTokenCounter.from_pretrained(arguments.tokenizer_path),  # type: ignore[arg-type]
    )
    await _preflight_model_route(client, served_model_name)
    semaphore = asyncio.Semaphore(arguments.concurrency)

    async def run_case(case: _Case) -> NativeInteractiveAttempt:
        async with semaphore:
            try:
                environment = await asyncio.to_thread(case.create_environment)
            except (OSError, RuntimeError, ValueError):
                attempt = NativeInteractiveAttempt(
                    case.task.task_id,
                    case.task.benchmark,
                    None,
                    None,
                    0,
                    0,
                    0,
                    False,
                    False,
                    "EnvironmentCreationError",
                )
            else:
                attempt = await run_native_interactive_task(client, case.task, environment)
            previous = prior.get(case.task.task_id)
            journal.append(
                InteractiveJournalRecord(
                    task_id=case.task.task_id,
                    request_attempt=1 if previous is None else previous.request_attempt + 1,
                    execution_contract_id=contract_id,
                    outcome=_attempt_value(attempt),
                )
            )
            return attempt

    pending = tuple(
        case
        for case in cases
        if case.task.task_id not in prior or can_resume(prior[case.task.task_id])
    )
    await asyncio.gather(*(run_case(case) for case in pending))
    final = journal.load()
    if set(final) != case_ids or any(can_resume(record) for record in final.values()):
        raise RuntimeError("interactive panel still contains infrastructure retry slots")
    attempts = tuple(_attempt_from_value(final[case.task.task_id].outcome) for case in cases)
    with (arguments.private_output_dir / "per-task-results.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for attempt in attempts:
            stream.write(json.dumps(_attempt_value(attempt), sort_keys=True) + "\n")
    summary = _aggregate(attempts, protocol)
    (arguments.private_output_dir / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_attempt_marker(
        arguments.private_output_dir / "attempt-complete.json",
        {
            **start,
            "status": "complete",
            "final_record_count": len(attempts),
        },
    )
    return summary


def _execution_contract_id(
    cases: tuple[_Case, ...], served_model_name: str, *, context_length: int
) -> str:
    identities = sorted(
        {
            "::".join(
                (
                    case.task.benchmark.value,
                    case.task.population_id,
                    case.task.prompt_profile_id,
                    case.task.profile.profile_id,
                    case.task.parser_profile_id,
                    str(case.task.run_seed),
                    str(case.task.max_steps),
                    str(case.task.history_window_steps),
                    str(case.task.history_maximum_characters),
                    case.task.invalid_candidate_policy.value,
                    case.task.invalid_environment_action_policy.value,
                    case.task.reasoning_mode.value,
                )
            )
            for case in cases
        }
    )
    return "|".join(
        ("protocol13-interactive@2", served_model_name, str(context_length), *identities)
    )


async def _preflight_model_route(
    client: OpenAICompatibleDirectClient, served_model_name: str
) -> None:
    probe = await client.generate(
        DirectGenerationRequest(
            request_id="direct-interactive-runtime-capability-probe",
            messages=({"role": "user", "content": "Reply with OK."},),
            profile=DirectDecodingProfile(
                "interactive-runtime-capability-probe@1",
                False,
                0.0,
                1.0,
                1,
                0.0,
                0.0,
                1.0,
                8,
                sampling_mode="greedy",
            ),
        )
    )
    if probe.response_model != served_model_name:
        raise RuntimeError("interactive capability probe returned another model route")


async def _preflight_environments(
    cases: tuple[_Case, ...], selected: frozenset[DirectBenchmark]
) -> None:
    """Exercise reset, one declared action, outcome, and close per runtime."""

    for benchmark in sorted(selected, key=lambda item: item.value):
        case = next(item for item in cases if item.task.benchmark is benchmark)
        try:
            environment = await asyncio.to_thread(case.create_environment)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"{benchmark.value} environment preflight failed during creation"
            ) from exc
        try:
            state = await environment.reset()
            rendered_state = state.render() if isinstance(state, NativePublicState) else state
            action = _first_declared_action(rendered_state, benchmark)
            step = await environment.step(action)
            if not step.observation.strip():
                raise RuntimeError("preflight step returned an empty observation")
            outcome = await environment.outcome()
            if not 0.0 <= outcome.reward <= 1.0:
                raise RuntimeError("preflight outcome reward is invalid")
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                await environment.close()
            except (OSError, RuntimeError):
                pass
            raise RuntimeError(
                f"{benchmark.value} environment preflight failed during reset/step"
            ) from exc
        try:
            await environment.close()
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"{benchmark.value} environment preflight failed during close"
            ) from exc


def _first_declared_action(observation: str, benchmark: DirectBenchmark) -> str:
    marker = "Admissible actions:"
    _prefix, separator, suffix = observation.rpartition(marker)
    if not separator and benchmark is DirectBenchmark.SCIENCE_WORLD:
        return "look around"
    if not separator:
        raise ValueError("preflight observation does not declare admissible actions")
    actions = tuple(
        line.removeprefix("-").strip()
        for line in suffix.splitlines()
        if line.strip().startswith("-") and line.removeprefix("-").strip()
    )
    if not actions:
        raise ValueError("preflight observation has no admissible action")
    return actions[0]


def _attempt_value(attempt: NativeInteractiveAttempt) -> dict[str, object]:
    return {
        "benchmark": attempt.benchmark.value,
        "infrastructure_error": attempt.infrastructure_error,
        "cleanup_error": attempt.cleanup_error,
        "invalid_actions": attempt.invalid_actions,
        "reward": attempt.reward,
        "steps": attempt.steps,
        "submission_produced": attempt.submission_produced,
        "success": attempt.success,
        "task_id": attempt.task_id,
        "terminal_reached": attempt.terminal_reached,
        "terminated_by_horizon": attempt.terminated_by_horizon,
        "valid_actions": attempt.valid_actions,
        "termination_reason": attempt.termination_reason,
        "budget_exhausted": attempt.budget_exhausted,
        "terminal_success": attempt.terminal_success,
        "unfinished_subgoal_at_horizon": attempt.unfinished_subgoal_at_horizon,
        "trace": [
            {
                "step_index": step.step_index,
                "raw_text": step.raw_text,
                "reasoning_text": step.reasoning_text,
                "parsed_action": step.parsed_action,
                "parse_reason": step.parse_reason,
                "observation_before": step.observation_before,
                "available_actions_before": list(step.available_actions_before),
                "action_listed_before": step.action_listed_before,
                "official_action_valid": step.official_action_valid,
                "observation_after": step.observation_after,
                "available_actions_after": list(step.available_actions_after),
                "native_reward_after": step.native_reward_after,
                "official_terminal_after": step.official_terminal_after,
                "finish_reason": step.finish_reason,
                "prompt_tokens": step.prompt_tokens,
                "completion_tokens": step.completion_tokens,
                "response_model": step.response_model,
                "response_id": step.response_id,
                "service_instance_id": step.service_instance_id,
                "service_slot": step.service_slot,
                "structured_memory_before": step.structured_memory_before,
                "structured_memory_after": step.structured_memory_after,
                "visible_thought": step.visible_thought,
                "memory_update_status": step.memory_update_status,
                "repeated_action": step.repeated_action,
                "unchanged_state": step.unchanged_state,
                "revisited_product_or_location": step.revisited_product_or_location,
                "purchase_with_unmet_constraints": step.purchase_with_unmet_constraints,
            }
            for step in attempt.trace
        ],
    }


def _attempt_from_value(value: dict[str, object]) -> NativeInteractiveAttempt:
    trace_value = value.get("trace")
    if not isinstance(trace_value, list):
        raise ValueError("interactive journal trace is absent")
    trace: list[InteractiveStepRecord] = []
    for item in trace_value:
        if not isinstance(item, dict):
            raise ValueError("interactive journal trace row is not an object")
        if "observation_before" in item:
            trace.append(
                InteractiveStepRecord(
                    step_index=cast(int, item["step_index"]),
                    observation_before=cast(str, item["observation_before"]),
                    available_actions_before=tuple(
                        cast(list[str], item["available_actions_before"])
                    ),
                    raw_text=cast(str, item["raw_text"]),
                    reasoning_text=cast(str | None, item["reasoning_text"]),
                    parsed_action=cast(str | None, item["parsed_action"]),
                    parse_reason=cast(str, item["parse_reason"]),
                    action_listed_before=cast(bool | None, item["action_listed_before"]),
                    official_action_valid=cast(bool | None, item["official_action_valid"]),
                    observation_after=cast(str | None, item["observation_after"]),
                    available_actions_after=tuple(cast(list[str], item["available_actions_after"])),
                    native_reward_after=cast(float | None, item["native_reward_after"]),
                    official_terminal_after=cast(bool, item["official_terminal_after"]),
                    finish_reason=cast(str, item["finish_reason"]),
                    prompt_tokens=cast(int | None, item["prompt_tokens"]),
                    completion_tokens=cast(int | None, item["completion_tokens"]),
                    response_model=cast(str | None, item.get("response_model")),
                    response_id=cast(str | None, item.get("response_id")),
                    service_instance_id=cast(str | None, item.get("service_instance_id")),
                    service_slot=cast(int, item.get("service_slot", 0)),
                    structured_memory_before=cast(str | None, item.get("structured_memory_before")),
                    structured_memory_after=cast(str | None, item.get("structured_memory_after")),
                    visible_thought=cast(str | None, item.get("visible_thought")),
                    memory_update_status=cast(
                        str, item.get("memory_update_status", "not-applicable")
                    ),
                    repeated_action=cast(bool, item.get("repeated_action", False)),
                    unchanged_state=cast(bool, item.get("unchanged_state", False)),
                    revisited_product_or_location=cast(
                        bool, item.get("revisited_product_or_location", False)
                    ),
                    purchase_with_unmet_constraints=cast(
                        bool, item.get("purchase_with_unmet_constraints", False)
                    ),
                )
            )
            continue
        trace.append(
            InteractiveStepRecord(
                step_index=cast(int, item["step_index"]),
                observation_before=cast(str, item["observation"]),
                available_actions_before=(),
                raw_text=cast(str, item["raw_text"]),
                reasoning_text=cast(str | None, item["reasoning_text"]),
                parsed_action=cast(str | None, item["parsed_action"]),
                parse_reason=cast(str, item["parse_reason"]),
                action_listed_before=None,
                official_action_valid=cast(bool | None, item["action_valid"]),
                observation_after=cast(str, item["observation"]),
                available_actions_after=(),
                native_reward_after=cast(float | None, item["reward"]),
                official_terminal_after=cast(bool, item["terminal"]),
                finish_reason=cast(str, item["finish_reason"]),
                prompt_tokens=cast(int | None, item["prompt_tokens"]),
                completion_tokens=cast(int | None, item["completion_tokens"]),
            )
        )
    return NativeInteractiveAttempt(
        task_id=cast(str, value["task_id"]),
        benchmark=DirectBenchmark(cast(str, value["benchmark"])),
        reward=cast(float | None, value["reward"]),
        success=cast(bool | None, value["success"]),
        steps=cast(int, value["steps"]),
        valid_actions=cast(int, value["valid_actions"]),
        invalid_actions=cast(int, value["invalid_actions"]),
        submission_produced=cast(bool, value["submission_produced"]),
        terminal_reached=cast(bool, value["terminal_reached"]),
        infrastructure_error=cast(str | None, value["infrastructure_error"]),
        terminated_by_horizon=cast(bool, value["terminated_by_horizon"]),
        cleanup_error=cast(str | None, value["cleanup_error"]),
        termination_reason=cast(str | None, value["termination_reason"]),
        trace=tuple(trace),
        budget_exhausted=cast(bool, value.get("budget_exhausted", False)),
        terminal_success=cast(bool | None, value.get("terminal_success")),
        unfinished_subgoal_at_horizon=cast(str | None, value.get("unfinished_subgoal_at_horizon")),
    )


def _aggregate(
    attempts: tuple[NativeInteractiveAttempt, ...] | list[NativeInteractiveAttempt],
    protocol: DirectReferenceProtocol,
) -> dict[str, object]:
    grouped: dict[str, list[NativeInteractiveAttempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.benchmark.value].append(attempt)
    result: dict[str, object] = {}
    for benchmark, rows in sorted(grouped.items()):
        spec = protocol.benchmark(DirectBenchmark(benchmark))
        definitive = [row for row in rows if row.infrastructure_error is None]
        generation_infrastructure = sum(
            row.infrastructure_error == "DirectGenerationError" for row in rows
        )
        environment_infrastructure = sum(
            row.infrastructure_error is not None
            and row.infrastructure_error != "DirectGenerationError"
            for row in rows
        )
        complete = len(rows) == spec.sample_count and len(definitive) == spec.sample_count
        trace_rows = [step for row in definitive for step in row.trace]
        parsed_action_rows = [step for step in trace_rows if step.parsed_action is not None]
        known_action_rows = [step for step in parsed_action_rows if step.action_valid is not None]
        diagnostic_average = (
            100 * sum(cast(float, row.reward) for row in definitive) / len(definitive)
            if definitive
            else None
        )
        diagnostic_success = (
            100 * sum(row.success is True for row in definitive) / len(definitive)
            if definitive
            else None
        )
        observed_by_metric = {
            "average_score": diagnostic_average,
            "success_rate": diagnostic_success,
        }
        metrics = []
        for contract in spec.metrics:
            observed = observed_by_metric[contract.metric_id]
            formal = observed if complete else None
            gap = abs(formal - float(contract.reference_percent)) if formal is not None else None
            metrics.append(
                {
                    "metric": contract.metric_id,
                    "reference_percent": str(contract.reference_percent),
                    "diagnostic_observed_percent": str(observed) if observed is not None else None,
                    "formal_observed_percent": str(formal) if formal is not None else None,
                    "absolute_gap_pp": str(gap) if gap is not None else None,
                    "numeric_status": (
                        "PASS"
                        if gap is not None and gap < float(protocol.parity.max_gap_pp_exclusive)
                        else "INCOMPLETE"
                        if not complete
                        else "FAIL"
                    ),
                }
            )
        result[benchmark] = {
            "benchmark": benchmark,
            "comparability": spec.comparability.value,
            "population": spec.population,
            "dataset_revision": spec.dataset_revision,
            "selection_rule": spec.selection_rule,
            "prompt_profile": spec.prompt_profile,
            "decoding_profile": spec.decoding_profile,
            "parser_profile": spec.parser_profile,
            "scorer_profile": spec.scorer_profile,
            "seed_aggregation": spec.seed_aggregation.mode.value,
            "dataset_variant": spec.dataset_variant,
            "comparability_evidence": {
                "population": spec.evidence.population.value,
                "prompt": spec.evidence.prompt.value,
                "decoding": spec.evidence.decoding.value,
                "seed_aggregation": spec.evidence.seed_aggregation.value,
                "scorer": spec.evidence.scorer.value,
                "environment": spec.evidence.environment.value,
            },
            "availability": "complete" if complete else "incomplete",
            "metrics": metrics,
            "coverage": {
                "planned_count": spec.sample_count,
                "final_record_count": len(rows),
                "candidate_response_count": len(definitive),
                "definitive_verdict_count": len(definitive),
                "generation_infrastructure_failures": generation_infrastructure,
                "scorer_infrastructure_failures": 0,
                "environment_infrastructure_failures": environment_infrastructure,
            },
            "diagnostic_average_score_percent": diagnostic_average,
            "formal_average_score_percent": diagnostic_average if complete else None,
            "diagnostic_success_rate_percent": diagnostic_success,
            "formal_success_rate_percent": diagnostic_success if complete else None,
            "submission_rate_percent": (
                100 * sum(row.submission_produced for row in definitive) / len(definitive)
                if definitive
                else None
            ),
            "valid_action_rate_percent": (
                100
                * sum(step.action_valid is True for step in known_action_rows)
                / len(known_action_rows)
                if known_action_rows
                else None
            ),
            "parse_invalid_rate_percent": (
                100 * sum(step.parsed_action is None for step in trace_rows) / len(trace_rows)
                if trace_rows
                else None
            ),
            "environment_invalid_rate_percent": (
                100
                * sum(step.action_valid is False for step in known_action_rows)
                / len(known_action_rows)
                if known_action_rows
                else None
            ),
            "official_terminal_rate_percent": (
                100 * sum(row.terminal_reached for row in definitive) / len(definitive)
                if definitive
                else None
            ),
            "mean_steps": (
                sum(row.steps for row in definitive) / len(definitive) if definitive else None
            ),
            "horizon_rate_percent": (
                100 * sum(row.terminated_by_horizon for row in definitive) / len(definitive)
                if definitive
                else None
            ),
            "candidate_invalid_termination_rate_percent": (
                100
                * sum(row.termination_reason == "candidate-invalid" for row in definitive)
                / len(definitive)
                if definitive
                else None
            ),
            "repeated_action_count": sum(step.repeated_action for step in trace_rows),
            "unchanged_state_count": sum(step.unchanged_state for step in trace_rows),
            "revisited_product_or_location_count": sum(
                step.revisited_product_or_location for step in trace_rows
            ),
            "purchase_with_unmet_constraints_count": sum(
                step.purchase_with_unmet_constraints for step in trace_rows
            ),
            "memory_missing_count": sum(
                step.memory_update_status in {"missing-carried", "missing-empty"}
                for step in trace_rows
            ),
            "unfinished_subgoal_at_horizon_count": sum(
                row.unfinished_subgoal_at_horizon is not None for row in definitive
            ),
            "cleanup_error_count": sum(row.cleanup_error is not None for row in rows),
            "partial_reward_rate_percent": (
                100
                * sum(row.reward is not None and 0.0 < row.reward < 1.0 for row in definitive)
                / len(definitive)
                if definitive
                else None
            ),
            "infrastructure_failures": generation_infrastructure + environment_infrastructure,
        }
    return result


def main() -> None:
    print(json.dumps(asyncio.run(_run(_arguments())), sort_keys=True))


if __name__ == "__main__":
    main()
