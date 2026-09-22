"""Private fixed-source development comparisons through the real readonly collector.

This is neither IID/A0 nor training. Targets remain in native evaluators. No
application, optimizer, posterior or evolution runtime is constructed. A repeated
candidate uses a new output and the original selection-private.json as reference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, cast

from skillev.contracts import JsonValue, normalize_json
from skillev.policy import QwenMultimodalBackboneConfig, QwenTokenizerAdapter
from skillev.rollout import PolicySnapshot
from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator
from skillev.rollout.readonly_collection import evaluation_isolation
from skillev.runtime import SkillLibraryState
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.runtime.serving_profile import (
    require_same_profile,
    require_training_service,
    serving_profile,
)
from skillev.training.inflight import durable_json
from skillev.training.performance_config import TrainingPerformanceConfig
from skillev.training.rollout_workflow import RolloutWorkflowResources
from skillev.training.run_condition import EffectiveRunCondition
from skillev_private.benchmarks.evaluation_episode import EvaluationEpisodeRecord
from skillev_private.benchmarks.mbpp_scoring import resolve_mbpp_profile
from skillev_private.benchmarks.official_process import ALFWorldGameDeployment
from skillev_private.benchmarks.protocol_v13_training_sessions import (
    build_protocol13_training_sessions,
    native_scorer_contracts,
)
from skillev_private.evaluation.iid_architecture import decode_record
from skillev_private.evaluation.iid_episode_sources import canonical_source_key
from skillev_private.evaluation.iid_live_binding import observe_service

from .development_forward import (
    bind_forward_gateway,
    fixed_snapshot,
    require_forward_candidate,
    resolve_forward,
)
from .formal_episode_config import formal_actor_transport
from .fresh_restart import load_fresh_config
from .readonly_replicas import (
    ReadonlyReplicaGenerator,
    configure_replica_admission,
    measured_capacity,
    replica_declaration,
    require_capacity,
)
from .skill_practice import PracticeSessionFactory, practice_controls
from .zero_update_bridge import collect_training_condition

FORMAT = "fixed-source-development-collection@1"
DOMAINS = {
    "hotpotqa": "hotpotqa",
    "triviaqa": "triviaqa",
    "aime-2026": "aime-historical",
    "healthbench": "healthbench",
    "alfworld": "alfworld",
    "mbpp-plus": "mbpp-plus",
    "humaneval": "humaneval",
}


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save(path: Path, value: Any) -> None:
    durable_json(path, cast(dict[str, JsonValue], normalize_json(value)))


def selection(
    request: dict[str, Any],
) -> tuple[tuple[EvaluationEpisodeRecord, ...], dict[str, Any]]:
    """Read/validate all isolation evidence BEFORE constructing any live dependency."""
    practice = practice_controls(request.get("development_practice"))
    replicas = replica_declaration(request)
    if request.get("format") != FORMAT or request.get("arm") not in {
        "skills-off",
        "initial-library",
    }:
        raise ValueError("an explicit development-only request and library arm are required")
    if not all(
        isinstance(request.get(k), str) and request[k].strip()
        for k in ("condition_id", "comparison_id")
    ):
        raise ValueError("candidate and fixed comparison identities must be declared")
    if type(request.get("chunk_size")) is not int or request["chunk_size"] < 1:
        raise ValueError("development collection requires a positive fixed chunk size")
    raw = [
        json.loads(line)
        for line in Path(request["source_records"]).read_text().splitlines()
        if line.strip()
    ]
    records = tuple(decode_record(row) for row in raw)
    manifest = read(request["source_manifest"])
    if manifest.get("comparison_id") != request["comparison_id"] or manifest.get("seed") != 0:
        raise ValueError("source manifest must declare this comparison and the single seed zero")
    sources = manifest["ordered_sources"]
    aliases = read(request["source_aliases"])
    for mapping in aliases.values():
        if not isinstance(mapping, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or mapping.get(v, v) != v
            for k, v in mapping.items()
        ):
            raise ValueError("declared aliases must map directly to canonical source identities")
    if not records or len(records) != len(sources):
        raise ValueError("the entire nonempty source manifest must be supplied exactly once")
    canonical = []
    for record, source in zip(records, sources, strict=True):
        domain = record.episode.benchmark.value
        if domain not in DOMAINS or source.get("report_domain") != DOMAINS[domain]:
            raise ValueError("development requires a declared seven-domain native source")
        if not all(
            isinstance(source.get(k), str) and source[k].strip()
            for k in ("source_dataset", "source_revision", "source_split")
        ):
            raise ValueError("real dataset/revision/split provenance is required")
        key = canonical_source_key((domain, record.episode.source_id), aliases)
        if (
            key != (source["benchmark"], source["source_id"])
            or source["population_id"] != record.episode.population_id
            or source["task_id"] != record.input.task_id
        ):
            raise ValueError("source manifest differs from actual ordered native records")
        if domain == "aime-2026":
            context = record.input.public_context
            payload = context.get("payload", {}) if isinstance(context, dict) else {}
            if source["source_dataset"].casefold() in {"aime2026", "aime-2026"} or (
                isinstance(payload, dict) and payload.get("benchmark_slice") == "AIME2026"
            ):
                raise ValueError("historical AIME development cannot claim the AIME2026 population")
        canonical.append(key)
    if len(set(canonical)) != len(canonical) or len({r.input.task_id for r in records}) != len(
        records
    ):
        raise ValueError("development comparisons require unique canonical sources and task IDs")
    excluded_files = request["exclusions"]
    if set(excluded_files) != {"a0", "training", "quality"}:
        raise ValueError("all frozen A0, planned-training and quality exclusions are required")
    exclusions = {name: read(path) for name, path in excluded_files.items()}
    exclusion_counts = {}
    for purpose, rows in exclusions.items():
        keys = {canonical_source_key((r["benchmark"], r["source_id"]), aliases) for r in rows}
        if set(canonical) & keys:
            raise ValueError(f"development source overlaps {purpose}; no model call is permitted")
        exclusion_counts[purpose] = len(keys)
    # Observed failure: a dataset split (valid_seen/valid_unseen) was supplied
    # as execution train_eval. Reuse the official pure deployment contract before
    # service inspection, hydration, model dispatch or environment reset.
    for record in records:
        if record.episode.benchmark.value != "alfworld":
            continue
        route = record.output.target.get("environment_route")
        if not isinstance(route, dict) or not isinstance(route.get("game_file"), str):
            raise ValueError("development ALFWorld needs its native game route")
        game_file = Path(cast(str, route["game_file"]))
        if not game_file.is_file():
            raise ValueError("development ALFWorld declared game file is absent")
        ALFWorldGameDeployment(game_file.parent, cast(str, route.get("mode")), record.input.query)
    policy, forward_initialization = resolve_forward(request)
    library = SkillLibraryState.from_value(read(request["initial_library"]))
    frozen = {
        "format": FORMAT,
        "purpose": "development-only",
        "comparison_id": request["comparison_id"],
        "seed": 0,
        "records": raw,
        "source_manifest": manifest,
        "source_aliases": aliases,
        "exclusions": exclusions,
        "exclusion_source_counts": exclusion_counts,
        "policy": policy.to_value(),
        "initial_library": library.to_value(),
        "arm": request["arm"],
        "chunk_size": request["chunk_size"],
        **replicas,
    }
    if forward_initialization:
        frozen["forward_initialization"] = forward_initialization
    if practice is not None:
        frozen["development_practice"] = practice
    if request.get("skill_utility_policy") is not None:
        from .skill_utility import SkillUtilityPolicy

        frozen["skill_utility_policy"] = SkillUtilityPolicy(
            **request["skill_utility_policy"]
        ).to_value()
    if request.get("autonomous_validation") is not None:
        from .autonomous_skill_validation import freeze_validation

        frozen["autonomous_validation"] = freeze_validation(
            request["autonomous_validation"], frozen
        )
    if request.get("comparison_reference") and read(request["comparison_reference"]) != frozen:
        raise ValueError("candidate changed fixed comparison sources/inputs/arm/policy/coordinates")
    return records, frozen


def aggregate(output: Path, frozen: dict[str, Any], *, elapsed_seconds: float) -> dict[str, Any]:
    """Raw native measurements only; missing outcomes stay null at the full denominator."""
    domains: dict[str, Any] = {}
    for position, source in enumerate(frozen["source_manifest"]["ordered_sources"]):
        row = domains.setdefault(
            source["report_domain"], {"planned": 0, "rewards": [], "success": [], "native": {}}
        )
        row["planned"] += 1
        path = output / "collection" / "episodes" / f"episode-{position:06d}-private.json"
        artifact = read(path).get("artifact") if path.exists() else None
        if artifact is None:
            continue
        reward = artifact["record"]["reward"]
        row["rewards"].append(reward["value"])
        row["success"].append(reward["success"])
        native = reward["native_payload"].get("public_metrics", {})
        if not isinstance(native, dict) or not native:
            native = {reward["native_metric_name"]: reward["value"]}
        for name, value in native.items():
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                row["native"].setdefault(name, []).append(value)
    measured = {}
    for domain, row in domains.items():
        count = row["planned"]
        complete = len(row["rewards"]) == count
        measured[domain] = {
            "planned": count,
            "completed": len(row["rewards"]),
            "reward_mean": math.fsum(row["rewards"]) / count if complete else None,
            "success_count": sum(row["success"]) if complete else None,
            "success_rate": sum(row["success"]) / count if complete else None,
            "native_metrics": {
                k: math.fsum(v) / count if len(v) == count else None
                for k, v in row["native"].items()
            },
        }
    ledger = output / "collection" / "episodes" / "ledger-private.json"
    return {
        "format": FORMAT,
        "record_kind": "development-evaluation",
        "eligible_for_a0_acceptance": False,
        "development_practice": frozen.get("development_practice"),
        **(
            {"forward_initialization": frozen["forward_initialization"]}
            if "forward_initialization" in frozen
            else {}
        ),
        "autonomous_skill_choice_measurement": frozen.get("development_practice") is None,
        "purpose": "independent development, not IID/A0 or formal training",
        "comparison_id": frozen["comparison_id"],
        "domains": measured,
        "source_provenance": frozen["source_manifest"]["ordered_sources"],
        "native_scorer_adapter_note": (
            "aime-2026 is the existing native integer scorer route, "
            "NOT this historical source population"
        ),
        "elapsed_seconds": elapsed_seconds,
        "consumed_budget": read(ledger)["settled"] if ledger.exists() else None,
        "phase_evidence": "collection/episodes/rollout-progress.json",
        "raw_outcomes": "collection/episodes/episode-*-private.json",
        **evaluation_isolation(),
    }


async def run(request: dict[str, Any], output: Path) -> dict[str, Any]:
    records, frozen = selection(request)
    formal = load_fresh_config(Path(request["formal_config"]))
    backbone = QwenMultimodalBackboneConfig.from_value(read(request["backbone"]))
    formal.require_backbone(backbone)
    policy = PolicySnapshot.from_value(frozen["policy"])
    if policy.tokenizer_id != backbone.tokenizer_id or formal.sampling_config.base_seed != 0:
        raise ValueError("development requires the declared tokenizer and single seed zero")
    library = SkillLibraryState.from_value(frozen["initial_library"])
    forward_initialization = frozen.get("forward_initialization", {})
    require_forward_candidate(
        forward_initialization,
        backbone=backbone.to_value(),
        sampling=formal.sampling_config.to_value(),
        library=library.to_value(),
    )
    if request["arm"] == "skills-off":
        library = SkillLibraryState.from_seed_documents(())
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    started = time.monotonic()
    save(output / "selection-private.json", frozen)
    save(output / "request-private.json", request)
    originals = output / "originals"
    originals.mkdir(mode=0o700)
    for key in (
        "source_records",
        "source_manifest",
        "source_aliases",
        "formal_config",
        "backbone",
        "initial_library",
        "deployments",
        "serving_profile",
    ):
        shutil.copyfile(request[key], originals / key)
    if forward_initialization:
        keys = (
            ("preparation", "effective_condition", "skill_sources")
            if forward_initialization["kind"] == "ttb-checkpoint"
            else ("preparation", "policy")
        )
        for key in keys:
            shutil.copyfile(forward_initialization[key], originals / f"forward-{key}.json")
    else:
        shutil.copyfile(request["step0_policy"], originals / "step0_policy")
    for key, path in request["exclusions"].items():
        shutil.copyfile(path, originals / f"exclude-{key}.json")
    generator = None
    try:
        endpoint = request["endpoint"]
        replicas = replica_declaration(request)
        actor_endpoints = tuple(
            replicas["actor_replica_pool"]["actor_endpoints"] if replicas else (endpoint,)
        )
        actor_requests = request.get("replica_actor_requests")
        inspected_endpoints = tuple(dict.fromkeys((endpoint, *actor_endpoints)))
        profiles = {}
        capacities = {}
        for member_endpoint in inspected_endpoints:
            observed = observe_service(member_endpoint)
            if replicas:
                capacities[member_endpoint] = measured_capacity(
                    observed["server_info"], include_request_limit=actor_requests is not None
                )
                save(output / "replica-capacities-before-private.json", capacities)
                if member_endpoint.rstrip("/").removesuffix("/v1") in actor_endpoints:
                    require_capacity(
                        capacities[member_endpoint],
                        input_tokens=formal.max_input_tokens,
                        actor_requests=actor_requests,
                        output_tokens=max(
                            formal.maximum_reasoning_tokens, formal.maximum_action_tokens
                        ),
                    )
            profile = serving_profile(observed["server_info"])
            require_same_profile(read(request["serving_profile"]), profile)
            require_training_service(
                profile,
                model_path=backbone.base_model_path,
                tokenizer_path=backbone.tokenizer_path or backbone.base_model_path,
                base_model=cast(str, profile["served_model_name"]),
                minimum_context=formal.max_input_tokens
                + max(formal.maximum_reasoning_tokens, formal.maximum_action_tokens),
                actor=True,
            )
            if (
                profile["dtype"] not in {"bfloat16", "torch.bfloat16"}
                or profile["quantization"] is not None
            ):
                raise ValueError("development requires actual unquantized BF16 service")
            if observed["model_info"]["model_path"] != backbone.base_model_path:
                raise ValueError("actual service model differs from the declaration")
            profiles[member_endpoint] = profile
        profile = profiles[endpoint]
        performance = TrainingPerformanceConfig.load(formal.performance_profile)
        workflow = performance.workflow()
        resources = RolloutWorkflowResources(workflow)
        configure_replica_admission(resources, replicas)
        mbpp = resolve_mbpp_profile(
            {"source_root": request["evalplus_source_root"], "profile": request["mbpp_profile"]}
        )
        scorers = native_scorer_contracts(mbpp, formal.healthbench_judge, domains=formal.domains)
        deployments = read(request["deployments"])
        controls = {
            "purpose": "development-only",
            "formal": formal.to_value(),
            "backbone": backbone.to_value(),
            "policy": policy.to_value(),
            "library": library.to_value(),
            "service": profile,
            "scorers": scorers,
            "deployments": deployments,
            "workflow": workflow.to_value(),
            "selection": frozen,
            "development_practice": frozen.get("development_practice"),
            **replicas,
            **(
                {"replica_service_profiles": profiles, "replica_capacity_before": capacities}
                if replicas
                else {}
            ),
        }
        save(output / "controls-private.json", controls)
        condition = EffectiveRunCondition.create(
            condition_id=request["condition_id"],
            scientific=cast(
                dict[str, JsonValue],
                normalize_json(
                    {
                        key: value
                        for key, value in controls.items()
                        if key != "replica_capacity_before"
                    }
                ),
            ),
            execution={
                "purpose": "development-only",
                "endpoint": endpoint,
                **({"replica_capacity_before": normalize_json(capacities)} if replicas else {}),
            },
        )
        gateway = bind_forward_gateway(
            forward_initialization,
            endpoint=endpoint,
            base_model=cast(str, profile["served_model_name"]),
            policy=policy,
        )
        journal = output / "evaluation-requests.sqlite3"
        _, sessions = await build_protocol13_training_sessions(
            records,
            deployments_path=Path(request["deployments"]),
            endpoint_base=endpoint,
            base_model=cast(str, profile["served_model_name"]),
            resources=resources,
            mbpp_interpreter=Path(request["mbpp_interpreter"]),
            mbpp_source_root=Path(request["evalplus_source_root"]),
            mbpp_profile=mbpp,
            hotpot_deliberation=formal.hotpot_deliberation,
            rollout_budget=formal.task_budget,
            static_rollout_budget=formal.static_task_budget,
            domain_rollout_budgets=formal.domain_task_budgets,
            lazy_environments=True,
            request_journal_path=journal,
            healthbench_judge=formal.healthbench_judge,
        )
        tokenizer = QwenTokenizerAdapter.from_config(backbone)
        request_journal = DurableRequestJournal(journal)
        members = []
        try:
            for member_endpoint in actor_endpoints:
                members.append(
                    ExternalSGLangRolloutGenerator(
                        config=formal_actor_transport(
                            member_endpoint, worker_threads=workflow.transport_worker_threads
                        ),
                        tokenizer=tokenizer,
                        gateway=gateway,
                        snapshot_provider=lambda: fixed_snapshot(gateway, policy),
                        request_journal=request_journal,
                    )
                )
            generator = (
                ReadonlyReplicaGenerator(tuple(members), request_journal)
                if replicas
                else members[0]
            )
        except BaseException:
            from contextlib import ExitStack

            with ExitStack() as stack:
                for member in members:
                    stack.callback(member.close)
            raise
        await collect_training_condition(
            root=output / "collection",
            condition=condition,
            tasks=tuple(r.input for r in records),
            generator=generator,
            base_sessions=sessions
            if request.get("development_practice") is None
            else PracticeSessionFactory(sessions, request["development_practice"]),
            library_state=library,
            trainer=formal.application_config(request["condition_id"]).trainer,
            maximum_h0_tokens=formal.max_input_tokens,
            workflow=workflow,
            workflow_resources=resources,
            sampling_schedule_id=request["comparison_id"],
            ordered_task_sequence_id=request["comparison_id"],
            sampled_policy_step=0,
            chunk_size=request["chunk_size"],
            execution_controls=cast(dict[str, JsonValue], normalize_json(controls)),
        )
        fixed_snapshot(gateway, policy)
        after_capacities = {}
        for member_endpoint in inspected_endpoints:
            after = observe_service(member_endpoint)
            if replicas:
                after_capacities[member_endpoint] = measured_capacity(
                    after["server_info"], include_request_limit=actor_requests is not None
                )
                save(output / "replica-capacities-after-private.json", after_capacities)
                if member_endpoint.rstrip("/").removesuffix("/v1") in actor_endpoints:
                    require_capacity(
                        after_capacities[member_endpoint],
                        input_tokens=formal.max_input_tokens,
                        actor_requests=actor_requests,
                        output_tokens=max(
                            formal.maximum_reasoning_tokens, formal.maximum_action_tokens
                        ),
                    )
            require_same_profile(profiles[member_endpoint], serving_profile(after["server_info"]))
            if after["model_info"]["model_path"] != backbone.base_model_path:
                raise ValueError("live replica model changed during collection")
        if replicas:
            save(
                output / "controls-after-private.json",
                {
                    **controls,
                    "replica_capacity_after": after_capacities,
                },
            )
        if read(request["deployments"]) != deployments:
            raise ValueError("native deployment controls changed during collection")
    except BaseException as error:
        save(
            output / "failure-private.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            },
        )
        save(
            output / "summary.json",
            aggregate(output, frozen, elapsed_seconds=time.monotonic() - started),
        )
        raise
    finally:
        if generator is not None:
            generator.close()
    summary = aggregate(output, frozen, elapsed_seconds=time.monotonic() - started)
    save(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = asyncio.run(run(read(args.request), args.output.resolve()))
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
