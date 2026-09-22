"""Owner-authorized item-level operational recovery, not a fresh evaluation arm.

Keep completed in-budget answers, even incorrect ones. Select replacements using
only transport/timing/token metadata, before scoring; retain original task IDs
and immutable source runs. All generated plans, source rows and scores are private.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.integrity_final_validation import validate_persisted_owner_source
from skillev.evaluation.integrity_metric_schema import (
    NATIVE_VERIFIER_VERSIONS,
    aggregate_secondary_metrics,
)
from skillev.evaluation.integrity_results import NativeScore, exact_panel_join, native_mean
from skillev.evaluation.sealed_candidates import CandidateJournal, CandidateReader, EventOrigin

from .ood_scoring import score_ood
from .ood_sources import load_ood_panel


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def write_new(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def recovery_reason(
    reader: CandidateReader, scope: tuple[str, str, str], *, token_limit: int, wall_limit: float
) -> str | None:
    """No answer, correctness, native verdict or judge result is consulted."""
    try:
        final = reader.get(*scope)
    except KeyError:
        attempted = reader.connection.execute(
            "SELECT 1 FROM episode_attempts WHERE run_id=? AND arm_id=? AND episode_id=?", scope
        ).fetchone()
        return "unfinished-attempt" if attempted else "unstarted"
    if final.completion_tokens > token_limit:
        return "output-token-limit"
    timings = reader.traces(scope, "episode-timing", origin=EventOrigin.MODEL_TRANSPORT)
    if any(isinstance(row, dict) and float(row["wall_seconds"]) > wall_limit for row in timings):
        return "episode-wall-limit"
    # Older runs did not record per-episode wall time. Missing timing is unknown,
    # not evidence of a timeout and not a license to regenerate a valid answer.
    return None


def plan_recovery(
    *,
    original_config: Path,
    old_journal: Path,
    old_run_id: str,
    new_run_id: str,
    new_arm_id: str,
    recipe: Path,
    output: Path,
    previous_plan: Path | None = None,
) -> dict[str, Any]:
    base, limits = read_json(original_config), read_json(recipe)
    source = load_ood_panel(base)
    old_arm = base["arms"][0]["condition_id"]
    if len(base["arms"]) != 1 or new_run_id == old_run_id:
        raise ValueError("recovery requires one owner and a distinct run identity")
    previous = read_json(previous_plan) if previous_plan else None
    prior = {row["task_id"]: row for row in previous["records"]} if previous else {}
    expected = {entry.task_id for entry in source.panel.entries}
    if previous and (len(prior) != len(previous["records"]) or set(prior) != expected):
        raise ValueError("previous recovery does not cover the same original panel")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    new_config_path = output / "runtime-private.json"
    new_journal = output / "run" / "candidates-private.sqlite"
    records, selected_ids = [], set()
    reasons: dict[str, Counter[str]] = {}
    readers: dict[str, CandidateReader] = {}
    try:
        for entry in source.panel.entries:
            old = (
                prior[entry.task_id]["selected"]
                if previous
                else {
                    "journal": str(old_journal.resolve()),
                    "config": str(original_config.resolve()),
                    "run_id": old_run_id,
                    "arm_id": old_arm,
                }
            )
            path = old["journal"]
            if path not in readers:
                readers[path] = CandidateReader(Path(path))
            token_limit = int(limits["budgets"][entry.benchmark]["total_output_tokens"])
            reason = recovery_reason(
                readers[path],
                (old["run_id"], old["arm_id"], entry.task_id),
                token_limit=token_limit,
                wall_limit=float(limits["episode_timeout_seconds"]),
            )
            chosen = old
            if reason is not None:
                selected_ids.add(entry.task_id)
                chosen = {
                    "journal": str(new_journal.resolve()),
                    "config": str(new_config_path.resolve()),
                    "run_id": new_run_id,
                    "arm_id": new_arm_id,
                }
            records.append(
                {
                    "task_id": entry.task_id,
                    "benchmark": entry.benchmark,
                    "reason": reason or "retained-completed-in-budget",
                    "previous": old,
                    "selected": chosen,
                }
            )
            reasons.setdefault(entry.benchmark, Counter())[reason or "retained"] += 1
    finally:
        for reader in readers.values():
            reader.close()
    config = {**base, **limits}
    config["arms"] = [{**base["arms"][0], "condition_id": new_arm_id}]
    config["ood_sources"], config["evaluation_sample_counts"] = {}, {}
    config["ood_panel_exposure"] = "owner-authorized-operational-recovery-subset"
    config["ood_provenance"] = {
        "original_population": base["ood_provenance"],
        "selection": "operational-token-wall-or-unfinished-only; no score-based selection",
        "original_config": str(original_config.resolve()),
        "previous_plan": str(previous_plan.resolve()) if previous_plan else None,
    }
    config["scorers"] = {name: dict(value) for name, value in base["scorers"].items()}
    config["scorers"]["omni-math"]["judgements_path"] = str(output / "luna-judgements-private.json")
    for name in ("livecodebench", "apps-introductory"):
        config["scorers"][name]["wall_timeout_seconds"] = 300
    for benchmark, path in base["ood_sources"].items():
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        rows = [row for row in rows if row["task_id"] in selected_ids]
        if not rows:
            continue
        subset = output / f"{benchmark}-private.jsonl"
        with subset.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        config["ood_sources"][benchmark] = str(subset)
        config["evaluation_sample_counts"][benchmark] = len(rows)
    if selected_ids:
        load_ood_panel(config)
        write_new(new_config_path, config)
    plan = {
        "profile": "ood-operational-recovery@1",
        "exposure": "mixed-decoding-conditions; not a uniform fresh run or paired comparison",
        "original_config": str(original_config.resolve()),
        "previous_plan": str(previous_plan.resolve()) if previous_plan else None,
        "recipe": limits,
        "counts": reasons,
        "records": records,
        "historical_missing_episode_timing": "unknown; not inferred from token count",
        "replacement_count": len(selected_ids),
        "total_count": len(records),
    }
    write_new(output / "plan-private.json", plan)
    return plan


async def aggregate_recovery(plan_path: Path) -> dict[str, Any]:
    """Native grading only, resumable by immutable source scope; no generation."""
    plan = read_json(plan_path)
    base = read_json(Path(plan["original_config"]))
    source = load_ood_panel(base)
    rows = {row["task_id"]: row for row in plan["records"]}
    if len(rows) != len(plan["records"]) or set(rows) != {e.task_id for e in source.panel.entries}:
        raise ValueError("recovery lineage must join the complete original panel")
    readers: dict[str, CandidateJournal] = {}
    configs: dict[str, dict[str, Any]] = {}
    scores: dict[str, list[NativeScore]] = {}
    tokens: Counter[str] = Counter()
    selected_sources: dict[str, Counter[str]] = {}
    try:
        for entry in source.panel.entries:
            row = rows[entry.task_id]["selected"]
            scope = (row["run_id"], row["arm_id"], entry.task_id)
            path = row["journal"]
            if path not in readers:
                readers[path] = CandidateJournal(Path(path))
            reader = readers[path]
            final = reader.get(*scope)
            validate_persisted_owner_source(
                reader, final, benchmark=entry.benchmark, native_thinking=False, adapter_name=None
            )
            reason = recovery_reason(
                reader,
                scope,
                token_limit=int(plan["recipe"]["budgets"][entry.benchmark]["total_output_tokens"]),
                wall_limit=float(plan["recipe"]["episode_timeout_seconds"]),
            )
            if reason:
                raise RuntimeError(f"selected record still requires operational recovery: {reason}")
            stored = reader.stored_score(scope)
            if stored is not None:
                score = NativeScore.from_value(stored)
            else:
                config_path = row["config"]
                if config_path not in configs:
                    configs[config_path] = read_json(Path(config_path))
                config = configs[config_path]
                settings = {key: dict(value) for key, value in config["scorers"].items()}
                for name in ("livecodebench", "apps-introductory"):
                    settings[name]["wall_timeout_seconds"] = 300

                def diagnostics(
                    value: dict[str, Any],
                    *,
                    journal: CandidateJournal = reader,
                    bound_scope: tuple[str, str, str] = scope,
                ) -> None:
                    journal.record(
                        bound_scope, "native-ood-verdict", value, origin=EventOrigin.SCORER
                    )

                score = await score_ood(
                    reader,
                    scope,
                    entry.benchmark,
                    source.targets[entry.task_id],
                    settings=settings,
                    sandbox=ActorSandbox.current(Path(base["public_source"])),
                    diagnostics=diagnostics,
                )
                # Append native grading to its actual source journal, never copy
                # or relabel a candidate into a new arm to satisfy score ownership.
                reader.save_score(scope, asdict(score))
            scores.setdefault(entry.benchmark, []).append(score)
            tokens[entry.benchmark] += final.completion_tokens
            selected_sources.setdefault(entry.benchmark, Counter())[row["run_id"]] += 1
            print(
                json.dumps(
                    {
                        "scored": sum(map(len, scores.values())),
                        "total": len(rows),
                        "benchmark": entry.benchmark,
                    }
                ),
                flush=True,
            )
    finally:
        for reader in readers.values():
            reader.close()
    domains = {}
    for benchmark, values in scores.items():
        expected = tuple(e.task_id for e in source.panel.entries if e.benchmark == benchmark)
        joined = exact_panel_join(
            expected, tuple(values), expected_verifier=NATIVE_VERIFIER_VERSIONS[benchmark]
        )
        domains[benchmark] = {
            "count": len(joined),
            "metric": joined[0].metric,
            "value": native_mean(joined),
            "secondary_metrics": aggregate_secondary_metrics(joined),
            "completion_tokens": tokens[benchmark],
            "source_run_counts": selected_sources[benchmark],
        }
    summary = {
        "profile": plan["profile"],
        "exposure": plan["exposure"],
        "domains": domains,
        "total_count": len(rows),
        "lineage": str(plan_path.resolve()),
    }
    write_new(plan_path.parent / "summary-private.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for name in ("original-config", "old-journal", "recipe", "output"):
        plan.add_argument("--" + name, type=Path, required=True)
    for name in ("old-run-id", "new-run-id", "new-arm-id"):
        plan.add_argument("--" + name, required=True)
    plan.add_argument("--previous-plan", type=Path)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--plan", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = (
        plan_recovery(**args)
        if command == "plan"
        else asyncio.run(aggregate_recovery(args["plan"]))
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
