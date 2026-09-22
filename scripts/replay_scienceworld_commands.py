"""Replay a fixed private command packet without a policy, scorer or candidate selection.

This is a diagnostic environment execution, NOT a new scored model evaluation.
Never rewrites a command, resets after failure, or promotes a replay to a result.
Compatible with the official environment's Python 3.8 interpreter.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def public_observation(result: dict[str, Any]) -> str:
    names = ("current_look", "inventory", "task_description")
    return "\n\n".join(
        [
            result["observation_text"],
            *(name + ":\n" + result["public_state"][name] for name in names),
        ]
    )


def replay_one(
    directory: Path,
    specification: dict[str, Any],
    output: Path,
    worker_class: Any,
    jar_override: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    task = json.loads((directory / "task.json").read_text())
    frozen = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
    literal = (directory / "actions.txt").read_text().splitlines()
    if literal != [row["command"] for row in frozen]:
        raise ValueError("the packet's two frozen command representations disagree")
    manifest, case = specification["manifest"], specification["case"]
    deployment = manifest["deployments"][case["deployment"]]
    runtime = manifest["runtimes"][deployment["runtime"]]
    target = output / directory.name
    target.mkdir()
    worker = None
    report: dict[str, Any] = {
        "task_id": task["task_id"],
        "planned_actions": len(frozen),
        "executed_actions": 0,
        "differences": [],
        "model_calls": 0,
        "candidate_replacement": False,
        "environment_source_revision": runtime["source_revision"],
        "diagnostic_jar": str(jar_override) if jar_override else None,
    }
    try:
        worker = worker_class(
            runtime["source_root"],
            {
                "jar_path": str(jar_override) if jar_override else deployment["jar_path"],
                "seed": task["seed"],
                "simplification": task["simplification"],
                "private_diagnostics": True,
            },
            {
                "task_name": case["payload"]["task_name"],
                "variation_index": case["payload"]["variation_index"],
                "max_steps": case["max_steps"],
            },
        )
        initial = worker.reset()
        (target / "reset-private.json").write_text(
            json.dumps(initial, ensure_ascii=False, indent=2)
        )
        rendered = "Task: " + initial["task_description"] + "\n\n" + public_observation(initial)
        report["initial_public_observation_equal"] = (
            rendered == task["initial_public_state"]["observation_text"]
        )
        if not report["initial_public_observation_equal"]:
            report["differences"].append({"step": 0, "field": "initial_public_observation"})
        with (target / "transitions-private.jsonl").open("x") as stream:
            for index, original in enumerate(frozen, 1):
                result = worker.step(original["command"])
                comparisons = {
                    "observation_text": (
                        result["observation_text"],
                        original["transition"]["observation"],
                    ),
                    "score": (result["score"], original["transition"]["native_score"]),
                    "terminal": (result["terminal"], original["transition"]["terminal"]),
                    "moves": (result["native_moves"], original["transition"]["simulator_moves"]),
                    "public_observation": (
                        public_observation(result),
                        original["environment_response"]["observation"],
                    ),
                }
                mismatches = [key for key, (left, right) in comparisons.items() if left != right]
                report["differences"].extend({"step": index, "field": key} for key in mismatches)
                stream.write(
                    json.dumps(
                        {
                            "step": index,
                            "command": original["command"],
                            "result": result,
                            "differences": mismatches,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stream.flush()
                report["executed_actions"] = index
                report["final_native_score"] = result["score"]
                report["terminal_reached"] = result["terminal"]
                report["first_failure"] = result["private_diagnostics"].get("first_failure")
                if result["terminal"]:
                    break
        report["status"] = "replayed"
        if report["executed_actions"] != len(frozen):
            report["differences"].append({"field": "termination_position"})
    except Exception as error:
        report.update(
            status="infrastructure-error", error=repr(error), traceback=traceback.format_exc()
        )
    finally:
        if worker is not None:
            worker.close()
    report["elapsed_seconds"] = time.monotonic() - started
    (target / "report-private.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jar-override", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.source_jsonl.read_text().splitlines()]
    sources = {row["task_id"]: row["interactive"] for row in records}
    roots = {
        raw["source_root"]
        for spec in sources.values()
        for raw in spec["manifest"]["runtimes"].values()
    }
    if len(roots) != 1:
        raise ValueError("fixed replay requires one original native source deployment")
    # As in the official worker's initialize path, the upstream package must
    # precede our bridge directory (which also contains a scienceworld.py).
    sys.path[:0] = [next(iter(roots)), str(args.worker_script.parent)]
    spec = importlib.util.spec_from_file_location("official_environment_worker", args.worker_script)
    if spec is None or spec.loader is None:
        raise ImportError("the standalone environment worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    episodes = sorted((args.packet / "episodes").iterdir())
    reports, started = [], time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(
                replay_one,
                directory,
                sources[json.loads((directory / "task.json").read_text())["task_id"]],
                args.output,
                module.ScienceWorldWorker,
                args.jar_override,
            )
            for directory in episodes
        ]
        for future in concurrent.futures.as_completed(jobs):
            report = future.result()
            reports.append(report)
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "done": len(reports),
                        "total": len(jobs),
                        "status": report["status"],
                        "actions": sum(row["executed_actions"] for row in reports),
                        "episodes_per_minute": len(reports) / elapsed * 60,
                        "eta_seconds": elapsed / len(reports) * (len(jobs) - len(reports)),
                    }
                ),
                flush=True,
            )
            (args.output / "summary-private.json").write_text(
                json.dumps(
                    {
                        "planned": len(jobs),
                        "completed": len(reports),
                        "model_calls": 0,
                        "replay_only": True,
                        "reports": sorted(reports, key=lambda row: row["task_id"]),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
