"""Per-step structure statistics for an EvoSteer run, computed from its trajectories.

Usage (on the pod): python3 analyze_run.py <run_dir>  -> writes <run_dir>/analysis.json

For every committed batch and each source (current = pi_theta, natural_reference = rho)
and family: mean reward, fraction of episodes that add a verifier, fraction whose
output node is a verifier, mean team size, and action-kind frequencies.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path


def _executions(trajectory: dict) -> list[dict]:
    """Executed node events (request + result) from the terminal public state."""
    if not trajectory["decisions"]:
        return []
    raw = trajectory.get("terminal_state_json") or trajectory["decisions"][-1]["state_json"]
    state = json.loads(raw) if isinstance(raw, str) else raw
    events = []
    for event in state.get("history", []):
        obs = event.get("observation") or {}
        if obs.get("node_request") and obs.get("node_result"):
            events.append({"action": event.get("action") or {}, **obs})
    return events, state


def episode(trajectory: dict) -> dict:
    actions = [json.loads(d["action_json"]) for d in trajectory["decisions"]]
    executed, state = _executions(trajectory) if trajectory["decisions"] else ([], {})
    reruns = [e for e in executed if (e["action"] or {}).get("kind") == "RERUN_AGENT"]
    output_id = (state.get("graph") or {}).get("output_node_id")
    last_output = [e for e in executed if e.get("node_id") == output_id]
    outcome = lambda e: ((e["node_result"].get("metadata") or {}).get("execution_outcome") or {})
    roles = {a["node_id"]: a["role_id"] for a in actions if a["kind"] == "ADD_AGENT"}
    outputs = [a["node_id"] for a in actions if a["kind"] == "SET_OUTPUT"]
    return {
        "reward": trajectory["reward"],
        # A skill is bound either when an agent is created with it or by BIND_SKILL.
        "skill": any(a.get("skill_id") for a in actions),
        "verifier": "verifier" in roles.values(),
        "verifier_output": bool(outputs) and roles.get(outputs[-1]) == "verifier",
        "team_size": len(roles),
        # Repair: a node executed again, or feedback wiring between agents.
        "repair": any(a["kind"] in ("RERUN_AGENT", "ADD_EDGE") for a in actions),
        "minimal": [a["kind"] for a in actions] == ["ADD_AGENT", "SET_OUTPUT", "STOP"],
        "executions": len(executed),
        "truncated": sum(outcome(e).get("failure_kind") == "truncated" for e in executed),
        "reruns": len(reruns),
        # A rerun only adds information if the node received other agents' output.
        "informed_reruns": sum(bool(e["node_request"].get("messages")) for e in reruns),
        "output_truncated": bool(last_output)
        and outcome(last_output[-1]).get("failure_kind") == "truncated",
        "roles_used": sorted({a.get("role_id") for a in actions if a["kind"] == "ADD_AGENT"}),
        "kinds": Counter(a["kind"] for a in actions),
    }


def summarize(rows: list[dict]) -> dict:
    kinds = Counter()
    for row in rows:
        kinds.update(row["kinds"])
    n = len(rows)
    return {
        "n": n,
        "reward": sum(r["reward"] for r in rows) / n,
        "verifier_rate": sum(r["verifier"] for r in rows) / n,
        "skill_rate": sum(r["skill"] for r in rows) / n,
        "verifier_output_rate": sum(r["verifier_output"] for r in rows) / n,
        "team_size": sum(r["team_size"] for r in rows) / n,
        "repair_rate": sum(r["repair"] for r in rows) / n,
        "truncation_rate": sum(r["truncated"] for r in rows) / max(1, sum(r["executions"] for r in rows)),
        "informed_rerun_rate": sum(r["informed_reruns"] for r in rows)
        / max(1, sum(r["reruns"] for r in rows)),
        "output_truncated_rate": sum(r["output_truncated"] for r in rows) / n,
        "role_use": {
            role: sum(role in r["roles_used"] for r in rows) / n
            for role in sorted({x for r in rows for x in r["roles_used"] if x})
        },
        "minimal_rate": sum(r["minimal"] for r in rows) / n,
        "actions_per_episode": {k: v / n for k, v in sorted(kinds.items())},
    }


def main(run: Path) -> None:
    batches = []
    for path in sorted((run / "trajectories").glob("batch-*.jsonl")):
        groups: dict[tuple[str, str], list[dict]] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            trajectory = json.loads(line)
            item = episode(trajectory)
            for family in (trajectory["task"]["family"], "all"):
                groups.setdefault((trajectory["source"], family), []).append(item)
        batches.append(
            {
                "batch": path.stem,
                "committed_at": os.path.getmtime(run / "batches" / f"{path.stem}.json"),
                "stats": {f"{s}/{f}": summarize(rows) for (s, f), rows in sorted(groups.items())},
            }
        )
    (run / "analysis.json").write_text(json.dumps(batches, indent=1))
    for batch in batches:
        stats = batch["stats"]
        line = [batch["batch"]]
        for family in ("all", "hotpotqa", "nq_open", "medqa", "aime_2026", "mbpp_plus"):
            cur, ref = stats.get(f"current/{family}"), stats.get(f"natural_reference/{family}")
            if cur and ref:
                line.append(
                    f"{family}: R {cur['reward']:.2f}/{ref['reward']:.2f} "
                    f"V {cur['verifier_rate']:.2f}/{ref['verifier_rate']:.2f} "
                    f"S {cur['skill_rate']:.2f}/{ref['skill_rate']:.2f} "
                    f"Rep {cur['repair_rate']:.2f}/{ref['repair_rate']:.2f}"
                )
        print(" | ".join(line))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
