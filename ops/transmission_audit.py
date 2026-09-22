"""Audit what each executed agent actually received in the live run."""
import json, glob, sys, collections
run = sys.argv[1]
rows = {}
for l in open("/workspace/evosteer/SKILLEV-new-main/data/paper_benchmarks/paper_benchmarks_9b_mixed.jsonl"):
    r = json.loads(l); rows[r["task_id"]] = r
sys.path.insert(0, "/workspace/evosteer/SKILLEV-new-main/src")
from skillev.experiments.curve_benchmarks import _task_prompt
stats = collections.Counter(); by_role = collections.Counter(); msg_by_role = collections.Counter()
omitted = []; example = None; skill_example = None
for path in sorted(glob.glob(run + "/trajectories/batch-*.jsonl")):
    for line in open(path):
        t = json.loads(line)
        task_id = t["task"]["task_id"]
        terminal = json.loads(t["decisions"][-1]["state_json"]) if t["decisions"] else {}
        for event in terminal.get("history", []):
            obs = event.get("observation") or {}
            req, res = obs.get("node_request"), obs.get("node_result")
            if not req or not res:
                continue
            stats["executions"] += 1
            role = req["role"]["role_id"]; by_role[role] += 1
            stats["task_text_complete"] += req["task_prompt"] == _task_prompt(rows[task_id])
            if req["messages"]:
                msg_by_role[role] += 1
            if req["skills"]:
                stats["with_skill"] += 1
                if skill_example is None: skill_example = (task_id, req, res)
            if req["previous_output"] is not None:
                stats["with_previous_output"] += 1
            meta = res.get("metadata", {})
            om = (meta.get("prompt_fit") or {}).get("omitted_characters", 0)
            if om: omitted.append(om)
            outcome = meta.get("execution_outcome", {})
            stats["truncated_output"] += outcome.get("failure_kind") == "truncated"
            if example is None and role == "verifier" and req["messages"] and len(req["task_prompt"]) < 900:
                example = (task_id, req, res)
print(json.dumps(stats), "\nexecutions by role", dict(by_role), "\nexecutions that received inputs from other agents, by role", dict(msg_by_role))
print("executions with shortened inputs:", len(omitted), "chars omitted (max)", max(omitted) if omitted else 0)
from skillev.orchestration.model_executor import _render
from skillev.orchestration.graph import NodeExecutionRequest, RoleSpec
def show(ex, label):
    task_id, req, res = ex
    r = NodeExecutionRequest(runtime_id="x", node_id="x", role=RoleSpec.from_value(req["role"]), task_prompt=req["task_prompt"],
        skills=tuple((s["skill_id"], s["body"]) for s in req["skills"]), messages=tuple(req["messages"]),
        previous_output=req["previous_output"], execution_index=1, seed=0)
    text = _render(r, list(req["messages"]), req["previous_output"])
    print(f"\n===== {label}: {task_id} (prompt {len(text)} chars) =====\n" + (text if len(text) < 3500 else text[:2000] + "\n[...]\n" + text[-1200:]))
    print(f"----- output ({len(res['output'])} chars) -----\n" + res["output"][:600])
if example: show(example, "VERIFIER INPUT EXAMPLE")
if skill_example: show(skill_example, "NODE WITH A BOUND SKILL")
