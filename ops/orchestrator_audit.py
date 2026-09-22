"""What the orchestrator did in the live run: decisions, structures, and one worked episode."""
import json, glob, sys, collections
run = sys.argv[1]
trajs = []
for path in sorted(glob.glob(run + "/trajectories/batch-*.jsonl")):
    trajs += [json.loads(l) for l in open(path)]
natural = [t for t in trajs if t["source"] in ("current", "natural_reference")]
kinds = collections.Counter(); per_ep = []
def structure(actions):
    roles = {a["node_id"]: a.get("role_id") for a in actions if a["kind"] == "ADD_AGENT"}
    k = [a["kind"] for a in actions]
    tags = []
    if "verifier" in roles.values(): tags.append("verifier")
    if "RERUN_AGENT" in k: tags.append("rerun")
    if "ADD_EDGE" in k: tags.append("edge")
    if "DROP_AGENT" in k: tags.append("drop")
    if any(a.get("skill_id") for a in actions): tags.append("skill")
    if sum(1 for r in roles.values() if r == "solver") > 1: tags.append("multi-solver")
    return "+".join(tags) or "single solver only"
by_struct = collections.defaultdict(list)
for t in natural:
    actions = [json.loads(d["action_json"]) for d in t["decisions"]]
    kinds.update(a["kind"] for a in actions)
    per_ep.append(len(actions))
    by_struct[structure(actions)].append(t["reward"])
print(f"batches {sorted({t['batch_id'] for t in trajs})}, natural episodes {len(natural)}, decisions {sum(per_ep)} (mean {sum(per_ep)/len(per_ep):.2f}/episode)")
print("action kinds:", dict(kinds.most_common()))
print("structure -> count, mean reward:")
for s, r in sorted(by_struct.items(), key=lambda x: -len(x[1])):
    print(f"  {s:28s} n={len(r):3d} reward={sum(r)/len(r):.2f}")
# legal-action menu size at each decision
sizes = [len(d["legal_token_paths"]) for t in natural for d in t["decisions"]]
print("legal actions per decision: min", min(sizes), "mean", round(sum(sizes)/len(sizes), 1), "max", max(sizes))
# worked example: an episode with a verifier and a rerun
def pick():
    for t in natural:
        acts = [json.loads(d["action_json"])["kind"] for d in t["decisions"]]
        if "RERUN_AGENT" in acts and len(acts) >= 5 and t["task"]["family"] in ("aime_2026", "hotpotqa", "medqa"):
            return t
    return natural[0]
t = pick()
print(f"\n===== EPISODE {t['sample_id']} reward={t['reward']} =====")
first = json.loads(t["decisions"][0]["state_json"])
print("orchestrator sees at decision 1 (keys of the public state):", sorted(first.keys()))
print("  roles:", [r["role_id"] for r in first["roles"]], " skill_menu:", first.get("skill_menu"))
for i, d in enumerate(t["decisions"], 1):
    a = json.loads(d["action_json"])
    state = json.loads(d["state_json"])
    print(f"\n[decision {i}] value_estimate={d['value_estimate']:.3f} legal_actions={len(d['legal_token_paths'])} graph_nodes={len(state['graph']['nodes'])}")
    print("  ACTION:", json.dumps(a))
nxt = json.loads(t["decisions"][-1]["state_json"])
term = t.get("terminal_state_json")
final = json.loads(term) if isinstance(term, str) else nxt
for ev in final.get("history", []):
    obs = ev.get("observation") or {}
    req, res = obs.get("node_request"), obs.get("node_result")
    act = ev.get("action")
    line = f"  -> {json.dumps(act)[:110]}"
    if req:
        out = res["output"]
        oc = res["metadata"].get("execution_outcome", {})
        line += (f"\n     executed {req['role']['role_id']} node; input: task + {len(req['messages'])} message(s)"
                 f" {'+ previous output' if req['previous_output'] else ''} {'+ skill' if req['skills'] else ''};"
                 f" output {len(out)} chars, {res['usage']['output_tokens']} tokens, {oc.get('failure_kind') or 'answered'}"
                 f"\n     output tail: {out[-160:]!r}")
    print(line)
print("final output node:", final["graph"].get("output_node_id"))
