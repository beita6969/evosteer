"""Did the policy's structure choices improve relative to the frozen reference?

Per step and source: share of episodes whose output node is a mid-product role
(planner/researcher), share binding a skill, verifier share, mean team size.
Compares early vs late steps, and pi against rho on the same steps.
"""
import json, glob, sys, collections, math
run = sys.argv[1]; early = int(sys.argv[2]) if len(sys.argv) > 2 else 9
MID = {"planner", "researcher"}
per = collections.defaultdict(lambda: collections.defaultdict(list))
for path in sorted(glob.glob(run + "/trajectories/batch-*.jsonl")):
    step = int(path.rsplit("-", 1)[1][:6])
    for line in open(path):
        t = json.loads(line)
        if t["source"] not in ("current", "natural_reference"):
            continue
        acts = [json.loads(d["action_json"]) for d in t["decisions"]]
        roles = {a["node_id"]: a.get("role_id") for a in acts if a["kind"] == "ADD_AGENT"}
        outs = [a["node_id"] for a in acts if a["kind"] == "SET_OUTPUT"]
        row = per[(step, t["source"])]
        row["mid_output"].append(1.0 if outs and roles.get(outs[-1]) in MID else 0.0)
        row["skill"].append(1.0 if any(a.get("skill_id") for a in acts) else 0.0)
        row["verifier"].append(1.0 if "verifier" in roles.values() else 0.0)
        row["team"].append(float(len(roles)))
        row["reward"].append(t["reward"])
steps = sorted({s for s, _ in per})
late = [s for s in steps if s > max(steps) - 10]
first = [s for s in steps if s <= early]
def agg(keys, source, metric):
    vals = [v for s in keys for v in per[(s, source)][metric]]
    n = len(vals); m = sum(vals) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / max(1, n - 1)) / math.sqrt(n)
    return m, sd, n
print(f"run {run.rsplit('/',1)[1]}: steps {min(steps)}-{max(steps)}; early = 1-{early}, late = {min(late)}-{max(late)}")
for metric in ("mid_output", "skill", "verifier", "team", "reward"):
    line = f"  {metric:11s}"
    for source in ("current", "natural_reference"):
        e = agg(first, source, metric); l = agg(late, source, metric)
        line += f" | {source[:4]}: early {e[0]:.3f}±{e[1]:.3f} late {l[0]:.3f}±{l[1]:.3f}"
    print(line)
# pi vs rho on the late steps (same tasks, same steps)
print("  late-step pi - rho:")
for metric in ("mid_output", "skill", "verifier", "team", "reward"):
    c = agg(late, "current", metric); r = agg(late, "natural_reference", metric)
    d = c[0] - r[0]; se = math.sqrt(c[1] ** 2 + r[1] ** 2)
    print(f"    {metric:11s} {d:+.3f} ± {se:.3f}  ({'significant' if abs(d) > 2 * se else 'not significant'})")
