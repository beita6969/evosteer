"""Warmup-vs-skills summary for one run: rewards, paired evidence, structure, per family."""
import json, glob, sys, collections, math
run = sys.argv[1]; split = int(sys.argv[2]) if len(sys.argv) > 2 else 9   # last warmup step
rows = [json.loads(l) for l in open(run + "/metrics.jsonl")]
def block(rs, key):
    vals = [r[key]["mean"] if isinstance(r[key], dict) else r[key] for r in rs]
    n = len(vals); m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / max(1, n - 1))
    return m, sd / math.sqrt(n), n
warm = [r for r in rows if r["batch_index"] <= split]
late = [r for r in rows if r["batch_index"] > split]
print(f"== {run.rsplit('/',1)[1]}  steps={len(rows)}  warmup=1-{split}")
for key, label in (("current_reward", "pi_theta reward"), ("natural_reference_reward", "rho reward"),
                   ("anchor_tb_loss", "AnchorTB loss"), ("value_loss", "value MSE (pre-fit)"),
                   ("value_family_mean_loss", "value family baseline")):
    w = block(warm, key); l = block(late, key) if late else (float("nan"),) * 3
    print(f"  {label:24s} warmup {w[0]:.3f}±{w[1]:.3f}   with skills {l[0]:.3f}±{l[1]:.3f}   delta {l[0]-w[0]:+.3f}")
d = [r["current_reward"]["mean"] - r["natural_reference_reward"]["mean"] for r in rows]
dl = [r["current_reward"]["mean"] - r["natural_reference_reward"]["mean"] for r in late]
se = lambda v: (sum(v) / len(v), math.sqrt(sum((x - sum(v)/len(v)) ** 2 for x in v) / max(1, len(v) - 1)) / math.sqrt(len(v)))
m, s = se(d); print(f"  pi-rho (same tasks)      all steps {m:+.4f}±{s:.4f}", end="")
if dl:
    m2, s2 = se(dl); print(f"   with skills {m2:+.4f}±{s2:.4f}")
else: print()
fam = collections.defaultdict(lambda: ([], []))
for r in rows:
    for f, v in r["reward_by_family"].items():
        (fam[f][0] if r["batch_index"] <= split else fam[f][1]).append(v["current"]["mean"])
print("  per family (pi):")
for f, (w, l) in sorted(fam.items()):
    mw = sum(w)/len(w) if w else float("nan"); ml = sum(l)/len(l) if l else float("nan")
    print(f"    {f:10s} warmup {mw:.3f}  with skills {ml:.3f}  delta {ml-mw:+.3f}")
paired = {"treat": {}, "control": {}}
bind = collections.defaultdict(lambda: [0, 0])
for path in sorted(glob.glob(run + "/trajectories/batch-*.jsonl")):
    step = int(path.rsplit("-", 1)[1][:6])
    for line in open(path):
        t = json.loads(line); acts = [json.loads(x["action_json"]) for x in t["decisions"]]
        if t["source"] == "current":
            bind[step][0] += 1; bind[step][1] += any(a.get("skill_id") for a in acts)
        elif t["source"] in ("paired_treatment", "paired_control"):
            paired["treat" if t["source"] == "paired_treatment" else "control"][(t["task"]["task_id"], step)] = t["reward"]
common = sorted(set(paired["treat"]) & set(paired["control"]))
diffs = [paired["treat"][k] - paired["control"][k] for k in common]
if diffs:
    w = sum(1 for x in diffs if x > 0); l = sum(1 for x in diffs if x < 0)
    p = sum(math.comb(w + l, i) for i in range(w, w + l + 1)) / 2 ** (w + l) if w + l else 1.0
    print(f"  paired A/B (skill vs none, both continued by rho): n={len(diffs)} mean {sum(diffs)/len(diffs):+.4f} "
          f"W={w} L={l} T={len(diffs)-w-l} one-sided sign p={p:.4f}")
late_bind = [bind[s][1] / bind[s][0] for s in sorted(bind) if s > split and bind[s][0]]
if late_bind:
    print(f"  skill binding rate with skills: mean {sum(late_bind)/len(late_bind):.2f} range {min(late_bind):.2f}-{max(late_bind):.2f}")

# Same-task comparison: removes the task-difficulty confound of a rotating batch.
per_task = collections.defaultdict(lambda: ([], []))
for path in sorted(glob.glob(run + "/trajectories/batch-*.jsonl")):
    step = int(path.rsplit("-", 1)[1][:6])
    for line in open(path):
        t = json.loads(line)
        if t["source"] not in ("current", "natural_reference"):
            continue
        key = (t["source"], t["task"]["task_id"])
        (per_task[key][0] if step <= split else per_task[key][1]).append(t["reward"])
for source in ("current", "natural_reference"):
    pairs = [(sum(w)/len(w), sum(l)/len(l)) for (s, _), (w, l) in per_task.items() if s == source and w and l]
    if pairs:
        d = [l - w for w, l in pairs]
        m = sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1)) / math.sqrt(len(d))
        better = sum(1 for x in d if x > 0); worse = sum(1 for x in d if x < 0)
        print(f"  same-task before/after skills [{source:17s}]: n={len(d)} tasks  mean {m:+.4f}±{sd:.4f}  "
              f"better {better} worse {worse} unchanged {len(d)-better-worse}")
