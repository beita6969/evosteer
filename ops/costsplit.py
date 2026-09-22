"""Exact prefill/decode split, orchestrator vs executor, per episode and per step."""
import json, glob, sys
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(float)); n = defaultdict(int)
steps = set()
for f in sorted(glob.glob(sys.argv[1] + "/trajectories/*.jsonl")):
    idx = int(f.rsplit("-", 1)[1].split(".")[0])
    if idx < 10: continue
    steps.add(idx)
    for line in open(f):
        r = json.loads(line)
        src = r.get("source")
        u = r.get("usage_json") or {}
        if isinstance(u, str):
            try: u = json.loads(u)
            except Exception: u = {}
        tin, tout = u.get("input_tokens"), u.get("output_tokens")
        if tin is None: continue
        ts = r.get("terminal_state_json") or {}
        if isinstance(ts, str):
            try: ts = json.loads(ts)
            except Exception: ts = {}
        nu = ts.get("node_usage") or {}
        ein = eout = ecalls = 0
        if isinstance(nu, dict):
            ein = nu.get("input_tokens", 0) or 0; eout = nu.get("output_tokens", 0) or 0
            ecalls = nu.get("model_calls", 0) or 0
        a = agg[src]
        a["in"] += tin; a["out"] += tout or 0
        a["exec_in"] += ein; a["exec_out"] += eout; a["exec_calls"] += ecalls
        a["calls"] += u.get("model_calls", 0) or 0
        a["ms"] += u.get("wall_time_milliseconds", 0) or 0
        n[src] += 1
print(f"steps 10..{max(steps)}  ({len(steps)} steps)")
print(f"{'source':20s} {'n':>6s} {'in/ep':>9s} {'out/ep':>8s} {'exec_in':>9s} {'exec_out':>9s} {'orch_in':>9s} {'orch_out':>8s} {'calls':>6s} {'sec/ep':>7s}")
tot = defaultdict(float)
for s, a in sorted(agg.items()):
    c = n[s]
    print(f"{s:20s} {c:6d} {a['in']/c:9.0f} {a['out']/c:8.0f} {a['exec_in']/c:9.0f} {a['exec_out']/c:9.0f} "
          f"{(a['in']-a['exec_in'])/c:9.0f} {(a['out']-a['exec_out'])/c:8.0f} {a['calls']/c:6.2f} {a['ms']/c/1000:7.1f}")
    for k, v in a.items(): tot[k] += v
N = sum(n.values())
print(f"{'ALL':20s} {N:6d} {tot['in']/N:9.0f} {tot['out']/N:8.0f} {tot['exec_in']/N:9.0f} {tot['exec_out']/N:9.0f} "
      f"{(tot['in']-tot['exec_in'])/N:9.0f} {(tot['out']-tot['exec_out'])/N:8.0f} {tot['calls']/N:6.2f} {tot['ms']/N/1000:7.1f}")
ns = len(steps)
print()
print(f"PER TRAINING STEP: prefill {tot['in']/ns/1e6:.3f}M  decode {tot['out']/ns/1e6:.3f}M  "
      f"ratio decode/total {tot['out']/(tot['in']+tot['out'])*100:.1f}%")
print(f"  executor share of decode: {tot['exec_out']/tot['out']*100:.1f}%   orchestrator decode: {(tot['out']-tot['exec_out'])/ns/1e3:.1f}k/step")
