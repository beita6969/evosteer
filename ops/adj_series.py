import json, glob, sys, statistics as st
run_glob, pilot_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
pil = {}
for r in json.load(open(pilot_path))["results"]:
    pil.setdefault(r["task_id"], []).append(r["reward"])
pil = {k: sum(v)/len(v) for k, v in pil.items()}
rows = {}
for d in sorted(glob.glob(run_glob)):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        idx = int(f.rsplit("-", 1)[1].split(".")[0])
        for line in open(f):
            r = json.loads(line)
            t = r.get("task")
            tid = t.get("task_id") if isinstance(t, dict) else t
            if isinstance(tid, dict): tid = tid.get("task_id")
            rows.setdefault(idx, []).append((r.get("source"), tid, r.get("reward")))
series = []
for idx in sorted(rows):
    cur, ref, adj, base = {}, {}, [], []
    for src, tid, rw in rows[idx]:
        if rw is None: continue
        if src == "current": cur.setdefault(tid, []).append(rw)
        elif src == "natural_reference": ref.setdefault(tid, []).append(rw)
    for tid, v in cur.items():
        m = sum(v)/len(v)
        p = pil.get(tid)
        if p is not None:
            adj.append(m - p); base.append(p)
    pi = st.mean([sum(v)/len(v) for v in cur.values()]) if cur else None
    rh = st.mean([sum(v)/len(v) for v in ref.values()]) if ref else None
    common = set(cur) & set(ref)
    pr = st.mean([sum(cur[t])/len(cur[t]) - sum(ref[t])/len(ref[t]) for t in common]) if common else None
    series.append(dict(step=idx, n=len(cur), pi=pi, rho=rh, pilot=st.mean(base) if base else None,
                       adj=st.mean(adj) if adj else None,
                       adj_se=(st.stdev(adj)/len(adj)**0.5) if len(adj) > 1 else None,
                       pi_minus_rho_paired=pr, n_pair=len(common)))
json.dump(series, open(out, "w"))
def win(a, b, k):
    s = [x[k] for x in series if a <= x["step"] <= b and x[k] is not None]
    return (len(s), round(st.mean(s), 4), round(st.stdev(s)/len(s)**0.5, 4) if len(s) > 1 else 0) if s else None
for k in ("pi", "rho", "pilot", "adj", "pi_minus_rho_paired"):
    print(k.ljust(20), [win(a, b, k) for a, b in ((1, 9), (10, 30), (31, 60), (61, 90), (91, 130))])
