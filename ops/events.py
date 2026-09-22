"""Skill EVENTS: birth/promotion step -> measured causal effect of that skill."""
import json, glob, sys, random, statistics as st
from collections import defaultdict
random.seed(3)
births, promos, status_hist = {}, [], defaultdict(list)
for d in sorted(glob.glob(sys.argv[1])):
    run = d.rsplit("/", 1)[1]
    for b in sorted(glob.glob(d + "/batches/*.json")):
        idx = int(b.rsplit("-", 1)[1].split(".")[0])
        j = json.load(open(b))
        a = j.get("author") or {}
        wins = a.get("windows") or ([a] if a.get("skill_id") else [])
        for w in wins:
            if w.get("status") == "candidate_proposed" and w.get("skill_id"):
                births.setdefault(w["skill_id"], (idx, w.get("family"), run))
        for dec in (j.get("admission_decisions") or []):
            if dec.get("action") in ("promote", "retire"):
                promos.append(dict(step=idx, run=run, action=dec["action"], skill=dec.get("skill_id"),
                                   W=dec.get("wins"), L=dec.get("losses"), T=dec.get("ties"),
                                   p=dec.get("p_positive"), eff=dec.get("effect_size")))
# causal paired effect per skill
tr, ct, meta = {}, {}, {}
for d in sorted(glob.glob(sys.argv[1])):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            pid, src, rw = r.get("pair_id"), r.get("source"), r.get("reward")
            if not pid or rw is None: continue
            if src == "paired_treatment":
                tr[pid] = rw
                if r.get("candidate_id"): meta[pid] = (r["candidate_id"], (r.get("task") or {}).get("task_id"))
            elif src == "paired_control": ct[pid] = rw
bytask = defaultdict(lambda: defaultdict(list))
for pid in set(tr) & set(ct):
    if pid in meta:
        cid, tid = meta[pid]; bytask[cid][tid].append(tr[pid] - ct[pid])
def boot(tm, B=3000):
    ts = list(tm)
    if len(ts) < 3: return None
    obs = st.mean([st.mean(tm[t]) for t in ts]); ms = []
    for _ in range(B):
        s = [tm[ts[random.randrange(len(ts))]] for _ in ts]
        ms.append(st.mean([st.mean(x) for x in s]))
    ms.sort()
    return dict(effect=round(obs, 4), lo=round(ms[int(.025*B)], 4), hi=round(ms[int(.975*B)], 4),
                tasks=len(ts), pairs=sum(len(v) for v in tm.values()))
ev = []
for cid, tm in bytask.items():
    st_ = boot(tm)
    if not st_: continue
    step, fam, run = births.get(cid, (None, None, None))
    if step is None: continue
    ev.append(dict(step=step, run=run, family=fam, skill=cid[:22], **st_))
ev.sort(key=lambda r: (r["run"], r["step"]))
json.dump(dict(events=ev, promotions=promos), open(sys.argv[2], "w"))
print(f"{'run':22s} {'step':>5s} {'family':10s} {'effect':>9s} {'95% CI':>20s} {'pairs/tasks':>12s}")
for e in ev:
    sig = "*" if e["lo"] > 0 or e["hi"] < 0 else " "
    print(f"{e['run'][:22]:22s} {e['step']:5d} {e['family']:10s} {e['effect']:+9.4f}{sig} "
          f"[{e['lo']:+.3f},{e['hi']:+.3f}]".rjust(20) + f"  {e['pairs']:4d}p/{e['tasks']:3d}t")
print("\nPROMOTION / RETIREMENT EVENTS:")
for p in promos: print("  ", p)
