"""Skill effect matrix: run x family, task-clustered, plus phase contrast per run."""
import json, glob, sys, random, statistics as st
from collections import defaultdict
random.seed(11)
RUNMAP = [("v10", "9b_paper_v10_*"), ("v10d", "9b_paper_v10d_*"), ("v11", "9b_paper_v11_*"),
          ("v12b", "9b_paper_v12b_*"), ("v13", "9b_paper_v13_*"), ("v14", "9b_paper_v14_*")]
FAMS = ["aime_2026", "mbpp_plus", "medqa", "hotpotqa", "nq_open"]

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

out = {"skill_matrix": {}, "phase": {}}
for name, pat in RUNMAP:
    tr, ct, meta = {}, {}, {}
    cur, ref = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    warm_r, post_r = defaultdict(list), defaultdict(list)
    for d in sorted(glob.glob("/workspace/evosteer_runs/" + pat)):
        for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
            idx = int(f.rsplit("-", 1)[1].split(".")[0])
            for line in open(f):
                r = json.loads(line)
                t = r.get("task") or {}
                fam, tid, rw, src = t.get("family"), t.get("task_id"), r.get("reward"), r.get("source")
                if rw is None or not fam: continue
                pid = r.get("pair_id")
                if pid and src == "paired_treatment": tr[pid] = rw; meta[pid] = (fam, tid)
                elif pid and src == "paired_control": ct[pid] = rw
                if src == "natural_reference":
                    (warm_r if idx <= 9 else post_r)[tid].append(rw)
    fm = defaultdict(lambda: defaultdict(list))
    for pid in set(tr) & set(ct):
        fam, tid = meta[pid]
        fm[fam][tid].append(tr[pid] - ct[pid])
    out["skill_matrix"][name] = {f: boot(fm[f]) for f in FAMS if fm[f]}
    common = set(warm_r) & set(post_r)
    if len(common) >= 5:
        d = {t: [st.mean(post_r[t]) - st.mean(warm_r[t])] for t in common}
        out["phase"][name] = boot(d)
json.dump(out, open(sys.argv[1], "w"))
print(f"{'run':6s} " + " ".join(f"{f[:9]:>16s}" for f in FAMS) + f" {'rho phase jump':>18s}")
for name, _ in RUNMAP:
    row = out["skill_matrix"].get(name, {})
    cells = []
    for f in FAMS:
        v = row.get(f)
        cells.append("       —        " if not v else
                     f"{v['effect']:+.3f}{'*' if v['lo']>0 or v['hi']<0 else ' '}({v['tasks']:2d}t)".rjust(16))
    p = out["phase"].get(name)
    ps = "     —    " if not p else f"{p['effect']:+.3f}{'*' if p['lo']>0 or p['hi']<0 else ' '}({p['tasks']}t)".rjust(18)
    print(f"{name:6s} " + " ".join(cells) + " " + ps)
