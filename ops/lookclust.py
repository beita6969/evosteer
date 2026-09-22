"""Per-comparison alpha-spending LOOK re-scored with TASK-CLUSTERED statistics.

usage: lookclust.py <run_dir> <look_step> [<look_step> ...]
For each comparison evaluated at a look, recompute the paired effect with a
percentile cluster bootstrap over task_id (one cluster = one task), report the
clustered 95% CI and one-sided p, the one-vote-per-task win/loss tally, and the
naive per-pair p for contrast. Marks the engine's own action at that look.
"""
import json, glob, os, sys, random, statistics as st
from collections import defaultdict

random.seed(20260921)
# argv[1] may be a comma-separated list of run dirs (the pre-150 and post-150 halves)
RUNS = [x.rstrip('/') for x in sys.argv[1].split(',') if x.strip()]
LOOKS = [int(x) for x in sys.argv[2:]]

# --- engine decisions per look --------------------------------------------
actions = {}            # (look, comparison_id) -> action
for p in sorted(g for R in RUNS for g in glob.glob(R + '/batches/*.json')):
    n = int(os.path.basename(p).split('-')[1].split('.')[0])
    j = json.load(open(p))
    for d in (j.get('admission_decisions') or []):
        actions[(n, d.get('comparison_id'))] = d.get('action')

# --- paired outcomes -------------------------------------------------------
# pair_id = "<batch>/<family>/<task>/<comparison_id>[/...]"
tr, ct, meta = {}, {}, {}
for f in sorted(g for R in RUNS for g in glob.glob(R + '/trajectories/*.jsonl')):
    step = int(f.rsplit('-', 1)[1].split('.')[0])
    for line in open(f):
        r = json.loads(line)
        src, pid, rw = r.get('source'), r.get('pair_id'), r.get('reward')
        if not pid or rw is None or not str(src).startswith('paired_'):
            continue
        parts = pid.split('/')
        cid = next((x for x in parts if x.startswith('sha256:')), None)
        if cid is None:
            continue
        t = r.get('task') or {}
        meta[pid] = (cid, t.get('family'), t.get('task_id'), step)
        (tr if src == 'paired_treatment' else ct)[pid] = rw

def boot(task_map, B=20000):
    tasks = list(task_map)
    if len(tasks) < 2:
        return None
    obs = st.mean([st.mean(task_map[t]) for t in tasks])
    le, means = 0, []
    for _ in range(B):
        s = [task_map[tasks[random.randrange(len(tasks))]] for _ in tasks]
        m = st.mean([st.mean(x) for x in s])
        means.append(m)
        if m <= 0:
            le += 1
    means.sort()
    return obs, means[int(.025 * B)], means[int(.975 * B)], (le + 1) / (B + 1)

print('runs:', ', '.join(os.path.basename(R) for R in RUNS))
for look in LOOKS:
    per_cid = defaultdict(lambda: defaultdict(list))
    for pid in set(tr) & set(ct):
        cid, fam, tid, step = meta[pid]
        if step <= look:
            per_cid[cid][tid].append(tr[pid] - ct[pid])
    cids = sorted({c for (n, c) in actions if n == look})
    if not cids:
        print('\n=== look @ step %d : no decisions recorded ===' % look)
        continue
    print('\n=== look @ step %d  (pairs accumulated through step %d) ===' % (look, look))
    print('%-14s %-8s %6s %6s %10s %24s %9s %9s %10s' % (
        'comparison', 'engine', 'tasks', 'pairs', 'effect', 'clustered 95% CI',
        'p_clust', 'p_naive', '1vote W/L'))
    for cid in cids:
        tm = per_cid.get(cid, {})
        npairs = sum(len(v) for v in tm.values())
        b = boot(tm)
        act = actions[(look, cid)]
        short = cid.replace('sha256:', '')[:12]
        if b is None:
            print('%-14s %-8s %6d %6d   (too few task clusters)' % (short, act, len(tm), npairs))
            continue
        obs, lo, hi, p = b
        flat = [x for v in tm.values() for x in v]
        if len(flat) > 1 and st.stdev(flat) > 0:
            se = st.stdev(flat) / len(flat) ** 0.5
            # one-sided normal p for the naive per-pair test
            import math
            pn = 0.5 * math.erfc((st.mean(flat) / se) / math.sqrt(2))
        else:
            pn = float('nan')
        votes = [st.mean(v) for v in tm.values()]
        W = sum(1 for v in votes if v > 0)
        L = sum(1 for v in votes if v < 0)
        print('%-14s %-8s %6d %6d %+10.4f   [%+.4f,%+.4f] %9.4f %9.4f %5d/%-4d' % (
            short, act, len(tm), npairs, obs, lo, hi, p, pn, W, L))
