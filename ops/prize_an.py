"""Decode-time-weighted cost of low concurrency, and the counterfactual at a higher one."""
import re, sys, os, statistics as st
from collections import defaultdict
pat = re.compile(r'Decode batch, #running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+)')
with open(sys.argv[1], 'rb') as fh:
    fh.seek(max(0, os.path.getsize(sys.argv[1]) - 300_000_000)); fh.readline()
    data = fh.read().decode('utf-8', 'ignore')
b = defaultdict(list)
for m in pat.finditer(data):
    r, t = int(m.group(1)), float(m.group(2))
    if r >= 1 and t > 0: b[r].append(t)
thr = {r: st.mean(v) for r, v in b.items() if len(v) >= 40}
ITER = 40                                  # sglang logs one line per 40 decode iterations
tot_tok = tot_time = 0.0
for r, v in b.items():
    if r not in thr: continue
    n = len(v)
    tot_tok += n * ITER * r
    tot_time += n * ITER * r / thr[r]
print(f'MEASURED over the log window (n={sum(len(v) for v in b.values())} decode-batch samples):')
print(f'  decode tokens            : {tot_tok/1e6:10.2f} M')
print(f'  decode wall time         : {tot_time/3600:10.2f} h')
print(f'  achieved mean throughput : {tot_tok/tot_time:10.0f} tok/s')
for target in (16, 24, 32):
    t2 = sum(len(v)*ITER*r/thr[min(max(r, target), max(thr))] for r, v in b.items() if r in thr)
    # counterfactual: same tokens, but every batch served at >= `target` concurrency
    tt = tot_tok / thr[target]
    print(f'  IF all decode ran at conc={target:2d} ({thr[target]:.0f} tok/s): '
          f'{tt/3600:6.2f} h  -> {tot_time/tt:4.2f}x less decode wall time')
print()
print('share of decode WALL TIME (not samples) spent at each concurrency:')
acc = defaultdict(float)
for r, v in b.items():
    if r in thr: acc[r] = len(v)*ITER*r/thr[r]
for lo, hi in ((1,2),(2,5),(5,9),(9,17),(17,33),(33,65)):
    s = sum(t for r, t in acc.items() if lo <= r < hi)
    print('   conc %2d-%-2d : %6.1f%% of decode wall time' % (lo, hi-1, 100*s/tot_time))
