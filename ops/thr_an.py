"""Measured sglang decode throughput as a function of #running-req (the headroom curve)."""
import re, sys, os, statistics as st
from collections import defaultdict
pat = re.compile(r'Decode batch, #running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+)')
path = sys.argv[1]
win = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000_000
with open(path, 'rb') as fh:
    fh.seek(max(0, os.path.getsize(path) - win)); fh.readline()
    data = fh.read().decode('utf-8', 'ignore')
b = defaultdict(list)
for m in pat.finditer(data):
    r, t = int(m.group(1)), float(m.group(2))
    if r >= 1 and t > 0: b[r].append(t)
print('%-10s %9s %12s %12s   %s' % ('#running', 'samples', 'mean tok/s', 'per-req t/s', 'speedup vs conc=1'))
base = st.mean(b[1]) if b.get(1) else None
tot = sum(len(v) for v in b.values())
for r in sorted(b):
    v = b[r]
    if len(v) < 40: continue
    m = st.mean(v)
    print('%-10d %9d %12.0f %12.1f   %.2fx' % (r, len(v), m, m / r, (m / base) if base else float('nan')))
print('\ntotal decode-batch log samples n=%d' % tot)
# occupancy-weighted: where does the decode time actually go?
wt = {r: len(v) for r, v in b.items()}
tw = sum(wt.values())
cum = 0
print('\nshare of decode-batch samples spent at each concurrency band:')
for lo, hi in ((1,2),(2,5),(5,9),(9,17),(17,33),(33,65)):
    n = sum(c for r, c in wt.items() if lo <= r < hi)
    cum += n
    print('   %2d-%-2d : %6.1f%%' % (lo, hi-1, 100*n/tw))
