"""Is the SERVER slower over time at a FIXED concurrency, or is the TRAINER feeding it less?"""
import re, sys, os, statistics as st
from collections import defaultdict
pat = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d):.*?Decode batch, #running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+)')
with open(sys.argv[1], 'rb') as fh:
    fh.seek(max(0, os.path.getsize(sys.argv[1]) - 300_000_000)); fh.readline()
    data = fh.read().decode('utf-8', 'ignore')
fix = defaultdict(list)     # hour -> throughput at conc in [7,9]
occ = defaultdict(list)     # hour -> all #running-req
for line in data.split('\n'):
    m = pat.match(line)
    if not m: continue
    h, r, t = m.group(1), int(m.group(2)), float(m.group(3))
    occ[h].append(r)
    if 7 <= r <= 9 and t > 0: fix[h].append(t / r)      # per-request tok/s
hs = sorted(occ)
print('%-14s %9s %14s %12s %10s' % ('hour', 'samples', 'per-req t/s', 'n(conc 7-9)', 'mean conc'))
print('  (per-req t/s at FIXED concurrency 7-9 = server health; mean conc = what the trainer delivers)')
for h in hs:
    f = fix.get(h, [])
    if len(occ[h]) < 500: continue
    print('%-14s %9d %14s %12d %10.2f' % (
        h, len(occ[h]), ('%.1f' % st.mean(f)) if len(f) >= 30 else '-', len(f), st.mean(occ[h])))
