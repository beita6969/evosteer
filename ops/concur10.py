"""Executor concurrency in 10-minute buckets (to see an intervention's effect)."""
import re, sys, statistics as st
from collections import defaultdict
pat = re.compile(r'^\[(\d{4}-\d\d-\d\d) (\d\d):(\d\d):\d\d\].*?#running-req: (\d+).*?token usage: ([\d.]+)')
buck = defaultdict(list)
import os
path = sys.argv[1]
with open(path, 'rb') as fh:                      # only the tail; these logs are huge
    fh.seek(max(0, os.path.getsize(path) - 120_000_000)); fh.readline()
    for line in fh.read().decode('utf-8', 'ignore').split('\n'):
        if '#running-req' not in line: continue
        m = pat.match(line)
        if not m: continue
        key = '%s %s:%02d' % (m.group(1)[5:], m.group(2), int(m.group(3)) // 10 * 10)
        buck[key].append(int(m.group(4)))
ks = sorted(buck)
print('%-14s %9s %10s %8s' % ('time (10min)', 'samples', 'mean run', 'max'))
for k in ks[-18:]:
    v = buck[k]
    print('%-14s %9d %10.2f %8d' % (k, len(v), st.mean(v), max(v)))
