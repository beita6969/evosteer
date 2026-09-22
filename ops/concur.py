"""How many requests is the executor actually running, over time?"""
import re, sys, statistics as st
from collections import defaultdict
pat = re.compile(r'^\[(\d{4}-\d\d-\d\d) (\d\d):\d\d:\d\d\].*?#running-req: (\d+).*?token usage: ([\d.]+).*?#queue-req: (\d+)')
hour = defaultdict(lambda: ([], [], []))
n = 0
for line in open(sys.argv[1], errors='ignore'):
    if '#running-req' not in line: continue
    m = pat.match(line)
    if not m: continue
    n += 1
    d, h, run, use, q = m.group(1), m.group(2), int(m.group(3)), float(m.group(4)), int(m.group(5))
    a, b, c = hour[(d, h)]
    a.append(run); b.append(use); c.append(q)
print('parsed %d scheduler lines' % n)
print('%-16s %9s %9s %9s %9s' % ('date hour', 'samples', 'mean run', 'max run', 'tok usage'))
ks = sorted(hour)
for k in ks[::max(1, len(ks)//24)]:
    a, b, c = hour[k]
    print('%-16s %9d %9.2f %9d %9.3f' % ('%s %s:00' % k, len(a), st.mean(a), max(a), st.mean(b)))
if ks:
    a0 = hour[ks[0]][0]; a1 = hour[ks[-1]][0]
    print('\nFIRST hour mean running-req %.2f  ->  LAST hour %.2f' % (st.mean(a0), st.mean(a1)))
