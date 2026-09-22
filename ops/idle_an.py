"""How much wall time is the sglang server completely idle? (gaps between scheduler log lines)"""
import re, sys, os, datetime, statistics as st
pat = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*?#running-req: (\d+)')
path = sys.argv[1]
window = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000_000
with open(path, 'rb') as fh:
    fh.seek(max(0, os.path.getsize(path) - window)); fh.readline()
    lines = fh.read().decode('utf-8', 'ignore').split('\n')
ev = []
for line in lines:
    m = pat.match(line)
    if m:
        t = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        ev.append((t, int(m.group(2))))
if len(ev) < 100:
    print('too few events', len(ev)); sys.exit()
t0, t1 = ev[0][0], ev[-1][0]
span = (t1 - t0).total_seconds()
# second-resolution occupancy: for each wall second, max #running-req seen
occ = {}
for t, r in ev:
    s = int(t.timestamp())
    occ[s] = max(occ.get(s, 0), r)
total_secs = int(t1.timestamp()) - int(t0.timestamp()) + 1
busy_secs = len(occ)
idle_secs = total_secs - busy_secs
vals = sorted(occ.values())
print(f'log window: {t0} .. {t1}   span {span/3600:.2f} h  ({total_secs} s)')
print(f'seconds with ANY scheduler activity : {busy_secs:7d}  ({100*busy_secs/total_secs:5.1f}%)')
print(f'seconds with NO log line at all     : {idle_secs:7d}  ({100*idle_secs/total_secs:5.1f}%)  <- server had nothing to do')
print()
print('#running-req over ACTIVE seconds only (n=%d):' % len(vals))
print('   mean %.2f  p50 %d  p90 %d  p99 %d  max %d' % (
    st.mean(vals), vals[len(vals)//2], vals[int(.9*len(vals))], vals[int(.99*len(vals))], vals[-1]))
print()
print('  UNCONDITIONAL mean concurrency over the whole span (idle seconds counted as 0):')
print('   %.2f  against configured EVOSTEER_ROLLOUT_CONCURRENCY=64 and --max-running-requests 64'
      % (sum(vals) / total_secs))
# distribution of idle gap lengths
gaps = []
ss = sorted(occ)
for i in range(1, len(ss)):
    d = ss[i] - ss[i-1] - 1
    if d > 0: gaps.append(d)
if gaps:
    gaps.sort()
    print()
    print('idle gaps: n=%d  total=%ds  mean=%.1fs  p50=%ds  p90=%ds  max=%ds'
          % (len(gaps), sum(gaps), st.mean(gaps), gaps[len(gaps)//2], gaps[int(.9*len(gaps))], gaps[-1]))
