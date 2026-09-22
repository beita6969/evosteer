import glob, os, sys, statistics as st, datetime
V = sys.argv[1]
R = open(f'/workspace/evosteer_runs/active_9b_paper_{V}.out').read().strip()
b = sorted(glob.glob(R + '/batches/*.json'))
mt = [os.path.getmtime(x) for x in b]
n = [int(os.path.basename(x).split('-')[1].split('.')[0]) for x in b]
d = [(n[i], mt[i] - mt[i-1]) for i in range(1, len(mt))]
print(f'{V}: post-resume batches {n[0]}..{n[-1]}')
for lab, sel in (('all post-resume', d), ('last 20', d[-20:]), ('last 10', d[-10:]), ('last 5', d[-5:])):
    if sel: print('  %-16s %6.0f s/step  (n=%d)' % (lab, st.mean(s for _, s in sel), len(sel)))
pace = st.mean(s for _, s in d[-10:]) if len(d) >= 10 else st.mean(s for _, s in d)
rem = 250 - n[-1]
eta = datetime.datetime.now() + datetime.timedelta(seconds=rem * pace)
print(f'  at step {n[-1]}, {rem} left at {pace:.0f}s -> {rem*pace/3600:.1f} h -> ETA {eta:%Y-%m-%d %H:%M} (pod local)')
print(f'  now {datetime.datetime.now():%Y-%m-%d %H:%M}')
