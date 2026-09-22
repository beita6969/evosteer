import glob, os, sys
V = sys.argv[1]
dirs = []
p = f'/workspace/evosteer_runs/active_9b_paper_{V}_prev.out'
if os.path.exists(p): dirs.append(open(p).read().strip())
dirs.append(open(f'/workspace/evosteer_runs/active_9b_paper_{V}.out').read().strip())
rows = []
for R in dirs:
    for d in sorted(glob.glob(R + '/checkpoints/batch-*')):
        n = int(os.path.basename(d).split('-')[1])
        try:
            sj = os.path.getsize(d + '/state.json')
            tp = os.path.getsize(d + '/trainable.pt')
        except OSError:
            continue
        rows.append((n, sj, tp))
rows.sort()
print('%-8s %14s %14s %14s' % ('step', 'state.json MB', 'trainable.pt MB', 'batch json MB'))
bj = {}
for R in dirs:
    for f in glob.glob(R + '/batches/*.json'):
        bj[int(os.path.basename(f).split('-')[1].split('.')[0])] = os.path.getsize(f)
for n, sj, tp in rows[::12]:
    print('%-8d %14.2f %14.2f %14.2f' % (n, sj/1e6, tp/1e6, bj.get(n, 0)/1e6))
if rows:
    print('\nstate.json  first %.2f MB -> last %.2f MB  (x%.1f)' % (
        rows[0][1]/1e6, rows[-1][1]/1e6, rows[-1][1]/max(rows[0][1], 1)))
