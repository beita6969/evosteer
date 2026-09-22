import glob, os, datetime, sys
R = sys.argv[1]
b = sorted(glob.glob(R + '/batches/*.json'))
mt = [(int(os.path.basename(x).split('-')[1].split('.')[0]), os.path.getmtime(x)) for x in b]
for i in range(max(1, len(mt) - 9), len(mt)):
    print('  step %3d: %6.0f s  (done %s)' % (
        mt[i][0], mt[i][1] - mt[i-1][1],
        datetime.datetime.fromtimestamp(mt[i][1]).strftime('%m-%d %H:%M')))
