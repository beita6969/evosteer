import json, glob, sys
V = sys.argv[1]
a = json.load(open(f'/workspace/evosteer_9b_paper_{V}_config.json'))['application']
print('max_validated_per_family =', a.get('max_validated_per_family'),
      '  max_candidates_per_family =', a.get('max_candidates_per_family'),
      '  families =', a.get('task_families'))
R = open(f'/workspace/evosteer_runs/active_9b_paper_{V}.out').read().strip()
j = json.load(open(sorted(glob.glob(R + '/batches/*.json'))[-1]))
print('batch', j['batch_index'], 'library:')
for sid, st in sorted((j.get('skill_status') or {}).items()):
    print('   %-30s %s' % (sid, st))
