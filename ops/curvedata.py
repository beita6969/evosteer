"""Export the per-step curve series + skill events, merging the pre-150 and post-150 halves."""
import json, os, glob, sys
V = sys.argv[1]
chain = f'/workspace/evosteer_runs/chain_{V}.txt'
dirs = [l.strip() for l in open(chain)] if os.path.exists(chain) else []
cur = open(f'/workspace/evosteer_runs/active_9b_paper_{V}.out').read().strip()
if cur not in dirs: dirs.append(cur)
dirs = [d for d in dirs if d and os.path.isdir(d)]
out = {'run': [os.path.basename(d) for d in dirs], 'version': V, 'steps': [], 'events': []}
seen_steps, seen_ev = set(), set()
for R in dirs:
    for l in open(R + '/metrics.jsonl'):
        m = json.loads(l)
        if m['batch_index'] in seen_steps: continue
        seen_steps.add(m['batch_index'])
        out['steps'].append({
            'i': m['batch_index'], 'reward': m.get('mean_reward'),
            'ref': m.get('natural_reference_reward'), 'cur': m.get('current_reward'), 'loss': m.get('anchor_tb_loss'),
            'vloss': m.get('value_fit_loss'), 'gn': m.get('gradient_norm'),
            'lr': m.get('policy_abs_log_ratio'), 'kl': m.get('policy_kl_current'),
            'alpha': m.get('alpha_spent'), 'by_family': m.get('reward_by_family')})
    for p in sorted(glob.glob(R + '/batches/*.json')):
        j = json.load(open(p)); n = j['batch_index']
        if n in seen_ev: continue
        seen_ev.add(n)
        st = j.get('skill_status') or {}
        out['events'].append({
            'i': n,
            'validated': sum(1 for v in st.values() if v == 'validated'),
            'candidates': sum(1 for v in st.values() if v == 'candidate'),
            'author': (j.get('author') or {}).get('status'),
            'decisions': [(d.get('comparison_id'), d.get('action'))
                          for d in (j.get('admission_decisions') or [])],
            'pairs': j.get('pair_count')})
out['steps'].sort(key=lambda r: r['i']); out['events'].sort(key=lambda r: r['i'])
json.dump(out, open(f'/workspace/evosteer/curvedata_{V}.json', 'w'))
promos = [(e['i'], [c for c, a in e['decisions'] if a == 'promote']) for e in out['events'] if e['decisions']]
print(V, 'dirs', len(dirs), 'steps', len(out['steps']), 'last', out['steps'][-1]['i'])
for i, ps in promos:
    print('   look @%d: %d decisions, promoted %s' % (i, len([1 for e in out['events'] if e['i']==i for _ in e['decisions']]), [x[7:15] for x in ps] or 'none'))
