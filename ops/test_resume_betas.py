"""Prove the resume reversion is real and that the patch fixes it."""
import sys, torch, json, glob
sys.path.insert(0, '/workspace/evosteer/SKILLEV-new-main/src')
from skillev.training.evosteer import EvoTrainer, EvoOptimizerConfig, ADAM_BETA2

# --- what the LIVE v13 checkpoint actually stores -------------------------
R = open('/workspace/evosteer_runs/active_9b_paper_v13.out').read().strip()
ck = sorted(glob.glob(R + '/checkpoints/batch-*'))[-2]
t = torch.load(ck + '/trainable.pt', map_location='cpu', weights_only=True)
groups = t['optimizer']['param_groups']
print('LIVE CHECKPOINT %s' % ck.rsplit('/', 1)[1])
for i, g in enumerate(groups):
    print('  group %d: lr=%s betas=%s weight_decay=%s' % (i, g.get('lr'), g.get('betas'), g.get('weight_decay')))
cfg_live = json.load(open('/workspace/evosteer_9b_paper_v13_config.json'))['application']['optimizer']
print('  config says actor_beta1=%s head_beta1=%s' % (cfg_live.get('actor_beta1'), cfg_live.get('head_beta1')))

# --- reproduce the reversion with real torch ------------------------------
def make(beta1):
    a = [torch.nn.Parameter(torch.randn(4, 4))]
    h = [torch.nn.Parameter(torch.randn(3))]
    return torch.optim.AdamW(
        [{'params': a, 'lr': 5e-6, 'betas': (beta1, ADAM_BETA2)},
         {'params': h, 'lr': 1e-4, 'betas': (0.9, ADAM_BETA2)}], weight_decay=0.01), a, h

old, a, h = make(0.90)                       # the checkpoint was written at 0.90
for p in a + h:
    p.grad = torch.ones_like(p)
old.step(); old.step()                       # two real steps -> moments + step counts
saved = old.state_dict()

new, a2, h2 = make(0.95)                     # this run is configured at 0.95
assert new.param_groups[0]['betas'][0] == 0.95
new.load_state_dict(saved)
reverted = new.param_groups[0]['betas'][0]
print('\nAFTER load_state_dict: actor beta1 = %s   <-- %s' % (
    reverted, 'REVERTED (the bug)' if reverted == 0.90 else 'unchanged'))

steps_before = {i: s['step'].clone() for i, s in enumerate(new.state.values())}
moments_before = [s['exp_avg'].clone() for s in new.state.values()]

# --- apply the patched method, unbound, against this optimizer ------------
class Shim:
    pass
shim = Shim()
shim.optimizer = new
shim.config = EvoOptimizerConfig(actor_learning_rate=5e-6, head_learning_rate=1e-4,
                                 weight_decay=0.01, actor_beta1=0.95, head_beta1=0.9)
EvoTrainer.apply_configured_hyperparameters(shim)

fixed = new.param_groups[0]['betas'][0]
print('AFTER apply_configured_hyperparameters: actor beta1 = %s   <-- %s' % (
    fixed, 'FIXED' if fixed == 0.95 else 'STILL WRONG'))

same_steps = all(torch.equal(s['step'], steps_before[i]) for i, s in enumerate(new.state.values()))
same_moments = all(torch.equal(s['exp_avg'], moments_before[i]) for i, s in enumerate(new.state.values()))
print('moments preserved: %s   step counts preserved: %s' % (same_moments, same_steps))
print('head beta1: %s   actor lr: %s   weight_decay: %s' % (
    new.param_groups[1]['betas'][0], new.param_groups[0]['lr'], new.param_groups[0]['weight_decay']))
assert fixed == 0.95 and same_steps and same_moments
print('\nRESULT: PASS')
