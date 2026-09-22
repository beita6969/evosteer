"""Make a resumed run keep ITS OWN optimizer hyperparameters.

torch's Optimizer.load_state_dict restores the checkpoint's parameter-group
hyperparameters next to its moments, so a resume silently reverts actor_beta1
(and the rates/decay) to whatever the checkpoint was written at. Re-assert the
running configuration over the restored groups; moments and step counts stay.
"""
import re, sys

TRAINER = '/workspace/evosteer/SKILLEV-new-main/src/skillev/training/evosteer.py'
APP = '/workspace/evosteer/SKILLEV-new-main/src/skillev/evosteer_application.py'

METHOD = '''
    def apply_configured_hyperparameters(self) -> None:
        """Re-assert this run's AdamW hyperparameters over a restored state dict.

        torch's Optimizer.load_state_dict restores every parameter-group
        hyperparameter from the saved state, so a resumed run would otherwise
        silently keep the rates, decays and first moments its checkpoint was
        written at. Only the hyperparameters move: the moments and the step
        counts stay as saved, which is the point of resuming at all.
        """
        options = (
            {
                "lr": self.config.actor_learning_rate,
                "betas": (self.config.actor_beta1, ADAM_BETA2),
                "weight_decay": self.config.weight_decay,
            },
            {
                "lr": self.config.head_learning_rate,
                "betas": (self.config.head_beta1, ADAM_BETA2),
                "weight_decay": self.config.weight_decay,
            },
        )
        if len(self.optimizer.param_groups) != len(options):
            raise ValueError("restored optimizer has an unexpected parameter group count")
        for group, values in zip(self.optimizer.param_groups, options):
            group.update(values)

'''

s = open(TRAINER).read()
if 'def apply_configured_hyperparameters' in s:
    print('TRAINER: already patched')
else:
    anchor = '    def _export_value_replay(self, optimizer: Any, state_dict: dict[str, Any]) -> dict[str, Any]:'
    assert s.count(anchor) == 1, 'trainer anchor not unique: %d' % s.count(anchor)
    s = s.replace(anchor, METHOD.lstrip('\n') + anchor)
    open(TRAINER, 'w').write(s)
    print('TRAINER: patched')

a = open(APP).read()
if 'apply_configured_hyperparameters' in a:
    print('APP: already patched')
else:
    anchor = '            self.trainer.optimizer.load_state_dict(tensors["optimizer"])'
    assert a.count(anchor) == 1, 'app anchor not unique: %d' % a.count(anchor)
    a = a.replace(anchor, anchor + '\n'
        '            # torch restores the checkpoint\'s own parameter-group\n'
        '            # hyperparameters next to its moments, so this run\'s\n'
        '            # configuration is applied over them; the moments and the\n'
        '            # step counts stay exactly as saved.\n'
        '            self.trainer.apply_configured_hyperparameters()')
    open(APP, 'w').write(a)
    print('APP: patched')
