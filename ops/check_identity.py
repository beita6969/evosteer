"""Does the edited config still hash to the identity the live checkpoints carry?"""
import json, glob, sys
from pathlib import Path
sys.path.insert(0, '/workspace/evosteer/SKILLEV-new-main/src')
from skillev.evosteer_cli import _configuration

for v in ('v13', 'v14'):
    cfgp = f'/workspace/evosteer_9b_paper_{v}_config.json'
    try:
        values, app = _configuration(Path(cfgp))
    except Exception as e:
        print(f'{v}: CONFIG FAILED TO PARSE -> {e}')
        continue
    try:
        R = open(f'/workspace/evosteer_runs/active_9b_paper_{v}.out').read().strip()
    except FileNotFoundError:
        print(f'{v}: no active run pointer here'); continue
    cks = sorted(glob.glob(R + '/checkpoints/batch-*'))
    if not cks:
        print(f'{v}: no checkpoints under {R}'); continue
    st = json.load(open(cks[-2] + '/state.json'))
    ok = st['config'] == app.identity
    print(f'{v}: steps={values["steps"]}  checkpoint={cks[-2].rsplit("/",1)[1]}')
    print(f'    checkpoint config id = {st["config"]}')
    print(f'    edited   config id = {app.identity}')
    print(f'    MATCH: {ok}   -> resume {"WILL be accepted" if ok else "WILL BE REJECTED"}')
