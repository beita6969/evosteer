import json
from pathlib import Path
from skillev.evosteer_cli import _application_config, _model_config
for v in ("v13", "v14"):
    raw = json.load(open(f"/workspace/evosteer_9b_paper_{v}_config.json"))
    app = _application_config(raw["application"])
    _model_config(raw["model"], Path("/workspace"))
    o = app.optimizer
    print(f"{v}: OK  batch={raw['batch_size']} steps={raw['steps']}")
    print(f"     validation_interval={app.validation_interval} cand/family={app.max_candidates_per_family} repeated_pairs={app.allow_repeated_pair_tasks}")
    print(f"     actor_beta1={o.actor_beta1} head_beta1={o.head_beta1} beta={o.beta} lr={o.actor_learning_rate} clip={o.gradient_clip}")
    print(f"     rollouts cur/ref={app.current_rollouts}/{app.reference_rollouts}  targets={raw['model']['target_modules']}")
    caps = {r['role_id']: r['model_maximum']['output_tokens'] for r in raw['application']['roles']}
    print(f"     output caps={caps}")
    print(f"     identity={app.identity[:28]}")
