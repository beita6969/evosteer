# HealthBench terminal judge: Luna medium

New scoring condition: `healthbench-luna-medium-api-per-rubric@1`.
Model: `gpt-5.6-luna`, `reasoning_effort=medium`, official OpenAI Chat Completions.
The task actor remains one Qwen3.5-9B; this judge never generates task actions,
consults the actor or supplies rubric answers to its context.

## Scope and invariants

- Training and native IID evaluation use the same pinned simple-evals rubric path.
- Per rubric: JSON response, 8,000 completion-token cap, no temperature/top-p/seed
  override, `store=false`. SDK and transport automatic retries are disabled.
  The existing official semantic-format repair allowance is unchanged.
- An incomplete, timed-out or failed judge response is infrastructure failure,
  not a negative training label. Already committed native rewards remain intact.
- Raw rubric scores may be negative. TTB reward stays `clip(raw,0,1)`;
  binary success stays `raw >= 0.60` and no triggered negative rubric.
- API requests use an independent bounded pool, never the actor's SGLang lease.
  Grader token usage and partial-failure costs remain separate from actor usage.
- Secrets are read from `OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` at runtime,
  outside the repository. They are not copied into configuration, checkpoint,
  task context, telemetry or Git. The client pins the official OpenAI endpoint.

## New runs and complete-checkpoint continuation

The formal training YAML explicitly selects the new judge. Historical configs
that omit `healthbench_judge` retain their original local-Qwen identity.
Do not overwrite a running batch's scorer or silently reinterpret an old run.

To preserve an existing run's actual configuration, including budgets and skills:

```bash
uv run python scripts/configure_healthbench_judge.py \
  /private/run/condition-current.json /private/run/luna-condition.json
```

Use that generated config with `--resume` at a complete checkpoint and
`--allow-healthbench-judge`. The runtime checks that only the judge changed,
persists the new condition boundary, and keeps model/Adam/Z, cursor, posterior,
library and all historical labels. It never reuses old-condition in-flight data
as newly judged evidence. The helper alone does not stop or restart training.

For new native IID scoring, use
`configs/evaluation/healthbench_luna_medium.yaml`; it is also the default when
no explicit HealthBench profile was frozen. The old Qwen profile remains an
explicit historical option, not a fallback for missing API credentials.
Grader profiles and verifier versions are frozen into evaluation controls;
aggregation rejects mixed judge versions. Scores from different judge conditions
are not a like-for-like accuracy comparison.

## Validation identity

An actual OpenAI API transport check returned the requested model and complete
JSON (22 input + 18 output tokens). This is not a HealthBench benchmark result,
does not validate rubric calibration, and generated no training evidence.
No running training process is hot-swapped by this implementation.

Final validation (2026-09-12): CPU-only `make check` on the approved 22049 host
passed: 4,426 tests, 12 explicit GPU/private-tokenizer skips, ruff/mypy and both
wheels. Targeted routing, rubric failure, native-metric and complete-checkpoint
continuation tests passed first. Initial test-fixture/type issues were repaired;
no unrelated GPU numerical replay, independent review agent or additional
file-hash check was run. The local E-drive pytest/mypy attempts timed out on I/O;
the final checks ran on server-local storage instead.
