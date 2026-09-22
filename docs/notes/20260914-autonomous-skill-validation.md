# Autonomous skill validation — 2026-09-14

Implementation: `6374d0b`. This is development-only, not seven-domain A0 or TTB.
The earlier 40-episode zero-read collection remains failed.

- Preserved teaching demonstrations, actual body visibility, and attributable
  application annotations; added original action/token supervision accounting.
- Added a predeclared, no-teaching before/after warmup × skills-off/on check.
  Sources exclude previous teaching/warmup, training, quality and IID sources.
- Kept the saved five-update, F-only token-mean warmup unchanged. No new
  grouped weighting, prompt rewriting or on-policy evidence relabeling.
- Held the five-step launcher: demonstration success or falling NLL cannot
  substitute for autonomous application on independent development sources.

## Complete outcome

Six ALFWorld and six MBPP development sources, four arms, seed 0. HumanEval
was excluded because no unused source-disjoint records remained in the loaded
pool. These twelve sources are now development-used, not future unseen tests.

| Initialization / library | Successes | Mean reward | Actual reads |
| --- | ---: | ---: | ---: |
| Original F / off | 6/12 | 0.500000 | 0 |
| Original F / initial library | 7/12 | 0.583333 | 1 |
| Supervised F / off | 5/12 | 0.416667 | 0 |
| Supervised F / initial library | 8/12 | 0.666667 | 0 |

All 48 outcomes were retained. The one original-F read returned a body and
showed partial procedural correspondence, but the task failed. Supervised-F
skills-on had **zero reads and zero successful applications**. The autonomous
check failed; no five-step training was started. Score differences without
body reads are not evidence of skill-body benefit or causal efficacy.

Collection: 10:16–10:48 UTC, 1,927.16 seconds (32.12 minutes), approximately
1.49 episodes/minute. This is not training steps/hour. GPU6 and GPU1 served
the paired arms; the pre-existing GPU5/6/7 services were retained, and a GPU1
service was added after a prelaunch availability check. GPU6 reached 88°C,
with sampled SM clocks as low as 345 MHz and software thermal slowdown in
82.2% of its observations. Hardware remediation needs an administrator;
thermal slowdown does not explain or excuse zero autonomous reads.

Validation: 45 directly related CPU tests; one Linux CPU `make check`,
5,033 passed / 16 skipped, lint/types/wheels passed. No audit agents, manual
hash checks, new reward bonus, read quota, posterior update or evolution.
All per-source material, original actions and reviews remain private.

[Complete scalar run and failed qualification](https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/runs/autonomous-skill12-20260914-v3).

The next candidate needs better transfer from teaching to autonomous use
(e.g. explicitly versioned fading of teaching cues and more independent
application demonstrations). This run does not establish the cause of that
transfer failure, and its sources cannot be recycled as an unseen admission
panel. No additional candidate or training is automatically launched.
