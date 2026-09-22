# Owner-authorized training after failed autonomous cold start

On 2026-09-14 the owner explicitly requested continued training despite the
failed cold-start check, and approved GPU1 serving plus GPU7/GPU6 gradients
on endpoint 22049. This does **not** turn that check into a pass or claim
architecture-matched seven-domain IID admission.

- Implementation `57366b6` adds an explicit unqualified run condition. Ordinary
  fresh-run IID requirements and the separate five-step hold remain unchanged.
  Same-run continuation must retain its recorded condition and source exclusions.
- Run: `bayesian250-warmup-unqualified-20260914-v1`; upper bound 250 complete
  steps, B28, seed 0, 249 search steps plus one closure, checkpoint every 10
  steps and cooperative save-on-stop. It is not an automatic five-step stop.
- Initial forward adapter: the saved five-update supervised warmup. Backward,
  Z, optimizer, posterior, evolution and task cursor start fresh. No evaluation
  trajectory is imported into training; old runs and failed results are retained.
- Seven domains, four independent rollouts per source occurrence. The unchanged
  training lanes contain 1,472 distinct sources across 1,750 question occurrences;
  repeated sources are not reported as new independent questions.
- Existing candidate budgets: static 8 / ALFWorld 25 turns; thinking on only for
  AIME and HealthBench; domain reasoning budgets and HealthBench action cap stay
  unchanged. Autonomous catalog reads remain optional; no teaching prompt, read
  quota, call-count reward or loss change is introduced.
- Actual training loop entered at **11:15:28 UTC**. Two real gradient workers
  reported B28 capacity and global canonical contribution merging. GPU5 and
  unrelated processes were untouched. GPU6/GPU7 owned SGLang services were
  handed over to the training ranks and are restored after controlled exit.
- Committed loss, reward, success, action validity, domain metrics, skill and
  posterior/evolution counters, timings, tokens and declared GPU-hours go to the
  new scalar-only W&B run. No loss point is fabricated before a complete commit.

[Live training](https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/runs/bayesian250-warmup-unqualified-20260914-v1).

## First complete commit

Observed at 11:37 UTC: optimizer step 1 committed all 28 trajectories. TTB loss
1.53057814, mean native reward 0.52564103, binary success 12/28, one credited
skill invocation and one posterior update event. A read is not by itself proof
of useful application. Step wall was 1,285.22 seconds; rollout 1,244.04 seconds;
post-rollout gradient tail 35.94 seconds. The single warmup-step rate is about
2.80 steps/hour, not a steady-state throughput result. Extrapolating that one
step would put 249 remaining steps near 89 hours; the owner authorized continued
execution, not a promise to meet the 72-hour plan. GPU6 thermal slowdown was
observed during collection; no hardware limits were altered.

Validation for the explicit launch mode: 18 targeted CPU tests, then one Linux
CPU `make check`: 5,045 passed / 16 skipped, lint/types/wheels passed. No manual
hash checks or review agents; documentation updates do not repeat the full run.
