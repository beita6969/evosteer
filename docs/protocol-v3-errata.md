# Protocol-v3 historical-goal errata

`idea.tex` remains the immutable scientific authority and
[`method-v3-final-spec.md`](method-v3-final-spec.md) is the current engineering
authority. The phase goals are historical implementation records. The
instructions below are superseded and must not be reintroduced into active
code.

The benchmark-wire notes in this historical errata predate Protocol 10. They
describe old artifacts only; `evaluation.tex` and `EVALUATION_SPEC.md` govern
new benchmark work.

| Historical location | Superseded instruction | Current rule |
|---|---|---|
| Phase 4 rollout recovery text | Resume a live environment or repair an interrupted rollout in-process | A rollout is fresh and atomic inside one child attempt; infrastructure failure ends that attempt. |
| Phase 4 environment protocol | Expose idempotency keys, execution checkpoints, or restore methods | Every fresh environment session accepts each consecutive step once; duplicate/skip calls fail and no recovery surface is public. |
| Phase 5 collection/rejection text | Replace rejected or failed rollout attempts until a batch is full | Plan exactly `B` tasks and attempt each exactly once; a legal reward-zero trajectory stays in the batch, while infrastructure failure ends the attempt. |
| Phase 5 checkpoint retention text | Automatically keep/prune the newest checkpoints | Live checkpoint code publishes one caller-named immutable snapshot and never deletes or prunes; retention is explicit offline maintenance. |
| Phase 6 replay paragraphs | Reconstruct or repair live diagnostic state from event history | Live projections hydrate one exact typed runtime state; event replay is offline audit only. |
| Phase 6 stagnation representation | Use `None` for all unavailable/non-comparable residual windows | Residual-window evidence is a closed union: insufficient history, zero previous residual, or comparable windows. |
| Phase 7 calibration switch | Put an enabled/disabled Bayesian switch in the full application graph | Full core is always Bayesian; the no-Bayesian condition is an arm-owned application graph with its own source records. |
| Phase 8 action limits | Treat max-actions/max-generate truncation or no-action as a successful phase | A verified phase produces one complete nonempty Phi decision; insufficient evidence or budget fails the attempt. |
| Phase 8 authoring failure text | Skip one failed proposal and continue with a subset | Every proposal is authored exactly once before mutation; one failure invalidates the complete Phi attempt. |
| Phase 8 rollback/rebuild text | Use rollback or event rebuild as a live recovery authority | Active library/runtime state comes only from one exact snapshot; audit replay and maintenance are offline. |
| Phase 8 retrieval text | Retrieve a compact top-N/all-active metadata view | Production retrieval returns all task-applicable active skills with full instructions; context overflow is an error. |
| Phase 8 pending-phase text | Return success while a newly detected phase remains pending, or resume it in another attempt | A phase detected during the frozen phase-search interval of `run(plan)` is resolved in that same call and followed by the closure-training tail, or the attempt fails; forensic phase evidence is non-executable. |
| Earlier budget ledgers | Release, abort, unknown-use reconciliation, scopes, or duplicate-idempotent transitions | One attempt uses only strict `reserve -> settle`; duplicates and missing exact usage fail. |
| Earlier experiment matrices | Encode sampling and arms through mode strings plus nullable controls | Sampling and all seven protocol rows are tagged variants with exact fields. |
| Earlier rollout token admission | Require `encode(decode(sampled_action_ids)) == sampled_action_ids` and reject alternate token segmentations of the same text | Preserve the sampled action span for `K_t` and both scoring directions; use its deterministic decode for action text and artifact admission, without inverse re-encoding. |

The method name remains Protocol v3 because these corrections do not alter the
scientific equations. This implementation makes one incompatible wire
migration with no compatibility path:

- attempt IPC is `@5` and published attempts are `@4`, binding exact input,
  public identity, and every required source log;
- main source payloads are training `@4`, library `@4`, phase `@5`, and cycle
  `@7`;
- evolution contracts are `@4`, including task-family Split modes;
- runtime execution and snapshots are `@6`, including an exact run cursor and
  immutable snapshot identity; and
- benchmark protocol/freeze artifacts named here are historical `@8` evidence;
  the current Python reader is historical `@9`, while new formal execution is
  blocked until the frozen population-level Protocol 10 migration completes.

Existing older-version artifacts remain historical evidence only. They are
not accepted by executable or scientific-audit entrypoints.
