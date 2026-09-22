# Optional discovery and zero-coverage Generate

`zero-coverage-generate@1` is an **opt-in method extension**, explicitly authorized by the owner,
not a correction to `idea.tex` and not a claim that the original joint trigger
already permits cold start. `idea.tex` is unchanged. Previous runs retain their
original conditions and evidence.

## Rules

- Default remains residual plateau **AND** strictly decreasing invocation entropy.
  That rule takes priority whenever it succeeds.
- The extension retains the configured residual statistic, W and rho. It needs
  the complete two-window comparison (100 committed batches at W=50).
- It permits **Generate only** when the ordinary entropy gate cannot be satisfied
  and a public `(task_family, context, available_tools)` scope has **zero actual
  skill invocations in the complete window**. Calls in other scopes do not block it.
- Reuse the existing absolute importance floor AND whole-window upper-tail selector;
  malformed actions are not candidates. Do not change the population/normalization
  per worker. Authoring uses original public execution evidence, never hidden answers.
- Default support: selected evidence from at least **two trusted distinct source
  questions and two committed batches**; repeated rollouts do not count as distinct
  sources. Missing source identities cannot manufacture support. These are explicit
  extension controls, not new requirements for skill calls.
- Cold-start authoring uses at most **eight selected witness edges** per skill,
  selecting distinct trusted sources/batches first, then filling in canonical order.
  This bounds the mandatory per-edge authoring contract at W=50; it does not trim
  trajectories, diagnostic windows or posterior evidence. Normal Generate is unchanged.
- At most **one new skill per cold-start boundary**, in stable authority order.
  Cold-start mutations consume the same declared global evolution-cycle budget.
  No additional budget or early optimizer step is created.
- Existing phase transaction, author journal, complete mutation, Z-only reset,
  projection/library segment switch and adapter/checkpoint publication are reused.
  No-op preserves library and Z; its consumed cutoff only rearms after a full fresh
  comparison. Restores retain that cutoff. Ineligible cold-start scopes do not
  close a phase or delay the next normal evaluation.
- New skill IDs start from the configured prior (Beta(1,1) here). Authoring
  trajectories do **not** become retroactive invocations of the new skill.

## Optional skill use

The two-distinct-skills trajectory target is **withdrawn**. Use
`catalog-then-read@1`: show applicable catalog entries, let the single main Agent
choose whether/what to read, return the pinned document, then continue the task.
Reads still consume normal actions/turns. There is no forced execution, call-count
reward, failure filtering, or minimum calls for an accepted trajectory. `@2` remains
only to interpret historical conditions; an explicit `@2 → @1` continuation is supported.

Specificity is not proved by word count or call count. Generate's authoritative
applicability and public examples constrain scope; the extension requests concrete
use conditions, executable steps and a public outcome check, not generic slogans
or a task answer. Semantic usefulness still requires real subsequent observations.
The formal fresh-run seeds are still three general advisory procedures (planning,
tool use and public-evidence verification), not domain-specialized skills. They
are not rewritten under existing IDs; the lifecycle report exposes that limitation
instead of treating broad applicability as evidence of task relevance.

## Inspect the chain

`resolved_method_state` now includes a private `skill_lifecycle` report joining:

1. sampled policy/library, original matched/visible catalog IDs, task family;
2. each declared read's admission, returned library/document version and nonempty
   body receipt (a failed read stays failed);
3. subsequent public task actions and the original terminal success label;
4. per-skill/per-batch posterior events, first observed batch and active status,
   plus immutable document title/summary/applicability for relevance inspection.

Per-phase `rollout-progress.json` also records `catalog_offered_skill_ids`,
`catalog_visible_skill_ids` and `catalog_not_fully_visible_skill_ids`. If the input
window truncated H0, it checks complete catalog entries in the decoded **admitted
tokens**, not mere ID mentions. This sidecar changes no input/token/seed. The
committed H0 coverage aggregates describe the offered catalog, not guaranteed
visibility after every later sliding-window operation.

Coverage reporting no longer mistakes catalog visibility for body visibility.
Body receipt is not proof of comprehension, causal benefit, or use in a later
model request: a read at the final turn can have no following task action.
Unknown historical fields remain unknown. Nothing from this report enters reward,
sampling, TTB, the posterior weighting formula, or the IID actor.

Supplemental W&B telemetry separates `skills/visible_id_count`, `body_read_count`,
`body_read_failed_count`, `body_read_unknown_count`, `read_followed_by_action_count`,
`evolved_visible_count`, `evolved_invoked_count`, and `phase/cold_start_trigger_count`
from natural triggers and real mutations. It exports aggregates, not skill text or
per-question artifacts. Historical telemetry stores must be rebuilt into a new
output store when their enrichment schema changes; original commits remain unchanged.

## Declare, then resume at a complete checkpoint

Generate a **new private config** from the run's actual `condition-current.json`
(or formal config), rather than copying an old YAML with stale horizons/thinking:

```bash
uv run python scripts/configure_skill_cold_start.py CURRENT_CONDITION NEW_CONFIG
```

The helper changes only optional catalog exposure and the declared extension.
The formal entrypoint accepts `--allow-skill-cold-start` together with `--resume`
and the new config. The existing condition-boundary checkpoint records source,
target and effective next step. It rejects changes to model, scorer, seed/data,
budgets, thresholds, optimizer or run plan under that permission. Unfinished old
transactions must be resolved first. No hot replacement or implicit restart.

## Verification identity

Synthetic tests establish reachability, no fabricated posterior, source grouping,
no-op replay protection, concrete mutation → existing retriever → optional read →
next task action → fresh-prior posterior, and complete-checkpoint continuation.
They do **not** establish real Qwen natural invocation, useful skills, monotonic
training loss, or IID improvement. Those remain live-run measurements; zero use,
no mutation and no improvement are legitimate reportable outcomes.

Implementation validation (2026-09-12): targeted cold-start, optional-read,
visibility, telemetry and checkpoint tests passed. One final CPU-only `make check`
on 22049 passed formatting, lint, mypy, **4,403 tests**, and both wheel builds;
12 explicit GPU/private-tokenizer tests were skipped. No GPU task was started and
no active training process or W&B writer was replaced for this change.
