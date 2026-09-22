# Qwen3.5-9B Direct Reference 7pp Implementation Report

## Scope

This change implements the code-level augmented parity repairs for the separate,
adapter-free Paper Direct-Qwen track. It does not modify `idea.tex`, Protocol 10,
the Bayesian Trajectory-Balance method, or the training distribution. Licensed
questions, targets, model generations, patches, environment traces, and per-task
verdicts remain outside Git.

The current authoritative project IID catalog contains nine benchmarks:
HotpotQA, TriviaQA, AIME 2026, HealthBench, WebShop, ALFWorld, SpreadsheetBench,
SWE-bench, and HumanEval. Older direct-reference artifacts used a superseded
seven-IID split; they remain historical evidence rather than a replacement for
the current catalog. This repair is scoped only to SWE-bench's 128-instance IID
panel and official CPU evaluator; no OOD benchmark is rerun.

## Implemented repairs

### Formal result semantics

- Made the formal gate scope-aware. IID-only and OOD-only reports now return
  `NOT-EVALUATED`; only an all-role report containing the complete 14-benchmark
  protocol can return formal `GO` or `NO-GO`.
- Made the renderer reconstruct typed coverage and recompute references, gaps,
  statuses, and published aggregates from the executable protocol. Redundant
  aggregate fields and all protocol bindings must agree before rendering.
- Added strict coverage conservation across candidate responses and generation,
  scorer, and environment infrastructure failures. Incomplete official scoring
  remains `INCOMPLETE` rather than becoming a zero.
- Added explicit metric source scales so unit-interval scorer values cannot be
  confused with percentages.

### Protocol, runtime, and request identity

- Added typed upstream evidence, parity metric, multidimensional comparability
  evidence, dataset revision, selection rule, horizon policy, and complete
  prompt/decoding/parser/scorer bindings.
- Enforced greedy-versus-sampling numeric invariants and task/request seed
  equality. Replay now checks the declared seed, request attempt, served model,
  response shape, and attempt lifecycle.
- Added model-route capability checks, token-budget preflight, response-model
  identity checks, and a serving receipt covering model, tokenizer, chat template,
  context length, and adapter-free configuration.
- Removed legacy implicit contract IDs from formal task types.

### Durable private generation and replay

- Moved raw-response observation ahead of parsing and added a single asynchronous
  journal writer with bounded flushing plus attempt start/complete markers.
- Made scorer replay accept only complete attempts, collapse only explicitly
  classified infrastructure retries, and reject duplicate definitive attempts or
  contract drift. Candidate failures are never regenerated or retried.
- Made released IID population manifests select rows by source identity and then
  validate task order, dataset revision, and selection rule.

### Benchmark-specific behavior

- Replaced the truncated HotpotQA rendering inherited from the upstream
  preparation script with an answer-blind renderer that supplies all ten
  complete public context passages from each frozen record. The dedicated
  `hotpotqa-full-context@1` prompt and dataset-variant IDs make this request
  contract visible in aggregate reports.
- Kept answer-blind, versioned prompt/parser/scorer contracts for QA, AIME,
  multiple choice, code, patch, and native-action tasks. Added offline QA,
  interactive-trace, and AIME calibration diagnostics that operate only on
  non-final or private evidence.
- Corrected interactive telemetry: candidate-invalid and horizon outcomes no
  longer masquerade as official terminal states. Reports now separate official
  terminal, horizon, candidate-invalid, partial-reward, and cleanup signals.
- Made the SWE dataset variant protocol-driven and strengthened typed official
  verdict handling. The local official-evaluator bridge is restart-safe and does
  not turn missing official verdicts into failed candidates.
- Bound the declared seed 42 to every repository-agent request. Replaced the
  supervisor-to-MExec delegated edit with one Qwen using canonical
  worktree-bounded read-only navigation and one final unified diff. The formal
  contract disables the executor, incremental workspace edits, syntax/test
  feedback, environment progress memory, issue-derived member hints, and repeat
  steering. Thus neither an adapter-free two-call orchestration control nor a
  stronger editable-agent control can be reported as the Direct-Qwen score.

## Validation

Development used affected-scope checks rather than repeating the full gate after
each edit. The existing protocol/renderer/coverage/replay/static/interactive
selections had already passed in their affected batches. This SWE repair added
focused coverage for seed injection, conflicting-seed rejection, canonical
read boundaries, orchestration-hint removal, the read-only final-diff parser,
and all scorer bindings; its 10 focused tests passed on the Linux tmpfs checkout.
Affected Ruff format/lint and Python compilation checks passed locally.

The final clean milestone gate ran on the 22049 Linux local filesystem as
`CUDA_VISIBLE_DEVICES="" make check` with a node-local temporary directory
outside the repository:

- Ruff format: 630 files already formatted;
- Ruff lint: passed;
- mypy: no issues in 331 source files;
- pytest: 2,055 passed;
- both wheels built;
- model-wheel check passed.

An earlier consolidated attempt correctly exposed one replay test fixture that
still named the superseded HotpotQA prompt profile; its four affected tests were
rerun after correction. The next attempt was stopped by generated test fixtures
under a repository-local temporary directory being visible to Ruff, so the
clean final gate moved `TMPDIR` outside the checkout. The Linux-local full gate
above is authoritative. No independent audit/review agent was started because
repository instructions prohibit that class of sub-agent. No hash, SHA, or
SHA256 check was performed.

## Evaluation policy and retained evidence

- Each selected benchmark uses one frozen seed-42 panel. AIME uses all 30
  released records; every other non-SWE IID benchmark uses 128.
- Successful generations and candidate failures are immutable. Only explicit
  infrastructure failures may be retried with the same task and contract.
- The model route is frozen Qwen3.5-9B with no adapter, LoRA, skill library,
  retriever, posterior, or Z head.
- The corrected HotpotQA-only rerun covered all 128 frozen records with no
  generation, scorer, or environment infrastructure failure. EM increased from
  50.78125% to 63.28125%, and F1 increased from 62.52333603896104% to
  78.97818687133945% when the full context was supplied.
- The issue-only one-shot SWE diagnostic is complete at 8/128 resolved. The
  delegated, editable single-Qwen, and raw-tools editable controls reached
  35/128, 38/128, 33/128, and 39/128 under their separately versioned contracts;
  none is relabeled as the final Direct-Qwen result. The formal read-only
  inspection plus one-final-diff run has 128/128 definitive official verdicts:
  16 resolved, 20 test failures, 19 apply failures, and 73 candidate-invalid
  outputs, with zero generation, scorer, or environment infrastructure failure.
  Its 12.5% score is 4.69 percentage points from the 17.19% reference and
  therefore passes the strict `<7 pp` numeric gate.

## Scientific limits

The released comparison does not publish the exact direct-baseline prompt,
decoding arguments, seed aggregation, WebShop/ALFWorld payloads, or reconstructed
environment identity. Those unknowns are represented as evidence dimensions,
not repaired through final-panel prompt or seed selection. Consequently, the IID
scope remains scientific `NO-GO` even if any numeric component falls inside the
strict seven-percentage-point band, and the scoped report cannot decide the
formal training gate.
