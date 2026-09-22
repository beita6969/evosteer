# OOD native-contract repair

This is a code/diagnostic milestone following A13 and the separately frozen
A14/A15 runs. It is **not a new seven-benchmark generation round**. One frozen
Qwen3.5-9B owner remains responsible for every evaluated trajectory; skills-off,
no consultants and no answer-informed candidate selection remain in force.
`idea.tex`, `AGENTS.md` and the unrelated handoff edit are unchanged.

## Fixed contracts

- **ScienceWorld:** `NativeEnvironmentScore` and `LearningRewardProjection`
  separate raw measurements from bounded learning rewards. The primary metric
  is now `native-final-score`, raw final score / 100, verifier
  `scienceworld-official-native-final-score@ood3`. Legitimate negative scores
  are scored outcomes, not candidate-format failures. The clipped reward and
  success-at-100 are secondary metrics. Missing native evidence stays incomplete.
  The old `native-reward` / `@ood2` records remain readable without relabelling.
- **Public contracts:** one `BenchmarkSpec` registry supports all seven IID and
  seven OOD interfaces, including the training-public bridge. The independently
  selected `public-task-semantics@5` shares domain meaning with training, while
  retaining distinct native/training answer carriers. No references, hidden
  tests or rubrics enter the spec. NQ remains closed-book; no retrieval was added.
- **Identity/input delivery:** panel, source population, benchmark, revision,
  input profile, split and development exposure are separate fields. OOD is no
  longer labelled `frozen-eight-iid`. Unknown historical revisions remain
  unknown. New MuSiQue exports preserve all paragraphs and stable paragraph IDs,
  without supporting labels or decomposition answers. Trusted per-call records
  retain actual input tokens, required-field presence, paragraph order and
  omitted-history counts. Legacy exports cannot retroactively prove completeness.
- **Submission/budget:** typed outcomes distinguish received text, length stop,
  missing carrier, unique final, exhausted budget and sealed submission. Syntax
  validity and native test success are separate diagnostics. Original single-
  response 12,000-token conditions are unchanged. An opt-in, independently named
  reserve condition leaves 1,000 tokens inside the same total after its first
  11,000-token allowance. A valid final stops immediately; unused reserve is not
  spent. Same-stream native-thinking continuation remains a separate protocol.
  A recognized but syntactically wrong final is still submitted, not repaired
  using its AST or tests; a reserve is not a guarantee of a better score.
- **ScienceWorld observation:** the opt-in `public-state@1` profile forwards only
  the current look, inventory and task description already returned by the same
  native response. It does not query a gold plan, valid-action oracle, hidden
  predicates or private score progress, and does not choose actions. The old
  `text-only@1` profile is retained. Private transition diagnostics distinguish
  native rejection, negative terminal outcomes and unproven progress; native API
  calls and simulator moves are counted separately.
- **Scorers/reporting:** official APPS/LCB execution and positive-verdict
  aggregation are retained. All-positive results must cover the full test
  population; syntax checks only diagnose the already fixed candidate. Reports
  add source/input/submission gaps, owner cost, native failure kinds, actual
  returned judge identities, source composition and positive/negative rubric
  coverage. Missing metadata is not inferred from answers or scores.

## Deterministic checks on old records

`scripts/reproject_scienceworld_scores.py` produced a new private report from
all 64 A12 environment journals, with **zero new model calls or environment
steps** and no replacement of old scores. Native mean is **12.75/100**; clipped
learning reward is **51.8125/100**. There are exactly 25 final scores of -100,
28 final scores of 100, no missing final scores, and 30 cases at least 70.

Read-only A12 judge checks cover all 64 Omni cases and 360 medical criteria.
All stored final owner identities and recomputed native scores agree; all Omni
templates agree with the frozen template file and every report parses to its
stored verdict. All 424 API response model names are `gpt-5.6-luna`, with no
unresolved attempts or unknown usage. These checks made no judge requests and
**do not establish independent blind-agreement accuracy**.

LiveMedBench's snapshot is `v202601`: 125/238 positive criteria are met, earning
972 of 1,835 positive points; 22 negative criteria trigger -137 points. These
pooled coverage diagnostics do not replace the mean of case-level rubric
scores. Historical language/specialty metadata is missing for all 64 cases;
it is reported as unknown, not guessed.

## Independent experiments, not score-driven retries

`scripts/prepare_ood_condition.py` prepares, but never launches, separately
named native-public, training-shared-semantics, code-reserve and public-state
conditions. It retains the supplied sample population, seed, model and scorers.
New generations receive separate judge-cache namespaces; previous submissions'
judge ledgers cannot be used as their grades. Each intervention must actually
have its target benchmark population present.
The original 448 repeatedly inspected cases remain development regressions.
The answer-free source-question/variation exclusion helper rejects reused
source questions despite new task IDs; it does not prove semantic paraphrase
isolation or absence from base-model pretraining. No fresh final holdout or
new seven-domain result is claimed at this milestone.

## Validation and operations

Affected suites covered raw negative/partial/missing/history scores, shared
public/private input boundaries, actual actor/broker submission budgets, native
observation allowlisting and diagnostics. Deployed official code checkers also
passed **14/14 fixed synthetic programs** across stdin, callable, `__main__`,
wrong answer, syntax error, runtime error and timeout, using the existing CPU
sandbox, not a replacement grader.

The native ScienceWorld worker compatibility check also passed: a single public
`look around` call returns all three allowlisted state fields and only the free
`act` surface. It records **one native API step but zero simulator moves**,
demonstrating why those costs must be separate. No model generated an action or
answer, and this is not an evaluated episode. The first check omitted the
deployed Java environment and failed before initialization; its log is retained,
then the check passed using the original deployment's Java settings.

Final CPU-only `make check` passed **4,535 tests, 12 skipped**, Ruff, mypy
(638 source files), both wheel builds and the model-wheel boundary check.
Pytest took **312.55 seconds**; the whole check took about **335 seconds**.
Ten private-real-tokenizer tests skipped by the generic suite were separately
run against the deployed Qwen3.5-9B assets and **all ten passed in 7.31 seconds**,
without CUDA. The two CUDA-only qualifications remain intentionally unrun.

The first full check retained eight failures caused by incomplete old synthetic
budget fixtures (4,523 passed). Those fixtures were supplied actual token
budgets rather than relaxing production validation; both affected files then
passed **14 tests**. The core milestone passed **4,531 tests**. Final self-check
found and fixed the public-context receipt omission and new-condition judge
cache isolation; their **18 tests** and the final full check above passed.
Other targeted batches exercised native projection, shared OOD contracts,
single-owner submission budgets, observation profiles and native scoring.
Passed deployed-checker execution was not repeated for these reporting-only
changes. Documentation-only completion does not trigger another full check.

No independent review/consultant agent, hash check, formal training or W&B run
was introduced. The dedicated Luna agent handles Git only.
A14/A15 raw journals, outputs, configurations and logs are archived privately;
no licensed content is included in these public notes.

Personal post-run checks on all eight cards of 22049 confirmed the existing
GPU7 service only for this work: PIDs 472775/474607, `CUDA_VISIBLE_DEVICES=7`,
39,058 MiB reserved, health HTTP 200 and base Qwen3.5-9B / default weights.
A14/A15 coordinators have exited. Repair checks were CPU-only (check runners
432717, 497069, 537165), with no new SGLang service or use of another GPU.
The original service remains resident as requested. This milestone does not
claim the benchmark score targets, independent judge accuracy or a fresh
unseen final-evaluation population have been achieved.
