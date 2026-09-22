# Backbone-only Step-0: disjoint sample extension

This run extends [the completed baseline](backbone-step0-20260916.md), rather
than regenerating or re-scoring its cases. The old full-precision native scores
remain immutable. Aggregate each benchmark by actual case count, not by averaging
rounded percentages or treating different native metrics as interchangeable.

## Completed scope

| Set | Benchmark | Retained | Additional | Combined |
| --- | --- | ---: | ---: | ---: |
| IID | HotpotQA | 64 | 64 | 128 |
| IID | TriviaQA | 64 | 64 | 128 |
| IID | AIME2026 | 30 | 0 | 30 |
| IID | HealthBench | 64 | 64 | 128 |
| IID | ALFWorld | 64 | 64 | 128 |
| IID | MBPP+ | 64 | 64 | 128 |
| OOD | MuSiQue-Ans | 64 | 64 | 128 |
| OOD | NQ-Open, supervised DPR retrieval | 64 | 64 | 128 |
| OOD | Math-Hard | 64 | 64 | 128 |
| OOD | GPQA Diamond Biology + Organic Chemistry | 64 | 27 | 91 |
| OOD | ScienceWorld | 64 | 64 | 128 |
| OOD | APPS Introductory | 64 | 64 | 128 |

The frozen released-IID-v3 TriviaQA population contains 128 records but only 127
distinct public inputs: one additional record repeats a retained case despite
having a different source-position ID. It is excluded before generation, without
consulting scores. This is a limitation of the captured source population, **not**
the size of official TriviaQA. The owner subsequently approved expanding the
source. One official [TriviaQA RC validation](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
case was fixed before generation, using seed 0 among 9,338 eligible unique questions
after excluding released-IID-v3 and known train-v3 question texts. All three nonempty
released evidence documents were retained in source order, without answer-based
selection. It ran through the same formal collector and scorer, once. The combined
TriviaQA result is explicitly **127 released-IID-v3 inputs + one official RC input**,
not a claim that all 128 came from the previous source/projection.

The completed extension has **667 additional scores + 734 retained = 1,401 total**:
IID 670 and OOD 731. AIME was retained without any new calls. GPQA adds 27 Organic
Chemistry records to the retained 19 Biology and 45 Organic Chemistry records.
ALFWorld cases with similar task wording are still distinct native game states;
their identities are checked at game level, not by goal text alone.

## Results

All scores are on a 100-point scale; the native metrics remain distinct.
Additional counts are listed above, so GPQA is weighted by 64 + 27, not by an
unweighted average of two cohort percentages. Presentation uses half-up rounding;
aggregation uses the persisted, unrounded per-case native values.

| Set | Benchmark / primary metric | Retained score | Additional score | Combined score |
| --- | --- | ---: | ---: | ---: |
| IID | HotpotQA F1 | 82.08 | 83.52 | **82.80** |
| IID | TriviaQA F1 — mixed source above | 86.76 | 76.98 | **81.87** |
| IID | AIME2026 accuracy, n=30 | 70.00 | — | **70.00** |
| IID | HealthBench Luna-medium native rubric mean | 31.85 | 36.06 | **33.96** |
| IID | ALFWorld success — legacy input caveat | 68.75 | 64.06 | **66.41** |
| IID | MBPP+ Base AND Plus pass@1 | 79.69 | 76.56 | **78.13** |
| OOD | MuSiQue-Ans F1 | 68.68 | 60.61 | **64.64** |
| OOD | NQ-Open supervised-DPR EM | 42.19 | 37.50 | **39.84** |
| OOD | Math-Hard accuracy | 90.63 | 79.69 | **85.16** |
| OOD | GPQA Diamond BioOrganic accuracy, n=91 | 56.25 | 55.56 | **56.04** |
| OOD | ScienceWorld signed native final mean | 24.34 | 23.92 | **24.13** |
| OOD | APPS Introductory pass@1 | 81.25 | 65.63 | **73.44** |

These results do **not** meet all targets. Only HotpotQA, TriviaQA, MBPP+ and
Math-Hard exceed 90% of the specified numerical targets. ScienceWorld is below
the separately accepted 38-point threshold. This extension did not replace
failed samples, tune prompts, or select a better-scoring cohort.

## Conditions retained

- One Qwen3.5-9B owner per episode; adapter-free, skills-off, no consultants.
- Thinking on for AIME2026, HealthBench, Math-Hard and GPQA only.
- Zero training updates, posterior updates, skill evolution or training-evidence
  writes. The evaluation harness, not a training entry point, executes the work.
- IID reuses the captured formal controller, tokenizer, phase budgets, action
  interface, scorer and deterministic SGLang 0.5.15.post1 configuration.
  Only the already-completed AIME domain is omitted from the additional schedule.
- OOD reuses each benchmark's exact earlier code snapshot, parser, tools,
  budget, scorer and non-deterministic SGLang 0.5.9 configuration. One owned
  replica was restored to that configuration; the two other replicas continue
  serving IID. Batch remains 32. Scheduling/replica allocation is recorded
  separately from scientific controls; no reproducible OOD seed claim is made.
- NQ retains the separately declared NQ-supervised DPR condition. It is neither
  closed-book nor strict zero-shot OOD.
- HealthBench retains Luna medium. Only missing/unreturned judgments may use
  the authorized official-API recovery; returned judgments and owner answers
  cannot be replaced based on score.
- ScienceWorld reports signed native final score, including -100, not clipped
  reward. MBPP+ requires Base AND Plus. All failures remain in the denominator.
- The known ALFWorld legacy catalog/reset-goal conflict is **not repaired
  mid-extension**. Its combined score remains an interface-caveated diagnostic,
  not a certified clean native baseline.

These are development/regression panels, not untouched final tests. Selection
uses source identities and public-input duplication only; there is no score-based
replacement. New IDs, private configuration, process ownership, logs and scoring
provenance are persisted outside Git. No benchmark content or credentials are
included here.

## Recovery and complete trace accounting

One new HealthBench case had five unreturned gateway rubric requests: one upstream
400 error, three connection errors and one timeout. Following the existing owner
authorization, only those five missing judgments were sent to official
`gpt-5.6-luna`, medium, without automatic retries or a new owner answer. All five
returned. The unchanged native parser aggregated the stored responses; the original
incomplete ledger remains alongside an explicit completed-score supplement. Shared
Judge code was not edited. The new IID cohort has 319 engine artifacts plus this
one separately supplemented score, not 320 falsely labeled successful engine runs.

The interrupted IID shard left 63 original cases unstarted. Only that frozen suffix
ran, once, with its original global sampling coordinates; the settled prefix was
not rerun. Initial private helper/setup failures occurred before model dispatch and
are retained in the operational logs. All 667 additional cases now have a score;
none is silently omitted because its generation, submission, or judgment failed.

The read-only structural pass covered all new trajectories and **4,959 owner model
requests**. IID covered 1,391 action records: raw response/result traces and action
result/parse text agreed throughout. There were 21 format-error actions (HealthBench
4, ALFWorld 16, MBPP+ 1), retained with any same-owner subsequent handling and costs.
This is a structural comparison, not a manual semantic audit of every reasoning token.

All 64 new ALFWorld catalog/reset goal pairs were read. Both goal texts entered each
prompt history; at least two pairs have clear object or destination conflicts,
beyond ordinary wording differences. The existing input caveat therefore remains.
Fixing it requires a separately named, consistently applied architecture condition;
66.41 is not certified as a clean native ALFWorld baseline.

New OOD diagnostics retain eight Math-Hard and six GPQA empty submissions after their
bounded continuation allowance. One NQ case exhausted its eight-call allowance
without a final answer, despite 6,093 unused output tokens; this was not an HTTP
truncation. APPS submitted 64 candidates, of which 42 passed all tests; the native
checker classified 14 as wrong answer, three as runtime error/timeout and five as
compile/load error. Four payloads have invalid Python syntax. One further payload
is syntactically valid but contains only comments: the owner put that block after
its earlier program, so the frozen last-code-block contract submitted it unchanged.
Five responses ended by length. No earlier code block was selected using hidden tests.

ScienceWorld's 1,457 new simulator commands matched owner command, execution and
acknowledgment records. All 1,409 eligible next-request observations were present.
The additional cohort has 28 full successes and 19 native negative endings; the
combined 128 has **59 full successes and 39 negative endings**. Signed final mean
24.13 is primary, not the higher clipped-reward mean. These native failures were
not converted into transport retries or replacement samples.

## Cost, validation and resources

| Additional work | Model requests | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| IID owner | 2,782 | 29,446,415 | 969,032 |
| OOD owner | 2,177 | 21,447,648 | 986,469 |
| HealthBench Judge, returned usage | 664 attempts / 659 returned | 837,206 | 93,920 |

The Judge's five unreturned gateway attempts have unknown additional usage, not
zero cost. The 659 completed new rubric judgments include five official supplements;
returned model labels were `gpt-5.6-luna`. OOD recorded 1,601 tool calls and zero peer
model calls. Combined historical + additional owner totals are 10,027 requests,
106,088,736 input tokens and 4,368,423 output tokens; the historical portion was not
called again. Generation and scoring finished about 32.4 minutes after the first
additional OOD job started, approximately 1,237 completed new episodes per hour
across both sets, excluding preparation and final read-only review.

No public evaluation implementation changed. Preparation and final verification
covered source separation, effective controls, raw-response/action correspondence,
native score denominators, original-score preservation and cost aggregation.
Missing ALFWorld assets were added without replacing existing scenes. No new unit
test, full-suite run, build or lint was needed for this documentation/private
bookkeeping-only completion. The earlier milestone's 5,455 passed / 16 skipped,
Ruff, mypy and builds are recorded in the baseline report and were not repeated.
No independent review agent or new hash audit was introduced; the mandated Luna
agent handles Git delivery only. No formal training or W&B training stream ran.

All evaluation coordinators and their recorded children have exited. The three
owned SGLang services remain healthy and resident, batch 32: two use the formal IID
configuration and one preserves the old OOD configuration. Service PIDs, endpoints,
GPU allocation and private results are retained in the operational handoff, not
Git. No other users' GPU processes were terminated. The separate model-weight
download/upload queue continues using local G transit.
