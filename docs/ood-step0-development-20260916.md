# OOD Step-0 development — September 16

**Sampling clarification discovered later:** seed0 below is the requested model
seed, not proof of reproducible generation. The deployed SGLang 0.5.9 sampler
ignored request seeds with deterministic inference disabled. Frozen-panel CPU
selection is unaffected. See [the follow-up diagnosis](ood-step0-followup-20260916.md#actual-sampling-seed-limitation-discovered-in-the-paired-review).
Historical scores are retained, not replaced or retroactively reclassified as
deterministic comparisons.

## Current goal rule

The owner's latest rule is **primary score > 0.90 × target**, for both
32-to-64 promotion and acceptance of a complete 64-sample development panel.
Use unrounded values; equality fails. Configuration:
`configs/evaluation/current_ood_targets.json`. The standalone
`scripts/report_current_ood_targets.py` reports each run independently and
does not combine candidates or remove infrastructure failures.

These inspected panels are development evidence, not unseen final tests.
Trying another complete, preselected panel is now permitted for this development
goal. All attempts remain recorded; best-panel selection is not an unbiased estimate.

Current accepted **64-case development confirmations** are Math-Hard 89.0625,
APPS Intro 84.375, GPQA BioOrganic 64.0625 and MuSiQue F1 68.8628719479.
These come from separately identified runs/conditions below, not one combined
new six-domain run. Those four benchmarks stop their score-repair loop. NQ and
ScienceWorld remain unfinished; the overall goal is not complete.

## Completed fresh six-domain run

Run `ood32-wire-v7-20260916`: Qwen3.5-9B, adapter-free, single owner,
skills off. Math-Hard and GPQA thinking-on; the other four thinking-off.
Batch 32; seed 0; 8,000 output tokens, except APPS's unchunked 12,000.
ScienceWorld retains its 200 native-move horizon and signed terminal scores.
Training, posterior, skill-evolution and training-evidence updates are all zero.

| Benchmark | Samples | Native primary | Strict threshold | 32-sample gate |
|---|---:|---:|---:|---|
| MuSiQue-Ans F1 | 32 | 58.75 | 67.50 | below |
| NQ-Open closed-book EM | 32 | 18.75 | 44.28 | below |
| Math-Hard accuracy | 32 | 84.375 | 73.26 | passed |
| GPQA Diamond BioOrganic accuracy | 32 | 53.125 | 63.00 | below |
| ScienceWorld native final mean | 32 | 5.625 | 50.553 | below |
| APPS Intro pass@1 | 32 | 87.50 | 80.46 | passed |

All 192 records have native scoring outcomes; zero infrastructure failures.
Cost: 11,075,079 input tokens, 526,653 output tokens, 1,127 owner calls,
801 recorded tool calls including 796 simulator actions; no consultant calls.
The three existing services used worker08 GPU0/1/4. No new inference service
or other GPU allocation was created for this round.

## What the records establish

- The whole-record check covered all 192 trajectories, 1,127 calls and 796
  actions: no command/source mismatch, missing acknowledgement, lost immediately
  preceding public observation, continuation-prefix mismatch or incomplete input
  receipt was found. This is not proof that every possible software bug is absent.
- Bounded native thinking closes only the channel framing at the reserved
  boundary; the owner generates the answer. All forced framing tokens and calls
  are charged. Math-Hard still has five budget-exhausted empty submissions.
- GPQA has three empty submissions and two identifiable presentation failures:
  a bold final label and a single option label followed by its text. Five other
  replies redundantly stated the same final both in prose and in one tool call,
  causing unnecessary clarification. The v8 source snapshot fixes these literal
  transport cases, keeps contradictory submissions invalid, and versions the
  GPQA scorer as `gpqa-diamond-frozen-choice-exact@ood2`.
- APPS has one empty submission, one submitted example block rather than a
  completed program, and two wrong programs. No hidden tests select an earlier
  program; no automatic code repair is performed.
- ScienceWorld has 12 full successes, 13 native negative terminal results and
  seven partial results. Every action was retained and inspected. Wrong object
  focus, object/room references and incomplete experiments remain visible;
  correcting public API descriptions did not establish an overall capability gain.
- QA fields reach their scorer literally. Shorter MuSiQue submissions did not
  guarantee better answers; NQ remains closed-book, not a retrieval experiment.

## Separate follow-up, not a replacement six-domain result

`promoted-v8b-20260916` uses the v8 source snapshot for Math-Hard 64,
GPQA 32 (same fixed IDs), and APPS 64. Math-Hard uses seed-0 reservoir sampling.
APPS uses a new whole panel: seed-0 sample of 128 source indices, positions
64–127, fixed before generation. Its preceding 64-sample attempt scored 76.5625
and remains recorded. Neither run is overwritten or spliced into the other.
An earlier v8 launcher precheck rejected a partial catalog ordering before any
generation; v8b corrects that ordering without changing the selected cases.

This follow-up completed with zero infrastructure failures:

| Benchmark | Samples | Native primary | Goal status |
|---|---:|---:|---|
| Math-Hard accuracy | 64 | 89.0625 (57/64) | accepted 64 |
| GPQA Diamond BioOrganic accuracy | 32 | 68.75 (22/32) | promote to 64 |
| APPS Intro pass@1, new whole panel | 64 | 84.375 (54/64) | accepted 64 |

All 160 trajectories and 394 owner calls passed the same whole-record wire checks.
Cost: 1,777,052 input tokens and 648,965 output tokens, no tool/consultant calls.
Math retains six empty budget failures and one wrong answer; GPQA one empty
submission and nine wrong answers; APPS one empty submission and nine wrong
programs. Every nonempty APPS submission parsed as Python. GPQA's malformed-choice
and redundant final/tool clarification failures are absent in this run. These are
new model generations as well as changed interfaces, not a parser-only causal estimate.
The separate GPQA 64-sample follow-up retains all 19 Biology records plus 45 Organic
Chemistry records, seed 0, with unchanged v8 settings.

### GPQA 64-sample confirmation did not pass

`gpqa64-v8-20260916` completed all 64 records: **56.25 (36/64)**, below the
strict 63.00 threshold. The preceding 32-sample result is not substituted for
this failed confirmation. There were zero infrastructure failures, 58 scored
choices, four budget-exhausted empty submissions and two invalid choice carriers.
All 64 trajectories and 199 calls passed the recorded transport, continuation
prefix, token-ledger and public-input receipt checks; there were no tool or
consultant calls. Cost: 1,024,571 input tokens and 323,511 output tokens.

The two remaining carrier failures are concrete parser limitations: an empty
Markdown final-answer heading followed later by an explicit final letter, and
a leading labelled option followed by a separate explanation paragraph. Neither
has been silently repaired or rescored in this run. Even recovering both could
add at most 3.125 points, insufficient to pass 63.00. Four incomplete outputs
also remain; bounded thinking did not guarantee a completed final answer.

Consequently only Math-Hard and APPS have accepted 64-sample development panels
so far. MuSiQue, NQ-Open, GPQA and ScienceWorld have not met their respective
64-sample goals. The three coordinators for the runs above have finished; the
existing batch-32 SGLang services on GPU0/1/4 remain running intentionally.

## Follow-up: carrier v3 and a separate decoding comparison

The next source snapshot fixes the two demonstrated GPQA carriers, retaining
the historical parser and declaring `gpqa-diamond-frozen-choice-exact@ood3`.
Synthetic tests include conflicting heading/body choices, conflicting explicit
finals, alternative lists and absent choices; no reference is passed to the parser.
A separate diagnostic applies both parser versions to all 64 frozen outputs:
36 correct becomes 37 correct (57.8125), with both carrier errors resolved and
the four empty submissions retained. One recovered choice was wrong. This is
not new generation, does not replace the original run, and still fails the gate.

ScienceWorld now distinguishes the native clock horizon from the runner's action
count. A recorded episode had 32 actions but 211 simulator moves; the released
wrapper terminates when moves exceed its configured limit. Its termination was
previously unspecified. The fix changes private diagnostics only, preserves raw
negative scores and gives success/failure precedence over the horizon label.

`ood32-wire-v9-presence0-20260916` starts a new four-domain condition on the same
32 IDs per domain: MuSiQue, NQ-Open, GPQA and ScienceWorld. Presence penalty is
0 instead of 1.5; other sampling, thinking, prompt, tool, context and output
budgets are unchanged. This is an explicitly named decoding experiment, not an
assertion that the old penalty was a transport bug. GPQA uses the new scorer.
Math-Hard and APPS are not rerun after their accepted 64-sample results.

Before launch, the GPQA/parser/schema CPU batch passed 79 tests and the
ScienceWorld command/owner-finish CPU batch passed 41. Relevant Ruff checks passed.
No evaluation or review subagent was used; the Luna xhigh helper only committed
the specified public changes. No training or W&B training update was launched.

## Presence-zero round completed; separate corpus condition authorized

The four-benchmark `ood32-wire-v9-presence0` run completed 128 fresh trajectories
with zero infrastructure failures. Each benchmark retained the preceding 32 IDs.

| Benchmark | Primary score / 100 | Strict gate |
|---|---:|---:|
| MuSiQue answer F1 | 57.4107 | >67.50 |
| NQ-Open closed-book EM | 21.875 (7/32) | >44.28 |
| GPQA Diamond BioOrganic accuracy | 56.25 (18/32) | >63.00 |
| ScienceWorld signed terminal mean | 14.4375 | >50.553 |

None qualifies for promotion. Whole-record checks covered 1,001 owner calls and
815 environment actions: no command substitution, missing latest observation,
continuation-prefix mismatch, cost mismatch or incomplete required-input receipt
was detected. Cost was 10,997,140 input and 269,893 output tokens; 821 tool calls,
zero peer calls. All ScienceWorld action records were also read. Its 15 full
successes, 12 native negative terminals and five partial scores remain unmodified;
object-selection, navigation and experimental-reasoning failures persist. Reduced
presence penalty is a decoding experiment, not an established communication fix.

One additional GPQA carrier fault was demonstrated: an outer explicit final and
the same explicit final inside `submit_answer` were rejected when the parameter
also contained explanation. The literal transport now accepts agreeing declared
finals while preserving the entire parameter; contradictory declarations still
fail, and no reference answer participates. A fresh same-panel GPQA32 v10 run
tests this change. It is not a replacement for the failed prior runs.

The owner explicitly authorized a separate **NQ-Open retrieval-augmented**
condition. Its question IDs and original FiD-style EM scorer are unchanged;
closed-book results are retained. The condition exposes one read-only
`corpus_search(query)` tool to the same owner, with three queries, five passages
per query, unchanged 8,000-output-token budget and thinking off. No consultations,
reference-conditioned retrieval, answer selection or training updates occur.

The corpus is the complete public [DPR Wikipedia 100-word passage release](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py),
not DPR's NQ question-specific positive-passage pools. Import reads only its
`id/text/title` fields. A separately built SQLite FTS5 index uses Porter/Unicode
tokenization, BM25 title/text weights 5:1, a fixed common-word stoplist, the first
32 unique remaining query terms joined by OR, and rank-ordered top-five passages.
Queries have a 30-second execution limit; errors remain infrastructure failures.
The completed index is read-only during evaluation. Its public identity, search
budget, all literal owner queries, returned passages, latency and costs are
persisted privately. The broker validates each request against the actual owner
output, never against a reference answer. The new input profile is distinct from
closed-book and is part of architecture-matching controls.

The related real actor/broker and tool-carrier batch passed 76 tests. Corpus
acquisition is complete; full-corpus indexing and new NQ32 generation remain
pending at this update. No retrieval score is claimed yet.

## Literal-carrier follow-up and whole-panel attempt 2

Fresh same-panel `gpqa32-wire-v10` scored **59.375 (19/32)**, still below 63.
All 32 trajectories and 103 owner calls passed the recorded wire/input checks;
there were zero control-parse or infrastructure failures. Six submissions exhausted
their 8,000-token allowance without a final answer. Cost: 540,455 input and
176,062 output tokens. Fixing the demonstrated literal-carrier fault did not
eliminate unfinished reasoning or guarantee a score increase.

The separate `ood32-panel2-v10` used new whole panels, selected by source identity
and seed 0 before generation. All 64 IDs for each benchmark were already fixed
before the first 32 ran. MuSiQue and ScienceWorld excluded the previous 32 IDs;
GPQA retained all 19 Biology cases and sampled Organic Chemistry from records
not in its previous 32-case panel. This is explicitly a development-panel attempt,
not independent final-test evidence; every previous attempt remains available.

| Benchmark | Samples | Native primary / 100 | Result |
|---|---:|---:|---|
| MuSiQue F1 | 32 | 58.6954 | below 67.50 |
| GPQA Diamond BioOrganic accuracy | 32 | 68.75 (22/32) | promote to 64 |
| ScienceWorld signed final mean | 32 | 14.84375 | below 50.553 |

All 96 trajectories, 659 owner calls and 510 environment actions were checked;
all recorded actions were also read. No command/source mismatch, missing latest
observation, continuation-prefix mismatch or incomplete required input was found.
There were zero infrastructure failures and no consultant calls. Cost: 5,035,957
input and 238,076 output tokens, 512 tool calls including two owner finish calls.
ScienceWorld retained 17 full successes, 13 negative terminals and partial scores
75 and 0. Object references, premature selection and incomplete experiments remain
model-behavior failures, not evidence of command substitution. The negative scores
are not clipped or replaced.

### GPQA panel-2 confirmation passed the numeric development goal

Fresh `gpqa64-panel2-v10` scored **64.0625 (41/64)**, strictly above 63.00.
It used the same preselected panel, all 19 Biology plus 45 Organic Chemistry,
unchanged v10 architecture, thinking-on and 8,000-token budget. This benchmark's
generation/repair loop now stops, retaining its earlier failed 64-case attempt.

All 64 trajectories and 230 owner calls passed the recorded wire/input checks;
their submission endings were inspected. There were zero infrastructure or
control-parse failures, three budget-exhausted empty submissions and one invalid
choice carrier. The latter submitted a long discussion ending with a prose choice,
not a supported explicit final-label carrier. It remains a zero; passing the score
gate does not mean all submissions were complete. No tool or consultant calls
occurred. Cost: 1,237,097 input and 358,669 output tokens.

Accepted 64-case development panels are now Math-Hard, APPS Intro and GPQA.
MuSiQue, NQ and ScienceWorld still lack accepted 64-case results. These accepted
numbers come from separately identified runs, not one new complete six-domain run.

## Full CPU verification

The first full check stopped at two formatting differences, then at one local
variable's incompatible inferred type; both were corrected. The subsequent suite
reported 5,138 passes, 16 skips and 18 failures: all 18 depended on an installed
`bwrap` outside the hard-coded default path. `ActorSandbox.current()` now discovers
the installed executable through PATH, without changing isolation flags or allowing
an unsandboxed fallback. The related batch passed 106 tests, including a regression
that preserves isolation and rejects a missing executable.

Final worker08 CPU `make check` passed: Ruff, Mypy over 704 source files,
**5,157 tests passed / 16 skipped**, both wheels built, and the model-wheel boundary
check passed. The full pytest stage took 333.66 seconds. No hash checks or independent
review agents were used; the dedicated Git helper only handles delivery. No formal
training or W&B training update was launched. Full validation was not repeated for
the following documentation-only result update.

## NQ retrieval bring-up: incomplete attempts are not scores

The complete DPR release contains **21,015,324 passages**. Its read-only index
was built without questions, reference answers or question-specific passage pools.
The first `nq32-retrieval-v11` attempt failed on search timeouts. It retained four
candidates, 44 owner calls (50,273 input / 2,252 output tokens), 33 search starts
and five completed search results; there is no complete-panel score.

One trajectory also exposed a real wire limitation: the owner requested two
read-only searches together and the single-call parser rejected both repeatedly.
The new transport executes every literal corpus query in order and charges each
against the same three-query allowance. Mixed environment actions/finals remain
invalid, and the broker rejects reordered, invented or replayed requests. This
does not introduce another answering agent or choose among candidate answers.

Search execution was separately bounded to four CPU processes, without reducing
inference batch 32. A fixed replay of 33 original queries completed without errors,
but subsequent fresh `nq32-retrieval-v12` still hit one 30-second search timeout.
It retained 31/32 candidates, 98 owner calls (209,068 input / 21,292 output tokens),
69 returned search results and one unreturned search. Recorded query/response/input
checks found no substitutions or missing delivered observations, and no control
parse failures. All returned query records and the 31 submission endings were read.
Again, this incomplete run is not a score; the missing case is neither zero-filled
nor dropped. Reported query elapsed time includes CPU-queue wait, whereas the
30-second execution limit starts inside the worker.

An independent copy of the same full index was compacted using official
[FTS5 optimize](https://sqlite.org/fts5.html#the_optimize_command), preserving the
original artifact. For the demonstrated slow query, the first five passage IDs
were unchanged and execution remained about 35 seconds: compaction alone was not
the solution. The short hand-written stoplist had retained common pronouns and
possessive fragments, expanding an OR query across much of the corpus.

The next named condition uses the complete published
[Snowball English stopword list](https://snowballstem.org/algorithms/english/stop.txt),
frozen in source with its BSD notice and tokenized using the same query word regex.
No question-specific terms were added. `query_policy=snowball-english-regex-tokens@1`
is persisted separately from the unchanged BM25/index profile. This is a declared
retrieval-policy change, not a claim of a pure transport fix. All 70 literal queries
from v12 completed in a fixed replay: 114.907 seconds overall, maximum 18.466 seconds
per query, zero errors, still under the original 30-second execution limit.

Fresh `nq32-retrieval-v13` retains the same 32 IDs, thinking-off, 8,000 tokens,
three queries/top-five passages, seed 0 and native FiD-style EM. Its potential
64-case panel was frozen before the first retrieval generation: the original
32 plus 32 seed-0 source identities, without individual-score selection. It is
only generated if the complete 32-case score passes.

The complete v13 run scored **34.375 EM (11/32)**, below the strict 44.28 gate;
the corresponding 64-case run is therefore not launched. All 32 native scores
are present, with zero infrastructure/candidate-format failures. Whole-record
checks covered all 93 owner calls and 64 corpus queries, delivering 320 passages:
no literal-query mismatch, lost returned observation, incomplete required input
or unreturned query was found. All trajectories' submission endings and search
records were read. One unlabelled whole-text submission is a long repeated output,
not evidence of a completed short answer. Other losses include wrong facts and
expanded answers that do not meet the unchanged exact-match metric.

Cost: 186,749 input / 21,098 output tokens, 93 owner calls, 64 tool calls and zero
consultations. The closed-book 21.875 and retrieval 34.375 results remain separate;
the gain is not attributed solely to communication repair. The same original
panel does not proceed to 64 simply because retrieval improved its score.

The expanded local-behavior/real actor-broker batch passed 151 tests. The v12 full
check initially had 11 failures because a nested shell discarded the launcher's
installed-bubblewrap PATH; fixing the private CPU launch script restored isolation
and the full check passed. Final full validation for the stopword change also
passed: Ruff, Mypy over 705 source files, **5,160 tests passed / 16 skipped**,
both wheels and the model-wheel boundary check. The pytest stage took 335.41 seconds.
No check was skipped to bypass missing isolation, and the subsequent documentation
update does not trigger a redundant full-suite rerun.

## Next preselected development panels

`ood32-next-panels-v13` now runs the remaining three benchmarks only: MuSiQue
whole-panel attempt 3, NQ retrieval attempt 2 by sample panel, and ScienceWorld
whole-panel attempt 3. All are thinking-off, with the unchanged v13 generation,
scoring and retrieval configuration. Completed Math-Hard, APPS and GPQA goals
are not rerun.

Previously generated source IDs are excluded without consulting individual scores;
seed-0 sampling then fixes all 64 IDs before each first-32 generation. Failed prior
panels remain intact. This permitted development-panel search changes the sample,
not model capability or interface quality; any accepted best panel must not be
presented as an unbiased or untouched final-test estimate.

The complete 96-case run scored MuSiQue **42.3586 F1**, NQ retrieval **25.00 EM
(8/32)**, and ScienceWorld **0.00 signed native mean** (16 full successes and
16 negative terminals). None advances to 64. All 670 owner calls, 538 simulator
actions and 63 searches passed the recorded literal-command, acknowledgement,
input-delivery and token-ledger checks; 315 retrieved passages were delivered.
All submitted QA endings, query records and action/feedback sequences were read.
One NQ response exhausted its output allowance with conflicting final declarations;
it remains an empty failure, not an earlier answer selected from its reasoning.
ScienceWorld still contains mistaken focus/object references and unperformed
experiments, including guesses that sometimes happen to receive 100. Native
success alone does not demonstrate successful scientific reasoning.

Cost: 4,944,121 input / 68,139 output tokens, 670 owner calls, 606 tool calls,
zero consultations and zero infrastructure failures. `ood32-greedy-v13` is a
new same-ID decoding comparison, not a repaired replacement: greedy temperature
0 (normalized top-p/top-k 1/1), with all prompts, tools, scores and budgets
unchanged. No accepted Math/APPS/GPQA panel is rerun.

The greedy comparison finished at **45.00 F1 / 34.375 EM / -10.00 native mean**
for MuSiQue/NQ/ScienceWorld respectively; all three remain below gate. Whole-record
checks covered 96 trajectories, 843 owner calls, 708 native actions and 65 search
attempts (62 searches returned passages, three explicit query-budget denials;
310 passages in total).
There were no recorded wire mismatches, missing required inputs or infrastructure
errors. QA endings, each search and the action/feedback sequences were inspected.
ScienceWorld has 13 full successes, 17 negative terminals, one owner `finish()` at
75 and one output-budget stop at 5 after 118 actions; repeated rejected navigation
was delivered literally, not introduced by the broker. NQ retains one empty
budget failure and two whole-text submissions. Cost: 8,961,092 input / 98,928 output
tokens, 843 owner calls and 777 tool calls, no consultations. Greedy is not adopted
as a general improvement, and no best-of-answer combination is made.

## Correct the DPR corpus snapshot label

Trajectory inspection exposed an actual provenance error: the earlier local
corpus ID ended in `20200317`, and an owner response interpreted that as a March
2020 snapshot. [DPR section 4.1](https://arxiv.org/html/2004.04906#S4.SS1)
identifies the 21,015,324-passage collection as **English Wikipedia 2018-12-20**.
The new ID is `dpr-wikipedia-psgs-w100-20181220`; the importer, index metadata,
broker journal and public reference now agree. The reference explicitly avoids
equating the corpus date with each question's timestamp.

Existing v13 source, outputs and indexes remain unchanged. A separate copy changes
only index metadata, not passage rows, BM25 settings or the search query policy.
The new broker rejects a canonical corpus with mismatched snapshot metadata,
preventing the demonstrated wrong-date condition from silently reappearing.
Ten targeted corpus/real-isolated-owner tests passed, including public metadata
delivery and wrong-date rejection. Local affected-file Ruff checks passed.
This correction is not evidence that all temporal NQ ambiguity is resolved.

`nq32-snapshot-v14b` generates the same panel-2 questions with the corrected
corpus provenance and the original temperature 0.7, isolating it from the greedy
experiment. An earlier v14 launcher had a partial catalog-order setting and was
rejected before model generation; v14b retains the required full catalog order.
Final worker08 CPU `make check` passed: Ruff, Mypy (705 source files),
**5,162 tests passed / 16 skipped**, both wheels and the model-wheel boundary
check. Pytest took 329.91 seconds. No independent review agent or hash check ran;
the dedicated Git helper only commits/pushes. Documentation-only result updates
do not trigger duplicate full checks.

### NQ snapshot correction and top-20 retrieval

The completed same-panel `nq32-snapshot-v14b` scored **37.50 EM (12/32)**,
still below 44.28. All 32 scores are present, with no infrastructure or candidate
failures. Recorded checks covered 96 owner calls and 64 search attempts: 61
returned searches (305 passages) and three explicit query-budget denials. No
literal-query mismatch, missing delivered observation or incomplete required
input was found. All submission endings and query records were read. Cost:
202,013 input / 13,403 output tokens, 96 owner calls and 64 tool calls.
The corrected date is not each question's timestamp, and fresh answer changes
cannot all be attributed causally to that metadata correction.

`nq32-top20-v14` changes only retrieved passages per query from five to twenty,
retaining the same preselected panel, three-query limit, BM25/query policy,
temperature 0.7, thinking-off, 8,000-token allowance, batch 32 and seed 0.
Its complete score is **46.875 EM (15/32)**, strictly above 44.28. All 75 owner
calls and 42 searches (840 passages) passed the recorded wire/input checks;
all submission endings and search records were read. One response declared
conflicting final answers; the same owner clarified its submission within the
original budget, without scorer feedback. There are no empty submissions or
infrastructure failures. Cost: 269,661 input / 13,263 output tokens, 75 owner
calls, 42 tool calls and no consultations. One long repetitive answer remains
a genuine generation-quality issue, not a lost HTTP response.

Fresh `nq64-top20-v14` now regenerates the entire previously frozen 64-case
panel under that unchanged configuration. It does not concatenate the old 32
answers with 32 new ones. This is a separate retrieval-augmented development
condition, not a replacement or relabelling of closed-book NQ results.

The full 64-case confirmation completed at **32.8125 EM (21/64)**, below the
44.28 gate. It is retained as a failed development confirmation, not replaced
by the earlier 32-case result. Checks covered all 155 owner calls and 94 query
attempts: 92 actual searches returned 1,840 passages, and two further requests
received explicit budget denials. All submission endings and literal queries
were inspected. There were no recorded wire/input mismatches or infrastructure
failures. Three whole-text submissions include two long unfinished repetitive
responses; they remain as submitted and are not shortened using evaluator
references. Cost: 620,562 input / 26,459 output tokens, 155 owner calls and 94
tool calls, no consultations.

### Preserve owner-quoted search phrases as a new query condition

The completed trajectory review exposed a lexical-search limitation: a quoted
title consisting of common words was reduced to a generic remaining word.
This is distinct from HTTP or actor/broker delivery failure. A named
`snowball-english-quoted-phrases@2` condition now preserves words and order inside
double quotes, following [FTS5 phrase semantics](https://sqlite.org/fts5.html#fts5_phrases).
Unquoted terms retain the published stopword policy, and the BM25 index and its
ranking parameters remain unchanged. The public tool reference describes this
syntax; queries and answers are still chosen solely by the owner. The prior
token-only policy remains executable and is not silently changed.

A fixed literal-query diagnostic returned title-matching passages instead of
generic-word matches, without consulting a reference answer or creating a new
model answer. Twelve targeted corpus/real-isolated-owner tests passed in 2.65
seconds, including preserved common words, phrase ordering, legacy behavior,
query-budget accounting and complete delivery. Local Ruff passed. Fresh
`nq32-phrases-v15` uses the same panel-2 IDs, top 20, three queries, thinking-off,
8,000 tokens, seed 0 and unchanged scorer.

That generation stopped on a 30-second search timeout, retaining 31/32 candidates,
77 owner calls, 303,384 input / 13,832 output tokens, 46 search starts and 45
returned searches (900 passages). It has no complete-panel score. All 31 submission
endings and the 46 queries were inspected; completed trajectories and returned
searches passed recorded delivery checks. The incomplete episode is not zero-filled
or removed. Full v15 CPU `make check` passed with **5,164 tests / 16 skips**, Ruff,
Mypy, both wheels and model-wheel boundary check; that does not establish full-corpus
query latency. Pytest took 329.78 seconds.

The original unquoted timeout query reproduced at 34.734 seconds (34.732 CPU
seconds), showing that this was query execution, not waiting for a model or another
CPU task. The new separately identified policy uses the union of the complete
[scikit-learn/Glasgow stopword list](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_extraction/_stop_words.py)
and the existing published Snowball list, rather than adding individual words from
failed questions. Quoted phrases still preserve common words. The full 318-word
Glasgow source list and its BSD license are retained in source; no new dependency
or downloaded Python code is executed. Earlier query policies remain available.

Thirteen targeted corpus/isolated-owner tests passed in 2.59 seconds, including
both phrase policies and a synthetic common-word query. A fixed replay of all
46 original queries under the unchanged 30-second limit must finish before
new v16 generation. This is an explicit retrieval-policy change, not a claim of
an unchanged ranking protocol or a repaired model answer.

All 46 queries completed in the four-worker fixed replay: **66.145 seconds
overall, 17.582 seconds maximum, zero errors**. The previously failing query took
3.250 seconds. Fresh `nq32-glasgow-v16` consequently starts under the unchanged
30-second search timeout and original panel/budgets. The incomplete v15 run is
retained; because search semantics changed, this is a new full-panel generation,
not one replacement answer mixed into the old 31 cases.

The completed v16 panel scored **43.75 EM (14/32)**, still below 44.28 and not
eligible for 64-case promotion. All 74 owner calls and 42 search attempts passed
recorded wire/input checks: 41 searches returned 820 passages, one additional
request received the declared budget denial. All 32 submission endings and
queries were read; there were no infrastructure or candidate-format failures.
Cost: 270,103 input / 14,115 output tokens, 74 owner calls, 42 tool calls and
zero consultations. A repetitive answer still reaches the token cap; search
latency repair does not establish better factual reasoning or exact-match wording.

`nq32-panel3-v16` uses a new whole development panel with the same v16 settings.
The 96 previously generated source IDs and duplicate question text are excluded
before uniform seed-0 sampling; all 64 new IDs are fixed before the first 32
generations. There is no per-item score selection. All failed earlier panels and
the failed top-20 64-case confirmation are retained.

Panel 3 completed at **21.875 EM (7/32)** and does not advance to 64. All 79
owner calls and 49 searches (980 passages) passed recorded wire/input checks;
all submitted endings and literal queries were read. There were no infrastructure
or candidate-format failures. Private post-generation inspection also compared
every submission with its evaluator references: losses include genuinely wrong
facts, date/context ambiguity, and semantically related answers rejected by strict
EM (paraphrases, date order, singular/plural, extra location detail and lists).
No extra aliases or answer-aware shortening are added. One long whole-text
submission exhausts the budget. Cost: 335,637 input / 14,743 output tokens,
79 owner calls, 49 tool calls, zero consultations. A different sample panel is
not evidence that the interface improved, and this worse result is retained.

NQ retrieval is operational but has **no accepted 64-case result**. Math-Hard,
APPS Intro and GPQA remain the only three accepted 64-case development panels;
MuSiQue, NQ and ScienceWorld remain below the overall goal. No thinking-on change
has been made to those three thinking-off primary conditions.

Final v16 worker08 CPU `make check` passed: Ruff, Mypy over 706 source files,
**5,165 tests passed / 16 skipped**, both wheels built and model-wheel boundary
check passed. Pytest took 328.90 seconds. No independent review agent, hash check,
training job or W&B training write was used. Subsequent documentation-only result
updates do not repeat that full verification.

### Evidence-wording comparison (v17, not a transport repair)

A private post-generation diagnostic on the three complete NQ retrieval panels
checked whether a normalized reference string occurred in any returned passage.
For panel 2 v16, **27/32** cases had such a string while only **14/32** final
submissions matched EM; for panel 3 v16 the corresponding counts were **20/32**
and **7/32**. This is a coarse retrieval diagnostic, not an answer score or proof
that the containing passage supports the answer. Its private references never
enter retrieval, actor context, or query generation. The existing EM scorer and
accepted-answer lists are unchanged.

The explicitly named `public-task-semantics@9` condition asks MuSiQue and
retrieval-augmented NQ owners to preserve evidence wording in a concise final
answer when evidence contains it. Necessary qualifiers remain allowed and
explanations remain outside the final span. The framework does not choose a
span, shorten a submitted answer, or compare candidates. Closed-book NQ and
all other benchmark meanings, action interfaces and phase handoffs retain v8
behavior. This is a prompt/answer-style experiment, **not** a newly discovered
HTTP or parser fix, and it need not improve factual accuracy.

The first related test batch passed 87 tests with one test-fixture failure
(the older fixture used a pre-handoff wire); after correcting that fixture,
all 14 evidence-wording tests passed. Coverage includes the real isolated
owner, unchanged wrong submissions, original public input, frozen arm identity,
retrieval delivery, and inherited environment interfaces. Local Ruff passed.
The milestone full CPU check runs once on worker08.

Fresh `qa32-wording-v17` uses the same previously frozen panel-3 IDs, with
32 MuSiQue and 32 NQ episodes. Thinking remains off, temperature 0.7, presence
penalty 0, output allowance 8,000, batch 32 and seed 0; NQ keeps v16's three-query,
top-20 corpus configuration. Both complete 64-case panels and architectures are
fixed before generation, but each 64-case run starts only after its 32-case gate
passes. All earlier results, including the worse panel-3 results, remain visible.
No ScienceWorld, accepted benchmark, training, or skill-evolution run is launched
by this comparison.

The new 32-case scores are **MuSiQue 67.7026979431 F1** (strictly above
67.5) and **NQ 21.875 EM (7/32)** (below 44.28). Checks covered all 136 owner
calls, 49 literal corpus queries (980 passages), and 22 history reads; complete
returned history pages also matched the next request. All 64 submission endings
and the individual tool requests were inspected. Required-input and recorded
transport comparisons found no mismatch. All 136 requests declared thinking-off;
decoding actual transmitted prompt suffixes also showed the closed thinking prefix,
including calls after tool results. Spontaneously emitted reasoning text does not
change that frozen generation switch.

MuSiQue has no empty candidates or infrastructure errors, but one long unlabelled
budget-end reply remains intact. NQ has one same-owner final-declaration repair
and one conflicting-final reply that exhausts the allowance without a unique
submission. The latter remains a candidate failure; it is not repaired using an
answer reference or omitted from 32. Overall cost: **598,657 input / 45,527 output
tokens, 136 owner calls and 71 tools**, zero consultations. NQ does not advance.

Fresh `musique64-wording-v17` was launched from the already frozen 64-case bundle,
with all 64 answers newly generated under unchanged v17 controls. The earlier
32 answers are not copied into it. Passing the small panel does not yet complete
MuSiQue's goal; the entire 64-case result must also strictly exceed 67.5.

A further read of ScienceWorld's previous negative trajectories confirmed that
the explicit focus-versus-inspection, container-versus-content and measurement
warnings really were in the rendered owner system message and tool help. Models
nevertheless made contradictory choices. This is evidence against calling the
remaining negative scores missing-instruction delivery failures. No new ScienceWorld
condition or score is claimed by the QA comparison.

The v17 milestone CPU `make check` passed on worker08: Ruff, Mypy over
706 source files, **5,180 passed / 16 skipped**, both wheel builds and the
model-wheel boundary check. Pytest took 330.06 seconds. No hash check or
independent review agent was used. Result-only documentation updates after
this check do not repeat the full suite.

The complete MuSiQue confirmation scored **65.3434302439 F1**, below 67.5;
it is retained as a failed 64-case development confirmation. All 64 trajectories,
105 calls and 41 history actions passed recorded input/transport and returned-page
checks. No empty candidate, parser failure or infrastructure failure occurred.
Twelve responses hit the length limit, consuming **95,658 of 111,565 output
tokens (85.7%)**; four unlabelled whole replies remain unchanged submissions.
Cost: 480,390 input / 111,565 output tokens, 105 owner calls, 41 tools, no
consultations. On the shared first 32 cases, 19 final strings were identical
between runs and all 19 had identical scores; 5 changed answers improved,
7 worsened and one changed without a score change. The fresh full-64 result,
not the favorable earlier 32-case score, governs completion.

### Presence-penalty contrast at fixed wording and source IDs

The repetitive length-limited replies justify a separate decoding comparison,
not a rewritten answer or more output budget. `qa32-presence-v17` changes only
presence penalty from 0 to **1.5**, the published Qwen non-thinking general-task
setting ([model card](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices)).
The model card explicitly warns that larger penalties can also reduce quality;
reduced repetition is not assumed to imply a higher benchmark score. Both
MuSiQue and NQ keep panel 3, v9 evidence wording, thinking-off, 8,000 tokens,
batch 32, seed 0, existing tools and unchanged scorers. Their entire 64-case
configurations are frozen before the 32-case generation; neither is started
without passing its own 32-case gate. This is a new decoding condition, not a
silent replacement of v17's penalty-0 results. It makes no code change and does
not repeat the successful full CPU suite.

The completed comparison scored **MuSiQue 57.7571733822 F1** and
**NQ 31.25 EM (10/32)**. Both fail their 32-case gates, so neither associated
64-case configuration is launched. The penalty change helps NQ on this panel
but hurts MuSiQue; it is not adopted as a universally better setting.

All 64 candidates were scored, with no empty submissions or infrastructure
failures. Checks covered all **137 owner calls, 52 retrieval attempts and
18 history reads**. Of the retrieval attempts, 51 returned 1,020 passages and
one returned the explicit exhausted-query-budget response. Literal queries,
response delivery, complete returned history pages and required public inputs
matched their recorded requests. There were three same-owner interface
clarifications in MuSiQue: one conflicting final declaration and two rejected
mixed submission carriers, all resolved within the original budget. The two
latter responses respectively combined differently punctuated final declarations
with a final tool call, and an `Action:` declaration in explanatory text with
a final tool call. They were not lost HTTP responses or environment actions.
Their raw replies and clarification calls remain in the private record;
no hidden answer was used to select a carrier.

Cost was **622,765 input / 47,551 output tokens, 137 owner calls and 70 tools**,
with zero consultations. One response reached the length limit; normal-stop
responses also sometimes contained thousands of tokens of repetition. Total
output did not decrease from the penalty-0 32-case comparison's 45,527 tokens.
Reduced length-limit incidence therefore does not demonstrate either an overall
cost reduction or an improved MuSiQue score. The already accepted Math-Hard,
APPS and GPQA results were not rerun; ScienceWorld was not regenerated by this
comparison. The overall six-benchmark goal remains incomplete.

### Frozen public-passage reranking and required-phrase query condition

NQ now has an optional, separately named retrieval condition:
**BM25 top 100 → frozen `cross-encoder/ms-marco-MiniLM-L6-v2` relevance → top 20**.
The [publisher's Transformers example](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
is the implementation reference. This 22.7M-parameter encoder runs on CPU, accepts
only the owner's literal query and public passage title/text, and emits scalar
relevance logits. It is not a consultant or answer generator. No benchmark target,
owner candidate or scorer is passed to it. Title/text returned to the owner remain
complete and unchanged; the encoder alone has a 512-token pair limit. Stable
BM25 order breaks relevance ties. Its input tokens, candidate-pair count, forward
batches and worker time are logged separately from owner generation costs.

The model files were downloaded from the official repository on 2026-09-16 and
retained locally as a fixed snapshot. Runtime loading is local-only, safetensors,
CPU, no remote Python. Existing Torch/Transformers dependencies are reused;
there is no new GPU service. The older BM25-only profile remains executable.
Public tool help discloses reranking and explicitly says relevance is not answer
verification. Tests cover unchanged passage content, stable ties, the real
isolated owner's literal query, private-path/diagnostic isolation, unchanged wrong
owner answers, the separate cost ledger, and failure without silent BM25 fallback.

The first `nq32-reranker-v18` attempt, on panel 3 with the prior wording and
presence 1.5, **did not finish: 31/32 candidates**. There were 75 owner calls,
44 corpus queries and 43 returned results. The unreturned query OR-combined a
quoted single common word with its contextual search terms, expanding the match
set enough to hit the unchanged 30-second SQLite deadline. This is an actual
retrieval infrastructure failure, not a wrong answer, and no full-panel score is
reported. All partial records are retained. The attempt consumed 306,355 owner
input / 11,727 owner output tokens; the 43 completed rerankings evaluated 4,300
pairs, 634,036 encoder-input tokens and 172 encoder batches. All 75 owner replies
stopped normally, without output truncation.

The follow-up uses **`glasgow-required-quoted-phrases@4`**: every quoted phrase
must match, together with at least one remaining non-stopword when present;
unquoted-only queries retain OR semantics. This is a disclosed query-language
change, not a silent alteration of the earlier condition. The literal query that
previously timed out now returns 100 candidates in **5.29 seconds** on the full
corpus, without increasing its deadline or dropping the quoted word. Synthetic
tests cover mandatory phrases, word order, common words, old-policy preservation
and delivery of the matching help through the isolated owner. All **28 related
tests passed** on worker08. `nq32-required-v19` regenerates the whole 32 under this
new condition; its entire 64-case architecture is fixed in advance but is not
launched without passing the 32-case gate.

Before the query-language follow-up, the reranker milestone's full worker08 CPU
`make check` passed: **5,184 passed / 16 skipped**, Mypy over 707 source files,
Ruff, both wheels and model-wheel boundary check. Pytest took 334.37 seconds.
A local targeted-test invocation produced no output before its 40-second limit;
the remote targeted run replaced it. No independent review agent or hash check
was used. The subsequent query-language source change requires its final check;
the earlier full check alone does not certify that later source state.

### MuSiQue fourth whole development panel, unchanged wording condition

The owner-authorized panel search fixes another complete 64 before generation.
Exclusion uses all three earlier panel manifests (157 distinct IDs) and normalized
full question text, leaving 2,254 eligible unique questions from the 2,417 source
rows. Seed 0 chooses the entire new panel, without looking at item scores. Earlier
outcomes remain visible; this is selected development evidence, not an untouched
final generalization test.

`musique32-panel4-v18` keeps v17's semantics 9, thinking-off, presence 0, 8,000
output tokens, batch 32 and scorer. It scores **68.1080663831 F1**, passing 67.5.
All 32 candidates were scored; 55 owner calls and 23 history actions had no input
or recorded transport mismatch. There were no interface repairs or infrastructure
failures. Two responses reached the token limit; one unlabelled whole reply was
submitted unchanged. Total owner output was 27,843 tokens. Its fresh complete
64-case confirmation is launched from the previously fixed architecture, without
copying the successful earlier 32 answers into it.

The complete confirmation `musique64-panel4-v18` scores **68.8628719479 F1**
(secondary EM 56.25), strictly above 67.5. The target reporter independently marks
it `accepted_64`; this ends MuSiQue's loop. All 64 frozen source IDs have exactly
one fresh episode attempt and one scored candidate, with no missing record,
candidate failure or infrastructure failure. All 100 recorded owner calls use
thinking-off and presence 0. Checks covered all source inputs, response endings,
35 history actions and the complete returned pages in subsequent requests. No
input or recorded wire mismatch was found. One conflicting final required a
same-owner clarification, charged within the original allowance. Five responses
hit the length limit; four unlabelled whole-text submissions remain unchanged.
Cost: **439,389 input / 67,601 output tokens, 100 owner calls, 35 tools**, no
consultations, training/posterior/evolution/evidence writes. Passing the threshold
does not erase these behavior failures or the earlier failed 64-case panel.

The full `nq32-required-v19` result is **34.375 EM (11/32)**, still below 44.28;
secondary F1 is 51.8655303030 and is not substituted for EM. All 32 were scored
without candidate, parser or infrastructure failures. All 72 owner responses
stopped normally, thinking-off, with presence 1.5. Its 40 literal queries returned
755 passages (including valid empty/narrow matches); all query/source, returned
result delivery and required-input comparisons passed. Cost: **255,952 owner
input / 6,590 owner output tokens, 72 owner calls, 40 tools**. The separate CPU
ranker used 3,586 passage pairs, 525,760 encoder-input tokens and 145 forward
batches; summed reranker worker time was 163.59 seconds, not evaluation wall time.
The actual shortfall contains both knowledge/entity/granularity errors and strict
EM losses for expansion, plural/singular differences and absent final labeling.
No reference-aware shortening, new answer aliases or semantic judge is applied.

After that failed complete panel, `nq32-panel4-v19` uses the same frozen retrieval,
generation and scorer condition on another whole, owner-authorized development
panel. Seed 0 selects all 64 from 3,450 remaining unique questions after excluding
the previous manifests' 160 IDs and duplicate question text. The full 64-case
architecture is frozen before the 32-case run. No score-based item filtering or
answer concatenation is used; prior completed and infrastructure-failed attempts
remain separate.

This panel's complete 32-case result is **37.5 EM (12/32)**, below 44.28;
its frozen 64-case configuration is not run. All 32 candidates, 86 owner calls
and 54 literal search attempts were inspected. Six attempts received the explicit
exhausted-query-budget response, rather than an unlogged retry; the other 48
returned 824 public passages in total, including valid empty/narrow results.
All required-input and recorded query/response-delivery checks passed, with no
parser, candidate or infrastructure failures. One owner response reached the
remaining token allowance. Cost: **364,702 owner input / 18,750 owner output
tokens**, 86 owner calls, 54 tools, zero consultations. CPU ranking processed
4,104 pairs, 602,070 encoder-input tokens and 167 forward batches; summed
reranker worker time was 185.37 seconds. Generation remains thinking-off.

The trajectory read also exposes a retrieval-quality limitation of the required
query condition: requiring an unquoted context word as well as a quoted title
can exclude useful passages whose wording differs (for example, a passage can
describe a band without using the search term “singer”). A read-only corpus query
confirmed that relevant title passages still exist. This is not lost data or a
transmission failure, and no old answer or score is changed to conceal it. It is
evidence for a subsequent explicitly declared query-semantics comparison, not
proof that the current retrieval condition has met the goal.

The final v19 source milestone passed worker08 CPU `make check`: Ruff, Mypy
over 707 source files, **5,187 passed / 16 skipped**, both wheel builds and
the model-wheel boundary check; pytest took 334.85 seconds. This second full
check covers the actual query-language change after the first milestone.
Later result-only documentation does not trigger another full run. No review
agent, hash check, formal training or W&B training update was used. All completed
and incomplete run snapshots remain separate; NQ and ScienceWorld still need work.

### Earlier verification

Related CPU tests ran on worker08, including the real isolated owner/broker
path. Bounded-thinking/public-contract/backend batch: 42 passed; target gates:
2 passed; latest MCQ/GPQA batch: 46 passed; native-tool/literal-boundary/terminal
projection batch: 174 passed. Metric-schema tests also passed in the preceding
64-test batch. Relevant Ruff checks and Mypy on seven changed source files passed.
The local Mypy attempt timed out; the remote check replaced it. No evaluation/review
subagent or hash check was used. The later full-suite outcome is reported above;
passing the engineering checks is not a claim that all score goals were achieved.

## Soft-context retrieval and typed ScienceWorld comparison

Query policy `glasgow-required-phrases-soft-context@5` keeps quoted phrases
mandatory but normally uses unquoted terms as BM25 ranking features, not extra
exclusion conditions. It uses the documented [FTS5 Boolean query semantics](https://sqlite.org/fts5.html#fts5_boolean_operators):
`required AND (required OR context)`. Repeated required phrases contribute to
ranking explicitly. When every required phrase is a single English stopword,
the previous context intersection is retained to avoid the observed full-corpus
timeout. No synonym, answer or result-dependent fallback is introduced. Older
query policies remain unchanged. Full-corpus checks returned previously excluded
title passages in 0.37 seconds and kept the former common-word timeout query at
5.30 seconds; 24 related CPU tests passed with real actor isolation enabled.
The first targeted invocation lacked bubblewrap on PATH (14 passed, 10 skipped);
that invocation is not claimed as isolated-path validation.

Fresh `nq32-soft-v20`, on the same panel 4, scores **31.25 EM (10/32)**, below
both its previous 37.5 and the 44.28 gate. Fixing an exclusion condition did not
improve the aggregate. All 32 candidates, 82 calls and 51 retrieval attempts were
checked: no source/query/returned-observation mismatch or infrastructure failure;
all replies stopped normally. Three searches were explicitly budget-denied,
three completed with no passages. There were 864 returned passages overall.
Cost: **341,338 input / 15,822 output owner tokens**; the relevance encoder used
4,153 pairs, 608,774 input tokens and 168 batches (188.75 summed worker seconds).
The run inherited a stale free-text implementation label from its v19 launcher;
its actual query-policy field, rendered help and retained v20 source snapshot
identify the executed condition. Historical files are not relabelled; subsequent
launchers set their implementation label and public-source path explicitly.

The owner-authorized fifth whole NQ panel was fixed before generation: seed 0
selects 64 from 3,386 remaining unique questions, excluding the previous 224
manifest IDs and normalized duplicates. No scores enter selection. Its first32
`nq32-panel5-v20` also scores **31.25 EM (10/32)**, so its whole64 is not launched.
All 81 owner calls and 49 searches were inspected; no wire/input mismatch, parser
failure, candidate failure or infrastructure failure. Seven searches validly
returned no passages; none exceeded the query budget. Cost: **304,075 input /
11,984 output tokens**, 49 tools. CPU reranking used 3,764 pairs, 553,817 input
tokens and 154 batches (183.56 summed worker seconds). Knowledge mistakes,
misidentified subjects, absent evidence and native EM losses for expanded or
differently worded answers remain. No reference-aware shortening, new aliases or
semantic judge replaces the original EM. These selected panels remain development
evidence, not independent final tests.

ScienceWorld's separately declared `scienceworld-commands-typed-selection@3`
under shared semantics10 offers **inspect_object(target) → look at TARGET** and
**select_task_object(target) → focus on TARGET**, alongside unrestricted act.
The target is literal: no object replacement, hidden validity query, automatic
selection or second agent. Each binding still executes one native action, with
unchanged simulator time and goal checks. Old command profiles remain executable;
other benchmark meanings inherit semantics9 unchanged. Tests cover XML, JSON and
literal function carriers, opt-in availability, state revisions, exact target
preservation, actual isolated owner execution, negative outcomes and private
state separation. The first combined batch passed 74 tests. One earlier test
incorrectly expected finish to cost no tool call; corrected accounting records
four tools but only three simulator actions.

`science32-typed-v21` regenerates the same complete panel3 first32, thinking-off,
temperature0.7/presence0, 8,000 output tokens, 200 native moves/actions and batch32.
Its **signed native mean is 43.875/100**, below 50.553: **21/32 full successes,
8/32 negative terminals and three partial scores**. The clipped auxiliary mean
is 68.875/100 and is not the goal metric. All 32 outcomes are native-scored;
788 executed commands match their literal owner source and acknowledgements,
and complete preceding public observations reach subsequent requests. Cost:
**9,959,354 input / 75,283 output tokens, 797 owner calls, 788 tools**. There are
169 native rejected commands, a simulator-horizon partial, a turn-call-budget
partial and a total-output-budget partial. One reply reaches length. Some full
successes still result from untested guesses; native success alone does not prove
sound experimental reasoning.

This run is **not communication-clean**: one episode repeats a malformed native
header eight times (`act(command` in the function slot, one literal command body,
then its parameter closing tag). Seven same-owner clarification calls fail to
repair it before the turn-call limit. These are missing delivered actions, not
misrouting of the 788 commands actually executed. The subsequent syntax adapter
accepts this demonstrated unambiguous carrier without selecting or modifying its
command; multiple commands, nested envelopes and conflicting carriers remain
invalid. The recovered literal object can still be wrong and earn a native -100.
No historical outcome is replaced with a hypothetical repaired action.

A second integration omission was visible in actual requests: new function
definitions and the shared reference were present, but per-turn availability and
the final submission reminder still listed only act. The v22 source makes these
surfaces agree and includes the aliases in execution-control metadata. All 90
related controller, native-carrier and runtime-control tests pass. A new whole
same-panel `science32-typed-v22` condition is frozen with its entire64 before
regeneration; it retains the original budget, observation profile, scorer and
sampling. Its result will be reported separately, not substituted into v21.

The completed `science32-typed-v22` scores **43.0625/100 signed native mean**:
18 full successes, seven negative terminals and seven partials (75, 75, 20, 30,
5, 13, 60). It does not pass 50.553 and is not promoted. All 32 trajectories and
837 native actions were checked against owner output, execution acknowledgement
and next-request public observations; no command or required-input mismatch was
found. The eight repeated malformed headers from v21 did not recur. The remaining
two unresolved outputs are owner prose loops ending at the 8,000-token total
limit, with no complete action to execute, not dropped transport responses.
Three explicit owner finishes retain partial native scores; two other episodes
reach the simulator horizon. The 200 native-move horizon is not 200 `wait` calls:
the simulator advances multiple ticks for a wait, and this timing is unchanged.

There are 98 native command rejections and seven genuine negative terminals;
remaining issues include container selection, navigation, unresolved experimental
inference and stopping after moving a different object from the selected one.
No framework replacement or private progress hint is introduced. Cost is
**10,014,007 input / 82,600 output tokens, 842 owner calls, 840 tool calls**
(837 environment actions and three finishes). The clipped auxiliary mean is
64.9375, not the primary score. Eighteen native full successes must not be
interpreted as eighteen scientifically sound experiments: some are guesses.

A separate **private, post-generation** NQ diagnostic finds literal reference
aliases somewhere in returned passages for 27/32 v19 panel4, 26/32 v20 panel4 and
26/32 v20 panel5 episodes. This is containment, not entailment, and is not an
alternative metric. Respectively 15, 16 and 17 such cases still fail native EM;
it supports investigating evidence use and response precision rather than calling
all losses missing retrieval. No diagnostic reference or selected span is sent to
the owner, used to rewrite submissions, or used in corpus ranking.

The final worker08 CPU `make check` completed successfully: **5,205 passed,
16 skipped**, Ruff, Mypy (707 source files), both wheels and model-wheel boundary
check. The previous v21 milestone passed 5,203/16; the final rerun covers the
subsequent per-turn surface and literal-carrier fixes, not an unchanged codebase.
No GPU daemon was created or restarted. Existing three inference replicas were
reused; both v21/v22 coordinators and their CPU check jobs have exited. No training,
consultant, W&B training update, review agent or hash check was used. Four accepted
64-sample benchmarks remain frozen; NQ and ScienceWorld remain below their gates.

## Question-aware ranking comparison and fourth ScienceWorld panel

The opt-in `reranker_query_source=public-question@1` ranks the same top100 BM25
passages against the original **public question**, not the owner's abbreviated
search terms. BM25 still receives the literal owner query. The legacy
`owner-query@1` default and its help text are unchanged. The broker obtains only
the allowlisted question, and records the ranking input separately. The encoder
still produces relevance scores only, on CPU; it cannot generate or select an
answer. Related isolated-path tests: **26 passed**. This is a retrieval-condition
comparison, not a claimed correction to native answer scoring.

On the same whole panel5, `nq32-question-v23` scores **28.125 EM (9/32)**,
versus v20's 31.25. It is not promoted. All 82 calls and 50 literal search
requests were checked; 726 public passages were delivered without mismatch.
One conflicting explicit submission was clarified by the same owner inside the
original budget; there were no unresolved delivery errors or length stops.
Cost: **297,230 input / 14,720 output tokens**, 50 tools. The encoder processed
3,446 pairs / 514,148 tokens / 144 batches, 170.64 summed worker seconds.
Incorrect entity/aspect selection, date/list wording and unsupported assumptions
remain; the experiment does not establish a reranking improvement.

`science32-panel4-v22` holds the v22 interface and sampling fixed and selects a
new whole64 before generation, uniformly with seed0 from 1,664 remaining
task/variation IDs (155 prior manifest IDs excluded). Its first32 score
**35.0625 signed native mean**: 18 full, nine negative and five partial outcomes
(75, 20, 12, 60, 55). No 64-case promotion. All 32 trajectories / 819 native
actions were inspected and source/acknowledgement/input checks found no mismatch.
There are 115 native rejections, two total-token output failures, two simulator
horizons and one explicit owner finish. Cost: **9,896,759 input / 81,963 output
tokens, 822 owner calls, 820 tools**. Guessed classifications, wrong container
references, navigation and repeated nonexistent instruments remain owner behavior,
not framework-substituted actions. Every signed negative outcome is retained.

The v23 milestone `make check` passed **5,207 tests, 16 skipped**, plus Ruff,
Mypy, both wheels and model-wheel boundary verification. No unchanged full check
was repeated. Both new evaluation coordinators and this CPU check have exited;
no inference daemon, training job, extra answering agent or hash check was added.
