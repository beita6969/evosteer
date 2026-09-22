# OOD Step-0 follow-up, 2026-09-16

Continuation of [the development log](ood-step0-development-20260916.md).
The five accepted 64-case Math-Hard, APPS, GPQA, MuSiQue and **NQ-supervised
DPR** conditions remain frozen; ScienceWorld has not met its strict >90%-target
gate. NQ's closed-book results are not replaced or called zero-shot retrieval.
These repeatedly inspected panels are development evidence, not an unseen test.

## Whole-panel failures retained

| Condition, 32 cases | Primary score | Gate | Outcome |
| --- | ---: | ---: | --- |
| NQ retrieval `nq32-panel6-v20` | 40.625 EM (13/32) | >44.28 | No 64-case promotion |
| ScienceWorld `science32-panel5-v22` | -0.65625 signed native mean | >50.553 | No 64-case promotion |

NQ's whole64 was selected before generation with seed0 from 3,322 remaining
unique questions, excluding 288 prior full-panel IDs and normalized questions.
The condition retains query-policy5, owner-query MiniLM reranking, temperature0.7,
presence1.5, thinking-off, 8,000 output tokens and batch32. There are **73 owner
calls, 42 search attempts and 737 returned passages**, costing **268,825 input /
15,893 output tokens**. All queries, submitted endings, call receipts and input
ledgers were examined. One owner spends its remaining 7,967 tokens repeating and
changing explicit final declarations, ending at length with a conflicting partial
final. It remains an empty failure; the framework does not choose a convenient
earlier answer. One excess query is explicitly denied; missing source evidence,
incorrect entities/aspects and strict EM wording are not relabelled as wire bugs.

ScienceWorld retains temperature0.7/**presence0**, thinking-off, public-state@1,
command profile3/semantics10, 8,000 tokens and the 200 native horizon/actions.
Its 32 outcomes are **10 full successes, 14 negative terminals and eight partials**
(75, 38, 13, 13, 60, 60, 60, 60). The clipped auxiliary mean is **43.09375**;
it is not the primary score. All **925 executed actions** were checked in order
against owner commands, acknowledgements and the next public observation. No
route mismatch or missing required public input was found. There are 106 native
rejections, six length-stopped replies, one explicit owner finish and one native
simulator-horizon ending. Cost: **934 owner calls / 926 tools, 12,832,080 input /
93,731 output tokens**. The 14 negatives follow 12 focus commands and two numeric
focus confirmations. Wrong container/object selections and unfinished experiments
remain genuine behavior failures, including guesses that sometimes score 100.

The final caps of 1, 9 and 57 output tokens match the remaining episode ledger;
they are not HTTP truncations or a newly lowered overall budget. Actual requests
include both the local eight-call decision allowance and the distinct remaining
episode call/token allowance. No private progress or reference-driven correction
is supplied to the owner.

## Literal wait carrier, not a replacement action

One eventually successful episode issued the complete zero-argument native
command in a `wait` function envelope twice. The old controller rejected the name;
the same owner later emitted `act(command="wait")` and completed the task. Source24
accepts that demonstrated literal carrier as exactly the same native `wait` on a
ScienceWorld act surface. It does not select a duration, infer a command from
prose, finish an incomplete envelope, repair targets or bypass native ticking.
Wrong arguments, extra commands, another environment and stale state remain
rejected. This is not an explanation for the 14 genuine negative terminals.

Related CPU tests on worker08: **118 passed**, including actual isolated-owner
execution and accounting. Local Ruff check/format check passed. The full worker08
CPU `make check` passed **5,217 tests / 16 skipped** (340.02 seconds for pytest),
Ruff, Mypy over 707 source files, both wheels and model-wheel boundary verification.
Documentation-only updates did not repeat this unchanged-code full check.
No independent review/answering agent, hash check, training or W&B update is used.

## Next declared development panels

After this review, NQ whole-panel attempt7 and ScienceWorld attempt6 each freeze
all64 before first32 generation, using seed0 after prior source-ID exclusion
(and normalized-question exclusion for NQ). No per-item scores select the IDs;
no candidates are spliced or replaced. NQ keeps source20's retrieval condition.
ScienceWorld uses source24's literal wait transport with otherwise unchanged
budget, interface, sampling and scorer. A different panel's score cannot establish
that the parser repair improved model capability. No 64-case run is authorized by
a failing first32, and no accepted benchmark is regenerated.

## Dense-retrieval preparation only

The official [DPR question encoder](https://huggingface.co/facebook/dpr-question_encoder-single-nq-base)
and [compressed Wikipedia index](https://huggingface.co/datasets/facebook/wiki_dpr/tree/main/index)
were downloaded through the local connection, together with the official CPU
Faiss wheel. The approximately 3.33 GB acquisition completed; no dense retriever
has been installed, connected to an owner or used for a benchmark generation.
This encoder was supervised on NQ training examples. Permission for that separately
labelled condition was requested; it must not be described as strictly zero-shot
OOD, or confused with updates to this project's policy/posterior/library.
Read-only parquet probes at the first, middle and final shards found sequential
IDs locally, ending at 21,015,300 versus 21,015,324 rows in the existing corpus.
This is not yet full index-row alignment: no assumption that every index result
equals row+1 is accepted without checking the actual index and all mapping rows.


## Next-panel results and complete record review

`nq32-panel7-v20`: **37.50 EM (12/32)**; not promoted. All32 submissions and
82 calls were reviewed, with 50 literal queries returning 859 passages. Three
allowed queries returned no passages, and four beyond-budget requests were
explicitly denied. There were **no length stops, carrier failures, wire mismatches
or required-input omissions**. One unlabelled explanatory response remains a
whole-text submission rather than a reference-selected phrase. Wrong facts,
unsupported guesses, temporal/aspect choices and answer expansions remain.
Cost: **365,006 input / 13,188 output tokens, 82 owner calls / 50 tools**. CPU
ranking: 4,160 pairs / 610,655 tokens / 166 batches, 194.01 summed worker seconds.

`science32-panel6-v24`: **24.21875 signed native mean**; not promoted. Outcomes:
**15 full / 11 negative / six partial** (75, 75, 35, 70, 60, 60). All32
trajectories and **1,010 native actions** were examined; repeated command/feedback
pairs retain every revision index in the private review. Source, acknowledgement,
next observation and shared token accounting agree. There are 129 native command
rejections, two explicit owner finishes, one simulator horizon and three output
budget stops without a complete final action. Five real zero-argument wait
carriers now execute their literal native command without an extra repair call;
there are zero control-parse failures. This proves the narrow carrier fix works,
not that the new panel's changed mean measures its benefit.

Cost: **13,512,447 input / 88,338 output tokens, 1,015 owner calls / 1,012 tools**.
The remaining negatives and wrong experimental inferences are not automatically
retried or erased. Some full scores again come from guesses or unsound experiments.
All four accepted64 benchmark conditions remain unchanged. Both new evaluation
coordinators and the full CPU check have exited; existing three SGLang replicas
are retained, without adding or restarting an inference daemon.

The source-export completeness field is not populated for these non-MuSiQue
public projections, and the per-record upstream source revision is still unknown
in these historical-style manifests. Their required-rendered-input checks pass,
but must not be described as full source-revision/export verification. Frozen
private files, IDs, interfaces, deployment settings and run provenance remain
available; no unknown revision is fabricated or filled with a hash.

## Same-panel budget scope and retrieval breadth

Inspection of panel6's actual public requests found the correct 200-action and
8,000-token episode ledgers, but the owner repeatedly interpreted the separate
eight-call local decision allowance as its remaining task actions/calls. Source25
explains that this allowance is for preparing one action; renewing it after an
executed action never resets any episode budget. Static task wording is unchanged.
The regression test exercises a rejected carrier followed by two owner actions:
local calls 2/1/2, global calls 4/3/2 and environment actions 200/200/199.

The full same-panel32 `science32-budget-v25` comparison is **17.28125**, below
source24's **24.21875** and the >50.553 gate. There are **15 full, 12 negative,
five partial** outcomes (75, 43, 15, 60, 60); the auxiliary clipped mean is
54.78125, not the primary score. Seven cases improve, seven worsen, 18 stay equal;
the old results are retained rather than taking the best per case. All **945
owner calls and 941 executed actions** were checked, with zero literal-command,
acknowledgement or required-observation mismatches. Cost: **13,027,252 input /
84,674 output tokens**, 943 tools including two explicit finishes. There are
91 native rejections, two output-budget endings and one simulator horizon. The
12 negatives follow 11 focus commands and one numeric focus confirmation.
The former eight-call/action wording no longer appears in these owner replies;
this is not evidence of a task-score improvement.

One of its two length-stopped responses has a complete closed native action
followed by Qwen's literal message-end marker. That marker, rather than a missing
argument or closing tag, caused the parser to reject the call. Source27 accepts
that outer delimiter after a complete tool envelope, preserving raw tokens,
usage, arguments and actual stop reason. It does not complete partial syntax,
discard prose suffixes, select another call or execute any historical action.
A real isolated-owner test spends its full budget on this length-stopped response
and delivers its one unchanged action without a second model call. The new
same-panel32 framing condition retains all other source25 settings.

For NQ, a private offline diagnostic repeats all 46 allowed literal panel7 queries
without generating answers. Returned passages contain a normalized reference
alias for 29/32 questions; the top1000 BM25 pool contains one for 30/32. This is
only coarse lexical coverage, not a native score or proof of relevant evidence.
No labels enter search or ranking, and no diagnostic result is fed to an actor.
The new `bm25-1000-minilm-l6-v2@2` condition therefore tests retrieval breadth on
the same whole panel7: 1000 candidates instead of 100, otherwise the same frozen
MiniLM, owner queries, return20/query3 limits, 8,000-token budget and decoding.
The old 100-candidate condition remains the default. No NQ-supervised DPR encoder
has been installed or enabled.

## Wider retrieval and message framing: complete results

The same-panel NQ `nq32-wide-v26` result is **31.25 EM (10/32)**, below the
old 100-candidate condition's **37.50 (12/32)** and the >44.28 gate. The broader
pool is an optional named condition, not a new default or a successful repair.
All32 raw final responses, all **78 owner calls and 46 queries** were examined.
There are 742 returned passages, six allowed empty results and two explicitly
denied excess queries. All replies stop normally: zero length stops, carrier
failures, literal-query mismatches or missing required inputs. Two unlabelled
finals are retained whole; no reference-selected short phrase replaces them.
Cost: **282,494 input / 9,863 output tokens**. CPU ranking processes **33,412
pairs / 4,896,741 input tokens / 1,071 batches**, taking 1,516.51 summed worker
seconds. Wrong entities, temporal/aspect choices and answers inconsistent with
returned evidence persist. Twelve final texts match the old run; 20 change.
No 64-case generation is promoted.

`science32-framing-v27` scores **19.09375 signed native mean**, also not promoted.
All32 trajectories and all **999 executed actions** were read, including repeated
actions with every revision retained. There are **12 full / 11 negative / nine
partial** outcomes (83, 75, 75, 70, 13, 15, 60, 60, 60). The auxiliary clipped
mean is **53.46875**, not the primary score. Four cases improve, six worsen and
22 stay equal versus source25; historical outputs remain unchanged. Ten negative
terminals follow focus and one follows placement into an answer box. There are
102 native rejections, three explicit owner finishes and six output-budget
endings. One length-stopped response contains a complete command and executes
unchanged; the other five lack a complete carrier. None of this run's responses
contains the outer message-end marker, so this score is not evidence that the
narrow delimiter fix improved task capability.

Every action agrees with the owner's literal command, broker acknowledgement
and next complete public observation; required-input checks pass. Cost:
**1,007 owner calls / 1,002 tools, 13,792,624 input / 103,918 output tokens**.
Wrong container selection, unfinished experiments, navigation loops and guesses
remain model behavior, including guesses that happen to score 100. Partial
owner-finished states are not upgraded to successful tasks. Private full-record
reviews are retained for all three new32 runs; no historical action was replayed,
replacement candidate selected, negative score clipped in the main metric or
training update performed.

## Actual sampling seed limitation discovered in the paired review

Source25 and source27 have identical initial messages and input-token arrays for
all32 ScienceWorld cases, and identical numerical requested decoding including
seed0; nevertheless every first sampled response differs. A direct inspection
of all three running services finds SGLang **0.5.9**, FA3 attention, FlashInfer
sampling and **deterministic inference disabled**. Inspection of the installed
`sampling_batch_info.py::from_schedule_batch` confirms that this mode sets
`sampling_seed=None` rather than forwarding the request's seed. The active
FlashInfer top-k/top-p sampler uses that unseeded branch. This is stronger
evidence than merely observing different stochastic answers.

Accordingly, **seed0 was requested but did not control server sampling**. Past
scores remain actual stochastic observations, not reproducible seed0 results or
causal measurements of parser changes. This limitation also applies to other
sampled runs made on these service configurations, including the accepted64
development conditions; it does not silently invalidate, replace or rerun their
scores. There was no deliberate multi-seed sweep. A requested per-call seed
schedule alone never establishes batch-invariant model execution.

The existing raw serving controls already retain the relevant flags/backends
and server RNG seeds. The reporting fix now explicitly labels seed authority
per replica, preserves unknown states, reads the server version from its actual
top-level location, and removes the unconditional reproducibility wording from
the call-seed helper. A regression comparison ensures identical requested
decoding cannot conceal an actual sampler change in a purported policy-only
Step-0/trained comparison. Enabling a seeded serving mode would be a separately
named serving condition, with compatibility/throughput validation rather than
an assumed improvement. Permission for rolling validation has been requested;
no service was restarted or patched during this diagnosis.

## Verification and resource close-out

Targeted worker08 CPU checks covered budget accounting, reranking and literal
tool framing (including the 108-case framing-related set); the new serving-seed
reporting/paired-control tests and call-seed tests pass **9/9**. Local Ruff check
and formatting pass for all 11 changed Python files. The source27 milestone
full check passed **5,233 tests / 16 skipped**. After the newly discovered seed
reporting change, final source28 `CUDA_VISIBLE_DEVICES="" make check` passes
**5,234 tests / 16 skipped / 311 warnings**, pytest **335.46 seconds**, plus Ruff,
Mypy, both wheels and model-wheel boundary verification. The second full check
covers changed serving metadata, not a repeat on unchanged code. No extra E2E
run, independent review agent, hash check or repeat after documentation-only
edits was performed.

The three new32 coordinators and CPU checks have exited successfully. Existing
three inference replicas are retained; no inference daemon was added/restarted,
no other GPU used, and no training/posterior/library update or W&B write occurred.
NQ and ScienceWorld remain below their gates. Accepted64 scores remain frozen
with the sampling limitation disclosed; the overall goal is **not complete**.

## Query allowance contrast, with the same total owner budget

The earlier v13 greedy contrast already failed NQ and ScienceWorld, so it is not
repeated merely to avoid the disabled seeded sampler. Existing services remain
unchanged. The owner declined a restart and requested disclosure of the
non-deterministic sampler; no deterministic-serving experiment is launched.

`nq32-query6-v28b` instead allows **six rather than three corpus queries**, with
the same frozen panel7, 100-candidate MiniLM pool, top20, thinking-off, temperature
0.7/presence1.5, **eight total owner calls / 8,000 total output tokens / batch32**.
Two recent search attempts were explicitly denied by the three-query limit;
this is a tool-budget experiment, not a communication repair or a guaranteed
improvement. Current shared message-end framing is retained, so comparison to
v20 is not a strictly source-identical causal experiment. Actual server RNG
authority is disclosed. Both complete32 and potential complete64 architectures
are frozen before generation; >44.28 EM is still required to promote the64.
No accepted benchmark or ScienceWorld episode is regenerated for this contrast.

The first v28 launcher failed before runtime construction or model calls because
an interrupted source transfer left an incomplete archive. Its import traceback
and exit1 are retained as a launch failure, not a model answer or benchmark score.
The separate v28b attempt uses a completed compressed transfer and an extraction
that must succeed before launch. This correction does not change evaluation
source code, select a candidate, repeat a generated answer, or run a hash check.


### Query-six result and newly authorized conditions

`nq32-query6-v28b` completes at **34.375 EM (11/32)**, below >44.28; its64 is
not launched. All32 trajectories and **57 search actions / 880 returned passages**
were checked against literal queries, responses and the next public request:
zero wire mismatches or missing required inputs. There are 31 submissions and
one budget-ended empty candidate, not an infrastructure failure. Eleven allowed
searches return no passages; two further searches are explicitly denied. The
empty episode repeats overly narrow queries and keeps querying after the
zero-remaining-query feedback. The displayed remaining budgets are present.
Another response hits the output limit while repeating the same final statement;
no reference-aware extraction or fallback candidate is introduced.

Cost is **88 owner calls, 400,920 input / 16,515 output tokens**. The frozen
MiniLM processes 4,286 pairs / 634,268 input tokens in171 batches, using195.96
summed CPU worker seconds. Extra query capacity does not establish improved EM.
The owner now authorizes a separately labelled **NQ-supervised DPR** condition,
and a separate ScienceWorld thinking-on diagnostic. MuSiQue's accepted64 stays
frozen. Previously prepared next-panel launchers were paused, not silently run.

### ScienceWorld thinking-on diagnostic, not a main-goal result

`science32-thinking-v30` keeps panel6, public semantics10/state1, temperature0.7,
presence0, batch32, 200 native actions and the shared8,000 output-token budget.
Only native thinking is enabled; no final reserve/chunking is added. The result
is **20.375 signed native mean**, **14 full / 11 negative / seven partial**
(10,13,60,60,60,60,89). The clipped auxiliary mean is54.75 and is not the
primary metric. This diagnostic does not replace the off main result or pass
its gate, and does not justify a thinking-on64. Actual server RNG remains
non-deterministic, so small changes are not a clean causal estimate.

All32 trajectories and **629 native actions** were reviewed, retaining repeated
commands/revisions. Every executed command matches its owner source and broker
acknowledgement, and the full previous public observation reaches the next call.
There are58 simulator rejections, six output-budget endings and one owner finish;
14 completed native successes include some unverified guesses, not only completed
scientific experiments. Wrong target/container selection, incomplete melting and
conductivity experiments, stale disambiguation numbers and navigation errors
remain. All11 negative terminals are kept in the primary mean.

The response-level check finds **27 normal-stop outputs with a tool envelope
but no closing thinking delimiter**, despite an actual open-thinking prompt
suffix. These are not missing HTTP payloads: the raw outputs are retained and
not promoted from the reasoning channel to actions. Those27 plus one non-action
prose response are clarified by the same owner inside the original budget.
Six length-ended final responses remain unfinished. No framework action is
inferred from prose or chosen from an earlier draft. Cost: **664 owner calls /
630 tools, 7,202,348 input / 124,334 output tokens**. Main ScienceWorld remains off.

### Separate frozen DPR retrieval adapter

The new opt-in adapter leaves BM25/MiniLM defaults intact. Its public profile
explicitly discloses `facebook/dpr-question_encoder-single-nq-base` supervision
on NQ train, dense natural-language query semantics (quotes are not filters),
256 encoder-token truncation and passage-only output. The only answering agent
remains the Qwen owner. The retriever accepts only its literal public query;
it cannot access task answers, scorer labels or candidate submissions.

The CPU implementation uses the publisher's CLS `pooler_output`, without cosine
normalization, with the published compressed IVF4096/HNSW128/PQ128 inner-product
index, nprobe64 and efSearch128. Its21,015,300 index rows map in source order to
the original DPR TSV IDs. The [original upstream builder](https://github.com/huggingface/datasets/blob/1.18.4/datasets/wiki_dpr/wiki_dpr.py#L132-L155)
explicitly omits the last24 records lacking embeddings. Thus it is not silently
presented as covering all21,015,324 original corpus records. A24-passage public
check spanning three parquet shards confirms identical passage text and IDs;
HF retained TSV quote wrappers in some titles, while our existing CSV importer
correctly decodes them. This is a documented serialization difference, not
wrong paragraph mapping. We do not claim an exhaustive21-million-row comparison.

The [current official dataset builder](https://huggingface.co/datasets/facebook/wiki_dpr/blob/main/wiki_dpr.py)
and [question encoder instructions](https://huggingface.co/facebook/dpr-question_encoder-single-nq-base)
are the implementation references. Install the optional official CPU package
`faiss-cpu==1.15.0` in the trusted evaluation runtime alongside Torch/Transformers;
no old DPR training stack, context encoder, reader model or GPU context is needed.
Assets are downloaded before evaluation and accessed locally thereafter. Missing
assets, invalid index results or a retrieval timeout fail explicitly rather than
falling back to another condition. Public passage contents are unchanged and
in index rank order; private diagnostics retain retrieval tokens, CPU time and
index/model identity. Retrieval cost is separate from owner-generation cost.

The real CPU public-query smoke returns20 passages. The36 targeted DPR, corpus
broker and reranker tests pass with the real actor sandbox available. They cover
query preservation, rank/ID/text mapping, no vector normalization, unchanged
owner answers, cost persistence, explicit supervision and no silent fallback.
Initial test setup lacked the sandbox on PATH and skipped14 integration cases;
rerunning with the correct existing runtime PATH executes all36 successfully.
No hash check, independent review agent or inference-service restart is used.


### New32 DPR and ScienceWorld off results

`nq32-dpr-v31` completes at **40.625 EM (13/32)**, below >44.28. This is the
**NQ-supervised retrieval** condition, not closed-book or strict zero-shot OOD.
All32 trajectories and59 literal query actions were reviewed;57 allowed queries
return1,140 passages and two over-budget requests are denied. There are no
query/source/response/next-observation mismatches or incomplete required inputs.
31 candidates are submitted; one reaches8,000 total output tokens after repeating
conflicting final declarations and cannot be resolved without owner budget. It
is kept as a failed submission, not repaired by taking the most frequent answer.
Cost: **91 owner calls / 59 tools, 598,477 input / 21,247 output tokens**.
DPR costs **57 CPU forwards / 552 input tokens / 48.63 summed worker seconds**,
separate from owner cost. No64 is promoted from this run.

The raw output also reveals over-interpretation of the evidence-wording cue:
the owner repeatedly treats a concise span preference as a ban on inference or
combining evidence. A separately frozen `nq32-dpr-wording-v32` therefore keeps
the same panel7, DPR retriever, six-query allowance, generation budgets, decoding,
submission parser and native EM, but uses the existing NQ semantics8 without
that extra cue. Preflight compares the actual public semantics and requires that
only the evidence-wording suffix is removed. This is not a hidden rescoring,
reference-based extraction, new candidate selector or service restart. Both32
and the conditional fresh64 are frozen before generation. The32 must still
pass >44.28 before64 is allowed.

`science32-panel7-v29` uses a new whole panel selected uniformly before generation,
excluding previous manifests without using their scores. Main thinking stays
**off**. It scores **17.4375 signed native mean**, **15 full / 12 negative / five
partial** (75,77,13,19,74); auxiliary clipped mean54.9375 is not the goal metric.
All32 trajectories and **745 native actions** were reviewed, including every
repeated command's revisions. Literal commands, acknowledgements and next full
public observations agree. There are67 native rejections, two explicit owner
finishes, two output-budget endings, and one partial native horizon ending.
No framework action is selected or corrected; all12 native negative outcomes
remain. Two incorrect target choices fail on the first action; wrong container
selection and incomplete experiments persist even with explicit public help.
Some successes also follow guesses and are not proof of completed experiments.
Cost: **749 owner calls / 747 tools, 8,610,765 input / 70,972 output tokens**.
Its64 is not promoted. Merely changing the panel has not solved this gap.

Final CPU `make check` for the DPR code passes **5,242 tests / 16 skipped /
311 warnings**, pytest337.14seconds, plus formatting, lint, Mypy708 source files,
both wheels and model-wheel boundary. Initial Mypy caught a Transformers typing
wrapper issue, fixed with an explicitly typed model instance; the full check
covers the fix. Subsequent wording-profile changes use existing code, and no
second identical full check is run. Private assets and trajectories remain
outside Git. No training/posterior/library update or W&B write occurs.

### Evidence-wording contrast: completed, not promoted

`nq32-dpr-wording-v32` scores **34.375 EM (11/32)**. Removing the extra
evidence-wording preference does not establish an improvement, and this run
does not qualify for64. All32 trajectories, **81 raw owner responses and48
literal query actions** were examined. The960 returned passages reach the
owner unchanged; query transport and required public-input checks find no
mismatch. One pair of conflicting final declarations receives a same-owner
clarification inside the original budget, without scorer feedback. One
length-ended output spends the shared8,000 tokens on an unfinished calendar
enumeration: the whole unlabelled text is preserved, not mined for a year.
All32 receive native scores; zero infrastructure failures or dropped records.

Cost is **360,191 input / 17,689 output tokens**, plus **48 CPU DPR forwards /
470 retrieval input tokens / 41.96 summed worker seconds**. Incorrect entities,
temporal ambiguity, wrong answer aspects and strict-EM paraphrase losses remain;
none licenses answer rewriting by the framework. The same optional DPR profile
and existing semantics8 are now frozen for `nq32-panel8-dpr-v33`, using the
entire panel8 already selected before these DPR experiments. Its64 IDs are
fixed before32 generation. This is explicitly another development-panel
attempt, not a replacement of v31/v32 or an untouched generalization estimate.
No service restart, budget increase, new seed sweep or answer selection occurs.

### Panel8 DPR32 passes; fresh64 follows the predeclared gate

`nq32-panel8-dpr-v33` scores **46.875 EM (15/32)**, exceeding >44.28.
All32 trajectories, **78 owner responses / 46 query actions** were reviewed.
Every response stops normally and every candidate has an explicit final carrier;
no parse/transport/infrastructure failure or missing required input is observed.
The920 passages preserve their published ranking and source text. Three longer
submitted answers, mistaken entities, temporal guesses and exact-match failures
are retained as generated, not shortened against references.

Cost is **352,820 input / 7,618 output tokens**, plus **46 CPU DPR forwards /
497 retrieval input tokens / 40.54 summed worker seconds**. The passing32 gate
authorizes `nq64-panel8-dpr-v33`: the already frozen64 are all generated anew
under the same architecture, with no imported32 answers or candidate selection.
This remains supervised retrieval and development-panel search; accepted64
results of the other four benchmarks remain frozen.

The fresh64 completes at **39.0625 EM (25/64)** and **does not pass** >44.28.
The passing32 is therefore not a completed NQ goal. Every one of the64
trajectories, **155 raw owner responses and91 query actions** was reviewed.
All replies stop normally. There are89 authorized searches returning1,780
passages, plus two explicit query-budget denials. Source commands, tool returns,
next public observations and required inputs agree throughout. One owner
repeatedly searches after its query allowance ends and exhausts all eight calls
without submitting; it remains a failed candidate in the64 denominator. One
conflicting final is clarified by the same owner inside its existing budget.
No earlier answer is substituted, and no hidden answer resolves a conflict.

The remaining errors include misinterpreting the requested entity/aspect,
retrieval misses, guesses unsupported by the available passages and answers
that are not accepted by native EM. No new communication fault is established
by this lower score. Cost: **691,097 input / 17,135 output tokens**, plus
**89 CPU DPR forwards / 927 retrieval input tokens / 61.95 summed worker
seconds**. Generation finishes at roughly4,527 episodes/hour (about51seconds
for64; ETA0 at completion). There are63 submitted candidates, one candidate
failure and zero infrastructure failures; all64 remain reported. NQ and
ScienceWorld remain unresolved, and the overall goal is not complete.

The user explicitly declined inference-service restart. All three existing
replicas remain non-deterministic with batch32; requested seed0 is recorded
alongside actual server-global RNG authority, not described as reproducible.
The wording32, panel8 DPR32 and its fresh64 coordinators have exited. No extra
GPU service or training/W&B process is started. Code was delivered in493e814;
these subsequent edits only record completed runs, so the unchanged code's
successful5,242-test final check is not repeated.

For the32 shared IDs, the fresh64 has13 correct rather than the earlier15:
two improve, four worsen,26 keep the same score, and18 keep the same submitted
text. The additional32 contribute12 correct. These are actual sampled outcomes,
not evidence of an altered scorer or grounds to splice the better earlier
answers into64. At the final direct resource check (14:58UTC), all three HTTP
health checks return200; their parent/scheduler PIDs and approximately10-hour
uptimes are unchanged. Our GPU0/1/4 schedulers occupy approximately41/49/46GiB.
All evaluation coordinators and the full-check coordinator have exited;
only the pre-existing inference replicas/watchdog remain. Other GPUs and
other users' processes were not modified.

## Separate public retrieval and observation conditions (@34)

NQ's remaining retrieval misses motivate an optional **DPR100 + BM25100,
equal reciprocal-rank fusion, constant60, top20** condition. This follows the
rank-fusion method documented in [Pyserini's fusion implementation](https://github.com/castorini/pyserini/blob/master/pyserini/fusion/_base.py),
not its model-loading stack. Both legs receive the owner's literal query;
canonical passages are neither concatenated nor rewritten. No reader answers
the question. A failed leg fails the retrieval rather than silently changing
conditions. CPU dense cost and lexical/fusion cost are logged separately.
The DPR encoder remains NQ-supervised; this is not strict zero-shot OOD and
not claimed as a communication fix. Closed-book and earlier retrieval results
remain separate. Existing panel8, semantics8, six queries, eight owner calls,
8,000 output tokens, thinking-off and batch32 are unchanged.

ScienceWorld **public-state@2** retains all raw public-state@1 observations and
adds only an indented view of literal `(containing ...)` text already visible
in look/inventory. It does not query an object tree, reveal closed contents,
enumerate valid targets, choose disambiguation options or rewrite commands.
This is a separately named presentation contrast on existing panel7, with the
same semantics10/typed commands, thinking-off, presence0, 8,000 tokens and
200-action ceiling. Signed native final scores remain primary, including -100.

All64 IDs for each condition are fixed before32 generation; a fresh complete64
is authorized only after its32 gate passes. Four previously accepted64 results
are untouched. These are development regressions, not untouched final tests.
The three existing services are not restarted: requested seed0 still does not
establish deterministic inference, so score differences alone cannot identify
the effect of an interface change. Targeted CPU validation: **68 passed**;
no GPU model or training/W&B process was used for those tests.

The first hybrid32 attempt (`nq32-hybrid-v34`) ends with a **retrieval
infrastructure timeout**, not a benchmark score: nine candidates and13 returned
searches are retained, with23 started searches unanswered. Successful tool wall
times rise from9.9 to29.8seconds before the30second outer deadline. That deadline
included waiting behind four CPU processes. The broker now bounds queue
admission separately (120seconds), limits active CPU submissions to four, and
starts the execution deadline only after admission. The hybrid worker has
60seconds for encoder/index loading and both search legs; each lexical SQL
search still has its existing30second bound. Queue delay is persisted separately
from retrieval computation. No model output/token/action allowance changes.

This observed failure is covered by a32-request/four-worker regression;
the targeted retrieval suite passes **34 tests**. A separate `nq32-hybrid-v35`
generated the entire frozen32 anew, retaining v34's partial failure without
scoring it as zero or importing its nine answers. The corresponding64 condition
was fixed before generation and remains gated by32. ScienceWorld's independently
frozen layout condition does not use retrieval and is unaffected by this fix.

### Completed32 results: neither condition qualifies for64

| Condition | Actual main score | Strict gate | Outcome |
| --- | ---: | ---: | --- |
| NQ supervised DPR+BM25 fusion, v35 | 37.50 EM (12/32) | >44.28 | No64 |
| ScienceWorld public-state@2, v34 | -13.03125 native mean | >50.553 | No64 |

NQ's **32 trajectories, 74 owner responses and42 query actions** were inspected.
All replies stop normally, all candidates have explicit final carriers, and all
840 returned passages reach the next owner request unchanged. There are zero
parse, transport or infrastructure failures and no missing required input.
Two submissions longer than eight words remain unchanged. Incorrect entities,
answer aspects, temporal guesses, retrieval misses and native-EM wording losses
remain; the framework does not shorten answers against references. The queue
fix resolves the observed batch-timeout failure, not these answer errors.

Cost: **290,854 input / 5,757 output tokens**, 42 CPU encoder forwards with426
retrieval tokens, 42.84 dense worker seconds, and244.28 lexical/fusion worker
seconds. Summed queue delay across42 admitted queries is1,006.67seconds; this
is not elapsed run time. Generation completes at about1,410 episodes/hour
(81.7seconds for32, ETA0). The failed v34 attempt also remains accounted for:
32 started episodes, **45 owner responses / 96,841 input / 3,289 output tokens**,
36 started searches,13 returns and nine sealed candidates. All45 responses
were inspected; its unreturned searches remain infrastructure failures rather
than model wrong answers or discarded costs.

ScienceWorld's **32 trajectories, 751 owner responses and745 native actions**
were inspected. There are **11 full successes,17 native -100 endings, and four
partial scores (75,75,30,3)**. The auxiliary clipped mean40.09375 is not the
primary score or a passing result. Literal owner commands, broker executions,
acknowledgements and subsequent public observations agree. However, this is
**not a format-error-free run**: three multiple-tool envelopes require
same-owner clarification, and an output-budget ending leaves one unfinished
action unresolved. There are94 simulator command rejections. These remain
separate from transport errors and native negative task endings.

The remaining behavior includes treating target selection as inspection,
container/content confusion, cumulative stopwatch misinterpretation, treating
current temperature as a melting point, premature experimental conclusions
and repeated rejected navigation. Two owners finish at75; one episode reaches
the simulator horizon at30 (wait commands advance multiple ticks), and one
exhausts output at3. Some full native successes also follow guesses: native
success is reported as defined, not relabelled as proof of completed experiments.
No object is substituted, failed command repaired into a different action,
negative score clipped in the main metric, or older answer substituted.

ScienceWorld costs **9,597,425 input / 70,672 output tokens**, 751 owner calls,
747 tool calls and745 native actions. There are750 stop and one length replies.
Generation completes at about638 episodes/hour (180.7seconds for32, ETA0).
The layout is optional and did not produce a passing result. Non-deterministic
serving means this score difference alone does not isolate its causal effect.
Both runs still report32 missing per-task `source_revision` fields in the legacy
task metadata; persisted source/panel configuration is retained, but this is
not described as complete per-task revision provenance.

### Validation, delivery and resource boundary

The final CPU `make check` after the queue fix passes **5,251 tests / 16 skipped /
311 warnings** in337.59seconds, plus formatting, Ruff, Mypy710 source files,
both wheels and the model-wheel boundary. Earlier targeted checks passed68
related tests and, after the observed timeout fix,34 retrieval tests. Local
Mypy exceeded its50second WSL deadline; remote full Mypy completed successfully.
The earlier full check preceded the queue fix, which justified the final rerun;
no further unchanged-code full test is repeated for this results-only update.
No independent review agent, hash checks, training updates, training-evidence
writes or W&B writes are used. The dedicated Git-only agent handles delivery.

At the parent's direct15:31UTC resource check, all eight GPUs were inspected.
The three existing GPU0/1/4 services return HTTP200, with unchanged parent and
scheduler PIDs and approximately10h54m uptime. Their scheduler allocations are
about41/49/46GiB. This batch used those existing services for the two completed
32 runs and the retained failed hybrid attempt; all three evaluation
coordinators and the final CPU-check coordinator have exited. No new resident
service was created, no replica restarted, and no other user's process changed.
Requested seed0 remains recorded as **non-deterministic**, with no claim that
seed alone reproduces an answer. The four previously accepted64 results remain
frozen. NQ and ScienceWorld are unresolved; the overall goal is not complete.

## Next complete development panels (@36)

After inspecting every v35/v34 trajectory, no further systematic transport fault
explains their remaining gap. ScienceWorld's multiple `inspect_object` calls
are not read-only corpus-query batches: native `look at` advances time and can
trigger goal checks. The declared single-action interface therefore retains
same-owner clarification, rather than silently selecting one call or executing
a different action sequence. Its unresolved output-budget action remains a
reported failure; no old trajectory is rewritten.

Under the user's permission to try another whole panel, **NQ panel9** and
**ScienceWorld panel8** each fix all64 IDs before generating32. Uniform seed0
selection uses source identities, not previous item scores: NQ excludes480
previous IDs and draws from3,130 unique remaining questions; ScienceWorld
excludes411 previous task/variation IDs and draws from1,408 remaining entries.
These are additional development-panel attempts, not untouched test claims.
The four accepted64 benchmark results remain frozen.

NQ keeps the official **NQ-supervised DPR-only** condition, semantics8,
six queries/top20/eight owner calls, thinking-off and8,000 output tokens, with
the tested CPU queue correction. It is not the optional hybrid condition.
ScienceWorld keeps its earlier main **public-state@1**, semantics10 typed API,
thinking-off, presence0,8,000 output tokens and200 actions; the optional layout
contrast is not silently made the default. Both use batch32 and the existing
non-deterministic services without restart. Coordinators41271/41272 use the
three existing replicas; no new GPU service, training or W&B process starts.

## Completed @36 panels and fifth accepted 64-case condition

| Condition | Count | Native primary | Strict gate | Decision |
| --- | ---: | ---: | ---: | --- |
| NQ supervised DPR, panel9 | 32 | 50.000 EM (16/32) | >44.28 | Promoted |
| NQ supervised DPR, whole panel9 | 64 | 48.4375 EM (31/64) | >44.28 | Freeze; stop this loop |
| ScienceWorld public-state@1, panel8 | 32 | -6.5625 signed mean | >50.553 | No 64-case promotion |

The parent inspected all new owner responses, actions, retrieval queries,
submissions and associated public observations. The fresh NQ64 run has **161
owner calls, 97 search attempts**, including 96 successful queries returning
1,920 passages and one explicitly denied excess query. All64 final carriers
are unique and literal; there is no source/command mismatch, missing required
input, transport failure or parser failure. Seven submitted answers exceed
eight words. Two responses exhaust the8,000-token budget repeating declarations;
their single distinct final answer is retained without reference-driven selection.
Knowledge, retrieval misses, entity/aspect changes and strict-EM expansion losses
remain. Cost is **795,063 input /32,201 output tokens**, plus976 CPU retrieval
tokens,96 encoder forwards,64.725 summed retriever seconds and235.001 summed
queue seconds. NQ32 costs534,886 input /25,144 output tokens and91 owner calls.
Neither score is a claim of held-out generalization after development-panel search.

ScienceWorld has **13 full successes,16 native -100 terminals and three partial
outcomes**, with auxiliary clipped mean43.4375, not the main metric. Its913
owner responses and911 native actions retain their literal command/observation
links;136 native rejections are not communication failures. Two length/budget
endings remain recorded. One simulator-horizon termination follows native wait
ticks, not more than200 submitted actions. Cost is12,459,545 input /80,964 output
tokens. The legacy per-task revision field is still missing; persisted source
configuration is retained, not asserted to prove complete revision provenance.

Generation rates were approximately759,2,597 and429 episodes/hour for NQ32,
NQ64 and ScienceWorld32 respectively; all three finished, ETA0. No accepted
Math/APPS/GPQA/MuSiQue run was regenerated. The API-provider migration is a
separate pending change and does not alter these native-scored OOD results.

### Fixed-command ScienceWorld diagnosis, not a replacement evaluation

A CPU-only replay of four already stored episodes executed **all138 original
commands**, with no model call, alternate command or replacement candidate.
All four reproduced the same -100 endpoint, with zero differences in observed
text, native score, terminal state and simulator moves. Replay took10.29seconds.
Private object properties and goal-type diagnostics remain private and never
become owner observations; the exact native failure-branch string is unavailable.

The deployed JVM polarity checks agree with the upstream
[source/load terminal issue](https://github.com/allenai/ScienceWorld/issues/50)
and released
[electrical component implementation](https://github.com/allenai/ScienceWorld/blob/main/simulator/src/main/scala/scienceworld/objects/electricalcomponent/ElectricalComponent.scala).
The Python package reports1.2.3; the legacy manifest's claimed source revision
does not resolve upstream, so it is not described as independently verified.
This justifies an optional public instrument-reference condition, not modifying
the simulator, changing terminal rewards, or supplying hidden material properties.

The next controlled ScienceWorld32 contrast retains panel8's IDs, the same
public-state@1 observation, typed native actions,8000-token/200-action budget,
thinking-off, presence0, batch32 and all three unchanged service replicas.
Only `public-task-semantics@11` adds task-independent terminal-polarity and
measurement-snapshot semantics. The text contains no episode, object list,
classification answer, task solution or goal predicate. Every wire and target
still comes from the single owner. This is a new interface condition; the
@36 failures are retained and cannot be replaced by its outcomes.

### Completed @38 instrument-interface contrast

The same whole32 panel completed with **23.50 signed native mean**,16 full
successes,11 native negative terminals and five partial outcomes. The auxiliary
zero-clipped mean is57.875; it is not the primary score. This remains below the
strict50.553 gate, so no64-case promotion is authorized by these results.
The difference from @36 is a development contrast, not proof of a deterministic
interface effect: serving remains explicitly nondeterministic.

All959 request/response records and955 executed actions pass the automated
literal-command, acknowledgement and next-public-observation association checks;
there are no missing required public inputs or scorer infrastructure failures.
Two length/budget output failures remain, and127 native command rejections must
not be renamed transport failures. Manual reading of all32 owner trajectories
and every executed action is now complete, with private per-episode notes.
The32 trajectories cost12,529,216 input and92,813 output tokens. No accepted
benchmark was regenerated, and no training/posterior/library evidence was written.

The83 related CPU tests passed. The frozen interface-only source passed
CPU `make check`:5,255 tests,16 skips, ruff/mypy and both wheels (pytest339.55s).
This validation excludes the separate, still-changing Judge-provider migration.
The evaluation coordinator and CPU-check process both exited0. Existing three
SGLang services remain healthy without restart; no new resident service was added.

### @38 detailed findings and separate fixed-command diagnosis

The paired native outcomes improved on8 episodes, regressed on3 and stayed
unchanged on21. Both failures and native successes include premature object
selection, incomplete experiments and guesses. A native100 is preserved even
when the trajectory does not establish that a sound experiment was performed.
Public action meanings were delivered, but some owners still treated focus as
inspection, connected source/load polarity incorrectly, or used ambient
temperature as a melting-point measurement. These are not wire replacements.

A second CPU-only diagnosis replayed241 original commands from five selected
stored episodes, without any model call, alternate action or replacement answer.
All five final scores matched. Three episodes matched all checked intermediate
fields; two had seven steps with differing mobile-object/electrical observations.
This is evidence against claiming fully deterministic environment replay, not
grounds to replace old outcomes. Runtime was14.73s, about20.36 episodes/minute,
ETA0. No new GPU or resident process was created for the replay.

One apparent off/on contradiction was resolved by reading the complete public
response: the action reply said off, but the accompanying latest room view said
on. The owner's on statement was therefore supported. Another fixed replay
confirmed a measured object cooled across the task threshold during subsequent
actions; its negative terminal was not a scorer arithmetic error. The material
and goal diagnostics remain private and cannot guide owner actions.

The current budget is also more specific than merely200 owner actions: the
official Python wrapper independently terminates when native moves exceed200.
One wait-heavy trajectory reached207 moves after46 actions. This unchanged
native cutoff must be disclosed; removing it would be a new budget condition,
not a silent formatting fix. No time, reward, action or finalization rule changed.

After completing this review, whole development panel9 (@39) freezes all64 IDs
before generating its first32, excluding previously selected complete panels.
Selection reads task/variation IDs, not scores. It retains @38's source,
semantics11/state1, thinking-off, presence0,8000 tokens, both existing horizons
and batch32. Only the panel changes. No64 promotion is made unless its32-case
signed native mean strictly exceeds50.553. The five accepted OOD64 results stay
frozen; neither this panel search nor the earlier attempts are an unseen test.
The separate concurrently edited Judge-provider migration is not deployed by
this ScienceWorld run, and no training evidence or W&B data is written.

### Whole panel9 @39 completed; detailed review complete

The new32 completed with **36.875 signed native mean**,19 full successes,
nine negative terminals and four partial outcomes. The auxiliary clipped mean
is65.0, not the primary metric. The strict50.553 gate is not met: no64-case
run was launched. This different-panel result does not establish an interface
improvement over @38. All888 model calls and885 native executions were retained,
including two unresolved output decisions and80 native command rejections.
Input/output costs were12,059,412/91,984 tokens. Generation completed at roughly
596 episodes/hour, ETA0; coordinator exit0. The three existing inference
replicas were used without restart; no new resident service or training job.
All32 trajectories, their full owner replies and every native action have now
been manually read; per-item notes remain private. The separate complete wire
check found no command replacement or missing required public observation.
Remaining failures include task selection used as inspection, incorrect object
references, incomplete experiments, reversed electrical polarity and genuine
output exhaustion. Some native100 trajectories also used an invalid experiment
or a guess; those scores remain unchanged, not recast as experimental competence.

The existing public reference already distinguishes inspection/selection,
containers/contents, pickup/placement, adjacent navigation, numbered choices,
whole-object disconnection and source/load polarity. These delivered facts were
not consistently followed. No new parser relaxation, action repair, hidden-state
hint or success-conditioned retry is justified by this review. The score still
fails the gate, and its frozen64 continuation has not been run.

Whole development panel10 (@40) is the next authorized panel-search condition:
uniform seed0 selection from remaining task/variation IDs, all64 fixed before32,
with @39's frozen source/interface, thinking-off, decoding, budgets and scorer
unchanged. It does not isolate an interface improvement or establish unseen
generalization. Five accepted OOD64 benchmarks remain frozen; this work writes
no training evidence or W&B data. Since only private panel preparation and this
note changed, the prior successful full code check is not repeated.

### Whole panel10 @40:32 completed, no promotion

The unchanged architecture on the next whole development panel produced
**39.21875 signed native mean**:19 full successes, eight negative terminals,
five partial outcomes. The auxiliary clipped64.21875 is not the primary score.
The50.553 gate still fails; no64 continuation or replacement episode was run.
All868 model responses and864 literal native actions passed the wire/required
observation check, and all32 have native scores with no scorer infrastructure
failure. One incomplete tool carrier occurred at genuine total-token exhaustion:
the actual runtime message explicitly disclosed the remaining115 output tokens.
Budget notices were present in all868 requests, in the tool-role runtime surface;
checking only system/user roles would incorrectly suggest they were absent.

Costs were11,084,573 input and92,682 output tokens. Generation finished at about
548 episodes/hour, ETA0; coordinator exit0. Only the three existing permitted
inference replicas were used, without restart or a new resident process. Manual
review is complete for all32 trajectories and all their replies/actions. No code
or scoring change is claimed for this attempt; its below-gate result is retained.

The additional full-trajectory review finds the same unsafe selection behavior in
both failures and native100 outcomes: some owners focus to obtain information or
start an experiment, rather than deliberately select a task target. Native100
also does not establish that a melting-point experiment was actually completed.
Literal `look at around/task`, stale numbered choices, remote-object commands and
unfinished prerequisites are preserved, not rewritten by the broker. Repeated
waits can exhaust the native simulator horizon before the separate200-owner-action
allowance is exhausted; those counters must not be conflated. Per-episode details
remain in private review notes, not in Git.

### Scoped object-tool help candidate @41

The complete @40 review found an actual public-help scope mismatch:
`inspect_object` executes only `look at TARGET`, but its function-local description
included the whole observation-command family. Owners repeatedly supplied
`around/task` as object names. That is not a lost network message or proof that
this issue explains the score gap. Long repetition was already present before
history-window cropping; unsuccessful experiments and old measurements remain
separate behavior failures.

The new explicit condition is `public-task-semantics@12` with
`scienceworld-commands-scoped-tool-help@4`. It inherits @11 task/instrument facts
and changes only the object-inspection function's help, directing room, inventory
and task requests to their literal `act` commands. All bindings, selection help,
free commands, native time, stopping, private negative scores and old profiles
remain unchanged. No object is corrected or chosen for the owner.

The paired generation uses the same complete panel10 IDs and the same
thinking-off,8000-token,200-owner-action/native200-move, batch32/presence0 settings.
Its64 IDs were already fixed before @40; no case is selected by a previous score.
It is development regression evidence, not a new unseen test. The five accepted
OOD64 results are not rerun. CPU validation precedes this new32 generation;64
remains conditional on the strict50.553 gate and trajectory review.

CPU verification on the worker SSD passed Ruff, Mypy (712 source files), both
wheel builds and the complete suite: **5349 passed,16 skipped** in348.43s
(pytest rate about15.4 tests/s, ETA0; full check exit0). Related tests passed117
cases plus the final corrected help comparison. Initial test assumptions about
appended capability-cost text, the fixture's four-call allowance and the changed
profile label were corrected; production action/budget behavior was not loosened
to satisfy them. No independent review agent or hash check was used. The CPU
check processes ended, with no GPU test or new resident inference service.

### Paired scoped-help @41:32 completed and fully reviewed

The same complete32 IDs yielded **45.00 signed native mean**, still below the
strict50.553 gate. There are19 full successes, six negative terminals and seven
partial outcomes; auxiliary clipped63.75 is not the primary score. No64 run was
launched. Compared with @40, five cases improved, five regressed and22 were
unchanged. Full-success count stayed19; the higher mean does not establish more
completed tasks or a deterministic causal effect of the help change.

All867 owner replies and862 literal actions passed the complete wire/required
public-observation check, with zero command replacement, missing acknowledgement
or missing required observation. The32 native scores have no scorer infrastructure
failure. One incomplete tool marker was emitted with only one remaining output
token at total-budget exhaustion; this is not evidence of a network failure.
The malformed-carrier diagnostic remains recorded, not silently cleared.
Three replies stopped for length. Two native simulator horizons and five
owner-stop/budget outcomes remain distinct from the six native negative terminals.

The exact wrong-scope commands `look at around/task/inventory` fell from seven
to one; native rejections fell from165 to118. These descriptive observations do
not isolate model randomness or explain every score change. Manual reading is
complete for32/32 trajectories, all867 replies and862 actions and their complete
public-state changes, including all six negative trajectories. Behavior includes
container/content confusion, selection used as inspection, cumulative stopwatch
readings treated as independent durations, and unvalidated circuit/temperature
conclusions. Native100 also occurs after an invalid experiment; keep the original
native score without claiming that the scientific reasoning was correct.

CPU-only fixed-command diagnosis replayed81 saved actions from two growing-plant
episodes, without model calls or alternative actions. Observations, public states,
native moves, scores and terminal flags all matched. Selecting a container replaced
the previously monitored plant even as that plant later reached reproduction;
this is persistent focus semantics, not a missing-observation or score-arithmetic
fault. The initial diagnostic caller supplied an unsupported task field and
executed zero actions; correcting that private caller left benchmark code and
saved results untouched. The completed replay process exited0, ETA0. Final review
also distinguished repetitive prose/token exhaustion from unwatered plants and
repeated waits reaching the independent native horizon. No new generation, GPU
task or resident service was started during this diagnosis.

Input/output costs were10,602,820/92,606 tokens, with zero peer calls. Generation
completed at578 episodes/hour, ETA0; coordinator exit0 and no longer running.
Only the three existing inference replicas were used; all three health checks
passed afterwards, without a restart or new resident service. Seed0 was requested
but deterministic inference remains disabled as instructed. No training evidence,
W&B data or Judge request was produced. The code was delivered as `7acb574`;
only this result note changed after the successful full check, so that expensive
verification is not repeated. This session leaves Judge implementation/configuration
to the other session and does not stage its changes.

### Persistent-focus API candidate @42

The opt-in `public-task-semantics@13` / `scienceworld-commands-persistent-focus@5`
adds only the persistent/replacing nature of successful focus to shared API help,
including both literal invocation paths. This follows official
[ActionFocus](https://github.com/allenai/ScienceWorld/blob/main/simulator/src/main/scala/scienceworld/actions/ActionFocus.scala)
and the deployed fixed-command diagnosis; it gives no target, experiment policy
or private goal state. Old profiles, other tool help, bindings, observations,
scoring and budgets remain unchanged. The paired32/conditional64 use the same
pre-existing panel10 IDs, thinking-off,8000 output,200 owner actions/native moves,
batch32/presence0 and seed0-requested nondeterministic services. CPU tests precede
generation. This is exposed development evidence, not an unseen final panel.

Verification passed136 related tests in2.81s. The worker-SSD CPU `make check`
then passed Ruff, Mypy (712 files), both wheels, and **5394 passed,16 skipped**
in342.50s (pytest about15.7 tests/s, ETA0; check exit0 and process ended).
No independent review agent or hash check was used; later documentation-only
updates do not repeat the same full verification. No Judge path changed.

### Future external Judge routing: gateway first, official fallback

New terminal-only clients use `lab-gpt-5.6-luna` medium at the declared third-party
gateway, with official `gpt-5.6-luna` medium as the unavailable-provider fallback.
The HealthBench condition is
`healthbench-luna-medium-provider-failover-per-rubric@3`; its explicit YAML is
`configs/evaluation/healthbench_luna_medium_failover.yaml`.

A failed read-only availability probe routes the first unsent request to the
official API. A gateway POST with an uncertain outcome is preserved as incomplete
and is **not replayed**; only later unsent requests switch providers. SDK retries
are disabled, credentials remain separate, and requests are synchronous/tool-free.
FALSE judgements, malformed content and token exhaustion do not select another
provider. Actual provider, requested/returned model and failover reason are retained.

Historical direct-only and gateway-only HealthBench profiles retain their own
transport and verifier identities, including empty-submission scores. Existing
frozen processes are not hot-switched. A pinned legacy training spool remains
gateway-only; requesting the new failover condition through that unsupported
spool fails explicitly rather than mislabelling its route. This change starts no
training process and rewrites no old reward, posterior or benchmark result.

Both provider model-directory GETs returned HTTP200 and listed the requested Luna
model. No paid Judge request was sent in this verification. CPU validation of the
isolated API change set passed:20 targeted HealthBench tests in0.99s; final
`make check` passed Ruff formatting/lint, Mypy and both wheel builds, with
**5282 passed,16 skipped** in352.45s (about15 tests/s, ETA0). An earlier full pass
preceded the additional historical-score-identity fix; that fix received the
targeted tests and the final full pass. Initial check-directory staging and one
import-order issue were corrected; no GPU test was enabled. No redundant full
check followed the final documentation-only update.

Concurrent unrelated training edits are excluded from this snapshot/delivery.
For the shared training CLI, only the tested Judge-help hunk is included; the
other session's continuation change is left untouched. No independent review
agent or hash check was used. One byte comparison was justified by an actual
concurrent edit. The existing inference services were not restarted, and no
training evidence or W&B data was produced by this work.
