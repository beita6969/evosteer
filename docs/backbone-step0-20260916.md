# Backbone-only Step-0 — 2026-09-16

## Scope and results

All 734 frozen cases now have terminal evaluation results: IID 350 and OOD 384.
AIME2026 uses all 30 distinct questions; each other benchmark uses 64.
These are development/regression panels, not untouched final generalization tests.
Failures, empty submissions and regressions remain in the denominator; no sample
replacement, best-of selection or answer regeneration was used for recovery.

One Qwen3.5-9B owner per episode, adapter-free and skills-off; no consultants.
Training updates, posterior updates, skill evolution and training-evidence writes
are all zero. Thinking is on only for AIME2026, HealthBench, Math-Hard and GPQA.

Scores below are on a 100-point scale. F1, EM, pass@1, rubric scores and environment
scores retain their respective meanings; they are not interchangeable success rates.

| Set | Benchmark / primary metric | n | Score |
| --- | --- | ---: | ---: |
| IID | HotpotQA answer F1 | 64 | 82.08 |
| IID | TriviaQA answer F1 | 64 | 86.76 |
| IID | AIME2026 accuracy | 30 | 70.00 (21/30) |
| IID | HealthBench Luna-medium native rubric mean | 64 | 31.85 |
| IID | ALFWorld native success — **input caveat below** | 64 | 68.75 (44/64) |
| IID | MBPP+ Base AND Plus pass@1 | 64 | 79.69 (51/64) |
| OOD | MuSiQue-Ans answer F1 | 64 | 68.68 |
| OOD | NQ-Open supervised-DPR retrieval EM | 64 | 42.19 (27/64) |
| OOD | Math-Hard accuracy | 64 | 90.63 (58/64) |
| OOD | GPQA Diamond Biology + Organic Chemistry accuracy | 64 | 56.25 (36/64) |
| OOD | ScienceWorld signed native final mean | 64 | 24.34 |
| OOD | APPS Introductory pass@1 | 64 | 81.25 (52/64) |

The full set does **not** meet all user targets. Numerical gates use strictly more
than 90% of the supplied targets, with the separate owner-accepted ScienceWorld
38-point requirement. An external paper's score is not a matched control experiment.
NQ uses the explicitly authorized NQ-supervised DPR condition, not closed-book or
strict zero-shot OOD. GPQA is sampled from the approved Diamond BioOrganic-91 pool.

## Execution conditions and recovery

OOD preserves the previous frozen 64-case architecture, including the pre-upgrade
65,536-context, non-deterministic services. Its output allowance is 8,000 tokens per
episode except APPS, whose single initial response can use the whole 12,000.
Math-Hard/GPQA retain their already-declared same-owner bounded-thinking continuation.
ScienceWorld retains 200 simulator actions and the signed native final score.

After OOD completed, the three owned services were rolled to the formal IID
configuration: 81,920 context, deterministic inference, PyTorch sampling, batch 32,
SGLang 0.5.15.post1, BF16 and no loaded policy adapters. Dependencies were transferred
through G, not downloaded into the C-backed WSL filesystem. IID uses the captured
formal phase budgets, native action wire, tokenizer and controller. These IID phase
budgets are not the OOD whole-episode budget. Future trained-checkpoint comparisons
must preserve each benchmark's captured architecture; the upgraded service is not
silently equivalent to the earlier OOD service configuration.

Missing `blobfile` initially stopped 34 HealthBench terminal scorers **before any
Judge request**. The missing official dependencies were installed. Existing owner
responses were restored exactly from settled request journals; started episodes
could not issue replacement owner requests. Originally unstarted episodes ran once.

One later HealthBench gateway POST timed out after 120 seconds. Its 11 returned
rubric judgments were retained. Following explicit user authorization, only its
one missing rubric was sent to official `gpt-5.6-luna`, medium, with the same prompt
and 8,000-token limit. The official response arrived in 13.32 seconds. The unchanged
official parser aggregated the 11 old responses and this one response without any
additional model calls. The old incomplete ledger remains unchanged; the completed
score is a separate authorized supplement, not a fabricated successful engine run.

Consequently, IID has 349 original successful engine artifacts plus one explicitly
supplemented terminal score. All 64 HealthBench cases and all 838 rubrics are scored.
There were 839 actual Judge POST attempts, 838 returned responses and one unresolved
gateway usage record that may already have been billed. Actual returned model labels
were `gpt-5.6-luna`. This is a gateway-primary condition with one authorized official
supplement, not a claim that all cases used official API routing or the historical
32-case HealthBench protocol. No shared Judge implementation was edited here.

The 19 never-started final MBPP+ cases in the interrupted shard ran once with their
original global sampling positions. No prior environment or completed Judge result
was replayed to seek a better answer.

## Trace findings and limitations

The IID structural pass covered all 350 owner trajectories, 2,412 model requests
and 1,206 action records. Stored raw responses agree with result traces; action
text agrees across the result/parse boundary. There were 17 format-error actions
(AIME 1, HealthBench 2, ALFWorld 12, MBPP+ 2), not HTTP delivery substitutions.
These errors and the same owner's subsequent interface handling remain in the cost.
This structural check is not a claim to have manually assessed every reasoning token.

ALFWorld has a substantive legacy input problem: the formal catalog instruction
and simulator reset goal both enter all 64 prompt histories. Different wording alone
is not an error, but reading all 64 pairs confirmed **at least 8 clear object or
destination conflicts**. The frozen formal run used `catalog-instruction@1` and was
not silently switched to the available `reset-public-goal@2` condition. Its 68.75 is
therefore retained as a legacy-interface diagnostic, not certified as a clean native
baseline. Correcting the goal binding requires a named new condition and consistent
handling in the future formal-training evaluation architecture.

AIME has 20 length-stopped reasoning calls; MBPP+ has 20. A reasoning cap alone
does not imply a lost final answer: this IID architecture has a separate action
phase. All submitted programs were scored as produced, with no automatic repair or
hidden-test-based candidate selection.

The earlier full OOD structural pass retained five Math-Hard empty submissions,
six GPQA empty submissions plus one invalid final, and two APPS empty submissions.
ScienceWorld recorded 31 full successes and 20 native -100 endings; all 1,844 executed
commands matched their recorded owner actions and environment receipts. The clipped
55.59 auxiliary reward is **not** the primary 24.34 signed final mean. No failure was
converted into an unseen replacement sample. Several imported sources still have
unrecorded upstream revision metadata; frozen local identity is not a claim of a
fully pinned upstream dataset release.

## Cost, validation and resources

- IID owner: 2,412 requests, 24,271,287 input tokens, 1,308,195 output tokens.
- OOD owner: 2,656 requests, 30,923,386 input tokens, 1,104,727 output tokens;
  1,984 tool calls, zero peer-model calls.
- HealthBench Judge known usage: 1,071,804 input and 121,520 output tokens;
  the timed-out gateway request's unknown usage is additional, not zero.
- Relevant recovery tests passed; the latest binding fix passed 5 focused tests.
  Final evaluation-host CPU `make check` passed: **5,455 passed, 16 skipped**, plus
  Ruff, mypy (713 files), package builds and boundary checks. The final small binding
  fix had separate focused tests/mypy. Full checks were not repeated for this
  documentation-only completion or private deterministic aggregation.
- No independent review subagent; the mandated Luna agent handled Git delivery only.
  No new hash audit was added. Licensed cases, rubric text, trajectories, credentials
  and internal server paths remain outside Git.
- All evaluation coordinators have exited. The three owned SGLang replicas remain
  available as requested, with batch 32. Host identity, physical GPU allocation,
  service PIDs and ports are recorded only in the private operational handoff.
  No other GPU processes were terminated.

Model weights continue through the separately running local-G-to-evaluation-host transfer queue.
Qwen3.5-27B has all 11 indexed shards present remotely; no C-backed weight download
or WSL shutdown was introduced by this evaluation work.
