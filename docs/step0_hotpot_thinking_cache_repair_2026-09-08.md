# HotpotQA thinking and serving repair — September 8, 2026

Scope: the owner's existing fixed 32-task development panel, one untrained
Qwen3.5-9B owner, A2, skills off, thinking on, seed 0. No consultants, answer
selection, reference-based repairs, training or new panel selection.

## Diagnosis before the new run

The preceding complete thinking-on run scored **79.593855 answer F1** and
**59.375 answer EM**, both on a 0–100 scale. It remains unchanged.

- All 34 rendered calls contain the complete public task: the question and all
  ten passages for each of the 32 tasks. All 32 submitted payloads match the
  owner output, and independent application of the
  [official HotpotQA scorer](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
  matches all stored scores. The earlier distribution comparison is recorded in
  [the context diagnosis](step0_hotpot_context_diagnosis_2026-09-08.md); it was not
  unnecessarily repeated.
- All **34 calls actually used thinking**, ended with `stop`, and have complete
  channels. None used a length continuation. Reasoning tokens per call: mean
  **540.24**, median **429.5**, maximum **1,722**. Maximum complete response:
  **1,799** tokens; limits remain **8,192 per call / 32,768 per episode**.
  Maximum actual input is **3,618** tokens, below the 98,304-token context limit.
  These traces do not show output-budget exhaustion or context truncation.
- The installed Qwen template produces identical synthetic prompt text and
  token IDs for `reasoning_effort=low/medium/high/max` with thinking enabled.
  This evaluation renders tokens itself and uses native `/generate`; adding a
  chat-API effort label would not increase reasoning in this path. The
  [official Qwen template](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja)
  uses the thinking toggle, not those effort levels. The existing sampling
  profile already follows the model card's
  [thinking/general recommendations](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices).

More available tokens do not force a model that already stopped normally to
reason longer. No decoding parameter, task budget, answer projection or metric
was changed in response to these diagnoses.

The owner subsequently explicitly requested deeper thinking for **HotpotQA
only**. Thinking-on HotpotQA now receives uniform public guidance to take time,
connect passage evidence, resolve entity/relationship ambiguities and check the
exact question before answering. It does not specify answers, compulsory passes,
minimum token counts or extra independent solutions. Thinking-off HotpotQA and
all other benchmarks retain their previous instructions. This is a new declared
prompt condition, not a scorer repair or a guarantee of more reasoning or higher
accuracy. The fresh run measures both token use and score; it changes the prompt
and cache together and cannot isolate their individual effect on accuracy.

## Reproduced serving defect and repair

The old evaluation service reused BF16-rounded recurrent checkpoints in its
Mamba prefix cache. Four fixed synthetic inputs, unrelated to benchmark items,
were tested with 64 greedy output tokens: cold, warm, then cold recomputation.
All short token sequences matched, but cold/warm selected-token logprobs differed
by up to **0.0605485439**. Cold recomputation matched exactly.

The replacement uses the existing process-local FP32 checkpoint repair with
aligned 64-token pages and the `extra_buffer` strategy. It preserves BF16 output
activation rounding and does not modify the shared SGLang installation. On the
same four fixtures, corrected cold/warm/recomputed outputs and legacy cold versus
corrected cold have **identical token IDs and zero selected-logprob difference**.
The declared tolerance was 1e-6; it was not relaxed after inspection.

The diagnostic harness needed an idle wait because SGLang briefly rejected a
cache flush after returning a response. This changes neither numeric criteria
nor inference. Completed cases were retained; only interrupted cases resumed.

The native evaluation serving profile now enables that precision repair, and
effective execution controls record the observed cache page size. This prevents
silently retaining the old default. Synthetic agreement is bounded evidence,
not proof of equivalence for every request or proof that cache rounding caused
particular HotpotQA mistakes.

## Fresh fixed-32 evaluation

Run `seven-thinking-hotpotqa-deliberate-fp32-20260908`, condition
`A2-hotpotqa32-deliberate-thinking-fp32@1`, completed all 32 tasks without repair,
continuation, tool calls or consultant calls. Every call used thinking and ended
normally with a complete channel. Every rendered call contains the complete
public task and the declared deliberation instruction; panel entries are
unchanged. Independent application of the original official scorer matches
all 32 new F1/EM results. Historical answers and scores are not relabelled or combined.

| Measurement | Previous thinking run | New prompt + corrected cache |
|---|---:|---:|
| Answer F1, 0–100 | 79.593855 | **82.587482** |
| Answer EM, 0–100 | 59.375 | **68.75** |
| Owner model calls | 34 | 32 |
| Mean reasoning tokens / call | 540.24 | **518.34** |
| Mean reasoning tokens / episode | 574.00 | **518.34** |
| Maximum reasoning tokens / call | 1,722 | 1,560 |

The owner's **F1 80** goal is met on this development panel. However, reasoning
did **not** become longer: the prompt did not achieve increased token use. Do
not describe this as a successful effort-level increase. Longer reasoning is
not established as necessary by these results. Forcing additional computation
would require a separately declared same-owner verification stage or other
decoding change; neither was introduced or evaluated here.

There are 22 full-F1, eight partial-F1 and two zero-F1 submissions. Remaining
errors include semantic misinterpretation and native answer-span granularity;
they are not repaired using references or custom aliases. See the
[aggregate result](machine-results/step0_hotpot32_deliberate_thinking_2026-09-08.json).

Wall time was **215 seconds (3m35s)**, about **535.8 tasks/hour**, with 71,478
input and 19,925 output tokens. Only physical **GPU 6 on 22049** was used, with
four concurrent independent episodes, about 28 GB VRAM and observed 100% GPU
utilization during the run. The old service was retired before evaluation; the
new service remains available. Evaluation processes exited. The prior run had
two available slots and other serving work, so this is not an isolated cache
speedup measurement or a sustained suite-wide throughput claim.

## Validation and scope

- Local Ruff lint and format checks passed for the changed Python files.
- Local targeted pytest produced no result before its 150-second WSL timeout;
  it is not counted as passed. The targeted checks moved to the Linux CPU host:
  **24 passed in 5.72 seconds**, covering serving arguments, recorded page size,
  passage delivery and thinking-only Hotpot guidance. They also verify that
  TriviaQA and non-thinking Hotpot do not receive the new guidance.
- One final `CUDA_VISIBLE_DEVICES="" make check` passed on the frozen
  Linux-local source (`a76237c` plus this batch's overlay): **3,611 tests passed,
  one explicitly CUDA-only test skipped**, 136 warnings, pytest 291.38 seconds.
  Ruff, formatting, mypy over 548 source files, both wheels and the model-wheel
  boundary passed; the command exited zero. Later documentation-only changes
  did not trigger a redundant full check. Other sessions' subsequent changes
  are not implicitly covered by this snapshot's result.
- Existing source-distribution verification was reused. No unchanged benchmark
  rerun, repeated successful targeted command, separate E2E suite or checksum
  check was added. The fresh fixed-32 run is the integration evaluation.
- Main-thread implementation and self-review only; no coding/review subagents.
  The previously authorized Git-only agent synchronizes progress, not validation.
  Item contents and item-level evidence remain private.
