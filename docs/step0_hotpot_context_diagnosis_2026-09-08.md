# HotpotQA context and answer-interface check — 2026-09-08

The requested seven-IID round is `seven-iid32-a2-owner-2ae5b87-20260908T063713Z`.
This check uses its 32 frozen HotpotQA tasks and 33 actual owner requests, including
the one interface-repair request. It does not generate additional answers.

## Findings

- The earlier **128-task / F1 72.13** run was also checked: all **145** rendered
  owner requests contain all ten passages and the question, with no archived-message
  omissions. Its recorded inputs range from 1,626 to 3,527 tokens. This separate
  historical check added no actor calls and changed no scores.
- **Not question-only:** all ten source passages reach every rendered request and
  the decoded model input. Source/export mismatches: zero; dropped passages: zero.
- All question content is present. Two questions lose only trailing whitespace at
  the chat-template boundary. No question words disappear.
- Inputs contain 1,390–2,575 tokens, well below the declared context capacity.
  Recorded transport token counts match re-encoding; archived-message omissions: zero.
- F1 **75.0117243867**, EM **56.25**, denominator **32**: 18 full-F1, ten partial-F1,
  four zero-F1. Direct execution of the
  [official answer scorer](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
  agrees on every saved answer. This is answer F1, not supporting-fact or joint F1.
- Losses include answer-span granularity/paraphrases, a numeric spelling mismatch,
  a bare yes/no answer expanded into explanatory prose, and genuine reading errors.
  At least one source question/reference pair is inconsistent with its own passage.
  The subsequent independent dataset comparison below finds the same reference in
  the public distribution; this is not evidence of a reference mix-up in our export.

## Interface repair, without changing the metric

The owner layer removed a short-answer field's header before calling the shared
decoder. Thus `Final answer: <answer>` followed by another explanatory line could
be submitted as answer-plus-explanation, despite the direct decoder recognizing the
field correctly. Preserve the field boundary and version the QA owner parser as v3.
Conflicting final declarations still require owner clarification. Bare prose,
numeric spellings, names and reference aliases are not rewritten by this fix.

Five context-delivery tests cover both public passage carriers, the real isolated
actor/broker boundary, interface repair, answer isolation and insufficient capacity.
Twelve answer-field regressions cover framing, conflicting declarations, unchanged
bare answers and persisted owner provenance for both QA benchmarks.

Read-only old/new projection comparison on **all 32 HotpotQA + 32 TriviaQA** saved
owner finals finds **zero changed payloads**. Therefore this fix does not explain or
retroactively increase this round's score. The running round retains its frozen old
parser identity; its answers and grades are not overwritten. No targeted retries,
new aliases, passage selection, reference injection or answer voting were used.

All questions, references, passages, output traces and per-item diagnoses stay in
private storage. The single-owner topology, thinking policy and budgets are unchanged.

## Independent source comparison

The CMU download endpoint remained unavailable. Instead, a separate copy of the
[HotpotQA distractor validation distribution on Hugging Face](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
was downloaded on September 8: 7,405 records, not a model-generated reconstruction.
This is a comparison against that distribution, not a claim of byte identity with
the unavailable CMU file.

All **128** frozen panel questions match exactly one distribution record after
removing the export's wrapper and surrounding question whitespace. There are:

- **0** unmatched or ambiguous questions;
- **0** reference-answer differences;
- **0** passage-content differences across **1,280** passages, comparing titles,
  sentence content and passage order while ignoring whitespace-only differences.

The suspected question/reference inconsistency is therefore already present in
this separate distribution. No label, candidate, panel membership or official
metric was changed. Existing 32-task F1 remains **75.0117243867**, below the owner's
**80** target. A score below that target alone does not establish an evaluator bug.
This read-only comparison made no model calls and did not trigger an unchanged
evaluation rerun; private source files and item-level evidence remain outside Git.

## Subsequent optimization and fresh 32-task run

After the owner requested further optimization, revision `7588601` made the public
task instruction explicit: return an answer span using the passage's wording, or
yes/no for a yes/no question. These semantics follow
[HotpotQA's task definition, §2](https://aclanthology.org/D18-1259.pdf). No reasoning
steps, examples, reference-specific hints or additional model calls were prescribed.
This is an interface/prompt optimization, not a finding that the native scorer was wrong.

Fresh run: `hotpot32-a2-public-answer-7588601-20260908T080300Z`.

| Same fixed 32-task development panel | Earlier run | Public-answer instruction |
|---|---:|---:|
| Answer F1, 0–100 | 75.0117 | **81.0250** |
| Answer EM, 0–100 | 56.25 | **68.75** |
| Completed / scored | 32 / 32 | 32 / 32 |

The current panel exceeds the owner's F1 **80** target. Four task scores improved,
28 were unchanged and none regressed. New answers were generated for every task;
historical answers or scores were not substituted. The earlier seven-domain report
retains its original HotpotQA row and is not relabelled as this new condition.

- Same Qwen3.5-9B single owner, no adapter or training, skills off, thinking off, seed 0.
- Questions, ten passages, references, native parser/scorer and sampling are unchanged.
  Per-call output cap remains 8,192; episode cap 32,768; model-call ceiling eight.
- Only the existing physical GPU 5 service on endpoint 22048 was used, with 32
  concurrent episodes. The earlier run used two services; this is not a matched
  concurrent control experiment or proof of backbone superiority.
- **33** actual owner calls: 69,353 input and 2,780 output tokens. One owner-requested
  public history read accounts for the extra call; there were no consultant calls,
  scorer-driven repairs or candidate selectors.
- Launch to summary publication took **13.998 seconds**, including coordinator
  startup; it is a small-batch measurement, not a sustained suite-wide speed estimate.
  The coordinator exited. No additional GPU service was started.
- All **33** rendered requests preserve the complete public task and answer semantics,
  with zero archived-message omissions. All **32** submissions match their actual last
  owner output's deterministic projection and agree with the official F1/EM scorer.
- New distribution: 22 full-F1, seven partial-F1 and three zero-F1 answers. No candidate
  or infrastructure failures. Remaining mistakes were not corrected using references.

This panel is **development-exposed**. It does not establish performance on the full
128-task panel or unseen data, and no additional benchmark round was started after
meeting this panel's target. See the
[aggregate machine result](machine-results/step0_hotpot32_public_answer_2026-09-08.json).

### Validation and scope

- Local targeted command covered `test_hotpot_context_delivery.py`,
  `test_short_answer_field_boundary.py` and `test_step0_integrity_sources.py`.
  All 33 progress markers reached 100%, but process exit exceeded 300 seconds;
  this was **not** recorded as a completed pass. Local mypy likewise timed out.
- One complete `CUDA_VISIBLE_DEVICES="" make check` on the 22049 Linux local disk
  checked the isolated `7588601` source: **3,467 passed, one CUDA-only skip**;
  Ruff format/lint, mypy over 532 source files, both wheels and model-wheel boundary
  passed. This includes the targeted cases above. Pytest took 234.90 seconds and
  emitted 106 existing PEFT warnings. The check process exited successfully.
- Aggregate consistency, private-field exclusion and final diff checks passed.
  No second full check was run for the subsequent documentation-only changes.
  The actual 32-task run supplied the end-to-end validation; no extra suite was launched.
- The main thread performed implementation and self-review. No review/coding/watch
  subagent or checksum checks were used in this batch. Other sessions' uncommitted
  training changes were excluded from the tested source and left untouched.
