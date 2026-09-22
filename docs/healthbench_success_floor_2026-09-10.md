# HealthBench fixed-four success floor — 2026-09-10

Target: **at least 1/4 successful trajectories** on the existing fixed HealthBench
four-source development panel, using the untrained Qwen3.5-9B single owner.
Success remains native score >= 0.60 **and** zero triggered negative rubrics.
An average rubric score cannot substitute for this binary requirement.

## Read-only findings

The two previous native-v2 panels both have 0/4 successes: initial-policy mean
0.420863 and Step-10 mean 0.401525. Their source conversations are intact.
The actual 20 actor requests retain the complete conversations; all ten action
requests retain their corresponding current reasoning. No private rubric text
was found in those actor inputs. All 116 saved judge calls include the complete
source conversation and the owner's actual submitted answer. Each rubric was
graded once, all calls finished normally, and recomputing the scores and negative
counts from those verdicts agrees with the persisted results. Original evidence
and labels are unchanged. This verifies transport/arithmetic, not the clinical
reliability of every local-Qwen judgment.

The initial policy hit its 1,024-token reasoning cap in three of four trajectories.
One draft ends mid-sentence while discussing the requested phase rather than
finishing its reasoning. Complete submission alone does not establish an adequate
reasoning budget. Increasing that budget is a named experiment, not proof that a
low score must have been caused by a software defect.

## Declared candidate

- Same four source questions, original order, one newly generated trajectory each;
  no old answer import, selective retry, voting, reference-based prompt hint, or
  rubric/threshold change.
- Initial policy, skills on, seed zero, native thinking on, raw policy sampling.
- HealthBench reasoning cap **4,096** instead of 1,024; eight turns, 2,048 action
  tokens and 65,536 input tokens remain unchanged.
- Four-source diagnostic collection, **not** a change to the formal B28 batch.
  Its smaller panel has its own sampling coordinates; this is not a strict paired
  same-random-stream attribution experiment.
- Production collector, native-v2 carrier and unchanged temperature-0 training
  rubric grader; no optimizer, posterior or skill-library update.
- Existing inference service on endpoint 22048, physical GPU 6. No additional
  GPU server and no hot edit or restart of the other session's live training.

The private launcher passed syntax and effective-configuration checks, including
the actual HealthBench budget override and the separate diagnostic batch size.
Production source is unchanged; an identical full test/build check is not repeated
for this private launcher and documentation. No hash checks or independent review
agents are used. The complete four-case result, not a partial favorable answer,
will determine whether this candidate meets the floor.

## Budget-only result and confirmed planning-interface gap

The complete budget-only four-case run scored **0/4 successes**, mean **0.439986**,
in 288.66 seconds of collection. All 58 judge requests completed. It is retained
as a failed candidate, not replaced by a more favorable retry. The client exited
and unloaded its temporary adapter; it did not update the running trainer.

One reasoning call still exhausted 4,096 tokens. Its draft repeatedly questioned
whether the submission tool existed and how to name it. The actual interface
explains that declared tools are available, but the R request omits the names and
parameter schemas which are supplied only to A. This is a missing public planning
input, distinct from missing source dialogue or a rubric-scoring arithmetic error.

The explicit `reasoning_tool_catalog` candidate introduces `phase-context@2`.
R receives a read-only reference generated from the **same** public tool definitions
as A; R still has no executable tool-call channel. A and B retain their existing
tool interfaces, and B still excludes current reasoning. No medical checklist,
mandatory follow-up question, answer, or rubric enters the new reference. The flag
is opt-in and persisted in formal/rollout configuration and condition identity;
historical defaults and the other session's running source are not hot-edited.

This second candidate keeps the four-source sampler and the 4,096-token reasoning
budget of the failed budget-only run.

## Complete repaired result: target met

The complete four-source catalog-repair run has **1/4 successes (25%)**, with
mean native rubric score **0.461916**. Collection and grading took **158.82 seconds**
(2.65 minutes; 160.86 seconds including preparation/cleanup). All four trajectories
finished in one turn: eight actor requests and 58 judge requests, all completed.
The prior budget-only run remains 0/4. Neither result overwrites the original panel.
All four reasoning calls stopped normally rather than hitting the token limit;
they generated 3,212 reasoning tokens in total, and all four actions were accepted.
Recomputation from the 58 saved verdicts confirms the native scores, negative counts
and exactly one binary success. The judge received the complete dialogue and the
actual owner answer in every call, using frozen base weights. The two four-source
candidates retain identical source order and matching generation seeds at shared
sampling coordinates.

This meets the requested HealthBench floor on the fixed development panel. It is
not a 32-sample IID score, clinical certification, seven-domain T0 qualification,
or a formal-training throughput claim. Shared service load was not controlled for
a speed comparison. Do not splice these four new results into an older B28 panel.

Validation: **40 targeted tests passed**, including the actual private Qwen
tokenizer, old-condition compatibility, persisted configuration, static/typed
planning context, and sealed/provisional F/B prefix and action-span agreement.
One final CPU-only `make check` on 22049 passed **4,093 tests**, with two explicit
CUDA opt-in skips; format/lint, mypy and both wheel builds passed. No unchanged
CUDA gradient qualification, duplicate full check, manual hash check or independent
review agent was added. The main thread implemented and checked the interface change.

Both finite diagnostic clients used only the existing physical GPU-6 service,
then exited and unloaded their own temporary adapters. The service remained healthy
at the direct final check. No new GPU server or formal optimizer/posterior update
was started by this repair; the other session's live training was not reconfigured.
Enabling the candidate requires its explicit recorded condition rather than a
silent change to a running or restored historical configuration.
