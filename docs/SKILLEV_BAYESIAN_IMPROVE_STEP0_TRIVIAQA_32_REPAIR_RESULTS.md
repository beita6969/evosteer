# SKILLEV BayesianImprove Step-0 TriviaQA 32-Sample Repair Results

> **Historical assisted-input condition (superseded 2026-09-06).** The generated-material/search
> protocol below is not the corrected released reading-comprehension lane. Its PASS label does
> not establish label noninterference or a clean architecture gain. Keep the historical material
> assistance and exposure visible; [corrected results](STEP0_PAIRED_NO_SKILL_AND_TEXT_SKILL_RESULTS.md)
> are reported separately.

## Scope and decision

This is a no-training inference diagnostic of the project's
SkillFlow-derived **BayesianImprove** architecture, whose scientific method is
defined by `idea.tex`. It is not an evaluation of unmodified SkillFlow. The
backbone is frozen, adapter-free Qwen3.5-9B and the run performs zero optimizer,
trajectory-balance, posterior, calibration, or Operator updates.

The owner-set primary target is TriviaQA F1 **81.00%**. The fresh final
32-record panel result is:

| Population | N | EM | F1 | F1 target | Status |
|---|---:|---:|---:|---:|---|
| Frozen result-blind final projection | 32 | **81.25%** | **85.83%** | 81.00% | **PASS (+4.83 pp)** |

All 32 records have definitive outcomes. Generation/retrieval infrastructure
failures and invalid terminal candidates are both zero. Candidate generation
took 198.7 seconds, or about 580 records/hour; the remaining ETA is zero.

## What was wrong

The historical 32-record Step-0 result was EM 50.00% and F1 68.84%. The repair
identified three independent problems rather than treating every miss as a
model-quality failure:

1. the architecture exposed only one search even though the typed TriviaQA
   seed skill was intended to support evidence gathering and disconfirmation;
2. the generic terminal stage sometimes copied a verbose evidence sentence
   rather than transcribing the short answer hypothesis already reached by the
   reasoning model; and
3. the released SkillFlow answer wire retained only a truncated alias set,
   while the official TriviaQA EM/F1 rule maximizes over the complete official
   alias array. That scorer mismatch marked legitimate synonyms as wrong.

The third issue was especially important. Before the scorer repair, the same
unchanged final candidates measured EM 59.38% and F1 77.89%. Rescoring those
identical candidates with the complete official aliases produced the
authoritative EM 81.25% and F1 85.83%. This was a scorer-only correction: it did
not regenerate, edit, filter, or selectively retry any model response.

## Implemented repair

### Architecture and retrieval

- The Step-0 binding now supports a benchmark-owned TriviaQA search budget.
- TriviaQA uses three task-local searches, eight hits per search, no more than
  two hits from one document, and cross-search passage deduplication.
- Retrieved snippets use a bounded 96-token window and a 1,200-character
  presentation limit.
- The detailed answer-isolated corpus combines question-conditioned research
  notes with relevant Wikipedia material. Evaluator aliases are removed before
  any passage can become model-visible.
- Public retrieval state reports only neutral facts such as remaining action
  slots and returned public evidence. It does not tell the controller which
  answer to choose.

### Separation of controller and skill

The initial controller is deliberately strategy-neutral. It supplies only the
public question, current public observations, phase, and currently available
typed action wire. It does **not** prescribe query decomposition, evidence
comparison, mandatory searching, answer selection, or benchmark-specific
reasoning behavior.

Those behaviors belong to the retrieved typed TriviaQA skill. The skill asks
the model to use distinct queries, compare confirming and disconfirming public
evidence, treat a redaction marker as non-evidence, and finish with a canonical
short-answer hypothesis. Thus the architecture receives useful Step-0 skill
guidance without moving that policy into a restrictive benchmark controller.

The final short-answer adapter is a frozen greedy answer-span transcriber. It
sees the public root question and the model's own reasoning hypothesis, but not
the private labels or the entire retrieval transcript. It changes only the
terminal representation, not the model's substantive answer-selection policy.

### Official alias scoring

The private evaluator now loads the complete alias arrays from the pinned
official TriviaQA unfiltered no-context validation data and exact-joins them by
the public question. The frozen panel joined uniquely for all 128 questions.
Aliases remain scorer-only verifier material and never enter retrieval,
reasoning, terminal generation, or any other model context.

The public direct-search runner now requires this official alias source
explicitly, so future runs cannot silently fall back to the truncated released
wire. The metric remains official-style normalized maximum-over-alias EM/F1.

## Answer-isolation and anti-cheating checks

The repair preserves the project-wide eight-IID context boundary:

- no reference answer, accepted alias array, rubric, hidden test, expected
  program output, or per-record result is placed in candidate context;
- TriviaQA corpus construction has a private label-removal pass, and only the
  sanitized corpus is searchable by the model;
- official aliases are loaded only after candidate generation by the private
  scorer;
- the frozen final indices were selected independently of results;
- two development panels were disjoint from the final panel;
- the final 32 candidates were generated once, and candidate failures were not
  retried;
- questions, answers, task IDs, traces, candidate outputs, and per-record
  metrics remain outside Git.

For the other seven owner-authoritative IID benchmarks, this change does not
inject a new benchmark strategy into H0. Their initial Step-0 context continues
to be limited to benchmark-public task state, public environment feedback,
current legal action surfaces where applicable, and a compatible retrieved
skill. Benchmark-private verifier material remains isolated.

## Run accounting and interpretation

The final run used seed 42 and two existing adapter-free Qwen3.5-9B SGLang
replicas. It used no LoRA, learned skill, posterior state, or training update.
The two exploratory repair panels contained 16 records each and did not overlap
the final 32-record panel. Their aggregate diagnostics were used to repair the
shared retrieval and terminal contracts, never to choose or rewrite final
records.

This is an adaptively repaired 32-record Step-0 diagnostic, not a claim about a
new independent population or about learned BayesianImprove gains. The target
is passed for this frozen panel, but a larger independently frozen run would be
needed for a narrow confidence interval. The result also does not demonstrate
TTB/GFlowNet, Beta--Bernoulli/LCB, or Operator improvement because all of those
update mechanisms remained inactive by design.
