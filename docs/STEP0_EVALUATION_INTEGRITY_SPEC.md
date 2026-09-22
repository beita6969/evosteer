# Step-0 evaluation integrity

Method authority remains `idea.tex`, unchanged. This evaluation-only experiment does not
change TTB training, rollout semantics or checkpoint format.

## Conditions

Optimizer steps, native thinking, optional reasoning, topology and text skills are independent.
A1 is a single controller without skills; A2 permits independent peer conversations without
skills. Both can answer directly or voluntarily review their work, using the same total model-call
and output-token caps. Peers provide advice; **only the evaluated Qwen3.5-9B owner submits the
final answer or environment action**. Actual peer calls, not the arm's name, establish participation.
A3 is optional: A2 plus generic or public-capability-retrieved advisory text, with no other change.
Thinking-on is a separate matched experiment, including for AIME2026.

No voting, label-swapped selection, reference-answer fallback, targeted regeneration, model-based
serialization, hidden demonstrations, semantic action pruning or catalog autoplay is allowed.
No-skill bypasses both retrieval and advice injection, rather than merely omitting skill headings.
Historical composite implementations remain explicitly `legacy_*`, not current clean evidence.

The clean decoder is not forced into JSON: one explicit native `Action:` line or one JSON tool
call is accepted. The full tool schema describes capabilities, not a strategy or a decoder
restriction. Message/review acknowledgements state that no environment tool was invoked; only
actual native execution advances the environment revision.

## Boundaries and metrics

Actor workers have a private filesystem/process/network namespace. Only interpreter dependencies
and public Python source are mounted. Task-scoped standard-I/O RPC supplies public messages,
tokenization, generation and native environment observations. Targets, rubrics, tests, scores,
credentials and host network access are unavailable to actors and peers. Code evaluation uses
its own namespace after submission; hidden tests never become a repair signal.

Public source export is allowlist-based. HealthBench exports only conversation role/content.
HotpotQA retains all ten released distractor passages. TriviaQA's main lane uses the released
reading-comprehension context and question, **not** generated dossiers or label-redacted corpora;
official aliases are joined by public question text for the scorer only.

Native metric definitions follow [the benchmark guide](benchmark-evaluation-guide.md).
HealthBench retains negative item scores and clips only the panel mean; Qwen-local rubric grading
is not comparable to the official GPT judge. MBPP+ hard is the owner's display label for EvalPlus
MBPP v0.2.0 Base AND Plus; there is no distinct official hard split. HumanEval accepts one complete
module or one native completion through syntax-only projection. AIME accepts one unambiguous
integer, not the last of conflicting answers.

## Execution and interpretation

Insert-once private SQLite records retain unique final ownership, costs, ordered traces and
side-effect acknowledgements. Submitted candidates are never regenerated after grading.
Definitive scores are reused on resume; infrastructure failures do not become zeros or disappear
from the denominator. A missing action acknowledgement requires reconciliation, not blind replay.

Before scoring, all candidates in both arms are stored. Actual model/tokenizer/service, public
inputs, tools, environment, evaluator, parser, sampling and budgets must match. A2/A3 may differ
only in skill access. The full population is seven benchmarks × 128 plus AIME2026 × 30 = 926 per
arm, 1,852 for A1/A2. The historical panels are development-exposed, not pristine confirmatory data.

Exposure ledger: historical final-panel aggregates and failure trajectories informed earlier
development; complete per-person/per-item exposure to outputs, gold, tests and rubrics cannot be
reconstructed and is marked unknown, not unseen. This repair uses synthetic cases and explicitly
final-disjoint canaries for communication debugging. Canary actor traces were inspected, but their
quality scores were not used for prompt/condition selection. Evaluator loading of targets is
scorer-domain access, not actor access. All new per-item traces stay in private run directories.

Report exact paired joins, native metrics, cost, descriptive paired bootstrap intervals and binary
McNemar tests (with an eight-comparison Bonferroni value). Integrity, execution, communication,
performance and external comparability are separate statuses. A clean negative result remains
reportable; higher scores are not an integrity requirement.
