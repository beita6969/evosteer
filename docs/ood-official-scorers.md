# OOD scoring: upstream reuse, not a replacement agent

## Boundary and scope (2026-09-12)

Generation, reasoning, tools and final submissions remain owned by the project's
single frozen Qwen3.5-9B actor. This change does not start generation, change
thinking/budgets/prompts, add consultants, or expose evaluator data to that actor.
Only post-submission scoring and private environment evidence are changed.

The already inspected 64-item panels remain development evidence. Historical
A5/A6/A7 journals are not overwritten or combined. Scoring fixes do not justify
claiming that their scores reproduce a model card or constitute a new holdout.

### Future external judges: Luna medium via OpenAI API (owner update)

All **future external model judges** use `gpt-5.6-luna`, reasoning effort
`medium`, through OpenAI's official API. This is a new condition, not a rename
of historical Luna-high subagent scores. Qwen3.5-9B remains the only answer
owner. Native/rule scorers and frozen local-Qwen HealthBench conditions are
unchanged; the latter are not external-judge runs.

The answer-free defaults live in `evaluation/external_judge_policy.py`. The
private `evaluation/external_judge_api.py` constructs the official API client
from `OPENAI_API_KEY`, without embedding credentials in source/config, forwarding
them to actors, using a third-party base URL, or silently substituting models.
No secret from a conversation belongs in a tracked file. Rotate any key exposed
in chat, and set its replacement locally through a non-echoing prompt or a
private credential manager. API access/quota must be confirmed separately; the
mocked tests do not establish account access.

Omni uses the existing official-template export, then the separate terminal
runner below. Both paths contain private rubric/reference/candidate material and
must remain outside Git and the actor sandbox:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m skillev_private.evaluation.ood_api_judge \
  --batch "$PRIVATE_OMNI_BATCH" --output "$PRIVATE_OMNI_RESULTS"
```

- New exports/imports default to
  `omni-official-equivalence-prompt-luna-medium-api@2`; metric
  `luna-medium-api-equivalence-accuracy`. Freeze this profile in new scorer
  settings as `judge_profile` and use a new evaluation condition identity.
- Each judge request has one **8,000-token total completion cap**, effort medium,
  a 120-second request timeout, `store=false`, no tools, and no temperature/top-p
  override. It is a judge budget, not a change to the owner's code-task budget.
- API calls are sequential and have no hidden SDK/semantic retry. One transport,
  refusal, truncation or parsing failure stops further calls, preserves all
  planned rows and yields an incomplete batch, never a fabricated FALSE verdict.
  After inspecting the failure, explicit `--resume` retries only unresolved
  judging with the same original export. Both TRUE and FALSE resolved verdicts
  are retained, and every attempt remains recorded. Actor answers are never
  regenerated, repaired or selected by the judge.
- Private output retains the exact exported inputs, raw returned report, requested
  and returned model, effort, token cap, usage (including SDK usage details),
  response/request IDs, finish status, timings and failures. Unknown usage after
  a transport interruption stays unknown. Atomic private updates preserve progress;
  no row is silently removed from the denominator.
- Historical high imports require an explicit legacy `judge_profile`. Their
  metric/verifier remain high; aggregation rejects a high/medium mixture.
- For future **external** HealthBench runs, use
  `make_healthbench_external_sampler()` with `make_external_judge_client()` and
  the unchanged official per-rubric prompt/scoring code. Its declared profile is
  `healthbench-luna-medium-api-per-rubric@1`, not Protocol 14's historical low
  judge or the official GPT-4.1 score. It retains the existing bounded transport
  retries (maximum four, 120 seconds/request, 300 seconds total); no candidate
  repair is involved. Historical profile loaders are not the new-run defaults.

Official model/parameter references: [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
and [reasoning effort](https://developers.openai.com/api/docs/guides/reasoning).
This migration itself performs **no paid judge calls, candidate generation,
evaluation launch, or GPU/service changes**.

Migration validation: 119 scoped synthetic tests passed (5.76 s). Final
`CUDA_VISIBLE_DEVICES="" make check` on the approved 22049 Linux local disk
passed Ruff, mypy (625 source files), 4,412 tests (12 skipped), both wheels and
model-wheel separation; pytest took 307.10 s. Validation covered the committed
baseline plus this migration, not unrelated concurrent working-tree edits.
WSL pytest/type-check attempts had timed out during disk-bound startup, so they
were moved rather than repeatedly restarted locally. An initial remote type
check found one missing annotation; after its correction the final full check
passed, without separately repeating the already-passed scoped runtime suite.
Only self-review and the dedicated Git helper were used; no audit agent or
digest checks. Live API authorization, quota and actual judge outputs remain
unverified until a separately requested, credentialed run.

## Upstream implementation map

| Benchmark | Upstream code actually inspected | Project integration |
| --- | --- | --- |
| MuSiQue-Ans | [answer.py](https://github.com/StonyBrookNLP/musique/blob/main/metrics/answer.py), [evaluate_v1.0.py](https://github.com/StonyBrookNLP/musique/blob/main/evaluate_v1.0.py) | Private `benchmarks/musique_answer.py` adapts the complete answer metric; `score_musique_answers()` provides the thin typed wrapper. |
| NQ-Open | [FiD evaluation.py](https://github.com/facebookresearch/FiD/blob/main/src/evaluation.py), [harness task YAML](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/nq_open/nq_open.yaml) | Existing QA helper uses FiD-compatible normalization and maximum EM over aliases; no reader/retriever model or `has_answer()` is imported. |
| Omni-MATH | [GPT template](https://github.com/KbsdJames/Omni-MATH/blob/main/GPT_eval/gpt_evaluation_template.txt), [report parser](https://github.com/KbsdJames/Omni-MATH/blob/main/GPT_eval/get_result.py) | Private `evaluation/ood_luna_judge.py` renders the official template and frozen candidate; `ood_api_judge.py` calls terminal-only `gpt-5.6-luna`, effort `medium`. Older high results remain historical. |
| LiveCodeBench | [custom evaluator](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/runner/custom_evaluator.py), [testing_util.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py), [pass_k_utils.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/pass_k_utils.py) | `evaluation/ood_scoring.py::grade_code()` invokes the downloaded official `run_test()` on exactly the frozen task's tests, inside the existing sandbox. |
| APPS Intro | [official testing_util.py](https://github.com/hendrycks/apps/blob/main/eval/testing_util.py), [test_one_solution.py](https://github.com/hendrycks/apps/blob/main/eval/test_one_solution.py), [OpenCompass integration](https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/apps.py) | Same thin private wrapper, retaining native stdin/callable modes and comparison logic. No OpenCompass runtime is installed. |
| ScienceWorld | [official environment](https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py), [SwiftSage science_world branch](https://github.com/SwiftSage/SwiftSage/blob/science_world/eval_agent_fast_slow.py) | Official JVM environment and native `info["score"]`; private raw evidence alongside bounded reward. No SwiftSage dual-agent policy or score-preserving `no_stop` behavior is copied. |

These are the associated [MuSiQue paper](https://arxiv.org/abs/2108.00573),
[LiveCodeBench paper](https://arxiv.org/abs/2403.07974), and
[SwiftSage paper](https://arxiv.org/abs/2305.17390). Paper systems are useful
references for organizing runs; their model loading, training, planning policies
and reported scores are not automatically part of this project's condition.

### MuSiQue and NQ: exact answer scoring, not semantic matching

The MuSiQue port preserves normalization, repeated-token counts, maximum over
declared aliases, and the unrounded per-question mean. Its upstream empty-token
rule is important: two answers that both normalize to no tokens receive F1 1.
The old shared helper assigned F1 0 in that case. The new dedicated metric fixes
MuSiQue without changing the IID HotpotQA/TriviaQA rules. An absent model
submission remains an explicit candidate failure, not an invented empty answer.

The port retains the upstream AllenNLP attribution and includes the repository's
CC BY 4.0 license in the private package. Adaptations are typing, packaging,
equivalent normalization composition and removing the unused abstract base.
Support/sufficiency metrics are not claimed: this condition evaluates answers,
not predicted evidence indices.

NQ retains the project's QA implementation rather than copying FiD's complete
CC BY-NC 4.0 module. Its normalized alias EM is compared directly with FiD.
FiD uses the `regex` package's Unicode word boundaries; Python `re` differs on
combining accents. This difference was reproduced and corrected, with `regex`
declared explicitly as a dependency. NQ F1 remains supplementary; substring
retrieval recall is never substituted for EM. The harness's newline/period/comma
stops are **not** copied into an actor that may reason before its final answer.

### APPS and LCB: reuse execution, preserve protocol differences

Both official checkers were already in use before this round. The private source
snapshots avoid pulling obsolete model loaders or training dependencies into the
project. Existing frozen records identify tasks by source ID, not filtered row
number. This is a subset adapter to the official execution chain, not a claim
that the full-release `custom_evaluator` CLI was run on only 64 predictions.

- APPS receives a parsed `input_output` object; LCB receives its JSON string.
- `fn_name` distinguishes stdin and function/Solution invocation. The existing
  APPS `RuntimeModule` compatibility shim preserves `__main__` only for stdin
  programs. That earlier bug fix is retained, not claimed as a new discovery.
- A candidate passes only when its nonempty native verdict list is entirely
  positive. Negative error codes are not truthy successes. With one candidate
  per task, averaging these flags is the official pass@1 estimator in `[0,1]`.
- Per-test timeout now comes from the declared scorer settings, rather than
  silently ignoring that field. Defaults remain APPS 4 seconds / LCB 6 seconds.
  APPS's module-level timeout and LCB's function argument are both wired.
  Diagnostics retain effective per-test and outer wall limits.
- Error namespaces differ: APPS `-2` is compile/load failure; LCB metadata `-2`
  is wrong answer. APPS `-1` combines runtime error/timeout and is reported that
  way; no unsupported finer attribution is invented.
- Empty native verdicts, worker crashes and outer wall-budget failures remain
  infrastructure failures, not valid zero-score answers or silently omitted rows.

Generated programs still execute in bubblewrap with separate network, process
and filesystem namespaces, cleared environment, read-only mounts, a private
temporary filesystem, and CPU/memory limits. The checker's `reliability_guard()`
is not used as the security boundary. No credentials or other tasks' tests are
mounted. Official comparison quirks are retained rather than relaxed after
seeing a failed solution.

LCB's standard extraction takes the last fenced block; OpenCompass APPS takes
the first. The project's existing final-submission parser is not switched after
viewing scores, and frozen code is not re-extracted or repaired. Such extraction
changes would be separate generation/interface conditions, not scoring fixes.

### ScienceWorld: raw final score is not bounded reward

Official `reward` is a stepwise score delta; `info["score"]` is cumulative.
`done` can also indicate failure or a limit, not success. The bridge already used
the cumulative **last** score, not the delta or trajectory maximum, but clipped
negative values too early and discarded them.

The bridge now retains `raw_score` separately while preserving its existing
bounded reward for `TerminalReward`. New private OOD outcomes record:

```text
native_final_score       # raw final info["score"], including negatives
reward                  # max(0, native_final_score) / 100
success                 # score == 100, not done
native_steps
terminal_reached
terminated_by_horizon
termination_reason
```

If there was no native step, raw final score is unknown, not fabricated as zero.
Native termination without sufficient causal evidence is marked unspecified;
an early owner/budget stop is not automatically called the environment horizon.
Raw scores enter only private terminal evidence, never public observations.

The historical A5 **51.02/100 is a per-episode-zero-clipped final-score mean**;
it must not be renamed an unmodified raw mean. Lost historical negative scores
cannot be reconstructed from clipped zeroes. No environment replay or new model
generation is launched to manufacture missing evidence. Strict success remains
score 100; any score-70 success statistic is separately labeled.

### Omni: freeze the actual judge input; never shrink the denominator

The export now includes the exact template text and the fully rendered prompt
per frozen candidate, not just a loosely named profile or file path. Template
fields are replaced once, so text inserted from a problem is not templated again.
All of these files contain private reference material and remain outside Git.

The existing importer requires exactly one scoped, boolean verdict with an
explanation for each finalized candidate. Missing, duplicate or unparseable
judgments stop scoring; it does not copy the upstream parser's silent skips.
The model ID, effort, profile and original candidate ownership are retained.
Luna remains a terminal judge, not the official GPT judge or another answer owner.

[Omni-MATH-Rule](https://github.com/KbsdJames/omni-math-rule/tree/main/evaluation)
is a different, rule-compatible population/protocol. Its symbolic grader is not
used to replace this full-source panel's Luna scores. This round does not invoke
any judge or import new judgments.

## Verification without another generation run

`tests/evaluation/test_ood_official_scoring.py` contains synthetic regressions for
aliases, empty tokens, Unicode boundaries, error-code namespaces, timeout wiring,
raw-score privacy, final-vs-peak semantics and judge-template rendering.

`scripts/check_ood_official_scorers.py` additionally loads explicitly supplied
official source snapshots and compares QA metrics and code execution. It requires
these source filenames in a private directory (from the links above):

```text
musique-answer.py, musique-metric.py, fid-evaluation.py,
lcb-testing.py, lcb-passk.py, apps-testing.py
```

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/check_ood_official_scorers.py \
  --sources "$PRIVATE_UPSTREAM_SOURCES" --output "$PRIVATE_NEW_RESULT"
```

The script does not download or install repositories, import their model stacks,
run benchmark questions, or call a model. It exercises stdin, main guards,
functions, Solution methods, wrong answers, syntax/runtime failures and timeouts
through the real sandbox. Source retrieval dates/URLs and results stay private.
Use a new output path; do not overwrite a previous comparison.

Updated verifier identities distinguish the repaired implementation from older
frozen runs. Historical score files retain their original labels and values.
No score improvement should be claimed merely because these compatibility
checks pass; generation quality, sample exposure, model settings, source window
and judge differences remain separate limitations.

### Completed checks in this round

- Corrected targeted tests: **76 passed**. The initial expanded run had 110
  passes and three fixture failures; the old ScienceWorld stubs were updated to
  use the actual official result type with raw-score evidence.
- Upstream comparison: **171 QA comparisons and 16 real-checker synthetic code
  cases**, no differences. The production APPS/LCB source snapshots and Omni
  template also matched the newly retrieved upstream text.
- Scoring-only comparison: all **256 frozen A5/A7 QA scores were unchanged**;
  all **250 stored nonempty A5/A6 code verdict vectors** retained their pass
  flags under official positive-verdict aggregation. No code answers were
  re-executed or re-extracted, and no candidate was regenerated.
- One final remote Linux CPU `CUDA_VISIBLE_DEVICES="" make check`: **4,357
  passed, 12 skipped** (301.15 seconds for pytest); formatting, Ruff, mypy on
  616 source files, both wheels and model-wheel separation passed.
- Local WSL pytest hit its 120-second I/O timeout and was moved to remote CPU,
  not repeatedly retried locally. Documentation-only reporting did not trigger
  another full check. No independent review agent or digest check was used.

Real model/judge calls and new GPU tasks in this round: **zero**. The changes
establish scoring consistency; they do not raise the previously measured scores
or remove the remaining generation-quality and protocol-comparability limits.
