# Qwen3.5-9B paper-direct reference protocol

## Scope and scientific identity

This is a separate **Paper Direct-Qwen Reproduction Track**. It never enters the
TTB training path and does not alter the training distribution described in
`idea.tex`. The model is the adapter-free `Qwen/Qwen3.5-9B` base route: no LoRA,
learned skill, retriever, posterior, or Z head is constructed. Benchmark-native
environments remain available where the released protocol requires interaction.
In particular, SWE-bench reuses the locked SkillFlow repository environment with
an empty skill workspace, but the formal Direct-Qwen lane exposes only read-only
repository navigation followed by one final unified-diff submission. A second
MExec model call and incremental editable-workspace feedback are disabled; this
is not the project's learned-skill rollout codec.

The existing Protocol 10 result remains a structured-agent control and is not
renamed or rescaled. Scores from these two lanes must not be compared without an
explicit protocol label.

The authoritative **project IID catalog** is now: HotpotQA, TriviaQA, AIME 2026,
HealthBench, WebShop, ALFWorld, SpreadsheetBench, SWE-bench, and HumanEval. Older
artifacts that call MedQA IID, call HumanEval OOD, or substitute AppWorld/MBPP+
for SWE-bench/HumanEval are superseded and require catalog migration before a
new all-IID aggregate is published. This SWE-only run already classifies
SWE-bench as IID and is unaffected by that separate migration.

## Locked upstream evidence

| Item | Lock |
|---|---|
| SkillFlow repository | `https://github.com/beita6969/SkillFlow` |
| SkillFlow revision | `74be52bb6bd9f0e9e68dacb72636b75649197983` |
| Released IID data revision | `07bb38bcc62fa8bebab6af86c39ba23b0293c97d` |
| Preparation RNG | one shared Python RNG, seed 42, source order from `data/prepare_v3.py` |
| Base model | `Qwen/Qwen3.5-9B`, adapter-free route |

The released IID file contains exactly 128 examples for each non-AIME family and
all 30 AIME 2026 questions. The seven OOD task-ID panels, exact paper prompts,
and exact decoding arguments are not published in the locked repository, the
released dataset, or the paper source. The inaccessible anonymous artifact is
not treated as evidence. Consequently:

- the released static IID panels (HotpotQA, TriviaQA, AIME, MedQA, and
  SWE-bench) retain their published task identities and are labeled
  `exact-paper` at the population level;
- released WebShop rows preserve goal text but not the official environment
  goal payload, and released ALFWorld rows are synthetic templates rather than
  game identities, so their official-valid reconstructions are labeled
  `approximate-only`;
- OOD panels rebuilt from pinned official splits with seed 42 are labeled
  `approximate-only` until author-published IDs are recovered;
- no approximate OOD score can satisfy the exact-paper Go gate, even if its
  numeric gap is strictly below the Protocol's seven-percentage-point threshold.

This label is deliberately conservative and must not be silently upgraded.

## Population and answer isolation

Answer-bearing records, task questions, answer aliases, patches, environment
state, and per-task outputs remain in the private evaluation workspace. Git may
contain only source/revision names, aggregate counts, public prompt templates,
and aggregate metrics. The model-facing request is built from the public half of
each case; the private target is passed only to an official scorer after
generation. A target may never influence parsing, candidate selection, retries,
or prompt construction.

Every benchmark uses at most 128 records; AIME uses its complete 30-record
population. Sampling is result-blind. A panel is frozen before any final answer
is generated, and a candidate failure is not retried.

For HotpotQA specifically, the released record's pre-rendered question is not
the complete model input: the upstream preparation script retains only the first
300 characters of each distractor passage in that field. The executable
`hotpotqa-full-context@1` contract instead extracts the final question and
supplies all ten complete public context passages from the same frozen record.
The private target and supporting-fact annotations are not consulted while
building the request.

## Direct request contract

For ordinary static tasks, `DirectGenerationClient` sends one
OpenAI-compatible base-model request with all decoding controls explicit:
thinking mode, temperature, top-p, top-k, min-p, presence penalty, repetition
penalty, output budget, stop list, and seed. The served model name rejects
adapter-, LoRA-, and skill-like routes. Static prompts produce native text, a
choice label, an integer, boxed math, or source code—never the project's action
JSON codec.

SWE-bench is the one static-population exception. It is a 28-step, single-Qwen
repository-agent episode with read-only `list_files`, `search_code`, and
`view_file`. The adapter-free base route uses non-thinking generation,
temperature 0.8, a 512-token action budget, and the declared seed 42 on every
request. The skill workspace is empty and MExec is replaced by a fail-closed
disabled executor. Requested tool observations contain no environment-added
progress ledger, issue-derived member hints, or repeat steering. The model
eventually submits one final unified diff; it receives no edit/syntax/test
feedback during inspection and never receives a gold patch, official test,
gold-affected-file hint, or scorer result. All model-supplied read paths resolve
inside the episode's isolated worktree. The issue-only one-shot, delegated
supervisor+MExec, and incrementally editable agent runs remain diagnostic
controls, not the formal single-model Direct-Qwen parity score.

The profiles are derived from the official Qwen3.5 model-card recommendations:

- general thinking: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5;
- precise code: temperature 0.6, top-p 0.95, top-k 20;
- general non-thinking: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5;
- math has an 81,920-token output ceiling; ordinary tasks use a smaller frozen ceiling.

The full machine-readable values are in
`configs/evaluation/qwen35_skillflow_direct_reference.yaml`.

The YAML is the sole executable binding for model identity, population,
prompt, decoding, parser, scorer, metric, seed semantics, and aggregate rows.
Runtime code resolves versioned registries from those IDs and rejects unknown
or duplicate bindings rather than maintaining a second benchmark map.

## Benchmark-native contracts

| Family | Model output | Scoring/environment |
|---|---|---|
| HotpotQA | all ten complete public context passages plus the question; concise final answer | dataset-native normalization, EM and token F1 |
| TriviaQA, MuSiQue, NQ-Open | concise final answer | dataset-native normalization, EM and token F1 |
| AIME 2026 | last explicit final integer in 0–999 | official integer accuracy |
| MedQA, GPQA Diamond | one A–D label with frozen option order | exact accuracy |
| MATH-Hard | one final boxed expression | official symbolic-equivalence bridge |
| HumanEval | executable Python source, no JSON escaping | official execution harness, pass@1 |
| SWE-bench Verified | 28-step no-skill read-only repository agent; one final unified diff | official Docker harness; infrastructure failure is not unresolved |
| WebShop, ALFWorld, ScienceWorld | one native ReAct command per turn | official environment and terminal reward |
| Mind2Web | native offline element/operation/value prediction | official candidate and action scorer |

Interactive histories are sequential within a trajectory. Different trajectories
may run concurrently, but completion order never changes panel or aggregation
order. Invalid native actions are candidate failures; they are not rewritten or
resampled. Environment and scorer failures are infrastructure failures and do
not become zero-score candidates.

## Parser rules

Parsers may remove a public final-answer marker, a thinking block, whitespace, or
a Markdown code fence. They may call the official normalizer. They may not see a
target, choose among contradictory candidates using correctness, enlarge answer
aliases, repair actions, or replace an official scorer with an LLM judge. The
SWE repository-agent path accepts only one unambiguous unified diff after
read-only inspection; prose and contradictory multiple patches are candidate
failures rather than repaired or selected using scorer feedback.

## Reporting and gate

For every metric the report includes reference, full-precision and displayed
observed percentages, absolute percentage-point gap, population, planned and
scored counts, comparability, prompt/decoding profile, seed semantics,
submission rate, scorer or terminal reach, native-action validity when
applicable, and infrastructure failures. Numeric parity requires an absolute
gap **strictly below 7.0 points**. A diagnostic mean over a partial population
is never a formal score. Scientific parity additionally requires exact-paper
identity and known comparison semantics.

An IID-only or OOD-only report has gate status **NOT-EVALUATED**: a scoped
report cannot decide formal training. Only an all-scope report that exactly
covers all 14 declared benchmarks may emit GO or NO-GO. At that point GO also
requires all exact panels, all required metrics strictly inside the threshold,
zero infrastructure failures, known comparison semantics, and a passing final
CPU `make check`. This evaluation condition does not change the scientific
training method.

## Unresolved paper-protocol questions

The locked public SkillFlow repository, released IID dataset, and paper source
do not answer every direct-baseline question. The table distinguishes the
frozen executable reconstruction from paper evidence; `unknown` is intentional
and prevents result-aware protocol selection.

| # | Question | Frozen executable contract | Paper evidence status |
|---:|---|---|---|
| 1 | Thinking mode per benchmark | AIME, MedQA, MATH-Hard, GPQA, and HumanEval use thinking; QA and native interactive profiles use non-thinking. SWE-bench's single repository agent is non-thinking. | Exact per-benchmark mapping: `unknown`. |
| 2 | Temperature, top-p, top-k, penalties, and token budget | Versioned profiles freeze every field in the YAML: general thinking `1.0/0.95/20`, code `0.6/0.95/20`, non-thinking `0.7/0.8/20`; math allows 81,920 tokens. SWE uses the frozen 512-token per-action budget. | Exact paper request arguments: `unknown`. |
| 3 | System+user versus single-user prompt | Static renderers emit one public system message and one answer-free user message. SWE-bench instead uses the released repository-agent message/tool loop. | Exact paper role layout: `unknown`. |
| 4 | Multiple-choice output contract | MedQA and GPQA require one final A–D label; JSON is not accepted by these profiles. | Exact paper output wire format: `unknown`. |
| 5 | HumanEval deterministic decoding | Greedy, temperature 0, top-p 1, top-k 1, one completion, thinking enabled, 32,768-token ceiling. | Exact paper service parameters beyond “deterministic”: `unknown`. |
| 6 | Runs and seeds behind table `±` | One frozen reconstruction run uses seed 42. | Run count, seed set, and reducer: `comparison-semantics-unknown`. |
| 7 | Exact 128 OOD task IDs | Explicit private manifests freeze reconstruction IDs before generation. | Author-published exact panels: `unknown`; OOD remains `approximate-only`. |
| 8 | WebShop goals, corpus, search backend, horizon | Official-valid seed-42 reconstruction with the pinned 100k corpus/search backend and a manifest-frozen horizon. | Exact goals, backend artifact, and horizon: `unknown`; result remains `approximate-only`. |
| 9 | ALFWorld games, split ratio, and horizon | Manifest freezes official valid seen/unseen games and per-case horizons before generation. | Exact game IDs, ratio, and horizon: `unknown`; result remains `approximate-only`. |
| 10 | ScienceWorld tasks, variations, simplification, and success | Manifest freezes official task/variation identities; the official environment supplies terminal reward. | Exact paper panel and simplification: `unknown`; result remains `approximate-only`. |
| 11 | Mind2Web split, sampling unit, candidate top-k, and Action F1 string | Explicit step manifest and `operation-value-string@1` component contract; scorer reports component metrics separately. | Exact split quotas, task/step unit, top-k, and string reducer: `unknown`; result remains `approximate-only`. |
| 12 | SWE-bench variant and direct input/tool contract | Released IID IDs are validated against SWE-bench Verified. The frozen executable contract supplies single-model read-only full-repository navigation for at most 28 steps, an empty skill workspace, seed 42 on every request, and one final unified diff for the official evaluator. | The public environment is known, but the paper does not establish whether its direct row used issue-only generation, delegated MExec, editable tools, or read-only inspection; comparison semantics remain `unknown`. Evaluator outages never become zero. |

Therefore a numerically close single run is only a numeric diagnostic wherever
comparison semantics are unknown. It cannot upgrade an approximate panel or
make scientific/formal training GO.
