# Qwen3.5-9B Protocol 13 Exact-Eight Implementation

## Scope

Protocol 13 freezes the current IID catalog to exactly:

1. HotpotQA
2. TriviaQA
3. AIME 2026
4. HealthBench Full 2025
5. WebShop
6. ALFWorld
7. MBPP+
8. HumanEval

Seven panels contain 128 records and AIME 2026 contains all 30 records, for 926 final
records. The owner-final training shape is 2,000 questions: 250 per domain in a step-major
balanced schedule. Each step uses one question per domain and four trajectories per question,
giving 32 trajectories per update, 8,000 trajectories over 250 optimizer steps, and 25 cadence
checkpoints at 10-step intervals.
SpreadsheetBench, AppWorld, and SWE-bench implementations remain available for historical or
diagnostic use but cannot enter the Protocol 13 catalog, target registry, receipt composer, or
formal gate.

The runtime additionally enforces the 72-hour throughput floor documented in
`docs/PROTOCOL13_FORMAL_TRAINING_PERFORMANCE.md`: two warmup steps, at least three measured steps,
a hard floor of 3.4722 steps/hour, and a deployment target of 3.8 steps/hour. This is an execution
readiness gate, not a change to the scientific protocol.

## Implemented contracts

- `src/skillev/evaluation/current_iid/protocol13/` owns an immutable explicit catalog, strict
  execution contracts, target registry v4, conserved public receipts, fail-closed admission,
  aggregation, gate evaluation, and deterministic answer-free rendering.
- `src/skillev/experiments/protocol_v13.py` loads the exact-eight source roles without changing
  Protocol 12 compatibility code.
- `configs/evaluation/protocol_v13*.yaml` and `qwen35_protocol13_iid*.yaml` freeze the catalog,
  three population roles, runner profiles, conditions, target identities, and target status.
- The composer accepts only eight Protocol 13 receipts. A receipt is rejected for any execution,
  denominator, projection, formula, aggregation, metric-set, population, or infrastructure
  mismatch. The parity threshold is strict: a gap below 7 pp passes; exactly 7 pp fails.
- Private adapters join per-task outcomes against a frozen manifest before emitting only counts,
  metrics, provenance, and diagnostics. Questions, answers, rubrics, tests, traces, and task IDs
  remain outside Git.
- Interactive evaluation now has a durable task-keyed journal. Only typed infrastructure
  outcomes may occupy a retry slot; candidate-invalid, horizon, native failure, partial reward,
  and success outcomes are immutable.

## Benchmark repairs

### HotpotQA and TriviaQA

- HotpotQA uses the released full-context prompt, checks context structure before generation, and
  applies the official special-answer behavior for `yes`, `no`, and `noanswer`.
- TriviaQA preserves every non-empty unique released alias rather than truncating to five.
- Both lanes retain deterministic, answer-blind parsing and official-style normalized max-over-
  alias EM/F1 scoring.

### AIME 2026

- Dataset year and population role are typed evidence rather than substring heuristics.
- Exactly one externally evidenced formal profile is required.
- The owner-defined current minimum score goal is 80.00%; architecture non-regression against a
  paired backbone remains a separate gate.
- The formal prompt/parser pair requires the final complete boxed integer in `[0, 999]`; decimals,
  out-of-range values, and incomplete boxes are rejected rather than coerced.

### HealthBench

- Candidate generation is a direct, non-thinking Qwen completion with the official helpful-
  assistant system message and official-like generation configuration.
- The owner-required Qwen SGLang rubric judge is physically separated from candidate records and
  uses pinned `simple-evals` `HealthBenchEval.grade_sample()` semantics.
- Candidate and grade journals are independent. Per-row transport and local implementation
  failures are typed and are never converted to zero scores.
- The reported headline is mean-then-clip `native_rubric_mean`; it is explicitly not an official
  GPT-4.1-comparable score.

### WebShop and ALFWorld

- Both use pinned official environment processes and native terminal rewards.
- The v3 ReAct interface returns public observations and declared actions to the actor, performs
  reset/step/outcome/close preflight, preserves horizon as a definitive candidate outcome, and
  records trace-funnel diagnostics.
- WebShop reports native average score and exact native success; ALFWorld reports exact native
  success over the frozen seen/unseen manifest.

### MBPP+ and HumanEval

- MBPP+ uses a frozen result-blind `Random(0)` 128-task manifest, a benchmark-specific deterministic
  profile, the official EvalPlus full carrier, and separate generation-infrastructure,
  candidate-invalid, Base failure, Plus-only failure, and scorer-infrastructure outcomes.
- HumanEval selection is manifest-authoritative by official task ID and preserves original source
  positions even if the source file order changes.

## External implementations used as behavioral references

The implementation follows small, independently rewritten behavior patterns rather than copying
whole external runners or importing their reported scores as targets.

| Area | Reference code | Adopted boundary |
|---|---|---|
| QA normalization | [Search-R1 `qa_em.py`](https://github.com/PeterGriffinJin/Search-R1/blob/598e61b/verl/utils/reward_score/qa_em.py) | Article/punctuation/whitespace normalization; not its answer-block extractor |
| Alias aggregation | [Search-o1 `evaluate.py`](https://github.com/RUC-NLPIR/Search-o1/blob/c76a700/scripts/evaluate.py) | Maximum EM/F1 over all aliases; not substring accuracy or greedy boxed parsing |
| AIME identity | [tinker-cookbook AIME evaluator](https://github.com/thinking-machines-lab/tinker-cookbook/tree/9dfcc3a) | AIME 2026 identity and 0–999 integer domain; no float coercion |
| Native WebShop/ALFWorld metrics | [AgentSquare](https://github.com/tsinghua-fib-lab/AgentSquare/tree/8f5b3fe/tasks) | Native environment reward/success aggregation; no restricted top-three wrapper |
| Evaluation/training reward split | [verl-agent environments](https://github.com/langfengQ/verl-agent/tree/20bd331/agent_system/environments/env_package) | Native evaluation score remains distinct from shaped training reward |
| HealthBench | [OpenAI simple-evals `healthbench_eval.py`](https://github.com/openai/simple-evals/blob/652c89d/healthbench_eval.py) | Pinned official rubric prompt, per-rubric grading, and mean-then-clip aggregation |
| Health diagnostics | [SERPO HealthBench evaluator](https://github.com/chiefovoavicii/SERPO/tree/main/eval) | Per-sample grader observability; not its batch prompt as the formal scorer |
| MBPP+ | [EvalPlus evaluator](https://github.com/evalplus/evalplus/tree/26d6d00/evalplus) | Official data loader, Base+Plus execution, isolation, and verdicts |
| Adapter boundary | [EvoAgentX benchmarks](https://github.com/ANative-Lab/EvoAgentX/tree/main/evoagentx/benchmark) | Separation of public task, private target, and scorer; ordinary MBPP is not substituted for MBPP+ |
| ALFWorld taxonomy | [StateAct](https://github.com/ai-nikolai/StateAct) | Per-task-type diagnostics; no action correction or JSON repair |

The broader 2025–2026 conference and repository review remains in
`docs/EXTERNAL_AGENT_BENCHMARK_EVALUATION_CODE_RESEARCH_2025_2026.md`.

## Target and gate interpretation

- Exact or matched targets may enter the numeric gate only when every execution-contract field
  matches.
- HealthBench's public Qwen anchor is `reconstructed-external`; WebShop, ALFWorld, and HumanEval
  retain reconstructed comparability; these labels are not silently upgraded.
- MBPP+ remains `undefined` until an independently frozen matched reference is established.
- Consequently, implementation conformance and a complete 926-record diagnostic do not by
  themselves authorize SKILLEV formal training. The gate fails closed while any target is
  undefined, non-matched, infrastructure-incomplete, or at least 7 pp from its target.

## Post-repair runtime authority

- The native static and interactive runners consume `ExecutionContractV3` directly. Runtime
  generation, parser, environment, grader, population, horizon, and evaluator identities have one
  manifest-owned source rather than a compatibility YAML that can silently diverge.
- Execution start/completion journals and service snapshots are produced by the running client;
  receipt builders no longer accept caller-invented execution revisions. Multi-endpoint routing is
  assigned from the frozen record ordinal, not from coroutine lock acquisition order.
- Token-budget preflight passes the frozen `enable_thinking` value into the Qwen chat template.
  Interactive history is a contiguous newest suffix and therefore cannot keep an older short turn
  after dropping a newer long one.
- Target evidence v4 separates exact-panel evidence, protocol-incomplete paper reports, external
  aggregate anchors, diagnostics, and undefined targets. A successful repeated local run cannot
  be promoted into the reference target for the same candidate.

## Returned-state ReAct contract

For both interactive benchmarks, every transition persists the pre-action public state, emitted
action, post-action public state, and returned available/admissible actions. The task and a
contiguous suffix of complete transitions are rendered on the next turn. ALFWorld includes the
task on the first turn. WebShop and ALFWorld demonstrations are successful train-environment
replays, and their action validity is checked against the recorded native surface. These are
ordinary execution requirements, not answer-bearing memory or action correction.

## Code execution profiles

- MBPP+ pins EvalPlus revision `26d6d00`, creates the official evaluator environment once, keeps
  generation and scoring resumable as distinct phases, supplies the selected dataset explicitly,
  and maps the evaluator's native `pass`/`fail` statuses into Base/Plus outcomes.
- HumanEval records the Python executable, timeout, CPU, memory, process, and original-test profile
  in the execution condition. The frozen 128-ID original HumanEval panel cannot be substituted
  with HumanEval+ tests.
