# Protocol 13 exact-eight final formal-training dataset

## Scope

The current IID catalog is exactly:

1. HotpotQA
2. TriviaQA
3. AIME 2026
4. HealthBench
5. WebShop
6. ALFWorld
7. MBPP+
8. HumanEval

The owner-facing label “MBPP+ hard” means EvalPlus MBPP+ v0.2.0 scored against both Base and
Plus tests. EvalPlus has no standalone split named `hard`, so the machine identity remains the
truthful `mbpp-plus`. SpreadsheetBench, SWE-bench, and AppWorld are not members of this dataset.

This is the owner-final training shape. It supersedes the earlier Protocol 13 draft that used 512
records per domain and an unconstrained global shuffle.

## Final schedule

| Control | Final value |
|---|---:|
| IID domains | 8 |
| Questions per domain | 250 |
| Persisted questions | 2,000 |
| Questions sampled per optimizer step | 8: exactly one per domain |
| Trajectories per question | 4 |
| Effective trajectory batch | 32 |
| Optimizer steps | 250 |
| Total trajectories | 8,000 |
| Checkpoint cadence | every 10 optimizer steps |
| Cadence checkpoints | 25 |
| Scientific seed | 0 |

The stored stream is **step-major and dataset-balanced**. Every consecutive eight records form one
optimizer step and contain exactly one question from every IID domain. Domain order within a step
and source order within each domain are deterministic seed-0-derived shuffles. A source population
smaller than 250 is repeated through deterministic shuffled cycles; every occurrence receives a
new episode ID.

At runtime, each eight-question step is expanded into four independently seeded trajectory passes.
The persisted artifact therefore contains 2,000 questions rather than copying the same verifier
payload four times, while the trainer consumes exactly 32 unique rollout bindings per update.

Preparing this dataset does not open the formal execution gate or itself start the 250-step run.
The runtime keeps three rolling per-step recovery snapshots and separately anchors
`cadence-step-00000010` through `cadence-step-00000250`; retention never removes those 25 cadence
products. The final successful snapshot is an additional named completion anchor rather than a
26th scheduled checkpoint.

## Unified record contract

Every line of private `training.jsonl` has the same four root fields:

```json
{
  "format": "skillev-private-protocol13-training-record@2",
  "episode": {
    "benchmark": "<closed Protocol 13 ID>",
    "population_id": "<training population>",
    "episode_id": "<unique question occurrence ID>",
    "source_id": "<private source ID>",
    "repeat_ordinal": 0,
    "block_position": 0,
    "optimizer_step": 1,
    "global_position": 0
  },
  "input": {
    "task_id": "<episode_id>",
    "environment_id": "<public environment identity>",
    "task_family": "<benchmark-namespaced family>",
    "context_id": "<training context identity>",
    "query": "<model-visible task>",
    "available_tools": [],
    "public_context": {},
    "action_surface": null,
    "budget_profile": null,
    "model_visible_messages": []
  },
  "output": {
    "evaluator_kind": "<closed trusted evaluator>",
    "target": {}
  }
}
```

`input` is the canonical answer-free `RolloutTask`. `output` is not a teacher-forced response: it
is the private answer, test, rubric, or native-environment route used only by the trusted terminal
evaluator after a trajectory completes. Policy code must never receive it. The artifact also emits
`model-inputs.jsonl`, an exact ordered projection containing only `episode_id`, `format`, and
`input`; this is the safe model-facing file.

## Source and input decisions

| Benchmark | Training source and model-visible input | Private evaluator output |
|---|---|---|
| HotpotQA | 250 released training records. Complete evidence paragraphs are embedded explicitly in `query`. | Accepted answer and supporting-fact metadata for official-style EM/F1. |
| TriviaQA | 250 released training records with question-only initial input. Retrieval is a runtime capability, not an answer-bearing field. | All non-empty accepted aliases for official-style max-over-alias EM/F1. |
| AIME 2026 | 250 historical AIME problems from pre-2026 years; the 2026 final set is excluded. | Exact accepted integer answer. |
| HealthBench | 250 Full-2025 records after excluding the official `Random(0)` 128-record final panel. Native message roles and turns are preserved. | Original per-case rubrics and conversation for the Qwen3.5-9B local judge. |
| WebShop | 250 official train goals after final-panel identity/text exclusion. | Native official-environment goal route and reward. |
| ALFWorld | 250 unique official train games with the canonical instruction unchanged. | Native game route and success target; final games remain disjoint. |
| MBPP+ | 218 training IDs after excluding the frozen final panel and deterministic 32-ID validation holdout; 32 repeated occurrences complete the lane. | Canonical solution and official Base+Plus test carrier. |
| HumanEval | 32 training IDs after excluding the frozen final panel and deterministic 4-ID validation holdout; 218 repeated occurrences complete the lane. | Canonical solution and original HumanEval test carrier. |

EvalPlus non-finite floating-point test values use the private
`skillev-private-nonfinite-float@1` tagged carrier. The trusted evaluator restores them immediately
before execution; they never appear in model input. It also applies EvalPlus' official
task-aware `mbpp_deserialize_inputs()` conversion before reference or candidate execution. This
restores serialized tuples, sets, and complex numbers instead of treating their JSON carriers as
the benchmark's runtime values.

## Expected materialized counts

| Benchmark | Questions | Unique source IDs | Repeated questions | Maximum occurrences/source |
|---|---:|---:|---:|---:|
| HotpotQA | 250 | 250 | 0 | 1 |
| TriviaQA | 250 | 250 | 0 | 1 |
| AIME 2026 | 250 | 250 | 0 | 1 |
| HealthBench | 250 | 250 | 0 | 1 |
| WebShop | 250 | 250 | 0 | 1 |
| ALFWorld | 250 | 250 | 0 | 1 |
| MBPP+ | 250 | 218 | 32 | 2 |
| HumanEval | 250 | 32 | 218 | 8 |
| **Total** | **2,000** | **1,750** | **250** | — |

## Training semantics

The BayesianImprove runner retains the method defined in `idea.tex`: forward/backward policies,
Tempered Trajectory Balance, log-Z, flow diagnostics, Beta–Bernoulli calibration, and the phase
transition Operator. It matches the applicable SkillFlow controls—four trajectories per question,
seed 0, outcome-only terminal rewards, beta 1.0, epsilon minimum 0.1, per-step adapter publication,
adapter/log-Z learning rates of 1e-4, and 10-step checkpoint cadence—while keeping
BayesianImprove-specific state and conservative
production diagnostic windows. Runtime-compatible Qwen3.5 LoRA and memory controls are explicit in
the preparation artifact rather than silently claimed as exact upstream SkillFlow parity.

The deployment throughput gate and answer-free per-phase telemetry are specified in
`docs/PROTOCOL13_FORMAL_TRAINING_PERFORMANCE.md`. They tune concurrency and distribute the same
sealed TTB batch across two training ranks; they do not change this dataset, its 8-by-4 sampling
shape, the rollout policy, or the one-update-per-step semantics.

## Isolation and validation

The materializer verifies:

- exact membership and order of the eight-domain catalog;
- 250 questions per domain and 2,000 total;
- every consecutive eight-question step has exactly one item per IID domain;
- unique episode IDs, contiguous global/domain positions, and contiguous repeat ordinals;
- identical root and `input`/`output` shapes across all records;
- exact equality between merged-record inputs and the safe model-only projection;
- structural absence of answers, rubrics, executable tests, native goal indices, game paths, and
  evaluator routes from `model-inputs.jsonl`;
- official task-aware MBPP+ input deserialization inside the trusted worker, after the answer-free
  model boundary;
- final-panel ID/content exclusion and strict JSON encoding;
- owner-only permissions (`0700` directory and `0600` files).

## Implementation and private location

- Schema, balanced selection, rollout expansion, writer, and validator:
  `skillev_private.benchmarks.protocol_v13_training`
- Source adapters and final-panel exclusion:
  `skillev_private.benchmarks.protocol_v13_training_sources`
- CLI: `scripts/materialize_protocol_v13_training.py`
- Public shape: `configs/evaluation/protocol_v13.yaml`

Questions, answers, tests, rubrics, source IDs, native routes, per-record outputs, and absolute
storage locations remain on approved private storage and are intentionally absent from Git. The
current private artifact location is recorded only in ignored `HANDOFF.md`.
