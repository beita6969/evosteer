# Fresh restart implementation status

This is implementation status, not an accepted A0 or a completed training run.
`idea.tex` is unchanged. Historical runs, profiles and their failures remain
historical evidence; none supplies the new optimizer or posterior state.

## Candidate and execution boundary

- `bayesianimprove_fresh_restart.yaml` explicitly selects the native two-phase
  wire, public task semantics, autonomous `catalog-then-read@1`, seven domains,
  seed zero, B28, 250 updates and checkpoint interval 10.
- Thinking is on only for AIME2026 and HealthBench. Static tasks have at most
  eight turns; ALFWorld has at most 25. Resolved per-domain budgets are saved.
- `zero-coverage-generate@1` is an explicit method extension. It does not replace
  the ordinary W50 joint detector or require artificial skill calls.
- The candidate serving profile reserves space for the largest declared
  reasoning output as well as the admitted input. It is not deployment evidence.
  BF16 and process seed zero are explicit launch arguments; unresolved
  `dtype=auto` is not treated as a measured BF16 deployment by IID acceptance.

## Implemented functional batches

| Modules | Actual implementation boundary |
| --- | --- |
| M01, M14 | Explicit fresh controls, original initialization, independent initial parameter reference, actual application-state inspection and resume binding to the original accepted A0. |
| M02, M03, M06 | Formal training and read-only IID share `execute_episode`; evaluation has separate records, arbitrary panel/chunk sizes, frozen controls and policy/library-only substitutions. Quality axes and baseline conditions are distinct. |
| M04, M05, M12 | Complete raw artifacts, commit indexing, real asynchronous durable mirror, controller-owned status, separated metric coordinates and original finalized phase/request accounting. Missing measurements remain unknown. |
| M08–M10 | Persisted catalog/body visibility, actual post-read context, multiple-read attribution, canonical source grouping and explicit cold-start/new-library lifecycle. No forced read quota. |
| M11, M13 | Action-failure classification from original records; separate native reward, binary success, structural/admission/execution metrics; opt-in actual F/B/Z and Adam update observations. |

At each complete boundary the controller first passes finalized timings to the
evidence observer, then indexes the commit. A periodic monitor is not the source
of truth for whether those phases reached the durable mirror. Mirror failure
pauses at a complete boundary and is not reported as successful evidence export.
Actor phase requests have exact batch/policy/library attribution. Legacy judge
rows and author costs without a corresponding request binding are reported as
unassigned/unknown rather than charged to a guessed step.

## Seven native input/scoring paths (M07)

All rows use the same single-owner native action/complete interface and preserve
original generated tokens. Targets and private scoring inputs stay outside the
owner context.

| Domain | Public input | Native terminal scoring |
| --- | --- | --- |
| HotpotQA | Original question and every supplied passage | Answer-blind short-answer projection; native EM/F1 |
| TriviaQA | Original released question and supplied reading context | Scorer-only aliases; native EM/F1, not fabricated closed-book equivalence |
| AIME2026 | Actual released 2026 problem and original source identity | Integer submission and exact native answer |
| HealthBench | Full original role/content conversation | Declared Luna-medium per-rubric judge; native score distinct from success |
| ALFWorld | Original public task and official reset/step observations | Independent official TextWorld episode and terminal success |
| MBPP+ | Exact official public prompt/interface/examples | Explicit shared EvalPlus Base/Plus profile and isolated scorer process |
| HumanEval | Original Python prompt | Answer-blind code projection and original isolated tests |

The new A0 is a fixed architecture-acceptance reference, not proof of an unseen
final population. Prior project evaluation exposure is retained separately from
the new training/development/quality exclusion lists. Historical AIME training
sources are not relabelled as the actual 2026 evaluation population.

## Owner-approved A0 acceptance condition

Before A0 sampling, the owner confirmed the HealthBench floor and revised the
HumanEval floor to 90 percent. The explicit rules are in
`configs/evaluation/architecture_matched_iid_a0_acceptance.json`:
HotpotQA F1 >=80, TriviaQA F1 >75, AIME2026 accuracy >=80%,
ALFWorld success >=80%, MBPP+ Base+Plus >=80%, HumanEval pass@1 >=90%,
and HealthBench Luna-medium rubric score >=45. The previous HumanEval
92.1875% reference is not this run's minimum. HealthBench acceptance belongs to
the declared Luna judge condition, not an assertion of equality to old judges.

The frozen population contains 32 sources per domain except the actual
30-problem AIME2026 population: 222 per arm. Skills-off and initial-library
use the same frozen architecture and sample coordinates; execution chunks of
32 are not training batches. Neither arm can update TTB, F/B/Z, posterior,
evolution or training evidence. Model failures remain in the denominators;
infrastructure failure stops progression to the next arm for repair.

## Remaining acceptance work

- Integrated CPU `make check` passed: 4,732 tests passed and 13 opt-in tests
  skipped; formatting, lint, mypy, both wheels and model-wheel contents passed.
  A separate CPU-only run with the deployed Transformers/tokenizer subsequently
  passed all 11 native-wire, shared-task-semantics and skill-body visibility
  cases that require private tokenizer assets. This does not qualify the two
  remaining CUDA-specific tests. The unchanged full suite was not repeated.
- A separate controlled development CLI now runs author/mutation and new-process
  catalog/read qualification. It recovers the original author response without
  regenerating it. Controlled visibility is not autonomous use, task utility,
  natural phase change or proof of skill usefulness.
- The real frozen-base author produced one valid Generate document; another
  process recovered the same response and mutation with zero new author calls.
  Its first read attempt failed before actor dispatch because the CLI omitted an
  explicit adapter-free gateway argument. The corrected constructor and narrowly
  scoped stopped/pre-dispatch recovery are covered by targeted CPU tests; the
  original failed attempt is retained.
- Fresh-F0 development collection has its own declared policy and output, not a
  relabelled adapter-free trajectory. The actual owner made four model calls over
  two turns, read the newly authored skill, and received its body in a subsequent
  admitted input. This was a controlled read request, not unprompted discovery.
- A separate real-Qwen CPU scorer used the original full edge objective and
  projection: four edge-direction calls took 322.39 seconds, with the independent
  initial F/B/Z parameter reference unchanged. The first new-skill development
  posterior event changed Beta(1,1) to Beta(2,1). Another process restored exactly
  the same projection with zero model calls or additional posterior updates.
  Its submission-presence label is explicitly not task correctness or skill
  utility. The projection ordinal is not an optimizer step; actual and formal
  optimizer steps remain zero, and formal posterior events remain zero.
- The two qualification test files passed 25 CPU
  tests on the remote Linux local filesystem; a WSL run expired without a test
  result and was not counted as a pass.
- New seed-zero F/B/Z initialization was saved on CPU with an independent
  parameter reference. A separate CUDA deployment binding uses those same
  tensors without reinitialization; this is not CUDA numerical qualification.
- The live candidate reports explicit BF16 and seed zero. Its actual initial
  forward adapter passed six fixed public base/F0/base route probes across
  thinking-on/off, with equal output token IDs. This is not a full gradient or
  task-capability qualification.
- Real source relationships, scorer/judge provenance and thresholds were frozen;
  both read-only seven-domain A0 arms completed 222 episodes. Neither arm met
  all frozen capability floors; collection completion is not A0 acceptance.
- The controlled author/read/development-posterior chain and process recovery
  above are qualified using non-IID material. They do not replace a complete real
  training transaction, Z-reset/Adam numerical qualification, autonomous useful
  skill use or production-window natural evolution. Synthetic coverage is
  labelled separately.
- Only after accepted A0 and the fresh-state checks, start the new training run
  on authorized resources. Report all complete steps, failures, recovery costs
  and quality observations; four steps do not prove W50 natural evolution.

No new A0 acceptance, formal updates, natural mutations or throughput gains are
claimed by this note. No private examples or per-question results belong in Git.

## A0 execution interruption and recovery (2026-09-13)

The skills-off arm collected 96 of 222 episodes before an official ALFWorld
environment-preparation failure stopped further dispatch. Its completed native
domain aggregates were HotpotQA F1 81.63, TriviaQA F1 83.93 and AIME2026 accuracy
73.33% (22/30). AIME did not meet its frozen 80% floor. The two completed
HealthBench observations are not a completed domain score; the other domains
and the initial-library arm were not evaluated. The remaining 126 entries are
missing/not-started observations, not 126 wrong answers or independent errors.

The actual failed collection took 3,125.98 seconds. A separate CPU-only native
initialization reproduced Fast Downward failing to remove a temporary directory
on network storage. With an owned Linux-local runtime `TMPDIR`, the same official
init/reset/close path succeeded on two cases in 1.55 and 1.67 seconds, with no
model or scorer requests. Runtime temporary storage is separate from checkpoint
publication staging. Neither evaluation controls nor native environment rules
were changed. The GPU7 SGLang service remained resident; formal learning remained
disabled. Original failed outputs and completed responses are retained for
explicit same-condition continuation, not replacement sampling.

The read-only collector now records the original exception and traceback in a
private chunk-level preparation event; unstarted episodes are not assigned a
fabricated failing task. Explicit continuation requires the original execution
request and unchanged panel/controls/snapshot, retains the original completed
artifact bytes and request identities, and restores the original settled budget
once. Pending/unknown requests and started-but-incomplete episodes are not
silently retried. The old output remains unchanged. The targeted 23 tests,
four-module type check and scoped formatting/lint passed; a CPU preflight with
the actual old output restored 96 artifacts and 204 settled entries without
model requests. This storage/recovery repair does not change frozen A0 scores
or authorize training after a failed capability floor.

The integrated Linux CPU `make check` for this recovery change passed: 4,740
tests passed, 13 opt-in cases skipped, with formatting/lint/type checks and both
wheels successful (pytest: 343.11 seconds). The unchanged deployed-tokenizer
qualification was not repeated. Actual AIME request telemetry confirms all 36
reasoning phases used native thinking-on with the frozen 16,384-token cap;
skills-off refers to the library axis, not to disabling thinking. Action phases
remain non-thinking submission phases, as in the frozen architecture.

### Completed two-arm A0 comparison

Both arms completed all 222 frozen episodes. No missing terminal labels
remained; the original failed attempt is retained separately. The controller
exited normally at 11:01:16 UTC with `complete-criteria-not-met`, not a training
launch. The same frozen architecture and base weights were used in both arms;
only the declared library axis differed.

| Domain | Skills off | Initial library | Frozen floor met in both |
| --- | ---: | ---: | --- |
| HotpotQA | F1 81.63 | F1 85.66 | Yes |
| TriviaQA | F1 83.93 | F1 83.48 | Yes |
| AIME2026 | 73.33% (22/30) | 76.67% (23/30) | No |
| HealthBench, Luna-medium | 30.54 | 30.95 | No |
| ALFWorld | 65.63% (21/32) | 59.38% (19/32) | No |
| MBPP+ Base+Plus | 84.38% (27/32) | 84.38% (27/32) | Yes |
| HumanEval | 90.63% (29/32) | 93.75% (30/32) | Yes, against the approved 90% floor |

Cumulative skills-off collection time, including the preserved original prefix,
was 5,948.79 seconds (99.15 minutes). Wall time from the first attempt's start to
that arm's completion was about 129.7 minutes, including interruption and repair.
Initial-library collection took 5,682.56 seconds (94.71 minutes). Together the
collection intervals total 193.86 minutes; first-attempt-to-final-completion wall
time was about 224.4 minutes. These are IID collection timings, not training
steps/hour or a measured speedup caused by skills.

For the skills-off diagnostic below, the 32 HealthBench actions were accepted
and ended normally, with no empty
outputs; their action-token lengths ranged from 137 to 738. All 32 reasoning
phases used thinking-on. These observations do not identify action truncation
or parsing as the cause of the lower rubric score, nor prove a causal comparison
with historical judges.

Neither arm performs learning or trains on IID evidence. The same three domain
floors failed in both arms, so these results do not authorize formal training
or automatic threshold/budget changes. Any revised candidate condition requires
an explicit decision rather than relabelling these results as accepted.

Read-only diagnosis of the original completed records further separates budget
and interface observations. All 36 AIME reasoning phases used thinking-on;
30 reached the 16,384-token limit, including all 10 phases belonging to the
eight unsuccessful episodes. Six action phases reached their output limit and
had invalid carriers; four belonged to episodes that subsequently succeeded.
This association does not establish that increasing the budget would fix the
answers. ALFWorld had 467 accepted actions out of 471, with four invalid-carrier
outcomes; all 11 unsuccessful episodes exhausted the frozen 25-turn budget.
Interface admission is not environment progress or task success. HealthBench
had three reasoning length limits but no rejected final actions.

The phase records contain cache/prefill token counts and server queue readings;
separate server prefill/decode compute-time fields remain unavailable, not zero.
Near-zero model-service stage-transition durations are not GPU compute time.
These observations neither change the frozen candidate nor constitute new
training evidence. The initial-library arm's first 64 completed episodes had
catalog exposure but no recorded skill invocation; visibility alone is not use.

Across all 222 initial-library episodes, there were five autonomous read events
in five trajectories, covering two distinct skills. AIME, HealthBench, ALFWorld,
MBPP+ and HumanEval each contributed one read event. Catalog entries were visible
in all 1,366 phase records; returned skill bodies were visible in 34 subsequent
phase records. The skills-off arm had no catalog exposure or reads. These are
mechanical discovery/read/body-delivery observations, not causal usefulness or
posterior evidence. Both isolation records confirm no TTB loss contribution,
F/B/Z training, posterior/evolution updates or training evidence writes.

W&B's stale current-summary aliases were corrected and read back separately from
preserved historical observations. An incomplete arm has pending acceptance,
whereas a completed arm failing its floors is complete and not accepted. The
resident GPU7 inference service is intentionally retained; formal training has
not started. A revised-candidate decision and an authorized formal multi-GPU
allocation remain outstanding, as do the real full-B28 optimizer transaction,
2+2 process-recovery qualification and the requested long-run acceptance work.

The original A0 evidence was also copied to local private storage in a
63,160,879-byte archive. Reading it confirmed both sets of 222 final artifacts
and preservation of the original failed attempt. Raw records remain outside
Git and W&B; no model calls, rescoring or hash checks were used for this backup.
