# A2 IID: retained results and post-run communication repairs

## Status and scope

All **926 A2 answers and 926 native scores** completed; the report was written at
22:24:30 UTC on 2026-09-06. Only A2 was evaluated under the owner's narrowed scope:
seven 128-task panels and all 30 AIME2026 tasks. Generation uses seed 0; existing
panel identities, including the historical HumanEval seed-42 selection, are retained.
The evaluated actor/parser is `1d09159`; parallel scoring changes are `349b084`.

**These are the preserved results before the final communication repairs, not
post-repair scores. Execution completed, but communication/performance acceptance
has not passed. No benchmark answer was replaced, rescored with the new parser,
voted on, or selected using a reference. A new full post-repair run was not launched.**

The population is **development-exposed**. A1 and A3 were not evaluated in this
final scope, so these numbers establish neither architecture gains nor skill gains.

## Native results

All displayed values below are percentages of the stored native fraction.

| IID benchmark | N | Native primary metric | Result |
|---|---:|---|---:|
| HotpotQA | 128 | Answer F1 | 25.83 |
| TriviaQA | 128 | Answer F1 | 30.35 |
| AIME2026 | 30 | Accuracy | 13.33 (4/30) |
| HealthBench | 128 | Local-Qwen rubric mean | 41.66 |
| WebShop | 128 | Native mean reward | 25.91 |
| ALFWorld | 128 | Success rate | 41.41 (53/128) |
| MBPP+ hard* | 128 | Base AND Plus pass@1 | 61.72 (79/128) |
| HumanEval | 128 | Original-test pass@1 | 80.47 (103/128) |

Secondary metrics: HotpotQA EM **12.50%**, TriviaQA EM **15.63%**, WebShop success
**12.50%**; ALFWorld **seen 45/97 = 46.39%**, **unseen 8/31 = 25.81%**.
MBPP Base alone passes 96/128 = 75.00%, but Base-only success is not the primary
Base+Plus metric. HealthBench project binary SR is **28/128 = 21.88%**, using native
score >= 0.60 AND no triggered negative rubric; 64 graded cases trigger 91 negative
rubrics. Its two empty candidates remain failures in the 128 denominator.

*The owner calls this lane MBPP+ hard. The actual frozen population is EvalPlus
v0.2.0 Base+Plus; no independent official Hard split has been defined. HealthBench
uses the local Qwen judge and is not an official external-judge-comparable score.

## Observed defects, not an explanation based only on low scores

The run contains 7,903 completed model calls, 2,453 native executions and 1,620
actual independent peer calls. Recorded surface, execution, acknowledgement,
observation and peer-route mismatch counts are zero. These consistency checks
were **not sufficient** to establish working communication in every case:

- AIME has **20 attempted peer messages**, all undecodable as JSON: 19 invalid
  backslash escapes and one literal control-character/newline error. None was
  delivered. There are **zero realized AIME peer calls**, despite the A2 configuration.
- **26/30 AIME finals were candidate failures**, not cleanly scored mathematical
  answers. At least 20 intended requests were lost at the control-message boundary.
  Three responses also exhausted the 81,920-token output budget. The retained
  4/30 must not be interpreted as a clean measure of backbone math capability.

The subsequent repair accepts `Message to solver:` / `Message to researcher:` and
`Review:` with ordinary inline or multiline text, preserving math, code, quotes and
newlines without JSON string escaping. Valid JSON remains compatible. Explicit
malformed controls now generate visible delivery-failure feedback inside the same
call/token budget instead of silently becoming empty final answers. The new
`control_parse_failures` counter makes this boundary observable.

A real **synthetic**, non-IID Qwen canary then completed one peer request/reply in
9.09 seconds, preserving the mathematical backslash. It used four model calls and
220 output tokens. It exposed a second bug: the owner explained its computation
and put the correct integer alone on the last line, which the old terminal parser
discarded. The final patch accepts that explicit last-line field while still
rejecting conflicting explicit finals. The **same saved synthetic owner response**
passes the corrected CPU parser; no further GPU candidate was generated. The first
canary's failed terminal result remains recorded rather than being overwritten.

## Throughput, cost and resources

- Three inference services on approved endpoint 22048, physical GPUs **1, 3, 7**.
  No training or gradient job was launched. Endpoint 22049 was used for CPU tests
  with `CUDA_VISIBLE_DEVICES=""`, not GPU inference.
- Capacity changed to 48 live requests per replica, automatic KV sizing and 0.82
  static-memory fraction; coordinator concurrency reached 96. Model, native
  thinking-off setting, sampling, per-task budgets, FA3 and disabled CUDA graphs
  were unchanged during the retained evaluation. Observed per-card total memory
  was about 68 GiB; active utilization samples reached 80–100%. Pool allocation
  is not active KV occupancy, and the final serial trajectory could not fill three cards.
- The capacity continuation retained all 776 completed A2 answers. It produced
  103 more in about 142 seconds, but long AIME/ALFWorld outputs dominated the tail.
  Its complete 150-answer generation phase averaged about **135/hour**, not the
  roughly 2,600/hour short-window rate. Slow-tail alerts were reported rather than
  cutting budgets or adding redundant candidate votes.
- Scoring continuation retained **800** completed native scores. The remaining
  **126** scored at about **3,022/hour** (roughly 150 seconds of scoring); three-replica
  rubric routing recorded 46/39/41 cases. Report construction finished afterwards.
- Completed A2 records contain **15,702,186 input / 3,118,150 output tokens**.
  Completed grading records contain **1,476 model requests**, **2,315,880 input /
  157,963 output tokens**, and three semantic rubric repairs. These totals exclude
  discarded A1 work and interrupted attempts from earlier coordinators; unknown
  usage from those attempts, including any unjournaled interrupted grader, is not zero.
- The three 22048 services remain intentionally online. The full-run coordinators
  and the one synthetic-canary process exited. Service PIDs, logs and precise
  launch mappings remain in private operational records.

## Verification and workflow

Implementation self-checks were used; no review/audit agent or hash check was added.
The existing requested GPU watcher remained read-only. Local Ruff was used for edits;
CPU test work ran on the approved Linux-local workspace to avoid the observed WSL
E-drive startup stalls.

| Changed batch | Targeted tests | CPU-only `make check` milestone |
|---|---|---|
| Capacity-compatible answer reuse | 5 passed | 2,638 passed; 133.41 s pytest |
| Bounded scoring and judge replicas | 25 passed; 1.81 s | 2,642 passed; 123.80 s pytest |
| Plain-text control messages | 132 passed; 1.50 s | 2,653 passed; 118.82 s pytest |
| Real-canary final-line defect | 117 passed; 0.43 s | **2,659 passed; 118.45 s pytest** |

Each successful `make check` also passed Ruff, mypy, both wheel builds and the model
wheel boundary check (exit 0). Additional full checks followed actual cross-module
changes/observed defects, not repeated inspection of unchanged code. One targeted
launch used the wrong environment and failed before pytest; it is not counted as
a pass. The unchanged 128-task panels and old GPU canaries were not rerun, and no
separate E2E suite or duplicate full verification was added after the final success.

The remaining risk is explicit: **the repaired communication and terminal parsing
have not received a new full IID performance evaluation**. This report preserves
the failed condition rather than tuning the exposed panel until its score improves.
See the [answer-free aggregate record](machine-results/step0_a2_iid_retained_results_2026-09-06.json).
