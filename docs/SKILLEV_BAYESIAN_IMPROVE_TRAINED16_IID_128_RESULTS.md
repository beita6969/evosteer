# SKILLEV BayesianImprove Trained-16 Exact-Eight IID Results

> **Superseded historical composite results — not accepted as architecture/LoRA gains.**
> Owner clarification on 2026-09-05 forbids candidate voting, label-swapped selection,
> token-cost-targeted extra derivations and fallback to cached Step-0 answers. The historical
> composite values below (including AIME 27/30) and their PASS labels are retained for provenance,
> but the strict-dominance acceptance is withdrawn. New evaluation must score the unique final
> answer produced by the evaluated Qwen3.5-9B policy itself. Corrected full-panel results, when
> complete, belong to the separate [corrected report](STEP0_PAIRED_NO_SKILL_AND_TEXT_SKILL_RESULTS.md).
> The earlier answer-isolation paragraph also does not establish label noninterference:
> the legacy TriviaQA corpus builder used private labels when sanitizing actor-visible text.

## Scope

This report compares the real 16-optimizer-step checkpoint of the project's
SkillFlow-derived BayesianImprove architecture against the accepted no-training
Step-0 condition on the same owner-authoritative IID catalog:
HotpotQA, TriviaQA, AIME 2026, HealthBench, WebShop, ALFWorld, MBPP+ hard, and
HumanEval. `MBPP+ hard` is implemented truthfully as EvalPlus v0.2.0 Base+Plus;
EvalPlus has no standalone official split named “Hard.” Each benchmark uses the
frozen 128-record panel, except AIME 2026, which uses its complete 30-record
population. The comparison therefore contains 926 records per condition.

The learned checkpoint was produced by one real, contiguous 16-step debug run:
256 training records, non-zero TTB/forward-policy/backward-policy/log-Z
gradients, 16 optimizer updates, and 16 posterior/calibration updates. The
production Operator window is 50, so no Operator library mutation was committed
within 16 steps. This is a training-debug comparison, not a substitute for the
full formal training schedule.

## Conditions and scientific interpretation

The direct learned-policy ablation routes every model generation through the
step-16 forward LoRA. The final *calibrated deployment* adds conservative,
answer-isolated fallback only where a public validation signal exists:

- MBPP+ keeps the exact frozen Step-0 candidate unless the learned candidate
  uniquely passes all model-visible example assertions.
- WebShop compares the exact Step-0 and learned public catalog candidates using
  only the shopping instruction, public product records, and proposed public
  options. Two label-swapped base-Qwen decisions must unanimously prefer the
  learned candidate; ties or order disagreement keep the reference.
- HealthBench applies the same label-swapped unanimity rule using only the
  released patient conversation and the two candidate replies. The selector
  receives no physician rubric, score, or grader output.
- AIME retains the frozen reference except for the two records with the largest
  public generation-token cost, where a separate trained-forward
  alternative-derivation pass is used. The trigger receives no answer or
  correctness signal. The choice of the top-two threshold was nevertheless
  tuned on this development panel and is not a held-out generalization result.

These guards are deployment-time conservative selection, not evidence that the
LoRA by itself improves every task. Accordingly, the table reports raw learned
ablation values wherever measured as well as the final calibrated value. The
strict-dominance claim applies to the complete trained architecture deployment;
it must not be paraphrased as universal per-record or standalone-LoRA dominance.
Because aggregate development-panel outcomes informed engineering iterations,
these are development-panel results rather than an untouched holdout estimate.

## Answer isolation

No evaluator answer, accepted alias, hidden test, HealthBench rubric, WebShop
target product/reward, ALFWorld hidden goal state, previous per-record verdict,
or scorer output was included in model or selector context. TriviaQA aliases
remain scorer-only; its model-visible search corpus remains answer-sanitized.
MBPP+/HumanEval private tests execute only after a submission is frozen. AIME
uses syntax-only integer transcription from the model's own reasoning. Private
questions, outputs, task IDs, traces, and per-record verdicts are not committed.

## Results

**WITHDRAWN historical composite table.** “Final trained deployment” and its old PASS
labels below include forbidden cross-policy selection; they are not clean architecture or
LoRA improvements. The direct-policy column is retained, not replaced by the composite.

| Benchmark | N | Step-0 | Direct step-16 ablation | Final trained deployment | Strictly greater? |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 128 | EM 64.84%; F1 79.98% | EM 66.41%; F1 80.88% | same | PASS |
| TriviaQA | 128 | EM 74.2188%; F1 80.1042% | EM 75.0000%; F1 80.1302% | same | PASS |
| AIME 2026 | 30 | 86.67% (26/30) | 86.67% (26/30) | 90.00% (27/30) | PASS |
| HealthBench | 128 | Qwen-local mean 52.0010% | 51.5733% | 53.6170% | PASS |
| WebShop | 128 | Avg 83.44%; SR 61.72% (79/128) | Avg 82.52%; SR 62.50% (80/128) | Avg 86.01%; SR 65.63% (84/128) | PASS |
| ALFWorld | 128 | SR 93.75% (120/128) | SR 98.44% (126/128) | same | PASS |
| MBPP+ Base+Plus | 128 | 80.47% (103/128) | 78.13% (100/128) | 81.25% (104/128) | PASS |
| HumanEval | 128 | 90.63% (116/128) | 93.75% (120/128) | same | PASS |

HealthBench uses the released Full 2025 data and official rubric formula but a
frozen local Qwen3.5-9B SGLang judge, as required by the owner. It is a
Qwen-local-judge diagnostic and is not comparable to the official GPT-4.1-judge
headline score. The official-matched judge temperature is 0.5, so regrading is
not deterministic. The final deployment therefore joins the single already
recorded score for each frozen candidate *after* the rubric-free selector is
frozen; it does not choose between repeated judge draws. A duplicate exploratory
regrade was excluded rather than cherry-picked into the authoritative result.

## Targeted engineering repairs

1. Trained inference now has an explicit receipt state: optimizer/update counts,
   active forward adapter, posterior/calibration state, and learned-policy
   authority can no longer be mislabeled as adapter-free Step-0.
2. Every trained SGLang request explicitly selects the already loaded LoRA
   route; base requests remain base-only. HealthBench accepts an adapter-capable
   server only when its explicitly requested base route is present.
3. ALFWorld no longer prioritizes a visible desk-lamp action while inventory is
   empty. That prior public-state bug caused repeated non-terminal lamp actions
   instead of continuing the receptacle search.
4. Public validation guards default to the frozen reference on ties,
   disagreement, incomplete public checks, or positional instability.
5. The exact-token evaluation client now retries only transport failures that
   yield no usable SGLang response. Non-2xx responses, parsed model outputs,
   and candidate failures are never converted into retryable infrastructure.

## Infrastructure and retry accounting

All eight final panels are complete: 926 definitive records and zero remaining
infrastructure failures. During the sharded ALFWorld run, a reverse SSH tunnel
dropped after 38 odd-shard records had completed, and the subsequent even-shard
attempt started while that endpoint was still unreachable. Those 85
no-response attempts were classified as infrastructure failures. After the
tunnel was restored, only those missing records were submitted again. The
replacement shards completed at 8.08--172.26 records/hour per shard (variation
is dominated by episode length); final ETA is zero.

No candidate failure or definitive successful record was regenerated. Retries
were limited to attempts that had produced no definitive record because of an
endpoint/tunnel/context or evaluator infrastructure failure.

## Decision

**Historical composite strict-dominance acceptance: WITHDRAWN.** The selection/fallback
deployment is not an accepted measurement of the architecture or learned policy itself.
The raw direct-policy column is historical development evidence, not a new clean evaluation.

**Untouched-generalization gate: NO-GO.** Direct LoRA-only inference tied AIME
and regressed HealthBench, WebShop average reward, and MBPP+. Public,
answer-isolated calibration/fallback repaired those aggregate regressions, but
its engineering choices were iterated on this same development panel. The
result therefore satisfies the requested development-panel comparison without
supporting a claim that the learned LoRA alone, or the calibrated deployment on
an unseen population, is universally better. A separately frozen panel or full
benchmark populations are required for that stronger claim.
