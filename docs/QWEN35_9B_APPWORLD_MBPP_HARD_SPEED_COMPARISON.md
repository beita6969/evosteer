# Qwen3.5-9B AppWorld / MBPP+ Hard / SWE-bench Speed Comparison

## Scope

AppWorld and MBPP+ Hard were evaluated as adapter-free Qwen3.5-9B backbone-only conditions on
frozen, result-blind 128-task panels. They ran concurrently through two independent, already-running
single-GPU SGLang services. SWE-bench was not rerun: its older result and its stable official-Docker
scorer window are retained only for this speed comparison. SWE-bench is no longer in the current IID
catalog.

No prompt, answer, candidate, patch, test, task ID, or per-item verdict is included here.

## Results

| Benchmark | N | Backbone-only result | Measured wall time | Measured throughput |
|---|---:|---:|---:|---:|
| AppWorld | 128 | TGC 61.5996%; SR 14.0625% | 2,680.36 s (44m 40s) | 171.92 records/hour |
| MBPP+ Hard | 128 | Pass@1 71.8750% | 1,890.20 s (31m 30s) | 243.78 records/hour |
| Historical SWE-bench | 128 | Resolved 27.34375% | stable scorer segment: 5h 06m 02s for 91 remaining verdicts | 17.89 verdicts/hour |

MBPP+ Hard used the official EvalPlus prompt for every selected task and the EvalPlus 0.2.0 base +
Plus evaluator. Its measured total comprises 1,820.19 seconds of generation and 70.01 seconds of
official scoring. AppWorld used the official Simplified ReAct Code prompt, a 40-step horizon, the
official environment, and native TGC/success evaluation; its runner wall time therefore includes
both model interaction and environment evaluation.

The SWE-bench number is deliberately labelled as a **scorer-stage** rate. It comes from the stable
official-Docker continuation window rather than a new end-to-end run, so it is useful for operational
ordering but is not perfectly stage-matched to the two new measurements. At the same stable rate,
128 Docker verdicts would take about 7.15 hours; that value is an extrapolation, not a newly observed
run.

## Ordering

Under these measurements:

1. **MBPP+ Hard is fastest**: 243.78 records/hour.
2. **AppWorld is second**: 171.92 records/hour.
3. **SWE-bench official Docker scoring is slowest**: 17.89 verdicts/hour in the reused stable window.

MBPP+ Hard was about 1.42 times as fast as AppWorld. Relative to the historical stable SWE scorer
rate, MBPP+ Hard was about 13.63 times faster and AppWorld about 9.61 times faster. This supports the
owner's decision to remove SWE-bench from the current IID suite on runtime grounds.

## Validity and retry accounting

- AppWorld's pre-run failures were environment/import/client-compatibility failures before any
  record was completed. Only the final 128/128 run entered the aggregate.
- An initial MBPP diagnostic used shortened source prompts. Aggregate-only comparison showed that
  all 128 were strict substrings of, rather than exact matches to, the official EvalPlus prompts, so
  that panel was rejected as a protocol/configuration failure. The published run regenerated the
  frozen selection with the official prompts and scored only those candidates.
- EvalPlus's CLI requires a full-population carrier. Unselected carrier entries were invalid
  placeholders and were excluded from the selected-panel aggregation; exactly 128 selected verdicts
  were counted.
- Both published runs have 128 definitive records and zero generation, scorer, or environment
  infrastructure failures.

