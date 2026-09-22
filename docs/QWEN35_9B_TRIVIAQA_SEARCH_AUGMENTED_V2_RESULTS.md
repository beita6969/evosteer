# Qwen3.5-9B TriviaQA detailed local-search result

This is the independent `qwen35-triviaqa-search-augmented@2` diagnostic defined
in `QWEN35_9B_TRIVIAQA_SEARCH_AUGMENTED_V2_PROTOCOL.md`. It uses the frozen
adapter-free Qwen3.5-9B backbone and does not overwrite the question-only or v1
results.

## Aggregate score

| Population | Records | EM | Token F1 | Invalid candidates | Generation infra | Retrieval infra |
|---|---:|---:|---:|---:|---:|---:|
| SkillFlow released TriviaQA IID v3 | 128/128 | 58.59375% | 67.88318452380952% | 0 | 0 | 0 |

All 128 records received exactly one task-local search and one definitive
official-scorer verdict. The search-use rate was 100%, the mean search count was
1.0, and the empty-hit rate was 0%. Evaluation took 42.79 seconds at 179.50
tasks/minute.

## Corpus telemetry

| Item | Observed |
|---|---:|
| Detailed Codex dossiers | 128 |
| Dossier length | 2,441–3,000 characters |
| Queries per task | 8 |
| Research entities per task | 9–12 |
| Wikipedia page associations | 1,534 |
| Unique Wikipedia pages | 1,509 |
| Indexed dossier passages | 384 |
| Indexed entity passages | 128 |
| Indexed Wikipedia passages | 15,178 |
| Total indexed passages | 15,690 |
| Passages or titles changed by literal-label removal | 3,404 |

The released RC passage previews were removed before both corpus preparation
and generation. Codex and Wikipedia query selection saw only each bare public
question. A private final pass removed literal official aliases from the
task-local index before Qwen inference. Questions, labels, corpus text,
provenance, traces, and per-task predictions remain outside Git.

## Comparison

| Track | EM | Token F1 | EM change vs question-only | F1 change vs question-only |
|---|---:|---:|---:|---:|
| Question-only Direct-Qwen | 52.34375% | 59.813988095238095% | — | — |
| Local search v1 | 57.8125% | 63.917410714285715% | +5.46875 pp | +4.10342261904762 pp |
| Detailed local search v2 | 58.59375% | 67.88318452380952% | +6.25 pp | +8.069196428571425 pp |

Relative to v1, v2 improves EM by 0.78125 percentage points and token F1 by
3.965773809523805 percentage points. This comparison changes the inference
protocol and retrieval corpus, so it is a diagnostic result rather than a new
paper-direct parity claim.

Two complete private diagnostic attempts exposed an overly strict command
parser before the final protocol was frozen. Their samples were discarded in
full: no prediction was reused, merged, retried, or included in the score above.
