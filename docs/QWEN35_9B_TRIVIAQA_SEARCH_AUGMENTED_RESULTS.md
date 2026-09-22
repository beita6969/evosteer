# Qwen3.5-9B TriviaQA local-search result

This is the independent `qwen35-triviaqa-search-augmented@1` diagnostic track
defined in `QWEN35_9B_TRIVIAQA_SEARCH_AUGMENTED_PROTOCOL.md`. It does not
overwrite the question-only Direct-Qwen baseline and is not a training result.

## Aggregate score

| Population | Records | EM | Token F1 | Generation infra | Retrieval infra |
|---|---:|---:|---:|---:|---:|
| SkillFlow released TriviaQA IID v3 | 128/128 | 57.8125% | 63.917410714285715% | 0 | 0 |

All 128 records received a definitive official-scorer verdict. Thirty-eight
trajectories ended in a malformed or empty final command and were retained as
candidate failures rather than repaired or retried. Thus 90/128 trajectories
(70.3125%) submitted a valid short final answer.

## Search and corpus telemetry

| Item | Observed |
|---|---:|
| Search-use rate | 100.00% |
| Mean search calls per task | 1.328125 |
| Empty-hit rate | 0.00% |
| Private Codex background notes | 128 |
| Wikipedia page associations | 511 |
| Indexed passages | 2,688 |
| Evaluation wall time | 135.90 seconds |
| Evaluation throughput | 56.51 tasks/minute |

Corpus construction used only the public question for each task. Targets and
accepted aliases were available only to the unchanged official scorer after
generation. Questions, targets, aliases, notes, article text, per-task search
traces, and per-task predictions remain outside Git.

## Separate comparison with question-only Direct-Qwen

| Track | EM | Token F1 |
|---|---:|---:|
| Question-only Direct-Qwen | 52.34375% | 59.813988095238095% |
| Local-search diagnostic | 57.8125% | 63.917410714285715% |
| Difference | +5.46875 pp | +4.10342261904762 pp |

The comparison changes the inference protocol by adding a task-partitioned
local retrieval tool, so the improvement is diagnostic rather than evidence
that the question-only paper-direct protocol was reproduced more exactly.
