# Qwen3.5-9B HotpotQA Full-Context Result

## Correction

The earlier 128-record backbone-only evaluation did not omit context entirely,
but its inherited upstream rendering truncated every one of the ten distractor
passages to at most 300 characters. This could remove evidence needed for the
multi-hop answer and therefore was not a faithful full-context request.

The corrected adapter-free evaluation uses the same frozen 128-record panel,
base model, tokenizer, chat template, decoding profile, parser, scorer, and seed.
For every request it extracts the final question and supplies all ten complete
public context passages from that record. It does not load a LoRA, adapter,
skill, retriever, posterior, or Z head, and it does not use the private target or
supporting-fact labels during generation.

## Aggregate result

| Metric | Truncated-context run | Full-context run | Change | Reference | Parity |
|---|---:|---:|---:|---:|---|
| Exact Match | 50.78125% | 63.28125% | +12.50000 pp | 60.94% | PASS |
| Token F1 | 62.52333603896104% | 78.97818687133945% | +16.45485083237841 pp | 75.70% | PASS |

Both corrected metrics are inside the executable protocol's strict
`absolute gap < 7.0 percentage points` numeric band: EM differs by 2.34125
points and F1 by 3.27818687133945 points.

## Coverage

- Frozen records: 128
- Candidate responses: 128
- Definitive scorer verdicts: 128
- Generation/scorer/environment infrastructure failures: 0/0/0
- Scorer submission/reach rate: 99.21875%

The corrected HotpotQA component is numerically `PASS`. This scoped rerun does
not by itself decide the formal-training gate, and unknown paper prompt/seed
semantics remain a scientific comparability limitation.
