# Protocol 10 matched external targets

Protocol 10 is the structured-agent control. It is not the SkillFlow paper's
direct-Qwen row and is never rescaled to that row.

| Benchmark | External target status | Permitted interpretation |
|---|---|---|
| HealthBench | unmatched until population, actor prompt, judge model/prompt and aggregation all match | report the local-Qwen-judge metric by that name; no five-point gate |
| SpreadsheetBench | unmatched until split, workbook runtime, tool interface and no-skill prompt all match | internal structured-agent result only |
| AppWorld Normal | matched public comparison | five-point regression control |
| AppWorld Challenge | matched public comparison | five-point regression control |
| MBPP+ fixed-100 | target undefined | internal fixed-panel regression; HumanEval is not its target |

The already published AppWorld values remain the negative control: Normal 21.22
versus 25.2 and Challenge 15.45 versus 15.4. Any shared runner change must retain
those matched gaps. The paper-direct implementation is isolated, so it does not
modify this control lane.
