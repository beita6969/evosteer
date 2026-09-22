# Active documentation map

| File | Role |
|---|---|
| [`../idea.tex`](../idea.tex) | Current scientific idea and methodology for SKILLEV-new |
| [`../evaluation.tex`](../evaluation.tex) | Frozen Protocol 10 benchmark science |
| [`../README.md`](../README.md) | Project and repository boundary |
| [`method-v3-final-spec.md`](method-v3-final-spec.md) | Current Protocol-v3 cross-layer engineering specification |
| [`protocol-v3-errata.md`](protocol-v3-errata.md) | Incompatible execution-wire migration and historical-goal corrections |
| [`phase1-goal.md`](phase1-goal.md) | Phase 1 contracts goal |
| [`phase2-goal.md`](phase2-goal.md) | Phase 2 policy-backbone goal |
| [`phase3-goal.md`](phase3-goal.md) | Phase 3 scoring/objective goal |
| [`phase4-goal.md`](phase4-goal.md) | Phase 4 rollout goal |
| [`phase5-goal.md`](phase5-goal.md) | Phase 5 training-loop goal |
| [`phase6-goal.md`](phase6-goal.md) | Phase 6 flow-diagnostics goal |
| [`phase7-goal.md`](phase7-goal.md) | Phase 7 Bayesian-calibration goal |
| [`phase8-goal.md`](phase8-goal.md) | Phase 8 evolution-loop goal |
| [`benchmark-evaluation-guide.md`](benchmark-evaluation-guide.md) | Frozen benchmark roles, reward boundary, and evaluation protocol |
| [`EVALUATION_SPEC.md`](EVALUATION_SPEC.md) | Protocol 10 engineering mapping and migration gate |
| [`method-handoff-contracts.md`](method-handoff-contracts.md) | Sole production handoff at each method-layer boundary |
| [`reuse-boundary.md`](reuse-boundary.md) | Explicit migration and non-reuse boundary |

The scientific method always comes from `../idea.tex`; the phase goals are
historical implementation records, while their top-of-file amendments plus
`method-v3-final-spec.md` and `protocol-v3-errata.md` describe the final
engineering semantics. The current executable runtime/snapshot wire is `@6`.
Protocol 10 benchmark science and its population-level reader are frozen but not yet executable;
the `@9` reader is historical. Historical `@3` protocol
artifacts under `historical/protocol-v3/` are not executable inputs. Historical
FIRE-GFN material remains outside the active documentation path under
`../old_backup_do_not_open_or_push/` and must not be used as a method input. A
historical source may be consulted only for an explicit infrastructure
extraction governed by `reuse-boundary.md`.
