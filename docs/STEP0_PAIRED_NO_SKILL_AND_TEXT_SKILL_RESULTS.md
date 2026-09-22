# Corrected Step-0 paired results

**Historical execution notes, not a currently active or completed paired run.**
The owner subsequently restricted evaluation to A2 only. Current separated runs and incomplete
attempts are indexed in [STEP0_RUN_INDEX.md](STEP0_RUN_INDEX.md). The dated narrative below
records the historical execution state; it is not a present connection or process-status claim.

The complete committed `1d09159` snapshot started a fresh 1,852-candidate run at
2026-09-06 15:15 UTC on the two existing inference services at `185.212.56.211:22048`.
It retains the original arms, population, within-domain ordering and model budgets, schedules
whole interactive domains first, imports no previous candidates and scores only after complete
generation. The final CPU-only check passed 2,633 tests and both wheel builds.

The expanded nonfinal native canary completed 24 candidates, 289 executions and 631 model
transports with zero recorded surface, execution, acknowledgement, observation or peer-route
mismatches. It took about 52 minutes (27.7 candidates/hour), below the 40/hour operational
alert threshold. Extrapolating that small development rate to 512 interactive candidates gives
about 18.5 hours, excluding static tasks and scoring; this is a planning estimate, not a measured
full-run ETA or completion guarantee. No quality improvement is inferred from these diagnostics.

The new immutable `b140e33` attempt started at 2026-09-06 13:04 UTC on the existing two
inference services at `185.212.56.211:22048`. It retained 66 WebShop candidates and no scores
before its coordinator was interrupted at 13:53 UTC and confirmed stopped at 13:59 UTC.
Additional unambiguous operation-header/JSON and equals-sign argument notation was rejected.
Its four interrupted requests have unknown usage; no old candidates are imported elsewhere.
See the [separate incomplete record](machine-results/step0_paired_2026-09-06_named_arguments_incomplete.json).
The preceding four-candidate native communication canary
completed 60 executions with zero recorded surface, execution, acknowledgement, observation or
peer-route mismatches, but did not expose all notation variants seen later. It is not evidence
of full notation coverage or quality improvement. Development validation is expanded to six
final-disjoint tasks per native domain, including the six existing ALFWorld train task types.

The fresh `7f80f26` attempt started at 2026-09-06 09:05 UTC on the two existing inference
replicas at the approved `185.212.56.211:22048` endpoint. It uses in-flight load balancing,
one transport attempt and a 7,200-second HTTP deadline. Panels, seed, native thinking,
actor call/token limits and scorers are unchanged. All four static QA panels completed, including
all 60 AIME candidates and at least one normally completed 81,920-token call. However, unambiguous
literal function calls and named arguments were still rejected by the native-action codec.
The coordinator was interrupted at 12:44 UTC and confirmed stopped at 12:48 UTC, retaining
866 candidates and zero scores. Its four unfinished transport records have unknown usage.
The two inference services were not stopped. See the separate
[incomplete-attempt record](machine-results/step0_paired_2026-09-06_literal_calls_incomplete.json).
This is long-request execution evidence, not a completed eight-IID comparison or quality result.

The literal-call repair preserves exact arguments and current native permissions; it does not
execute Python, infer targets or select answers. Reusing completed unaffected domains under
explicitly separate version provenance was proposed to the owner but not authorized or
implemented, so the new run uses the original complete-generation protocol.

The `3f903e3` full attempt retained 559/1,852 candidates and produced no scores. Long AIME calls
were interrupted by the HTTP timeout and repeated; failure cleanup then waited for active HTTP
threads. The partial run remains private and is not a complete comparison. Its successful-response
token totals omit aborted attempts and must not be presented as total cost. Transport and replica
scheduling repairs require a fresh frozen attempt, not replacement or splicing of old candidates.

The intended primary comparison is A1 single/no-skill against A2 peer-capable/no-skill, both
Qwen3.5-9B, optimizer step zero and native thinking off, with one seed and matched total budgets.
A3 text skills are optional, not silently added to the primary comparison.

Each full arm requires 926 unique tasks: HotpotQA, TriviaQA, HealthBench, WebShop, ALFWorld,
MBPP+ hard and HumanEval each 128, plus AIME2026 all 30. Two arms require 1,852 stored candidates.
The frozen historical panels are development-exposed. Nonfinal canary tasks are excluded from
these denominators and are deployment checks, not evidence of quality improvement.

Historical AIME 27/30 and label-swapped/fallback composites are withdrawn as evidence of standalone
policy gains; see [the historical report](SKILLEV_BAYESIAN_IMPROVE_TRAINED16_IID_128_RESULTS.md).
No replacement full-run score, gain, completion time or statistical significance is claimed here.

After execution, publish the answer-free runtime summary with actual revision, arms, native metric
denominators, paired effects/intervals, communication/intervention counts and costs. Keep per-item
answers, tasks, rubric grades, test sources and server-internal paths private.
