# Skill catalog continuation

The latest requested training condition uses `skill_exposure: catalog-then-read@2`:
H0 contains the applicable skill directory, not skill bodies. The single agent
may call `read_skill`; the existing catalog environment returns the pinned body
and records the actual invocation. The prompt asks it to read at least two
**different** visible skills during each trajectory, before final submission or
environment completion. Repeated reads of one ID do not meet this target.

Reads still count as actions and consume the existing tool/turn budget (including
the 8/25-turn caps); neither the training horizon nor TTB normalization changes.
This is a prompt target, not a forced-call gate: no framework-selected calls,
submission masking, added reward, trajectory filtering or resampling. Fewer than
two available skills, ignored instructions and exhausted budgets remain visible
unmet outcomes. Exposure alone is not posterior evidence.

`catalog-then-read@1` retains its original optional-reading prompt. Version 2 is
separate so the new instruction is not silently added to old trajectories.

For a full-inline or version-1 catalog run, drain the active batch and resume its complete checkpoint
with the existing formal entrypoint plus `--allow-catalog-read`. Derive the target
configuration from that run's current condition, changing **only**
`skill_exposure`. Do not substitute a repository default configuration for a
running experiment's frozen configuration.

The continuation preserves optimizer/model state, library, posterior history,
detector, task/seed cursor and budgets. It saves a new condition-boundary snapshot
and records the first affected optimizer step. In-flight full-inline artifacts
must not be reused under the catalog condition. Later resumes of the new boundary
are ordinary same-condition resumes; no prompt change is applied mid-batch.

This is an explicit input-condition change, not a promise of more calls, higher
reward or natural evolution. Historical zero-invocation batches stay unchanged.
Compare exposure, actual calls, posterior updates and task outcomes separately.
The version-2 telemetry adds `skills/two_distinct_met_trajectory_count`,
`skills/two_distinct_met_fraction`, `skills/distinct_per_trajectory_mean`, and
unknown/insufficient-catalog coverage. Old conditions have no retroactive target;
missing invocation evidence stays null. `train/action_count` is unchanged.

Deployment status (2026-09-11): local continuation implementation prepared;
activation on the running training host is pending restoration of SSH access to
the approved 22048 endpoint. No effective step is claimed yet.

Prior version-1 validation: catalog/continuation targeted tests passed, including a restored next
step with real scripted skill-read credit and unchanged historical posterior.
The 22049 CPU full suite covered 4,269 passing tests and 12 opt-in skips across
the initial run and environment-correction reruns. The reused validation
environment initially exposed old subprocess imports and EvalPlus; those were
corrected in an isolated environment, not in the training environment. Format,
lint, mypy and both wheel builds also passed. No GPU qualification was run;
already passing unrelated tests were not repeated after the environment fixes.

Version-2 validation: 84 targeted cases passed, including two actual skill reads
followed by submission consuming three actions/turns, checkpoint continuation,
distinct-ID accounting and W&B field mapping. The 22049 CPU full check covered
4,297 passing tests and 12 opt-in skips across the initial run and a targeted
rerun: 127 isolated-actor tests initially could not import PyYAML inside their
sandbox. Installing it in the isolated validation environment resolved all 127;
no evaluation behavior was changed. Full format, lint, mypy and both wheel builds
passed. Already passing unrelated tests were not repeated. No GPU run or live
condition transition was performed; real-model two-skill compliance is unverified.
