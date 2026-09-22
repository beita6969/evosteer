# Independent development before a new A0

The owner authorized candidate revision on independent development material on
2026-09-13. The previous two A0 arms, their failures and capability floors remain
unchanged. Development outcomes cannot authorize training or replace A0.

## Fixed comparison

The private seed-zero development manifest contains eight historical AIME2025
sources (not AIME2026), eight HealthBench sources, and twelve official ALFWorld
games: each of the six task types from both validation splits. Sources were
selected without outcomes and exclude all declared training, quality and prior
final-evaluation source identities. Game availability is recorded separately;
every rollout uses its own official environment. No private task or result is
stored in Git.

`development_collection` uses the existing read-only native session/collector
path with base weights and an explicit library arm. Candidate comparisons retain
the complete source manifest, original targets, policy, library and sampling
coordinates. Responses, original timings and failures are private evidence. All
optimizer, posterior, evolution and training-evidence writes remain zero.

The first baseline preparation failed because dataset split names had been used
as ALFWorld execution modes, before any model request. Its output was retained.
The repaired binding uses the official mode contract before live dispatch and
preserves the original source/split identities. This was an execution-binding
repair, not a replacement answer or a set of negative labels.

## Explicit candidate, not an approved new default

`bayesianimprove_development_handoff.yaml` selects public semantics @7 and the
persisted phase-deliverable @4 envelope for three public domains:

- HealthBench distinguishes preparing a substantive conversational reply from
  delivering that reply in `submit_answer`, without inventing detail merely to
  lengthen it. Its reasoning cap changes from 4,096 to 8,192.
- AIME keeps native thinking and the 16,384 reasoning cap, with explicit handoff
  of a supported integer result rather than a promise of a later calculation.
- ALFWorld keeps thinking off, the 8,192 reasoning cap and the hard 25-turn limit;
  its handoff distinguishes one actual next action from imagined future feedback.

All seven domains keep the original task inputs, callable schemas, parser,
sampling, action/input budgets and native scoring. The other four domains retain
their original phase messages. The initial library remains autonomous and no
number of reads is required. Original reasoning/action tokens remain the scoring
targets; forward/provisional/sealed preparation shares the persisted phase
condition, and hindsight does not gain the current reasoning draft.

These are declared **prompt/budget condition changes**, not claims of an
equivalent infrastructure speedup or demonstrated quality gains. The preserved
fresh-restart YAML is not silently replaced. Any selected candidate must be
frozen and pass both architecture-matched A0 arms before fresh formal training.

`development_comparison` reports every paired source, native values, action/read
evidence and original costs. Missing observations stay unknown and incomplete
comparisons cannot claim a complete result. A backend decode-rate number without
an actual decode duration is also unknown; an after-prefill host span is not
relabeled as pure decode or CUDA kernel time.

## Owner-authorized three-step training diagnostic

The owner subsequently prioritized a short real training diagnostic instead of
further development collection. `local_training_diagnostic` explicitly supports
one external actor and one local gradient owner; this is not a relaxation of the
formal two-gradient-worker topology or an A0 acceptance result. It starts from
fresh Step-0 state using the original training-source split, never development
or IID episodes, and retains three complete B28 batches with a checkpoint each
step. The candidate's budgets, phase semantics and autonomous skill reads stay
unchanged. Both incomplete development attempts and startup failures are retained.

Actual skill reads, subsequent body visibility, posterior updates and mutations
are reported separately. Three steps cannot validate natural phase changes with
the unchanged W=50 residual windows; zero mutation is a valid observed result.
No reads, author calls or library changes are forced to manufacture acceptance.

### Completed diagnostic and authorized continuation

The three complete updates finished on 2026-09-13 at 21:58 UTC. These are
training-source observations, not an IID panel or a formal training acceptance:

| Step | TTB loss | Mean reward | Binary successes | Step wall seconds | Skill reads |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.196803 | 0.662088 | 16/28 | 766.24 | 0 |
| 2 | 1.407551 | 0.532967 | 14/28 | 2346.75 | 1 |
| 3 | 0.929293 | 0.642857 | 16/28 | 1480.31 | 1 |

Both real reads returned the body and made it visible in the following reasoning
and action inputs. They produced two posterior updates in separate context cells;
neither associated trajectory succeeded. There were no library mutations. This
verifies the read-to-evidence path, not skill efficacy or natural evolution.
The three-step observed mean was 2.35 steps/hour, not steady-state throughput.

The owner then authorized continuation to Step10, stopping after its complete
checkpoint. The append-only plan preserves the old search/search/closure slots;
it adds six search steps and one closure step. It restores Step3 F/B/Z, named
Adam state, posterior, library, detector, source cursor and original budget
charges without replaying or rewriting steps1–3. The same source population,
seed, candidate configuration, single-local-gradient exception and W&B run are
retained; the extended schedule has 280 trajectories, including the original84.
No old closure is retrospectively reclassified as a phase search. Ten steps still
do not fill the unchanged production residual windows.
