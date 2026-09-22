# ScienceWorld public API repair — September 13

This is an **interface development condition**, not a new scored model result.
The repeatedly inspected 64-case panel remains development regression evidence.
One Qwen3.5-9B owner makes every decision; no consultants, object substitution,
automatic experimental actions, or private-score-driven stopping are introduced.

## Interface condition

`public-task-semantics@6` selects `scienceworld-commands@2` and
`scienceworld-pending-choice@1`. Earlier semantic versions keep the original
command reference. The same catalog supplies the task meaning, system reference,
and native `act` definition in both the actor and its trusted broker. This avoids
changing a description in one path but not the other. Execution controls persist
the new identities. The observation whitelist remains **public-state@1** and
the explicit stopping condition remains **owner-finish-cooperative-deadline@1**.

The reference explains:
- focus designates a task object, not an inspection; an incorrect choice/order
  can end a task, while required initial focus operations remain allowed;
- containers, their contents, instruments and terminals are distinct referents;
  native containment text and indentation are preserved without object-tree queries;
- an unacknowledged/rejected prerequisite is not completed, and seeing/focusing
  an instrument is neither pickup nor successful use;
- numerical disambiguation is tied to one pending command and public revision;
  selection or native cancellation clears it. No number/object is chosen or blocked;
- thermometer use measures the target, whereas inspecting the thermometer reports
  its own temperature; the stopwatch displays accumulated ticks;
- every act consumes an evaluation interaction. Simulator time is separate:
  look-around/inventory/task are time-free, not all inspection commands are.

These are API facts, not task-specific experimental recipes, target identities,
box colors, or gold paths. Sources: [goal checks](https://github.com/allenai/ScienceWorld/blob/f6d8f5ec41eadfdcad23cc3ab097f0903dc1378b/simulator/src/main/scala/scienceworld/tasks/goals/specificgoals/GoalFind.scala),
[input parser](https://github.com/allenai/ScienceWorld/blob/f6d8f5ec41eadfdcad23cc3ab097f0903dc1378b/simulator/src/main/scala/scienceworld/input/InputParser.scala),
[action time](https://github.com/allenai/ScienceWorld/blob/f6d8f5ec41eadfdcad23cc3ab097f0903dc1378b/simulator/src/main/scala/scienceworld/input/ActionHandler.scala),
[thermometer](https://github.com/allenai/ScienceWorld/blob/f6d8f5ec41eadfdcad23cc3ab097f0903dc1378b/simulator/src/main/scala/scienceworld/objects/devices/Thermometer.scala),
[stopwatch](https://github.com/allenai/ScienceWorld/blob/f6d8f5ec41eadfdcad23cc3ab097f0903dc1378b/simulator/src/main/scala/scienceworld/objects/devices/StopWatch.scala).

## Private diagnostics and fixed-command execution

The worker can separately log before/after native score and simulator moves,
current goal type/index, pending parser state and the first failure transition.
This private structure travels only to the private transition journal, never to
`public_state`, actor requests, tools or training evidence. A missing diagnostic
is explicitly unavailable and neither erases native scoring nor triggers a retry.
An unavailable previous goal state remains unknown: it cannot be used to claim
that the current action caused the first failure.

`scripts/build_scienceworld_diagnostics.py` creates a **different jar**, compiling
only added logging fields in Goal/GoalReturn and AgentInterface. The deployed jar
and original source files are untouched. The first native taskFailure captures
its call site, goal type/index, move and before/after-physics phase; resolved
native action classes/object bindings are logged. Predicates, score arithmetic,
action execution and ticks are unchanged. Without this build, branch evidence
is explicitly unavailable, not guessed from the final command.

The prepared, **not launched**, A18 architecture uses this separate
`scienceworld-diagnostic-overlay@1` jar and the new public interface. Its 64 IDs
and 8,000-token / 200-action / batch-32 limits are unchanged. The jar and private
configuration are stored separately from the original deployment. This is an
explicit interface-plus-logging condition, not an in-place change to A17.

`scripts/replay_scienceworld_commands.py` executes each stored command literally
with the original task, variation, simplification and initialization. It compares
initial observation, every raw/public observation, score, moves and terminal flag,
retaining discrepancies and exceptions. No alternate command, automatic reset,
model call, answer selection or historical result replacement occurs. Replay
outputs, source tasks and failure branches remain private and outside Git.

The three fixed-command executions (two original-jar passes and one logging
build) each reached 30 native negative terminals, with 411 actions per pass:
one task terminated a command earlier than the archived trajectory. All 30
logging-build runs captured a first native failure branch. Original versus
logging-build final scores and termination positions agreed, but **not every
observation agreed**. Only 17/30 traces in each first pass matched the archive
exactly. Repeating the original jar also changed public observations/choice
order in five cases, without changing that repeat's scores or termination
positions. Thus observation variability is not solely a diagnostic-build effect.
The historical one-step electrical timing difference is retained as unresolved;
do not call this bit-identical replay or silently stabilize the physics/parser.

## Validation and remaining model evidence

Interface and privacy tests cover shared descriptions, legacy behavior, literal
container selection, active/cancelled/stale numerical choices, first-failure
retention, negative scoring and real isolated actor/broker transport. Fixed-command
replay is separate from those tests and from any scored generation experiment.

Two focused groups passed 49 tests each (some overlap), with the added latch
case covered in the later group; scoped mypy passed seven source files. Local
WSL test/type-check startup timed out, so completion validation used an isolated
source snapshot on endpoint 22049 CPU, with CUDA hidden. Other concurrently
edited training files were not included. The first whole-suite attempt passed
4,578 tests but two nested-pytest fixtures rejected an inherited absolute
`PYTEST_ADDOPTS --basetemp`; this was a launch-environment error, not a reason to
weaken those tests. Removing that inherited setting retains a dedicated TMPDIR.
The corrected snapshot passed 4,580 tests (12 skipped). Final `make check` on
the newer committed main plus this repair passed **4,714 tests, 13 skipped**
(pytest: 326.68 seconds), Ruff, mypy on 667 source files, both wheel builds and
the model-wheel boundary check. The skips require explicitly selected CUDA or
private deployed tokenizers; they are not claimed as executed GPU validation.
The 11-test command-contract group also passed, including missing-prior-state
and isolated actor/broker regressions. Unrelated uncommitted serving/training
edits remained excluded. No further whole check is needed for this report-only
update.

No new generated score is claimed. The next approved generation comparison must
include **all 64** cases, not just the 30 negatives, with the frozen 8,000-output-
token / 200-action / batch-32 limits. For an old/new-interface comparison, hold
the logging jar and other runtime components constant on both sides rather than
calling the old A17 archive a single-axis control;
thinking and presence-penalty contrasts are separate, single-axis conditions.
Do not enlarge budgets, replace failures or tune against a success threshold.
Report raw native mean, full successes, negative terminals, voluntary/budget stops,
per-task categories, actual tokens and native moves. Reduced negatives without
more completed tasks may be a stopping-behavior change, not better reasoning.
No final unseen-generalization claim is appropriate for this development panel.
No independent review agent or hash validation was used. Whole checks were
repeated for the corrected launch environment and the newer main/diagnostic
fix, not for each edited file. The repeated original-jar replay specifically
investigated observed ordering differences; it was not model resampling or
selection of better scores.
