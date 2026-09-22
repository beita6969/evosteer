# ALFWorld training-path communication repair

## Diagnosis

Read the local training-session history and the private fixed-panel T0
artifacts. The inspected failures do **not** show a missing task/reset context
or a scorer turning successful environment episodes into failures.

The concrete communication defect is in the reasoning/action history carrier:

- Owner reasoning sometimes invents `Action:`, `Observation:` and `### Step`
  sections. The previous flat transcript used the same headings for actual
  execution records. Later reasoning/actions treat imagined observations as
  established state.
- Examples in the private traces include acting as if a still-closed container
  had been opened, claiming inventory changes without a successful pickup, and
  assuming a lighting task was finished after repeatedly issuing `look`.
- Some resulting commands are not currently admissible; some action responses
  mix prose with a native call and are correctly rejected by the existing wire.
  Failed attempts remain failures. Increasing reasoning capacity alone does
  not fix the distinction between a draft and executed history.
- Public command meaning was underspecified. `look` reports a view; it does not
  perform the `use` operation of a lamp. See the [official command
  catalog](https://github.com/alfworld/alfworld/blob/master/alfworld/data/alfred.twl2)
  and [ALFWorld's interface example](https://alfworld.github.io/).

These are observed failure mechanisms, not proof that every low score has the
same cause or that correcting communication guarantees a success threshold.

## Explicit new condition

Select `task_semantic_guidance: public-task-semantics@3` alongside phase context,
native single-tool calls, and public action semantics. Version 1 and legacy
conditions retain their original rendering; old results are not reinterpreted.

For ALFWorld, version 2 adds:

1. Generic public command meanings, without task-specific demonstrations,
   locations, action plans, private goals, or scorer feedback.
2. `typed-public-history@1`: controller-produced records with separate owner
   draft, submitted action, and execution-feedback fields. Embedded headings
   stay inside the raw draft string rather than becoming new history records.
3. A final controller message carrying the latest **actual** public observation,
   current admissible commands, previous attempt feedback, and the controller's
   remaining turn allowance. A parse failure or skill read does not replace the
   last environment state. The native environment limit is not confused with
   the smaller controller limit.

The v2 real diagnostic exposed another misconception: the owner treated
"clean" as an object-name category and inferred cleanliness from the absence
of "dirty" in an ID. It also treated the absence of a currently admissible
cleaning command as evidence that cleaning was unnecessary. **Version 3** adds
the missing general semantics: processed states are separate from object
names; cleaning/heating/cooling depend on holding the object near the matching
appliance; the current command list does not enumerate all actions that can
become available later. These statements follow the [public domain
specification](https://github.com/alfworld/alfworld/blob/master/alfworld/data/alfred.pddl),
not private episode goals. Version 2 remains frozen, including its failures.

The full owner reasoning remains available, including text around `</think>`.
Controller data messages avoid Qwen's historical-assistant reasoning stripping.
No generated action is repaired, masked, replaced, selected, or re-scored under
a different candidate. One Qwen3.5-9B owner remains responsible for every action.

Sampling, restored full-trajectory scoring and provisional scoring use the same
renderer. Forward receives current reasoning; backward receives current
execution observation **without current reasoning/action**. Original sampled
action token spans, terminal reward, skill evolution and the TTB math remain
unchanged. The other six IID task meanings are unchanged in versions 2 and 3.

## Validation and diagnostic scope

- 30 targeted communication/shared-semantics tests passed on 22049 CPU.
  Initial WSL test/type-check attempts timed out during startup at 180 seconds.
  An inherited remote temporary-directory ownership conflict was avoided by
  using a new private temporary directory; only the six affected tests were
  rerun, not the 24 that had passed.
- Two deployed-Qwen-tokenizer checks passed on 22048 CPU, covering thinking on
  and off and preservation of the real-state/draft boundary.
- Full `make check` runs on a frozen source snapshot on 22049's Linux memory
  filesystem, with `CUDA_VISIBLE_DEVICES=""`. Its first pass found one import
  outside the established rollout facade, plus nine subprocess tests loading
  an older editable installation. The import now goes through `policy.interface`;
  explicit source paths correct the subprocess environment without changing
  any shared installation. All ten affected tests passed before the final
  complete check was started. That complete check passed **3991 tests, 2 GPU-only
  skips**, formatting, lint, typing and both wheel builds. The subsequent v3
  catalog-only clarification passed 28 affected tests and targeted typing;
  unchanged scoring and transport tests were not repeated for a prose change.
- A separately labeled, collect-only ALF4 diagnostic uses the four original
  fixed-panel ALFWorld sources, saved initial policy, full-inline skills,
  thinking off, 25 turns, reasoning cap 8192 and action cap 2048. It uses seed 0
  and declared B4 sampling coordinates, **not** the original B28 coordinates.
  It therefore is not a pure same-seed B28 A/B comparison, a formal quality-gate
  result, a new IID32 score, or evidence of a training improvement.
- The ALF4 diagnostic reuses the project's existing inference service on
  22048 physical GPU6; this session starts no new inference service or 250-step
  training. Outputs, private source records and endpoint-local paths stay out
  of Git. The diagnostic has a 20-minute whole-run limit and cannot claim a
  complete score if interrupted.

No review/coding subagent or manual hash checking was used. Final measured
results will be appended after the diagnostics finish.

### Frozen version-2 result

The first successful ALF4 launch completed all four trajectories: **1/4 native
success (25%)**, 82 edges, 1166.34 seconds (19.44 minutes). The successful
trajectory took 7 turns; the other three exhausted 25 turns. Four action parse
failures remain; they are retained, not repaired after generation. One failure
searched unsuccessfully within the horizon, another incorrectly assumed the
required processed state, so this result does not establish that all remaining
failures are transport defects. The temporary adapter was unloaded and the
diagnostic client exited. An earlier launcher-preparation failure made zero
model requests and is not a benchmark observation.

Version 3 is tested as a fresh complete ALF4 condition, not as a replacement
answer for any failed version-2 trajectory. No cross-condition successes are
combined into a reported score.
