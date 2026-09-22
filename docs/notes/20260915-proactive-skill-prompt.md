# Owner-requested proactive skill prompt — 2026-09-15

The owner explicitly asked to tell the model to use skills more often. The new
`catalog-then-read@3` condition encourages early selection of a relevant visible
skill, a real `read_skill` action, and use of the returned method in subsequent
reasoning/actions. It discourages irrelevant/repeated reads and clarifies that
reading advice is neither an environment check nor proof of success.

There is **no invocation quota**, injected action, read reward or new loss.
Direct completion remains allowed. Reading still consumes turns/tool calls;
skill bodies are not visible before reading. F/B conditions, per-edge action
normalization, Z, the full B28 commit and ordinary Bayesian/Phi rules stay intact.
This changes the prompt/input condition, not just monitoring, and cannot by
itself demonstrate learned skill use or successful cold start.

`bayesianimprove_autonomous_ttb_proactive_skills.yaml` differs from the original
pure-TTB configuration only in `skill_exposure`. Continuation requires a full
checkpoint and explicit `--allow-catalog-read`; it preserves optimizer, posterior,
library, source order and cursor. Historical evidence keeps its original condition
labels, including when missing observer projections are recovered later. The new
catalog uses the same actual-input/body-visibility instrumentation as earlier
catalog protocols.

The live run requested a cooperative pause during step 2: it must finish and save
the entire batch before this prompt is deployed. No partial batch is discarded
to install the hint. Runtime deployment/progress will be appended after checking
the actual saved boundary.

## Observation before the prompt change

The first step committed at **04:45:36 UTC**: TTB loss **1.705718**, mean native
reward **0.561355**, binary success **13/28**, structural validity **99/101**, and
**zero skill reads / posterior updates**. Wall time was **905.44 s**, rollout
**571.70 s**, gradient tail **329.77 s**. Its single-step rate is **3.98 steps/h**,
not steady-state throughput. Constant-rate extrapolation for the remaining 249
steps is approximately **62.6 h**, excluding future variability and overhead.
Thermal slowdown recurs under load and needs administrator attention; hardware
protection and foreign processes were not changed.

W&B GPU-hour export was repaired privately to bind the actual performance
`process_instance_id`, rather than bare operating-system PIDs. Step 1's missing
resource fields were backfilled as separate telemetry, without duplicating loss
or optimizer commits. GPU hours represent reserved roles, not measured utilization.

## Verification

The first CPU full check exposed a hard-coded two-version visibility branch;
it was repaired rather than ignoring the failing catalog-body evidence tests.
Targeted catalog/continuation/observer/authorization tests then passed **67/67**.
Final remote Linux CPU `CUDA_VISIBLE_DEVICES="" make check` passed **5,078 tests,
16 skipped**, Ruff, mypy, both wheel builds and model-wheel checks. The local WSL
targeted invocation timed out without a verdict; it is not counted as a pass.
No subagent or hash audit was used. No additional full run is needed for this
documentation-only verification record.

## Step 2 boundary and monitor repair

Step 2 committed at **05:24:47 UTC** and was completely saved and mirrored:
loss **0.921749**, reward **0.669937**, success **18/28**, and **3 actual skill
reads / posterior events** under the original prompt. Those reads were on failed
ALFWorld trajectories, not evidence of successful skill use. Step wall was
**2,348.31 s** (rollout **1,265.73 s**, gradient tail **1,078.35 s**). The two-step
activity rate is only **2.21 steps/h**; simple remaining-work extrapolation is
about **112 h**, not a steady-state guarantee. Thermal/long-edge costs still need
human attention rather than a hidden reduction in the workload.

At **05:28 UTC**, full-checkpoint restoration with the new prompt succeeded, but
the old aggregate metric exporter attempted to relabel the previous two commits
and correctly triggered a cooperative pause **before any step-3 sampling**.
The database was neither removed nor rewritten to hide the conflict. Export now
uses the same declared per-step condition history as the evidence observer.
The actual failure was reproduced on a private copy of the metric database;
the repaired export added zero commits and preserved both original rows. New
monitor/metrics/formal-entry targeted tests passed **53/53**.
The repair's final Linux CPU `make check` passed **5,080 tests, 16 skipped**,
Ruff, mypy, both builds and wheel checks. Standby SGLangs retained the two owned
gradient devices during this repair; no step-3 trajectory was regenerated.

## Live deployment

Runtime source **`f38140a`** resumed the full step-2 checkpoint at **05:42:50 UTC**.
The prompt is effective from step 3; the prior two metric/evidence rows retain
their original condition. At **05:47:42 UTC**, step 3 had **21/28 artifacts and
21/28 gradient contributions**, with no monitor failure or batch abort. All 21
completed artifacts carried the actual new model-visible hint. One TriviaQA
trajectory read the relational-evidence skill on turn 1; its body was present
in four later phase inputs and the trajectory succeeded. This is a read-then-
success observation, not proof of causal benefit or learning.

GPU4 continues serving through parent/scheduler **196926/197475**. New controller
**76190**, torchrun **76812**, and gradient owners **80132 (GPU6) / 80133 (GPU0)**
are active. CPU metric source **83826** and local W&B uploader **1294659** are
active; authenticated W&B readback confirmed `running`, last committed step 2,
and prompt transition at step 3. Earlier source/upload processes exited; no
parallel uploader remains for this run. During the saved-checkpoint handoffs,
owned standby SGLang parents **633960/633961**, then **5145/5146**, were stopped
only after replacement real gradient owners established CUDA contexts. The
first resumed gradient attempt **648627/648628** ended safely before sampling.
No foreign process was changed. Three physical GPUs remain in use; GPU4 and GPU6
again reported thermal slowdown at the latest observation. The new prompt had
no complete-step speed measurement at that observation.

## First complete step with the proactive prompt

At **06:05:54 UTC**, step 3 was committed and step 4 had started. The full step-3
checkpoint has its completion marker, optimizer state, runtime state and policy.
Authenticated W&B readback at **06:08:13 UTC** confirmed the new committed row:

| Metric | Step 3 |
| --- | ---: |
| TTB loss | 0.958192 |
| Mean native reward | 0.702610 |
| Binary success | 18/28 |
| Structurally valid actions | 97/100 |
| Parse errors / schema errors | 3 / 0 |
| Actual skill reads / posterior events | 3 / 3 |
| Installed library mutations | 0 |
| Step wall / rollout / gradient tail | 1,331.85 / 1,187.41 / 139.99 s |
| Logical input / output tokens | 2,523,207 / 171,457 |

The three reads came from three trajectories and two source questions. All were
model-generated on turn 1, with the returned body visible in later phase inputs.
One ALFWorld trajectory and one TriviaQA trajectory succeeded; the other ALFWorld
trajectory failed. This is stronger read-then-success evidence than the prior
step, but only **3/28 trajectories** read a skill; the total read count of three
is unchanged from step 2. It does not demonstrate causal skill benefit, learned autonomous use, or
a passed cold start. No invocation quota, bonus or new training objective was added.

The last ALFWorld trajectory reached turn 25. Its preceding reasoning request
generated the full 8,192-token budget in **208.50 s**, versus **3.30 s** for the
following action; cumulative environment calls were under one second. That
reasoning request reported **30,720 cached / 35,973 input tokens**. Thus this
particular observed wait was not evidence of an environment IPC hang or disabled
prefix caching. Unknown pure-decode timing remains unknown.

Across all three committed steps, activity throughput is **2.36 steps/h**.
Constant-rate remaining-work extrapolation is about **105 h**, not a steady-state
forecast; prompt conditions and source questions differ across the steps. GPU4
and GPU6 still reported thermal slowdown and need administrator attention. The
owner-authorized run continues on the same three GPUs, without changing hardware
protection or foreign processes. No new GPU or resident process was started during
this monitoring interval. This is a documentation-only update; the unchanged
runtime already passed the 5,080-test full check above, so it was not repeated.
