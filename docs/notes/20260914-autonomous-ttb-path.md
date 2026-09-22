# Autonomous TTB candidate, separate from supervised cold-start experiments

Subsequent deployment and completed remote verification are recorded in
[the September 15 fresh-start note](20260915-autonomous-ttb-fresh-start.md).
The implementation-time status below describes September 14, not live training.

This is an implementation condition, **not a successful cold-start result**.
Historical teaching, five-update forward-only warmup and the owner-authorized
unqualified training run remain separate and unchanged. No new GPU run was
started for this change. `idea.tex` is unchanged and remains authoritative.

## Inputs and learning boundary

Select `configs/training/bayesianimprove_autonomous_ttb.yaml` explicitly. Its
`learning_protocol: autonomous-ttb@1` uses the saved pre-skill-SFT initialization,
catalog-then-read@1, native action wire, and visible tools during reasoning.
It rejects teaching metadata and the supervised initialization export. There is
no read quota, teacher, reasoning-token/NLL/KL/auxiliary loss, or read reward.
All sampled outcomes enter the usual full B28 TTB transaction. The bounded plan
remains 250 real updates (249 search + one closure); the first 100 search steps
can supply the original two W50 windows. They are not an uncounted Step-0.
Continuation restores the entire existing application checkpoint, not F alone.

The new `public-method-cards@5` initial library contains seven answer-free method
cards: code dependencies, boundary/invariant reasoning, relational evidence,
mathematical domains/cases, response organization, and two public ALFWorld
procedures. Descriptions name inputs and non-applicability; bodies remain behind
real reads. These are **unvalidated candidates**, not established skill gains.
Old library profiles and IDs are not overwritten. Once a run begins, changes to
its library remain the responsibility of the ordinary Phi boundary.

The private binding's `data_condition.autonomous_ttb_sources` must contain:

```json
{
  "format": "public-task-needs@1",
  "ordered_sources": [
    {
      "benchmark": "mbpp-plus",
      "source_id": "synthetic-example-only",
      "role": "procedure-applicable",
      "method_family": "boundary-invariant",
      "public_basis": "A pre-outcome public contract requires order preservation."
    }
  ],
  "source_aliases": {},
  "excluded_sources": {"iid": [], "development": [], "quality": []}
}
```

The example is a schema illustration, **not a runnable source population**.
Supply all seven real training lanes, actual exclusions and reasonable direct
controls. Allowed roles are `direct-control`, `procedure-applicable`, and
`exploration`. Roles never enter prompts, rewards or loss weights. The order is
frozen before sampling; no unsuccessful or uninvoked trajectory is filtered.
Identical historical acquisition repeats can map to one canonical source, but
conflicting variants are rejected. The complete expanded training schedule and
source-role counts are retained with the effective condition. Applicability is
not a claim that reading improves the task.

## Confirmed phase-boundary correction

The historical, explicitly enabled `zero-coverage-generate@1` extension permits
a Generate boundary without decreasing entropy. It remains readable for that
historical experiment but **cannot be enabled in this new protocol**.

The ordinary detector also had a narrower defect: a sequence of observed skill
distributions followed by an empty window could treat the serialized empty
entropy value 0 as a real decrease. It now requires positive invocation counts
in every window used for the entropy comparison. Genuine concentration to one
observed skill remains valid. Historical empty-entropy serialization is retained;
no evidence, failure labels or posterior updates are fabricated.

Existing F/B conditions, per-edge K normalization, (Delta/T)^2 loss, raw Delta^2
phase statistics, full-batch posterior normalization and Z-only reset are not
replaced. The current implementation conventions are explicitly not uniquely
specified by the paper: entropy uses rolling invocation frequencies with the
configured consecutive-drop rule; invoking-edge flow weights normalize to mean
one within the full batch; retrieval, finite-sample flow estimation and confidence
thresholds retain their existing configured definitions. No new convention was
invented to force a cold-start success.

## Read-only real TTB checkpoint validation

Historical `autonomous-skill-validation@1` still denotes warmup comparisons.
Use `autonomous-skill-validation@2`, `comparison_kind: ttb-checkpoint` and a
`ttb_binding` for a real TTB comparison. The same binding is passed as the after
collector's `forward_initialization`:

```json
{
  "format": "development-ttb-checkpoint@1",
  "kind": "ttb-checkpoint",
  "run_root": "<original private run>",
  "checkpoint": "<complete checkpoint in that run's checkpoints directory>",
  "preparation": "<original pre-skill-SFT preparation.json>",
  "effective_condition": "<same-run effective-condition-process-N.json>",
  "skill_sources": "<private skill construction inventory.json>",
  "published_forward_adapter": "<already published actual forward revision route>"
}
```

The reader uses existing runtime/policy metadata and the committed transaction;
a kind string or detached forward export is insufficient. It neither loads
optimizer tensors nor publishes adapters. A single version-pinned prepublished
SGLang endpoint is supported. Before uses the same run's saved original
`fresh-forward` binding. Real training step and fixed evaluation draw coordinate
are recorded separately; after is not renamed training Step-0.

Both on arms must use the **same** frozen initial or checkpoint library. Actual
initial/current libraries and committed cycle count are reported separately.
The full planned training source set—not merely successes or read examples—is
excluded. `skill-construction-sources@1` inventories every initial and retained
evolved document, including inactive lineage:

- Every entry contains the actual `document`, `kind`, and `source_files` list.
- `public-procedure` is only valid for an actual initial document and requires
  `public_basis`; no task sources can be declared when none were used.
- `source-derived` requires nonempty archived source records.
- `training-evolution` requires its saved `mutation_record`, containing the
  actual body and a step no later than the checkpoint. The entire training
  population is also excluded, including failed/no-read trajectories.
- Source files contain source rows (`benchmark`, `source_id`) or an
  `ordered_sources` manifest. Previously inspected construction/development
  material must be included, not renamed as new unseen validation.

The four arms retain all outcomes and original read/application reviews. Their
policy-axis effects use TTB names, not warmup labels. Evaluation writes no
training evidence, posterior, optimizer or Phi state. Results do not authorize
restarting/resetting training or imply seven-domain IID/A0 acceptance.

## Remaining execution work

Actual private source selection, frozen new-development availability, actor
publication and empirical four-arm collection remain to be done. Prior twelve
autonomy sources are development-used. Do not assume 24 unused MBPP+ sources are
available without checking the complete exclusions. No new sample or seed was
collected here, and the prior failed cold-start verdict is not changed.

Validation and deployment status will be reported separately from this input
contract. Remote CPU full-check and any GPU experiment require the approved
endpoint to be reachable; loss of monitoring connectivity is not evidence that
the old training process stopped.

Local verification: 127 targeted input/checkpoint/catalog/F-B/provisional tests
passed, plus 61 unchanged-branch compatibility tests for the existing collection,
warmup, IID and formal entry paths. A missing synthetic fixture directory was
fixed before the final targeted run. Changed-file Ruff lint/format checks passed.
Local mypy timed out without a verdict. The required remote CPU `make check`
remains pending because SSH on the approved check endpoint refused connection;
this patch is not deployed and no empirical cold-start qualification is claimed.
