# Failure playbook

## CUDA OOM in the gradient worker

1. Abort the whole optimizer step; do not retain partial gradients.
2. Keep the last completed checkpoint and exact ordered batch intact.
3. Fail the attempt without launching another GPU or automatically retrying.
4. Report the required memory evidence for a human decision; do not silently shrink the scientific
   batch.

An explicit later resume must start from the last committed checkpoint. The supervisor has no OOM
standby or expansion state.

## SGLang adapter swap

Drain supervisor requests, unload the old supervisor adapter, load the new immutable version, and
verify it through the model endpoint. Executor and skill-creator base requests remain available.
If loading fails, reload the prior adapter before releasing supervisor traffic. If rollback also
fails, stop the run rather than serving an unknown generation.

## Data or verifier failure

Missing/malformed rows, duplicate IDs, cross-split overlap, unavailable source revisions, and
verifier contract violations are hard failures. Never substitute sample data. Quarantine any
unverified training event; it cannot update a posterior.

## Process cleanup

Use only project-owned recorded PIDs. Check ownership and full command line before sending a signal.
Never kill by broad name and never alter another user's process or GPU allocation.
