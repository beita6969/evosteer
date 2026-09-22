"""Requested call substreams of one evaluation experiment seed."""

SAMPLING_SEED_SCHEDULE = "episode-call-counter@1"


def evaluation_call_seed(seed: int, admitted_calls: int, *, sampling_mode: str) -> int:
    """Advance on every sampled call, independently of its content or outcome.

    With SGLang's deterministic sampler enabled, an explicit seed restarts a
    stateless per-position sampler. Reusing it for every call does not advance
    a conversation's random stream. This counter is scheduling-independent,
    but sending it alone does NOT enable that server feature: SGLang 0.5.9
    ignores request seeds when deterministic inference is disabled. Actual
    serving arguments are recorded separately; this schedule does not promise
    batch-invariant model outputs. No labels or candidate selection are used.
    Greedy generation retains its original (unused) sampling seed.
    """
    return seed if sampling_mode == "greedy" else (seed + admitted_calls) % (1 << 64)
