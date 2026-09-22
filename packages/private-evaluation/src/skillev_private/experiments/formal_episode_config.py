"""Shared transport allowance, distinct from model token/turn budgets."""

from skillev.rollout import ExternalSGLangRolloutConfig


def formal_actor_transport(endpoint: str, *, worker_threads: int) -> ExternalSGLangRolloutConfig:
    # Real AIME native reasoning exceeded the former 300/600 second transport
    # limits. The read-only architecture must not silently restore those limits.
    return ExternalSGLangRolloutConfig(
        endpoint_base=endpoint,
        request_timeout_seconds=1800.0,
        transport_worker_threads=worker_threads,
    )
