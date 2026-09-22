"""Read-only serving counters; never use a metrics request as a model request."""

from __future__ import annotations

import time
import urllib.request
from urllib.parse import urlsplit


def serving_metrics(endpoint: str) -> dict[str, object]:
    """Interval context, not invented per-request TTFT/prefill attribution.

    Request IDs and reported backend timestamps live in each phase record. These
    Prometheus series describe the service population at this sampling instant.
    They may include frozen-judge calls; never assign the aggregate to one actor.
    """
    started = time.monotonic()
    result: dict[str, object] = {
        "sampled_at_monotonic": started,
        "scope": "service-aggregate-not-per-request",
    }
    try:
        if urlsplit(endpoint).scheme not in {"http", "https"}:
            raise ValueError("serving metrics require an HTTP endpoint")
        request = urllib.request.Request(  # noqa: S310 - HTTP(S) scheme checked above.
            endpoint.rstrip("/") + "/metrics", method="GET"
        )
        with urllib.request.urlopen(request, timeout=3) as response:  # noqa: S310 - validated above.
            body = response.read(2 * 1024 * 1024 + 1)
        if len(body) > 2 * 1024 * 1024:
            raise ValueError("metrics response exceeds sidecar byte budget")
        names = (
            "time_to_first_token",
            "inter_token",
            "e2e_request",
            "queue_time",
            "num_running_reqs",
            "num_queue_reqs",
            "cache_hit",
            "cached_tokens",
            "prefill",
            "decode_throughput",
            "generation_tokens",
        )
        result.update(
            available=True,
            series=[
                line
                for line in body.decode().splitlines()
                if line.startswith("sglang:") and any(n in line.split("{", 1)[0] for n in names)
            ],
        )
    except (OSError, ValueError) as error:
        # Advisory monitoring failure remains explicit; it produces no label,
        # model retry, partial transaction, or change to serving behavior.
        result.update(available=False, error_class=type(error).__name__, series=None)
    result["sampling_wall_seconds"] = time.monotonic() - started
    return result
