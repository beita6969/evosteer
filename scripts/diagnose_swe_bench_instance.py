#!/usr/bin/env python3
"""Run one private, deterministic SWE-bench gold instance through the formal path."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import swe_bench_eval  # noqa: E402


def main() -> int:
    os.environ["SKILLEV_FORMAL_RUNTIME"] = "1"
    started = time.monotonic()
    swe_bench_eval._load_verified_dataset()
    candidates = [
        row
        for _, row in sorted(swe_bench_eval._verified_cache.items())
        if str(row.get("patch", "")).strip() and row.get("FAIL_TO_PASS")
    ]
    if not candidates:
        raise RuntimeError("verified dataset contains no diagnosable instance")
    verified = candidates[0]
    instance_id = str(verified["instance_id"])

    try:
        resolved, _, status = swe_bench_eval._evaluate_patch_with_official_harness(
            instance_id=instance_id,
            model_patch=str(verified["patch"]),
            verified=verified,
            timeout=int(os.environ.get("SWE_DIAGNOSTIC_TEST_TIMEOUT_SECONDS", "900")),
        )
        import docker

        client = docker.from_env(timeout=60)
        try:
            info = client.info()
        finally:
            client.close()
        output = {
            "status": "passed" if resolved else status,
            "instance_hash": swe_bench_eval._short_blake2b(instance_id),
            "resolved": resolved,
            "report_present": True,
            "duration_seconds": round(time.monotonic() - started, 1),
            "docker_server": info.get("ServerVersion"),
            "storage_driver": info.get("Driver"),
        }
        print(json.dumps(output, sort_keys=True))
        return 0 if resolved else 1
    except swe_bench_eval.SWEHarnessInfrastructureError as error:
        diagnostic = error.diagnostic
        print(
            json.dumps(
                {
                    "status": "failed",
                    "classification": diagnostic.classification,
                    "phase": diagnostic.phase,
                    "retryable": diagnostic.retryable,
                    "instance_hash": diagnostic.instance_hash,
                    "report_present": diagnostic.report_present,
                    "test_output_present": diagnostic.test_output_present,
                    "container_exit_code": diagnostic.container_exit_code,
                    "container_oom_killed": diagnostic.container_oom_killed,
                    "image_key_hash": diagnostic.image_key_hash,
                    "duration_seconds": round(time.monotonic() - started, 1),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
