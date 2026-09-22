"""Frozen EvalPlus execution settings and private native-verdict diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev_private.benchmarks.mbpp_scoring import (
    MBPPScorerProfile as MBPPScorerProfile,
)
from skillev_private.benchmarks.mbpp_scoring import (
    decode_mbpp_verdict,
    mbpp_request,
)
from skillev_private.benchmarks.mbpp_scoring import mbpp_failure_kind as mbpp_failure_kind
from skillev_private.benchmarks.mbpp_scoring import (
    resolve_mbpp_profile as resolve_mbpp_profile,
)


def grade_mbpp(
    candidate: str, target: dict[str, Any], settings: dict[str, Any], sandbox: ActorSandbox
) -> dict[str, Any]:
    from skillev_private.benchmarks import protocol_v13_mbpp_worker

    profile = resolve_mbpp_profile(settings)
    prefix = list(sandbox.command())[:-1]
    source = Path(settings["source_root"])
    prefix.extend(
        (
            "--ro-bind",
            str(source),
            "/evalplus",
            "--setenv",
            "PYTHONPATH",
            os.pathsep.join(("/evalplus", *(str(path) for path in sandbox.python_path))),
        )
    )
    # Only the target for this one candidate enters the isolated scorer. No
    # hidden input or native verdict is returned to the evaluated actor.
    payload = mbpp_request(
        profile=profile,
        private_target=target["private_target"],
        prompt=target["reference_prompt"],
        source_task_id=target["source_task_id"],
        submission=candidate,
        task_id=target["source_task_id"],
    )
    worker = Path(protocol_v13_mbpp_worker.__file__).read_text()
    try:
        completed = subprocess.run(  # noqa: S603 -- isolated trusted native evaluator
            [*prefix, str(sandbox.interpreter), "-c", worker],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=profile.outer_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "EvalPlus outer worker timeout; no candidate score was produced"
        ) from error
    if completed.returncode:
        raise RuntimeError(
            f"EvalPlus worker exited {completed.returncode} before a native verdict: "
            f"{completed.stderr[-3000:]}"
        )
    value = decode_mbpp_verdict(json.loads(completed.stdout), profile)
    return {**value, "worker_returncode": completed.returncode, "outer_timeout": False}
