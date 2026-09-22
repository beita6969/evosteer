"""Private, durable per-submission rubric evidence; never an actor input.

A settled score is reused exactly, including low scores. An unfinished prior
judgment is not silently sampled again. No rubric, case or response enters the
public metric projection; the reward retains only this private file reference.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

from skillev.contracts import normalize_json
from skillev.runtime.request_journal import UnknownRequestOutcomeError
from skillev.training.inflight import durable_json


class HealthBenchCriterionLedger:
    def __init__(self, root: Path, task_id: str, binding: dict[str, Any]) -> None:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = root / (quote(task_id, safe="") + ".json")
        self._lock = threading.RLock()
        expected = normalize_json({"task_id": task_id, **binding})
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["binding"] != expected:
                raise ValueError(
                    "saved HealthBench judgment belongs to another submission/condition"
                )
            if self.state["status"] != "completed":
                raise UnknownRequestOutcomeError(
                    "prior HealthBench judgment is incomplete; do not resample"
                )
        else:
            self.state = {
                "format": "healthbench-criterion-ledger@1",
                "binding": expected,
                "status": "started",
                "requests": [],
                "result": None,
            }
            # Exclusive creation distinguishes a new judgment from another owner.
            with self.path.open("x", encoding="utf-8") as stream:
                os.chmod(self.path, 0o600)
                json.dump(self.state, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def update(self, **fields: Any) -> None:
        with self._lock:
            if self.state["status"] == "completed":
                raise ValueError("a completed HealthBench judgment is immutable")
            self.state.update(fields)
            durable_json(self.path, self.state)

    def record_requests(self, rows: list[dict[str, Any]]) -> None:
        self.update(requests=rows)

    @property
    def completed(self) -> bool:
        return bool(self.state["status"] == "completed")

    def reference(self) -> dict[str, Any]:
        return {
            "format": self.state["format"],
            "path": str(self.path),
            "status": self.state["status"],
            "request_count": len(self.state["requests"]),
        }


def official_criterion_messages(
    grade_sample: Any,
    prompt: list[dict[str, str]],
    candidate: str,
    rubrics: list[Any],
    *,
    system_message: str | None = None,
) -> list[list[dict[str, str]]] | None:
    """Mirror the installed official grade_sample's exact public call construction.

    This associates measurements only, never creates a request or computes a
    score. If another official implementation exposes no template, report the
    association unknown instead of guessing criterion substrings. Duplicate
    identical criterion messages retain all matching indices.
    """
    template = getattr(grade_sample, "__globals__", {}).get("GRADER_TEMPLATE")
    if not isinstance(template, str):
        return None
    conversation = "\n\n".join(
        f"{message['role']}: {message['content']}"
        for message in [*prompt, {"role": "assistant", "content": candidate}]
    )
    return [
        ([{"role": "system", "content": system_message}] if system_message is not None else [])
        + [
            {
                "role": "user",
                "content": template.replace("<<conversation>>", conversation).replace(
                    "<<rubric_item>>", str(rubric)
                ),
            }
        ]
        for rubric in rubrics
    ]
