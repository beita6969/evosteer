"""Typed SWE-bench evaluator interfaces with complete per-instance verdicts."""

from __future__ import annotations

import asyncio
import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast


class SWEDatasetVariant(StrEnum):
    ORIGINAL = "princeton-nlp/SWE-bench"
    LITE = "princeton-nlp/SWE-bench_Lite"
    VERIFIED = "princeton-nlp/SWE-bench_Verified"


class SWEVerdictKind(StrEnum):
    RESOLVED = "resolved"
    APPLY_FAILURE = "apply-failure"
    TEST_FAILURE = "test-failure"
    TIMEOUT = "timeout"
    CANDIDATE_INVALID = "candidate-invalid"


@dataclass(frozen=True, slots=True)
class SWEPrediction:
    instance_id: str
    model_patch: str


@dataclass(frozen=True, slots=True)
class SWEInstanceVerdict:
    instance_id: str
    kind: SWEVerdictKind

    @property
    def resolved(self) -> bool:
        return self.kind is SWEVerdictKind.RESOLVED


@dataclass(frozen=True, slots=True)
class SWEEvaluationResult:
    dataset_variant: SWEDatasetVariant
    verdicts: tuple[SWEInstanceVerdict, ...]
    evaluator_version: str

    def __post_init__(self) -> None:
        ids = tuple(item.instance_id for item in self.verdicts)
        if len(ids) != 128 or len(set(ids)) != 128:
            raise ValueError("SWE evaluator requires 128 unique verdicts")
        if not self.evaluator_version.strip():
            raise ValueError("SWE evaluator version must be non-empty")


class SWEOfficialEvaluationInfrastructureError(RuntimeError):
    pass


class SWEEvaluator(Protocol):
    async def preflight(self) -> None: ...

    async def evaluate(
        self,
        *,
        variant: SWEDatasetVariant,
        predictions: tuple[SWEPrediction, ...],
    ) -> SWEEvaluationResult: ...


@dataclass(frozen=True, slots=True)
class RemoteSWEEvaluatorClient:
    endpoint_base: str
    timeout_seconds: float = 3600.0

    async def preflight(self) -> None:
        await asyncio.to_thread(self._request, "/health", None, 10.0)

    async def evaluate(
        self,
        *,
        variant: SWEDatasetVariant,
        predictions: tuple[SWEPrediction, ...],
    ) -> SWEEvaluationResult:
        if len(predictions) != 128:
            raise ValueError("SWE evaluation requires 128 predictions")
        value = await asyncio.to_thread(
            self._request,
            "/evaluate-sync",
            {
                "dataset_variant": variant.value,
                "predictions": [
                    {"instance_id": item.instance_id, "model_patch": item.model_patch}
                    for item in predictions
                ],
            },
            self.timeout_seconds,
        )
        return _result(value, expected_variant=variant)

    def _request(self, path: str, payload: dict[str, object] | None, timeout: float) -> object:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(  # noqa: S310 - operator-frozen endpoint
            self.endpoint_base.rstrip("/") + path,
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                if response.status != 200:
                    raise SWEOfficialEvaluationInfrastructureError(
                        f"SWE evaluator returned HTTP {response.status}"
                    )
                return json.loads(response.read())
        except (
            OSError,
            TimeoutError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
        ) as exc:
            raise SWEOfficialEvaluationInfrastructureError(
                f"SWE evaluator request failed: {type(exc).__name__}"
            ) from exc


@dataclass(frozen=True, slots=True)
class LocalCommandSWEEvaluator:
    command: tuple[str, ...]
    work_directory: Path
    timeout_seconds: float = 3600.0

    async def preflight(self) -> None:
        if not self.command or not self.work_directory.is_dir():
            raise SWEOfficialEvaluationInfrastructureError("local SWE harness is unavailable")
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                (*self.command, "--preflight"),
                text=True,
                capture_output=True,
                cwd=self.work_directory,
                timeout=min(self.timeout_seconds, 300.0),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SWEOfficialEvaluationInfrastructureError(
                f"local SWE harness preflight failed: {type(exc).__name__}"
            ) from exc
        if completed.returncode != 0:
            raise SWEOfficialEvaluationInfrastructureError(
                "local SWE harness preflight returned failure"
            )

    async def evaluate(
        self,
        *,
        variant: SWEDatasetVariant,
        predictions: tuple[SWEPrediction, ...],
    ) -> SWEEvaluationResult:
        if len(predictions) != 128:
            raise ValueError("SWE evaluation requires 128 predictions")
        payload = json.dumps(
            {
                "dataset_variant": variant.value,
                "predictions": [
                    {"instance_id": item.instance_id, "model_patch": item.model_patch}
                    for item in predictions
                ],
            }
        )
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                self.command,
                input=payload,
                text=True,
                capture_output=True,
                cwd=self.work_directory,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SWEOfficialEvaluationInfrastructureError(
                f"local SWE harness failed: {type(exc).__name__}"
            ) from exc
        if completed.returncode != 0:
            raise SWEOfficialEvaluationInfrastructureError("local SWE harness returned failure")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise SWEOfficialEvaluationInfrastructureError(
                "local SWE harness returned invalid JSON"
            ) from exc
        return _result(value, expected_variant=variant)


def _result(value: object, *, expected_variant: SWEDatasetVariant) -> SWEEvaluationResult:
    if type(value) is not dict:
        raise SWEOfficialEvaluationInfrastructureError("SWE result must be an object")
    raw = cast(dict[str, object], value)
    try:
        variant = SWEDatasetVariant(str(raw["dataset_variant"]))
        version = str(raw["evaluator_version"])
        rows = raw["verdicts"]
    except (KeyError, ValueError) as exc:
        raise SWEOfficialEvaluationInfrastructureError("SWE result metadata is invalid") from exc
    if variant is not expected_variant or type(rows) is not list:
        raise SWEOfficialEvaluationInfrastructureError("SWE result variant or verdicts differ")
    verdicts: list[SWEInstanceVerdict] = []
    try:
        for row in cast(list[object], rows):
            if type(row) is not dict:
                raise ValueError
            item = cast(dict[str, object], row)
            verdicts.append(
                SWEInstanceVerdict(str(item["instance_id"]), SWEVerdictKind(str(item["status"])))
            )
        return SWEEvaluationResult(variant, tuple(verdicts), version)
    except (KeyError, ValueError, TypeError) as exc:
        raise SWEOfficialEvaluationInfrastructureError("SWE verdict payload is invalid") from exc
