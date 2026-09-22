"""Isolated EvalPlus CLI process boundary."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


class EvalPlusInfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EvalPlusExecutionContract:
    """Pinned EvalPlus source, dataset, CLI, and result-schema identity."""

    source_root: Path
    python_executable: Path
    code_revision: str
    dataset_name: str
    dataset_version: str
    result_schema_profile: str
    min_time_limit: float
    gt_time_limit_factor: float
    parallelism: int
    timeout_seconds: int
    profile_id: str = "evalplus-cli-26d6d00@1"

    def __post_init__(self) -> None:
        if not self.source_root.is_absolute() or not (self.source_root / "evalplus").is_dir():
            raise ValueError("EvalPlus source root must be an absolute checkout")
        if not self.python_executable.is_absolute():
            raise ValueError("EvalPlus Python executable must be absolute")
        texts = (
            self.code_revision,
            self.dataset_name,
            self.dataset_version,
            self.result_schema_profile,
            self.profile_id,
        )
        if any(not item.strip() for item in texts):
            raise ValueError("EvalPlus execution identity is incomplete")
        if self.dataset_name != "mbpp":
            raise ValueError("Protocol 13 EvalPlus evaluates MBPP+")
        if (
            self.min_time_limit <= 0
            or self.gt_time_limit_factor <= 0
            or self.parallelism <= 0
            or self.timeout_seconds <= 0
        ):
            raise ValueError("EvalPlus execution limits must be positive")

    def to_mapping(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "source_root": str(self.source_root),
            "python_executable": str(self.python_executable),
            "code_revision": self.code_revision,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "result_schema_profile": self.result_schema_profile,
            "min_time_limit": str(self.min_time_limit),
            "gt_time_limit_factor": str(self.gt_time_limit_factor),
            "parallelism": self.parallelism,
            "timeout_seconds": self.timeout_seconds,
        }


def minimal_evalplus_environment(source_root: Path) -> dict[str, str]:
    allowed = (
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "HUMANEVAL_OVERRIDE_PATH",
        "MBPP_OVERRIDE_PATH",
        "PATH",
        "XDG_CACHE_HOME",
    )
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment["PYTHONPATH"] = str(source_root)
    environment["PYTHONHASHSEED"] = "0"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment


def run_evalplus_26d6d00(
    contract: EvalPlusExecutionContract,
    *,
    carrier: Path,
    output_file: Path,
    stdout: Path,
    stderr: Path,
) -> dict[str, object]:
    """Execute the pinned CLI with an explicit output path and schema."""

    observed_revision = subprocess.run(  # noqa: S603 -- fixed Git identity query
        [  # noqa: S607 -- Git from the operator-controlled PATH
            "git",
            "-C",
            str(contract.source_root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_revision != contract.code_revision:
        raise EvalPlusInfrastructureError("EvalPlus source revision differs")
    if output_file.exists():
        raise EvalPlusInfrastructureError("EvalPlus output path already exists")
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    (cache_root / "evalplus").mkdir(parents=True, exist_ok=True)
    command = [
        str(contract.python_executable),
        "-m",
        "evalplus.evaluate",
        "--dataset",
        contract.dataset_name,
        "--version",
        contract.dataset_version,
        "--samples",
        str(carrier),
        "--parallel",
        str(contract.parallelism),
        "--min-time-limit",
        str(contract.min_time_limit),
        "--gt-time-limit-factor",
        str(contract.gt_time_limit_factor),
        "--output-file",
        str(output_file),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 -- pinned evaluator argv, no shell
            command,
            cwd=contract.source_root,
            env=minimal_evalplus_environment(contract.source_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=contract.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvalPlusInfrastructureError("pinned EvalPlus process exceeded deadline") from exc
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not output_file.is_file():
        raise EvalPlusInfrastructureError("pinned EvalPlus process failed")
    value = json.loads(output_file.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("eval"), dict):
        raise EvalPlusInfrastructureError("EvalPlus v26d6d00 result schema is invalid")
    return value


@dataclass(frozen=True, slots=True)
class EvalPlusProcessConfig:
    python_executable: Path
    dataset: str
    version: str
    workers: int
    timeout_seconds: int
    minimum_time_limit_seconds: float = 0.1
    ground_truth_time_limit_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.dataset not in {"mbpp", "humaneval"}:
            raise ValueError("EvalPlus dataset must be mbpp or humaneval")
        if self.workers <= 0 or self.timeout_seconds <= 0:
            raise ValueError("EvalPlus worker count and deadline must be positive")
        if self.minimum_time_limit_seconds <= 0 or self.ground_truth_time_limit_factor <= 0:
            raise ValueError("EvalPlus per-test time controls must be positive")


def run_evalplus(
    *,
    config: EvalPlusProcessConfig,
    samples: Path,
    output: Path,
    stdout: Path,
    stderr: Path,
) -> dict[str, object]:
    try:
        installed = version("evalplus")
    except PackageNotFoundError as exc:
        raise EvalPlusInfrastructureError("EvalPlus is not installed") from exc
    if installed != config.version.removeprefix("v"):
        raise EvalPlusInfrastructureError(
            f"EvalPlus version differs: expected {config.version}, observed {installed}"
        )
    derived_output = samples.with_name(f"{samples.stem}_eval_results.json")
    if output.exists() or derived_output.exists():
        raise EvalPlusInfrastructureError("EvalPlus output path already exists")
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    (cache_root / "evalplus").mkdir(parents=True, exist_ok=True)
    command = [
        str(config.python_executable),
        "-m",
        "evalplus.evaluate",
        "--dataset",
        config.dataset,
        "--samples",
        str(samples),
        "--parallel",
        str(config.workers),
        "--min-time-limit",
        str(config.minimum_time_limit_seconds),
        "--gt-time-limit-factor",
        str(config.ground_truth_time_limit_factor),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 -- pinned evaluator argv, no shell
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvalPlusInfrastructureError("EvalPlus exceeded its total deadline") from exc
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not derived_output.is_file():
        raise EvalPlusInfrastructureError("EvalPlus did not produce a complete result")
    derived_output.replace(output)
    value = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("eval"), dict):
        raise EvalPlusInfrastructureError("EvalPlus result schema is invalid")
    return value


__all__ = [
    "EvalPlusExecutionContract",
    "EvalPlusInfrastructureError",
    "EvalPlusProcessConfig",
    "minimal_evalplus_environment",
    "run_evalplus",
    "run_evalplus_26d6d00",
]
