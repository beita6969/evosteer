"""Collect hardware state and select four idle GPU roles for one run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from skillev.runtime.gpu_topology import (
    FourGPURolePolicy,
    GPUObservation,
    select_four_idle_gpus,
)


def _run(*command: str, timeout: float = 20.0) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed argument vector, never a shell command
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def _parse_gpu_rows(
    text: str,
    process_map: dict[str, tuple[int, ...]],
) -> tuple[GPUObservation, ...]:
    observations: list[GPUObservation] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            raise ValueError("unexpected nvidia-smi GPU row")
        index, name, uuid, total, used, utilization = fields
        observations.append(
            GPUObservation(
                physical_index=int(index),
                name=name,
                uuid=uuid,
                memory_total_mib=int(total),
                memory_used_mib=int(used),
                utilization_percent=int(utilization),
                compute_pids=process_map.get(uuid, ()),
            )
        )
    return tuple(observations)


def _parse_compute_rows(text: str) -> dict[str, tuple[int, ...]]:
    mutable: dict[str, list[int]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 2:
            raise ValueError("unexpected nvidia-smi process row")
        uuid, pid = fields[:2]
        mutable.setdefault(uuid, []).append(int(pid))
    return {uuid: tuple(sorted(set(pids))) for uuid, pids in mutable.items()}


def collect_gpu_observations() -> tuple[GPUObservation, ...]:
    process_text = _run(
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    )
    gpu_text = _run(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    return _parse_gpu_rows(gpu_text, _parse_compute_rows(process_text))


def build_report(policy: FourGPURolePolicy) -> dict[str, object]:
    observations = collect_gpu_observations()
    assignment = select_four_idle_gpus(policy, observations)
    disk = shutil.disk_usage(Path.cwd())
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "cuda_visible_devices_at_preflight": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git": {
            "branch": _run("git", "branch", "--show-current"),
            "commit": _run("git", "rev-parse", "HEAD"),
            "dirty": bool(_run("git", "status", "--porcelain")),
        },
        "gpu_driver": _run(
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ).splitlines()[0],
        "gpus": [item.to_value() for item in observations],
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "role_assignment": assignment.to_value(),
        "topology": _run("nvidia-smi", "topo", "-m"),
        "resources": {
            "cpu_count": os.cpu_count(),
            "disk_free_bytes": disk.free,
            "disk_total_bytes": disk.total,
            "file_descriptor_soft_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hardware/four_gpu_roles.yaml",
        help="four-role hardware policy YAML",
    )
    parser.add_argument("--output", required=True, help="new JSON report path")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(output)
    report = build_report(FourGPURolePolicy.read_yaml(arguments.config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "role_assignment": report["role_assignment"]}))


if __name__ == "__main__":
    main()
