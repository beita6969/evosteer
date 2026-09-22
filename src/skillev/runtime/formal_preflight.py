"""CUDA-free preflight for the corrected SkillFlow baseline entrypoint."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import yaml

from skillev.contracts import JsonValue, normalize_json

from .gpu_topology import FourGPURoleAssignment, GPUObservation

RUN_MANIFEST_FORMAT: Final = "skillev-skillflow-run-manifest@1"
EXPECTED_TRAIN_PACKAGES: Final = {
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "peft": "0.19.1",
    "causal-conv1d": "1.6.2.post1",
    "flash-linear-attention": "0.5.2",
    "datasets": "4.4.1",
    "sentence-transformers": "5.2.3",
    "faiss-cpu": "1.13.2",
    "docker": "7.1.0",
    "alfworld": "0.5.0",
    "textworld": "1.7.0",
    "fast-downward-textworld": "20.6.4",
    "spacy": "3.8.15",
    "en-core-web-sm": "3.8.0",
    "beautifulsoup4": "4.11.1",
    "cleantext": "1.1.4",
    "flask": "2.1.2",
    "werkzeug": "2.1.2",
    "gym": "0.26.2",
    "pyserini": "0.17.0",
    "pyjnius": "1.7.0",
    "onnxruntime": "1.22.1",
    "selenium": "4.2.0",
    "rank-bm25": "0.2.2",
    "thefuzz": "0.19.0",
    "chardet": "5.2.0",
    "ghapi": "2.0.5",
    "fastcore": "2.1.16",
    "fastspec": "0.1.3",
    "GitPython": "3.1.57",
    "pre-commit": "4.6.1",
    "python-dotenv": "1.2.2",
    "requests": "2.32.5",
    "rich": "15.0.0",
    "unidiff": "1.0.0",
    "tqdm": "4.70.0",
}

_RESOURCE_ENV_FIELDS: Final = (
    "alfworld_repository_path_env",
    "alfworld_data_path_env",
    "alfworld_config_path_env",
    "webshop_repository_path_env",
    "webshop_items_path_env",
    "webshop_attributes_path_env",
    "webshop_human_attributes_path_env",
    "webshop_search_index_path_env",
    "swe_verified_path_env",
    "swe_harness_path_env",
    "swe_evaluation_output_path_env",
    "medrag_textbooks_path_env",
    "embedding_model_path_env",
)

_SWE_DOCKER_STORAGE_DRIVERS: Final = frozenset({"overlay2", "fuse-overlayfs", "btrfs"})
_SWE_DOCKER_SMOKE_IMAGE: Final = "busybox:1.37.0-glibc"
_TRAINING_GPU_MAP_ENV: Final = "SKILLEV_TRAINING_GPU_MAP"
_TRAINING_CUDA_VISIBLE_DEVICES_ENV: Final = "SKILLEV_TRAINING_CUDA_VISIBLE_DEVICES"
_DISABLE_GRADIENT_STANDBY_ENV: Final = "SKILLEV_DISABLE_GRADIENT_STANDBY"
_EXTERNAL_INFERENCE_GPU_UUID_ENV: Final = "SKILLEV_EXTERNAL_INFERENCE_GPU_UUID"


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class FormalBaselineConfig:
    raw: Mapping[str, object]
    source_name: str

    @classmethod
    def read(cls, path: str | Path) -> FormalBaselineConfig:
        source = Path(path)
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw = _mapping(value, "baseline config")
        if set(raw) != {
            "data",
            "format",
            "model",
            "name",
            "resources",
            "runtime",
            "services",
            "training",
            "upstream_revision",
        }:
            raise ValueError("baseline config has an incompatible top-level field set")
        if raw["format"] != "skillev-skillflow-baseline@1":
            raise ValueError("unsupported baseline config format")
        config = cls(raw=raw, source_name=source.name)
        config.validate()
        return config

    @property
    def data(self) -> dict[str, object]:
        return _mapping(self.raw["data"], "data config")

    @property
    def model(self) -> dict[str, object]:
        return _mapping(self.raw["model"], "model config")

    @property
    def training(self) -> dict[str, object]:
        return _mapping(self.raw["training"], "training config")

    @property
    def resources(self) -> dict[str, object]:
        return _mapping(self.raw["resources"], "resource config")

    @property
    def runtime(self) -> dict[str, object]:
        return _mapping(self.raw["runtime"], "runtime config")

    @property
    def supervisor(self) -> dict[str, object]:
        services = _mapping(self.raw["services"], "services config")
        if set(services) != {"supervisor"}:
            raise ValueError("baseline may configure only the external supervisor service")
        return _mapping(services["supervisor"], "supervisor service config")

    def validate(self) -> None:
        data = self.data
        if data.get("strict_data") is not True:
            raise ValueError("formal baseline requires strict_data=true")
        for split in ("train", "validation"):
            _text(data.get(f"{split}_path_env"), f"{split}_path_env")
            digest = _text(data.get(f"{split}_blake2b"), f"{split}_blake2b")
            if len(digest) != 64:
                raise ValueError(f"{split}_blake2b must be a 64-character digest")
            _integer(data.get(f"expected_{split}_rows"), f"expected_{split}_rows", minimum=1)
        resources = self.resources
        if set(resources) != {
            *_RESOURCE_ENV_FIELDS,
            "alfworld_repository_revision",
            "embedding_model_revision",
            "expected_medrag_textbook_rows",
            "expected_swe_verified_rows",
            "medrag_textbooks_revision",
            "minimum_swe_docker_free_gib",
            "swe_docker_namespace",
            "swe_harness_revision",
            "swe_verified_revision",
            "webshop_goal_split",
            "webshop_repository_revision",
            "webshop_use_small",
        }:
            raise ValueError("baseline resource config has an incompatible field set")
        for field in _RESOURCE_ENV_FIELDS:
            _text(resources.get(field), field)
        _integer(
            resources.get("expected_swe_verified_rows"),
            "expected_swe_verified_rows",
            minimum=1,
        )
        _integer(
            resources.get("expected_medrag_textbook_rows"),
            "expected_medrag_textbook_rows",
            minimum=1,
        )
        for field in (
            "alfworld_repository_revision",
            "embedding_model_revision",
            "medrag_textbooks_revision",
            "swe_harness_revision",
            "swe_verified_revision",
            "webshop_repository_revision",
        ):
            revision = _text(resources.get(field), field)
            if len(revision) != 40:
                raise ValueError(f"{field} must be a complete Git revision")
        _integer(
            resources.get("minimum_swe_docker_free_gib"),
            "minimum_swe_docker_free_gib",
            minimum=1,
        )
        _text(resources.get("swe_docker_namespace"), "swe_docker_namespace")
        if resources.get("webshop_use_small") is not False:
            raise ValueError("formal WebShop must use the complete product catalog")
        if resources.get("webshop_goal_split") != "train":
            raise ValueError("formal WebShop must use the upstream training goal split")
        runtime = self.runtime
        assigned = (
            _integer(runtime.get("coordinator_physical_gpu"), "coordinator GPU"),
            _integer(runtime.get("primary_gradient_physical_gpu"), "primary gradient GPU"),
            _integer(runtime.get("standby_gradient_physical_gpu"), "standby gradient GPU"),
        )
        visible = runtime.get("steady_visible_physical_gpus")
        if visible != list(assigned[:2]):
            raise ValueError("steady CUDA visibility must contain only coordinator and primary")
        forbidden = runtime.get("forbidden_physical_gpus")
        if not isinstance(forbidden, list) or any(type(item) is not int for item in forbidden):
            raise TypeError("forbidden_physical_gpus must be an integer list")
        if set(assigned) & set(forbidden):
            raise ValueError("assigned GPUs must be allowed by the run configuration")
        initial = _integer(runtime.get("initial_micro_batch"), "initial_micro_batch", minimum=1)
        minimum = _integer(runtime.get("minimum_micro_batch"), "minimum_micro_batch", minimum=1)
        if minimum > initial:
            raise ValueError("minimum micro-batch cannot exceed the initial value")
        if runtime.get("tracking_mode") != "disabled":
            raise ValueError("formal baseline tracking must default to disabled")
        supervisor = self.supervisor
        if supervisor.get("mode") != "external":
            raise ValueError("formal baseline requires an external SGLang service")
        if (
            supervisor.get("require_lora") is not True
            or supervisor.get("require_metrics") is not True
        ):
            raise ValueError("external SGLang must require LoRA and metrics")
        inference = _integer(supervisor.get("expected_physical_gpu"), "inference GPU")
        if inference not in range(8) or any(index not in range(8) for index in assigned):
            raise ValueError("physical GPU indices must be between zero and seven")
        if inference in forbidden:
            raise ValueError("inference GPU must be allowed by the run configuration")
        _integer(supervisor.get("expected_context_length"), "context length", minimum=1)
        if any(key.casefold().startswith("bayesian") for key in self.training):
            raise ValueError("BayesianImprove controls are forbidden in the baseline")

    def resolved_environment(self) -> dict[str, str]:
        keys = {
            "model_path": _text(self.model.get("path_env"), "model path env"),
            "model_revision": _text(self.model.get("revision_env"), "model revision env"),
            "tokenizer_path": _text(self.model.get("tokenizer_path_env"), "tokenizer path env"),
            "train_data": _text(self.data.get("train_path_env"), "train path env"),
            "validation_data": _text(self.data.get("validation_path_env"), "validation path env"),
            "supervisor_api_base": _text(self.supervisor.get("api_base_env"), "supervisor API env"),
            "output_root": _text(self.runtime.get("output_root_env"), "output root env"),
        }
        keys.update(
            {
                field.removesuffix("_path_env"): _text(self.resources.get(field), field)
                for field in _RESOURCE_ENV_FIELDS
            }
        )
        resolved: dict[str, str] = {}
        for label, key in keys.items():
            value = os.environ.get(key)
            if value is None or not value.strip():
                raise RuntimeError(f"required environment variable is unset: {key}")
            resolved[label] = value
        return resolved

    def normalized_value(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], normalize_json(self.raw))


def blake2b_file(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_tree_identity(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.blake2b(digest_size=32)
    found = False
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode())
        digest.update(path.read_bytes())
    if not found:
        raise FileNotFoundError("model/tokenizer snapshot has no identity metadata")
    return digest.hexdigest()


def _parse_compute_processes(output: str) -> dict[str, list[int]]:
    process_map: dict[str, list[int]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2:
            raise RuntimeError("nvidia-smi returned an incompatible process row")
        uuid, pid = fields
        if "[N/A]" in fields:
            continue
        try:
            process_map.setdefault(uuid, []).append(int(pid))
        except ValueError as error:
            raise RuntimeError("nvidia-smi returned a non-numeric process id") from error
    return process_map


def _parse_gpu_utilization(value: str) -> int:
    # A driver-excluded device reports ``[N/A]``. Keep it visible to the
    # eight-device inventory, but never classify it as idle.
    return 100 if value == "[N/A]" else int(value)


def collect_gpu_observations() -> tuple[GPUObservation, ...]:
    processes = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    process_map = _parse_compute_processes(processes)
    gpus = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    observations: list[GPUObservation] = []
    for line in gpus.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError("nvidia-smi returned an incompatible GPU row")
        index, name, uuid, total, used, utilization = fields
        observations.append(
            GPUObservation(
                physical_index=int(index),
                name=name,
                uuid=uuid,
                memory_total_mib=int(total),
                memory_used_mib=int(used),
                utilization_percent=_parse_gpu_utilization(utilization),
                compute_pids=tuple(process_map.get(uuid, ())),
            )
        )
    return tuple(observations)


def resolve_gpu_assignment(config: FormalBaselineConfig) -> FourGPURoleAssignment:
    """Resolve infrastructure-only training GPU placement.

    The baseline YAML remains part of the checkpoint identity, while physical
    ordinals are host-local and may differ after a safe cross-host resume.  An
    explicit environment override therefore changes only the training roles;
    the external inference role remains fixed by the service contract. Hosts
    without a permitted standby can disable that role explicitly.
    """

    runtime = config.runtime
    raw_override = os.environ.get(_TRAINING_GPU_MAP_ENV)
    standby_disabled = os.environ.get(_DISABLE_GRADIENT_STANDBY_ENV) == "1"
    training_indices: tuple[int, int, int | None]
    if raw_override is None:
        training_indices = (
            _integer(runtime["coordinator_physical_gpu"], "coordinator GPU"),
            _integer(runtime["primary_gradient_physical_gpu"], "gradient primary GPU"),
            (
                None
                if standby_disabled
                else _integer(runtime["standby_gradient_physical_gpu"], "gradient standby GPU")
            ),
        )
    else:
        fields = raw_override.split(",")
        expected_fields = 2 if standby_disabled else 3
        if len(fields) != expected_fields or any(not field.isdecimal() for field in fields):
            raise ValueError(f"{_TRAINING_GPU_MAP_ENV} has an incompatible role count")
        values = [int(field) for field in fields]
        training_indices = (values[0], values[1], None if standby_disabled else values[2])
        if any(index < 0 or index >= 8 for index in training_indices if index is not None):
            raise ValueError(f"{_TRAINING_GPU_MAP_ENV} may use only physical GPUs 0 through 7")

    return FourGPURoleAssignment(
        inference=_integer(config.supervisor["expected_physical_gpu"], "inference GPU"),
        coordinator=training_indices[0],
        gradient_primary=training_indices[1],
        gradient_standby=training_indices[2],
    )


def resolve_training_cuda_visibility(config: FormalBaselineConfig, *, expanded: bool) -> str:
    """Resolve logical CUDA visibility without assuming host ordinal stability."""

    assignment = resolve_gpu_assignment(config)
    if expanded and assignment.gradient_standby is None:
        raise RuntimeError("gradient standby expansion is disabled for this run")
    raw_override = os.environ.get(_TRAINING_CUDA_VISIBLE_DEVICES_ENV)
    if raw_override is None:
        devices = [str(assignment.coordinator), str(assignment.gradient_primary)]
        if expanded:
            assert assignment.gradient_standby is not None
            devices.append(str(assignment.gradient_standby))
        return ",".join(devices)

    devices = raw_override.split(",")
    expected_devices = 2 if assignment.gradient_standby is None else 3
    if len(devices) != expected_devices or any(not device.startswith("GPU-") for device in devices):
        raise ValueError(f"{_TRAINING_CUDA_VISIBLE_DEVICES_ENV} has an incompatible role count")
    if len(set(devices)) != expected_devices:
        raise ValueError(f"{_TRAINING_CUDA_VISIBLE_DEVICES_ENV} GPU UUIDs must be distinct")
    return ",".join(devices if expanded else devices[:2])


def validate_training_gpu_ownership(
    config: FormalBaselineConfig, observations: tuple[GPUObservation, ...]
) -> FourGPURoleAssignment:
    assignment = resolve_gpu_assignment(config)
    by_index = {item.physical_index: item for item in observations}
    if set(by_index) != set(range(8)):
        raise RuntimeError("formal server must expose exactly physical GPUs 0 through 7")
    external_inference_uuid = os.environ.get(_EXTERNAL_INFERENCE_GPU_UUID_ENV)
    if external_inference_uuid is None and not by_index[assignment.inference].compute_pids:
        raise RuntimeError("external inference GPU has no serving process")
    if external_inference_uuid is not None and not external_inference_uuid.startswith("GPU-"):
        raise ValueError(f"{_EXTERNAL_INFERENCE_GPU_UUID_ENV} must contain a GPU UUID")
    training_indices = tuple(
        index
        for index in (
            assignment.coordinator,
            assignment.gradient_primary,
            assignment.gradient_standby,
        )
        if index is not None
    )
    for index in training_indices:
        observation = by_index[index]
        if not observation.is_launchable_idle:
            raise RuntimeError(f"training GPU {index} is unavailable or occupied")
        if observation.memory_free_mib < 70_000:
            raise RuntimeError(f"training GPU {index} lacks the required free memory")
    if external_inference_uuid in {
        by_index[assignment.coordinator].uuid,
        by_index[assignment.gradient_primary].uuid,
        *(
            ()
            if assignment.gradient_standby is None
            else (by_index[assignment.gradient_standby].uuid,)
        ),
    }:
        raise RuntimeError("external inference GPU overlaps a training role")
    uuid_override = os.environ.get(_TRAINING_CUDA_VISIBLE_DEVICES_ENV)
    if uuid_override is not None:
        expected_uuids = ",".join(by_index[index].uuid for index in training_indices)
        if uuid_override != expected_uuids:
            raise RuntimeError("training CUDA UUID visibility differs from the physical role map")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected = resolve_training_cuda_visibility(config, expanded=False)
    if visible != expected:
        raise RuntimeError(f"steady launch requires CUDA_VISIBLE_DEVICES={expected}")
    return assignment


def _json_request(url: str, *, payload: object | None, timeout: float) -> JsonValue:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(  # noqa: S310 - URL is validated by the config loader
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            if response.status != 200:
                raise RuntimeError("external SGLang returned a non-success status")
            return cast(JsonValue, normalize_json(json.loads(response.read(16 * 1024 * 1024))))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError("external SGLang preflight request failed") from error


def check_external_sglang(config: FormalBaselineConfig, api_base: str) -> dict[str, JsonValue]:
    supervisor = config.supervisor
    root = api_base.rstrip("/").removesuffix("/v1")
    models = _json_request(root + "/v1/models", payload=None, timeout=10)
    if not isinstance(models, dict) or not isinstance(models.get("data"), list):
        raise RuntimeError("external SGLang model list is malformed")
    model_data = models["data"]
    assert isinstance(model_data, list)
    expected_model = _text(supervisor["served_model"], "served model")
    entries = [item for item in model_data if isinstance(item, dict)]
    selected = next((item for item in entries if item.get("id") == expected_model), None)
    if selected is None:
        raise RuntimeError("expected served model is absent")
    if selected.get("max_model_len") != supervisor["expected_context_length"]:
        raise RuntimeError("external SGLang context length differs")
    metrics_request = urllib.request.Request(root + "/metrics", method="GET")  # noqa: S310
    with urllib.request.urlopen(metrics_request, timeout=10) as response:  # noqa: S310
        metrics = response.read(1024 * 1024).decode("utf-8", errors="replace")
    if "sglang:http_requests_total" not in metrics:
        raise RuntimeError("external SGLang metrics are not enabled")
    openapi = _json_request(root + "/openapi.json", payload=None, timeout=10)
    paths = openapi.get("paths") if isinstance(openapi, dict) else None
    required_routes = {"/load_lora_adapter", "/unload_lora_adapter"}
    if not isinstance(paths, dict) or not required_routes.issubset(paths):
        raise RuntimeError("external SGLang lacks transactional LoRA routes")
    completion = _json_request(
        root + "/v1/chat/completions",
        payload={
            "max_tokens": 1,
            "messages": [{"content": "Reply with OK.", "role": "user"}],
            "model": expected_model,
            "temperature": 0,
        },
        timeout=float(
            _integer(supervisor["request_timeout_seconds"], "request timeout", minimum=1)
        ),
    )
    if not isinstance(completion, dict) or not completion.get("choices"):
        raise RuntimeError("external SGLang OpenAI request failed validation")
    return {
        "context_length": selected["max_model_len"],
        "lora_control_routes": cast(JsonValue, sorted(required_routes)),
        "metrics": True,
        "mode": "external",
        "openai_request": "passed",
        "served_model": expected_model,
    }


def collect_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package, expected in EXPECTED_TRAIN_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required training package is missing: {package}") from error
        if actual.split("+", 1)[0] != expected:
            raise RuntimeError(f"training package version differs: {package}")
        versions[package] = actual
    return versions


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"required runtime resource is not a directory: {label}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"required runtime resource file is missing: {label}")


def _official_swe_instance_ids(train_path: Path) -> set[str]:
    instance_ids: set[str] = set()
    serialized = train_path.read_text(encoding="utf-8")
    try:
        records = json.loads(serialized)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in serialized.splitlines() if line.strip()]
    if not isinstance(records, list):
        raise RuntimeError("formal training data must be an array or JSONL stream")
    for record in records:
        if not isinstance(record, Mapping) or record.get("task_type") != "code_generation":
            continue
        extra = record.get("extra", {})
        if isinstance(extra, str):
            extra = json.loads(extra)
        if not isinstance(extra, Mapping) or not isinstance(extra.get("instance_id"), str):
            raise RuntimeError("formal SWE-bench record has no instance identity")
        code_files = record.get("code_files")
        if not isinstance(code_files, Mapping) or not code_files:
            raise RuntimeError("formal SWE-bench record has no source workspace")
        instance_ids.add(cast(str, extra["instance_id"]))
    if not instance_ids:
        raise RuntimeError("formal training mixture has no SWE-bench instances")
    return instance_ids


def inspect_swe_docker(minimum_free_gib: int) -> dict[str, JsonValue]:
    """Validate the Docker endpoint through its API, including one real container.

    The filesystem reported by ``DockerRootDir`` may live on another host when
    ``DOCKER_HOST`` points at an SSH-forwarded rootless socket.  Measuring free
    space from a disposable container keeps this check daemon-local instead of
    assuming that the client can stat the daemon's host path.
    """

    import docker  # type: ignore[import-untyped]

    client = None
    try:
        client = docker.from_env(timeout=20)
        if client.ping() is not True:
            raise RuntimeError("official SWE-bench Docker daemon did not answer ping")
        info = client.info()
        driver = str(info.get("Driver", ""))
        if driver not in _SWE_DOCKER_STORAGE_DRIVERS:
            raise RuntimeError("SWE-bench Docker storage driver is unsupported")
        try:
            client.images.get(_SWE_DOCKER_SMOKE_IMAGE)
        except docker.errors.ImageNotFound:
            client.images.pull(_SWE_DOCKER_SMOKE_IMAGE)
        output = client.containers.run(
            _SWE_DOCKER_SMOKE_IMAGE,
            ["sh", "-c", "df -Pk / | tail -1"],
            network_disabled=True,
            remove=True,
            stdout=True,
            stderr=True,
        )
        if not isinstance(output, bytes):
            raise RuntimeError("Docker storage smoke returned an incompatible result")
        fields = output.decode("utf-8", errors="strict").strip().split()
        if len(fields) < 6:
            raise RuntimeError("Docker storage smoke returned malformed df output")
        available_kib = int(fields[3])
        free_gib = available_kib // 1024**2
        if free_gib < minimum_free_gib:
            raise RuntimeError("SWE-bench Docker storage is below the configured safety floor")
        security_options = info.get("SecurityOptions", [])
        if not isinstance(security_options, list):
            raise RuntimeError("Docker security options are malformed")
        return {
            "driver": driver,
            "endpoint": "environment" if os.environ.get("DOCKER_HOST") else "default",
            "free_gib": free_gib,
            "rootless": "name=rootless" in security_options,
            "server_version": str(info.get("ServerVersion", "")),
            "smoke": "passed",
        }
    except (
        docker.errors.DockerException,
        OSError,
        TimeoutError,
        UnicodeError,
        ValueError,
    ) as error:
        raise RuntimeError("official SWE-bench Docker daemon is unavailable") from error
    finally:
        if client is not None:
            client.close()


def validate_swe_harness_runtime(swe_harness: Path) -> None:
    """Import the pinned official evaluator in an isolated preflight process."""

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(swe_harness), existing_pythonpath) if part
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(  # noqa: S603 - the current interpreter is trusted
        [sys.executable, "-c", "import swebench.harness.run_evaluation"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("official SWE-bench evaluator dependencies are unavailable")


def validate_runtime_resources(
    config: FormalBaselineConfig, environment: Mapping[str, str]
) -> dict[str, JsonValue]:
    """Validate every non-model resource used by the official seven-task mixture.

    This intentionally runs before importing torch.  Private paths are never returned;
    the manifest records only non-sensitive readiness facts.
    """

    if shutil.which("javac") is None:
        raise RuntimeError("WebShop requires a Java compiler on PATH")

    alf_repo = Path(environment["alfworld_repository"])
    alf_data = Path(environment["alfworld_data"])
    alf_config = Path(environment["alfworld_config"])
    webshop_repo = Path(environment["webshop_repository"])
    webshop_items = Path(environment["webshop_items"])
    webshop_attributes = Path(environment["webshop_attributes"])
    webshop_human_attributes = Path(environment["webshop_human_attributes"])
    webshop_search_index = Path(environment["webshop_search_index"])
    swe_verified = Path(environment["swe_verified"])
    swe_harness = Path(environment["swe_harness"])
    medrag = Path(environment["medrag_textbooks"])
    embedding = Path(environment["embedding_model"])

    _require_directory(alf_repo / "alfworld", "alfworld_repository")
    _require_directory(alf_data, "alfworld_data")
    _require_file(alf_config, "alfworld_config")
    webshop_package = (
        webshop_repo / "web_agent_site"
        if (webshop_repo / "web_agent_site").is_dir()
        else webshop_repo / "webshop" / "web_agent_site"
    )
    _require_directory(webshop_package, "webshop_repository")
    _require_file(webshop_items, "webshop_items")
    _require_file(webshop_attributes, "webshop_attributes")
    _require_file(webshop_human_attributes, "webshop_human_attributes")
    _require_directory(webshop_search_index, "webshop_search_index")
    _require_directory(swe_verified, "swe_verified")
    _require_file(swe_verified / ".source_revision", "swe_verified_revision")
    _require_directory(swe_harness / "swebench" / "harness", "swe_harness")
    _require_file(medrag / "bm25_index.pkl", "medrag_index")
    _require_file(medrag / "all_chunks.jsonl", "medrag_corpus")
    _require_file(medrag / ".source_revision", "medrag_revision")
    _require_file(embedding / "config.json", "embedding_model")
    _require_file(embedding / ".source_revision", "embedding_model_revision")
    for repository, field, label in (
        (alf_repo, "alfworld_repository_revision", "ALFWorld"),
        (webshop_repo, "webshop_repository_revision", "WebShop"),
    ):
        revision = subprocess.run(  # noqa: S603 - fixed executable and argv
            ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        if revision != config.resources[field]:
            raise RuntimeError(f"{label} repository revision differs")
    if (medrag / ".source_revision").read_text(encoding="utf-8").strip() != str(
        config.resources["medrag_textbooks_revision"]
    ):
        raise RuntimeError("MedRAG textbook source revision differs")
    if (embedding / ".source_revision").read_text(encoding="utf-8").strip() != str(
        config.resources["embedding_model_revision"]
    ):
        raise RuntimeError("embedding model source revision differs")
    with (medrag / "all_chunks.jsonl").open("rb") as corpus:
        medrag_rows = sum(1 for _ in corpus)
    expected_medrag_rows = cast(int, config.resources["expected_medrag_textbook_rows"])
    if medrag_rows != expected_medrag_rows:
        raise RuntimeError("MedRAG textbook snippet count differs")

    import datasets  # type: ignore[import-untyped]

    verified = datasets.load_from_disk(str(swe_verified))
    expected_rows = cast(int, config.resources["expected_swe_verified_rows"])
    if len(verified) != expected_rows:
        raise RuntimeError("SWE-bench Verified row count differs")
    if (swe_verified / ".source_revision").read_text(encoding="utf-8").strip() != str(
        config.resources["swe_verified_revision"]
    ):
        raise RuntimeError("SWE-bench Verified source revision differs")
    harness_revision = subprocess.run(  # noqa: S603 - fixed executable and argv
        ["/usr/bin/git", "-C", str(swe_harness), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    if harness_revision != config.resources["swe_harness_revision"]:
        raise RuntimeError("SWE-bench harness revision differs")
    validate_swe_harness_runtime(swe_harness)
    minimum_docker_free_gib = cast(int, config.resources["minimum_swe_docker_free_gib"])
    docker_status = inspect_swe_docker(minimum_docker_free_gib)
    verified_by_id = {str(row["instance_id"]): row for row in verified}
    instance_ids = _official_swe_instance_ids(Path(environment["train_data"]))
    missing_instances = instance_ids.difference(verified_by_id)
    if missing_instances:
        raise RuntimeError("formal training SWE-bench instance is not in Verified")
    import_paths = (str(alf_repo), str(webshop_package.parent), str(swe_harness))
    old_path = list(sys.path)
    try:
        sys.path[:0] = [path for path in import_paths if path not in sys.path]
        for module in (
            "alfworld",
            "bs4",
            "cleantext",
            "flask",
            "gym",
            "pyserini",
            "rank_bm25",
            "rich",
            "spacy",
            "swebench",
            "textworld",
            "thefuzz",
            "web_agent_site",
        ):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(f"required runtime module is unavailable: {module}")
    finally:
        sys.path[:] = old_path

    return {
        "alfworld": "ready",
        "embedding_model": "ready",
        "medrag": "ready",
        "medrag_textbook_rows": expected_medrag_rows,
        "swe_bench_verified_rows": expected_rows,
        "swe_harness_revision": harness_revision,
        "swe_docker_driver": docker_status["driver"],
        "swe_docker_endpoint": docker_status["endpoint"],
        "swe_docker_free_gib": docker_status["free_gib"],
        "swe_docker_rootless": docker_status["rootless"],
        "swe_docker_server_version": docker_status["server_version"],
        "swe_docker_smoke": docker_status["smoke"],
        "swe_verified_revision": str(config.resources["swe_verified_revision"]),
        "swe_training_instances": len(instance_ids),
        "swe_training_source_workspaces": len(instance_ids),
        "webshop": "ready",
    }


def build_run_manifest(
    config: FormalBaselineConfig,
    *,
    git_commit: str,
    observations: tuple[GPUObservation, ...],
) -> dict[str, JsonValue]:
    environment = config.resolved_environment()
    assignment = validate_training_gpu_ownership(config, observations)
    model_root = Path(environment["model_path"])
    tokenizer_root = Path(environment["tokenizer_path"])
    train_path = Path(environment["train_data"])
    validation_path = Path(environment["validation_data"])
    if blake2b_file(train_path) != config.data["train_blake2b"]:
        raise RuntimeError("formal training data identity differs")
    if blake2b_file(validation_path) != config.data["validation_blake2b"]:
        raise RuntimeError("formal validation data identity differs")
    output = Path(environment["output_root"])
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    disk = shutil.disk_usage(output)
    if disk.free < 30 * 1024**3:
        raise RuntimeError("output filesystem is below the initial 30 GiB safety floor")
    by_index = {item.physical_index: item for item in observations}
    external_inference_uuid = os.environ.get(_EXTERNAL_INFERENCE_GPU_UUID_ENV)
    roles: dict[str, JsonValue] = {
        role: {
            "logical_rank": None if role == "inference" else rank,
            "physical_index": index,
            "uuid": (
                external_inference_uuid
                if role == "inference" and external_inference_uuid is not None
                else by_index[index].uuid
            ),
        }
        for role, index, rank in (
            ("inference", assignment.inference, None),
            ("coordinator", assignment.coordinator, 0),
            ("gradient_primary", assignment.gradient_primary, 1),
        )
    }
    roles["gradient_standby"] = (
        {"enabled": False, "logical_rank": None, "physical_index": None, "uuid": None}
        if assignment.gradient_standby is None
        else {
            "enabled": True,
            "logical_rank": None,
            "physical_index": assignment.gradient_standby,
            "uuid": by_index[assignment.gradient_standby].uuid,
        }
    )
    manifest: dict[str, object] = {
        "config": config.normalized_value(),
        "config_blake2b": hashlib.blake2b(
            json.dumps(config.normalized_value(), sort_keys=True).encode(), digest_size=32
        ).hexdigest(),
        "data": {
            "train": {
                "blake2b": config.data["train_blake2b"],
                "rows": config.data["expected_train_rows"],
            },
            "validation": {
                "blake2b": config.data["validation_blake2b"],
                "rows": config.data["expected_validation_rows"],
            },
        },
        "format": RUN_MANIFEST_FORMAT,
        "git_commit": git_commit,
        "gpu_roles": roles,
        "model": {
            "metadata_blake2b": metadata_tree_identity(
                model_root,
                ("config.json", "model.safetensors.index.json", "generation_config.json"),
            ),
            "repo_id": config.model["repo_id"],
            "revision": environment["model_revision"],
        },
        "tokenizer": {
            "metadata_blake2b": metadata_tree_identity(
                tokenizer_root,
                (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.json",
                    "merges.txt",
                ),
            )
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "resources": validate_runtime_resources(config, environment),
        "software": collect_package_versions(),
        "storage": {
            "free_bytes_at_preflight": disk.free,
            "total_bytes": disk.total,
        },
        "supervisor": check_external_sglang(config, environment["supervisor_api_base"]),
    }
    return cast(dict[str, JsonValue], normalize_json(manifest))


__all__ = [
    "EXPECTED_TRAIN_PACKAGES",
    "RUN_MANIFEST_FORMAT",
    "FormalBaselineConfig",
    "blake2b_file",
    "build_run_manifest",
    "check_external_sglang",
    "collect_gpu_observations",
    "collect_package_versions",
    "inspect_swe_docker",
    "metadata_tree_identity",
    "resolve_gpu_assignment",
    "resolve_training_cuda_visibility",
    "validate_runtime_resources",
    "validate_training_gpu_ownership",
]
