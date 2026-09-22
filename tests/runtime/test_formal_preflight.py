from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skillev.runtime.formal_preflight import (
    FormalBaselineConfig,
    _official_swe_instance_ids,
    _parse_compute_processes,
    _parse_gpu_utilization,
    inspect_swe_docker,
    resolve_gpu_assignment,
    resolve_training_cuda_visibility,
    validate_runtime_resources,
    validate_swe_harness_runtime,
    validate_training_gpu_ownership,
)
from skillev.runtime.gpu_topology import GPUObservation


def _config(tmp_path: Path) -> FormalBaselineConfig:
    raw = yaml.safe_load(Path("configs/baseline/paper_v1_250step.yaml").read_text())
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return FormalBaselineConfig.read(path)


def _gpu(index: int, *pids: int, utilization: int = 0) -> GPUObservation:
    return GPUObservation(
        physical_index=index,
        name="NVIDIA H800",
        uuid=f"GPU-{index}",
        memory_total_mib=81559,
        memory_used_mib=64000 if index == 1 else 4,
        utilization_percent=utilization,
        compute_pids=tuple(pids),
    )


def test_formal_config_requires_external_service_without_fixed_gpu_exclusions(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    assert config.supervisor["mode"] == "external"
    assert not config.runtime["forbidden_physical_gpus"]
    assert config.runtime["steady_visible_physical_gpus"] == [2, 3]


def test_compute_process_parser_ignores_driver_na_rows() -> None:
    assert _parse_compute_processes("GPU-0, 123\nGPU-1, [N/A]\n[N/A], [N/A]\n") == {"GPU-0": [123]}


@pytest.mark.parametrize("inference_index", range(8))
def test_baseline_accepts_all_host_local_physical_role_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inference_index: int
) -> None:
    monkeypatch.delenv("SKILLEV_TRAINING_GPU_MAP", raising=False)
    monkeypatch.delenv("SKILLEV_DISABLE_GRADIENT_STANDBY", raising=False)
    raw = yaml.safe_load(Path("configs/baseline/paper_v1_250step.yaml").read_text())
    indices = tuple((inference_index + offset) % 8 for offset in range(4))
    raw["services"]["supervisor"]["expected_physical_gpu"] = indices[0]
    for name, index in zip(
        (
            "coordinator_physical_gpu",
            "primary_gradient_physical_gpu",
            "standby_gradient_physical_gpu",
        ),
        indices[1:],
        strict=True,
    ):
        raw["runtime"][name] = index
    raw["runtime"]["steady_visible_physical_gpus"] = list(indices[1:3])
    path = tmp_path / "all-gpu-indices.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = FormalBaselineConfig.read(path)

    assert resolve_gpu_assignment(config).physical_indices == indices


def test_compute_process_parser_rejects_malformed_pid() -> None:
    with pytest.raises(RuntimeError):
        _parse_compute_processes("GPU-0, not-a-pid\n")


def test_gpu_utilization_parser_marks_driver_na_as_unavailable() -> None:
    assert _parse_gpu_utilization("[N/A]") == 100
    assert _parse_gpu_utilization("37") == 37


def test_training_visibility_supports_an_explicitly_disabled_standby(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKILLEV_DISABLE_GRADIENT_STANDBY", "1")
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "4,7")
    monkeypatch.setenv(
        "SKILLEV_TRAINING_CUDA_VISIBLE_DEVICES",
        "GPU-coordinator,GPU-primary",
    )

    assignment = resolve_gpu_assignment(_config(tmp_path))
    assert assignment.gradient_standby is None
    assert resolve_training_cuda_visibility(_config(tmp_path), expanded=False) == (
        "GPU-coordinator,GPU-primary"
    )
    with pytest.raises(RuntimeError):
        resolve_training_cuda_visibility(_config(tmp_path), expanded=True)


def test_training_visibility_can_use_physical_gpu_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKILLEV_DISABLE_GRADIENT_STANDBY", "1")
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "0,7")

    assignment = resolve_gpu_assignment(_config(tmp_path))

    assert (assignment.coordinator, assignment.gradient_primary) == (0, 7)


def test_training_ownership_rejects_a_driver_unavailable_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKILLEV_DISABLE_GRADIENT_STANDBY", "1")
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "4,7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    observations = tuple(
        _gpu(index, *((61104,) if index == 1 else ()), utilization=100 if index == 4 else 0)
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        validate_training_gpu_ownership(_config(tmp_path), observations)


def test_preflight_accepts_configured_external_service_and_idle_training_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    observations = tuple(
        _gpu(index, *((44325,) if index == 0 else (61104,) if index == 1 else ()))
        for index in range(8)
    )

    assignment = validate_training_gpu_ownership(_config(tmp_path), observations)

    assert assignment.physical_indices == (1, 2, 3, 4)


def test_preflight_accepts_explicit_idle_host_local_training_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "3,4,7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    observations = tuple(
        _gpu(index, *((61104,) if index == 1 else (438796,) if index == 2 else ()))
        for index in range(8)
    )

    assignment = validate_training_gpu_ownership(_config(tmp_path), observations)

    assert assignment.physical_indices == (1, 3, 4, 7)


def test_preflight_accepts_uuid_visibility_for_host_local_training_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "3,4,7")
    monkeypatch.setenv("SKILLEV_TRAINING_CUDA_VISIBLE_DEVICES", "GPU-3,GPU-4,GPU-7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-3,GPU-4")
    observations = tuple(
        _gpu(index, *((61104,) if index == 1 else (438796,) if index == 2 else ()))
        for index in range(8)
    )

    assignment = validate_training_gpu_ownership(_config(tmp_path), observations)

    assert assignment.physical_indices == (1, 3, 4, 7)
    assert resolve_training_cuda_visibility(_config(tmp_path), expanded=True) == (
        "GPU-3,GPU-4,GPU-7"
    )


def test_preflight_rejects_uuid_visibility_that_differs_from_role_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "3,4,7")
    monkeypatch.setenv("SKILLEV_TRAINING_CUDA_VISIBLE_DEVICES", "GPU-4,GPU-3,GPU-7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-4,GPU-3")
    observations = tuple(
        _gpu(index, *((61104,) if index == 1 else (438796,) if index == 2 else ()))
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        validate_training_gpu_ownership(_config(tmp_path), observations)


def test_preflight_accepts_external_inference_on_another_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "3,4,7")
    monkeypatch.setenv("SKILLEV_TRAINING_CUDA_VISIBLE_DEVICES", "GPU-3,GPU-4,GPU-7")
    monkeypatch.setenv("SKILLEV_EXTERNAL_INFERENCE_GPU_UUID", "GPU-remote-inference")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-3,GPU-4")
    observations = tuple(_gpu(index, *((438796,) if index == 2 else ())) for index in range(8))

    assignment = validate_training_gpu_ownership(_config(tmp_path), observations)

    assert assignment.physical_indices == (1, 3, 4, 7)


def test_preflight_rejects_external_inference_that_overlaps_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "3,4,7")
    monkeypatch.setenv("SKILLEV_TRAINING_CUDA_VISIBLE_DEVICES", "GPU-3,GPU-4,GPU-7")
    monkeypatch.setenv("SKILLEV_EXTERNAL_INFERENCE_GPU_UUID", "GPU-3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-3,GPU-4")
    observations = tuple(_gpu(index) for index in range(8))

    with pytest.raises(RuntimeError):
        validate_training_gpu_ownership(_config(tmp_path), observations)


@pytest.mark.parametrize("value", ["", "3,4", "3,4,7,6", "1,3,4", "3,x,7"])
def test_explicit_training_map_rejects_invalid_or_overlapping_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", value)

    with pytest.raises((ValueError, TypeError)):
        resolve_gpu_assignment(_config(tmp_path))


@pytest.mark.parametrize("visible", [None, "0,1", "2,3,4", "3,2"])
def test_preflight_rejects_wrong_cuda_visibility_before_torch_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visible: str | None,
) -> None:
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    observations = tuple(
        _gpu(index, *((44325,) if index == 0 else (61104,) if index == 1 else ()))
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        validate_training_gpu_ownership(_config(tmp_path), observations)


def test_preflight_rejects_an_occupied_gradient_gpu(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    observations = tuple(
        _gpu(index, *((61104,) if index == 1 else (9000,) if index == 3 else ()))
        for index in range(8)
    )

    with pytest.raises(RuntimeError):
        validate_training_gpu_ownership(_config(tmp_path), observations)


def test_baseline_rejects_bayesian_feature_flags(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/baseline/paper_v1_250step.yaml").read_text())
    raw["training"]["bayesian_improve"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError):
        FormalBaselineConfig.read(path)


def test_runtime_resource_preflight_fails_closed_without_private_assets(
    tmp_path: Path,
) -> None:
    environment = {
        field.removesuffix("_path_env"): str(tmp_path / field)
        for field in _config(tmp_path).resources
        if field.endswith("_path_env")
    }

    with pytest.raises(RuntimeError):
        validate_runtime_resources(_config(tmp_path), environment)


def test_runtime_resource_preflight_requires_javac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = {
        field.removesuffix("_path_env"): str(tmp_path / field)
        for field in _config(tmp_path).resources
        if field.endswith("_path_env")
    }
    monkeypatch.setattr("skillev.runtime.formal_preflight.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError):
        validate_runtime_resources(_config(tmp_path), environment)


def test_formal_swe_records_require_a_source_workspace(tmp_path: Path) -> None:
    training_data = tmp_path / "train.json"
    training_data.write_text(
        '[{"task_type":"code_generation","extra":{"instance_id":"example"}}]',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        _official_swe_instance_ids(training_data)


def test_swe_harness_runtime_import_checks_transitive_dependencies(tmp_path: Path) -> None:
    package = tmp_path / "swebench" / "harness"
    package.mkdir(parents=True)
    (tmp_path / "swebench" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run_evaluation.py").write_text(
        "import dependency_that_is_not_installed\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        validate_swe_harness_runtime(tmp_path)


def test_swe_harness_runtime_import_accepts_complete_evaluator(tmp_path: Path) -> None:
    package = tmp_path / "swebench" / "harness"
    package.mkdir(parents=True)
    (tmp_path / "swebench" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run_evaluation.py").write_text("READY = True\n", encoding="utf-8")

    validate_swe_harness_runtime(tmp_path)


class _DockerImages:
    def __init__(self, *, missing: bool = False) -> None:
        self.missing = missing
        self.pulled: list[str] = []

    def get(self, image: str) -> object:
        if self.missing:
            import docker

            raise docker.errors.ImageNotFound("not present")
        return object()

    def pull(self, image: str) -> object:
        self.pulled.append(image)
        return object()


class _DockerContainers:
    def __init__(self, output: bytes) -> None:
        self.output = output
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(self, *args: object, **kwargs: object) -> bytes:
        self.calls.append((args, kwargs))
        return self.output


class _DockerClient:
    def __init__(
        self,
        *,
        driver: str = "overlay2",
        missing_image: bool = False,
        free_kib: int = 800 * 1024**2,
    ) -> None:
        self.images = _DockerImages(missing=missing_image)
        self.containers = _DockerContainers(f"overlay 1000000000 1 {free_kib} 1% /\n".encode())
        self.driver = driver
        self.closed = False

    def ping(self) -> bool:
        return True

    def info(self) -> dict[str, object]:
        return {
            "Driver": self.driver,
            "SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless"],
            "ServerVersion": "29.3.0",
        }

    def close(self) -> None:
        self.closed = True


def test_swe_docker_preflight_uses_api_and_real_container_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import docker

    client = _DockerClient(missing_image=True)
    monkeypatch.setattr(docker, "from_env", lambda *, timeout: client)
    monkeypatch.setenv("DOCKER_HOST", "unix:///private/forwarded-docker.sock")

    result = inspect_swe_docker(120)

    assert result["driver"] == "overlay2"
    assert result["rootless"] is True
    assert result["endpoint"] == "environment"
    assert client.images.pulled
    assert client.containers.calls
    _, kwargs = client.containers.calls[0]
    assert kwargs["network_disabled"] is True
    assert kwargs["remove"] is True
    assert client.closed is True


@pytest.mark.parametrize(
    ("client", "minimum_free_gib"),
    [
        (_DockerClient(driver="vfs"), 120),
        (_DockerClient(free_kib=10 * 1024**2), 120),
    ],
)
def test_swe_docker_preflight_rejects_unsafe_storage(
    client: _DockerClient,
    minimum_free_gib: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import docker

    monkeypatch.setattr(docker, "from_env", lambda *, timeout: client)

    with pytest.raises(RuntimeError):
        inspect_swe_docker(minimum_free_gib)

    assert client.closed is True
