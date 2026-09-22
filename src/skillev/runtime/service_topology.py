"""Private physical placement, separate from task policy and training state."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from skillev.contracts import JsonValue, normalize_json


@dataclass(frozen=True, slots=True)
class InferenceService:
    service_id: str
    endpoint: str
    gpu_uuid: str
    request_capacity: int = 8
    token_capacity: int | None = None

    def __post_init__(self) -> None:
        if type(self.request_capacity) is not int or self.request_capacity < 1:
            raise ValueError("request capacity must be positive")
        if self.token_capacity is not None and (
            type(self.token_capacity) is not int or self.token_capacity < 1
        ):
            raise ValueError("token capacity must be positive")
        parsed = urlsplit(self.endpoint)
        if not self.service_id or not self.gpu_uuid.startswith("GPU-"):
            raise ValueError("service identity and physical GPU UUID are required")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("service endpoint must be an HTTP URL without credentials")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "service_id": self.service_id,
            "endpoint": self.endpoint,
            "gpu_uuid": self.gpu_uuid,
            "request_capacity": self.request_capacity,
            "token_capacity": self.token_capacity,
        }

    @classmethod
    def from_value(cls, value: object) -> InferenceService:
        if (
            not isinstance(value, dict)
            or not {"service_id", "endpoint", "gpu_uuid"} <= set(value)
            or set(value)
            - {"service_id", "endpoint", "gpu_uuid", "request_capacity", "token_capacity"}
        ):
            raise ValueError("incompatible service placement")
        if any(not isinstance(value[k], str) for k in ("service_id", "endpoint", "gpu_uuid")):
            raise TypeError("service placement values must be text")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ServiceTopology:
    """One physical service may serve several explicit logical base/actor roles.

    Pools refer to the catalog instead of allocating the same device twice.
    Rank0 is both coordinator and gradient worker; no idle coordinator rank.
    No scheduler here claims physical isolation from unrelated SSH processes.
    """

    services: tuple[InferenceService, ...]
    actor_pool: tuple[str, ...]
    judge_pool: tuple[str, ...]
    author_pool: tuple[str, ...]
    gradient_workers: tuple[str, ...]

    def __post_init__(self) -> None:
        ids = [v.service_id for v in self.services]
        endpoints = [v.endpoint.rstrip("/").removesuffix("/v1") for v in self.services]
        devices = [v.gpu_uuid for v in self.services] + list(self.gradient_workers)
        if not ids or len(ids) != len(set(ids)) or len(endpoints) != len(set(endpoints)):
            raise ValueError("service identities and endpoints must be unique")
        if len(devices) != len(set(devices)):
            raise ValueError("different physical workers cannot own the same GPU")
        if len(self.gradient_workers) < 2 or any(
            not v.startswith("GPU-") for v in self.gradient_workers
        ):
            raise ValueError("formal execution requires at least two participating gradient GPUs")
        used: set[str] = set()
        for pool in (self.actor_pool, self.judge_pool, self.author_pool):
            if not pool or len(pool) != len(set(pool)) or not set(pool) <= set(ids):
                raise ValueError("each role requires distinct registered service references")
            used.update(pool)
        if used != set(ids):
            raise ValueError("every allocated service must have an explicit role")

    def members(self, role: str) -> tuple[InferenceService, ...]:
        pools = {"actor": self.actor_pool, "judge": self.judge_pool, "author": self.author_pool}
        catalog = {v.service_id: v for v in self.services}
        return tuple(catalog[key] for key in pools[role])

    def require_device_mapping(self, visible: str, world_size: int) -> None:
        if visible.split(",") != list(self.gradient_workers) or world_size != len(
            self.gradient_workers
        ):
            raise ValueError(
                "torchrun visibility/order must match the participating gradient ranks"
            )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": "skillev-service-topology@1",
            "services": [v.to_value() for v in self.services],
            "actor_pool": list(self.actor_pool),
            "judge_pool": list(self.judge_pool),
            "author_pool": list(self.author_pool),
            "gradient_workers": list(self.gradient_workers),
        }

    @classmethod
    def from_value(cls, value: object) -> ServiceTopology:
        data = normalize_json(value)
        if (
            not isinstance(data, dict)
            or set(data)
            != {"format", "services", "actor_pool", "judge_pool", "author_pool", "gradient_workers"}
            or data["format"] != "skillev-service-topology@1"
        ):
            raise ValueError("incompatible service topology")
        services = data["services"]
        if not isinstance(services, list):
            raise TypeError("service catalog must be a list")
        pools = []
        for name in ("actor_pool", "judge_pool", "author_pool", "gradient_workers"):
            pool = data[name]
            if not isinstance(pool, list) or any(not isinstance(v, str) for v in pool):
                raise TypeError("role membership must be a list of identifiers")
            pools.append(tuple(pool))
        return cls(tuple(InferenceService.from_value(v) for v in services), *pools)
