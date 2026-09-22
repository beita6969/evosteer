"""Strict file-backed deployment for the three official process environments.

The official WebShop, ALFWorld, and ScienceWorld packages require incompatible
interpreters.  The process-preparation CLI therefore imports one private
``module:symbol`` factory.  This module supplies the production implementation
of that factory while keeping every host path and task identity in an
untracked, canonical JSON configuration.

The configuration does not discover repositories, games, deployment assets,
or substitute interpreters.  Repository locations follow the sole acquisition
layout, revisions come from the committed acquisition lock, and every
remaining deployment input is explicit in the private configuration.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias, cast

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.experiments import FIXED_SEED, Benchmark
from skillev.rollout import RolloutTask

from .acquisition import (
    BenchmarkAcquisitionLock,
    _read_published_canonical_record,
)
from .official_process import (
    ALFWorldGameDeployment,
    JsonArrayWebShopDeployment,
    OfficialALFWorldProcessFactory,
    OfficialScienceWorldProcessFactory,
    OfficialWebShopProcessFactory,
    PinnedOfficialProcess,
    SQLiteWebShopDeployment,
    WebShopDeployment,
)
from .official_process_preparation import OfficialProcessPreparationDeployment
from .process_catalog import ProductionProcessCatalogDependencies
from .process_preparation import ExternalProcessTaskRecord, ProcessTaskManifestInput

OFFICIAL_PROCESS_DEPLOYMENT_FORMAT = "skillev-private-official-process-deployment@2"
_PROCESS_BENCHMARKS = (
    Benchmark.WEBSHOP,
    Benchmark.ALFWORLD,
    Benchmark.SCIENCE_WORLD,
)
_EXTERNAL_PROCESS_BENCHMARKS = (
    Benchmark.APPWORLD,
    Benchmark.BFCL_V3,
)
_REVISION_LENGTH = 40


def _wire_object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an invalid field set")
    return normalized


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value:
        raise ValueError(f"{field} must be text without NUL")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} cannot be empty")
    return value


def _relative_path(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError(f"{field} must be a normalized relative POSIX path")
    return text


def _absolute_path(value: object, *, field: str) -> Path:
    text = _text(value, field=field)
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return path


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _relative_files(value: object, *, benchmark: Benchmark) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{benchmark.value} asset_relative_files must be a non-empty array")
    if any(type(item) is not str for item in value):
        raise TypeError(f"{benchmark.value} asset_relative_files must contain text")
    files = tuple(
        _relative_path(item, field=f"{benchmark.value} asset path")
        for item in cast(list[str], value)
    )
    if files != tuple(sorted(files)) or len(set(files)) != len(files):
        raise ValueError(f"{benchmark.value} asset_relative_files must be sorted and unique")
    prefix = f"{benchmark.value}/"
    if any(not path.startswith(prefix) for path in files):
        raise ValueError(f"{benchmark.value} assets must remain below its benchmark root")
    return files


def _asset_tuple(
    value: object,
    *,
    benchmark: Benchmark,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{benchmark.value} asset_relative_files must be a tuple")
    return _relative_files(list(value), benchmark=benchmark)


def _existing_target_path(
    target_root: Path,
    relative_path: str,
    *,
    directory: bool,
    label: str,
) -> Path:
    path = target_root / PurePosixPath(relative_path)
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} must identify an existing {kind}")
    return path


def _relative_to_target(target_root: Path, path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(target_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must remain below target_root") from error


@dataclass(frozen=True, slots=True)
class OfficialProcessRuntimeConfig:
    """One explicit interpreter and worker timeout."""

    interpreter_path: Path
    request_timeout_seconds: float

    def __post_init__(self) -> None:
        interpreter = self.interpreter_path
        if not isinstance(interpreter, Path) or not interpreter.is_absolute():
            raise ValueError("official interpreter_path must be an absolute Path")
        if not interpreter.is_file():
            raise FileNotFoundError(interpreter)
        object.__setattr__(
            self,
            "request_timeout_seconds",
            _positive_number(
                self.request_timeout_seconds,
                field="official request_timeout_seconds",
            ),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "interpreter_path": self.interpreter_path.as_posix(),
            "request_timeout_seconds": self.request_timeout_seconds,
        }

    @classmethod
    def from_value(cls, value: object) -> OfficialProcessRuntimeConfig:
        data = _wire_object(
            value,
            fields=frozenset({"interpreter_path", "request_timeout_seconds"}),
            label="official process runtime",
        )
        return cls(
            interpreter_path=_absolute_path(
                data["interpreter_path"],
                field="official interpreter_path",
            ),
            request_timeout_seconds=_positive_number(
                data["request_timeout_seconds"],
                field="official request_timeout_seconds",
            ),
        )


@dataclass(frozen=True, slots=True)
class JsonArrayWebShopDeploymentConfig:
    runtime: OfficialProcessRuntimeConfig
    products_relative_path: str
    search_index_relative_path: str
    asset_relative_files: tuple[str, ...]
    inventory_size: int
    kind: str = "json-array"

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, OfficialProcessRuntimeConfig):
            raise TypeError("WebShop runtime config is incompatible")
        products = _relative_path(
            self.products_relative_path,
            field="WebShop products_relative_path",
        )
        search_index = _relative_path(
            self.search_index_relative_path,
            field="WebShop search_index_relative_path",
        )
        if not products.startswith("webshop/") or not search_index.startswith("webshop/"):
            raise ValueError("WebShop deployment paths must remain below webshop/")
        assets = _asset_tuple(
            self.asset_relative_files,
            benchmark=Benchmark.WEBSHOP,
        )
        if products not in assets:
            raise ValueError("WebShop products file must belong to its asset snapshot")
        if self.inventory_size not in {100, 1_000, 100_000}:
            raise ValueError("WebShop JSON inventory size is unsupported")
        if self.kind != "json-array":
            raise ValueError("WebShop JSON deployment kind is incompatible")
        object.__setattr__(self, "products_relative_path", products)
        object.__setattr__(self, "search_index_relative_path", search_index)
        object.__setattr__(self, "asset_relative_files", assets)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "asset_relative_files": list(self.asset_relative_files),
            "inventory_size": self.inventory_size,
            "kind": self.kind,
            "products_relative_path": self.products_relative_path,
            "runtime": self.runtime.to_value(),
            "search_index_relative_path": self.search_index_relative_path,
        }

    @classmethod
    def from_value(cls, value: object) -> JsonArrayWebShopDeploymentConfig:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "asset_relative_files",
                    "inventory_size",
                    "kind",
                    "products_relative_path",
                    "runtime",
                    "search_index_relative_path",
                }
            ),
            label="WebShop process deployment",
        )
        return cls(
            runtime=OfficialProcessRuntimeConfig.from_value(data["runtime"]),
            products_relative_path=_relative_path(
                data["products_relative_path"],
                field="WebShop products_relative_path",
            ),
            search_index_relative_path=_relative_path(
                data["search_index_relative_path"],
                field="WebShop search_index_relative_path",
            ),
            asset_relative_files=_relative_files(
                data["asset_relative_files"],
                benchmark=Benchmark.WEBSHOP,
            ),
            inventory_size=_positive_int(data["inventory_size"], field="WebShop inventory_size"),
            kind=_text(data["kind"], field="WebShop kind"),
        )


@dataclass(frozen=True, slots=True)
class SQLiteWebShopDeploymentConfig:
    runtime: OfficialProcessRuntimeConfig
    store_relative_path: str
    goals_relative_path: str
    search_index_relative_path: str
    asset_relative_files: tuple[str, ...]
    kind: str = "sqlite"

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, OfficialProcessRuntimeConfig):
            raise TypeError("WebShop SQLite runtime config is incompatible")
        paths = {
            "store_relative_path": self.store_relative_path,
            "goals_relative_path": self.goals_relative_path,
            "search_index_relative_path": self.search_index_relative_path,
        }
        normalized = {
            field: _relative_path(value, field=f"WebShop {field}") for field, value in paths.items()
        }
        if any(not value.startswith("webshop/") for value in normalized.values()):
            raise ValueError("WebShop deployment paths must remain below webshop/")
        assets = _asset_tuple(self.asset_relative_files, benchmark=Benchmark.WEBSHOP)
        for field in ("store_relative_path", "goals_relative_path"):
            if normalized[field] not in assets:
                raise ValueError(f"WebShop {field} must belong to its asset snapshot")
        if self.kind != "sqlite":
            raise ValueError("WebShop SQLite deployment kind is incompatible")
        for field, value in normalized.items():
            object.__setattr__(self, field, value)
        object.__setattr__(self, "asset_relative_files", assets)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "asset_relative_files": list(self.asset_relative_files),
            "goals_relative_path": self.goals_relative_path,
            "kind": self.kind,
            "runtime": self.runtime.to_value(),
            "search_index_relative_path": self.search_index_relative_path,
            "store_relative_path": self.store_relative_path,
        }

    @classmethod
    def from_value(cls, value: object) -> SQLiteWebShopDeploymentConfig:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "asset_relative_files",
                    "goals_relative_path",
                    "kind",
                    "runtime",
                    "search_index_relative_path",
                    "store_relative_path",
                }
            ),
            label="WebShop SQLite process deployment",
        )
        return cls(
            runtime=OfficialProcessRuntimeConfig.from_value(data["runtime"]),
            store_relative_path=_relative_path(
                data["store_relative_path"], field="WebShop store_relative_path"
            ),
            goals_relative_path=_relative_path(
                data["goals_relative_path"], field="WebShop goals_relative_path"
            ),
            search_index_relative_path=_relative_path(
                data["search_index_relative_path"],
                field="WebShop search_index_relative_path",
            ),
            asset_relative_files=_relative_files(
                data["asset_relative_files"], benchmark=Benchmark.WEBSHOP
            ),
            kind=_text(data["kind"], field="WebShop kind"),
        )


WebShopProcessDeploymentConfig: TypeAlias = (
    JsonArrayWebShopDeploymentConfig | SQLiteWebShopDeploymentConfig
)


def _webshop_config_from_value(value: object) -> WebShopProcessDeploymentConfig:
    if type(value) is not dict:
        raise TypeError("WebShop deployment must be an object")
    kind = value.get("kind")
    if kind == "json-array":
        return JsonArrayWebShopDeploymentConfig.from_value(value)
    if kind == "sqlite":
        return SQLiteWebShopDeploymentConfig.from_value(value)
    raise ValueError("unsupported WebShop deployment kind")


@dataclass(frozen=True, slots=True)
class ALFWorldGameDeploymentConfig:
    game_id: str
    data_directory_relative_path: str
    train_eval: str
    instruction_text: str

    def __post_init__(self) -> None:
        game_id = _text(self.game_id, field="ALFWorld game_id")
        directory = _relative_path(
            self.data_directory_relative_path,
            field="ALFWorld data_directory_relative_path",
        )
        if not directory.startswith("alfworld/"):
            raise ValueError("ALFWorld game directory must remain below alfworld/")
        if self.train_eval not in {
            "train",
            "eval_in_distribution",
            "eval_out_of_distribution",
        }:
            raise ValueError("ALFWorld train_eval is unsupported")
        instruction = _text(
            self.instruction_text,
            field="ALFWorld instruction_text",
        )
        object.__setattr__(self, "game_id", game_id)
        object.__setattr__(self, "data_directory_relative_path", directory)
        object.__setattr__(self, "instruction_text", instruction)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "data_directory_relative_path": self.data_directory_relative_path,
            "game_id": self.game_id,
            "instruction_text": self.instruction_text,
            "train_eval": self.train_eval,
        }

    @classmethod
    def from_value(cls, value: object) -> ALFWorldGameDeploymentConfig:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "data_directory_relative_path",
                    "game_id",
                    "instruction_text",
                    "train_eval",
                }
            ),
            label="ALFWorld game deployment",
        )
        return cls(
            game_id=_text(data["game_id"], field="ALFWorld game_id"),
            data_directory_relative_path=_relative_path(
                data["data_directory_relative_path"],
                field="ALFWorld data_directory_relative_path",
            ),
            train_eval=_text(data["train_eval"], field="ALFWorld train_eval"),
            instruction_text=_text(
                data["instruction_text"],
                field="ALFWorld instruction_text",
            ),
        )


@dataclass(frozen=True, slots=True)
class ALFWorldProcessDeploymentConfig:
    runtime: OfficialProcessRuntimeConfig
    config_relative_path: str
    games: tuple[ALFWorldGameDeploymentConfig, ...]
    asset_relative_files: tuple[str, ...]
    max_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, OfficialProcessRuntimeConfig):
            raise TypeError("ALFWorld runtime config is incompatible")
        config = _relative_path(
            self.config_relative_path,
            field="ALFWorld config_relative_path",
        )
        if not config.startswith("alfworld/"):
            raise ValueError("ALFWorld config must remain below alfworld/")
        assets = _asset_tuple(
            self.asset_relative_files,
            benchmark=Benchmark.ALFWORLD,
        )
        if config not in assets:
            raise ValueError("ALFWorld config must belong to its asset snapshot")
        if not self.games or any(
            not isinstance(game, ALFWorldGameDeploymentConfig) for game in self.games
        ):
            raise ValueError("ALFWorld deployment requires explicit game records")
        game_ids = tuple(game.game_id for game in self.games)
        if game_ids != tuple(sorted(game_ids)) or len(set(game_ids)) != len(game_ids):
            raise ValueError("ALFWorld games must be sorted by unique game_id")
        object.__setattr__(
            self,
            "max_steps",
            _positive_int(self.max_steps, field="ALFWorld max_steps"),
        )
        object.__setattr__(self, "config_relative_path", config)
        object.__setattr__(self, "asset_relative_files", assets)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "asset_relative_files": list(self.asset_relative_files),
            "config_relative_path": self.config_relative_path,
            "games": [game.to_value() for game in self.games],
            "max_steps": self.max_steps,
            "runtime": self.runtime.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> ALFWorldProcessDeploymentConfig:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "asset_relative_files",
                    "config_relative_path",
                    "games",
                    "max_steps",
                    "runtime",
                }
            ),
            label="ALFWorld process deployment",
        )
        games = data["games"]
        if not isinstance(games, list):
            raise TypeError("ALFWorld games must be an array")
        return cls(
            runtime=OfficialProcessRuntimeConfig.from_value(data["runtime"]),
            config_relative_path=_relative_path(
                data["config_relative_path"],
                field="ALFWorld config_relative_path",
            ),
            games=tuple(ALFWorldGameDeploymentConfig.from_value(game) for game in games),
            asset_relative_files=_relative_files(
                data["asset_relative_files"],
                benchmark=Benchmark.ALFWORLD,
            ),
            max_steps=_positive_int(data["max_steps"], field="ALFWorld max_steps"),
        )


@dataclass(frozen=True, slots=True)
class ScienceWorldProcessDeploymentConfig:
    runtime: OfficialProcessRuntimeConfig
    jar_relative_path: str
    asset_relative_files: tuple[str, ...]
    simplification: str
    max_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, OfficialProcessRuntimeConfig):
            raise TypeError("ScienceWorld runtime config is incompatible")
        jar = _relative_path(
            self.jar_relative_path,
            field="ScienceWorld jar_relative_path",
        )
        if not jar.startswith("scienceworld/"):
            raise ValueError("ScienceWorld jar must remain below scienceworld/")
        assets = _asset_tuple(
            self.asset_relative_files,
            benchmark=Benchmark.SCIENCE_WORLD,
        )
        if jar not in assets:
            raise ValueError("ScienceWorld jar must belong to its asset snapshot")
        simplification = _text(
            self.simplification,
            field="ScienceWorld simplification",
            allow_empty=True,
        )
        object.__setattr__(
            self,
            "max_steps",
            _positive_int(self.max_steps, field="ScienceWorld max_steps"),
        )
        object.__setattr__(self, "jar_relative_path", jar)
        object.__setattr__(self, "simplification", simplification)
        object.__setattr__(self, "asset_relative_files", assets)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "asset_relative_files": list(self.asset_relative_files),
            "jar_relative_path": self.jar_relative_path,
            "max_steps": self.max_steps,
            "runtime": self.runtime.to_value(),
            "simplification": self.simplification,
        }

    @classmethod
    def from_value(cls, value: object) -> ScienceWorldProcessDeploymentConfig:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "asset_relative_files",
                    "jar_relative_path",
                    "max_steps",
                    "runtime",
                    "simplification",
                }
            ),
            label="ScienceWorld process deployment",
        )
        return cls(
            runtime=OfficialProcessRuntimeConfig.from_value(data["runtime"]),
            jar_relative_path=_relative_path(
                data["jar_relative_path"],
                field="ScienceWorld jar_relative_path",
            ),
            asset_relative_files=_relative_files(
                data["asset_relative_files"],
                benchmark=Benchmark.SCIENCE_WORLD,
            ),
            simplification=_text(
                data["simplification"],
                field="ScienceWorld simplification",
                allow_empty=True,
            ),
            max_steps=_positive_int(data["max_steps"], field="ScienceWorld max_steps"),
        )


@dataclass(frozen=True, slots=True)
class ExternalProcessManifestDeploymentConfig:
    """One answer-free official task inventory prepared outside this package."""

    benchmark: Benchmark
    split: str
    task_manifest_relative_path: str
    asset_relative_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.benchmark not in _EXTERNAL_PROCESS_BENCHMARKS:
            raise ValueError("external manifest benchmark is not part of protocol v9")
        split = _text(self.split, field=f"{self.benchmark.value} split")
        manifest = _relative_path(
            self.task_manifest_relative_path,
            field=f"{self.benchmark.value} task_manifest_relative_path",
        )
        prefix = f"{self.benchmark.value}/"
        if not manifest.startswith(prefix):
            raise ValueError("external task manifest must remain below its benchmark root")
        assets = _asset_tuple(self.asset_relative_files, benchmark=self.benchmark)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "task_manifest_relative_path", manifest)
        object.__setattr__(self, "asset_relative_files", assets)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "asset_relative_files": list(self.asset_relative_files),
            "benchmark": self.benchmark.value,
            "split": self.split,
            "task_manifest_relative_path": self.task_manifest_relative_path,
        }

    @classmethod
    def from_value(cls, value: object) -> ExternalProcessManifestDeploymentConfig:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "asset_relative_files",
                    "benchmark",
                    "split",
                    "task_manifest_relative_path",
                }
            ),
            label="external process manifest deployment",
        )
        benchmark = Benchmark(_text(data["benchmark"], field="external benchmark"))
        return cls(
            benchmark=benchmark,
            split=_text(data["split"], field=f"{benchmark.value} split"),
            task_manifest_relative_path=_relative_path(
                data["task_manifest_relative_path"],
                field=f"{benchmark.value} task_manifest_relative_path",
            ),
            asset_relative_files=_relative_files(data["asset_relative_files"], benchmark=benchmark),
        )


@dataclass(frozen=True, slots=True)
class OfficialProcessDeploymentConfig:
    """Complete host-private deployment configuration for process preparation."""

    webshop: WebShopProcessDeploymentConfig
    alfworld: ALFWorldProcessDeploymentConfig
    scienceworld: ScienceWorldProcessDeploymentConfig
    external_manifests: tuple[ExternalProcessManifestDeploymentConfig, ...]
    seed: int = FIXED_SEED

    def __post_init__(self) -> None:
        if not isinstance(
            self.webshop,
            JsonArrayWebShopDeploymentConfig | SQLiteWebShopDeploymentConfig,
        ):
            raise TypeError("official process deployment requires WebShop config")
        if not isinstance(self.alfworld, ALFWorldProcessDeploymentConfig):
            raise TypeError("official process deployment requires ALFWorld config")
        if not isinstance(self.scienceworld, ScienceWorldProcessDeploymentConfig):
            raise TypeError("official process deployment requires ScienceWorld config")
        external = tuple(item.benchmark for item in self.external_manifests)
        if external != _EXTERNAL_PROCESS_BENCHMARKS:
            raise ValueError("external process manifests must follow protocol v9 order")
        if type(self.seed) is not int or self.seed != FIXED_SEED:
            raise ValueError("official process deployment must use the preregistered seed")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "alfworld": self.alfworld.to_value(),
            "external_manifests": [item.to_value() for item in self.external_manifests],
            "format": OFFICIAL_PROCESS_DEPLOYMENT_FORMAT,
            "scienceworld": self.scienceworld.to_value(),
            "seed": self.seed,
            "webshop": self.webshop.to_value(),
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @classmethod
    def from_value(cls, value: object) -> OfficialProcessDeploymentConfig:
        data = _wire_object(
            value,
            fields=frozenset(
                {
                    "alfworld",
                    "external_manifests",
                    "format",
                    "scienceworld",
                    "seed",
                    "webshop",
                }
            ),
            label="official process deployment config",
        )
        if data["format"] != OFFICIAL_PROCESS_DEPLOYMENT_FORMAT:
            raise ValueError("unsupported official process deployment format")
        seed = data["seed"]
        if type(seed) is not int:
            raise TypeError("official process deployment seed must be an integer")
        external = data["external_manifests"]
        if not isinstance(external, list):
            raise TypeError("external process manifests must be an array")
        return cls(
            webshop=_webshop_config_from_value(data["webshop"]),
            alfworld=ALFWorldProcessDeploymentConfig.from_value(data["alfworld"]),
            scienceworld=ScienceWorldProcessDeploymentConfig.from_value(data["scienceworld"]),
            external_manifests=tuple(
                ExternalProcessManifestDeploymentConfig.from_value(item) for item in external
            ),
            seed=seed,
        )


def publish_official_process_deployment_config(
    config: OfficialProcessDeploymentConfig,
    output_path: Path,
) -> None:
    """Publish one immutable canonical private deployment configuration."""

    if not isinstance(config, OfficialProcessDeploymentConfig):
        raise TypeError("config must be OfficialProcessDeploymentConfig")
    if not isinstance(output_path, Path) or not output_path.is_absolute():
        raise ValueError("output_path must be an absolute Path")
    if not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    value = {**config.to_value(), "content_hash": config.content_hash}
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_official_process_deployment_config(
    path: Path,
) -> OfficialProcessDeploymentConfig:
    """Load and verify one immutable canonical private deployment config."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("deployment config path must be an absolute Path")
    value = _wire_object(
        _read_published_canonical_record(
            path,
            label="official process deployment config",
        ),
        fields=frozenset(
            {
                "alfworld",
                "content_hash",
                "external_manifests",
                "format",
                "scienceworld",
                "seed",
                "webshop",
            }
        ),
        label="published official process deployment config",
    )
    content_hash = value.pop("content_hash")
    if type(content_hash) is not str:
        raise TypeError("official process deployment content_hash must be text")
    config = OfficialProcessDeploymentConfig.from_value(value)
    if content_hash != config.content_hash:
        raise ValueError("official process deployment content hash differs")
    return config


def _locked_revision(lock: BenchmarkAcquisitionLock, benchmark: Benchmark) -> str:
    for entry in lock.benchmarks:
        if entry.benchmark is benchmark:
            revision = entry.source.revision
            if len(revision) != _REVISION_LENGTH:
                raise ValueError("official source revision is not a full Git commit")
            return revision
    raise ValueError(f"acquisition lock is missing {benchmark.value}")


def _runtime(
    config: OfficialProcessRuntimeConfig,
    *,
    lock: BenchmarkAcquisitionLock,
    target_root: Path,
    benchmark: Benchmark,
) -> PinnedOfficialProcess:
    source_root = target_root / benchmark.value / "repository"
    return PinnedOfficialProcess(
        interpreter_path=config.interpreter_path,
        source_root=source_root,
        source_revision=_locked_revision(lock, benchmark),
        request_timeout_seconds=config.request_timeout_seconds,
    )


def _verify_assets(
    target_root: Path,
    *,
    benchmark: Benchmark,
    relative_files: tuple[str, ...],
) -> frozenset[str]:
    assets = frozenset(relative_files)
    if len(assets) != len(relative_files):
        raise ValueError(f"{benchmark.value} assets contain duplicates")
    for relative in relative_files:
        _existing_target_path(
            target_root,
            relative,
            directory=False,
            label=f"{benchmark.value} deployment asset",
        )
    return assets


def _require_directory_files_are_assets(
    target_root: Path,
    *,
    directory: Path,
    assets: frozenset[str],
    label: str,
) -> None:
    files = tuple(path for path in sorted(directory.rglob("*")) if path.is_file())
    if not files:
        raise ValueError(f"{label} cannot be empty")
    for path in files:
        relative = _relative_to_target(target_root, path, label=label)
        if relative not in assets:
            raise ValueError(f"{label} file is absent from asset_relative_files")


def _external_manifest_input(
    config: ExternalProcessManifestDeploymentConfig,
    *,
    lock: BenchmarkAcquisitionLock,
    target_root: Path,
) -> ProcessTaskManifestInput:
    """Load one public-only canonical RolloutTask JSONL inventory."""

    _verify_assets(
        target_root,
        benchmark=config.benchmark,
        relative_files=config.asset_relative_files,
    )
    path = _existing_target_path(
        target_root,
        config.task_manifest_relative_path,
        directory=False,
        label=f"{config.benchmark.value} task manifest",
    )
    records: list[ExternalProcessTaskRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            raise ValueError("external process task manifest contains an empty record")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("external process task manifest is invalid JSONL") from error
        task = RolloutTask.from_value(raw)
        if (
            not isinstance(task.public_context, dict)
            or task.public_context.get("benchmark_id") != config.benchmark.value
        ):
            raise ValueError("external process task has an incompatible benchmark identity")
        records.append(
            ExternalProcessTaskRecord(
                benchmark=config.benchmark,
                source_split=config.split,
                task=task,
            )
        )
    if not records:
        raise ValueError("external process task manifest cannot be empty")
    records.sort(key=lambda item: item.task.task_id)
    if len({item.task.task_id for item in records}) != len(records):
        raise ValueError("external process task manifest contains duplicate task IDs")
    revision = _locked_revision(lock, config.benchmark)
    return ProcessTaskManifestInput(
        benchmark=config.benchmark,
        dataset_revision=revision,
        split=config.split,
        task_manifest_relative_path=(f"_derived/process/{config.benchmark.value}.jsonl"),
        asset_relative_files=config.asset_relative_files,
        environment_source_revision=revision,
        records=tuple(records),
    )


@dataclass(frozen=True, slots=True)
class FileOfficialProcessPreparationFactory:
    """Concrete CLI factory bound to one immutable private JSON configuration."""

    config_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.config_path, Path) or not self.config_path.is_absolute():
            raise ValueError("config_path must be an absolute Path")
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)

    def build(
        self,
        *,
        lock: BenchmarkAcquisitionLock,
        target_root: Path,
    ) -> OfficialProcessPreparationDeployment:
        if not isinstance(lock, BenchmarkAcquisitionLock):
            raise TypeError("lock must be BenchmarkAcquisitionLock")
        if not isinstance(target_root, Path) or not target_root.is_absolute():
            raise ValueError("target_root must be an absolute Path")
        if not target_root.is_dir():
            raise NotADirectoryError(target_root)

        config = load_official_process_deployment_config(self.config_path)
        webshop_assets = _verify_assets(
            target_root,
            benchmark=Benchmark.WEBSHOP,
            relative_files=config.webshop.asset_relative_files,
        )
        alfworld_assets = _verify_assets(
            target_root,
            benchmark=Benchmark.ALFWORLD,
            relative_files=config.alfworld.asset_relative_files,
        )
        scienceworld_assets = _verify_assets(
            target_root,
            benchmark=Benchmark.SCIENCE_WORLD,
            relative_files=config.scienceworld.asset_relative_files,
        )

        webshop_runtime = _runtime(
            config.webshop.runtime,
            lock=lock,
            target_root=target_root,
            benchmark=Benchmark.WEBSHOP,
        )
        search_index_path = _existing_target_path(
            target_root,
            config.webshop.search_index_relative_path,
            directory=True,
            label="WebShop search index",
        )
        _require_directory_files_are_assets(
            target_root,
            directory=search_index_path,
            assets=webshop_assets,
            label="WebShop search index",
        )
        deployment: WebShopDeployment
        match config.webshop:
            case JsonArrayWebShopDeploymentConfig() as webshop_config:
                products_path = _existing_target_path(
                    target_root,
                    webshop_config.products_relative_path,
                    directory=False,
                    label="WebShop products",
                )
                deployment = JsonArrayWebShopDeployment(
                    runtime=webshop_runtime,
                    products_path=products_path,
                    index_path=search_index_path,
                    inventory_size=webshop_config.inventory_size,
                    seed=config.seed,
                )
            case SQLiteWebShopDeploymentConfig() as webshop_config:
                store_path = _existing_target_path(
                    target_root,
                    webshop_config.store_relative_path,
                    directory=False,
                    label="WebShop product store",
                )
                goals_path = _existing_target_path(
                    target_root,
                    webshop_config.goals_relative_path,
                    directory=False,
                    label="WebShop goal stream",
                )
                deployment = SQLiteWebShopDeployment(
                    runtime=webshop_runtime,
                    store_path=store_path,
                    goals_path=goals_path,
                    index_path=search_index_path,
                    seed=config.seed,
                )
            case _ as unmatched:
                from typing import assert_never

                assert_never(unmatched)
        webshop = OfficialWebShopProcessFactory(deployment=deployment)

        alfworld_runtime = _runtime(
            config.alfworld.runtime,
            lock=lock,
            target_root=target_root,
            benchmark=Benchmark.ALFWORLD,
        )
        alfworld_config_path = _existing_target_path(
            target_root,
            config.alfworld.config_relative_path,
            directory=False,
            label="ALFWorld config",
        )
        games: dict[str, ALFWorldGameDeployment] = {}
        for game in config.alfworld.games:
            directory = _existing_target_path(
                target_root,
                game.data_directory_relative_path,
                directory=True,
                label=f"ALFWorld game {game.game_id}",
            )
            _require_directory_files_are_assets(
                target_root,
                directory=directory,
                assets=alfworld_assets,
                label=f"ALFWorld game {game.game_id}",
            )
            games[game.game_id] = ALFWorldGameDeployment(
                data_directory=directory,
                train_eval=game.train_eval,
                instruction_text=game.instruction_text,
            )
        alfworld = OfficialALFWorldProcessFactory(
            runtime=alfworld_runtime,
            config_path=alfworld_config_path,
            games=games,
            seed=config.seed,
        )

        scienceworld_runtime = _runtime(
            config.scienceworld.runtime,
            lock=lock,
            target_root=target_root,
            benchmark=Benchmark.SCIENCE_WORLD,
        )
        scienceworld_jar = _existing_target_path(
            target_root,
            config.scienceworld.jar_relative_path,
            directory=False,
            label="ScienceWorld jar",
        )
        if config.scienceworld.jar_relative_path not in scienceworld_assets:
            raise ValueError("ScienceWorld jar is absent from deployment assets")
        scienceworld = OfficialScienceWorldProcessFactory(
            runtime=scienceworld_runtime,
            jar_path=scienceworld_jar,
            simplification=config.scienceworld.simplification,
            seed=config.seed,
        )

        external_inputs = tuple(
            _external_manifest_input(
                item,
                lock=lock,
                target_root=target_root,
            )
            for item in config.external_manifests
        )

        return OfficialProcessPreparationDeployment(
            dependencies=ProductionProcessCatalogDependencies(
                webshop=webshop,
                alfworld=alfworld,
                scienceworld=scienceworld,
            ),
            webshop_asset_relative_files=config.webshop.asset_relative_files,
            alfworld_asset_relative_files=config.alfworld.asset_relative_files,
            scienceworld_asset_relative_files=config.scienceworld.asset_relative_files,
            alfworld_max_steps=config.alfworld.max_steps,
            scienceworld_max_steps=config.scienceworld.max_steps,
            external_manifest_inputs=external_inputs,
        )


__all__ = [
    "OFFICIAL_PROCESS_DEPLOYMENT_FORMAT",
    "ALFWorldGameDeploymentConfig",
    "ALFWorldProcessDeploymentConfig",
    "ExternalProcessManifestDeploymentConfig",
    "FileOfficialProcessPreparationFactory",
    "JsonArrayWebShopDeploymentConfig",
    "OfficialProcessDeploymentConfig",
    "OfficialProcessRuntimeConfig",
    "SQLiteWebShopDeploymentConfig",
    "ScienceWorldProcessDeploymentConfig",
    "WebShopProcessDeploymentConfig",
    "load_official_process_deployment_config",
    "publish_official_process_deployment_config",
]
