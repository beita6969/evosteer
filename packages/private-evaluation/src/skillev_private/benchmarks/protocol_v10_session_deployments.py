"""Private production deployments for all nine Protocol 10 session routes.

The committed code describes the closed deployment shape, while the JSON
instance stays on the server and owns every interpreter, repository, dataset,
and scratch path.  This keeps Protocol 10 independent from the historical
Protocol 9 catalog and its acquisition/deployment objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from skillev.experiments import Benchmark
from skillev.experiments.protocol_v10 import BenchmarkV10
from skillev.runtime import SGLangGateway
from skillev.training import RolloutWorkflowResources

from .external_process_runtime import (
    ExternalProcessRuntime,
    ExternalProcessSessionFactory,
)
from .healthbench_qwen_sglang import (
    HealthBenchQwenGraderConfig,
    HealthBenchQwenVerifierIdentity,
    QwenSGLangHealthBenchDeployment,
)
from .official_process import (
    ALFWorldGameDeployment,
    OfficialALFWorldProcessFactory,
    OfficialWebShopProcessFactory,
    PinnedOfficialProcess,
    SQLiteWebShopDeployment,
)
from .protocol_v10_materialization import LoadedProtocolV10Catalog
from .protocol_v10_population import ProtocolV10PopulationSessionRegistry
from .protocol_v10_sessions import (
    ProtocolV10ALFWorldSessionBuilder,
    ProtocolV10AppWorldSessionBuilder,
    ProtocolV10CodeSessionBuilder,
    ProtocolV10HealthSessionBuilder,
    ProtocolV10PopulationSessionBuilder,
    ProtocolV10StaticSessionBuilder,
    ProtocolV10WebShopSessionBuilder,
    build_protocol_v10_population_sessions,
    load_protocol_v10_private_record_file,
)
from .protocol_v10_spreadsheet import ProtocolV10SpreadsheetSessionBuilder
from .protocol_v10_spreadsheet_runtime import (
    BubblewrapSpreadsheetExecutor,
    IsolatedSpreadsheetWorkspaceFactory,
    ProtocolV10SpreadsheetAssetResolver,
)
from .protocol_v10_workers import (
    SpreadsheetBenchProcessDeployment,
)

PROTOCOL_V10_SESSION_DEPLOYMENTS_FORMAT: Final = (
    "skillev-private-protocol-v10-session-deployments@2"
)


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _fields(value: object, *, expected: set[str], label: str) -> dict[str, object]:
    result = _mapping(value, label=label)
    if set(result) != expected:
        raise ValueError(f"{label} has incompatible fields")
    return result


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _path(value: object, *, label: str, directory: bool) -> Path:
    result = Path(_text(value, label=label))
    if not result.is_absolute():
        raise ValueError(f"{label} must be absolute")
    exists = result.is_dir() if directory else result.is_file()
    if not exists:
        raise ValueError(f"{label} does not identify an existing deployment asset")
    return result


def _positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _path_array(value: object, *, label: str, directory: bool = True) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    return tuple(_path(item, label=f"{label} item", directory=directory) for item in value)


@dataclass(frozen=True, slots=True)
class ProtocolV10WebShopDeployment:
    interpreter: Path
    source_root: Path
    source_revision: str
    product_store: Path
    goals: Path
    search_index: Path
    timeout_seconds: float

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10WebShopDeployment:
        data = _fields(
            value,
            expected={
                "goals",
                "interpreter",
                "product_store",
                "search_index",
                "source_revision",
                "source_root",
                "timeout_seconds",
            },
            label="Protocol 10 WebShop deployment",
        )
        return cls(
            interpreter=_path(data["interpreter"], label="WebShop interpreter", directory=False),
            source_root=_path(data["source_root"], label="WebShop source", directory=True),
            source_revision=_text(data["source_revision"], label="WebShop revision"),
            product_store=_path(
                data["product_store"], label="WebShop product store", directory=False
            ),
            goals=_path(data["goals"], label="WebShop goals", directory=False),
            search_index=_path(data["search_index"], label="WebShop search index", directory=True),
            timeout_seconds=_positive(data["timeout_seconds"], label="WebShop timeout"),
        )

    def build(self) -> OfficialWebShopProcessFactory:
        runtime = PinnedOfficialProcess(
            self.interpreter,
            self.source_root,
            self.source_revision,
            self.timeout_seconds,
        )
        return OfficialWebShopProcessFactory(
            SQLiteWebShopDeployment(
                runtime,
                self.product_store,
                self.goals,
                self.search_index,
                0,
            )
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10ALFWorldDeployment:
    interpreter: Path
    source_root: Path
    source_revision: str
    config_path: Path
    dataset_root: Path
    timeout_seconds: float

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10ALFWorldDeployment:
        data = _fields(
            value,
            expected={
                "config_path",
                "dataset_root",
                "interpreter",
                "source_revision",
                "source_root",
                "timeout_seconds",
            },
            label="Protocol 10 ALFWorld deployment",
        )
        return cls(
            interpreter=_path(data["interpreter"], label="ALFWorld interpreter", directory=False),
            source_root=_path(data["source_root"], label="ALFWorld source", directory=True),
            source_revision=_text(data["source_revision"], label="ALFWorld revision"),
            config_path=_path(data["config_path"], label="ALFWorld config", directory=False),
            dataset_root=_path(data["dataset_root"], label="ALFWorld dataset", directory=True),
            timeout_seconds=_positive(data["timeout_seconds"], label="ALFWorld timeout"),
        )

    def build(self, loaded: LoadedProtocolV10Catalog) -> OfficialALFWorldProcessFactory:
        games: dict[str, ALFWorldGameDeployment] = {}
        for population in loaded.catalog.populations:
            if population.spec.benchmark is not BenchmarkV10.ALFWORLD:
                continue
            records = load_protocol_v10_private_record_file(
                loaded.private_record_files[population.spec.population_id]
            )
            by_source = {record.source_id: record.private_payload for record in records}
            train_eval = _alfworld_train_eval(population.spec.population_id)
            for item in population.items:
                payload = by_source[item.source_id]
                if not isinstance(payload, dict):
                    raise ValueError("ALFWorld private route must be an object")
                game_id = payload.get("game_id")
                relative = payload.get("trajectory_relative_path")
                if type(game_id) is not str or type(relative) is not str:
                    raise ValueError("ALFWorld private route is incomplete")
                directory = (self.dataset_root / "json_2.1.1" / relative).resolve()
                if (
                    not directory.is_relative_to(self.dataset_root.resolve())
                    or not directory.is_dir()
                ):
                    raise ValueError("ALFWorld game route escaped or is absent")
                deployment = ALFWorldGameDeployment(
                    directory,
                    train_eval,
                    item.task.query,
                )
                previous = games.setdefault(game_id, deployment)
                if previous != deployment:
                    raise ValueError("ALFWorld game identity has conflicting deployments")
        if not games:
            raise ValueError("Protocol 10 ALFWorld games are unavailable")
        return OfficialALFWorldProcessFactory(
            PinnedOfficialProcess(
                self.interpreter,
                self.source_root,
                self.source_revision,
                self.timeout_seconds,
            ),
            self.config_path,
            games,
            0,
        )


def _alfworld_train_eval(population_id: str) -> str:
    if "train" in population_id:
        return "train"
    if "unseen" in population_id:
        return "eval_out_of_distribution"
    if "seen" in population_id:
        return "eval_in_distribution"
    raise ValueError("ALFWorld population has no official train/evaluation route")


@dataclass(frozen=True, slots=True)
class ProtocolV10AppWorldDeployment:
    interpreter: Path
    source_root: Path
    state_root: Path
    timeout_seconds: float

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10AppWorldDeployment:
        data = _fields(
            value,
            expected={"interpreter", "source_root", "state_root", "timeout_seconds"},
            label="Protocol 10 AppWorld deployment",
        )
        return cls(
            interpreter=_path(data["interpreter"], label="AppWorld interpreter", directory=False),
            source_root=_path(data["source_root"], label="AppWorld source", directory=True),
            state_root=_path(data["state_root"], label="AppWorld state", directory=True),
            timeout_seconds=_positive(data["timeout_seconds"], label="AppWorld timeout"),
        )

    def build(self) -> ExternalProcessSessionFactory:
        return ExternalProcessSessionFactory(
            ExternalProcessRuntime(
                Benchmark.APPWORLD,
                self.interpreter,
                self.source_root,
                self.state_root,
                self.timeout_seconds,
            )
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10HealthDeployment:
    interpreter: Path
    source_root: Path
    model_revision: str
    tokenizer_revision: str
    sglang_version: str
    request_timeout_seconds: float
    worker_timeout_seconds: float

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10HealthDeployment:
        data = _fields(
            value,
            expected={
                "interpreter",
                "model_revision",
                "request_timeout_seconds",
                "sglang_version",
                "source_root",
                "tokenizer_revision",
                "worker_timeout_seconds",
            },
            label="Protocol 10 HealthBench deployment",
        )
        return cls(
            interpreter=_path(
                data["interpreter"], label="HealthBench interpreter", directory=False
            ),
            source_root=_path(data["source_root"], label="HealthBench source", directory=True),
            model_revision=_text(data["model_revision"], label="HealthBench model revision"),
            tokenizer_revision=_text(
                data["tokenizer_revision"], label="HealthBench tokenizer revision"
            ),
            sglang_version=_text(data["sglang_version"], label="HealthBench SGLang version"),
            request_timeout_seconds=_positive(
                data["request_timeout_seconds"], label="HealthBench request timeout"
            ),
            worker_timeout_seconds=_positive(
                data["worker_timeout_seconds"], label="HealthBench worker timeout"
            ),
        )

    def build(self, gateway: SGLangGateway) -> QwenSGLangHealthBenchDeployment:
        if gateway.config.base_model != self.model_revision:
            raise ValueError("HealthBench grader must use the SGLang base-model route")
        return QwenSGLangHealthBenchDeployment(
            interpreter_path=self.interpreter,
            official_source_root=self.source_root,
            config=HealthBenchQwenGraderConfig(
                endpoint_base=gateway.config.endpoint_base,
                base_model=gateway.config.base_model,
                identity=HealthBenchQwenVerifierIdentity(
                    model_revision=self.model_revision,
                    tokenizer_revision=self.tokenizer_revision,
                    sglang_version=self.sglang_version,
                ),
                request_timeout_seconds=self.request_timeout_seconds,
                worker_timeout_seconds=self.worker_timeout_seconds,
            ),
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10SpreadsheetDeployment:
    interpreter: Path
    source_root: Path
    training_archive: Path
    verified_root: Path
    libreoffice: Path
    workspace_root: Path
    temporary_root: Path
    extraction_cache_root: Path
    bubblewrap: Path
    prlimit: Path
    python_environment: Path
    python_package_roots: tuple[Path, ...]
    runtime_readonly_paths: tuple[Path, ...]
    edit_timeout_seconds: float
    oj_timeout_seconds: float

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10SpreadsheetDeployment:
        data = _fields(
            value,
            expected={
                "bubblewrap",
                "edit_timeout_seconds",
                "extraction_cache_root",
                "interpreter",
                "libreoffice",
                "oj_timeout_seconds",
                "prlimit",
                "python_environment",
                "python_package_roots",
                "runtime_readonly_paths",
                "source_root",
                "temporary_root",
                "training_archive",
                "verified_root",
                "workspace_root",
            },
            label="Protocol 10 SpreadsheetBench deployment",
        )
        return cls(
            interpreter=_path(
                data["interpreter"],
                label="SpreadsheetBench interpreter",
                directory=False,
            ),
            source_root=_path(data["source_root"], label="SpreadsheetBench source", directory=True),
            training_archive=_path(
                data["training_archive"],
                label="spreadsheet training archive",
                directory=False,
            ),
            verified_root=_path(
                data["verified_root"],
                label="SpreadsheetBench verified root",
                directory=True,
            ),
            libreoffice=_path(data["libreoffice"], label="LibreOffice", directory=False),
            workspace_root=_path(
                data["workspace_root"],
                label="spreadsheet workspace root",
                directory=True,
            ),
            temporary_root=_path(
                data["temporary_root"],
                label="spreadsheet temporary root",
                directory=True,
            ),
            extraction_cache_root=_path(
                data["extraction_cache_root"],
                label="spreadsheet extraction cache",
                directory=True,
            ),
            bubblewrap=_path(data["bubblewrap"], label="bubblewrap", directory=False),
            prlimit=_path(data["prlimit"], label="prlimit", directory=False),
            python_environment=_path(
                data["python_environment"],
                label="spreadsheet Python environment",
                directory=True,
            ),
            python_package_roots=_path_array(
                data["python_package_roots"], label="spreadsheet Python package roots"
            ),
            runtime_readonly_paths=_path_array(
                data["runtime_readonly_paths"], label="spreadsheet runtime paths"
            ),
            edit_timeout_seconds=_positive(
                data["edit_timeout_seconds"], label="spreadsheet edit timeout"
            ),
            oj_timeout_seconds=_positive(
                data["oj_timeout_seconds"], label="spreadsheet OJ timeout"
            ),
        )

    def build(self, resources: RolloutWorkflowResources) -> IsolatedSpreadsheetWorkspaceFactory:
        oj = SpreadsheetBenchProcessDeployment(
            interpreter_path=self.interpreter,
            official_source_root=self.source_root,
            training_archive=self.training_archive,
            verified_root=self.verified_root,
            libreoffice_path=self.libreoffice,
            workspace_root=self.workspace_root,
            temporary_root=self.temporary_root,
            timeout_seconds=self.oj_timeout_seconds,
        ).build(process_limiter=resources.process_graders)
        return IsolatedSpreadsheetWorkspaceFactory(
            workspace_root=self.workspace_root,
            assets=ProtocolV10SpreadsheetAssetResolver(
                self.training_archive,
                self.verified_root,
                self.extraction_cache_root,
            ),
            sandbox=BubblewrapSpreadsheetExecutor(
                bubblewrap_path=self.bubblewrap,
                prlimit_path=self.prlimit,
                python_environment=self.python_environment,
                python_package_roots=self.python_package_roots,
                runtime_readonly_paths=self.runtime_readonly_paths,
                timeout_seconds=self.edit_timeout_seconds,
            ),
            oj=oj,
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10SessionDeployments:
    webshop: ProtocolV10WebShopDeployment
    alfworld: ProtocolV10ALFWorldDeployment
    appworld: ProtocolV10AppWorldDeployment
    healthbench: ProtocolV10HealthDeployment
    spreadsheetbench: ProtocolV10SpreadsheetDeployment
    format: str = PROTOCOL_V10_SESSION_DEPLOYMENTS_FORMAT

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10SessionDeployments:
        data = _fields(
            value,
            expected={
                "alfworld",
                "appworld",
                "format",
                "healthbench",
                "spreadsheetbench",
                "webshop",
            },
            label="Protocol 10 session deployments",
        )
        if data["format"] != PROTOCOL_V10_SESSION_DEPLOYMENTS_FORMAT:
            raise ValueError("Protocol 10 session deployment format is unsupported")
        return cls(
            webshop=ProtocolV10WebShopDeployment.from_value(data["webshop"]),
            alfworld=ProtocolV10ALFWorldDeployment.from_value(data["alfworld"]),
            appworld=ProtocolV10AppWorldDeployment.from_value(data["appworld"]),
            healthbench=ProtocolV10HealthDeployment.from_value(data["healthbench"]),
            spreadsheetbench=ProtocolV10SpreadsheetDeployment.from_value(data["spreadsheetbench"]),
        )

    @classmethod
    def read(cls, path: Path) -> ProtocolV10SessionDeployments:
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Protocol 10 session deployment input must be an absolute file")
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def build_registry(
        self,
        loaded: LoadedProtocolV10Catalog,
        resources: RolloutWorkflowResources,
        gateway: SGLangGateway,
    ) -> ProtocolV10PopulationSessionRegistry:
        health = self.healthbench.build(gateway)
        builders: dict[BenchmarkV10, ProtocolV10PopulationSessionBuilder] = {
            BenchmarkV10.HOTPOT_QA: ProtocolV10StaticSessionBuilder(),
            BenchmarkV10.TRIVIA_QA: ProtocolV10StaticSessionBuilder(),
            BenchmarkV10.AIME_2026: ProtocolV10StaticSessionBuilder(),
            BenchmarkV10.HEALTHBENCH: ProtocolV10HealthSessionBuilder(
                lambda population, records: health.build(
                    population,
                    records,
                    model_request_limiter=resources.model_requests,
                    health_grader_limiter=resources.process_graders,
                )
            ),
            BenchmarkV10.WEBSHOP: ProtocolV10WebShopSessionBuilder(self.webshop.build()),
            BenchmarkV10.ALFWORLD: ProtocolV10ALFWorldSessionBuilder(self.alfworld.build(loaded)),
            BenchmarkV10.SPREADSHEETBENCH: ProtocolV10SpreadsheetSessionBuilder(
                self.spreadsheetbench.build(resources)
            ),
            BenchmarkV10.APPWORLD: ProtocolV10AppWorldSessionBuilder(self.appworld.build()),
            BenchmarkV10.MBPP_PLUS_FIXED_100: ProtocolV10CodeSessionBuilder(),
        }
        return build_protocol_v10_population_sessions(
            loaded.catalog,
            loaded.private_record_files,
            builders,
        )


__all__ = [
    "PROTOCOL_V10_SESSION_DEPLOYMENTS_FORMAT",
    "ProtocolV10ALFWorldDeployment",
    "ProtocolV10AppWorldDeployment",
    "ProtocolV10HealthDeployment",
    "ProtocolV10SessionDeployments",
    "ProtocolV10SpreadsheetDeployment",
    "ProtocolV10WebShopDeployment",
]
