"""Enumerate answer-free task manifests from the pinned official simulators.

WebShop and ScienceWorld expose their inventories only through their official
runtime packages, so they are queried through the same pinned interpreter and
source checkout used by production episodes.  ALFWorld's game inventory is
already explicit in :class:`OfficialALFWorldProcessFactory`; each game is reset
once to capture the exact public observation and admissible commands.

The resulting records contain no terminal reward, goal predicate, gold action,
or evaluator payload.  They can be passed directly to
``prepare_locked_process_benchmarks``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from skillev.contracts import JsonValue, stable_hash
from skillev.experiments import Benchmark

from .acquisition import BenchmarkAcquisitionLock
from .official_process import (
    OfficialALFWorldProcessFactory,
    OfficialScienceWorldProcessFactory,
    OfficialWebShopProcessFactory,
    OfficialWorkerClient,
    _initialization_payload,
    _object,
    _string_array,
    _text,
)
from .process_catalog import ProductionProcessCatalogDependencies
from .process_preparation import (
    ALFWorldProcessTaskRecord,
    ProcessTaskManifestInput,
    ScienceWorldProcessTaskRecord,
    WebShopProcessTaskRecord,
)

_MANIFEST_PAGE_SIZE = 128
_ALFWORLD_MANIFEST_BATCH_SIZE = 8


@dataclass(frozen=True, slots=True)
class OfficialProcessPreparationDeployment:
    """Explicit private deployment used to enumerate all process manifests."""

    dependencies: ProductionProcessCatalogDependencies
    webshop_asset_relative_files: tuple[str, ...]
    alfworld_asset_relative_files: tuple[str, ...]
    scienceworld_asset_relative_files: tuple[str, ...]
    alfworld_max_steps: int
    scienceworld_max_steps: int
    external_manifest_inputs: tuple[ProcessTaskManifestInput, ...] = field(default=())

    def manifest_inputs(self) -> tuple[ProcessTaskManifestInput, ...]:
        """Enumerate the pinned official inventories through the shared builder."""

        official = build_official_process_manifest_inputs(
            self.dependencies,
            webshop_asset_relative_files=self.webshop_asset_relative_files,
            alfworld_asset_relative_files=self.alfworld_asset_relative_files,
            scienceworld_asset_relative_files=self.scienceworld_asset_relative_files,
            alfworld_max_steps=self.alfworld_max_steps,
            scienceworld_max_steps=self.scienceworld_max_steps,
        )
        if not self.external_manifest_inputs:
            return official
        expected_external = (
            Benchmark.APPWORLD,
            Benchmark.BFCL_V3,
        )
        if tuple(item.benchmark for item in self.external_manifest_inputs) != expected_external:
            raise ValueError("external process manifest inputs must cover protocol v9 order")
        by_benchmark = {
            item.benchmark: item for item in (*official, *self.external_manifest_inputs)
        }
        order = (
            Benchmark.WEBSHOP,
            Benchmark.ALFWORLD,
            Benchmark.APPWORLD,
            Benchmark.SCIENCE_WORLD,
            Benchmark.BFCL_V3,
        )
        return tuple(by_benchmark[benchmark] for benchmark in order)


class OfficialProcessPreparationFactory(Protocol):
    """Private ``module:symbol`` boundary for incompatible official runtimes."""

    def build(
        self,
        *,
        lock: BenchmarkAcquisitionLock,
        target_root: Path,
    ) -> OfficialProcessPreparationDeployment: ...


def _digest(value: dict[str, JsonValue]) -> str:
    return stable_hash(value).removeprefix("sha256:")


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _initialize_manifest(
    client: OfficialWorkerClient,
    payload: Mapping[str, JsonValue],
) -> None:
    result = client.request("initialize_manifest", payload)
    ready = _object(
        result,
        fields={"ready"},
        label="official manifest worker initialization",
    )["ready"]
    if ready is not True:
        raise RuntimeError("official manifest worker is not ready")


def _manifest_inventory(client: OfficialWorkerClient) -> tuple[dict[str, JsonValue], ...]:
    count_result = _object(
        client.request("manifest_count", {}),
        fields={"count"},
        label="official manifest count",
    )
    count = _positive_int(count_result["count"], field="official manifest count")
    records: list[dict[str, JsonValue]] = []
    for offset in range(0, count, _MANIFEST_PAGE_SIZE):
        limit = min(_MANIFEST_PAGE_SIZE, count - offset)
        result = _object(
            client.request(
                "manifest_batch",
                {"limit": limit, "offset": offset},
            ),
            fields={"records"},
            label="official manifest batch",
        )
        raw_records = result["records"]
        if not isinstance(raw_records, list) or len(raw_records) != limit:
            raise ValueError("official manifest batch has an incompatible length")
        for value in raw_records:
            if not isinstance(value, dict):
                raise TypeError("official manifest record must be an object")
            records.append(value)
    if len(records) != count:
        raise ValueError("official manifest inventory is incomplete")
    return tuple(records)


def enumerate_official_webshop_tasks(
    factory: OfficialWebShopProcessFactory,
) -> tuple[WebShopProcessTaskRecord, ...]:
    """Read every shuffled official goal without executing or scoring it."""

    if not isinstance(factory, OfficialWebShopProcessFactory):
        raise TypeError("factory must be OfficialWebShopProcessFactory")
    runtime = factory.deployment.runtime
    client = OfficialWorkerClient(runtime)
    _initialize_manifest(
        client,
        _initialization_payload(
            benchmark="webshop",
            runtime=runtime,
            deployment=factory.deployment_value(),
            task={},
        ),
    )
    inventory = _manifest_inventory(client)
    client.close_successfully()

    records: list[WebShopProcessTaskRecord] = []
    for value in inventory:
        row = _object(
            value,
            fields={"category", "goal_index", "instruction_text"},
            label="official WebShop manifest record",
        )
        goal_index = _non_negative_int(
            row["goal_index"],
            field="WebShop goal index",
        )
        query = _text(row["instruction_text"], field_name="WebShop instruction")
        category = _text(row["category"], field_name="WebShop category")
        digest = _digest(
            {
                "benchmark": Benchmark.WEBSHOP.value,
                "goal_index": goal_index,
                "source_revision": runtime.source_revision,
            }
        )
        records.append(
            WebShopProcessTaskRecord(
                task_id=f"webshop:{digest}",
                task_family=f"webshop/{category}",
                query=query,
                public_context={"observation_format": "official-text"},
                goal_id=f"webshop-goal:{digest}",
                session_id=f"webshop-session:{digest}",
                goal_index=goal_index,
            )
        )
    return tuple(sorted(records, key=lambda record: record.task_id))


def enumerate_official_alfworld_tasks(
    factory: OfficialALFWorldProcessFactory,
    *,
    task_families: Mapping[str, str],
    max_steps: int,
    progress_observer: Callable[[int, int], None] | None = None,
) -> tuple[ALFWorldProcessTaskRecord, ...]:
    """Reset every explicit game once using caller-supplied official families."""

    if not isinstance(factory, OfficialALFWorldProcessFactory):
        raise TypeError("factory must be OfficialALFWorldProcessFactory")
    max_steps = _positive_int(max_steps, field="ALFWorld max_steps")
    if not isinstance(task_families, Mapping) or set(task_families) != set(factory.games):
        raise ValueError("ALFWorld task families must cover exactly the pinned games")

    records: list[ALFWorldProcessTaskRecord] = []
    client = OfficialWorkerClient(factory.runtime)
    deployment_games: list[JsonValue] = [
        {
            "data_directory": str(factory.games[game_id].data_directory),
            "game_id": game_id,
            "instruction_text": factory.games[game_id].instruction_text,
            "train_eval": factory.games[game_id].train_eval,
        }
        for game_id in sorted(factory.games)
    ]
    _initialize_manifest(
        client,
        _initialization_payload(
            benchmark="alfworld",
            runtime=factory.runtime,
            deployment={
                "config_path": str(factory.config_path),
                "games": deployment_games,
                "seed": factory.seed,
            },
            task={
                "batch_size": min(
                    _ALFWORLD_MANIFEST_BATCH_SIZE,
                    len(factory.games),
                ),
                "max_steps": max_steps,
            },
        ),
    )
    inventory = _manifest_inventory(client)
    expected_game_ids = tuple(sorted(factory.games))
    inventory_game_ids = tuple(
        _text(
            _object(
                value,
                fields={"game_id"},
                label="ALFWorld manifest inventory item",
            )["game_id"],
            field_name="ALFWorld manifest game_id",
        )
        for value in inventory
    )
    if inventory_game_ids != expected_game_ids:
        raise ValueError("ALFWorld worker inventory differs from its explicit deployment")
    batch_size = min(_ALFWORLD_MANIFEST_BATCH_SIZE, len(expected_game_ids))
    for offset in range(0, len(expected_game_ids), batch_size):
        batch_game_ids = expected_game_ids[offset : offset + batch_size]
        indices = list(range(offset, offset + len(batch_game_ids)))
        indices.extend([indices[-1]] * (batch_size - len(indices)))
        index_values: list[JsonValue] = list(indices)
        batch_result = _object(
            client.request(
                "manifest_materialize_batch",
                {"indices": index_values},
            ),
            fields={"records"},
            label="ALFWorld materialized batch",
        )
        raw_records = batch_result["records"]
        if not isinstance(raw_records, list) or len(raw_records) != batch_size:
            raise ValueError("ALFWorld materialized batch has an incompatible length")
        materialized = {}
        for value in raw_records:
            reset = _object(
                value,
                fields={
                    "admissible_commands",
                    "game_id",
                    "instruction_text",
                    "observation_text",
                },
                label="ALFWorld manifest reset",
            )
            game_id = _text(
                reset["game_id"],
                field_name="ALFWorld materialized game_id",
            )
            materialized[game_id] = reset
        if set(materialized) != set(batch_game_ids):
            raise ValueError("ALFWorld materialized another explicit game batch")
        for game_id in batch_game_ids:
            reset = materialized[game_id]
            task_family = _text(
                task_families[game_id],
                field_name="ALFWorld task family",
            )
            deployment = factory.games[game_id]
            instruction = _text(
                reset["instruction_text"],
                field_name="ALFWorld instruction",
            )
            if instruction != deployment.instruction_text:
                raise ValueError("ALFWorld reset instruction differs from its deployment")
            observation = _text(
                reset["observation_text"],
                field_name="ALFWorld initial observation",
            )
            commands = _string_array(
                reset["admissible_commands"],
                label="ALFWorld admissible commands",
            )
            digest = _digest(
                {
                    "benchmark": Benchmark.ALFWORLD.value,
                    "game_id": game_id,
                    "source_revision": factory.runtime.source_revision,
                }
            )
            records.append(
                ALFWorldProcessTaskRecord(
                    task_id=f"alfworld:{digest}",
                    task_family=task_family,
                    query=deployment.instruction_text,
                    game_id=game_id,
                    initial_observation=observation,
                    admissible_commands=commands,
                    max_steps=max_steps,
                )
            )
        if progress_observer is not None:
            progress_observer(
                min(offset + batch_size, len(expected_game_ids)),
                len(expected_game_ids),
            )
    client.close_successfully()
    return tuple(sorted(records, key=lambda record: record.task_id))


def load_official_alfworld_task_families(
    factory: OfficialALFWorldProcessFactory,
) -> dict[str, str]:
    """Read each pinned game's public task category from ``traj_data.json``."""

    if not isinstance(factory, OfficialALFWorldProcessFactory):
        raise TypeError("factory must be OfficialALFWorldProcessFactory")
    families: dict[str, str] = {}
    for game_id in sorted(factory.games):
        deployment = factory.games[game_id]
        candidates = tuple(sorted(deployment.data_directory.glob("**/traj_data.json")))
        if len(candidates) != 1:
            raise ValueError(f"ALFWorld game {game_id!r} must contain exactly one traj_data.json")
        value = json.loads(candidates[0].read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"ALFWorld game {game_id!r} traj_data.json must be an object")
        task_type = _text(
            value.get("task_type"),
            field_name=f"ALFWorld game {game_id!r} task_type",
        )
        families[game_id] = f"alfworld/{task_type}"
    return families


def enumerate_official_scienceworld_tasks(
    factory: OfficialScienceWorldProcessFactory,
    *,
    max_steps: int,
) -> tuple[ScienceWorldProcessTaskRecord, ...]:
    """Enumerate and reset each official test variation in one JVM process."""

    if not isinstance(factory, OfficialScienceWorldProcessFactory):
        raise TypeError("factory must be OfficialScienceWorldProcessFactory")
    max_steps = _positive_int(max_steps, field="ScienceWorld max_steps")
    client = OfficialWorkerClient(factory.runtime)
    _initialize_manifest(
        client,
        _initialization_payload(
            benchmark="scienceworld",
            runtime=factory.runtime,
            deployment={
                "jar_path": str(factory.jar_path),
                "seed": factory.seed,
                "simplification": factory.simplification,
            },
            task={"max_steps": max_steps, "split": "test"},
        ),
    )
    inventory = _manifest_inventory(client)
    records: list[ScienceWorldProcessTaskRecord] = []
    for index, value in enumerate(inventory):
        row = _object(
            value,
            fields={"task_name", "variation_index"},
            label="official ScienceWorld manifest record",
        )
        task_name = _text(
            row["task_name"],
            field_name="ScienceWorld task name",
        )
        variation_index = _non_negative_int(
            row["variation_index"],
            field="ScienceWorld variation index",
        )
        materialized = _object(
            client.request("manifest_materialize", {"index": index}),
            fields={"initial_observation", "query"},
            label="official ScienceWorld materialized task",
        )
        digest = _digest(
            {
                "benchmark": Benchmark.SCIENCE_WORLD.value,
                "source_revision": factory.runtime.source_revision,
                "task_name": task_name,
                "variation_index": variation_index,
            }
        )
        records.append(
            ScienceWorldProcessTaskRecord(
                task_id=f"scienceworld:{digest}",
                task_family=f"scienceworld/{task_name}",
                query=_text(
                    materialized["query"],
                    field_name="ScienceWorld query",
                ),
                task_name=task_name,
                variation_index=variation_index,
                initial_observation=_text(
                    materialized["initial_observation"],
                    field_name="ScienceWorld initial observation",
                ),
                max_steps=max_steps,
            )
        )
    client.close_successfully()
    return tuple(sorted(records, key=lambda record: record.task_id))


def build_official_process_manifest_inputs(
    dependencies: ProductionProcessCatalogDependencies,
    *,
    webshop_asset_relative_files: tuple[str, ...],
    alfworld_asset_relative_files: tuple[str, ...],
    scienceworld_asset_relative_files: tuple[str, ...],
    alfworld_max_steps: int,
    scienceworld_max_steps: int,
) -> tuple[ProcessTaskManifestInput, ...]:
    """Build the complete three-source input tuple in production catalog order."""

    if not isinstance(dependencies, ProductionProcessCatalogDependencies):
        raise TypeError("dependencies must be ProductionProcessCatalogDependencies")
    webshop = dependencies.webshop
    alfworld = dependencies.alfworld
    scienceworld = dependencies.scienceworld
    alfworld_task_families = load_official_alfworld_task_families(alfworld)
    return (
        ProcessTaskManifestInput(
            benchmark=Benchmark.WEBSHOP,
            dataset_revision=webshop.deployment.runtime.source_revision,
            split="train",
            task_manifest_relative_path="_derived/process/webshop.jsonl",
            asset_relative_files=webshop_asset_relative_files,
            environment_source_revision=webshop.deployment.runtime.source_revision,
            records=enumerate_official_webshop_tasks(webshop),
        ),
        ProcessTaskManifestInput(
            benchmark=Benchmark.ALFWORLD,
            dataset_revision=alfworld.runtime.source_revision,
            split="train",
            task_manifest_relative_path="_derived/process/alfworld.jsonl",
            asset_relative_files=alfworld_asset_relative_files,
            environment_source_revision=alfworld.runtime.source_revision,
            records=enumerate_official_alfworld_tasks(
                alfworld,
                task_families=alfworld_task_families,
                max_steps=alfworld_max_steps,
            ),
        ),
        ProcessTaskManifestInput(
            benchmark=Benchmark.SCIENCE_WORLD,
            dataset_revision=scienceworld.runtime.source_revision,
            split="test",
            task_manifest_relative_path="_derived/process/scienceworld.jsonl",
            asset_relative_files=scienceworld_asset_relative_files,
            environment_source_revision=scienceworld.runtime.source_revision,
            records=enumerate_official_scienceworld_tasks(
                scienceworld,
                max_steps=scienceworld_max_steps,
            ),
        ),
    )


__all__ = [
    "OfficialProcessPreparationDeployment",
    "OfficialProcessPreparationFactory",
    "build_official_process_manifest_inputs",
    "enumerate_official_alfworld_tasks",
    "enumerate_official_scienceworld_tasks",
    "enumerate_official_webshop_tasks",
    "load_official_alfworld_task_families",
]
