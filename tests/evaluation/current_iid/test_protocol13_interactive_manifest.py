import json
from pathlib import Path

from skillev_private.direct_reference.protocol13_runner import (
    load_protocol13_environment_manifest_identity,
)

from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.current_iid.protocol13.config import load_execution_contracts_v3
from skillev.experiments.protocol_v13 import load_protocol_v13

ROOT = Path(__file__).parents[3]


def _execution():
    protocol = load_protocol_v13(
        ROOT / "configs/evaluation/protocol_v13.yaml",
        ROOT / "configs/evaluation/protocol_v13_sources.yaml",
    )
    return load_execution_contracts_v3(
        ROOT / "configs/evaluation/protocol_v13_conditions.yaml", protocol=protocol
    )[Protocol13Benchmark.WEB_SHOP]


def test_interactive_runtime_identity_contains_deployment_and_every_case(
    tmp_path: Path,
) -> None:
    manifest = {
        "format": "skillev-qwen35-direct-interactive@3",
        "runtimes": {
            "runtime": {
                "interpreter_path": "/private/python",
                "source_root": "/private/webshop",
                "source_revision": "revision",
            }
        },
        "deployments": {
            "deployment": {
                "kind": "webshop-sqlite",
                "runtime": "runtime",
                "store_path": "/private/store.json",
                "goals_path": "/private/goals.json",
                "index_path": "/private/index",
                "seed": 0,
            }
        },
        "cases": [
            {
                "task_id": f"webshop:{index}",
                "benchmark": "webshop",
                "source_identity": f"WebShop/goal-{index}",
                "deployment": "deployment",
                "max_steps": 10,
                "payload": {
                    "goal_id": f"goal-{index}",
                    "goal_index": index,
                    "session_id": f"session-{index}",
                },
            }
            for index in range(128)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    identity = load_protocol13_environment_manifest_identity(path, execution=_execution())
    assert identity["deployment_id"] == "deployment"
    assert len(identity["case_identities"]) == 128
    assert identity["case_identities"][0]["max_steps"] == 10
