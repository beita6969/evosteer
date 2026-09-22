from skillev_private.benchmarks.protocol_v11_materialization import materialization_receipt
from skillev_private.benchmarks.protocol_v11_population import build_protocol_v11_training_mix
from skillev_private.benchmarks.protocol_v11_sources import ProtocolV11PrivateRecord

from skillev.experiments.protocol_v11 import BenchmarkV11, PopulationRoleV11
from skillev.rollout import RolloutTask


def _record(benchmark: BenchmarkV11, role: PopulationRoleV11) -> ProtocolV11PrivateRecord:
    source_id = f"{benchmark.value}/{role.value}"
    return ProtocolV11PrivateRecord(
        benchmark,
        role,
        source_id,
        RolloutTask(
            task_id=source_id,
            environment_id=f"environment:{benchmark.value}",
            task_family=benchmark.value,
            context_id=role.value,
            query="public task",
            available_tools=(),
            public_context={},
        ),
        {"private_route": source_id},
    )


def test_all_roles_are_disjoint_and_training_mix_closes() -> None:
    records = tuple(
        _record(benchmark, role) for benchmark in BenchmarkV11 for role in PopulationRoleV11
    )
    assert materialization_receipt(records, final_ids_frozen=True).complete
    mixture = build_protocol_v11_training_mix(records)
    assert len(mixture) == 5_120
    assert len({item.episode_id for item in mixture}) == 5_120
    assert {item.benchmark for item in mixture} == set(BenchmarkV11)
