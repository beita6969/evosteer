from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from skillev.contracts import stable_hash
from skillev.evolution import SkillAuthoringAuthority
from skillev.experiments import (
    ABLATION_PROTOCOLS,
    ACTIVE_BENCHMARKS,
    BENCHMARK_METRIC_PROTOCOLS,
    BENCHMARK_SPECS,
    FIXED_SEED,
    AblationArm,
    Benchmark,
    BenchmarkMetricProtocol,
    BenchmarkRole,
    BenchmarkTaskCount,
    CappedFlowWeightArmProtocol,
    ClippedImportanceArmProtocol,
    DatasetSnapshotIdentity,
    EvaluationReportingProtocol,
    ExperimentProtocol,
    FixedSubsampleEvaluation,
    FrozenSinglePassOOD,
    FrozenTaskSequenceIdentity,
    FullArmProtocol,
    FullMethodIdentity,
    MetricRole,
    NoBayesianCalibrationArmProtocol,
    NoFlowWeightingArmProtocol,
    PopulationThresholdEvaluation,
    PosteriorMeanArmProtocol,
    ProtocolFreeze,
    ProtocolFrozenError,
    ProtocolNotFrozenError,
    ResidualOnlyPhaseArmProtocol,
    ResultBlindProtocolGate,
    ResultProtocolMismatchError,
    SchedulePurpose,
    TrainingUse,
    arm_protocol_from_value,
    build_preregistered_protocol,
    create_protocol,
    protocol_artifact_values,
    write_preregistered_protocol_artifacts,
)
from skillev.policy import TrainableStateIdentity
from skillev.runtime import BudgetVector, ExactAttemptRunPlan
from tests.experiments.formal_execution_helpers import (
    make_formal_application,
    make_formal_execution,
)


def _snapshots() -> tuple[DatasetSnapshotIdentity, ...]:
    return tuple(
        DatasetSnapshotIdentity(
            name=spec.dataset_snapshot_name,
            version="official-snapshot-v3",
            snapshot_hash=stable_hash(
                {"dataset": spec.dataset_snapshot_name, "version": "official-snapshot-v3"}
            ),
        )
        for spec in BENCHMARK_SPECS
    )


def _protocol() -> ExperimentProtocol:
    return create_protocol(
        _snapshots(),
        schedule_identities=_schedules(),
        formal_execution=_formal_execution(),
    )


def _schedules() -> tuple[FrozenTaskSequenceIdentity, ...]:
    by_purpose = {
        SchedulePurpose.IID_TRAINING: tuple(
            spec.benchmark
            for spec in BENCHMARK_SPECS
            if spec.training_use is TrainingUse.TRAINING_MIX
        ),
        SchedulePurpose.IID_EVALUATION: tuple(
            spec.benchmark for spec in BENCHMARK_SPECS if spec.role is BenchmarkRole.IID
        ),
        SchedulePurpose.OOD_EVALUATION: tuple(
            spec.benchmark for spec in BENCHMARK_SPECS if spec.role is BenchmarkRole.OOD
        ),
    }
    return tuple(
        FrozenTaskSequenceIdentity(
            purpose=purpose,
            ordered_task_ids_hash=stable_hash({"purpose": purpose.value, "test": "result-blind"}),
            task_count=len(benchmarks),
            benchmark_counts=tuple(
                BenchmarkTaskCount(benchmark=benchmark, count=1)
                for benchmark in sorted(benchmarks, key=lambda item: item.value)
            ),
            schedule_algorithm="test-result-blind-schedule@1",
        )
        for purpose, benchmarks in by_purpose.items()
    )


def _formal_execution():
    run_plan = ExactAttemptRunPlan(
        phase_search_steps=8,
        closure_steps=1,
        maximum_cycles=1,
    )
    deployment_hash = stable_hash({"deployment": "formal-protocol-test"})
    authority = SkillAuthoringAuthority(
        input_schema_id="formal-input@1",
        output_schema_id="formal-output@1",
        license_id="unit-test",
        allowed_task_families=("debug/task-family",),
        allowed_tools=(),
    )
    return make_formal_execution(
        application=make_formal_application(batch_size=1),
        run_plan=run_plan,
        training_sequence=next(
            item for item in _schedules() if item.purpose is SchedulePurpose.IID_TRAINING
        ),
        backbone_deployment_hash=deployment_hash,
        initial_trainable_state=TrainableStateIdentity.create(
            backbone_deployment_hash=deployment_hash,
            forward_adapter_hash=stable_hash("formal-forward"),
            backward_adapter_hash=stable_hash("formal-backward"),
            z_head_hash=stable_hash("formal-z"),
        ),
        initial_library_version=stable_hash("formal-library"),
        initial_skill_library_state_hash=stable_hash("formal-library-state"),
        authoring_authority=authority,
        attempt_budget=BudgetVector(input_tokens=100, output_tokens=100, model_calls=100),
        phi_per_cycle_maximum=BudgetVector(input_tokens=4, output_tokens=4, model_calls=1),
        catalog_freeze_hash=stable_hash("formal-catalog"),
    )


def test_closed_suite_has_fixed_result_blind_roles() -> None:
    assert len(BENCHMARK_SPECS) == 18
    assert tuple(spec.benchmark for spec in BENCHMARK_SPECS) == ACTIVE_BENCHMARKS
    assert sum(spec.role is BenchmarkRole.IID for spec in BENCHMARK_SPECS) == 9
    assert sum(spec.role is BenchmarkRole.OOD for spec in BENCHMARK_SPECS) == 9

    training = tuple(
        spec.benchmark for spec in BENCHMARK_SPECS if spec.training_use is TrainingUse.TRAINING_MIX
    )
    assert training == (
        Benchmark.HOTPOT_QA,
        Benchmark.TRIVIA_QA,
        Benchmark.AIME_2026,
        Benchmark.MED_QA,
        Benchmark.WEBSHOP,
        Benchmark.ALFWORLD,
        Benchmark.BIRD_SQL,
        Benchmark.APPWORLD,
        Benchmark.MBPP_PLUS,
    )
    assert all(
        spec.training_use is TrainingUse.TRAINING_MIX
        for spec in BENCHMARK_SPECS
        if spec.role is BenchmarkRole.IID
    )
    assert all(
        spec.training_use is TrainingUse.EVALUATION_ONLY
        for spec in BENCHMARK_SPECS
        if spec.role is BenchmarkRole.OOD
    )


def test_evaluation_sampling_is_a_closed_tagged_union() -> None:
    assert all(
        isinstance(spec.evaluation_sampling, PopulationThresholdEvaluation)
        for spec in BENCHMARK_SPECS
        if spec.benchmark not in {Benchmark.GPQA_DIAMOND, Benchmark.BFCL_V3}
    )
    assert all(
        spec.evaluation_sampling.to_value() == {"sampling_type": "full"}
        for spec in BENCHMARK_SPECS
        if spec.benchmark in {Benchmark.GPQA_DIAMOND, Benchmark.BFCL_V3}
    )
    sampling = PopulationThresholdEvaluation()
    assert sampling.THRESHOLD == 1000
    assert sampling.SAMPLE_SIZE == FixedSubsampleEvaluation.SAMPLE_SIZE == 500


def test_every_benchmark_has_one_exact_native_metric_protocol() -> None:
    reporting = EvaluationReportingProtocol()

    assert tuple(item.benchmark for item in BENCHMARK_METRIC_PROTOCOLS) == ACTIVE_BENCHMARKS
    for metric_protocol in BENCHMARK_METRIC_PROTOCOLS:
        assert BenchmarkMetricProtocol.from_value(metric_protocol.to_value()) == metric_protocol
        assert reporting.metric_protocol(metric_protocol.benchmark) == metric_protocol
        assert sum(item.role is MetricRole.PRIMARY for item in metric_protocol.metrics) == 1
        assert set(metric_protocol.public_report_order) == {
            item.metric_name for item in metric_protocol.metrics
        }

    assert reporting.metric_protocol(Benchmark.HOTPOT_QA).required_metric_names == (
        "exact-match",
        "token-f1",
    )
    webshop = reporting.metric_protocol(Benchmark.WEBSHOP)
    assert webshop.required_metric_names == ("webshop-score", "webshop-success")
    webshop.require_episode_metrics(
        reward_value=1.0,
        reward_native_metric_name="webshop-score",
        metric_values={"webshop-score": 1.0, "webshop-success": 1.0},
    )
    with pytest.raises(ValueError):
        webshop.require_episode_metrics(
            reward_value=0.5,
            reward_native_metric_name="webshop-score",
            metric_values={"webshop-score": 0.5},
        )
    scienceworld = reporting.metric_protocol(Benchmark.SCIENCE_WORLD)
    scienceworld.require_episode_metrics(
        reward_value=0.25,
        reward_native_metric_name="scienceworld-score",
        metric_values={"scienceworld-native-score": 25.0, "scienceworld-score": 0.25},
    )
    with pytest.raises(ValueError):
        scienceworld.require_episode_metrics(
            reward_value=0.5,
            reward_native_metric_name="scienceworld-score",
            metric_values={"scienceworld-native-score": 25.0, "scienceworld-score": 0.5},
        )


def test_full_method_identity_is_literal_and_immutable() -> None:
    identity = FullMethodIdentity()
    assert identity.to_value() == {
        "flow_weighting": "mean-normalized-uncapped",
        "format": "skillev-full-method-identity@4",
        "gradient_transform": "none",
        "importance_estimator": "raw-log-importance",
        "optimizer": "adamw-weight-decay-0",
        "phase_rule": "residual-and-entropy",
        "policy_distribution": "raw-categorical-softmax",
        "posterior_decision": "confidence-bound",
    }
    assert FullMethodIdentity.from_value(identity.to_value()) == identity
    with pytest.raises(ValueError):
        FullMethodIdentity.from_value({**identity.to_value(), "flow_weighting": "capped"})


def test_ablation_arms_are_seven_closed_tagged_variants() -> None:
    assert tuple(item.arm for item in ABLATION_PROTOCOLS) == tuple(AblationArm)
    assert tuple(type(item) for item in ABLATION_PROTOCOLS) == (
        FullArmProtocol,
        NoBayesianCalibrationArmProtocol,
        NoFlowWeightingArmProtocol,
        CappedFlowWeightArmProtocol,
        ClippedImportanceArmProtocol,
        PosteriorMeanArmProtocol,
        ResidualOnlyPhaseArmProtocol,
    )
    for arm in ABLATION_PROTOCOLS:
        assert arm_protocol_from_value(arm.to_value()) == arm
    with pytest.raises(ValueError):
        arm_protocol_from_value({"arm": AblationArm.FULL.value, "cap": 10.0})


def test_ood_contract_is_one_literal_tag() -> None:
    contract = FrozenSinglePassOOD()
    assert FrozenSinglePassOOD.from_value(contract.to_value()) == contract
    with pytest.raises(ValueError):
        FrozenSinglePassOOD.from_value({**contract.to_value(), "policy_updates_allowed": True})


def test_dataset_snapshot_wire_is_identity_only() -> None:
    snapshot = _snapshots()[0]
    assert DatasetSnapshotIdentity.from_value(snapshot.to_value()) == snapshot
    for sensitive_field in ("question", "answer", "gold", "items", "native_payload"):
        with pytest.raises(ValueError):
            DatasetSnapshotIdentity.from_value({**snapshot.to_value(), sensitive_field: "private"})


def test_protocol_round_trip_freeze_and_single_seed() -> None:
    protocol = _protocol()
    assert ExperimentProtocol.from_value(protocol.to_value()) == protocol
    assert protocol.seed == FIXED_SEED
    assert protocol.formal_execution.seed == FIXED_SEED
    with pytest.raises(FrozenInstanceError):
        protocol.seed = 7  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(protocol, seed=FIXED_SEED + 1)
    with pytest.raises(ValueError):
        ExperimentProtocol.from_value({**protocol.to_value(), "format": "v1"})

    gate = ResultBlindProtocolGate(protocol)
    with pytest.raises(ProtocolNotFrozenError):
        gate.admit_result(protocol_hash=protocol.content_hash)
    freeze = gate.freeze()
    assert ProtocolFreeze.from_value(freeze.to_value()) == freeze
    gate.admit_result(protocol_hash=protocol.content_hash)
    with pytest.raises(ResultProtocolMismatchError):
        gate.admit_result(protocol_hash=stable_hash("different"))
    with pytest.raises(ProtocolFrozenError):
        gate.replace_protocol(protocol)


def test_persisted_freeze_requires_exact_protocol() -> None:
    protocol = _protocol()
    freeze = ProtocolFreeze.create(protocol)
    ResultBlindProtocolGate(protocol, freeze=freeze)
    changed_snapshot = replace(
        protocol.dataset_snapshots[0],
        version="official-snapshot-v4",
        snapshot_hash=stable_hash("official-snapshot-v4"),
    )
    changed = replace(
        protocol,
        dataset_snapshots=(changed_snapshot, *protocol.dataset_snapshots[1:]),
    )
    with pytest.raises(ValueError):
        ResultBlindProtocolGate(changed, freeze=freeze)


def test_protocol_artifacts_are_generated_from_the_canonical_builder(tmp_path: Path) -> None:
    protocol = build_preregistered_protocol(
        dataset_snapshots=_snapshots(),
        schedule_identities=_schedules(),
        formal_execution=_formal_execution(),
    )
    protocol_path = tmp_path / "protocol.json"
    freeze_path = tmp_path / "protocol-freeze.json"

    freeze = write_preregistered_protocol_artifacts(
        protocol=protocol,
        protocol_path=protocol_path,
        freeze_path=freeze_path,
    )
    expected_protocol, expected_freeze = protocol_artifact_values(protocol)

    assert json.loads(protocol_path.read_text(encoding="utf-8")) == expected_protocol
    assert json.loads(freeze_path.read_text(encoding="utf-8")) == expected_freeze
    assert freeze.to_value()["result_ingestion_count"] == 0
    with pytest.raises(FileExistsError):
        write_preregistered_protocol_artifacts(
            protocol=protocol,
            protocol_path=protocol_path,
            freeze_path=freeze_path,
        )


def test_protocol_rejects_the_retired_v3_wire_format() -> None:
    with pytest.raises(ValueError):
        ExperimentProtocol.from_value(
            {**_protocol().to_value(), "format": "skillev-benchmark-protocol@3"}
        )


def test_protocol_source_has_no_model_dependency() -> None:
    path = Path(__file__).resolve().parents[2] / "src" / "skillev" / "experiments" / "protocol.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])
    assert imported_roots.isdisjoint({"torch", "transformers", "peft", "tokenizers"})
