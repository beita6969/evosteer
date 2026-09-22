from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from skillev.evaluation.current_iid.protocol14.aggregation import aggregate_protocol_v14
from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import (
    load_execution_contracts_v4,
    load_targets_v5,
)
from skillev.evaluation.current_iid.protocol14.publication import (
    render_protocol_v14_markdown,
)
from skillev.evaluation.current_iid.protocol14.receipts import (
    AttemptRole,
    MetricObservationV4,
    OutcomeCountsV5,
    Protocol14RunReceipt,
)
from skillev.experiments.protocol_v14 import load_protocol_v14

ROOT = Path(__file__).parents[4]


def _contracts_and_targets():
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    executions = load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )
    targets = load_targets_v5(
        ROOT / "configs/evaluation/qwen35_protocol14_iid_targets.yaml",
        executions=executions,
    )
    return executions, targets


def test_qwen_local_healthbench_is_reported_separately_from_formal_gate() -> None:
    executions, targets = _contracts_and_targets()
    started = datetime(2026, 8, 31, tzinfo=UTC)
    receipts = {}
    for benchmark, execution in executions.items():
        target = targets[benchmark]
        metrics = []
        for contract in target.metrics:
            observed = contract.reference_percent
            if benchmark is Protocol14Benchmark.HEALTHBENCH:
                observed += Decimal("8")
            metrics.append(
                MetricObservationV4(
                    metric_id=contract.metric_id,
                    unit=contract.unit,
                    numerator=observed * Decimal(execution.expected_count) / Decimal(100),
                    denominator=execution.expected_count,
                    denominator_id=contract.denominator_id,
                    projection=contract.projection,
                    observed_percent=observed,
                    formula=contract.formula,
                    aggregation=contract.aggregation,
                )
            )
        receipts[benchmark] = Protocol14RunReceipt(
            benchmark=benchmark,
            role=AttemptRole.CANDIDATE,
            execution=execution,
            attempt_id=f"candidate-{benchmark.value}",
            planned_count=execution.expected_count,
            outcomes=OutcomeCountsV5(execution.expected_count, 0, 0, 0, 0),
            metrics=tuple(metrics),
            generation_code_revision="candidate-revision",
            scoring_code_revision="candidate-revision",
            evaluator_version="matched-evaluator",
            started_at=started.isoformat(),
            completed_at=(started + timedelta(minutes=1)).isoformat(),
            runtime_contract_matched=True,
            final_panel_used_for_selection=False,
        )

    aggregate = aggregate_protocol_v14(receipts, targets)

    assert aggregate["overall_status"] == "pass"
    assert aggregate["formal_gate_status"] == "pass"
    assert aggregate["diagnostic_completion_status"] == "complete"
    assert aggregate["diagnostic_benchmarks"] == ["healthbench"]
    assert "healthbench" not in aggregate["formal_gate_benchmarks"]
    health = aggregate["benchmarks"]["healthbench"]
    assert health["formal_status"] == "diagnostic-only"
    assert health["diagnostic_status"] == "above-reference"
    assert not health["formal_target_admitted"]
    assert health["admission"] == {}
    assert (
        health["diagnostic_comparison"]["overall_score"]["absolute_gap_pp"] == "8.0000000000000000"
    )

    markdown = render_protocol_v14_markdown(aggregate)
    assert "## Formal parity results" in markdown
    assert "## Diagnostic results" in markdown
    assert "same adapter-free Qwen3.5-9B SGLang model family" in markdown
    assert "never by OpenAI or GPT" in markdown
