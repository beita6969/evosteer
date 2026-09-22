from pathlib import Path

from skillev.evaluation.swe_scaffolds import SWEScaffoldStatus, load_swe_scaffolds

ROOT = Path(__file__).parents[3]


def test_only_preregistered_upstream_scaffold_is_formal() -> None:
    registry = load_swe_scaffolds(ROOT / "configs/evaluation/swe_scaffolds.yaml")
    formal = registry["released-skillflow-repository-agent@1"]
    readonly = registry["qwen-direct-readonly-repository-agent@1"]
    assert formal.permits_formal_comparison
    assert readonly.status is SWEScaffoldStatus.POSTHOC_DIAGNOSTIC
    assert not readonly.permits_formal_comparison
    assert formal.editable
    assert formal.test_execution_allowed
    assert formal.horizon == 28
