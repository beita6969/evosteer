from skillev_private.benchmarks.mind2web import score_mind2web_submission
from skillev_private.benchmarks.source_cases import PrivateMind2WebStepTarget


def _target() -> PrivateMind2WebStepTarget:
    return PrivateMind2WebStepTarget(
        task_id="task",
        annotation_id="annotation",
        action_uid="action",
        operation="TYPE",
        original_operation="TYPE",
        value="new york",
        positive_backend_node_ids=("node-1", "node-2"),
        action_repr="TYPE node-1 new york",
    )


def test_action_f1_and_element_accuracy_are_distinct_components() -> None:
    result = score_mind2web_submission(
        _target(), operation="TYPE", backend_node_id="wrong-node", value="new york"
    )
    assert result.element_accuracy == 0
    assert result.operation_accuracy == 1
    assert result.value_f1 == 1
    assert result.action_f1 == 1
    assert result.step_success == 0


def test_multiple_positive_elements_and_wrong_value() -> None:
    result = score_mind2web_submission(
        _target(), operation="TYPE", backend_node_id="node-2", value="new"
    )
    assert result.element_accuracy == 1
    assert result.value_f1 < 1
    assert result.step_success == 0
