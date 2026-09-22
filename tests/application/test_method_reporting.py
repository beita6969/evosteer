from __future__ import annotations

import asyncio

from skillev.application_reporting import resolved_method_state
from skillev.calibration import EXTRACTOR_VERSION
from skillev.runtime import EventType
from skillev.runtime.event_log_reader import read_event_history
from tests.application.test_full_vertical_loop import build_application_fixture


def test_resolved_settings_and_current_evidence_are_reported_without_changing_state(
    tmp_path, training_backbone_config
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    app = fixture.application
    initial = resolved_method_state(app)
    config = fixture.public_identity.application_config
    assert initial["application_config"] == config.to_value()
    assert initial["feature_extractor_version"] == EXTRACTOR_VERSION
    assert initial["last_evidence_batch"] is None
    events = read_event_history(fixture.event_log.path)
    assert any(event.event_type is EventType.METHOD_STATE_RECORDED for event in events)
    asyncio.run(app.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1))
    state = app.projections.runtime_state()
    report = resolved_method_state(app)
    assert report["optimizer_step"] == 1
    assert report["last_evidence_policy"] != report["policy_snapshot_id"]
    assert report["last_evidence_library"] == report["library_version"]
    assert sum(cell["distinct_trajectory_count"] for cell in report["posterior_cells"]) == 2
    assert all(cell["update_count"] == 1 for cell in report["posterior_cells"])
    assert app.projections.runtime_state() is state
    non_phase = [
        event
        for event in read_event_history(fixture.event_log.path)
        if event.event_type is EventType.PHASE_DETECTION_RECORDED
    ]
    assert len(non_phase) == 1
    assert non_phase[0].payload["invoking_edge_count"] == 2
