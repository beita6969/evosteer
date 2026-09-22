import asyncio

from skillev.contracts import EvolutionCycleCommitted, TrainingStepCommit
from skillev.runtime import EventType
from skillev.runtime.event_log_reader import read_event_history
from tests.application.test_full_vertical_loop import _optimizer_steps, build_application_fixture


def test_two_controlled_cycles_reset_z_twice_and_train_each_new_library(
    tmp_path, training_backbone_config
):
    fixture = build_application_fixture(tmp_path, training_backbone_config, cycles=2)
    app = fixture.application
    app.evolution_loop._phase_checkpoint_cycle_ordinals = (1, 2)
    summary = asyncio.run(app.evolution_loop.run(fixture.run_plan))
    assert summary.cycles_committed_this_attempt == 2
    events = read_event_history(fixture.event_log.path)
    cycles = [
        EvolutionCycleCommitted.from_value(event.payload)
        for event in events
        if event.event_type is EventType.EVOLUTION_CYCLE_COMMITTED
    ]
    steps = [
        TrainingStepCommit.from_value(event.payload)
        for event in events
        if event.event_type is EventType.TRAINING_STEP_COMMITTED
    ]
    assert [cycle.optimizer_step for cycle in cycles] == [2, 4]
    assert cycles[0].library_version_after == cycles[1].library_version_before
    assert cycles[0].z_reset_seed != cycles[1].z_reset_seed
    assert steps[2].library_version == cycles[0].library_version_after
    assert steps[4].library_version == cycles[1].library_version_after
    assert (
        len([event for event in events if event.event_type is EventType.PHASE_CHECKPOINT_PUBLISHED])
        == 2
    )
    groups = app.backbone.parameter_groups()
    assert set(
        _optimizer_steps(app.training_loop.optimizer, (*groups.forward, *groups.backward))
    ) == {5}
    assert set(_optimizer_steps(app.training_loop.optimizer, groups.z_head)) == {1}
    assert len(app.projections.runtime_state().posterior_provenance.batches) == 5
