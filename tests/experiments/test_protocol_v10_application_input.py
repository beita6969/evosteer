from __future__ import annotations

from dataclasses import replace

import pytest
from skillev_private.experiments.protocol_v10_application_input import (
    ProtocolV10ApplicationIdentity,
)

from skillev.experiments import FormalMethodV10
from skillev.runtime import AttemptRunCursorState, BudgetVector
from tests.experiments.formal_execution_helpers import make_formal_application
from tests.v3_helpers import make_public_identity, make_run_plan, make_skill_document


def _application_identity() -> ProtocolV10ApplicationIdentity:
    application = make_formal_application()
    run_plan = make_run_plan(maximum_cycles=3)
    public_identity = make_public_identity(
        config=application,
        run_plan=run_plan,
        seed_documents=(make_skill_document("seed"),),
        task_ids=("task-1",),
        attempt_budget=BudgetVector(model_calls=10),
        phi_per_cycle_maximum=BudgetVector(model_calls=1),
    )
    return ProtocolV10ApplicationIdentity(
        method=FormalMethodV10.BAYESIAN_IMPROVE_FULL,
        application_config=application,
        run_plan=run_plan,
        initial_run_cursor=AttemptRunCursorState.fresh(run_plan),
        snapshot_identity=public_identity.runtime_snapshot_identity(),
        phase_checkpoint_cycle_ordinals=(),
    )


@pytest.mark.parametrize("ordinal", [1, 3])
def test_protocol_v10_phase_checkpoint_ordinals_are_one_based_inclusive(
    ordinal: int,
) -> None:
    identity = _application_identity()

    assert replace(identity, phase_checkpoint_cycle_ordinals=(ordinal,))


@pytest.mark.parametrize("ordinal", [0, 4])
def test_protocol_v10_phase_checkpoint_ordinals_reject_out_of_range_values(
    ordinal: int,
) -> None:
    identity = _application_identity()

    with pytest.raises(ValueError):
        replace(identity, phase_checkpoint_cycle_ordinals=(ordinal,))
