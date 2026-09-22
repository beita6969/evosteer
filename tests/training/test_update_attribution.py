from dataclasses import replace

import pytest

from skillev.training.update_attribution import attribution_summary, direct_coefficients
from tests.v3_helpers import make_artifact, make_source


def test_direct_terms_use_global_batch_and_edge_token_denominator_without_filtering():
    artifacts = (
        make_artifact("positive", reward_success=True),
        make_artifact("negative", reward_success=False),
    )
    source = make_source(artifacts=artifacts)
    residuals = (
        replace(
            source.stats.residuals[0],
            delta=2.0,
            log_z=source.stats.residuals[0].log_z + 2.0 - source.stats.residuals[0].delta,
        ),
        replace(
            source.stats.residuals[1],
            delta=-3.0,
            log_z=source.stats.residuals[1].log_z - 3.0 - source.stats.residuals[1].delta,
        ),
    )
    edges = direct_coefficients(source.records, residuals)
    assert len(edges) == 2
    assert edges[0].forward_logprob_derivative == pytest.approx(
        2 / source.records[0].steps[0].action_token_count
    )
    assert edges[1].forward_logprob_derivative == pytest.approx(
        -3 / source.records[1].steps[0].action_token_count
    )
    assert all(e.forward_logprob_derivative == -e.backward_logprob_derivative for e in edges)
    assert sum(g["edge_count"] for g in attribution_summary(edges)["groups"].values()) == 2
    with pytest.raises(ValueError):
        direct_coefficients(source.records[::-1], residuals)
