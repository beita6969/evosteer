from dataclasses import replace

import pytest

from skillev.contracts import ContextFeature, FailureMode, HorizonBucket, TokenBucket
from skillev.evolution.retriever import applicability_mismatch_reasons
from skillev.runtime import SkillApplicability
from skillev.training.evidence_context import TrajectoryEvidenceContext
from skillev.training.invocation_evidence import invocation_execution_links
from skillev.training.posterior_state import PosteriorEvidenceBatch
from tests.training.test_bayesian_chain import pipeline
from tests.training.test_evidence_reporting import artifact
from tests.v3_helpers import make_record, make_source


def test_declaration_span_and_terminal_outcome_are_not_a_solved_subprogram():
    record, _, _ = make_record(
        "calls", skills_by_step=(("skill-alpha",), (), ("skill-alpha",), ()), reward_success=False
    )
    first, second = invocation_execution_links(record)
    assert first.following_execution_steps == (2,)
    assert second.following_execution_steps == (4,)
    assert first.admitted
    assert not first.terminal_success


def test_each_posterior_event_links_to_its_admitted_declaration_and_label():
    item = artifact("a")
    projection, _ = pipeline()
    projection.commit(projection.preview(make_source(artifacts=(item,))))
    batch = projection.posterior_provenance.batches[0]
    context = batch.trajectory_contexts[0]
    assert context.invocation_links[0].admitted
    assert TrajectoryEvidenceContext.from_value(context.to_value()) == context
    corrupted = replace(
        context, invocation_links=(replace(context.invocation_links[0], admitted=False),)
    )
    with pytest.raises(ValueError):
        replace(batch, trajectory_contexts=(corrupted,))
    assert PosteriorEvidenceBatch.from_value(batch.to_value()) == batch


def test_online_retrieval_refuses_post_hoc_features():
    future = ContextFeature("family", FailureMode.SUCCESS, TokenBucket.LE_1K, HorizonBucket.LE_3)
    applicability = SkillApplicability(("*",), ("*",), (), ())
    with pytest.raises(TypeError):
        applicability_mismatch_reasons(applicability, future)
