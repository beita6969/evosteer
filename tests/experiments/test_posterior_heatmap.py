from __future__ import annotations

from skillev.contracts import (
    ContextFeature,
    FailureMode,
    HorizonBucket,
    PosteriorCellState,
    TokenBucket,
)
from skillev.experiments import (
    posterior_heatmap_cells,
    render_posterior_heatmap_markdown,
)


def _cell(skill_id: str, context: str, *, alpha: float, beta: float) -> PosteriorCellState:
    return PosteriorCellState(
        skill_id=skill_id,
        z=ContextFeature(
            context=context,
            failure_mode=FailureMode.SUCCESS,
            token_bucket=TokenBucket.LE_1K,
            horizon_bucket=HorizonBucket.LE_3,
        ),
        alpha=alpha,
        beta_count=beta,
        update_count=2,
        last_event_id=f"event-{skill_id}-{context}",
    )


def test_posterior_heatmap_compares_one_skill_across_contexts_and_renders() -> None:
    shopping = _cell("skill-search", "webshop", alpha=4.0, beta=2.0)
    question_answering = _cell("skill-search", "hotpotqa", alpha=2.0, beta=5.0)
    other = _cell("skill-plan", "webshop", alpha=3.0, beta=3.0)

    rows = posterior_heatmap_cells(
        (shopping, question_answering, other),
        confidence_multiplier=1.0,
    )

    assert tuple((row.skill_id, row.context) for row in rows) == (
        ("skill-plan", "webshop"),
        ("skill-search", "hotpotqa"),
        ("skill-search", "webshop"),
    )
    by_context = {row.context: row for row in rows if row.skill_id == "skill-search"}
    assert by_context["webshop"].posterior_mean == shopping.mean()
    assert by_context["webshop"].lower_confidence_bound == shopping.lcb(1.0)
    assert by_context["webshop"].evidence_weight == 4.0
    assert by_context["hotpotqa"].posterior_mean == question_answering.mean()

    rendered = render_posterior_heatmap_markdown(rows)
    assert "| skill | context | failure |" in rendered
    assert "| skill-search | hotspot" not in rendered
    assert "| skill-search | hotpotqa | success |" in rendered
    assert "| skill-search | webshop | success |" in rendered
