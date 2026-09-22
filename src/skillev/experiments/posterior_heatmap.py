"""Context-conditional posterior table used by the differentiation analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

from skillev.contracts import (
    FailureMode,
    HorizonBucket,
    JsonValue,
    PosteriorCellState,
    TokenBucket,
)


@dataclass(frozen=True, slots=True)
class PosteriorHeatmapCell:
    """One observed ``(skill, z)`` cell rendered as posterior mean and LCB."""

    skill_id: str
    context: str
    failure_mode: FailureMode
    token_bucket: TokenBucket
    horizon_bucket: HorizonBucket
    posterior_mean: float
    lower_confidence_bound: float
    evidence_weight: float
    update_count: int

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "context": self.context,
            "evidence_weight": self.evidence_weight,
            "failure_mode": self.failure_mode.value,
            "horizon_bucket": self.horizon_bucket.value,
            "lower_confidence_bound": self.lower_confidence_bound,
            "posterior_mean": self.posterior_mean,
            "skill_id": self.skill_id,
            "token_bucket": self.token_bucket.value,
            "update_count": self.update_count,
        }


def posterior_heatmap_cells(
    cells: tuple[PosteriorCellState, ...],
    *,
    confidence_multiplier: float = 1.0,
) -> tuple[PosteriorHeatmapCell, ...]:
    """Aggregate audited posterior state into deterministic heat-map rows."""

    if not isinstance(cells, tuple) or any(
        not isinstance(cell, PosteriorCellState) for cell in cells
    ):
        raise TypeError("posterior heat-map requires PosteriorCellState values")
    if isinstance(confidence_multiplier, bool) or not isinstance(
        confidence_multiplier, int | float
    ):
        raise TypeError("confidence multiplier must be numeric")
    k = float(confidence_multiplier)
    if not math.isfinite(k) or k < 0.0:
        raise ValueError("confidence multiplier must be finite and non-negative")
    keys = tuple(cell.z.cell_key(cell.skill_id) for cell in cells)
    if len(set(keys)) != len(keys):
        raise ValueError("posterior heat-map input repeats a cell")

    rows = tuple(
        PosteriorHeatmapCell(
            skill_id=cell.skill_id,
            context=cell.z.context,
            failure_mode=cell.z.failure_mode,
            token_bucket=cell.z.token_bucket,
            horizon_bucket=cell.z.horizon_bucket,
            posterior_mean=cell.mean(),
            lower_confidence_bound=cell.lcb(k),
            evidence_weight=(float(cell.alpha) - float(cell.alpha_0))
            + (float(cell.beta_count) - float(cell.beta_0)),
            update_count=cell.update_count,
        )
        for cell in cells
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.skill_id,
                row.context,
                row.failure_mode.value,
                row.token_bucket.value,
                row.horizon_bucket.value,
            ),
        )
    )


def render_posterior_heatmap_markdown(
    rows: tuple[PosteriorHeatmapCell, ...],
) -> str:
    """Render observed context cells as a compact, reviewable Markdown table."""

    if not isinstance(rows, tuple) or any(
        not isinstance(row, PosteriorHeatmapCell) for row in rows
    ):
        raise TypeError("posterior heat-map rows are invalid")
    header = (
        "| skill | context | failure | token bucket | horizon bucket | mean | LCB | evidence |",
        "|---|---|---|---|---|---:|---:|---:|",
    )
    body = tuple(
        "| "
        + " | ".join(
            (
                row.skill_id.replace("|", "\\|"),
                row.context.replace("|", "\\|"),
                row.failure_mode.value,
                row.token_bucket.value,
                row.horizon_bucket.value,
                f"{row.posterior_mean:.6f}",
                f"{row.lower_confidence_bound:.6f}",
                f"{row.evidence_weight:.6f}",
            )
        )
        + " |"
        for row in rows
    )
    return "\n".join((*header, *body)) + "\n"


__all__ = [
    "PosteriorHeatmapCell",
    "posterior_heatmap_cells",
    "render_posterior_heatmap_markdown",
]
