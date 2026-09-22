"""Native evaluation metrics for Protocol 13 interactive benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class WebShopPanelMetrics:
    average_score_percent: Decimal
    success_rate_percent: Decimal


def webshop_panel_metrics(rewards: tuple[Decimal, ...]) -> WebShopPanelMetrics:
    if len(rewards) != 128:
        raise ValueError("WebShop panel must contain 128 rewards")
    if any(not Decimal(0) <= value <= Decimal(1) for value in rewards):
        raise ValueError("WebShop reward outside [0,1]")
    return WebShopPanelMetrics(
        average_score_percent=sum(rewards, Decimal(0)) / Decimal(128) * Decimal(100),
        success_rate_percent=Decimal(sum(value == Decimal(1) for value in rewards))
        / Decimal(128)
        * Decimal(100),
    )


def alfworld_success_rate(successes: tuple[bool, ...]) -> Decimal:
    if len(successes) != 128:
        raise ValueError("ALFWorld panel must contain 128 verdicts")
    return Decimal(sum(successes)) / Decimal(128) * Decimal(100)


__all__ = ["WebShopPanelMetrics", "alfworld_success_rate", "webshop_panel_metrics"]
