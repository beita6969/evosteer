#!/usr/bin/env python3
"""Aggregate nested Protocol 13 WebShop episode traces."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import cast

_MAX_PRICE = re.compile(
    r"(?i)(?:under|below|less than|maximum(?: price)?(?: of)?)\s*\$?([0-9]+(?:\.[0-9]+)?)"
)
_PAGE_PRICE = re.compile(r"(?i)(?:price\s*:?|\$)\s*\$?([0-9]+(?:\.[0-9]+)?)")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    (
        "a an and are be buy find for from i in is it me my of on or please that the to want with"
    ).split()
)


def classify_webshop_episode(
    row: dict[str, object],
    *,
    task: str | None = None,
) -> Counter[str]:
    result: Counter[str] = Counter(episode=1)
    prior_action: str | None = None
    prior_before: str | None = None
    queries: list[str] = []
    products: list[str] = []
    option_values: list[str] = []
    bought_from: str | None = None
    price_limit = _maximum_price(task or "")
    observed_price: float | None = None
    first_product_step: int | None = None
    purchase_step: int | None = None
    search_after_product = 0
    late_searches = 0
    result_surfaces = 0
    historical_surfaces: list[tuple[str, ...]] = []
    saw_unlisted_action = False
    max_steps = int(cast(int, row.get("max_steps", 0)))
    for step in cast(list[dict[str, object]], row.get("trace") or []):
        action = step.get("parsed_action")
        before = str(step.get("observation_before") or "")
        after = str(step.get("observation_after") or "")
        step_index = int(cast(int, step.get("step_index", 0)))
        available_after = tuple(
            str(item) for item in cast(list[str], step.get("available_actions_after") or [])
        )
        available_before = tuple(
            str(item) for item in cast(list[str], step.get("available_actions_before") or [])
        )
        result_surfaces += _looks_like_result_surface(available_after)
        if isinstance(action, str):
            lowered = action.casefold()
            result["action"] += 1
            result["search"] += lowered.startswith("search[")
            result["click"] += lowered.startswith("click[")
            result["buy_attempt"] += "buy now" in lowered
            result["consecutive_repeat"] += action == prior_action
            result["same_page_repeat"] += action == prior_action and before == prior_before
            prior_action = action
            prior_before = before
            if lowered.startswith("search["):
                queries.append(action[len("search[") : -1])
                search_after_product += first_product_step is not None
                late_searches += bool(max_steps and step_index > max_steps - 3)
            elif lowered == "click[buy now]":
                bought_from = before
                observed_price = _page_price(before)
                purchase_step = step_index
            elif lowered.startswith("click["):
                value = action[len("click[") : -1]
                if any(
                    marker in after.casefold() for marker in ("price:", "buy now", "description")
                ):
                    products.append(value)
                    if first_product_step is None:
                        first_product_step = step_index
                elif value.casefold() not in {
                    "back to search",
                    "buy now",
                    "description",
                    "features",
                    "next >",
                    "< prev",
                }:
                    option_values.append(value)
        result["result_page"] += "search result" in after.casefold()
        result["product_page"] += any(
            marker in after.casefold() for marker in ("price:", "buy now", "description")
        )
        result["option_surface"] += "options" in after.casefold()
        result["listed_action"] += step.get("action_listed_before") is True
        result["unlisted_action"] += step.get("action_listed_before") is False
        if step.get("action_listed_before") is False:
            result["first_unlisted_action"] += not saw_unlisted_action
            result["unlisted_after_first"] += saw_unlisted_action
            result["unlisted_unchanged_state"] += step.get("unchanged_state") is True
            result["stale_historical_surface_action"] += isinstance(action, str) and any(
                _action_matches_surface(action, surface) for surface in historical_surfaces
            )
            saw_unlisted_action = True
        result["purchase_with_unmet_constraints"] += (
            step.get("purchase_with_unmet_constraints") is True
        )
        historical_surfaces.append(available_before)
    reward = row.get("reward")
    result["reward_zero"] += reward == 0
    result["reward_partial"] += (
        isinstance(reward, int | float) and not isinstance(reward, bool) and 0 < reward < 1
    )
    result["reward_full"] += reward == 1
    result["horizon"] += row.get("terminated_by_horizon") is True
    result["budget_exhausted"] += row.get("budget_exhausted") is True
    result["official_terminal"] += row.get("terminal_reached") is True
    result["steps"] += int(cast(int, row.get("steps", 0)))
    failed = row.get("success") is False
    candidate_invalid = row.get("termination_reason") == "candidate-invalid"
    result["failure"] += failed
    result["candidate_invalid"] += candidate_invalid
    result["early_terminal_failure"] += (
        failed and row.get("terminal_reached") is True and not candidate_invalid
    )
    purchase_failure = failed and bought_from is not None
    result["premature_purchase"] += purchase_failure
    reward = row.get("reward")
    result["probable_product_selection_error"] += (
        purchase_failure
        and isinstance(reward, int | float)
        and not isinstance(reward, bool)
        and reward < 0.5
    )
    task_lower = (task or "").casefold()
    result["probable_option_error"] += purchase_failure and any(
        value.casefold() not in task_lower for value in option_values
    )
    result["price_exceeded"] += (
        purchase_failure
        and price_limit is not None
        and observed_price is not None
        and observed_price > price_limit
    )
    result["repeated_search"] += len(queries) - len(set(map(str.casefold, queries)))
    result["revisited_product"] += len(products) - len(set(map(str.casefold, products)))
    result["search_query_degradation"] += _query_degraded(task or "", queries)
    result["result_surface"] += result_surfaces
    result["search_after_product"] += search_after_product
    result["late_search"] += late_searches
    result["excessive_search_episode"] += len(queries) >= 3
    result["purchase_at_last_step"] += purchase_step == max_steps and max_steps > 0
    result["horizon_without_purchase"] += (
        failed and row.get("budget_exhausted") is True and purchase_step is None
    )
    result["product_inspected_without_purchase"] += (
        failed and bool(products) and purchase_step is None
    )
    result["option_selected_without_purchase"] += (
        failed and bool(option_values) and purchase_step is None
    )
    result["no_product_inspected"] += failed and not products
    result["zero_reward_without_purchase"] += reward == 0 and purchase_step is None
    result["partial_reward_purchase"] += (
        purchase_step is not None
        and isinstance(reward, int | float)
        and not isinstance(reward, bool)
        and 0 < reward < 1
    )
    result["commitment_failure"] += (
        failed
        and first_product_step is not None
        and purchase_step is None
        and (search_after_product > 0 or row.get("budget_exhausted") is True)
    )
    result["episode_with_unlisted_action"] += saw_unlisted_action
    return result


def _maximum_price(task: str) -> float | None:
    match = _MAX_PRICE.search(task)
    return float(match.group(1)) if match else None


def _page_price(observation: str) -> float | None:
    values = [float(match.group(1)) for match in _PAGE_PRICE.finditer(observation)]
    return values[-1] if values else None


def _query_degraded(task: str, queries: list[str]) -> bool:
    goal = {word for word in _WORD.findall(task.casefold()) if word not in _STOP}
    if not goal or len(queries) < 2:
        return False
    overlap = [len(goal.intersection(_WORD.findall(query.casefold()))) for query in queries]
    return overlap[-1] < max(overlap[:-1])


def _looks_like_result_surface(actions: tuple[str, ...]) -> bool:
    product_clicks = 0
    for action in actions:
        lowered = action.casefold()
        if not lowered.startswith("click["):
            continue
        value = lowered[len("click[") : -1]
        if value in {
            "back to search",
            "buy now",
            "description",
            "features",
            "reviews",
            "attributes",
            "next >",
            "< prev",
        }:
            continue
        if len(value) == 10 and value.isalnum():
            product_clicks += 1
    return product_clicks > 0


def _action_matches_surface(action: str, surface: tuple[str, ...]) -> bool:
    normalized_action = " ".join(action.strip().casefold().split())
    normalized_surface = {" ".join(item.strip().casefold().split()) for item in surface}
    if normalized_action in normalized_surface:
        return True
    return (
        "search" in normalized_surface
        and normalized_action.startswith("search[")
        and normalized_action.endswith("]")
        and bool(normalized_action[len("search[") : -1].strip())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    arguments = parser.parse_args()
    counts: Counter[str] = Counter()
    tasks: dict[str, str] = {}
    if arguments.manifest is not None:
        manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
        tasks = {
            str(item["task_id"]): str(item["task"])
            for item in manifest["cases"]
            if item.get("benchmark") == "webshop"
        }
    with arguments.trace.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("trace row must be an object")
            task_id = str(row.get("task_id"))
            counts.update(
                classify_webshop_episode(
                    cast(dict[str, object], row),
                    task=tasks.get(task_id),
                )
            )
    print(json.dumps(dict(sorted(counts.items())), indent=2))


if __name__ == "__main__":
    main()
