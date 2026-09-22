"""Historical symbolic policies; never imported by the clean controller."""

from __future__ import annotations

import re

from .direct_baseline.parsing import ParseStatus, parse_visible_react
from .direct_baseline.prompts import WebShopPublicCatalogCandidate
from .step0_interactive_agent import (
    EpisodeMemoryState,
    _alfworld_destination_matches_task,
    _alfworld_target_entity_matches_task,
    _alfworld_target_take_actions,
    _canonical_alfworld_entity,
    _normalize_alfworld_instance,
    _required_visible_option_actions,
)
from .step0_types import StepZeroActionMode, StepZeroTaskBinding


def _blocked_native_actions(
    binding: StepZeroTaskBinding,
    memory: EpisodeMemoryState | None,
) -> tuple[str, ...]:
    """Keep verified WebShop failures out of later constrained action surfaces."""

    if (
        binding.action_mode is not StepZeroActionMode.WEB_SHOP
        or memory is None
        or memory.webshop is None
    ):
        return ()
    blocked = [f"click[{asin}]" for asin in memory.webshop.rejected_asins]
    current_asin = memory.webshop.selected_asin
    if current_asin is not None:
        blocked.extend(
            f"click[{tab}]"
            for asin, tab in memory.webshop.visited_product_tabs
            if asin.casefold() == current_asin.casefold()
        )
        blocked.extend(f"click[{value}]" for _, value in memory.webshop.selected_options)
        if not memory.webshop.purchase_ready:
            blocked.append("click[buy now]")
    return tuple(blocked)


def _webshop_catalog_navigation_action(
    candidates: tuple[WebShopPublicCatalogCandidate, ...],
    memory: EpisodeMemoryState,
    available_actions: tuple[str, ...],
    observation: str = "",
    *,
    option_surface: tuple[str, ...] | None = None,
) -> str | None:
    """Return one public-catalog navigation edge when it is executable now.

    This is an inference-time orchestration tool, not the phase-transition
    evolution operator. Candidate ordering and fields are derived before the
    episode from the public instruction and public product catalog. The tool
    routes exact searches and clicks already present on the authoritative
    current surface. On a selected catalog candidate, it applies task-named
    options and permits Buy Now only when the public controller ledger reports
    the candidate purchase-ready. Hidden goals and scorer feedback are never
    inputs to this decision.
    """

    if not candidates or memory.webshop is None:
        return None
    ledger = memory.webshop
    rejected = {value.casefold() for value in ledger.rejected_asins}
    current_by_fold = {action.casefold(): action for action in available_actions}
    candidate_ids = {candidate.product_id.casefold() for candidate in candidates}
    selected = ledger.selected_asin
    if (
        selected is not None
        and selected.casefold() in candidate_ids
        and selected.casefold() not in rejected
    ):
        selected_candidate = next(
            candidate
            for candidate in candidates
            if candidate.product_id.casefold() == selected.casefold()
        )
        chosen_options = {value.casefold() for _, value in ledger.selected_options}
        required_options = _required_catalog_option_actions(
            selected_candidate,
            memory.constraints,
            available_actions if option_surface is None else option_surface,
        )
        if not selected_candidate.options:
            required_options = _required_visible_option_actions(
                memory.constraints,
                available_actions if option_surface is None else option_surface,
                observation,
            )
        for option in required_options:
            if option.casefold() not in chosen_options:
                action = current_by_fold.get(f"click[{option}]".casefold())
                if action is not None:
                    return action
        catalog_ready = all(option.casefold() in chosen_options for option in required_options)
        if (
            catalog_ready if selected_candidate.options else ledger.purchase_ready
        ) and "click[buy now]" in current_by_fold:
            return current_by_fold["click[buy now]"]
    for candidate in candidates:
        if candidate.product_id.casefold() in rejected:
            continue
        click = f"click[{candidate.product_id}]"
        if click.casefold() in current_by_fold:
            return current_by_fold[click.casefold()]
    if ledger.search_queries:
        active_query = " ".join(ledger.search_queries[-1].casefold().split())
        matching_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.product_id.casefold() not in rejected
                and active_query
                in {
                    " ".join(candidate.suggested_search_query.casefold().split()),
                    candidate.product_id.casefold(),
                }
            ),
            None,
        )
        next_page = current_by_fold.get("click[next >]")
        if matching_candidate is not None:
            visited_pages = {
                page
                for query, page in ledger.visited_result_pages
                if " ".join(query.casefold().split()) == active_query
            }
            # The pinned WebShop environment exposes at most five ten-item
            # result pages.  Continuing beyond that point only repeats an
            # unchanged surface, so return to the live search box instead of
            # manufacturing progress from a stale Next action.
            if next_page is not None and len(visited_pages) < 5:
                return next_page
            back_to_search = current_by_fold.get("click[back to search]")
            if back_to_search is not None:
                return back_to_search
    if not any(action.casefold() == "search" for action in available_actions):
        return None
    tried_queries = {" ".join(value.casefold().split()) for value in ledger.search_queries}
    for candidate in candidates:
        if candidate.product_id.casefold() in rejected:
            continue
        query = " ".join(candidate.suggested_search_query.split())
        if query.casefold() not in tried_queries:
            return f"search[{query}]"
    return None


def _required_catalog_option_actions(
    candidate: WebShopPublicCatalogCandidate,
    constraints: tuple[str, ...],
    available_actions: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve one public task-named value per catalog-declared option group."""

    current_targets = {
        " ".join(action[len("click[") : -1].casefold().split()): action[len("click[") : -1]
        for action in available_actions
        if action.startswith("click[") and action.endswith("]")
    }
    required: list[str] = []
    suggested = {group.casefold(): value for group, value in candidate.suggested_options}
    for group, values in candidate.options:
        group_targets = tuple(
            current_targets[normalized]
            for value in values
            if (normalized := " ".join(value.casefold().split())) in current_targets
        )
        if not group_targets:
            continue
        matched = _required_visible_option_actions(
            constraints,
            tuple(f"click[{value}]" for value in group_targets),
        )
        if matched:
            required.append(matched[0])
            continue
        recommended = suggested.get(group.casefold())
        if recommended is not None:
            normalized = " ".join(recommended.casefold().split())
            if normalized in current_targets:
                required.append(current_targets[normalized])
    return tuple(required)


def _effective_native_actions(
    binding: StepZeroTaskBinding,
    memory: EpisodeMemoryState,
    available_actions: tuple[str, ...],
    observation: str,
) -> tuple[str, ...]:
    """Project public WebShop evidence into the next model-visible action surface.

    The environment remains authoritative.  This only removes actions already
    proven incompatible by public prices or the episode rejection ledger. It
    deliberately does not infer a product-type violation from an accessory,
    case, band, or protector title: released tasks can describe those products
    indirectly through their requested attributes. It prevents the action pass
    from reopening a known failure while retaining pagination and search
    recovery.
    """

    if binding.action_mode is StepZeroActionMode.ALF_WORLD:
        return _effective_alfworld_actions(memory, available_actions)
    if binding.action_mode is not StepZeroActionMode.WEB_SHOP or memory.webshop is None:
        return available_actions
    blocked = {item.casefold() for item in _blocked_native_actions(binding, memory)}
    records = _visible_webshop_results(observation)
    limit = memory.webshop.price_limit
    product_actions = tuple(
        action
        for action in available_actions
        if action.casefold().startswith("click[")
        and action.endswith("]")
        and len(action[len("click[") : -1]) == 10
        and action[len("click[") : -1].isalnum()
    )
    for action in product_actions:
        asin = action[len("click[") : -1].casefold()
        record = records.get(asin)
        if record is None:
            continue
        _title, price = record
        if _public_price_exceeds_limit(price, limit):
            blocked.add(action.casefold())
    has_next = any(action.casefold() == "click[next >]" for action in available_actions)
    current_query = memory.webshop.search_queries[-1] if memory.webshop.search_queries else None
    visited_pages = {
        page
        for query, page in memory.webshop.visited_result_pages
        if current_query is not None and query.casefold() == current_query.casefold()
    }
    rejected_for_query = {
        asin
        for query, asin in memory.webshop.rejected_products_by_query
        if current_query is not None and query.casefold() == current_query.casefold()
    }
    query_page_budget_exhausted = len(visited_pages) >= 5
    query_candidate_budget_exhausted = len(rejected_for_query) >= 6
    if product_actions:
        # A result page has already exposed every product that can be selected
        # from it.  Returning to an older result page cannot reveal new public
        # evidence and was the source of an observed Next/Prev cycle.  Keep
        # forward pagination while it exists; expose Back to Search only on the
        # final page so the agent can issue a genuinely different query.
        blocked.add("click[< prev]")
        if has_next and not query_page_budget_exhausted and not query_candidate_budget_exhausted:
            blocked.add("click[back to search]")
        if query_page_budget_exhausted or query_candidate_budget_exhausted:
            blocked.add("click[next >]")
        if query_candidate_budget_exhausted:
            blocked.update(action.casefold() for action in product_actions)
    filtered = tuple(action for action in available_actions if action.casefold() not in blocked)
    return filtered or available_actions


def _effective_alfworld_actions(
    memory: EpisodeMemoryState,
    available_actions: tuple[str, ...],
) -> tuple[str, ...]:
    """Prefer public ALFWorld actions that advance the typed subgoal state."""

    state = memory.alfworld
    if state is None or not available_actions:
        return available_actions
    placed_objects = {_normalize_alfworld_instance(item) for item in state.placed_objects}
    rejected_targets = {
        _normalize_alfworld_instance(item) for item in state.rejected_target_objects
    }
    target_takes = tuple(
        action
        for action in _alfworld_target_take_actions(
            memory.constraints,
            available_actions,
            task_type=state.task_type,
        )
        if _normalize_alfworld_instance(action.split(" from ", 1)[0].split(" ", 1)[-1])
        not in placed_objects | rejected_targets
    )
    if state.held_object is None and target_takes:
        return target_takes
    non_target_takes = {
        action.casefold()
        for action in available_actions
        if action.casefold().startswith(("take ", "pick up ")) and action not in target_takes
    }
    blocked_placements = {
        action.casefold()
        for action in available_actions
        if re.fullmatch(r"(?:move|put|place) .+? (?:to|in|into|on) .+", action, re.I)
        and (
            state.transform == "look"
            or (state.transform != "none" and not state.transform_completed)
            or not _alfworld_destination_matches_task(
                re.split(r"\s+(?:to|in|into|on)\s+", action, maxsplit=1, flags=re.I)[-1],
                memory.constraints,
                state.task_type,
            )
        )
    }

    current = (
        _canonical_alfworld_entity(state.current_location)
        if state.current_location is not None
        else ""
    )
    held = _canonical_alfworld_entity(state.held_object or "")
    transform_appliance = {
        "clean": "sinkbasin",
        "cool": "fridge",
        "heat": "microwave",
    }.get(state.transform)
    open_current = tuple(
        action
        for action in available_actions
        if action.casefold().startswith("open ")
        and current
        and _canonical_alfworld_entity(action[5:]) == current
    )

    if state.held_object is not None:
        held_is_target = (
            _alfworld_target_entity_matches_task(
                state.held_object,
                memory.constraints,
                state.task_type,
            )
            and _normalize_alfworld_instance(state.held_object) not in rejected_targets
        )
        if not held_is_target:
            release_actions = tuple(
                action
                for action in available_actions
                if re.fullmatch(
                    r"(?:move|put|place) .+? (?:to|in|into|on) .+",
                    action,
                    re.I,
                )
                and held
                and held in _canonical_alfworld_entity(action)
            )
            if release_actions:
                return release_actions
            if open_current:
                return open_current
        if state.transform == "look":
            look_completions = tuple(
                action
                for action in available_actions
                if action.casefold().startswith("use ")
                and "lamp" in _canonical_alfworld_entity(action)
            )
            if look_completions:
                return look_completions
            known_light_location = _normalize_alfworld_instance(state.look_source_location or "")
            known_light_routes = tuple(
                action
                for action in available_actions
                if action.casefold().startswith("go to ")
                and known_light_location
                and _normalize_alfworld_instance(action[6:]) == known_light_location
            )
            if known_light_routes:
                return known_light_routes
            lamp_routes = tuple(
                action
                for action in available_actions
                if action.casefold().startswith("go to ")
                and "lamp" in _canonical_alfworld_entity(action[6:])
            )
            if lamp_routes:
                return lamp_routes
        if state.transform != "none" and not state.transform_completed:
            transforms = tuple(
                action
                for action in available_actions
                if action.casefold().startswith(state.transform + " ")
                and (not held or held in _canonical_alfworld_entity(action))
            )
            if transforms:
                return transforms
            if transform_appliance is not None and transform_appliance not in current:
                routes = _alfworld_go_actions_matching_entity(
                    available_actions,
                    transform_appliance,
                )
                if routes:
                    return routes
            if open_current:
                return open_current
        else:
            placements = tuple(
                action
                for action in available_actions
                if re.fullmatch(r"(?:move|put|place) .+? (?:to|in|into|on) .+", action, re.I)
                and held
                and held in _canonical_alfworld_entity(action)
                and _alfworld_destination_matches_task(
                    re.split(r"\s+(?:to|in|into|on)\s+", action, maxsplit=1, flags=re.I)[-1],
                    memory.constraints,
                    state.task_type,
                )
            )
            if placements:
                return placements
            if open_current and _alfworld_destination_matches_task(
                state.current_location or "",
                memory.constraints,
                state.task_type,
            ):
                return open_current
            destination_routes = tuple(
                action
                for action in available_actions
                if action.casefold().startswith("go to ")
                and _alfworld_destination_matches_task(
                    action[6:],
                    memory.constraints,
                    state.task_type,
                )
                and (
                    transform_appliance is None
                    or transform_appliance not in _canonical_alfworld_entity(action[6:])
                )
            )
            if destination_routes:
                return destination_routes

    # A visible desk-lamp action cannot complete a look task until the target is
    # in inventory.  Prioritising it while empty-handed traps the controller in
    # a publicly observable non-terminal ``use desklamp`` loop and prevents the
    # remaining receptacles from being searched.
    if state.task_type == "look_at_obj" and state.held_object is not None:
        look_completions = tuple(
            action
            for action in available_actions
            if (
                action.casefold().startswith("use ")
                and "lamp" in _canonical_alfworld_entity(action)
            )
            or (
                action.casefold().startswith("examine ")
                and " with " in action.casefold()
                and any(
                    _alfworld_target_entity_matches_task(
                        part,
                        memory.constraints,
                        state.task_type,
                    )
                    or "lamp" in _canonical_alfworld_entity(part)
                    for part in re.split(r"\s+with\s+", action[8:], flags=re.I)
                )
            )
        )
        if look_completions:
            return look_completions

    if open_current:
        return open_current

    visited = {_normalize_alfworld_instance(item) for item in state.visited_receptacles}
    exhausted = {_normalize_alfworld_instance(item) for item in state.exhausted_locations}
    go_actions = tuple(
        action for action in available_actions if action.casefold().startswith("go to ")
    )
    if state.placed_count > 0 and state.placed_count < state.required_count:
        source = _normalize_alfworld_instance(state.target_source_location or "")
        known_source_routes = tuple(
            action
            for action in go_actions
            if source
            and source not in exhausted
            and _normalize_alfworld_instance(action[6:]) == source
        )
        if known_source_routes:
            return known_source_routes
    unvisited_routes = tuple(
        action
        for action in go_actions
        if _normalize_alfworld_instance(action[6:]) not in visited | exhausted
    )
    if unvisited_routes:
        return unvisited_routes
    filtered = tuple(
        action
        for action in available_actions
        if action.casefold() not in non_target_takes
        and action.casefold() not in blocked_placements
        and not action.casefold().startswith(("close ", "help", "inventory", "look", "examine "))
        and not (
            action.casefold().startswith("go to ")
            and _normalize_alfworld_instance(action[6:]) in exhausted
        )
    )
    if filtered:
        return filtered
    safe_fallback = tuple(
        action
        for action in available_actions
        if action.casefold() not in non_target_takes and action.casefold() not in blocked_placements
    )
    return safe_fallback or available_actions


def _alfworld_go_actions_matching_entity(
    available_actions: tuple[str, ...],
    entity: str,
) -> tuple[str, ...]:
    return tuple(
        action
        for action in available_actions
        if action.casefold().startswith("go to ")
        and entity in _canonical_alfworld_entity(action[6:])
    )


def _visible_webshop_results(observation: str) -> dict[str, tuple[str, str | None]]:
    fields = tuple(part.strip() for part in re.split(r"\s*\[SEP]\s*", observation))
    records: dict[str, tuple[str, str | None]] = {}
    for index, value in enumerate(fields):
        if not re.fullmatch(r"[A-Za-z0-9]{10}", value) or index + 1 >= len(fields):
            continue
        title = fields[index + 1]
        price = None
        if index + 2 < len(fields):
            match = re.fullmatch(r"\$\s*(\d+(?:\.\d+)?)", fields[index + 2])
            if match is not None:
                price = match.group(1)
        records[value.casefold()] = (title, price)
    return records


def _public_price_exceeds_limit(price: str | None, limit: str | None) -> bool:
    if price is None or limit is None:
        return False
    try:
        return float(price) > float(limit)
    except ValueError:
        return False


def _reasoning_aligned_actions(
    binding: StepZeroTaskBinding,
    current_actions: tuple[str, ...],
    reasoning_text: str,
    *,
    memory: EpisodeMemoryState | None = None,
) -> tuple[str, ...]:
    """Transcribe Qwen's own explicit current-surface decision when unambiguous.

    This adapter is deliberately task-strategy neutral. It never chooses a product,
    household subgoal, or recovery action. It only prevents the separately constrained
    JSON pass from replacing an exact legal action that the same frozen policy already
    wrote in its reasoning pass--the failure observed in both WebShop and ALFWorld.
    """

    del binding, memory
    if not current_actions:
        return current_actions
    parsed = parse_visible_react(reasoning_text).action
    if parsed.status is ParseStatus.EXTRACTED and parsed.value in current_actions:
        return (parsed.value,)

    folded = reasoning_text.casefold()
    referenced: list[tuple[int, str]] = []
    for action in current_actions:
        candidate = action.casefold()
        last = -1
        for match in re.finditer(re.escape(candidate), folded):
            before = folded[match.start() - 1] if match.start() else ""
            after = folded[match.end()] if match.end() < len(folded) else ""
            if candidate[0].isalnum() and before.isalnum():
                continue
            if candidate[-1].isalnum() and after.isalnum():
                continue
            last = match.start()
        if last >= 0:
            referenced.append((last, action))
    if referenced:
        latest = max(position for position, _ in referenced)
        choices = tuple(action for position, action in referenced if position == latest)
        if len(choices) == 1:
            return choices
    return current_actions


def _repeats_webshop_search(
    binding: StepZeroTaskBinding,
    action: str,
    memory: EpisodeMemoryState | None,
) -> bool:
    if (
        binding.action_mode is not StepZeroActionMode.WEB_SHOP
        or memory is None
        or memory.webshop is None
    ):
        return False
    match = re.fullmatch(r"search\[(.+)]", action, re.I)
    if match is None:
        return False
    query = " ".join(match.group(1).casefold().split())
    return query in {" ".join(item.casefold().split()) for item in memory.webshop.search_queries}


def _webshop_search_has_price_prose(
    binding: StepZeroTaskBinding,
    action: str,
) -> bool:
    if binding.action_mode is not StepZeroActionMode.WEB_SHOP:
        return False
    match = re.fullmatch(r"search\[(.+)]", action, re.I)
    if match is None:
        return False
    return (
        re.search(
            r"(?:\$\s*\d)|(?:\b(?:under|below|less than|lower than|at most|no more than)\s*"
            r"\$?\s*\d)|(?:\b\d+(?:\.\d+)?\s*(?:dollars?|usd)\b)",
            match.group(1),
            re.I,
        )
        is not None
    )
