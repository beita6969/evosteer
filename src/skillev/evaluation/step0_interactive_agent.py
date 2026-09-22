"""Episode-scoped state contracts for Step-0 native interactive evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from skillev.evaluation.direct_baseline.interactive_tasks import (
        NativeEnvironmentOutcome,
        NativeEnvironmentStep,
        NativeInteractiveTask,
        NativePublicState,
    )


@dataclass(frozen=True, slots=True)
class WebShopConstraintLedger:
    product_type: str | None = None
    required_attributes: tuple[str, ...] = ()
    price_limit: str | None = None
    selected_asin: str | None = None
    selected_options: tuple[tuple[str, str], ...] = ()
    unmet_constraints: tuple[str, ...] = ()
    search_queries: tuple[str, ...] = ()
    visited_result_pages: tuple[tuple[str, int], ...] = ()
    rejected_asins: tuple[str, ...] = ()
    rejected_products_by_query: tuple[tuple[str, str], ...] = ()
    visited_product_tabs: tuple[tuple[str, str], ...] = ()
    product_information_evidence: tuple[tuple[str, str, str], ...] = ()
    required_information_tabs_remaining: tuple[str, ...] = ()
    current_visible_price: str | None = None
    purchase_action_available: bool = False
    purchase_ready: bool = False


@dataclass(frozen=True, slots=True)
class ALFWorldSubgoalState:
    task_type: str
    target_object: str | None = None
    target_receptacle: str | None = None
    required_count: int = 1
    placed_count: int = 0
    placed_objects: tuple[str, ...] = ()
    rejected_target_objects: tuple[str, ...] = ()
    held_object: str | None = None
    target_source_location: str | None = None
    look_source_location: str | None = None
    transform: str = "none"
    transform_completed: bool = False
    current_location: str | None = None
    visited_receptacles: tuple[str, ...] = ()
    exhausted_locations: tuple[str, ...] = ()
    next_subgoal: str = "inspect the current admissible actions"


@dataclass(frozen=True, slots=True)
class EpisodeMemoryState:
    facts: tuple[str, ...] = ()
    completed_subgoals: tuple[str, ...] = ()
    active_subgoal: str | None = None
    constraints: tuple[str, ...] = ()
    last_native_action: str | None = None
    last_observation_summary: str | None = None
    webshop: WebShopConstraintLedger | None = None
    alfworld: ALFWorldSubgoalState | None = None

    def render(self) -> str:
        lines = [
            "Authoritative current task (verbatim): "
            + ("; ".join(self.constraints) if self.constraints else "public task"),
            "Constraints: " + ("; ".join(self.constraints) if self.constraints else "public task"),
            "Environment status entering this decision: NOT TERMINAL; task incomplete",
            "Last native action: " + (self.last_native_action or "none"),
            "Last public result: " + (self.last_observation_summary or "initial state"),
        ]
        if self.webshop is not None:
            ledger = self.webshop
            lines.extend(
                (
                    "Current product: " + (ledger.selected_asin or "none"),
                    "Selected options: "
                    + (
                        "; ".join(f"{key}={value}" for key, value in ledger.selected_options)
                        if ledger.selected_options
                        else "none"
                    ),
                    "Unmet constraints: "
                    + (
                        "; ".join(ledger.unmet_constraints)
                        if ledger.unmet_constraints
                        else "unknown"
                    ),
                    "Search queries tried: "
                    + ("; ".join(ledger.search_queries) if ledger.search_queries else "none"),
                    "Result pages visited by query: "
                    + (
                        "; ".join(
                            f"{query}:page-{page}" for query, page in ledger.visited_result_pages
                        )
                        if ledger.visited_result_pages
                        else "none"
                    ),
                    "Rejected products (verified hard violation or explicit abandonment): "
                    + ("; ".join(ledger.rejected_asins) if ledger.rejected_asins else "none"),
                    "Rejected products by query: "
                    + (
                        "; ".join(
                            f"{query}:{asin}" for query, asin in ledger.rejected_products_by_query
                        )
                        if ledger.rejected_products_by_query
                        else "none"
                    ),
                    "Visited product information tabs: "
                    + (
                        "; ".join(f"{asin}:{tab}" for asin, tab in ledger.visited_product_tabs)
                        if ledger.visited_product_tabs
                        else "none"
                    ),
                    "Persistent public information evidence for current product: "
                    + (
                        "; ".join(
                            f"{tab}={evidence}"
                            for product, tab, evidence in ledger.product_information_evidence
                            if ledger.selected_asin is not None
                            and product.casefold() == ledger.selected_asin.casefold()
                        )
                        or "none"
                    ),
                    "Information tabs remaining before purchase: "
                    + (
                        "; ".join(ledger.required_information_tabs_remaining)
                        if ledger.required_information_tabs_remaining
                        else "none"
                    ),
                    "Current visible price: " + (ledger.current_visible_price or "unknown"),
                    "Purchase action exposed: "
                    + ("yes" if ledger.purchase_action_available else "no"),
                )
            )
        if self.alfworld is not None:
            state = self.alfworld
            target_span, destination_span = _alfworld_task_entity_segments(
                self.constraints,
                state.task_type,
            )
            lines.extend(
                (
                    f"Task type: {state.task_type}",
                    "Required target phrase from the authoritative task: "
                    + (target_span.strip() or "derive from the authoritative task"),
                    "Required destination phrase from the authoritative task: "
                    + (destination_span.strip() or "none for this task type"),
                    "Target object: " + (state.target_object or "not yet observed"),
                    "Target destination: "
                    + (state.target_receptacle or "derive from authoritative task"),
                    "Held object: " + (state.held_object or "none"),
                    "Current location: " + (state.current_location or "unknown"),
                    "Visited locations: "
                    + (
                        "; ".join(state.visited_receptacles)
                        if state.visited_receptacles
                        else "none"
                    ),
                    "Exhausted locations (target not exposed after inspection): "
                    + (
                        "; ".join(state.exhausted_locations)
                        if state.exhausted_locations
                        else "none"
                    ),
                    f"Transform: {state.transform}; completed: {state.transform_completed}",
                    f"Placed count: {state.placed_count}/{state.required_count}",
                    "Placed target instances: "
                    + ("; ".join(state.placed_objects) if state.placed_objects else "none"),
                    "Rejected target instances after nonterminal placement: "
                    + (
                        "; ".join(state.rejected_target_objects)
                        if state.rejected_target_objects
                        else "none"
                    ),
                    "Known target source: " + (state.target_source_location or "unknown"),
                    "Known light source location: " + (state.look_source_location or "unknown"),
                )
            )
        return "\n".join(lines)


@dataclass(slots=True)
class StepZeroEpisodeState:
    episode_id: str
    task_id: str
    root_query: str
    policy_snapshot_id: str
    library_version: str
    retrieved_skill_ids: tuple[str, ...]
    invoked_skill_ids: set[str]
    controller_memory: EpisodeMemoryState
    outer_step_index: int
    architecture_turn_count: int
    current_observation: str
    current_available_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArchitectureEpisodeArtifact:
    episode_id: str
    task_id: str
    outer_steps: int
    architecture_turns: int
    retrieved_skill_ids: tuple[str, ...]
    terminal_success: bool


@runtime_checkable
class ArchitectureInteractiveAgent(Protocol):
    async def begin_interactive_episode(
        self,
        task: NativeInteractiveTask,
        state: NativePublicState,
    ) -> None: ...

    async def observe_interactive_episode(
        self,
        task_id: str,
        action: str | None,
        result: NativeEnvironmentStep | NativePublicState,
    ) -> None: ...

    def current_interactive_memory(self, task_id: str) -> str: ...

    async def finish_interactive_episode(
        self,
        task_id: str,
        outcome: NativeEnvironmentOutcome,
    ) -> ArchitectureEpisodeArtifact: ...


def initial_episode_memory(
    *,
    benchmark: str,
    task_text: str,
    task_type: str | None,
) -> EpisodeMemoryState:
    compact_task = " ".join(task_text.split())[:1000]
    if benchmark == "webshop":
        price = _price_limit(compact_task)
        return EpisodeMemoryState(
            constraints=(compact_task,),
            active_subgoal="search for a product satisfying the public constraints",
            webshop=WebShopConstraintLedger(
                price_limit=price,
                unmet_constraints=(compact_task,),
            ),
        )
    if benchmark == "alfworld":
        kind = task_type or "unknown"
        transform = next(
            (name for name in ("clean", "heat", "cool", "look") if name in kind.casefold()),
            "none",
        )
        count = 2 if "two" in kind.casefold() else 1
        return EpisodeMemoryState(
            constraints=(compact_task,),
            active_subgoal="locate the target object using an admissible command",
            alfworld=ALFWorldSubgoalState(
                task_type=kind,
                required_count=count,
                transform=transform,
            ),
        )
    raise ValueError("episode memory supports WebShop and ALFWorld")


def update_episode_memory(
    memory: EpisodeMemoryState,
    *,
    action: str | None,
    observation: str,
    available_actions: tuple[str, ...],
    terminal_success: bool | None = None,
    webshop_option_groups: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> EpisodeMemoryState:
    """Update auxiliary memory from executed public state; never reject an action."""

    summary = " ".join(observation.split())[:800]
    if action is None:
        return replace(memory, last_observation_summary=summary)
    if memory.webshop is not None:
        ledger = memory.webshop
        asin = ledger.selected_asin
        lowered_action = action.casefold()
        queries = ledger.search_queries
        visited_result_pages = ledger.visited_result_pages
        visited_tabs = ledger.visited_product_tabs
        information_evidence = ledger.product_information_evidence
        rejected_by_query = ledger.rejected_products_by_query
        selected_options = ledger.selected_options
        query_match = re.fullmatch(r"search\[(.+)]", action, re.I)
        if query_match is not None:
            query = " ".join(query_match.group(1).split())
            if query and query.casefold() not in {item.casefold() for item in queries}:
                queries = (*queries, query)
        page_match = re.search(r"\bPage\s+(\d+)\b", observation, re.I)
        if page_match is not None and queries:
            visited_result_pages = _append_unique_result_page(
                visited_result_pages,
                (queries[-1], int(page_match.group(1))),
            )
        if lowered_action.startswith("click["):
            target = action[len("click[") : -1] if action.endswith("]") else action
            if len(target) == 10 and target.isalnum():
                if asin is None or asin.casefold() != target.casefold():
                    selected_options = ()
                asin = target.casefold()
            elif target.casefold() in {"description", "features", "reviews", "attributes"}:
                if asin is not None:
                    visited_tabs = _append_unique_pair(
                        visited_tabs,
                        (asin, target.casefold()),
                    )
                    information_evidence = _replace_product_information_evidence(
                        information_evidence,
                        asin,
                        target.casefold(),
                        summary,
                    )
            elif asin is not None and target.casefold() not in {
                "back to search",
                "< prev",
                "next >",
                "buy now",
            }:
                option_group = _catalog_option_group(target, webshop_option_groups)
                if option_group is None:
                    option_group = _visible_option_group(
                        observation,
                        target,
                        available_actions,
                    )
                selected_options = _replace_selected_option(
                    selected_options,
                    option_group,
                    target,
                )
        purchase_action_available = any(
            item.casefold() == "click[buy now]" for item in available_actions
        )
        visible_price = _visible_product_price(observation) if purchase_action_available else None
        rejected = ledger.rejected_asins
        if asin is not None and _price_exceeds_limit(visible_price, ledger.price_limit):
            rejected = _append_unique(rejected, asin)
        previous_asin = ledger.selected_asin
        returned_from_product = (
            lowered_action in {"click[back to search]", "click[< prev]"}
            and ledger.purchase_action_available
            and previous_asin is not None
        )
        if returned_from_product and previous_asin is not None:
            rejected = _append_unique(rejected, previous_asin)
            if queries:
                rejected_by_query = _append_unique_pair(
                    rejected_by_query,
                    (queries[-1], previous_asin),
                )
        if lowered_action == "click[back to search]" or (
            lowered_action == "click[< prev]" and not purchase_action_available
        ):
            asin = None
            selected_options = ()
        required_options = _required_visible_option_actions(
            memory.constraints,
            available_actions,
            observation,
        )
        selected_values = {value.casefold() for _, value in selected_options}
        remaining_options = tuple(
            target for target in required_options if target.casefold() not in selected_values
        )
        current_clicks = {
            item[len("click[") : -1].casefold()
            for item in available_actions
            if item.startswith("click[") and item.endswith("]")
        }
        visited_for_current = {
            tab for product, tab in visited_tabs if asin is not None and product == asin
        }
        remaining_information_tabs = tuple(
            tab
            for tab in ("description", "features")
            if tab in current_clicks and tab not in visited_for_current
        )
        ready = (
            purchase_action_available
            and asin is not None
            and asin not in rejected
            and not remaining_options
        )
        next_subgoal = (
            "select required option: " + "; ".join(remaining_options)
            if remaining_options
            else (
                "if a mandatory descriptive attribute is not positively visible, inspect "
                f"{remaining_information_tabs[0]}; otherwise purchase the current product"
            )
            if purchase_action_available and remaining_information_tabs
            else "purchase the current product with the exact current Buy Now action"
            if purchase_action_available
            else "choose one action from the current public surface"
        )
        return replace(
            memory,
            active_subgoal=next_subgoal,
            last_native_action=action,
            last_observation_summary=summary,
            webshop=replace(
                ledger,
                selected_asin=asin,
                selected_options=selected_options,
                search_queries=queries,
                visited_result_pages=visited_result_pages,
                rejected_asins=rejected,
                rejected_products_by_query=rejected_by_query,
                visited_product_tabs=visited_tabs,
                product_information_evidence=information_evidence,
                required_information_tabs_remaining=remaining_information_tabs,
                current_visible_price=visible_price,
                purchase_action_available=purchase_action_available,
                purchase_ready=ready,
            ),
        )
    if memory.alfworld is not None:
        state = memory.alfworld
        lowered = action.casefold()
        current_location = state.current_location
        visited = state.visited_receptacles
        exhausted = state.exhausted_locations
        target_object = state.target_object
        target_receptacle = state.target_receptacle
        go_match = re.fullmatch(r"go to (.+)", action, re.I)
        if go_match is not None:
            current_location = " ".join(go_match.group(1).casefold().split())
            visited = _append_unique(visited, current_location)
        held = state.held_object
        target_source = state.target_source_location
        look_source = state.look_source_location
        rejected_targets = state.rejected_target_objects
        if lowered.startswith(("take ", "pick up ")):
            taken_object = action.split(" from ", 1)[0].split(" ", 1)[-1]
            held = taken_object
            if _alfworld_target_entity_matches_task(
                taken_object,
                memory.constraints,
                state.task_type,
            ) and _normalize_alfworld_instance(taken_object) not in {
                _normalize_alfworld_instance(item) for item in rejected_targets
            }:
                target_object = taken_object
                if current_location is not None:
                    target_source = current_location
        placement = re.fullmatch(
            r"(?:move|put|place) (.+?) (?:to|in|into|on) (.+)",
            action,
            re.I,
        )
        placed_count = state.placed_count
        placed_objects = state.placed_objects
        placement_counts = False
        if placement is not None:
            placed_object = placement.group(1)
            placement_destination = " ".join(placement.group(2).casefold().split())
            correct_target = _alfworld_target_entity_matches_task(
                placed_object,
                memory.constraints,
                state.task_type,
            )
            correct_destination = _alfworld_destination_matches_task(
                placement_destination,
                memory.constraints,
                state.task_type,
            )
            placement_counts = (
                correct_target
                and correct_destination
                and (state.required_count > 1 or terminal_success is True)
            )
            if placement_counts:
                target_receptacle = placement_destination
                placed_count = min(state.required_count, placed_count + 1)
                placed_objects = _append_unique(placed_objects, placed_object)
            elif (
                correct_target
                and correct_destination
                and state.required_count == 1
                and terminal_success is False
            ):
                rejected_targets = _append_unique(rejected_targets, placed_object)
            held = None
        transformed = state.transform_completed or (
            state.transform != "none"
            and (state.transform in lowered or state.transform in summary.casefold())
        )
        lamp_visible = _alfworld_visible_lamp(observation, available_actions)
        if lamp_visible is not None and current_location is not None:
            look_source = current_location
        target_take_visible = bool(
            _alfworld_target_take_actions(
                memory.constraints,
                available_actions,
                task_type=state.task_type,
            )
        )
        current_closed = bool(re.search(r"\b(?:is|are)\s+closed\b", observation, re.I))
        inspected_current = lowered.startswith(("go to ", "open ", "look", "examine "))
        if (
            inspected_current
            and current_location is not None
            and not target_take_visible
            and not (state.transform == "look" and lamp_visible is not None)
            and not current_closed
        ):
            exhausted = _append_unique(exhausted, current_location)
        next_subgoal = (
            "environment confirmed terminal success"
            if terminal_success is True
            else "environment remained nonterminal; reacquire target and use exact destination"
            if placement is not None and terminal_success is False and not placement_counts
            else "environment remained nonterminal; reacquire target and use exact destination"
            if placed_count >= state.required_count
            else "locate the light source and use it while holding the target"
            if held is not None and state.transform == "look"
            else "place the unrelated held object to clear inventory, then resume target search"
            if held is not None
            and not _alfworld_target_entity_matches_task(
                held,
                memory.constraints,
                state.task_type,
            )
            else "locate and take another target object"
            if held is None and placed_count < state.required_count and placed_count > 0
            else "place the transformed target in its destination"
            if transformed and held is not None
            else "apply the required transform to the held target"
            if held is not None and state.transform != "none"
            else "place the held target in its destination"
            if held is not None
            else "locate and take the target object"
        )
        return replace(
            memory,
            active_subgoal=next_subgoal,
            last_native_action=action,
            last_observation_summary=summary,
            alfworld=replace(
                state,
                target_object=target_object,
                target_receptacle=target_receptacle,
                placed_count=placed_count,
                placed_objects=placed_objects,
                rejected_target_objects=rejected_targets,
                held_object=held,
                target_source_location=target_source,
                look_source_location=look_source,
                transform_completed=transformed,
                current_location=current_location,
                visited_receptacles=visited,
                exhausted_locations=exhausted,
                next_subgoal=next_subgoal,
            ),
        )
    return replace(memory, last_native_action=action, last_observation_summary=summary)


def _price_limit(task: str) -> str | None:
    match = re.search(
        r"(?:under|below|less than|lower than|at most|no more than)\s*\$?\s*"
        r"(\d+(?:\.\d+)?)",
        task,
        re.I,
    )
    return None if match is None else match.group(1)


def _alfworld_target_take_actions(
    constraints: tuple[str, ...],
    available_actions: tuple[str, ...],
    *,
    task_type: str | None = None,
) -> tuple[str, ...]:
    matches: list[str] = []
    for action in available_actions:
        match = re.fullmatch(r"(?:take|pick up) (.+?) from .+", action, re.I)
        if match is None:
            continue
        entity_matches = (
            _alfworld_target_entity_matches_task(match.group(1), constraints, task_type)
            if task_type is not None
            else _alfworld_entity_mentioned_in_task(match.group(1), constraints)
        )
        if entity_matches:
            matches.append(action)
    return tuple(matches)


def _alfworld_entity_mentioned_in_task(
    entity: str,
    constraints: tuple[str, ...],
) -> bool:
    """Match an admissible entity to task words without substring contamination."""

    return _alfworld_entity_matches_text(entity, " ".join(constraints))


def _alfworld_target_entity_matches_task(
    entity: str,
    constraints: tuple[str, ...],
    task_type: str | None,
) -> bool:
    target_text, _ = _alfworld_task_entity_segments(constraints, task_type)
    return _alfworld_entity_matches_text(entity, target_text)


def _alfworld_destination_matches_task(
    entity: str,
    constraints: tuple[str, ...],
    task_type: str | None,
) -> bool:
    _, destination_text = _alfworld_task_entity_segments(constraints, task_type)
    return bool(destination_text) and _alfworld_entity_matches_text(entity, destination_text)


def _alfworld_task_entity_segments(
    constraints: tuple[str, ...],
    task_type: str | None,
) -> tuple[str, str]:
    """Split public task prose into target and final-placement spans.

    Transform appliances may also occur in the instruction.  Matching against
    the whole sentence made a cooling fridge or a source counter look like the
    final receptacle.  The final placement preposition provides the correct
    authority boundary without consulting private trajectory metadata.
    """

    task = " ".join(constraints)
    if task_type is not None and "look" in task_type.casefold():
        return task, ""
    placement = re.search(r"\b(?:put|move|place|take|bring|carry)\b", task, re.I)
    if placement is None:
        return task, ""
    tail = task[placement.start() :]
    prepositions = tuple(re.finditer(r"\b(?:inside\s+of|inside|into|onto|in|on|to)\b", tail, re.I))
    if not prepositions:
        return task, ""
    boundary = prepositions[-1]
    split_start = placement.start() + boundary.start()
    split_end = placement.start() + boundary.end()
    return task[:split_start], task[split_end:]


_ALFWORLD_PUBLIC_ALIASES: dict[str, tuple[str, ...]] = {
    "butterknife": (r"\b(?:knife|knives)\b",),
    "cart": (r"\b(?:metal\s+)?racks?\b",),
    "countertop": (
        r"\bcounters?(?:tops?)?\b",
        r"\b(?:left|right)\s+of\s+(?:the\s+)?(?:stove|stovetop)\b",
        r"\bisland(?:\s+counter)?\b",
    ),
    "creditcard": (r"\b(?:credit\s*)?cards?\b",),
    "desklamp": (r"\b(?:desk\s*)?(?:lamps?|lights?)\b",),
    "diningtable": (
        r"\b(?:kitchen|dining)\s+tables?\b",
        r"(?<!side )(?<!coffee )(?<!end )(?<!bedside )\btables?\b",
    ),
    "dishsponge": (r"\b(?:dish\s*)?sponges?\b",),
    "floorlamp": (r"\b(?:floor\s*)?(?:lamps?|lights?)\b",),
    "garbagecan": (r"\b(?:garbage|trash)(?:\s+(?:bins?|cans?))?\b",),
    "peppershaker": (r"\b(?:pepper\s*)?shakers?\b",),
    "saltshaker": (r"\b(?:salt\s*)?shakers?\b",),
    "spraybottle": (r"\bspray\s+bottles?\b",),
    "toiletpaper": (r"\b(?:rolls?\s+of\s+)?toilet\s+paper(?:\s+rolls?)?\b",),
    "toiletpaperhanger": (r"\btoilet\s+paper\s+(?:holders?|hangers?)\b",),
    "toiletpaperroll": (r"\b(?:rolls?\s+of\s+)?toilet\s+paper(?:\s+rolls?)?\b",),
}


def _alfworld_entity_matches_text(entity: str, text: str) -> bool:
    target = _canonical_alfworld_entity(entity)
    if not target:
        return False
    # ALFWorld's released human annotations sometimes call a simulator
    # ``PepperShaker`` a salt shaker.  Preserve that published alias so a
    # nonterminal placement can reject the literal salt instance and recover with
    # the alternate executable label.  The reverse is safe to disambiguate: an
    # explicit human "pepper shaker" must not admit a simulator SaltShaker.
    explicit_pepper = re.search(r"\bpepper\s*shakers?\b", text, re.I) is not None
    explicit_salt = re.search(r"\bsalt\s*shakers?\b", text, re.I) is not None
    if target == "saltshaker" and explicit_pepper and not explicit_salt:
        return False
    if target == "toilet" and re.search(r"\btoilet\s+paper\b", text, re.I):
        return False
    if any(re.search(pattern, text, re.I) for pattern in _ALFWORLD_PUBLIC_ALIASES.get(target, ())):
        return True

    task = text.casefold()
    replacements = (
        (r"\b(?:sets? of )?keys\b", " keychain "),
        (r"\brefrigerators?\b", " fridge "),
        (r"\bkitchen tables?\b", " diningtable "),
        (r"\bend ta(?:ble|le)s?\b", " sidetable "),
        (r"\bcounters?\b", " countertop "),
        (r"\balarm clocks?\b", " alarmclock "),
    )
    for pattern, replacement in replacements:
        task = re.sub(pattern, replacement, task)
    words = tuple(_singularize_alfworld_word(item) for item in re.findall(r"[a-z0-9]+", task))
    for start in range(len(words)):
        for width in range(1, min(4, len(words) - start + 1)):
            if "".join(words[start : start + width]) == target:
                return True
    return False


def _alfworld_visible_lamp(
    observation: str,
    available_actions: tuple[str, ...],
) -> str | None:
    public_text = f"{observation} {' '.join(available_actions)}"
    match = re.search(r"\b((?:desk|floor)\s*lamp)\s+(\d+)\b", public_text, re.I)
    if match is None:
        return None
    return f"{''.join(match.group(1).casefold().split())} {match.group(2)}"


def _singularize_alfworld_word(value: str) -> str:
    irregular = {
        "knives": "knife",
        "shelves": "shelf",
        "potatoes": "potato",
        "tomatoes": "tomato",
        "watches": "watch",
        "boxes": "box",
    }
    if value in irregular:
        return irregular[value]
    if len(value) > 4 and value.endswith("s") and not value.endswith(("ss", "us")):
        return value[:-1]
    return value


def _canonical_alfworld_task(value: str) -> str:
    normalized = value.casefold()
    replacements = (
        (r"\b(?:sets? of )?keys\b", " keychain "),
        (r"\brefrigerator\b", " fridge "),
        (r"\bkitchen table\b", " diningtable "),
        (r"\bend ta(?:ble|le)\b", " sidetable "),
        (r"\bcounter\b", " countertop "),
        (r"\balarm clock\b", " alarmclock "),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _canonical_alfworld_entity(value: str) -> str:
    without_instance = re.sub(r"\s+\d+\s*$", "", value.casefold())
    return re.sub(r"[^a-z0-9]+", "", without_instance)


def _normalize_alfworld_instance(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _visible_option_group(
    observation: str,
    target: str,
    available_actions: tuple[str, ...],
) -> str:
    """Return the nearest public option label preceding a clicked value."""

    fields = tuple(part.strip() for part in re.split(r"\s*\[SEP]\s*", observation))
    clickable = {
        _normalize_webshop_surface_value(action[len("click[") : -1])
        for action in available_actions
        if action.startswith("click[") and action.endswith("]")
    }
    for index, field in enumerate(fields):
        if _normalize_webshop_surface_value(field) != _normalize_webshop_surface_value(target):
            continue
        for candidate in reversed(fields[:index]):
            normalized = " ".join(candidate.split())
            if (
                normalized
                and _normalize_webshop_surface_value(normalized) not in clickable
                and "..." not in normalized
                and len(normalized) <= 64
                and not normalized.casefold().startswith(("instruction:", "price:", "rating:"))
                and "$" not in normalized
            ):
                return normalized.casefold()
    return "option"


def _catalog_option_group(
    target: str,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
) -> str | None:
    """Map a live option to its public catalog-declared mutually exclusive group."""

    normalized_target = _normalize_webshop_surface_value(target)
    for group, values in groups:
        if normalized_target in {_normalize_webshop_surface_value(value) for value in values}:
            return group.casefold()
    return None


def _normalize_webshop_surface_value(value: str) -> str:
    """Normalize harmless display spacing/punctuation when relating options to actions."""

    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _required_visible_option_actions(
    constraints: tuple[str, ...],
    available_actions: tuple[str, ...],
    observation: str = "",
) -> tuple[str, ...]:
    """Identify at most one task-named public value per visible option group.

    WebShop often exposes several overlapping values in one group (for example
    ``black``, ``2 black``, and ``black-black``).  Treating every value that
    shares a task token as required makes the controller either select several
    mutually exclusive values or, as happened in the Step-0 trace, select none.
    The environment observation is public and contains the group labels, so use
    it to choose the closest explicitly named value within each group.
    """

    reserved = {
        "back to search",
        "< prev",
        "next >",
        "buy now",
        "description",
        "features",
        "reviews",
        "attributes",
    }
    targets = tuple(
        action[len("click[") : -1]
        for action in available_actions
        if action.startswith("click[")
        and action.endswith("]")
        and action[len("click[") : -1].casefold() not in reserved
        and not (len(action[len("click[") : -1]) == 10 and action[len("click[") : -1].isalnum())
    )
    if not targets:
        return ()
    task_tokens = _webshop_option_tokens(" ".join(constraints))
    best_by_group: dict[str, tuple[tuple[int, int, int], str]] = {}
    for target in targets:
        target_tokens = _webshop_option_tokens(target)
        if not target_tokens:
            continue
        score = _webshop_option_match_score(task_tokens, target_tokens, target)
        if score is None:
            continue
        group = (
            _visible_option_group(observation, target, available_actions)
            if observation
            else "option"
        )
        current = best_by_group.get(group)
        if current is None or score > current[0]:
            best_by_group[group] = (score, target)
    return tuple(value for _, value in best_by_group.values())


_OPTION_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "twelve": "12",
    "sixteen": "16",
    "twenty": "20",
    "thirty": "30",
    "sixty four": "64",
}


def _webshop_option_tokens(value: str) -> tuple[str, ...]:
    text = value.casefold()
    for word, number in _OPTION_NUMBER_WORDS.items():
        text = re.sub(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", number, text)
    text = re.sub(r"\b(?:pack|set)\s+of\s+1\b", " ", text)
    text = re.sub(r"\bgigs?\b|\bgigabytes?\b", " gb ", text)
    text = re.sub(r"\binches?\b", " inch ", text)
    text = re.sub(r"\b(?:feet|foot)\b", " ft ", text)
    text = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text)
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text):
        # Product option labels are terse, while human instructions insert
        # conjunctions around the same values (``x + y`` vs ``x and y``) and
        # prose inside packaging spans (``16 oz each ... pack of 6``).  These
        # words do not distinguish selectable values, so remove them on both
        # sides before testing a contiguous public match.
        if token in {
            "a",
            "an",
            "and",
            "come",
            "comes",
            "each",
            "in",
            "make",
            "of",
            "sure",
            "the",
            "they",
            "with",
        }:
            continue
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 5 and token.endswith(("ches", "shes", "xes", "zes")):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tuple(tokens)


def _contains_token_sequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    return bool(needle) and any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def _contains_token_multiset(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    """Return whether every normalized option token occurs with sufficient multiplicity."""

    return bool(needle) and all(
        haystack.count(token) >= needle.count(token) for token in set(needle)
    )


def _webshop_option_match_score(
    task: tuple[str, ...],
    target: tuple[str, ...],
    raw_target: str,
) -> tuple[int, int, int] | None:
    """Score only public literal/normalized evidence for a selectable value."""

    if _contains_token_sequence(task, target):
        return (len(target), 3, -len(target))
    # Human instructions frequently state packaging tokens in a different
    # order from the compact option label (``pack of 6, one ounce`` versus
    # ``1 ounce (pack of 6)``).  Full normalized token inclusion is still
    # exact public evidence and distinguishes ``pack`` from ``case``.
    if len(target) >= 2 and _contains_token_multiset(task, target):
        return (len(target), 2, -len(target))
    alternatives = tuple(
        _webshop_option_tokens(part) for part in re.split(r"\s*(?:\||/|,)\s*", raw_target)
    )
    matched_alternative = max(
        (len(value) for value in alternatives if _contains_token_sequence(task, value)),
        default=0,
    )
    if matched_alternative:
        return (matched_alternative, 2, -len(target))
    # A catalog value may append a packaging/storage configuration that the
    # public instruction omits (``12 inch (pack of 1)``).  Accept the longest
    # contiguous public prefix, but never a lone generic unit/word.
    for width in range(len(target) - 1, 1, -1):
        if any(
            _contains_token_sequence(task, target[index : index + width])
            for index in range(len(target) - width + 1)
        ):
            return (width, 1, -len(target))
    return None


def _replace_selected_option(
    values: tuple[tuple[str, str], ...],
    group: str,
    target: str,
) -> tuple[tuple[str, str], ...]:
    normalized_group = group.casefold()
    retained = tuple(item for item in values if item[0].casefold() != normalized_group)
    return (*retained, (normalized_group, target.casefold()))


def _visible_product_price(observation: str) -> str | None:
    match = re.search(r"\bPrice:\s*\$\s*(\d+(?:\.\d+)?)", observation, re.I)
    return None if match is None else match.group(1)


def _price_exceeds_limit(price: str | None, limit: str | None) -> bool:
    if price is None or limit is None:
        return False
    try:
        return float(price) > float(limit)
    except ValueError:
        return False


def _append_unique(values: tuple[str, ...], value: str) -> tuple[str, ...]:
    normalized = value.casefold()
    if normalized in {item.casefold() for item in values}:
        return values
    return (*values, normalized)


def _append_unique_pair(
    values: tuple[tuple[str, str], ...],
    value: tuple[str, str],
) -> tuple[tuple[str, str], ...]:
    normalized = (value[0].casefold(), value[1].casefold())
    if normalized in {(first.casefold(), second.casefold()) for first, second in values}:
        return values
    return (*values, normalized)


def _append_unique_result_page(
    values: tuple[tuple[str, int], ...],
    value: tuple[str, int],
) -> tuple[tuple[str, int], ...]:
    normalized = (" ".join(value[0].casefold().split()), value[1])
    if normalized in {(" ".join(query.casefold().split()), page) for query, page in values}:
        return values
    return (*values, normalized)


def _replace_product_information_evidence(
    values: tuple[tuple[str, str, str], ...],
    product: str,
    tab: str,
    evidence: str,
) -> tuple[tuple[str, str, str], ...]:
    normalized_product = product.casefold()
    normalized_tab = tab.casefold()
    retained = tuple(
        item
        for item in values
        if (item[0].casefold(), item[1].casefold()) != (normalized_product, normalized_tab)
    )
    compact_evidence = " ".join(evidence.split())[:800]
    return (*retained[-11:], (normalized_product, normalized_tab, compact_evidence))


__all__ = [
    "ALFWorldSubgoalState",
    "ArchitectureEpisodeArtifact",
    "ArchitectureInteractiveAgent",
    "EpisodeMemoryState",
    "StepZeroEpisodeState",
    "WebShopConstraintLedger",
    "initial_episode_memory",
    "update_episode_memory",
]
