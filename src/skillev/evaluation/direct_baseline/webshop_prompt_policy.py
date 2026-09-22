"""Public-action-surface and horizon policies for WebShop prompts.

The short purchase workflow is an independent adaptation of the public WebShop
scaffolds in SkillFlow (``training/environment.py``), AgentSquare
(``tasks/webshop/plan_prompt.py``), and the state-hint pattern in Harness-R1.
The thought-free state profile follows StateAct's WebShop ablation: keep the goal,
location/surface, and selection state explicit while asking only for the action.
The bounded action contract independently adapts IPR's conference implementation:
search is usable only on a search surface and clicks must come from current clickables.
The single-product commitment profile follows both SkillFlow's released instruction and
the official ICLR 2023 ReAct WebShop example: search once, choose one reasonable match,
select required options, and purchase.
The efficient-replay profile keeps the v10 policy fixed while pairing it with the short
source-proven train replays used by the v5 asset, following AgentBoard's one-trajectory
WebShop scaffold and ExpeL's use of successful trajectories as in-context experience.
The single-replay profile retrieves one full efficient train trajectory, matching the
one-shot prompt shape released by AgentBoard and StateAct while retaining ExpeL-style
task-similarity selection.
Only public task/action/history state is rendered here.  In particular, this
module never blocks, replaces, canonicalizes, or executes the model's action.
"""

from __future__ import annotations

PUBLISHED_WEBSHOP_INTERACTIVE_SYSTEMS = {
    "webshop-native-react-memory-v8@1": (
        "You are a WebShop agent following a short purchase pipeline within a strict 10-action "
        "budget. The CURRENT ACTION SURFACE is authoritative for this turn and every surface in "
        "history is expired. Search once with the product type and required attributes; make at "
        "most one targeted retry after a confirmed hard violation. On results, choose the first "
        "reasonable currently listed match rather than searching for perfection. On a product "
        "page, select only task-required option values that are currently visible, then buy. "
        "Description, Features, Reviews, Back to Search, and < Prev are not default exploration "
        "actions. Preserve a compact factual Memory, copy one exact current action, and never "
        "invent or programmatically correct an action. Respond only as "
        "`Memory:\n...\n\nThought:\n...\n\nAction:\nsearch[...]` or `Action:\nclick[...]`."
    ),
    "webshop-native-stateact-memory-v9@1": (
        "You are a WebShop agent using explicit state tracking within a strict 10-action budget. "
        "At every turn, repeat the goal constraints in a compact factual Memory, update the "
        "current page/surface and selected options from public feedback, then output exactly one "
        "action without a Thought section. The CURRENT ACTION SURFACE is authoritative and every "
        "surface in history is expired. Search once with the product type and required "
        "attributes; use at most one targeted retry after a confirmed hard violation. Choose a "
        "reasonable current result, select only required visible options, and buy. Copy one exact "
        "current action and never invent or programmatically correct it. Respond only as "
        "`Memory:\n...\n\nAction:\nsearch[...]` or `Action:\nclick[...]`."
    ),
    "webshop-native-stateact-memory-v10@1": (
        "You are a WebShop agent using explicit state tracking and a bounded purchase workflow "
        "within exactly 10 actions. At every turn, repeat the goal constraints and current state "
        "in compact factual Memory, then output one Action with no Thought section. Search is "
        "legal only when the CURRENT ACTION SURFACE contains the literal search capability; on "
        "every other surface output one exact currently listed click. Use at most two searches. "
        "After the second search, the current product is the final candidate: select required "
        "visible options and buy the closest item instead of returning to search. After the first "
        "search, leave a product only for a publicly confirmed wrong product type, over-limit "
        "price, or absent mandatory selectable option—not for an unknown descriptive attribute. "
        "Never invent, replace, or programmatically correct an action. Respond only as "
        "`Memory:\n...\n\nAction:\nsearch[...]` or `Action:\nclick[...]`."
    ),
    "webshop-native-stateact-memory-v11@1": (
        "You are a WebShop agent using explicit state tracking and SkillFlow's released short "
        "purchase workflow within exactly 10 actions. At every turn, keep compact factual "
        "Memory and output one Action with no Thought section. Search once using the product "
        "type plus required attributes and option values, choose one reasonable product from "
        "the first result page, select every required visible option, and click Buy Now. Once "
        "you open a product, commit to that single product: do not click Back to Search, do not "
        "open another product, and do not spend actions on Description, Features, Reviews, or "
        "Attributes. The CURRENT ACTION SURFACE is authoritative; copy one exact current click "
        "line, or fill search[...] only when the literal current search capability is present. "
        "This is advice to the model only: never invent, replace, gate, or programmatically "
        "correct an action. Respond only as `Memory:\n...\n\nAction:\nsearch[...]` or "
        "`Action:\nclick[...]`."
    ),
    "webshop-native-stateact-memory-v13@1": (
        "You are a WebShop agent using explicit state tracking and a bounded purchase workflow "
        "within exactly 10 actions. At every turn, repeat the goal constraints and current state "
        "in compact factual Memory, then output one Action with no Thought section. Search is "
        "legal only when the CURRENT ACTION SURFACE contains the literal search capability; on "
        "every other surface output one exact currently listed click. Use at most two searches. "
        "After the second search, the current product is the final candidate: select required "
        "visible options and buy the closest item instead of returning to search. After the first "
        "search, leave a product only for a publicly confirmed wrong product type, over-limit "
        "price, or absent mandatory selectable option—not for an unknown descriptive attribute. "
        "Never invent, replace, or programmatically correct an action. Respond only as "
        "`Memory:\n...\n\nAction:\nsearch[...]` or `Action:\nclick[...]`."
    ),
    "webshop-native-stateact-memory-v14@1": (
        "You are a WebShop agent using explicit state tracking and a bounded purchase workflow "
        "within exactly 10 actions. At every turn, repeat the goal constraints and current state "
        "in compact factual Memory, then output one Action with no Thought section. Search is "
        "legal only when the CURRENT ACTION SURFACE contains the literal search capability; on "
        "every other surface output one exact currently listed click. Use at most two searches. "
        "After the second search, the current product is the final candidate: select required "
        "visible options and buy the closest item instead of returning to search. After the first "
        "search, leave a product only for a publicly confirmed wrong product type, over-limit "
        "price, or absent mandatory selectable option—not for an unknown descriptive attribute. "
        "Never invent, replace, or programmatically correct an action. Respond only as "
        "`Memory:\n...\n\nAction:\nsearch[...]` or `Action:\nclick[...]`."
    ),
    "webshop-native-stateact-memory-v15@1": (
        "You are a WebShop agent using explicit state tracking and a bounded purchase workflow "
        "within exactly 10 actions. At every turn, repeat the goal constraints and current state "
        "in compact factual Memory, then output one Action with no Thought section. Search is "
        "legal only when the CURRENT ACTION SURFACE contains the literal search capability; on "
        "every other surface output one exact currently listed click. Use at most two searches. "
        "After the second search, the current product is the final candidate: select required "
        "visible options and buy the closest item instead of returning to search. After the first "
        "search, leave a product only for a publicly confirmed wrong product type, over-limit "
        "price, or absent mandatory selectable option—not for an unknown descriptive attribute. "
        "Never invent, replace, or programmatically correct an action. Respond only as "
        "`Memory:\n...\n\nAction:\nsearch[...]` or `Action:\nclick[...]`."
    ),
    "webshop-native-stateact-memory-v16@1": (
        "You are a WebShop agent using explicit state tracking and a bounded purchase workflow "
        "within exactly 10 actions. At every turn, repeat the goal constraints and current state "
        "in compact factual Memory, then output one Action with no Thought section. Search is "
        "legal only when the CURRENT ACTION SURFACE contains the literal search capability; on "
        "every other surface output one exact currently listed click. Use at most two searches. "
        "After the second search, the current product is the final candidate: select required "
        "visible options and buy the closest item instead of returning to search. After the first "
        "search, leave a product only for a publicly confirmed wrong product type, over-limit "
        "price, or absent mandatory selectable option—not for an unknown descriptive attribute. "
        "Never invent, replace, or programmatically correct an action. Respond only as "
        "`Memory:\n...\n\nAction:\nsearch[...]` or `Action:\nclick[...]`."
    ),
    "webshop-native-stateact-memory-v17@1": (
        "You are a WebShop agent using explicit state tracking and a bounded purchase workflow "
        "within exactly 10 actions. At every turn, repeat the goal constraints and current state "
        "in compact factual Memory, then output one Action with no Thought section. Search is "
        "legal only when the CURRENT ACTION SURFACE contains the literal search capability; on "
        "every other surface output one exact currently listed click. Use at most two searches. "
        "After the second search, the current product is the final candidate: select required "
        "visible options and buy the closest item instead of returning to search. After the first "
        "search, leave a product only for a publicly confirmed wrong product type, over-limit "
        "price, or absent mandatory selectable option—not for an unknown descriptive attribute. "
        "Never invent, replace, or programmatically correct an action. Respond only as "
        "`Memory:\n...\n\nAction:\nsearch[...]` or `Action:\nclick[...]`."
    ),
    "webshop-native-react-memory-v18@1": (
        "You are a WebShop ReAct agent using explicit state tracking and a bounded purchase "
        "workflow. At every turn, update the compact factual Memory, give one concise Thought "
        "about the next unfinished purchase subgoal, then output one Action. Search is legal "
        "only when the CURRENT ACTION SURFACE contains the literal search capability; on every "
        "other surface copy one exact currently listed click. Keep price limits out of search "
        "queries because price is screened from public results. Never repeat an identical search "
        "or reopen a controller-listed rejected product. A visible over-limit result is already "
        "rejected and does not need inspection. Leave a violating product with < Prev when that "
        "current line exists, preserving the result page. WebShop instructions can describe a "
        "bundle, case, band, protector, "
        "or another product through its requested attributes instead of naming its catalog type "
        "cleanly. Therefore words such as accessory, case, band, or screen protector are not by "
        "themselves a product-type violation: retain and inspect a candidate when its public "
        "title matches several rare task attributes or exposes a required option. If every "
        "viable current result is exhausted, "
        "use the current Next > line. Visit each information tab at most once for one product; "
        "after returning from a tab, never reopen it. Reuse the controller's persistent public "
        "information evidence after leaving that tab instead of forgetting it. A mandatory "
        "descriptive attribute should use positive public evidence when available. After both "
        "Description and Features have been checked, absence of the exact wording is uncertainty, "
        "not a confirmed violation: if product type, visible price, and required options remain "
        "compatible, buy the plausible product instead of looping or rejecting it. Never purchase "
        "a confirmed hard violation. "
        "Respond only as `Memory:\n...\n\nThought:\n...\n\nAction:\nsearch[...]` or "
        "`Action:\nclick[...]`."
    ),
}

PUBLISHED_WEBSHOP_EMPTY_STRUCTURED_MEMORY = {
    "webshop-native-react-memory-v8@1": (
        "Hard constraints (verbatim):\n"
        "Stable search query core:\n"
        "Search attempts (hard plan maximum 2): 0\n"
        "Current candidate / visible price:\n"
        "Confirmed task matches:\n"
        "Required selectable options remaining:\n"
        "Options selected on current candidate:\n"
        "Confirmed hard violations:\n"
        "Last action / public result:\n"
        "Workflow stage / intended exact current action:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-stateact-memory-v9@1": (
        "Goal / hard constraints:\n"
        "Current location / surface:\n"
        "Current selection:\n"
        "Stable search query core:\n"
        "Search attempts (maximum 2): 0\n"
        "Current candidate / visible price:\n"
        "Confirmed matches:\n"
        "Required options remaining:\n"
        "Confirmed hard violations:\n"
        "Last public result:\n"
        "Next purchase-pipeline stage:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-stateact-memory-v10@1": (
        "Goal / hard constraints:\n"
        "Current location / surface:\n"
        "Current selection:\n"
        "Stable search query core:\n"
        "Search attempts (maximum 2): 0\n"
        "Current candidate / visible price:\n"
        "Confirmed matches:\n"
        "Required options remaining:\n"
        "Confirmed hard violations:\n"
        "Return-to-search eligibility:\n"
        "Last public result:\n"
        "Next purchase-pipeline stage:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-stateact-memory-v11@1": (
        "Goal / hard constraints:\n"
        "Current location / surface:\n"
        "Current selection:\n"
        "Stable one-search query:\n"
        "Search attempts (planned maximum 1): 0\n"
        "Committed product / visible price:\n"
        "Confirmed matches:\n"
        "Required options remaining:\n"
        "Options selected:\n"
        "Last public result:\n"
        "Next short-workflow stage:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-stateact-memory-v13@1": (
        "Goal / hard constraints:\n"
        "Current location / surface:\n"
        "Current selection:\n"
        "Stable search query core:\n"
        "Search attempts (maximum 2): 0\n"
        "Current candidate / visible price:\n"
        "Confirmed matches:\n"
        "Required options remaining:\n"
        "Confirmed hard violations:\n"
        "Return-to-search eligibility:\n"
        "Last public result:\n"
        "Next purchase-pipeline stage:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-stateact-memory-v14@1": (
        "Goal / hard constraints:\n"
        "Current location / surface:\n"
        "Current selection:\n"
        "Stable search query core:\n"
        "Search attempts (maximum 2): 0\n"
        "Current candidate / visible price:\n"
        "Confirmed matches:\n"
        "Required options remaining:\n"
        "Confirmed hard violations:\n"
        "Return-to-search eligibility:\n"
        "Last public result:\n"
        "Next purchase-pipeline stage:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-stateact-memory-v15@1": (
        "Goal / hard constraints:\n"
        "Current location / surface:\n"
        "Current selection:\n"
        "Stable search query core:\n"
        "Search attempts (maximum 2): 0\n"
        "Current candidate / visible price:\n"
        "Confirmed matches:\n"
        "Required options remaining:\n"
        "Confirmed hard violations:\n"
        "Return-to-search eligibility:\n"
        "Last public result:\n"
        "Next purchase-pipeline stage:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-stateact-memory-v16@1": (
        "Goal / hard constraints:\n"
        "Current location / surface:\n"
        "Current selection:\n"
        "Stable search query core:\n"
        "Search attempts (maximum 2): 0\n"
        "Current candidate / visible price:\n"
        "Confirmed matches:\n"
        "Required options remaining:\n"
        "Confirmed hard violations:\n"
        "Return-to-search eligibility:\n"
        "Last public result:\n"
        "Next purchase-pipeline stage:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-stateact-memory-v17@1": (
        "Goal / hard constraints:\n"
        "Current location / surface:\n"
        "Current selection:\n"
        "Stable search query core:\n"
        "Search attempts (maximum 2): 0\n"
        "Current candidate / visible price:\n"
        "Confirmed matches:\n"
        "Required options remaining:\n"
        "Confirmed hard violations:\n"
        "Return-to-search eligibility:\n"
        "Last public result:\n"
        "Next purchase-pipeline stage:\n"
        "Ready to purchase: no"
    ),
    "webshop-native-react-memory-v18@1": (
        "Goal / hard constraints:\n"
        "Current location / surface:\n"
        "Current selection:\n"
        "Stable search query core:\n"
        "Search queries tried:\n"
        "Rejected product IDs:\n"
        "Current candidate / visible price:\n"
        "Confirmed matches:\n"
        "Required options remaining:\n"
        "Confirmed hard violations:\n"
        "Return-to-search eligibility:\n"
        "Last public result:\n"
        "Next purchase-pipeline stage:\n"
        "Ready to purchase: no"
    ),
}

_SURFACE_GROUNDED_PROFILES = frozenset(
    {
        "webshop-native-react-memory-v7@1",
        "webshop-native-react-memory-v8@1",
        "webshop-native-stateact-memory-v9@1",
        "webshop-native-stateact-memory-v10@1",
        "webshop-native-stateact-memory-v11@1",
        "webshop-native-stateact-memory-v13@1",
        "webshop-native-stateact-memory-v14@1",
        "webshop-native-stateact-memory-v15@1",
        "webshop-native-stateact-memory-v16@1",
        "webshop-native-stateact-memory-v17@1",
        "webshop-native-react-memory-v18@1",
    }
)

_THOUGHT_FREE_PROFILES = frozenset(
    {
        "webshop-native-stateact-memory-v9@1",
        "webshop-native-stateact-memory-v10@1",
        "webshop-native-stateact-memory-v11@1",
        "webshop-native-stateact-memory-v13@1",
        "webshop-native-stateact-memory-v14@1",
        "webshop-native-stateact-memory-v15@1",
        "webshop-native-stateact-memory-v16@1",
        "webshop-native-stateact-memory-v17@1",
    }
)


def is_webshop_surface_grounded_profile(profile_id: str) -> bool:
    """Whether historical surfaces must be marked expired for this profile."""

    return profile_id in _SURFACE_GROUNDED_PROFILES


def is_webshop_thought_free_profile(profile_id: str) -> bool:
    """Whether visible Thought sections are omitted from output and replay history."""

    return profile_id in _THOUGHT_FREE_PROFILES


def published_webshop_task_decomposition(profile_id: str) -> str | None:
    """Return the source-adapted short workflow for published prompt profiles."""

    if profile_id not in PUBLISHED_WEBSHOP_INTERACTIVE_SYSTEMS:
        return None
    if profile_id == "webshop-native-stateact-memory-v11@1":
        return (
            "1. Extract the product type and every required attribute/option value; make one "
            "stable query containing those discriminating terms but no price prose.\n"
            "2. Search once and choose the first reasonable currently listed product whose "
            "visible title matches the type and most constraints.\n"
            "3. Commit to that one product. Do not return to search, open a second product, or "
            "visit Description, Features, Reviews, or Attributes.\n"
            "4. Select each visible option value explicitly required by the task.\n"
            "5. Click the exact current `click[buy now]` action."
        )
    if profile_id == "webshop-native-react-memory-v18@1":
        return (
            "1. Extract the product type, mandatory attributes/options, and hard price limit; "
            "keep them verbatim in controller-owned Memory.\n"
            "2. Search with the product type and discriminating attributes, but omit price prose "
            "and dollar amounts. Never repeat an identical query.\n"
            "3. On results, skip every known price/type violation and every controller-listed "
            "rejected product. Do not infer a type violation merely from accessory, case, band, "
            "or protector wording when the same title matches requested attributes. If no viable "
            "current result remains and Next > is exposed, "
            "paginate instead of reopening a failure.\n"
            "4. On a violating product page, use the current `< Prev` action to return to that "
            "result page; do not reset to the search box when `< Prev` is available. Visit any "
            "needed information tab at most once per product.\n"
            "5. Select mandatory visible options and buy only after every mandatory descriptive "
            "attribute has positive public evidence. If Description and Features do not confirm "
            "one, reject this product and return to results."
        )
    return (
        "1. Extract the product type, required attributes/options, and price limit; "
        "form one stable query core without price prose.\n"
        "2. Search once. A second search is only for one concrete hard violation; never "
        "plan a third search.\n"
        "3. From current results, choose the first reasonable title matching the type "
        "and most constraints; do not browse for a perfect item.\n"
        "4. On the product page, select each currently visible value explicitly required "
        "by the task. Do not select unrelated variants or browse information tabs by "
        "default.\n"
        "5. If the visible product has no confirmed hard violation and required visible "
        "options are selected, click the exact current `click[buy now]` action."
    )


def _current_surface_policy(available_actions: tuple[str, ...]) -> str:
    normalized = tuple(action.strip().casefold() for action in available_actions)
    click_values = tuple(
        action[len("click[") : -1]
        for action in normalized
        if action.startswith("click[") and action.endswith("]")
    )
    if "search" in normalized:
        return (
            "CURRENT SURFACE TYPE: search. Ground the next action as `search[query]`; do not "
            "reuse a historical product ID, tab, Back to Search, or Buy Now action."
        )
    if any(len(value) == 10 and value.isalnum() for value in click_values):
        return (
            "CURRENT SURFACE TYPE: results. Inspect one exact product-ID line currently listed; "
            "do not issue a search or reuse a product ID from an older result page."
        )
    if "buy now" in click_values:
        return (
            "CURRENT SURFACE TYPE: product. Choose an exact current option/tab/navigation line, "
            "or `click[buy now]` only when the ledger supports commitment."
        )
    return (
        "CURRENT SURFACE TYPE: subpage/navigation. Buy Now is not currently legal. If the current "
        "candidate remains viable, use an exact listed return/navigation line (normally "
        "`click[< Prev]`) to reach its product page; otherwise use another exact current line."
    )


def _published_workflow_policy(
    *,
    remaining_steps: int,
    available_actions: tuple[str, ...],
    prior_actions: tuple[str, ...],
) -> str:
    """Render the compact public workflow used by the v8 prompt.

    The counters are derived only from actions that the evaluated model already
    emitted.  They are advisory memory, not a programmatic action policy.
    """

    normalized_prior = tuple(action.strip().casefold() for action in prior_actions)
    search_count = sum(action.startswith("search[") for action in normalized_prior)
    opened_products = sum(
        action.startswith("click[")
        and action.endswith("]")
        and len(action[len("click[") : -1]) == 10
        and action[len("click[") : -1].isalnum()
        for action in normalized_prior
    )
    returned_to_search = sum(action == "click[back to search]" for action in normalized_prior)
    normalized_surface = tuple(action.strip().casefold() for action in available_actions)
    click_values = tuple(
        action[len("click[") : -1]
        for action in normalized_surface
        if action.startswith("click[") and action.endswith("]")
    )

    if "search" in normalized_surface:
        if search_count == 0:
            decision = (
                "SEARCH STAGE: issue one specific query containing the product type and all "
                "discriminating required attributes, but omit price prose."
            )
        elif search_count == 1:
            decision = (
                "FINAL SEARCH RETRY: use the stable product core plus one concrete missing hard "
                "attribute. This is the only planned retry; commit from the next result page."
            )
        else:
            decision = (
                "SEARCH-BUDGET RECOVERY: the two-search budget has already been spent. Use one "
                "stable-core query only because this surface has no product action, then commit "
                "to a reasonable first-page candidate without returning here again."
            )
    elif any(len(value) == 10 and value.isalnum() for value in click_values):
        decision = (
            "RESULT SELECTION: click the first currently listed product whose visible title "
            "matches the product type and the most required attributes at a plausible price. "
            "Do not click Back to Search or Next merely to seek a perfect candidate; one "
            "reasonable non-violating match is enough."
        )
    elif "buy now" in click_values:
        option_values = tuple(
            value
            for value in click_values
            if value
            not in {
                "back to search",
                "< prev",
                "next >",
                "description",
                "features",
                "reviews",
                "attributes",
                "buy now",
                "search",
            }
        )
        option_note = (
            "Current selectable option values exist: select only values explicitly required by "
            "the task, then buy."
            if option_values
            else "No selectable option value is exposed; verify the visible title/price, then buy."
        )
        decision = (
            "PRODUCT COMMITMENT: "
            + option_note
            + " Description, Features, Reviews, Back to Search, and < Prev are not default "
            "actions. Use an information tab only when one still-unverified mandatory "
            "descriptive constraint cannot be judged from the visible title/product page and "
            "more than four actions remain."
        )
    else:
        decision = (
            "DETAIL RETURN: this information subpage cannot purchase. Immediately copy the "
            "current return action (normally `click[< prev]`) to recover the product page; do "
            "not abandon the current candidate from here."
        )

    if remaining_steps <= 2:
        phase = "FINAL PHASE: take the shortest currently legal option-or-buy route."
    elif remaining_steps <= 4:
        phase = (
            "COMMIT PHASE: do not open another information tab or leave a viable product for "
            "an unknown descriptive attribute."
        )
    else:
        phase = "PIPELINE PHASE: advance exactly one stage toward purchase."
    return (
        f"PUBLIC PROGRESS: searches used {search_count}/2; products opened {opened_products}; "
        f"returns to search {returned_to_search}. {decision} {phase} Unknown is not a confirmed "
        "violation. Before responding, copy the complete action from CURRENT ACTION SURFACE "
        "(except filling the current literal search capability)."
    )


def _bounded_commitment_policy(
    *,
    available_actions: tuple[str, ...],
    prior_actions: tuple[str, ...],
) -> str:
    """Render an advisory current-surface and two-search contract for v10."""

    searches = sum(action.strip().casefold().startswith("search[") for action in prior_actions)
    normalized = tuple(action.strip().casefold() for action in available_actions)
    click_values = tuple(
        action[len("click[") : -1]
        for action in normalized
        if action.startswith("click[") and action.endswith("]")
    )
    if "search" in normalized:
        action_contract = (
            "LEGAL ACTION TYPE THIS TURN: search[...] only. Fill the current search capability; "
            "do not output click[...]."
        )
    else:
        action_contract = (
            "LEGAL ACTION TYPE THIS TURN: click[...] only. Copy one complete current click line; "
            "search[...] is unavailable and would perform nothing."
        )
    if "buy now" not in click_values:
        return action_contract
    if searches >= 2:
        commitment = (
            "FINAL CANDIDATE AFTER TWO SEARCHES: `click[back to search]` and further searching are "
            "outside the bounded plan. Do not leave this candidate. Select any task-required "
            "visible option still missing, then use the current Buy Now action for the closest "
            "available item."
        )
    else:
        commitment = (
            "FIRST-CANDIDATE CHECK: stay and commit unless this public page explicitly proves a "
            "wrong product type, a price above the limit, or the absence of a mandatory "
            "selectable option. Unknown descriptive attributes do not justify Back to Search."
        )
    return f"{action_contract} {commitment}"


def _step0_repair_policy(
    *,
    remaining_steps: int,
    available_actions: tuple[str, ...],
    prior_actions: tuple[str, ...],
) -> str:
    """Render the Step-0 loop-avoidance contract from public execution history."""

    normalized_prior = tuple(action.strip().casefold() for action in prior_actions)
    normalized_surface = tuple(action.strip().casefold() for action in available_actions)
    searches = tuple(
        action[len("search[") : -1]
        for action in normalized_prior
        if action.startswith("search[") and action.endswith("]")
    )
    opened = tuple(
        action[len("click[") : -1]
        for action in normalized_prior
        if action.startswith("click[")
        and action.endswith("]")
        and len(action[len("click[") : -1]) == 10
        and action[len("click[") : -1].isalnum()
    )
    click_values = tuple(
        action[len("click[") : -1]
        for action in normalized_surface
        if action.startswith("click[") and action.endswith("]")
    )
    if "search" in normalized_surface:
        used = "; ".join(dict.fromkeys(searches)) or "none"
        decision = (
            "SEARCH SURFACE: issue a query distinct from every prior query. Preserve the product "
            "type and change one discriminating attribute; never recreate an earlier result "
            f"loop. Prior queries: {used}."
        )
    elif any(len(value) == 10 and value.isalnum() for value in click_values):
        pagination = (
            " If every non-rejected visible product has a public hard violation, copy the current "
            "`click[next >]` line."
            if "next >" in click_values
            else " Do not reopen a controller-listed rejected product."
        )
        decision = (
            "RESULT SURFACE: choose a non-rejected current product only if its visible title and "
            "price do not prove a hard violation. A visible over-budget price is final and does "
            "not need product-page verification." + pagination
        )
    elif "buy now" in click_values:
        return_action = (
            " If this page proves a hard violation, copy `click[< prev]`; `click[back to search]` "
            "is forbidden while `< Prev` is available."
            if "< prev" in click_values
            else " If this page proves a hard violation, use the current navigation action."
        )
        decision = (
            "PRODUCT SURFACE: compare the current title, visible price, and required option values "
            "against authoritative constraints. Buy only with no confirmed violation."
            + return_action
        )
    else:
        decision = (
            "DETAIL/NAVIGATION SURFACE: use the exact current `< Prev` line to return to the "
            "current product; do not reset the search or use a historical product action."
        )
    urgency = (
        " FINAL PHASE: take the shortest non-repeating legal progress action."
        if remaining_steps <= 3
        else " Advance one non-repeating stage and preserve the controller rejection ledger."
    )
    return (
        f"PUBLIC PROGRESS: searches={len(searches)}; products opened={len(opened)}. "
        f"{decision}{urgency} Copy one complete line from CURRENT ACTION SURFACE."
    )


def _single_product_commitment_policy(
    *,
    remaining_steps: int,
    available_actions: tuple[str, ...],
    prior_actions: tuple[str, ...],
) -> str:
    """Render SkillFlow's released one-search/one-product workflow as prompt advice.

    Every counter comes from actions already emitted by the model.  The returned text
    does not filter or alter the next action.
    """

    normalized_prior = tuple(action.strip().casefold() for action in prior_actions)
    searches = sum(action.startswith("search[") for action in normalized_prior)
    opened_products = sum(
        action.startswith("click[")
        and action.endswith("]")
        and len(action[len("click[") : -1]) == 10
        and action[len("click[") : -1].isalnum()
        for action in normalized_prior
    )
    normalized_surface = tuple(action.strip().casefold() for action in available_actions)
    click_values = tuple(
        action[len("click[") : -1]
        for action in normalized_surface
        if action.startswith("click[") and action.endswith("]")
    )
    if "search" in normalized_surface:
        if searches == 0:
            decision = (
                "ONE SEARCH NOW: use product type plus all discriminating required attributes "
                "and option values; omit budget prose and generic shopping words."
            )
        else:
            decision = (
                "RECOVERY SEARCH SURFACE: a search was already used. Reuse the stable query "
                "core once because search is the only legal action type here, then choose one "
                "reasonable result and never return to search again."
            )
    elif any(len(value) == 10 and value.isalnum() for value in click_values):
        decision = (
            "CHOOSE ONE PRODUCT: click the first current product whose visible title reasonably "
            "matches the requested type and most discriminating terms. This product becomes the "
            "single committed candidate; do not plan another product inspection."
        )
    elif "buy now" in click_values:
        option_values = tuple(
            value
            for value in click_values
            if value
            not in {
                "back to search",
                "< prev",
                "next >",
                "description",
                "features",
                "reviews",
                "attributes",
                "buy now",
                "search",
            }
        )
        option_instruction = (
            "Select only the visible option values explicitly requested by the task, then buy."
            if option_values
            else "No selectable option is exposed; use the current Buy Now action."
        )
        decision = (
            "SINGLE-PRODUCT COMMITMENT: "
            + option_instruction
            + " Do not click Back to Search or an information tab."
        )
    else:
        decision = (
            "RETURN TO COMMITTED PRODUCT: copy the current return line (normally click[< Prev]) "
            "and then select a required option or buy; do not navigate elsewhere."
        )
    urgency = (
        "FINAL TWO ACTIONS: choose a required visible option now if one remains; otherwise buy now."
        if remaining_steps <= 2
        else "Advance exactly one search -> product -> required options -> Buy Now stage."
    )
    return (
        f"PUBLIC PROGRESS: searches used {searches}; products opened {opened_products}. "
        f"{decision} {urgency} Copy the complete current action line exactly."
    )


def webshop_step_operating_policy(
    profile_id: str,
    remaining_steps: int | None,
    available_actions: tuple[str, ...],
    prior_actions: tuple[str, ...] = (),
) -> str | None:
    """Return a policy derived only from public state and the frozen action budget."""

    if profile_id == "webshop-native-react-memory-v6-state@1":
        return (
            "Treat an attribute as a known violation only when a public page contradicts it. "
            "Preserve the product/query core across refinements and prefer inspecting a visible "
            "candidate over issuing an unrelated search."
        )
    if profile_id == "webshop-native-react-memory-v6@1":
        if remaining_steps is None:
            raise ValueError("WebShop v6 budget policy requires remaining_steps")
        if remaining_steps <= 2:
            phase = (
                "COMMIT PHASE: on a non-violating product page, select a still-required visible "
                "option or buy now; do not return to search unless the page proves a hard "
                "violation."
            )
        elif remaining_steps <= 4:
            phase = (
                "COMMIT SOON: choose the best visible non-violating candidate now and preserve at "
                "least two actions for required options and purchase."
            )
        else:
            phase = (
                "DISCOVERY PHASE: make one information-gaining search or candidate inspection, "
                "without abandoning the stable query core."
            )
        return (
            f"{phase} Unknown descriptive attributes are not known violations. Never purchase "
            "with a known price, product-type, or selectable-option violation."
        )
    if profile_id == "webshop-native-stateact-memory-v11@1":
        if remaining_steps is None:
            raise ValueError("WebShop single-product workflow requires remaining_steps")
        if not available_actions:
            raise ValueError("WebShop single-product workflow requires the current action surface")
        commitment = _single_product_commitment_policy(
            remaining_steps=remaining_steps,
            available_actions=available_actions,
            prior_actions=prior_actions,
        )
        return f"{_current_surface_policy(available_actions)} {commitment}"
    if profile_id == "webshop-native-react-memory-v18@1":
        if remaining_steps is None:
            raise ValueError("WebShop Step-0 repair workflow requires remaining_steps")
        if not available_actions:
            raise ValueError("WebShop Step-0 repair workflow requires the current action surface")
        return f"{_current_surface_policy(available_actions)} " + _step0_repair_policy(
            remaining_steps=remaining_steps,
            available_actions=available_actions,
            prior_actions=prior_actions,
        )
    if profile_id in {
        "webshop-native-react-memory-v8@1",
        "webshop-native-stateact-memory-v9@1",
        "webshop-native-stateact-memory-v10@1",
        "webshop-native-stateact-memory-v13@1",
        "webshop-native-stateact-memory-v14@1",
        "webshop-native-stateact-memory-v15@1",
        "webshop-native-stateact-memory-v16@1",
        "webshop-native-stateact-memory-v17@1",
    }:
        if remaining_steps is None:
            raise ValueError("WebShop published workflow requires remaining_steps")
        if not available_actions:
            raise ValueError("WebShop published workflow requires the current action surface")
        workflow = _published_workflow_policy(
            remaining_steps=remaining_steps,
            available_actions=available_actions,
            prior_actions=prior_actions,
        )
        bounded = (
            _bounded_commitment_policy(
                available_actions=available_actions,
                prior_actions=prior_actions,
            )
            if profile_id
            in {
                "webshop-native-stateact-memory-v10@1",
                "webshop-native-stateact-memory-v13@1",
                "webshop-native-stateact-memory-v14@1",
                "webshop-native-stateact-memory-v15@1",
                "webshop-native-stateact-memory-v16@1",
                "webshop-native-stateact-memory-v17@1",
            }
            else ""
        )
        return f"{_current_surface_policy(available_actions)} {bounded} {workflow}".strip()
    if profile_id != "webshop-native-react-memory-v7@1":
        return None
    if remaining_steps is None:
        raise ValueError("WebShop grounded budget policy requires remaining_steps")
    if not available_actions:
        raise ValueError("WebShop grounding policy requires the current action surface")
    if remaining_steps <= 2:
        phase = (
            "FINAL COMMIT PHASE: take the shortest currently legal route to a viable product "
            "page, required option, or purchase; do not explore a new candidate."
        )
    elif remaining_steps <= 4:
        phase = (
            "COMMIT SOON: preserve the best non-violating candidate and the actions needed "
            "to return from detail, select options, and purchase."
        )
    else:
        phase = (
            "DISCOVERY PHASE: gain one useful fact from the current surface without abandoning "
            "the stable query core."
        )
    return (
        f"{_current_surface_policy(available_actions)} {phase} Before finalizing, verify the "
        "Action against CURRENT ACTION SURFACE, not any historical surface. Unknown descriptive "
        "attributes are not known violations."
    )


__all__ = [
    "PUBLISHED_WEBSHOP_EMPTY_STRUCTURED_MEMORY",
    "PUBLISHED_WEBSHOP_INTERACTIVE_SYSTEMS",
    "is_webshop_surface_grounded_profile",
    "is_webshop_thought_free_profile",
    "published_webshop_task_decomposition",
    "webshop_step_operating_policy",
]
