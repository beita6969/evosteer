from skillev.evaluation.direct_baseline.webshop_prompt_policy import (
    published_webshop_task_decomposition,
    webshop_step_operating_policy,
)
from skillev.evaluation.legacy_step0_completion import native_action_constraint
from skillev.evaluation.step0_interactive_agent import (
    initial_episode_memory,
    update_episode_memory,
)


def test_memory_is_auxiliary_and_does_not_enter_the_executable_wire() -> None:
    memory = initial_episode_memory(
        benchmark="webshop",
        task_text="Find a red mug below 20 dollars.",
        task_type=None,
    )
    updated = update_episode_memory(
        memory,
        action="search[red mug]",
        observation="One public product result.",
        available_actions=("click[b012345678]",),
    )
    schema = native_action_constraint("webshop").json_schema
    assert updated.last_native_action == "search[red mug]"
    assert isinstance(schema, dict)
    for branch in schema["oneOf"]:  # type: ignore[index,union-attr]
        arguments = branch["properties"]["arguments"]  # type: ignore[index]
        assert "memory" not in arguments["properties"]  # type: ignore[operator]


def test_current_native_surface_constrains_webshop_clicks_and_alfworld_commands() -> None:
    webshop = native_action_constraint(
        "webshop",
        available_actions=("click[item-a]", "click[buy now]"),
    ).json_schema
    alfworld = native_action_constraint(
        "alfworld",
        available_actions=("go to shelf 1", "take mug 1 from shelf 1"),
    ).json_schema
    exact_search = native_action_constraint(
        "webshop",
        available_actions=("search[Synthetic cobalt watch case]",),
    ).json_schema

    assert webshop["properties"]["name"] == {"const": "click"}  # type: ignore[index,union-attr]
    assert webshop["properties"]["arguments"]["properties"]["target"] == {  # type: ignore[index,union-attr]
        "enum": ["item-a", "buy now"],
        "type": "string",
    }
    assert alfworld["properties"]["arguments"]["properties"]["command"] == {  # type: ignore[index,union-attr]
        "enum": ["go to shelf 1", "take mug 1 from shelf 1"],
        "type": "string",
    }
    assert exact_search["properties"]["name"] == {"const": "search"}  # type: ignore[index,union-attr]
    assert exact_search["properties"]["arguments"]["properties"]["query"] == {  # type: ignore[index,union-attr]
        "enum": ["Synthetic cobalt watch case"],
        "type": "string",
    }


def test_webshop_controller_retains_queries_and_blocks_verified_failed_products() -> None:
    memory = initial_episode_memory(
        benchmark="webshop",
        task_text="Find synthetic product alpha with price lower than 20 dollars.",
        task_type=None,
    )
    searched = update_episode_memory(
        memory,
        action="search[synthetic product alpha]",
        observation="results",
        available_actions=("click[x000000001]", "click[next >]"),
    )
    opened = update_episode_memory(
        searched,
        action="click[x000000001]",
        observation="Synthetic Product Alpha [SEP] Price: $25.00 [SEP] Buy Now",
        available_actions=("click[< Prev]", "click[back to search]", "click[buy now]"),
    )

    assert opened.webshop is not None
    assert opened.webshop.search_queries == ("synthetic product alpha",)
    assert opened.webshop.rejected_asins == ("x000000001",)
    assert opened.webshop.current_visible_price == "25.00"
    assert opened.webshop.purchase_action_available is True
    assert opened.webshop.purchase_ready is False
    assert "Rejected products" in opened.render()

    schema = native_action_constraint(
        "webshop",
        available_actions=(
            "click[x000000001]",
            "click[next >]",
            "click[< Prev]",
            "click[back to search]",
        ),
        blocked_actions=("click[x000000001]",),
    ).json_schema
    assert schema["properties"]["arguments"]["properties"]["target"] == {  # type: ignore[index,union-attr]
        "enum": ["next >", "< Prev"],
        "type": "string",
    }

    returned = update_episode_memory(
        opened,
        action="click[< Prev]",
        observation="Page 1 results",
        available_actions=("click[x000000001]", "click[next >]"),
    )
    assert returned.webshop is not None
    assert returned.webshop.selected_asin is None
    assert returned.webshop.rejected_asins == ("x000000001",)

    viable = update_episode_memory(
        searched,
        action="click[x000000002]",
        observation="Synthetic Product Alpha [SEP] Price: $15.00 [SEP] Buy Now",
        available_actions=("click[description]", "click[< Prev]", "click[buy now]"),
    )
    described = update_episode_memory(
        viable,
        action="click[description]",
        observation="Public description page",
        available_actions=("click[< Prev]",),
    )
    back_on_product = update_episode_memory(
        described,
        action="click[< Prev]",
        observation="Synthetic Product Alpha [SEP] Price: $15.00 [SEP] Buy Now",
        available_actions=(
            "click[description]",
            "click[features]",
            "click[< Prev]",
            "click[back to search]",
            "click[buy now]",
        ),
    )
    assert back_on_product.webshop is not None
    assert back_on_product.webshop.visited_product_tabs == (("x000000002", "description"),)
    tab_schema = native_action_constraint(
        "webshop",
        available_actions=(
            "click[description]",
            "click[features]",
            "click[< Prev]",
            "click[back to search]",
            "click[buy now]",
        ),
        blocked_actions=("click[description]",),
    ).json_schema
    assert tab_schema["properties"]["arguments"]["properties"]["target"] == {  # type: ignore[index,union-attr]
        "enum": ["features", "< Prev", "buy now"],
        "type": "string",
    }


def test_webshop_option_group_ignores_truncated_values_and_display_spacing() -> None:
    memory = initial_episode_memory(
        benchmark="webshop",
        task_text="Find a trishield nickel-plated fitting in black.",
        task_type=None,
    )
    observation = (
        "Instruction [SEP] Back to Search [SEP] < Prev [SEP] material type [SEP] "
        "copper, nickel plated fitting - black [SEP] "
        "burial 3ghz rg6, directv fitting - orang... [SEP] "
        "quad shield rg11 w |  weather seal - black [SEP] "
        "trishield nickel-plated fitting -black [SEP] Price: $20.00 [SEP] Buy Now"
    )
    actions = (
        "click[< Prev]",
        "click[copper, nickel plated fitting - black]",
        "click[quad shield rg11 w | weather seal - black]",
        "click[trishield nickel-plated fitting -black]",
        "click[buy now]",
    )
    opened = update_episode_memory(
        memory,
        action="click[x000000002]",
        observation=observation,
        available_actions=actions,
    )
    selected = update_episode_memory(
        opened,
        action="click[trishield nickel-plated fitting -black]",
        observation=observation,
        available_actions=actions,
    )

    assert selected.webshop is not None
    assert selected.webshop.selected_options == (
        ("material type", "trishield nickel-plated fitting -black"),
    )
    assert selected.webshop.purchase_ready is True

    length_memory = initial_episode_memory(
        benchmark="webshop",
        task_text="Find a 180 foot cable.",
        task_type=None,
    )
    length_page = "Instruction [SEP] size [SEP] 12ft [SEP] 180ft [SEP] Price: $20.00 [SEP] Buy Now"
    length_actions = ("click[12ft]", "click[180ft]", "click[buy now]")
    length_opened = update_episode_memory(
        length_memory,
        action="click[x000000003]",
        observation=length_page,
        available_actions=length_actions,
    )
    assert length_opened.webshop is not None
    assert length_opened.webshop.purchase_ready is False
    length_selected = update_episode_memory(
        length_opened,
        action="click[180ft]",
        observation=length_page,
        available_actions=length_actions,
    )
    assert length_selected.webshop is not None
    assert length_selected.webshop.selected_options == (("size", "180ft"),)
    assert length_selected.webshop.purchase_ready is True

    catalog_groups = (("color", ("yellow", "blue")), ("size", ("queen", "king")))
    ambiguous_page = (
        "Instruction [SEP] style [SEP] yellow [SEP] blue [SEP] queen [SEP] king [SEP] "
        "Price: $20.00 [SEP] Buy Now"
    )
    ambiguous_actions = (
        "click[yellow]",
        "click[blue]",
        "click[queen]",
        "click[king]",
        "click[buy now]",
    )
    multi_memory = initial_episode_memory(
        benchmark="webshop",
        task_text="Find a yellow king product.",
        task_type=None,
    )
    multi_opened = update_episode_memory(
        multi_memory,
        action="click[x000000004]",
        observation=ambiguous_page,
        available_actions=ambiguous_actions,
        webshop_option_groups=catalog_groups,
    )
    yellow_selected = update_episode_memory(
        multi_opened,
        action="click[yellow]",
        observation=ambiguous_page,
        available_actions=ambiguous_actions,
        webshop_option_groups=catalog_groups,
    )
    king_selected = update_episode_memory(
        yellow_selected,
        action="click[king]",
        observation=ambiguous_page,
        available_actions=ambiguous_actions,
        webshop_option_groups=catalog_groups,
    )
    assert king_selected.webshop is not None
    assert king_selected.webshop.selected_options == (("color", "yellow"), ("size", "king"))
    assert king_selected.webshop.purchase_ready is True


def test_webshop_step0_policy_uses_result_recovery_instead_of_closest_item_loop() -> None:
    profile = "webshop-native-react-memory-v18@1"
    decomposition = published_webshop_task_decomposition(profile)
    policy = webshop_step_operating_policy(
        profile,
        15,
        ("click[x000000001]", "click[next >]", "click[back to search]"),
        ("search[synthetic product alpha]", "click[x000000009]", "click[< Prev]"),
    )

    assert decomposition is not None
    assert "paginate" in decomposition
    assert policy is not None
    assert "click[next >]" in policy
    assert "visible over-budget price is final" in policy
    assert "closest current product" not in policy
