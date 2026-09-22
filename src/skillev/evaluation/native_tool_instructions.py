"""Public simulator API reference, separate from observations and text skills.

ALFWorld command meanings and preconditions follow its released alfred.twl2
interface and alfred.pddl actions, not a task-specific action sequence.
"""

from __future__ import annotations

from .scienceworld_commands import (
    COMMAND_PROFILE,
    LEGACY_COMMAND_PROFILE,
    TYPED_COMMAND_PROFILES,
    command_reference,
)


def native_tool_instructions(
    mode: str,
    *,
    native_functions: bool = False,
    scienceworld_profile: str = LEGACY_COMMAND_PROFILE,
) -> str:
    if native_functions:
        return (
            "Solve the user's task using public observations and the available functions. "
            "Function definitions describe their arguments; results provide the current "
            "observation and available commands. A confirmed execution means a command ran, "
            "not that the task succeeded. The simulator reports when the episode ends."
            + native_tool_reference(mode, scienceworld_profile=scienceworld_profile)
        )
    invocation = (
        "To invoke an environment tool, submit one native API call on its own final line; "
        "an Action: label is optional. "
        if mode == "webshop"
        else "To invoke an environment tool, submit Action: followed by one native action. "
    )
    transport = (
        "Solve the user's task using public observations and available tools. "
        + invocation
        + "A JSON tool call is also accepted: "
        '{"kind":"tool","resource_id":"'
        + mode
        + '","name":"tool name","arguments":{"argument name":"your value"}}. '
        "The current public action menu contains the commands available in the current state. "
        "An acknowledged execution includes the observation returned by the environment, "
        "even when the interface supplies no separate action-validity flag. "
        "A confirmed execution says that a command ran, not that the whole task succeeded. "
        "An active task status means the environment has not ended the episode. "
        "Only environment results confirm what happened in the simulator."
    )
    return transport + native_tool_reference(mode, scienceworld_profile=scienceworld_profile)


def native_tool_reference(mode: str, *, scienceworld_profile: str = LEGACY_COMMAND_PROFILE) -> str:
    """Public native API semantics, without a task-specific action policy."""
    if mode == "scienceworld":
        if scienceworld_profile in {COMMAND_PROFILE, *TYPED_COMMAND_PROFILES}:
            return command_reference(scienceworld_profile)
        if scienceworld_profile != LEGACY_COMMAND_PROFILE:
            raise ValueError("unknown ScienceWorld command reference")
        # The released simulator's get_possible_actions() API templates, not
        # get_valid_action_object_combinations(), object lists or gold paths:
        # https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py
        return (
            " This is the ScienceWorld text science simulator. act accepts one complete "
            "native command as its command argument. You may reason and use the available "
            "tools as needed. Calling act sends the command to the simulator; discussing "
            "an action does not execute it. The returned observation describes its result. "
            "Public command syntax (replace OBJ with "
            "the object's name from observations): activate OBJ; close OBJ; connect OBJ "
            "to OBJ; deactivate OBJ; disconnect OBJ; dunk OBJ in OBJ; eat OBJ; flush OBJ; "
            "focus on OBJ; go OBJ; inventory; look around; look at OBJ; look in OBJ; "
            "mix OBJ; move OBJ to OBJ; open OBJ; pick up OBJ; pour OBJ in OBJ; put down OBJ; "
            "read OBJ; reset task; task; use OBJ on OBJ; wait; wait1. "
            "Focus identifies an object for the task; looking at it, carrying it and "
            "moving it are different operations. move OBJ to OBJ places an object in "
            "a destination; use OBJ on OBJ applies a tool to a target. If the simulator "
            "asks which object you mean, the displayed choice number is a native command. "
            "In that disambiguation state, send only the chosen number as the act argument, "
            "not the annotated action description with parenthesized locations. "
            "An explicit unknown-action response reports that the command was rejected. "
            "Each act call, including observation, uses one interaction from this "
            "evaluation's horizon. These are syntax templates, not a list of currently "
            "valid or recommended actions. Invalid commands are handled by the simulator. "
            "Follow the actual goal shown at reset."
        )
    if mode == "webshop":
        return (
            " This is a simulated shopping task, with no real purchases or payments. "
            "search[query] submits a product query from the current page, "
            "including pages without a search form; "
            "its argument is query. click[target] activates a displayed product link, "
            "option or page control; its argument is target. Navigation changes the page "
            "and its available controls. The native click API lowercases the target "
            "before looking up its displayed control. 点击 means click and 搜索 means search. "
            "The displayed Buy Now control commits the current item and selected options and "
            "ends the episode; a chat recommendation does not."
        )
    if mode == "alfworld":
        return (
            " This is a text household simulator. act accepts a complete command argument. "
            "The zero-argument look, inventory and help functions execute those same native "
            "commands; they also consume an environment action. "
            "Public command reference (use the concrete spelling and identifiers in the menu): "
            "look displays the current surroundings; inventory lists held objects; examine "
            "inspects a visible object or receptacle; go to changes your location; open and "
            "close change receptacle access; take transfers an object from a receptacle into "
            "your inventory; move or put transfers a held object to a receptacle; "
            "heat, clean and cool change an object's corresponding state using the named "
            "receptacle; use toggles an operable object; slice cuts an object with the named "
            "sharp object. A placement command only places an object; it does not itself "
            "heat, clean or cool it. "
            "The menu lists actions executable at the current location and state, not all "
            "commands that can ever become available. Take requires empty hands, an accessible "
            "receptacle at your location, and a pickupable object; only one object is held at "
            "a time. Move requires a held object and an accessible, compatible receptacle "
            "at your location. Heat, clean and cool require an eligible held object and the "
            "corresponding appliance or sinkbasin at your location."
        )
    raise ValueError("native tool reference is defined only for supported simulators")
