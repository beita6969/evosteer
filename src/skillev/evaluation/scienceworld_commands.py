"""Versioned, task-independent ScienceWorld API facts, not an experiment policy.

Source: allenai/ScienceWorld's released action, instrument and input-parser
implementations. No task ID, object inventory, goal predicate or answer is an
input to this catalog. Legacy evaluation conditions retain their old reference.
"""

from dataclasses import dataclass, replace

LEGACY_COMMAND_PROFILE = "scienceworld-commands@1"
COMMAND_PROFILE = "scienceworld-commands@2"
TYPED_COMMAND_PROFILE = "scienceworld-commands-typed-selection@3"
SCOPED_TYPED_COMMAND_PROFILE = "scienceworld-commands-scoped-tool-help@4"
PERSISTENT_FOCUS_COMMAND_PROFILE = "scienceworld-commands-persistent-focus@5"
MATERIAL_CONTACT_COMMAND_PROFILE = "scienceworld-commands-material-contacts@6"
TYPED_COMMAND_PROFILES = frozenset(
    {
        TYPED_COMMAND_PROFILE,
        SCOPED_TYPED_COMMAND_PROFILE,
        PERSISTENT_FOCUS_COMMAND_PROFILE,
        MATERIAL_CONTACT_COMMAND_PROFILE,
    }
)
REFERENCE_PROFILE = "scienceworld-pending-choice@1"

# Literal API bindings, not object resolution, action selection or a skill.
TYPED_COMMANDS = {
    "inspect_object": "look at ",
    "select_task_object": "focus on ",
}


def native_functions(profile: str) -> tuple[str, ...]:
    if profile in TYPED_COMMAND_PROFILES:
        return ("act", *TYPED_COMMANDS)
    if profile not in {LEGACY_COMMAND_PROFILE, COMMAND_PROFILE}:
        raise ValueError("unknown ScienceWorld command profile")
    return ("act",)


def command_profile(task_semantic_guidance: str) -> str:
    if task_semantic_guidance == "public-task-semantics@14":
        return MATERIAL_CONTACT_COMMAND_PROFILE
    if task_semantic_guidance == "public-task-semantics@13":
        return PERSISTENT_FOCUS_COMMAND_PROFILE
    if task_semantic_guidance == "public-task-semantics@12":
        return SCOPED_TYPED_COMMAND_PROFILE
    if task_semantic_guidance in {"public-task-semantics@10", "public-task-semantics@11"}:
        return TYPED_COMMAND_PROFILE
    return (
        COMMAND_PROFILE
        if task_semantic_guidance
        in {
            "public-task-semantics@6",
            "public-task-semantics@7",
            "public-task-semantics@8",
            "public-task-semantics@9",
        }
        else LEGACY_COMMAND_PROFILE
    )


@dataclass(frozen=True, slots=True)
class ScienceWorldCommandSpec:
    syntax: str
    meaning: str
    reference: str
    task_selection: bool
    effects: str
    evidence: str
    version: str = COMMAND_PROFILE

    def help(self) -> str:
        return (
            f"{self.syntax}: {self.meaning} {self.reference} "
            f"Effects: {self.effects} Result: {self.evidence}"
        )


COMMANDS = (
    ScienceWorldCommandSpec(
        "focus on OBJ",
        "Designate OBJ as the current task object; this is not an inspection operation.",
        "Naming a container selects the container, not its contents.",
        True,
        "some tasks fail immediately on a wrong object or selection order. Follow the public "
        "task's requested order, including any required initial instrument/material focus; "
        "focus is not restricted to the final step.",
        "a focus confirmation proves selection only, not pickup, measurement, or an experiment.",
    ),
    ScienceWorldCommandSpec(
        "look around; look at OBJ; look in OBJ; inventory; read OBJ; task",
        "Observe surroundings, inspect an object or container, list carried objects, "
        "read a readable object, or display the task respectively.",
        "Keep the distinction between a named container and each explicitly visible content; "
        "unreported contents remain unknown.",
        False,
        "look around, inventory and task do not advance simulator time; other inspection "
        "commands such as look at do. Native goal checks can still occur. Every act call "
        "uses an evaluation interaction, including a time-free observation. "
        "The accompanying current_look/inventory/task_description fields are already supplied "
        "by that response, not extra actions.",
        "only the returned observations establish visible state; an old readout is not a "
        "new measurement of another target.",
    ),
    ScienceWorldCommandSpec(
        "pick up OBJ; put down OBJ; move OBJ to DEST; open OBJ; close OBJ; go LOCATION",
        "Change possession, placement, access, or location as specified by the command.",
        "A plant and its pot, a substance and its vessel, and a device and its terminals "
        "are distinct objects. Commands execute the literal referent, not the intent in prose. "
        "put down OBJ places it in the current room; to specify a destination, use "
        "move OBJ to DEST rather than appending a destination to put down. "
        "go LOCATION traverses an accessible connection, not a route through distant rooms.",
        True,
        "changes world state; placement into a task's classification container may commit "
        "a result and fail the task. A classification box is not a label-query tool.",
        "seeing or focusing an instrument does not carry it. A rejected command does not "
        "complete a required prerequisite.",
    ),
    ScienceWorldCommandSpec(
        "use TOOL on TARGET; activate OBJ; deactivate OBJ",
        "Apply an accessible instrument to a target, or switch an activatable device on/off.",
        "Use the instrument and target names separately, with native access/possession "
        "requirements confirmed by the response.",
        False,
        "instrument use or activation may change the instrument or world; native time applies.",
        "use thermometer on TARGET reports that target's current temperature, not a "
        "melting-point property; simply inspecting the thermometer reports its own temperature. "
        "An active stopwatch accumulates simulator ticks; a displayed "
        "total is not automatically the elapsed duration of the most recent experiment. "
        "activate stopwatch starts/resumes counting; an inactive tick resets its running "
        "counter while retaining the displayed reading. look at stopwatch displays that reading.",
    ),
    ScienceWorldCommandSpec(
        "connect TERMINAL to TERMINAL; disconnect OBJ",
        "Connect named electrical endpoints or remove a connection using the native interface.",
        "Use the endpoint names exposed by object observations, such as DEVICE anode, "
        "DEVICE cathode, or WIRE terminal 1/2, not an automatically selected contact. "
        "disconnect OBJ disconnects the whole named object, not a pair of terminals.",
        False,
        "changes connections; devices and the physical simulation update on native ticks.",
        "a connection acknowledgement proves that connection, not a powered complete circuit "
        "or any material classification.",
    ),
    ScienceWorldCommandSpec(
        "wait; wait1; mix OBJ; pour OBJ in OBJ; dunk OBJ in OBJ; eat OBJ; flush OBJ",
        "Use the released simulator's waiting or physical manipulation commands.",
        "Refer only to objects described in public observations; these templates are not "
        "a currently valid or recommended action list.",
        False,
        "may advance time and change physical state according to the native action.",
        "the returned state confirms what happened, not whether an unperformed experiment "
        "would succeed.",
    ),
)


def commands_for_profile(profile: str) -> tuple[ScienceWorldCommandSpec, ...]:
    if profile == MATERIAL_CONTACT_COMMAND_PROFILE:
        previous = commands_for_profile(PERSISTENT_FOCUS_COMMAND_PROFILE)
        # ActionConnectElectrical.getTerminal distinguishes explicit components
        # from generic material objects. The latter's contacts need not appear
        # in their short description; selection remains in the native simulator.
        electrical = replace(
            previous[4],
            reference=(
                "Polarized devices require DEVICE anode or DEVICE cathode; wires require "
                "WIRE terminal 1 or WIRE terminal 2. Generic material objects can also have "
                "terminal 1 and terminal 2 even when look at shows only their name. For "
                "such objects, connect accepts the object name and the native simulator "
                "uses an unconnected contact; an explicit OBJ terminal 1/2 is also allowed. "
                "The acknowledgement names the contact actually used; this harness does "
                "not infer or substitute contacts. Objects without available contacts are "
                "rejected. disconnect OBJ disconnects the whole named object."
            ),
            version=profile,
        )
        return (*previous[:4], electrical, *previous[5:])
    if profile != PERSISTENT_FOCUS_COMMAND_PROFILE:
        return COMMANDS
    # ActionFocus.runAction clears the monitor before adding the chosen object.
    # Confirmed against the deployed simulator by fixed saved-command diagnosis.
    # This describes the API, not which object any task requires.
    focus = replace(
        COMMANDS[0],
        meaning=COMMANDS[0].meaning
        + " Within an episode, successful focus replaces the previous focus; it does not "
        "add a second object. The selected object remains monitored across later commands "
        "until focus changes or the task is reset. Later goal checks may use that current "
        "selection.",
        version=profile,
    )
    return (focus, *COMMANDS[1:])


def task_contract() -> str:
    return (
        "Complete the public environment goal by choosing every command yourself. "
        "Task-object selection is different from inspection and may be irreversible. "
        "Use the command reference and the public task's required order; only actual "
        "execution feedback establishes completed prerequisites. Terminal does not itself "
        "mean success."
    )


def typed_description(name: str, profile: str = TYPED_COMMAND_PROFILE) -> str:
    spec = commands_for_profile(profile)[1 if name == "inspect_object" else 0]
    if (
        profile
        in {
            SCOPED_TYPED_COMMAND_PROFILE,
            PERSISTENT_FOCUS_COMMAND_PROFILE,
            MATERIAL_CONTACT_COMMAND_PROFILE,
        }
        and name == "inspect_object"
    ):
        # The old family-wide help misleadingly advertised look around/task as
        # operations of this object-only function. Its literal binding is unchanged.
        return (
            "Inspect one named object: execute exactly look at TARGET with your literal "
            "target string. The target is an object name, not a command or a viewing mode. "
            'For the room view call act(command="look around"); for carried objects call '
            'act(command="inventory"); for the task call act(command="task"). '
            + spec.reference
            + " This look-at operation uses one environment action and native simulator "
            "time/goal checks. It neither selects the task object nor picks it up. "
            + spec.evidence
            + " No target is inferred, replaced or selected by the framework."
        )
    return (
        f"Execute exactly {TYPED_COMMANDS[name]}TARGET, with your literal target string. "
        + spec.help()
        + " No target is inferred, replaced or selected by the framework."
    )


def act_description(profile: str = COMMAND_PROFILE) -> str:
    return (
        f"ScienceWorld native command interface ({profile}). "
        + commands_for_profile(profile)[0].help()
        + " Other command and instrument semantics are in the shared system reference."
    )


def command_reference(profile: str = COMMAND_PROFILE) -> str:
    return (
        f" ScienceWorld public command reference ({profile}). "
        "act(command) sends one literal command; prose does not execute actions. "
        "You decide which command and object to use.\n"
        + (
            "The separate inspect_object(target) and select_task_object(target) functions "
            "send exactly 'look at TARGET' and 'focus on TARGET' respectively. Their target "
            "argument is your literal object name; choosing a pot still refers to that pot, "
            "not its contents. Both consume one native action and retain native time and "
            "goal checks. Free-form act remains available for every command, including "
            "numbered disambiguation replies.\n"
            if profile in TYPED_COMMAND_PROFILES
            else ""
        )
        + "\n".join(item.help() for item in commands_for_profile(profile))
        + "\nObject ambiguity: the displayed numbered choices belong only to the pending "
        "command in that response's public revision. Send your chosen number, not the "
        "annotated command text. Do not reuse it after that choice is consumed or cancelled. "
        "No option is selected for you. Invalid commands still go to the native simulator. "
        "The reset task command is not an evaluation retry: no new episode or replacement "
        "answer is granted by this harness."
    )
