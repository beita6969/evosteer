"""Concrete public-interface procedures, still candidates rather than proven gains.

The ALFWorld mechanics below are paraphrased from its public action grammar and
generic domain operators, not game files, expert plans or benchmark answers:
https://github.com/alfworld/alfworld/blob/master/alfworld/data/alfred.twl2
https://github.com/alfworld/alfworld/blob/master/alfworld/data/alfred.pddl
Do not confuse checking those mechanics with measuring model-level usefulness.
"""

from skillev.runtime import SkillDocument, SkillRequirement

from ._evolution_preflight_seed import _seed_document
from .advisory_skill_candidates import procedure_seed_documents

PROFILE = "public-native-procedures@4"


def native_procedure_seed_documents() -> tuple[SkillDocument, ...]:
    """Keep unchanged non-ALF candidates; give revised ALF procedures new IDs."""
    shared = (
        SkillRequirement("public-interface-only", "Use only the actual public interface."),
        SkillRequirement(
            "advice-is-not-execution",
            "Reading supplies a procedure, not actions, observed state or a success verdict.",
        ),
        SkillRequirement(
            "optional-procedure",
            "Adapt or disregard this procedure when inappropriate; there is no read quota.",
            kind="evolvable-strategy",
        ),
    )
    alf = (
        _seed_document(
            skill_id="skill-alf-native-state-transitions-v4",
            title="ALFWorld appliance state transitions",
            summary=(
                "When a goal requires a clean, heated or cooled object: native command "
                "semantics, carried-object prerequisites and observation-based state tracking. "
                "Consult before planning appliance operations if these mechanics are uncertain."
            ),
            instructions=(
                "In TextWorld ALFWorld, placing an object inside an appliance and taking it "
                "out is not itself a clean, heat or cool operation. The public interface has "
                "distinct native commands: clean OBJECT with SINKBASIN; heat OBJECT with "
                "MICROWAVE; cool OBJECT with FRIDGE. These are templates, not literal IDs. "
                "Choose the exact currently admissible command for the observed object and "
                "receptacle; do not invent a command when it is absent. The object must be "
                "carried and the agent must be at the corresponding receptacle for these "
                "operations. If it was deposited there, use the public take command to carry "
                "it again, then check the newly returned admissible commands. Use a successful "
                "state-change response, or an actual examine response, as evidence of the "
                "new state. Ordinary move/take responses establish location or possession, "
                "not temperature or cleanliness. Heating and cooling replace the opposite "
                "temperature state; preserve the state required by the goal before final "
                "placement. Track goal object category, observed ID, required state and "
                "destination separately. Similar object names are not interchangeable. "
                "Continue with native feedback and the actual terminal signal; rereading this "
                "document executes and verifies nothing. It gives no object locations, "
                "instance-specific commands or hidden goal information."
            ),
            requirements=shared,
            contexts=("alfworld:task",),
            required_tools=("act",),
        ),
        _seed_document(
            skill_id="skill-alf-search-frontier-v4",
            title="ALFWorld search frontier and remaining-work budget",
            summary=(
                "When the target has not been located: maintain searched versus merely "
                "visited places, compare search costs and reserve actions for the goal's "
                "remaining transformations and placement. No location lookup is supplied."
            ),
            instructions=(
                "Keep a small frontier of receptacles from actual observations: unseen; "
                "visited but closed; contents observed without the target; target observed. "
                "Going to a closed container does not reveal its contents. Opening it can "
                "do so; use the returned contents rather than marking it searched merely "
                "because you visited it. Visiting an open surface often exposes contents "
                "without an extra open action. Compare these observed costs before spending "
                "the whole budget opening a long numbered sequence. Use task-relevant "
                "priors only as revisable guesses, never as claims about where an object is. "
                "Avoid revisiting an already observed empty place unless new evidence "
                "justifies it. On finding the exact target category, keep its observed ID "
                "and location and switch from search to the required manipulation. Reserve "
                "the remaining public actions needed to take it, reach any required state "
                "change location, perform the native operation, reach the destination and "
                "place it. This is a planning estimate, not permission to truncate a "
                "trajectory or skip task requirements. A short known next action may be "
                "more useful than repeatedly narrating the full search history. Reading "
                "this frontier procedure does not inspect a receptacle or find any object."
            ),
            requirements=shared,
            contexts=("alfworld:task",),
            required_tools=("act",),
        ),
    )
    return (
        tuple(
            doc
            for doc in procedure_seed_documents()
            if doc.manifest.skill_id != "skill-observation-ledger-v3"
        )
        + alf
    )
