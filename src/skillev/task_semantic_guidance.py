"""Versioned public task meaning shared by training and isolated evaluation.

Only a domain, a declared public-input profile, and a public reasoning switch
enter this catalog. Task text, references, rubrics and scores are not inputs.
Carrier-specific submission instructions remain separate from task meaning.
"""

LEGACY_TASK_SEMANTICS = "legacy"
PUBLIC_TASK_SEMANTICS_V1 = "public-task-semantics@1"
PUBLIC_TASK_SEMANTICS_V2 = "public-task-semantics@2"
PUBLIC_TASK_SEMANTICS_V3 = "public-task-semantics@3"
PUBLIC_TASK_SEMANTICS_V4 = "public-task-semantics@4"
PUBLIC_TASK_SEMANTICS_V5 = "public-task-semantics@5"
PUBLIC_TASK_SEMANTICS_V6 = "public-task-semantics@6"
PUBLIC_TASK_SEMANTICS_V7 = "public-task-semantics@7"
PUBLIC_TASK_SEMANTICS_V8 = "public-task-semantics@8"
PUBLIC_TASK_SEMANTICS_V9 = "public-task-semantics@9"
PUBLIC_TASK_SEMANTICS_V10 = "public-task-semantics@10"
PUBLIC_TASK_SEMANTICS_V11 = "public-task-semantics@11"
PUBLIC_TASK_SEMANTICS_V12 = "public-task-semantics@12"
PUBLIC_TASK_SEMANTICS_V13 = "public-task-semantics@13"
PUBLIC_TASK_SEMANTICS_V14 = "public-task-semantics@14"
TRAINING_PUBLIC_INPUT = "training-public-source-bridge@1"
RELEASED_PUBLIC_INPUT = "released-iid-source@1"
RELEASED_OOD_INPUT = "released-ood-source@1"


def validate_task_semantic_guidance(version: str) -> None:
    if version not in {
        LEGACY_TASK_SEMANTICS,
        PUBLIC_TASK_SEMANTICS_V1,
        PUBLIC_TASK_SEMANTICS_V2,
        PUBLIC_TASK_SEMANTICS_V3,
        PUBLIC_TASK_SEMANTICS_V4,
        PUBLIC_TASK_SEMANTICS_V5,
        PUBLIC_TASK_SEMANTICS_V6,
        PUBLIC_TASK_SEMANTICS_V7,
        PUBLIC_TASK_SEMANTICS_V8,
        PUBLIC_TASK_SEMANTICS_V9,
        PUBLIC_TASK_SEMANTICS_V10,
        PUBLIC_TASK_SEMANTICS_V11,
        PUBLIC_TASK_SEMANTICS_V12,
        PUBLIC_TASK_SEMANTICS_V13,
        PUBLIC_TASK_SEMANTICS_V14,
    }:
        raise ValueError("unsupported public task semantic guidance")


def public_task_semantics(
    benchmark: str,
    *,
    input_profile: str,
    hotpot_deliberation: bool = False,
    version: str = PUBLIC_TASK_SEMANTICS_V1,
) -> str:
    """Public task meaning and declared answer style, without private task data."""
    validate_task_semantic_guidance(version)
    # V12 scopes object-tool help; V13 explains persistent focus; V14 documents
    # material contacts. Task meaning is unchanged; command profiles are distinct.
    if version in {
        PUBLIC_TASK_SEMANTICS_V12,
        PUBLIC_TASK_SEMANTICS_V13,
        PUBLIC_TASK_SEMANTICS_V14,
    }:
        version = PUBLIC_TASK_SEMANTICS_V11
    if version == LEGACY_TASK_SEMANTICS:
        raise ValueError("legacy instructions do not use the shared semantic catalog")
    from skillev.evaluation.corpus_search import INPUT_PROFILE
    from skillev.evaluation.input_metric_contracts import CONTRACTS, OOD_BENCHMARKS

    if benchmark not in CONTRACTS or input_profile not in CONTRACTS[benchmark].input_profiles:
        raise ValueError("public task semantics require a declared supported input profile")
    if benchmark in OOD_BENCHMARKS and version not in {
        PUBLIC_TASK_SEMANTICS_V5,
        PUBLIC_TASK_SEMANTICS_V6,
        PUBLIC_TASK_SEMANTICS_V7,
        PUBLIC_TASK_SEMANTICS_V8,
        PUBLIC_TASK_SEMANTICS_V9,
        PUBLIC_TASK_SEMANTICS_V10,
        PUBLIC_TASK_SEMANTICS_V11,
    }:
        raise ValueError("OOD shared semantics require an OOD-capable catalog")
    if type(hotpot_deliberation) is not bool:
        raise TypeError("Hotpot deliberation must be boolean")
    instructions = {
        "hotpotqa": (
            "Answer the question using the supplied public passages. "
            "For a factoid question, return the answer span using the passage's "
            "wording. For a yes/no question, return yes or no."
        ),
        "triviaqa": (
            "Answer the question from the supplied public task. "
            if input_profile == TRAINING_PUBLIC_INPUT
            else "Answer the question; the supplied reading context is available as evidence. "
        )
        + "Return the short answer itself, rather than a sentence restating the question.",
        "aime-2026": (
            "Solve the problem and state your final integer from 0 through 999. "
            "A boxed integer is also accepted."
        ),
        "healthbench": (
            "You are a helpful assistant. Write the assistant's next reply to the "
            "user's latest message using the preceding dialogue as context. "
            "This is a conversation, not a short-answer exam: your reply can "
            "provide information, explain uncertainty, or ask for clarification "
            "as appropriate."
        ),
        "humaneval": (
            "Implement the requested function. Return executable Python source, "
            "preserving the supplied function signature and indentation."
        ),
        "mbpp-plus": (
            "Implement the requested Python function using the function name and "
            "calling convention shown in the public examples. The examples are "
            "part of the specification, including the expected return values. "
            "Return executable Python source."
        ),
        "alfworld": (
            "Follow the task stated by the current environment. Select one admissible "
            "command from the current public action surface using the latest public "
            "observation. The environment determines task completion."
        ),
        "musique": "Answer the question using all supplied public passages as evidence.",
        "nq-open": (
            "Answer the question using your knowledge and the available frozen corpus search."
            if input_profile == INPUT_PROFILE
            else "Answer the question from your own knowledge; this is a closed-book task."
        ),
        "omni-math": (
            "Solve the mathematical problem. Give your complete final answer, including a proof "
            "when requested. Mathematical explanation is part of your response."
        ),
        "math-hard": (
            "Solve the mathematical problem. State your final mathematical answer, "
            "preferably in a single \\boxed{...}; explanation may precede it. "
            "Only your final answer is submitted for equivalence scoring."
        ),
        "gpqa-diamond-bioorganic": (
            "Answer the scientific multiple-choice question using the supplied option order. "
            "State one final A, B, C or D choice; you may explain your reasoning."
        ),
        "livemedbench": (
            "Respond to the patient's request using the supplied narrative and core request. "
            "Provide the full appropriate clinical response, including explanation and uncertainty."
        ),
        "scienceworld": (
            "Complete the goal in the current public environment. You choose every native command; "
            "use actual execution observations to determine the next action. The environment "
            "determines termination, which does not by itself mean task success."
        ),
        "livecodebench": (
            "Write a complete Python solution to the supplied problem, following its public "
            "input/output or callable interface and starter code when present."
        ),
        "apps-introductory": (
            "Write a complete Python solution to the supplied problem, following its public "
            "input/output or callable interface and starter code when present."
        ),
    }
    try:
        instruction = instructions[benchmark]
    except KeyError as error:
        raise ValueError("benchmark has no shared public task semantics") from error
    if version in {
        PUBLIC_TASK_SEMANTICS_V9,
        PUBLIC_TASK_SEMANTICS_V10,
        PUBLIC_TASK_SEMANTICS_V11,
    } and (benchmark == "musique" or (benchmark == "nq-open" and input_profile == INPUT_PROFILE)):
        # A separately declared evidence-wording condition, not answer extraction
        # by the framework. The owner still chooses and submits the answer.
        instruction += (
            " When the evidence contains your answer, use a concise answer span "
            "in its original wording. Include qualifiers needed to answer the question, "
            "but leave background information and explanations outside the final answer."
        )
    if benchmark == "scienceworld" and version in {
        PUBLIC_TASK_SEMANTICS_V6,
        PUBLIC_TASK_SEMANTICS_V7,
        PUBLIC_TASK_SEMANTICS_V8,
        PUBLIC_TASK_SEMANTICS_V9,
        PUBLIC_TASK_SEMANTICS_V10,
        PUBLIC_TASK_SEMANTICS_V11,
    }:
        from skillev.evaluation.scienceworld_commands import task_contract

        instruction = task_contract()
        if version == PUBLIC_TASK_SEMANTICS_V11:
            from skillev.evaluation.scienceworld_equipment import equipment_reference

            instruction += equipment_reference()
    if benchmark == "alfworld" and version in {
        PUBLIC_TASK_SEMANTICS_V2,
        PUBLIC_TASK_SEMANTICS_V3,
        PUBLIC_TASK_SEMANTICS_V4,
        PUBLIC_TASK_SEMANTICS_V5,
        PUBLIC_TASK_SEMANTICS_V6,
        PUBLIC_TASK_SEMANTICS_V7,
        PUBLIC_TASK_SEMANTICS_V8,
        PUBLIC_TASK_SEMANTICS_V9,
        PUBLIC_TASK_SEMANTICS_V10,
        PUBLIC_TASK_SEMANTICS_V11,
    }:
        # Public command meanings, not demonstrations or a task-specific solution.
        # The official ALFWorld interface illustrates take/use for lighting tasks:
        # https://alfworld.github.io/
        instruction += (
            " The command list describes what is currently executable, not what has "
            "already happened. 'go to' navigates; 'open' and 'close' operate a nearby "
            "container. 'take ... from ...' picks up an object and 'move ... to ...' "
            "places a held object. 'inventory' reports held objects. 'look' reports "
            "the current view; it does not pick up objects or use appliances. "
            "'clean', 'heat', and 'cool' perform the named operation using the "
            "specified appliance. 'use' operates an available object: using a lamp "
            "while holding an object is how that object is examined under its light. "
            "Use the exact object identifiers and commands supplied by the environment. "
            "Only actual execution feedback changes the known state; an imagined "
            "action, observation, or success in reasoning does not."
        )
        if version in {
            PUBLIC_TASK_SEMANTICS_V3,
            PUBLIC_TASK_SEMANTICS_V4,
            PUBLIC_TASK_SEMANTICS_V5,
            PUBLIC_TASK_SEMANTICS_V6,
            PUBLIC_TASK_SEMANTICS_V7,
            PUBLIC_TASK_SEMANTICS_V8,
            PUBLIC_TASK_SEMANTICS_V9,
            PUBLIC_TASK_SEMANTICS_V10,
            PUBLIC_TASK_SEMANTICS_V11,
        }:
            # Public domain predicates and preconditions, not private episode goals:
            # https://github.com/alfworld/alfworld/blob/master/alfworld/data/alfred.pddl
            instruction += (
                " Cleaned, heated, and cooled are object states, not different object "
                "names. An unchanged object ID, or the absence of a word such as "
                "'dirty' in its name, does not establish the required state; use "
                "actual operation feedback as evidence. The admissible list is "
                "local to the current state. Cleaning, heating, and cooling "
                "require holding the object and being near the appropriate "
                "appliance. An operation absent from the current list can "
                "become available after its prerequisites are met; absence does "
                "not mean the task's state requirement is already satisfied."
            )
        if version in {
            PUBLIC_TASK_SEMANTICS_V4,
            PUBLIC_TASK_SEMANTICS_V5,
            PUBLIC_TASK_SEMANTICS_V6,
            PUBLIC_TASK_SEMANTICS_V7,
            PUBLIC_TASK_SEMANTICS_V8,
            PUBLIC_TASK_SEMANTICS_V9,
            PUBLIC_TASK_SEMANTICS_V10,
            PUBLIC_TASK_SEMANTICS_V11,
        }:
            # Public objectType and CleanObject/HeatObject/CoolObject semantics
            # in alfred.pddl. No episode, object ID, location, or verdict enters
            # this explanation; the owner still chooses every environment action.
            instruction += (
                " Different object types named by the simulator are distinct categories, "
                "even when everyday language treats them as similar. An object of a "
                "different listed type is not a substitute for the requested type. "
                "The operations have separate meanings: 'clean' with a sinkbasin "
                "sets the cleaned state; 'heat' with a microwave sets the hot state; "
                "'cool' with a fridge sets the cool state. Cleaning does not set "
                "the hot or cool state. Heating removes the cool state, and cooling "
                "removes the hot state. These meanings do not prescribe an action sequence."
            )
    if benchmark == "hotpotqa" and hotpot_deliberation:
        from skillev.benchmarks.reasoning_guidance import HOTPOT_DELIBERATION

        instruction = HOTPOT_DELIBERATION + instruction
    return instruction


def phase_deliverable(benchmark: str | None, *, version: str) -> str | None:
    """Opt-in controller handoff, selected by public domain, never by an answer.

    V7 is a development candidate extending the seven-domain V5 architecture.
    It leaves task content and the callable interface unchanged. Other versions
    keep their original phase messages, including V6's separate OOD condition.
    """
    if version in {
        PUBLIC_TASK_SEMANTICS_V12,
        PUBLIC_TASK_SEMANTICS_V13,
        PUBLIC_TASK_SEMANTICS_V14,
    }:
        version = PUBLIC_TASK_SEMANTICS_V11
    if (
        version
        not in {
            PUBLIC_TASK_SEMANTICS_V7,
            PUBLIC_TASK_SEMANTICS_V8,
            PUBLIC_TASK_SEMANTICS_V9,
            PUBLIC_TASK_SEMANTICS_V10,
            PUBLIC_TASK_SEMANTICS_V11,
        }
        or benchmark is None
    ):
        return None
    return {
        "healthbench": "conversation-reply",
        "aime-2026": "integer-answer",
        "alfworld": "environment-action",
    }.get(benchmark)


def evaluation_submission_instruction(benchmark: str) -> str:
    """Evaluation's textual submission carrier, not the training answer parameter."""
    if benchmark in {"hotpotqa", "triviaqa", "musique", "nq-open"}:
        return (
            "The submitted answer is a short answer phrase, not a sentence restating the "
            "question or an explanation. If you explain, put the short answer alone on a "
            "separate Final answer: line and keep the explanation outside that line."
        )
    if benchmark in {"livecodebench", "apps-introductory"}:
        return (
            "Bare Python or a fenced program is accepted. If several code blocks or final code "
            "sections are present, the last is your submitted program."
        )
    if benchmark == "gpqa-diamond-bioorganic":
        return (
            "After any explanation, put the single submitted choice on a separate "
            "Final answer: A line, replacing A with exactly one of A, B, C or D."
        )
    return ""


TRAINING_SUBMISSION_INSTRUCTION = "Put the final response in the answer parameter."
