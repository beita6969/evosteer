"""Public, answer-free procedure candidates; not established skill benefits.

Authored from public interface meanings, not benchmark answers, failure tests or
IID scores. Select as an independent library axis; old seed IDs remain intact.
"""

from skillev.runtime import SkillDocument, SkillRequirement

PROFILE = "public-procedure-advice@3"


def procedure_seed_documents() -> tuple[SkillDocument, ...]:
    from ._evolution_preflight_seed import _seed_document

    definitions = (
        (
            "skill-observation-ledger-v3",
            "Optional object-and-state evidence ledger",
            "Useful when an interactive goal has object, state and destination requirements, "
            "or when repeated commands have not produced native completion. Requires the "
            "public goal and actual observations; returns a comparison procedure, not a verdict.",
            ("alfworld:task",),
            "Make a compact comparison with columns: exact goal object type; required state; "
            "destination/relation; currently observed object ID; supporting execution feedback. "
            "Copy the goal's category words without substituting similar everyday categories. "
            "Keep 'unknown' wherever observation has not established a state. An object's name "
            "does not establish its temperature or cleanliness. After an actual command, update "
            "only the supported cells; a proposed action changes no cell. Compare unsatisfied or "
            "unknown cells with the current admissible interface and choose an action yourself. "
            "If the native episode remains nonterminal despite your completion belief, revisit "
            "that comparison rather than treating another document read as an environment check. "
            "This document does not locate objects, execute commands, inspect hidden goals or "
            "certify completion. It is optional; no extra read or fixed action sequence is needed.",
        ),
        (
            "skill-public-code-contract-v3",
            "Optional public-code contract worksheet",
            "Useful for implementing a supplied function when public helpers, examples or "
            "return conventions matter. Requires only the public code and examples, not tests.",
            ("humaneval:task", "mbpp-plus:task"),
            "Record the callable name/signature, public imports/helpers/constants, input/return "
            "conventions and each explicit example's constraint. Separate supplied definitions "
            "from definitions your implementation introduces. Trace the proposed algorithm on "
            "public examples without inferring hidden tests; keep unprovided cases as assumptions. "
            "Check that the actual submitted source uses the declared code carrier and contains "
            "Python rather than explanatory pseudocode. A tool run is evidence only if it actually "
            "returned a result. This worksheet neither executes code nor repairs it; you choose "
            "the implementation and can submit directly without using this advice.",
        ),
        (
            "skill-evidence-claim-map-v3",
            "Optional evidence-to-claim map",
            "Useful for a factual question with competing entities or several evidence hops. "
            "Requires the public question and any supplied passages; offers no answer lookup.",
            ("hotpotqa:task", "triviaqa:task"),
            "Separate the entity the question asks for, the relation to establish, and the "
            "available evidence for each hop. Track whether a claim is stated in a supplied "
            "passage, inferred, or recalled without supplied evidence. Resolve which entity "
            "fills the requested relation before formatting the final answer; a nearby related "
            "name is not evidence that it answers the relation. Keep the source wording when "
            "appropriate, and distinguish uncertain recall from observed text. Do not claim "
            "that this document queried a knowledge base or checked an alias list. No evidence "
            "or answer is supplied by reading it, and its use is optional.",
        ),
        (
            "skill-deliverable-checklist-v3",
            "Optional reasoning-to-deliverable checklist",
            "Useful when substantial reasoning must become one final reply or mathematical "
            "answer. Requires the original request and your own draft; does not grade it.",
            ("healthbench:task", "aime-2026:task"),
            "List the user's explicit requested deliverables and pair each with content in "
            "your draft. For a conversation, retain the user's actual constraints, relevant "
            "uncertainty, clarification and safety information; avoid replacing a useful reply "
            "with a promise that another assistant will answer. For a mathematical request, "
            "separate the candidate result, conditions used and unresolved assumptions. Then "
            "form one submission within the declared action budget. A reasoning length stop "
            "means a budget ended, not that a proof or reply is complete. This checklist neither "
            "provides medical judgment nor verifies mathematics, and does not prohibit appropriate "
            "refusal, uncertainty or direct submission without further checks.",
        ),
    )
    return tuple(
        _seed_document(
            skill_id=identity,
            title=title,
            summary=summary,
            instructions=body,
            requirements=(
                SkillRequirement(
                    "public-interface-only",
                    "Use only the actual public interface; never hidden evaluator material.",
                ),
                SkillRequirement(
                    "advice-is-not-execution",
                    "Reading supplies advice, not task execution, verification "
                    "or a success verdict.",
                ),
                SkillRequirement(
                    "optional-procedure",
                    "The owner may adapt or disregard this procedure; no read quota.",
                    kind="evolvable-strategy",
                ),
            ),
            contexts=tuple(sorted(contexts)),
        )
        for identity, title, summary, contexts, body in definitions
    )
