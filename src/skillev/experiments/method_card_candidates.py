"""Optional method cards for a NEW autonomous-TTB initial library.

These are public, synthetic procedures, not measured gains or task answers.
ALFWorld mechanics retain the existing public-interface procedures; new selection
descriptions distinguish usefulness from topical relevance. No teacher prompt,
automatic read, privileged tool or bonus accompanies these documents.
"""

from skillev.runtime import SkillDocument, SkillRequirement

from ._evolution_preflight_seed import _seed_document
from .native_procedure_candidates import native_procedure_seed_documents

PROFILE = "public-method-cards@5"


def method_card_seed_documents() -> tuple[SkillDocument, ...]:
    shared = (
        SkillRequirement(
            "public-inputs-only", "Use only public task inputs and actual tool observations."
        ),
        SkillRequirement(
            "advice-not-execution",
            "Reading does not execute, verify, grade or certify task completion.",
        ),
        SkillRequirement(
            "autonomous-applicability",
            "Adapt or disregard an inapplicable method; direct solutions remain allowed.",
            kind="evolvable-strategy",
        ),
    )
    definitions = (
        (
            "skill-code-context-contract-v5",
            "Preserve a public function's dependency contract",
            "Use when implementing a function inside supplied code with helpers, imports or "
            "constants. Needs the public signature and surrounding definitions. Not useful "
            "merely to recheck a self-contained implementation you already understand.",
            ("humaneval:task", "mbpp-plus:task"),
            "Build a dependency list from names referenced by the proposed implementation. "
            "Separate parameters/local bindings, builtins, public supplied helpers/imports, "
            "and newly introduced definitions. For every nonlocal name, identify the actual "
            "definition that will be available in the submitted program; a name mentioned only "
            "in prose supplies no definition. Preserve public helper semantics instead of "
            "renaming or replacing them inadvertently. For a fictional example, if a supplied "
            "normalize_item helper defines the intended normalization, implementing a caller "
            "with a different ad-hoc normalization changes the contract even if types match. "
            "Check the exact task's supported submission carrier, signature and return "
            "convention. Compare explicit examples and stated constraints, not hidden tests. "
            "This checklist does not assemble, execute or fix your program; use actual tool "
            "results if available, and submit directly when the contract is already clear.",
        ),
        (
            "skill-code-boundary-invariant-v5",
            "Derive boundary cases from an algorithm invariant",
            "Use when sequence boundaries, duplicates, mutation or loop state make an "
            "implementation uncertain. Needs the public function contract and candidate "
            "algorithm. Skip if the small direct implementation is already clear.",
            ("humaneval:task", "mbpp-plus:task"),
            "Write the relation that should hold before and after one loop iteration: what "
            "prefix has been processed, what the accumulator represents, and what remains. "
            "Derive initialization and termination from that relation. Then choose a few "
            "publicly valid boundary inputs that distinguish likely mistakes, rather than "
            "enumerating random examples: empty if permitted, singleton, equal elements if "
            "permitted, a boundary transition, and a case that distinguishes order from set "
            "membership. In a fictional stable grouping routine, two equal keys with distinct "
            "payloads reveal whether relative order was preserved; this is a technique, not "
            "a test for the current task. Track whether a check was mentally traced or truly "
            "executed. If an example contradicts the invariant, revise your algorithm and "
            "trace it again. Do not invent restrictions, unrequested input validation, or "
            "claims about hidden tests. The document supplies no execution result.",
        ),
        (
            "skill-relational-evidence-join-v5",
            "Resolve a relational question with typed evidence joins",
            "Use when several entities or evidence hops could fill the requested relation. "
            "Needs the question and any available passages or recalled facts; provides no "
            "lookup. A directly known unambiguous fact need not use this method.",
            ("hotpotqa:task", "triviaqa:task"),
            "Represent each candidate fact as subject, relation, object, and source status "
            "(quoted public evidence, inference, or uncertain recall). Identify the type of "
            "answer requested before joining facts. Join on the same entity, checking names, "
            "dates or qualifiers instead of relying on adjacent text. For a fictional query "
            "asking the birthplace of a work's editor, work-to-editor followed by "
            "editor-to-birthplace is the relevant chain; the author's birthplace answers a "
            "different relation. If a hop is missing, use only actual available tools or your "
            "own appropriately uncertain knowledge; reading this method did not search. "
            "Compare competing chains against the exact public question, then provide the "
            "requested answer through the task interface, without inventing citations.",
        ),
        (
            "skill-math-domain-cases-v5",
            "Preserve domains and exhaustive cases in a derivation",
            "Use when algebraic transformations, integer constraints, symmetry or case "
            "splits may introduce or lose solutions. Needs the stated problem and your "
            "derivation; not a solver or an answer source for routine arithmetic.",
            ("aime-2026:task",),
            "List each variable's stated domain and the conditions under which a planned "
            "transformation is reversible. Before dividing, separate the zero case; before "
            "squaring, remember that candidate roots require substitution in the original "
            "relation. Build disjoint cases whose union covers the stated domain, keeping "
            "boundary cases visible. A fictional equality x(x-1)=0 cannot be divided by x "
            "without separately retaining x=0. For counting, specify whether arrangements "
            "are ordered and whether symmetries have fixed points before dividing by a "
            "symmetry count. Verify candidate results against the original conditions, not "
            "only transformed equations. A token limit is not proof of completion; submit "
            "your best supported result through the actual answer interface. Reading has "
            "performed no derivation or verification on your behalf.",
        ),
        (
            "skill-conversation-response-map-v5",
            "Map a complex conversation to a bounded final reply",
            "Use when a multi-part conversation risks dropping a requested deliverable or "
            "confusing stated facts with assumptions. Needs the visible conversation and "
            "draft. This is response organization, not medical advice or a judge rubric.",
            ("healthbench:task",),
            "Separate what the user explicitly asks, relevant constraints they state, and "
            "uncertainties that affect the response. Map each requested deliverable to a "
            "part of the draft; remove promises to answer later when an actual answer is "
            "needed now. Distinguish information the user provided from assumptions or "
            "inferences. Present important limitations with the relevant claim instead of "
            "a detached generic disclaimer. Allocate the actual remaining output budget "
            "across the required reply, avoiding repeated introductions that consume it. "
            "For a fictional nonmedical planning request with two constraints, make both "
            "constraints explicit rather than answering only the first. This document adds "
            "no factual expertise, hidden rubric, safety verdict or required clinical "
            "recommendation. You remain responsible for a useful, truthful and appropriate "
            "reply and can disregard this organization method when unnecessary.",
        ),
    )
    documents = tuple(
        _seed_document(
            skill_id=identity,
            title=title,
            summary=summary,
            instructions=body,
            contexts=contexts,
            requirements=shared,
        )
        for identity, title, summary, contexts, body in definitions
    )
    alf = tuple(
        _seed_document(
            skill_id=doc.manifest.skill_id.replace("-v4", "-v5"),
            title=doc.title,
            summary=doc.summary + " Needs the public goal, observed state and admissible commands; "
            "skip when the next goal-directed operation is already known.",
            instructions=doc.instructions,
            contexts=doc.applicability.contexts,
            required_tools=doc.applicability.required_tools,
            requirements=shared,
        )
        for doc in native_procedure_seed_documents()
        if doc.manifest.skill_id.startswith("skill-alf-")
    )
    return documents + alf
