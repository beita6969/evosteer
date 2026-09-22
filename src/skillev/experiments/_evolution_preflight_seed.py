"""Public synthetic seed library for the result-blind evolution preflight."""

from __future__ import annotations

from skillev.contracts import JsonValue, stable_hash
from skillev.runtime import (
    SkillApplicability,
    SkillDocument,
    SkillManifest,
    SkillRequirement,
)


def _seed_document(
    *,
    skill_id: str,
    title: str,
    summary: str,
    instructions: str,
    requirements: tuple[SkillRequirement, ...],
    task_families: tuple[str, ...] = ("*",),
    contexts: tuple[str, ...] = ("*",),
    required_tools: tuple[str, ...] = (),
) -> SkillDocument:
    applicability = SkillApplicability(
        task_families=task_families,
        contexts=contexts,
        required_tools=required_tools,
        excluded_contexts=(),
    )
    content: dict[str, JsonValue] = {
        "applicability": applicability.to_value(),
        "instructions": instructions,
        "requirements": [item.to_value() for item in requirements],
        "summary": summary,
        "title": title,
    }
    return SkillDocument(
        manifest=SkillManifest(
            skill_id=skill_id,
            version="1",
            content_hash=stable_hash(content),
            input_schema_id="skillev-public-task@1",
            output_schema_id="skillev-structured-action@1",
            license_id="CC0-1.0",
            provenance_hash=stable_hash(
                {
                    "kind": "public-synthetic-formal-seed",
                    "skill_id": skill_id,
                    "version": 1,
                }
            ),
        ),
        title=title,
        summary=summary,
        instructions=instructions,
        applicability=applicability,
        requirements=requirements,
    )


def planned_seed_documents(profile: str = "public-advisory@2") -> tuple[SkillDocument, ...]:
    """Return advisory public seeds for a fresh training library, not task solvers.

    Existing runs retain their stored documents. These fresh seeds do not impose
    a fixed action sequence, require extra calls, or redefine the submission API.
    """

    if profile == "public-method-cards@5":
        from .method_card_candidates import method_card_seed_documents

        return method_card_seed_documents()
    if profile == "public-native-procedures@4":
        from .native_procedure_candidates import native_procedure_seed_documents

        return native_procedure_seed_documents()
    if profile == "public-procedure-advice@3":
        from .advisory_skill_candidates import procedure_seed_documents

        return procedure_seed_documents()
    if profile != "public-advisory@2":
        raise ValueError("unknown initial skill library profile")

    return (
        _seed_document(
            skill_id="skill-plan-decompose-v2",
            title="Plan and decompose",
            summary="Optional planning and decomposition for complex public tasks.",
            instructions=(
                "Optional advice: break a complex task into manageable parts when useful. "
                "Use public observations to revise your approach. As the sole task agent, "
                "you can solve a simple task directly, change the order, "
                "or disregard this advice; no written plan or extra call is required."
            ),
            requirements=(
                SkillRequirement(
                    requirement_id="plan-public-goal",
                    text="Keep the user's public goal in view.",
                    kind="evolvable-strategy",
                ),
                SkillRequirement(
                    requirement_id="plan-ordered-actions",
                    text="Offer revisable planning advice, not a mandatory action sequence.",
                    kind="evolvable-strategy",
                ),
                SkillRequirement(
                    requirement_id="plan-observation-update",
                    text="Distinguish public observations from planning assumptions.",
                ),
            ),
        ),
        _seed_document(
            skill_id="skill-tool-deliberation-v2",
            title="Deliberate tool use",
            summary="Choose and sequence available public tools without inventing capabilities.",
            instructions=(
                "Optional advice: use tools when their information or effects help with the "
                "task. The public API describes each tool and its arguments. Use returned "
                "observations to decide what to do next, and distinguish a proposed action "
                "from an executed one. You may retry, try another approach, or answer directly "
                "as appropriate; this advice does not require or forbid an available action."
            ),
            requirements=(
                SkillRequirement(
                    requirement_id="tool-declared-capability",
                    text="Explain tool use in terms of the available public capabilities.",
                ),
                SkillRequirement(
                    requirement_id="tool-exact-arguments",
                    text="Use the actual tool API, not a skill-specific wire.",
                ),
                SkillRequirement(
                    requirement_id="tool-observe-before-next",
                    text="Treat tool results as observations, not proposed actions as results.",
                ),
                SkillRequirement(
                    requirement_id="tool-adapt-plan",
                    text="When useful, revise tool selection after a failed attempt.",
                    kind="evolvable-strategy",
                ),
            ),
        ),
        _seed_document(
            skill_id="skill-verify-complete-v2",
            title="Verify before completion",
            summary="Optional checks of the proposed answer against the task and public evidence.",
            instructions=(
                "Optional advice: compare your proposed answer with the original task and "
                "available evidence. Check uncertain points when useful, or submit your best "
                "answer directly. You decide how much checking is worthwhile within the shared "
                "budget. This advice adds no response format and does not require another "
                "derivation, tool call, or peer opinion."
            ),
            requirements=(
                SkillRequirement(
                    requirement_id="verify-public-evidence",
                    text="Use public evidence and acknowledge uncertainty when relevant.",
                    kind="evolvable-strategy",
                ),
                SkillRequirement(
                    requirement_id="verify-output-shape",
                    text="Leave submission formatting to the task interface, not this skill.",
                ),
                SkillRequirement(
                    requirement_id="verify-no-hidden-answer",
                    text="Do not infer or request private evaluator material.",
                ),
            ),
        ),
    )


def planned_step_zero_seed_documents() -> tuple[SkillDocument, ...]:
    """Return the historical benchmark-specific Step-0 reproduction library.

    These documents do not replace the historical three-skill training preflight
    identity. The clean controller does not retrieve these restrictive legacy
    procedures: its skills are optional capability advice, with no catalog policy.
    """

    return (
        _seed_document(
            skill_id="skill-hotpot-evidence-chain",
            title="Hotpot evidence chain",
            summary="Resolve a multi-hop question from all supplied passages and answer briefly.",
            instructions=(
                "Read every supplied passage. Identify the two evidence hops that connect the "
                "question to one entity or value, resolve aliases, and check that the proposed "
                "short answer is explicitly supported. Return only the requested answer, not an "
                "explanation. Do not use facts outside the model-visible passages."
            ),
            requirements=(
                SkillRequirement("hotpot-two-hop", "Use the model-visible multi-hop evidence."),
                SkillRequirement("hotpot-short", "Produce one concise supported answer."),
            ),
            task_families=("qa",),
            contexts=("protocol13:hotpotqa:iid",),
        ),
        _seed_document(
            skill_id="skill-trivia-search-synthesis",
            title="Trivia search synthesis",
            summary=(
                "Use complementary public searches and synthesize one alias-compatible "
                "short answer."
            ),
            instructions=(
                "Use each search slot. Form a hypothesis from the public question and your own "
                "knowledge. First search key entities and relations; later use distinct queries "
                "to confirm or disconfirm it and resolve the decisive relation or alias. A "
                "`[label removed]` marker is missing text, not evidence: infer from the remaining "
                "public facts and your own knowledge. On the final pass compare the evidence and "
                "end with `Answer hypothesis: <canonical short answer>`. Submit only that answer, "
                "without alternatives or explanation."
            ),
            requirements=(
                SkillRequirement(
                    "trivia-search",
                    "Use the declared task-local searches with complementary queries.",
                ),
                SkillRequirement(
                    "trivia-synthesis",
                    "Synthesize one concise answer from snippets.",
                ),
            ),
            task_families=("qa",),
            contexts=("protocol13:triviaqa:iid",),
            required_tools=("search",),
        ),
        _seed_document(
            skill_id="skill-aime-integer-verification",
            title="AIME integer verification",
            summary="Solve carefully and verify the unique integer in the AIME range.",
            instructions=(
                "Reason step by step using the public problem. Once a coherent solution is "
                "complete, check the decisive calculation once without restarting or endlessly "
                "revising a settled derivation. End the reasoning with one explicit final integer "
                "from 0 through 999, and submit exactly that integer without alternatives."
            ),
            requirements=(
                SkillRequirement("aime-verify", "Check the decisive calculation once."),
                SkillRequirement("aime-domain", "Submit one integer from 0 through 999."),
            ),
            task_families=("math",),
            contexts=("protocol13:aime-2026:iid",),
        ),
        _seed_document(
            skill_id="skill-health-comprehensive-response",
            title="Comprehensive medical response",
            summary="Answer the full conversation safely, directly, and with useful next steps.",
            instructions=(
                "Respond to the user's last medical message in the context of the full public "
                "conversation. Address each explicit concern, explain uncertainty, state urgent "
                "warning signs when relevant, and give practical next steps without fabricating "
                "facts. Preserve nuance and do not shorten the response merely to satisfy the "
                "controller."
            ),
            requirements=(
                SkillRequirement("health-complete", "Address every explicit public concern."),
                SkillRequirement("health-safe", "Give proportionate safety guidance."),
            ),
            task_families=("health",),
            contexts=("protocol13:healthbench:iid",),
        ),
        _seed_document(
            skill_id="skill-webshop-constraint-ledger",
            title="WebShop constraint ledger",
            summary="Track public constraints and advance the current legal purchase action.",
            instructions=(
                "When `retrieved_public_catalog_candidates` is nonempty, it is answer-independent "
                "public tool output: start by issuing Candidate 1's public suggested search query, "
                "click its exact ID if the live result exposes it, select each public recommended "
                "option only when that exact value is on the current action surface, and move to "
                "the next ordered candidate only after an absent exact result or a visible hard "
                "violation. "
                "Otherwise search with the product type and discriminating attributes, without "
                "price prose. "
                "Keep a compact ledger of the requested product type, mandatory attributes, price "
                "limit, current product, selected options, and unmet constraints. Treat only the "
                "latest public action surface as executable and never repeat a stale action after "
                "an unchanged state. WebShop instructions may describe a compatible case, band, "
                "protector, bundle, or accessory through its intended use and rare attributes; do "
                "not reject such a candidate from the category word alone. Inspect public product "
                "evidence only as needed, treat missing descriptive wording as unknown rather than "
                "a violation, select every required visible option, and buy a reasonable matching "
                "product when no observed hard constraint is violated instead of scanning forever. "
                "Keep the reasoning concise and end it with `Action: <exact current native "
                "action>` so the separate action wire transcribes the same decision."
            ),
            requirements=(
                SkillRequirement("webshop-surface", "Choose one exact current public action."),
                SkillRequirement(
                    "webshop-ledger",
                    "Preserve public constraints across the episode.",
                ),
            ),
            task_families=("interactive-shopping",),
            contexts=("protocol13:webshop:iid",),
            required_tools=("click", "search"),
        ),
        _seed_document(
            skill_id="skill-alfworld-subgoal-machine",
            title="ALFWorld subgoal machine",
            summary=(
                "Track object state and execute one admissible command toward the next subgoal."
            ),
            instructions=(
                "Identify the task type, target object instance, required transform, destination, "
                "and count. Track held object, current location, completed transform, placed "
                "count, and exhausted locations only from public observations, preserving the "
                "exact target and destination words from the authoritative task. Open a closed "
                "receptacle when needed, take an exact visible target, prefer unvisited locations "
                "during search, perform the required clean/heat/cool operation while holding the "
                "target, then use an exact current placement command at the destination. For "
                "look/examine tasks, locate and use the required light source. If a command leaves "
                "the public state unchanged or a plausible placement remains nonterminal, do not "
                "declare completion or repeat it: choose a different compatible current object or "
                "admissible command. For two-object tasks, continue until both placements succeed. "
                "Keep the reasoning concise and end it with `Action: <exact current admissible "
                "command>` so the separate action wire transcribes the same decision."
            ),
            requirements=(
                SkillRequirement("alfworld-state", "Maintain the public task state machine."),
                SkillRequirement("alfworld-admissible", "Copy one current admissible command."),
            ),
            task_families=("interactive-household",),
            contexts=("protocol13:alfworld:iid",),
            required_tools=("act",),
        ),
        _seed_document(
            skill_id="skill-python-function-completion",
            title="Python function completion",
            summary="Implement the requested function as exact executable Python source.",
            instructions=(
                "Infer the full behavioral contract from the public prompt and examples, reason "
                "through edge cases, and emit executable Python source containing the required "
                "function. Preserve the required signature, avoid markdown around the submitted "
                "source, and do not include tests or prose in the terminal payload."
            ),
            requirements=(
                SkillRequirement("python-signature", "Preserve the required public signature."),
                SkillRequirement("python-source", "Submit executable source without prose."),
            ),
            task_families=("code",),
            contexts=("protocol13:humaneval:iid", "protocol13:mbpp-plus:iid"),
        ),
    )


__all__ = ["planned_seed_documents", "planned_step_zero_seed_documents"]
