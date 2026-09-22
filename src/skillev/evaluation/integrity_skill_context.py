"""Skill context budgeting and diagnostics around the existing production access API."""

from __future__ import annotations

from skillev.evolution.retriever import TaskRetrievalFeatures
from skillev.evolution.skill_access import ReadOnlySkillAccess

from .integrity_generation import IntegrityGeneration
from .public_context import PromptBlock
from .skill_library_config import FrozenSkillLibrary


class IntegritySkillContext:
    def __init__(
        self,
        snapshot: FrozenSkillLibrary,
        features: TaskRetrievalFeatures,
        budget: int,
        generation: IntegrityGeneration,
    ) -> None:
        self.snapshot, self.features = snapshot, features
        self.access = ReadOnlySkillAccess(snapshot.state)
        self.budget, self.generation = budget, generation
        self.blocks: dict[str, PromptBlock] = {}
        self.costs: dict[str, int] = {}
        self.retrieved: set[str] = set()
        generation.counts.skill_access_episodes += 1
        self._trace(
            "skill-access",
            {
                "allowed": True,
                "library_id": snapshot.library_id,
                "kind": snapshot.kind,
                "retrieval_rule": snapshot.retrieval_rule,
                "source_optimizer_step": snapshot.source_optimizer_step,
                "context_entry_mode": "automatic-retrieval-plus-owner-discovery",
                "task_family": features.task_family,
                "context": features.context,
                "available_tools": features.available_tools,
            },
        )
        self.retrieve(origin="automatic-initial-retrieval")

    def _trace(self, stage: str, value: object) -> None:
        self.generation.trace(self.features.task_id, stage, value)

    def _include(
        self, skill_id: str, content: str, *, prioritize: bool = False
    ) -> dict[str, object]:
        block = PromptBlock(
            "tool", f"Optional skill ({skill_id}):\n{content}", required=False, skill_id=skill_id
        )
        cost = len(self.generation.generator.tokenizer.encode(block.content))
        displaced = []
        if prioritize and cost <= self.budget:
            # The model selected this document, so old automatic context yields
            # to it. No document is truncated and no instruction is synthesized.
            for previous in tuple(self.blocks):
                if sum(self.costs.values()) - self.costs.get(skill_id, 0) + cost <= self.budget:
                    break
                if previous != skill_id:
                    del self.blocks[previous], self.costs[previous]
                    displaced.append(previous)
        available = self.budget - sum(self.costs.values()) + self.costs.get(skill_id, 0)
        if cost > available:
            self.generation.counts.skill_budget_skips += 1
            return {
                "skill_id": skill_id,
                "status": "skill-token-budget",
                "body_tokens": cost,
                "available_tokens": available,
            }
        self.blocks[skill_id], self.costs[skill_id] = block, cost
        return {
            "skill_id": skill_id,
            "status": "body-prepared",
            "body_tokens": cost,
            "displaced_skill_ids": displaced,
        }

    def retrieve(self, *, origin: str = "owner-request") -> dict[str, object]:
        selected = self.access.retrieve(self.features)
        self.generation.counts.skill_retrieval_requests += 1
        self.generation.counts.skill_retrieved += len(selected)
        self.generation.counts.skill_no_match += int(not selected)
        self.retrieved.update(item.metadata.skill_id for item in selected)
        result: dict[str, object] = {
            "origin": origin,
            "status": "matched" if selected else "no-match",
            "retrieved_skill_ids": [item.metadata.skill_id for item in selected],
            "applicability": self.access.applicability_report(self.features),
            "bodies": [self._include(item.metadata.skill_id, item.content) for item in selected],
        }
        self._trace("skill-retrieval", result)
        return result

    def execute(self, request: dict[str, object]) -> dict[str, object]:
        operation = request.get("operation")
        if request.get("name") == "invoke":
            # Production StructuredAction compatibility: an explicit invocation
            # of a text skill, not a new evaluation-only action language.
            if request.get("resource_id") != "skill-runtime" or request.get("arguments") != {}:
                raise ValueError(
                    "skill invocation requires the declared runtime and empty arguments"
                )
            operation = "invoke_skill"
            request = {"kind": "skill", "operation": operation, "skill_id": request.get("skill_id")}
        if operation == "list_skills":
            if request.keys() - {"kind", "operation", "cursor", "limit"}:
                raise ValueError("unsupported skill discovery argument")
            cursor, limit = request.get("cursor", 0), request.get("limit", 8)
            if type(cursor) is not int or type(limit) is not int:
                raise ValueError("skill page requires integer cursor and limit")
            result = self.access.discover(cursor=cursor, limit=limit)
            self.generation.counts.skill_discovery_calls += 1
        elif operation == "retrieve_skills":
            if request.keys() - {"kind", "operation"}:
                raise ValueError(
                    "retrieval uses the current public task, without private arguments"
                )
            result = self.retrieve()
        elif operation in {"read_skill", "invoke_skill"}:
            skill_id = request.get("skill_id")
            if type(skill_id) is not str or request.keys() - {"kind", "operation", "skill_id"}:
                raise ValueError("skill read or invocation requires one active skill ID")
            content = self.access.read(skill_id)
            self.generation.counts.skill_read_calls += 1
            result = self._include(skill_id, content, prioritize=True)
            if result["status"] == "body-prepared" and operation == "invoke_skill":
                result.update(self.access.invoke(skill_id))
                self.generation.counts.skill_calls += 1
        else:
            raise ValueError("unknown skill operation")
        self.generation.counts.tool_calls += 1
        self._trace("skill-tool", {"request": request, "result": result})
        return result
