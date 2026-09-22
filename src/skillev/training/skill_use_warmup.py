"""Explicit supervised initialization, never a hidden TTB auxiliary objective.

Only independently approved training-development demonstrations are admitted.
Read/body evidence establishes availability, not usefulness: usefulness or a
reasoned decision not to use a skill requires a separate attributable behavior annotation.
Artifacts, targets, and annotation references in checkpoints are PRIVATE.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import torch

from skillev.contracts import JsonValue
from skillev.policy import AdapterRole, PolicyBackbone
from skillev.policy.versions import TrainableVersions
from skillev.rollout import RolloutArtifact
from skillev.rollout.codec import ActionParseStatus, codec_for_initial_meta
from skillev.scoring.edge_plan import PreparedEdgePlan

from .inflight import durable_json
from .invocation_evidence import invocation_execution_links
from .optimizer_state import checkpoint_optimizer_state, require_optimizer_state_layout
from .warmup_supervision import load_annotation, supervision_report, validate_application_annotation

Decision = Literal["applied", "not-called", "abandoned-after-read"]
Source = tuple[str, str]  # benchmark, canonical source; population is not isolation.


@dataclass(frozen=True)
class WarmupDemonstration:
    artifact: RolloutArtifact
    plan: PreparedEdgePlan
    source: Source
    annotation_ref: str
    decision: Decision
    skill_id: str | None = None
    read_step: int | None = None
    decision_step: int | None = None
    annotation: dict[str, Any] | None = None

    def require(self) -> None:
        record = self.artifact.record
        if (
            self.plan.record != record
            or self.plan.initial_text != self.artifact.initial_context.text
        ):
            raise ValueError("warmup plan must belong to the complete original artifact")
        if self.plan.tokenizer_id != record.tokenizer_id:
            raise ValueError("warmup tokenizer differs")
        if not record.reward.success or not record.reward.verifier_version.strip():
            raise ValueError("warmup needs an actual successful native terminal verdict")
        if not self.annotation_ref.strip():
            raise ValueError("warmup needs an independent attributable application annotation")
        codec = codec_for_initial_meta(record.initial_context.meta)
        for step in record.steps:
            if codec.parse(step.action_text).status is not ActionParseStatus.VALID:
                raise ValueError("warmup cannot supervise a structurally invalid original action")
            edge = self.plan.edge(step.index, AdapterRole.FORWARD_POLICY)
            if (
                edge.role is not AdapterRole.FORWARD_POLICY
                or edge.action_ids != step.action_token_ids
            ):
                raise ValueError("warmup must preserve every original forward action token")
            if not edge.prefix_ids or not edge.action_ids:
                raise ValueError("warmup cannot supervise an empty action")
        links = invocation_execution_links(record, self.artifact.skill_input_evidence)
        if self.decision == "not-called":
            if links or any(s.invoked_skill_ids for s in record.steps):
                raise ValueError("not-called annotation conflicts with actual invocation")
            if self.read_step is not None or self.skill_id is not None:
                raise ValueError("not-called cannot name an actual read")
        elif self.decision in ("applied", "abandoned-after-read"):
            match = next(
                (
                    link
                    for link in links
                    if link.step_index == self.read_step and link.declared_skill_id == self.skill_id
                ),
                None,
            )
            if (
                match is None
                or not match.admitted
                or not match.body_returned
                or self.decision_step not in (match.body_visible_execution_steps or ())
            ):
                raise ValueError("annotation needs real read and body-visible later action")
        else:
            raise ValueError("unsupported attributed application decision")

    def to_value(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_value(),
            "edges": self.plan.wire_edges(),
            "source": list(self.source),
            "annotation_ref": self.annotation_ref,
            "decision": self.decision,
            "skill_id": self.skill_id,
            "read_step": self.read_step,
            "decision_step": self.decision_step,
            **({"annotation": self.annotation} if self.annotation is not None else {}),
        }


@dataclass(frozen=True)
class WarmupCorpus:
    demonstrations: tuple[WarmupDemonstration, ...]
    development_sources: tuple[Source, ...]
    excluded_sources: tuple[Source, ...]
    isolation_ref: str
    task_sources: tuple[tuple[str, str, str], ...]
    aliases: tuple[tuple[str, str, str], ...] = ()

    def canonical(self, source: Source) -> Source:
        mapping = {(domain, original): canonical for domain, original, canonical in self.aliases}
        seen: set[Source] = set()
        while source in mapping:
            if mapping[source] == source[1]:
                return source  # A declared canonical A -> A is a terminal, not a cycle.
            if source in seen:
                raise ValueError("source alias cycle")
            seen.add(source)
            source = source[0], mapping[source]
        return source

    def require(self) -> None:
        if not self.isolation_ref.strip() or not self.demonstrations:
            raise ValueError("explicit training-development isolation evidence is required")
        allowed = {self.canonical(s) for s in self.development_sources}
        excluded = {self.canonical(s) for s in self.excluded_sources}
        if allowed & excluded:
            raise ValueError("development sources overlap declared IID/quality exclusions")
        ids = [d.artifact.record.trajectory_id for d in self.demonstrations]
        if len(ids) != len(set(ids)):
            raise ValueError("repeated demonstration trajectory")
        decisions = {d.decision for d in self.demonstrations}
        if "applied" not in decisions or not decisions & {"not-called", "abandoned-after-read"}:
            raise ValueError("warmup needs annotated application and genuine negative controls")
        for example in self.demonstrations:
            example.require()
            canonical = self.canonical(example.source)
            if canonical not in allowed or canonical in excluded:
                raise ValueError("demonstration is not an isolated approved development source")
            tasks = {task: (domain, question) for task, domain, question in self.task_sources}
            if len(tasks) != len(self.task_sources):
                raise ValueError("development manifest repeats a task identity")
            task_id = example.artifact.manifest.task_id
            if task_id not in tasks or self.canonical(tasks[task_id]) != canonical:
                raise ValueError("artifact task differs from frozen development manifest")
            payload = example.artifact.record.reward.native_payload
            for key in ("training_evidence_source", "evaluation_evidence_source"):
                source = payload.get(key)
                if source is None:
                    continue  # Explicit frozen manifest is the source binding, not a relabel.
                if not isinstance(source, dict):
                    raise ValueError("invalid original source evidence")
                domain, question = source.get("benchmark_id"), source.get("source_question_id")
                if not isinstance(domain, str) or not isinstance(question, str):
                    raise ValueError("missing original canonical source coordinates")
                if self.canonical((domain, question)) != canonical:
                    raise ValueError("annotation source differs from the native evidence")

    def for_training(self, *, restored_annotations: bool = False) -> WarmupCorpus:
        """New updates require preserved @2 reviews, not legacy reference-only evidence."""
        examples = []
        for demonstration in self.demonstrations:
            note = (
                load_annotation(demonstration)
                if restored_annotations
                else json.loads(Path(demonstration.annotation_ref).read_text())
            )
            if not isinstance(note, dict):
                raise ValueError("application annotation must be a JSON object")
            if demonstration.annotation is not None and demonstration.annotation != note:
                raise ValueError("annotation reference changed from the preserved behavior review")
            examples.append(replace(demonstration, annotation=json.loads(json.dumps(note))))
        loaded = replace(self, demonstrations=tuple(examples))
        loaded.require()
        positives = set()
        for example in loaded.demonstrations:
            assert example.annotation is not None
            note = validate_application_annotation(example.annotation, example.artifact)
            if note.get("approved_for_warmup") is not True or any(
                note.get(key) != getattr(example, key)
                for key in ("decision", "skill_id", "read_step", "decision_step")
            ):
                raise ValueError(
                    "warmup approval/decision differs from the original behavior review"
                )
            if example.decision == "applied":
                if note["behavior_evidence"]["ceremonial_read"]:
                    raise ValueError("ceremonial reading is not a warmup application positive")
                positives.add(loaded.canonical(example.source))
        if len(positives) < 2:
            raise ValueError(
                "new warmup requires at least two independent canonical positive sources"
            )
        return loaded

    def supervision_report(self) -> dict[str, Any]:
        return supervision_report(self)

    def to_value(self) -> dict[str, Any]:
        return {
            "demonstrations": [d.to_value() for d in self.demonstrations],
            "development_sources": [list(s) for s in self.development_sources],
            "excluded_sources": [list(s) for s in self.excluded_sources],
            "isolation_ref": self.isolation_ref,
            "task_sources": [list(row) for row in self.task_sources],
            "aliases": [list(a) for a in self.aliases],
        }


@dataclass(frozen=True)
class WarmupConfig:
    learning_rate: float
    maximum_updates: int
    weight_decay: float = 0.0
    format: str = "skill-use-warmup@1"

    def __post_init__(self) -> None:
        if self.format != "skill-use-warmup@1":
            raise ValueError("unsupported warmup condition")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or type(self.maximum_updates) is not int
            or self.maximum_updates < 1
        ):
            raise ValueError("invalid explicit warmup optimizer configuration")


class SkillUseWarmup:
    """One full fixed corpus per update, token-mean original action NLL.

    No sampling, annotation inference, B/Z gradient, posterior, or reward update.
    Use an isolated backbone, not an attached active formal optimizer owner.
    """

    def __init__(
        self,
        *,
        backbone: PolicyBackbone,
        corpus: WarmupCorpus,
        config: WarmupConfig,
        run_id: str,
        _restored_annotation_contents: bool = False,
    ) -> None:
        corpus = corpus.for_training(restored_annotations=_restored_annotation_contents)
        if any(
            d.plan.tokenizer_id != backbone.tokenizer.tokenizer_id for d in corpus.demonstrations
        ):
            raise ValueError("warmup action tokens target another backbone tokenizer")
        if not run_id.strip() or "@" in run_id:
            raise ValueError("warmup run_id must be explicit and exclude version separators")
        self.backbone, self.corpus, self.config, self.run_id = backbone, corpus, config, run_id
        groups = backbone.parameter_groups()
        if any(p.grad is not None for p in (*groups.forward, *groups.backward, *groups.z_head)):
            raise ValueError("warmup cannot take over a backbone with outstanding gradients")
        self.original_versions = TrainableVersions.from_backbone(backbone)
        self.optimizer = torch.optim.AdamW(
            groups.forward, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.updates = 0
        self.history: list[dict[str, Any]] = []

    def step(self) -> dict[str, Any]:
        if self.updates >= self.config.maximum_updates:
            raise ValueError("declared warmup update budget is exhausted")
        groups = self.backbone.parameter_groups()
        tokens = sum(
            len(s.action_token_ids)
            for d in self.corpus.demonstrations
            for s in d.artifact.record.steps
        )
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        try:
            for demonstration in self.corpus.demonstrations:
                for step in demonstration.artifact.record.steps:
                    edge = demonstration.plan.edge(step.index, AdapterRole.FORWARD_POLICY)
                    scores = self.backbone.score(
                        edge.prefix_ids, edge.action_ids, AdapterRole.FORWARD_POLICY
                    )
                    if scores.numel() != len(edge.action_ids) or not torch.isfinite(scores).all():
                        raise ValueError("warmup received incomplete or nonfinite action scores")
                    loss = -scores.sum() / tokens
                    # autograd.grad limits ownership to F even if a backend is miswired.
                    gradients = torch.autograd.grad(loss, groups.forward, allow_unused=True)
                    if all(gradient is None for gradient in gradients):
                        raise ValueError("action score does not reach the forward adapter")
                    for parameter, gradient in zip(groups.forward, gradients, strict=True):
                        if gradient is not None:
                            if not torch.isfinite(gradient).all():
                                raise ValueError("nonfinite warmup gradient")
                            if parameter.grad is None:
                                parameter.grad = gradient.detach().clone()
                            else:
                                parameter.grad.add_(gradient.detach())
                    total += float(loss.detach())
            if not math.isfinite(total) or any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in groups.forward
            ):
                raise ValueError("nonfinite accumulated warmup objective or gradient")
            self.optimizer.step()
        finally:
            self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        self.backbone.synchronize_trainable_versions(
            TrainableVersions(
                f"skill-use-warmup/{self.run_id}@{self.updates}",
                self.original_versions.backward,
                self.original_versions.z,
            )
        )
        result = {
            "format": self.config.format,
            "initialization": "supervised-warmup-not-Step-0",
            "warmup_update": self.updates,
            "action_token_mean_nll": total,
            "action_tokens": tokens,
            "trajectories": len(self.corpus.demonstrations),
            "formal_optimizer_steps": 0,
            "posterior_updates": 0,
        }
        self.history.append(result)
        return result

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.backbone.save_checkpoint(str(directory / "policy"))
        with (directory / "optimizer.pt").open("xb") as stream:
            torch.save(checkpoint_optimizer_state(self.backbone, self.optimizer), stream)
            stream.flush()
            os.fsync(stream.fileno())
        durable_json(
            directory / "warmup.json",
            {
                "format": self.config.format,
                "run_id": self.run_id,
                "config": asdict(self.config),
                "corpus": self.corpus.to_value(),
                "original_versions": asdict(self.original_versions),
                "updates": self.updates,
                "history": cast(JsonValue, self.history),
                "supervision_report": self.corpus.supervision_report(),
            },
        )
        with (directory / "COMPLETE").open("x") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @classmethod
    def restore(
        cls,
        directory: Path,
        *,
        backbone: PolicyBackbone,
        corpus: WarmupCorpus,
        config: WarmupConfig,
        run_id: str,
    ) -> SkillUseWarmup:
        if not (directory / "COMPLETE").is_file():
            raise ValueError("warmup restore needs a complete saved update boundary")
        state = json.loads((directory / "warmup.json").read_text())
        corpus = corpus.for_training(restored_annotations=True)
        if (
            state["config"] != asdict(config)
            or state["run_id"] != run_id
            or state["corpus"] != corpus.to_value()
        ):
            raise ValueError("warmup restore cannot change demonstrations or condition")
        result = cls(
            backbone=backbone,
            corpus=corpus,
            config=config,
            run_id=run_id,
            _restored_annotation_contents=True,
        )
        optimizer_state = torch.load(directory / "optimizer.pt", weights_only=True)
        require_optimizer_state_layout(optimizer_state, backbone, result.optimizer)
        backbone.load_checkpoint(str(directory / "policy"))
        result.optimizer.load_state_dict(optimizer_state)
        result.original_versions = TrainableVersions(**state["original_versions"])
        result.updates, result.history = state["updates"], state["history"]
        return result
