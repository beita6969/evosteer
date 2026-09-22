"""Synthetic CPU wiring evidence, never real Qwen/skill-effectiveness evidence."""

import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from skillev.contracts import canonical_json
from skillev.policy import AdapterRole, PolicyParameterGroups
from skillev.policy.versions import TrainableVersions
from skillev.rollout import CanonicalInitialContextAssembler, DecodingSnapshot
from skillev.rollout.catalog import CatalogReadEnvironment
from skillev.scoring.edge_plan import prepare_edge_plan
from skillev.training.skill_use_warmup import (
    SkillUseWarmup,
    WarmupConfig,
    WarmupCorpus,
    WarmupDemonstration,
)
from tests.rollout.engine_fakes import (
    ByteTokenizer,
    FakeTerminalEvaluator,
    GenerationScript,
    ScriptedEnvironment,
    default_request,
    make_harness,
)
from tests.rollout.test_native_tool_wire import call, contract


class Model:
    def __init__(self):
        self.f = torch.nn.Parameter(torch.tensor([0.1, 0.3, -0.2]))
        self.b = torch.nn.Parameter(torch.tensor([0.8]))
        self.z = torch.nn.Parameter(torch.tensor([0.4]))
        self.versions = TrainableVersions("f@0", "b@0", "z@0")
        self.calls = []
        self.tokenizer = ByteTokenizer()
        self.loads = 0

    def named_trainable_parameters(self):
        return {"forward": self.f, "backward": self.b, "z": self.z}

    def parameter_groups(self):
        return PolicyParameterGroups((self.f,), (self.b,), (self.z,))

    def adapter_version(self, role):
        return (
            self.versions.forward if role is AdapterRole.FORWARD_POLICY else self.versions.backward
        )

    @property
    def z_version(self):
        return self.versions.z

    def synchronize_trainable_versions(self, versions):
        self.versions = versions

    def score(self, prefix, action, role):
        self.calls.append((prefix, action, role))
        return self.f.log_softmax(0)[torch.tensor(action) % 3]

    def save_checkpoint(self, directory):
        path = Path(directory)
        path.mkdir()
        torch.save({"f": self.f, "b": self.b, "z": self.z}, path / "model.pt")
        (path / "versions.json").write_text(json.dumps(asdict(self.versions)))

    def load_checkpoint(self, directory):
        self.loads += 1
        path = Path(directory)
        values = torch.load(path / "model.pt", weights_only=True)
        with torch.no_grad():
            for name, value in values.items():
                getattr(self, name).copy_(value)
        self.versions = TrainableVersions(**json.loads((path / "versions.json").read_text()))


def example(
    tmp_path, decision, *, success=True, invalid_action=None, name=None, body="SYNTHETIC_BODY"
):
    name = name or decision
    tokenizer = ByteTokenizer()
    request = default_request(trajectory_id=name)
    task = replace(request.task, task_id=name, action_surface=contract().surface)
    original = request.retrieved_skills[0]
    skill = replace(
        original,
        content=canonical_json(
            {
                "title": "A public procedure",
                "summary": "Use only when applicable",
                "applicability": {},
                "instructions": body,
                "requirements": [],
            }
        ),
    )
    request = replace(
        request,
        task=task,
        sampling_coordinate=replace(request.sampling_coordinate, task_id=task.task_id),
        retrieved_skills=(skill,),
        active_skill_ids=(skill.metadata.skill_id,),
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=128,
            max_action_tokens=4096,
            base_seed=0,
            action_boundary_version="native-model-stop@1",
        ),
    )
    texts = ["complete independently", call("submit_answer", answer="synthetic")]
    if decision != "not-called":
        texts = ["consider procedure", call("read_skill", skill_id=skill.metadata.skill_id), *texts]
    if invalid_action is not None:
        texts = ["unsuccessful protocol attempt", invalid_action, *texts]
    directory = tmp_path / name
    directory.mkdir()
    harness = make_harness(
        directory,
        tokenizer=tokenizer,
        request=request,
        max_turns=2,
        evaluator=FakeTerminalEvaluator(value=1 if success else 0),
        context_assembler=CanonicalInitialContextAssembler(
            maximum_h0_tokens=20000,
            phase_context=True,
            action_wire="native-single-tool-call@3",
            skill_exposure="catalog-then-read@1",
        ),
        environment=CatalogReadEnvironment(
            ScriptedEnvironment([]), (skill,), request.library_version
        ),
        scripts=[GenerationScript.text(tokenizer, text) for text in texts],
    )
    artifact = asyncio.run(harness.engine.run(request))
    plan = prepare_edge_plan(tokenizer, artifact.record, artifact.initial_context.text)
    annotation = directory / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "format": "skill-application-annotation@2",
                "reviewer": "synthetic-reviewer",
                "review_kind": "model-assisted-public-behavior",
                "reviewed_at": "synthetic-date",
                "rationale": "Synthetic mechanism fixture, not real effectiveness evidence.",
                "approved_for_warmup": True,
                "trajectory_id": name,
                "decision": decision,
                "skill_id": skill.metadata.skill_id if decision != "not-called" else None,
                "read_step": 1 if decision != "not-called" else None,
                "decision_step": 2 if decision != "not-called" else None,
                "behavior_evidence": {
                    "method_clauses": [{"clause_id": "procedure", "quote": "SYNTHETIC_BODY"}]
                    if decision != "not-called"
                    else [],
                    "actions": [
                        {
                            "step_index": 2 if decision != "not-called" else 1,
                            "behavior": "Synthetic reviewed action",
                            "clause_ids": ["procedure"] if decision != "not-called" else [],
                        }
                    ],
                    "ceremonial_read": False,
                    "attribution_limits": ["Synthetic fixture; no causal benefit demonstrated."],
                },
            }
        )
    )
    return WarmupDemonstration(
        artifact,
        plan,
        ("synthetic", name),
        str(annotation),
        decision,
        skill.metadata.skill_id if decision != "not-called" else None,
        1 if decision != "not-called" else None,
        2 if decision != "not-called" else None,
    )


def corpus(tmp_path, negative="not-called"):
    examples = (
        example(tmp_path, "applied"),
        example(tmp_path, "applied", name="applied-second"),
        example(tmp_path, negative),
    )
    return WarmupCorpus(
        examples,
        tuple(d.source for d in examples),
        (("synthetic", "iid-excluded"),),
        "frozen-development-manifest",
        tuple((d.artifact.manifest.task_id, *d.source) for d in examples),
    )


@pytest.mark.parametrize("negative", ["not-called", "abandoned-after-read"])
def test_only_original_f_actions_train_and_real_adam_recovers(tmp_path, negative):
    data = corpus(tmp_path, negative)
    original = data.to_value()
    model = Model()
    config = WarmupConfig(0.01, 2)
    trainer = SkillUseWarmup(backbone=model, corpus=data, config=config, run_id="synthetic")
    before = [p.detach().clone() for p in (model.f, model.b, model.z)]
    reference = torch.nn.Parameter(model.f.detach().clone())
    reference_adam = torch.optim.AdamW([reference], lr=0.01, weight_decay=0)
    ids = torch.tensor(
        [
            token
            for d in data.demonstrations
            for step in d.artifact.record.steps
            for token in step.action_token_ids
        ]
    )
    reference_loss = -reference.log_softmax(0)[ids % 3].mean()
    reference_loss.backward()
    reference_adam.step()
    report = trainer.step()
    assert torch.allclose(model.f, reference, atol=1e-7, rtol=1e-7)
    assert report["action_token_mean_nll"] == pytest.approx(float(reference_loss.detach()))
    assert report["formal_optimizer_steps"] == report["posterior_updates"] == 0
    assert not torch.equal(model.f, before[0])
    assert torch.equal(model.b, before[1])
    assert torch.equal(model.z, before[2])
    assert model.b.grad is model.z.grad is None
    assert model.versions.backward == "b@0"
    assert model.versions.z == "z@0"
    assert model.versions.forward.startswith("skill-use-warmup/")
    expected = [
        (
            d.plan.edge(s.index, AdapterRole.FORWARD_POLICY).prefix_ids,
            s.action_token_ids,
            AdapterRole.FORWARD_POLICY,
        )
        for d in data.demonstrations
        for s in d.artifact.record.steps
    ]
    assert model.calls == expected
    assert data.to_value() == original
    trainer.save(tmp_path / "checkpoint")
    restored_model = Model()
    restored = SkillUseWarmup.restore(
        tmp_path / "checkpoint",
        backbone=restored_model,
        corpus=data,
        config=config,
        run_id="synthetic",
    )
    assert restored.updates == 1
    assert restored.step() == trainer.step()
    assert torch.equal(restored_model.f, model.f)
    for key, value in trainer.optimizer.state[model.f].items():
        assert torch.equal(value, restored.optimizer.state[restored_model.f][key])
    with pytest.raises(ValueError):
        restored.step()
    with pytest.raises(FileExistsError):
        trainer.save(tmp_path / "checkpoint")


def test_no_automatic_read_success_enrollment_and_alias_isolation(tmp_path):
    data = corpus(tmp_path)
    with pytest.raises(ValueError):
        replace(data, demonstrations=data.demonstrations[:1]).require()
    with pytest.raises(ValueError):
        replace(
            data,
            excluded_sources=(("synthetic", "alias"),),
            aliases=(("synthetic", "alias", "applied"),),
        ).require()
    with pytest.raises(ValueError):
        replace(data, task_sources=(("unrelated", "synthetic", "applied"),)).require()
    applied = data.demonstrations[0]
    with pytest.raises(ValueError):
        replace(applied, annotation_ref="").require()
    with pytest.raises(ValueError):
        replace(applied, artifact=replace(applied.artifact, skill_input_evidence=None)).require()
    with pytest.raises(ValueError):
        replace(applied, decision="not-called", read_step=None, skill_id=None).require()


def test_restore_rejects_changed_corpus_and_partial_checkpoint_before_loading(tmp_path):
    data = corpus(tmp_path)
    model, config = Model(), WarmupConfig(0.01, 2)
    trainer = SkillUseWarmup(backbone=model, corpus=data, config=config, run_id="one")
    trainer.step()
    trainer.save(tmp_path / "checkpoint")
    other = Model()
    before = other.f.detach().clone()
    with pytest.raises(ValueError):
        SkillUseWarmup.restore(
            tmp_path / "checkpoint",
            backbone=other,
            corpus=replace(data, isolation_ref="changed"),
            config=config,
            run_id="one",
        )
    assert torch.equal(other.f, before)
    (tmp_path / "checkpoint" / "COMPLETE").unlink()
    with pytest.raises(ValueError):
        SkillUseWarmup.restore(
            tmp_path / "checkpoint", backbone=other, corpus=data, config=config, run_id="one"
        )


def test_native_failure_and_nonfinite_scoring_never_update(tmp_path):
    failed = example(tmp_path, "abandoned-after-read", success=False)
    with pytest.raises(ValueError):
        failed.require()
    data = corpus(tmp_path)
    model = Model()
    trainer = SkillUseWarmup(
        backbone=model, corpus=data, config=WarmupConfig(0.01, 1), run_id="one"
    )
    before = model.f.detach().clone()
    real_score = model.score
    model.score = lambda prefix, action, role: real_score(prefix, action, role) * float("nan")
    with pytest.raises(ValueError):
        trainer.step()
    assert torch.equal(model.f, before)
    assert not trainer.optimizer.state
    assert trainer.updates == 0
    assert model.f.grad is None


class TwoForwardModel(Model):
    def __init__(self, *, reverse=False):
        super().__init__()
        self.f2 = torch.nn.Parameter(torch.zeros_like(self.f))
        self.reverse = reverse

    def named_trainable_parameters(self):
        return {**super().named_trainable_parameters(), "forward-second": self.f2}

    def parameter_groups(self):
        forward = (self.f2, self.f) if self.reverse else (self.f, self.f2)
        return PolicyParameterGroups(forward, (self.b,), (self.z,))


def test_restore_checks_named_forward_layout_before_model_load(tmp_path):
    data, config = corpus(tmp_path), WarmupConfig(0.01, 2)
    trainer = SkillUseWarmup(backbone=TwoForwardModel(), corpus=data, config=config, run_id="one")
    trainer.step()
    checkpoint = tmp_path / "checkpoint"
    trainer.save(checkpoint)
    other = TwoForwardModel(reverse=True)
    with pytest.raises(ValueError):
        SkillUseWarmup.restore(checkpoint, backbone=other, corpus=data, config=config, run_id="one")
    assert other.loads == 0
    assert torch.equal(other.f, torch.tensor([0.1, 0.3, -0.2]))


def test_backbone_tokenizer_mismatch_rejected_without_scoring(tmp_path):
    data, model = corpus(tmp_path), Model()
    model.tokenizer.tokenizer_id = "different-tokenizer"
    with pytest.raises(ValueError):
        SkillUseWarmup(backbone=model, corpus=data, config=WarmupConfig(0.01, 1), run_id="one")
    assert not model.calls


@pytest.mark.parametrize("fault", ["accumulated-gradient", "summed-nll"])
def test_finite_edges_can_overflow_accumulation_but_never_step(tmp_path, fault):
    class LargeGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, parameter, count):
            ctx.save_for_backward(parameter)
            return torch.zeros(count)

        @staticmethod
        def backward(ctx, upstream):
            (parameter,) = ctx.saved_tensors
            return torch.full_like(parameter, 2e38), None

    data, model = corpus(tmp_path), Model()
    if fault == "accumulated-gradient":
        model.score = lambda prefix, action, role: LargeGradient.apply(model.f, len(action))
    else:
        model.score = lambda prefix, action, role: (
            model.f.sum() * 0 + torch.full((len(action),), -3e38)
        )
    trainer = SkillUseWarmup(
        backbone=model, corpus=data, config=WarmupConfig(0.01, 1), run_id="one"
    )
    before = model.f.detach().clone()
    with pytest.raises(ValueError):
        trainer.step()
    assert torch.equal(model.f, before)
    assert model.f.grad is None
    assert not trainer.optimizer.state
    assert trainer.updates == 0


@pytest.mark.parametrize("invalid_action", ["not a tool carrier", call("unknown_surface", x=1)])
def test_success_after_invalid_action_is_not_a_warmup_demonstration(tmp_path, invalid_action):
    demonstration = example(tmp_path, "not-called", invalid_action=invalid_action)
    before = demonstration.artifact.to_value()
    assert demonstration.artifact.record.reward.success
    assert demonstration.artifact.record.horizon == 2
    with pytest.raises(ValueError):
        demonstration.require()
    assert demonstration.artifact.to_value() == before


@pytest.mark.parametrize("transitive", [False, True])
def test_identity_alias_terminates_and_preserves_corpus_isolation(tmp_path, transitive):
    data = corpus(tmp_path)
    identities = tuple((domain, source, source) for domain, source in data.development_sources)
    aliases = (
        *identities,
        ("synthetic", "alias", "middle" if transitive else "applied"),
        ("synthetic", "middle", "applied"),
    )
    declared = replace(data, aliases=aliases)
    original = data.to_value()
    declared.require()
    assert declared.canonical(("synthetic", "applied")) == ("synthetic", "applied")
    assert declared.canonical(("synthetic", "alias")) == ("synthetic", "applied")
    for excluded in ("applied", "alias"):
        with pytest.raises(ValueError):
            replace(declared, excluded_sources=(("synthetic", excluded),)).require()
    assert data.to_value() == original


def test_real_alias_cycle_still_rejects_and_cannot_bypass_exclusions(tmp_path):
    data = corpus(tmp_path)
    cyclic = replace(
        data,
        aliases=(("synthetic", "applied", "alias"), ("synthetic", "alias", "applied")),
    )
    with pytest.raises(ValueError):
        cyclic.canonical(("synthetic", "applied"))
    with pytest.raises(ValueError):
        cyclic.require()
    with pytest.raises(ValueError):
        replace(cyclic, excluded_sources=(("synthetic", "alias"),)).require()
