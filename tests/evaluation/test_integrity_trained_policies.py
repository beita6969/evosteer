"""Explicit checkpoint routing, not label-based trained results or base-answer fallback."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from skillev_private.evaluation.integrity_policies import TrainedPolicyBinding

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_pipeline import run_paired
from skillev.evaluation.integrity_results import require_paired_controls
from skillev.evaluation.integrity_resume import EvaluationRunMode
from skillev.evaluation.step0_integrity import InferenceArm, SkillMode
from skillev.rollout.evaluation_sglang import (
    EvaluationPolicyDescriptor,
    EvaluationSGLangGenerationError,
)
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.evaluation.test_step0_integrity_pipeline import Runtime, panel
from tests.evaluation.test_step0_integrity_results import controls


def checkpoint_fixture(tmp_path):
    directory = tmp_path / "chosen-checkpoint"
    policy = directory / "policy"
    forward = policy / "forward_adapter"
    forward.mkdir(parents=True)
    (policy / "policy_state.json").write_text(
        json.dumps(
            {
                "format": "skillev-policy-checkpoint@4",
                "optimizer_step": 16,
                "backbone_id": "synthetic-backbone",
                "forward_version": "forward-synthetic@16",
            }
        )
    )
    (forward / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "/synthetic/Qwen3.5-9B",
                "r": 4,
                "target_modules": ["q_proj", "v_proj"],
            }
        )
    )
    # Deliberately not tensors: metadata binding must never load model weights on the CPU broker.
    (forward / "adapter_model.safetensors").write_bytes(b"synthetic-nonempty-weights")
    (directory / "COMPLETE").write_text("complete\n")
    (directory / "runtime_state.json").write_text(json.dumps({"optimizer_step": 16}))
    settings = {"checkpoint_directory": str(directory), "adapter_name": "synthetic-forward-16"}
    observed = [
        {
            "model_info": {"model_path": "/synthetic/Qwen3.5-9B", "model_type": "qwen3_5"},
            "server_info": {
                "server_args": {"enable_lora": True, "served_model_name": "synthetic-base"}
            },
            "models": {
                "data": [
                    {
                        "id": settings["adapter_name"],
                        "root": str(forward),
                        "parent": "synthetic-base",
                    }
                ]
            },
        }
    ]
    return settings, observed


def binding_fixture(tmp_path):
    settings, observed = checkpoint_fixture(tmp_path)
    return TrainedPolicyBinding.read(
        "synthetic-policy-16", settings, observed=observed, tokenizer_id="fixture"
    )


@pytest.mark.parametrize("direct_policy", [False, True])
def test_explicit_checkpoint_binding_reads_forward_only_and_matches_the_actual_route(
    tmp_path, direct_policy
):
    settings, observed = checkpoint_fixture(tmp_path)
    if direct_policy:
        settings["checkpoint_directory"] += "/policy"
    binding = TrainedPolicyBinding.read(
        "synthetic-policy-16", settings, observed=observed, tokenizer_id="fixture"
    )
    assert binding.policy.forward_adapter_version == "forward-synthetic@16"
    assert binding.inference_state.optimizer_steps == 16
    assert binding.inference_state.forward_adapter_active
    assert not any(
        (
            binding.inference_state.backward_adapter_active,
            binding.inference_state.posterior_active,
            binding.inference_state.calibration_active,
            binding.inference_state.operator_active,
        )
    )
    binding.require_arm(
        InferenceArm("trained", optimizer_steps=16, policy_id="synthetic-policy-16")
    )
    with pytest.raises(ValueError):
        binding.require_arm(
            InferenceArm("wrong-step", optimizer_steps=17, policy_id="synthetic-policy-16")
        )


@pytest.mark.parametrize(
    "defect",
    [
        "missing-route",
        "backward-route",
        "base-route",
        "disabled-lora",
        "other-base",
        "other-replica",
    ],
)
def test_missing_or_misbound_lora_never_silently_becomes_base_inference(tmp_path, defect):
    settings, observed = checkpoint_fixture(tmp_path)
    if defect == "missing-route":
        observed[0]["models"]["data"] = []
    elif defect == "backward-route":
        observed[0]["models"]["data"][0]["root"] = (
            settings["checkpoint_directory"] + "/policy/backward_adapter"
        )
    elif defect == "base-route":
        observed[0]["models"]["data"][0]["parent"] = None
    elif defect == "disabled-lora":
        observed[0]["server_info"]["server_args"]["enable_lora"] = False
    elif defect == "other-base":
        observed[0]["model_info"]["model_path"] = "/synthetic/different-base"
    else:
        observed.append(deepcopy(observed[0]))
        observed[1]["models"]["data"] = []
    with pytest.raises(ValueError):
        TrainedPolicyBinding.read(
            "synthetic-policy-16", settings, observed=observed, tokenizer_id="fixture"
        )


def test_remote_replica_paths_are_explicit_and_checkpoint_changes_are_visible(tmp_path):
    settings, observed = checkpoint_fixture(tmp_path)
    observed.append(deepcopy(observed[0]))
    settings["served_adapter_paths"] = [
        "/synthetic/replica-a/forward_adapter",
        "/synthetic/replica-b/forward_adapter",
    ]
    for item, path in zip(observed, settings["served_adapter_paths"], strict=True):
        item["models"]["data"][0]["root"] = path
    binding = TrainedPolicyBinding.read(
        "synthetic-policy-16", settings, observed=observed, tokenizer_id="fixture"
    )
    state_path = binding.checkpoint_directory / "policy/policy_state.json"
    state = json.loads(state_path.read_text())
    state["forward_version"] = "changed-forward"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        binding.validate(observed)


@pytest.mark.parametrize("defect", ["incomplete", "wrong-step", "empty-weights"])
def test_checkpoint_must_be_finished_and_agree_with_its_training_step(tmp_path, defect):
    settings, observed = checkpoint_fixture(tmp_path)
    directory = Path(settings["checkpoint_directory"])
    if defect == "incomplete":
        (directory / "COMPLETE").unlink()
    elif defect == "wrong-step":
        (directory / "runtime_state.json").write_text(json.dumps({"optimizer_step": 17}))
    else:
        (directory / "policy/forward_adapter/adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError):
        TrainedPolicyBinding.read(
            "synthetic-policy-16", settings, observed=observed, tokenizer_id="fixture"
        )


def policy_controls(arm):
    model = {
        "observed": {"model_path": "/synthetic/Qwen3.5-9B"},
        "policy": asdict(
            EvaluationPolicyDescriptor(
                arm.policy_id or "synthetic-base",
                "Qwen3.5-9B",
                "fixture",
                "forward-synthetic@16" if arm.optimizer_steps else "adapter-free",
            )
        ),
        "adapter_route_sent": "synthetic-forward-16" if arm.optimizer_steps else None,
    }
    if arm.optimizer_steps:
        model["checkpoint"] = {
            "optimizer_steps": arm.optimizer_steps,
            "forward_version": "forward-synthetic@16",
        }
    return replace(controls(), model=model)


def policy_arms():
    base = InferenceArm("base")
    return base, replace(
        base, arm_id="trained", policy_id="synthetic-policy-16", optimizer_steps=16
    )


def test_weight_only_and_trained_skill_only_controls_are_distinct_comparisons():
    base, trained = policy_arms()
    a, b = policy_controls(base), policy_controls(trained)
    assert require_paired_controls(a, b, base, trained) == "forward-policy"
    skilled = replace(trained, arm_id="skilled", skill_mode=SkillMode.GENERIC_TEXT)
    assert require_paired_controls(b, b, trained, skilled) == "skill-access"
    with pytest.raises(ValueError):
        require_paired_controls(a, b, base, skilled)


@pytest.mark.parametrize(
    "field", ["tokenizer", "service", "tools", "evaluator", "sampling", "budgets", "parser"]
)
def test_weight_comparison_does_not_excuse_changes_in_other_runtime_controls(field):
    base, trained = policy_arms()
    a, b = policy_controls(base), policy_controls(trained)
    with pytest.raises(ValueError):
        require_paired_controls(a, replace(b, **{field: {"different": "setting"}}), base, trained)


@pytest.mark.parametrize(
    "defect",
    [
        "other-base",
        "other-tokenizer",
        "missing-route",
        "wrong-step",
        "wrong-policy",
        "base-descriptor",
    ],
)
def test_weight_comparison_rejects_mislabeled_model_state(defect):
    base, trained = policy_arms()
    a, b = policy_controls(base), policy_controls(trained)
    model = deepcopy(b.model)
    if defect == "other-base":
        model["observed"]["model_path"] = "/synthetic/other-base"
    elif defect == "other-tokenizer":
        model["policy"]["tokenizer_id"] = "different-tokenizer"
    elif defect == "missing-route":
        model["adapter_route_sent"] = None
    elif defect == "wrong-step":
        model["checkpoint"]["optimizer_steps"] = 15
    elif defect == "wrong-policy":
        model["policy"]["snapshot_id"] = "another-policy"
    else:
        model = a.model
    with pytest.raises(ValueError):
        require_paired_controls(a, replace(b, model=model), base, trained)


class PolicyRuntime(Runtime):
    def controls(self, arm, entries):
        return replace(super().controls(arm, entries), model=policy_controls(arm).model)

    async def generate(self, entry, arm, run_id):
        return replace(
            await super().generate(entry, arm, run_id), policy_id=arm.policy_id or "synthetic-base"
        )

    def validate_candidate(self, reader, entry, arm, run_id):
        final = reader.get(run_id, arm.arm_id, entry.task_id)
        if final.policy_id != (arm.policy_id or "synthetic-base"):
            raise ValueError("synthetic stored final uses another policy")


@pytest.mark.parametrize("paired", [True, False])
def test_pipeline_runs_the_explicit_policy_without_changing_population_or_score(tmp_path, paired):
    base, trained = policy_arms()
    arms = (base, trained) if paired else (trained,)
    instance = PolicyRuntime()
    report = asyncio.run(
        run_paired(
            instance, panel(), arms, run_id="synthetic-policy-run", directory=tmp_path, canary=True
        )
    )
    assert instance.generated == instance.graded == len(panel().entries) * len(arms)
    assert instance.closed == 1
    assert report["schema"].startswith("policy-integrity-")
    if paired:
        comparison = report["comparisons"]["aime-2026"]
        assert comparison["intervention"] == "forward-policy"
        assert comparison["delta"] == 0
        assert comparison["performance_goal_status"] == "not-improved"


def test_trained_resume_keeps_original_answers_and_rejects_checkpoint_rebinding(tmp_path):
    arms = policy_arms()
    with pytest.raises(RuntimeError):
        asyncio.run(
            run_paired(
                PolicyRuntime(fail_score=True),
                panel(),
                arms,
                run_id="synthetic-resume",
                directory=tmp_path,
                canary=True,
            )
        )

    class ChangedCheckpoint(PolicyRuntime):
        def controls(self, arm, entries):
            original = super().controls(arm, entries)
            if not arm.optimizer_steps:
                return original
            model = deepcopy(original.model)
            model["checkpoint"]["checkpoint_directory"] = "/synthetic/different-checkpoint"
            return replace(original, model=model)

    changed = ChangedCheckpoint()
    with pytest.raises(ValueError):
        asyncio.run(
            run_paired(
                changed,
                panel(),
                arms,
                run_id="synthetic-resume",
                directory=tmp_path,
                canary=True,
                run_mode=EvaluationRunMode.SAME_RUN_RESUME,
            )
        )
    assert changed.generated == 0
    recovered = PolicyRuntime()
    asyncio.run(
        run_paired(
            recovered,
            panel(),
            arms,
            run_id="synthetic-resume",
            directory=tmp_path,
            canary=True,
            run_mode=EvaluationRunMode.SAME_RUN_RESUME,
        )
    )
    assert recovered.generated == 0
    assert recovered.graded > 0


class RouteTransport:
    def __init__(self, outputs, *, fail=False):
        self.outputs, self.fail, self.calls = outputs, fail, []

    async def request(self, *, payload, **kwargs):
        self.calls.append(payload)
        if self.fail:
            return 503, {"error": "synthetic unavailable route"}
        text = self.outputs[payload.get("lora_path")].pop(0)
        return 200, {
            "output_ids": [ord(char) for char in text],
            "meta_info": {
                "prompt_tokens": len(payload["input_ids"]),
                "completion_tokens": len(text),
                "finish_reason": {"type": "stop"},
            },
        }


def routed_runtime(tmp_path, *, fail=False):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    instance = runtime(tmp_path, entry, [])
    binding = binding_fixture(tmp_path)
    instance.config.update(
        endpoints=["http://127.0.0.1:1"], request_timeout_seconds=5, concurrency=2
    )
    instance.trained_policies = {binding.policy.snapshot_id: binding}
    instance.generators = instance._make_generators(instance.policy, None)
    instance.trained_generators = {
        binding.policy.snapshot_id: instance._make_generators(binding.policy, binding.adapter_name)
    }
    transport = RouteTransport(
        {
            None: ["Final answer: 17"],
            binding.adapter_name: [
                "Message to solver:\nHelp with this task.",
                "Review:\nCheck the available public evidence.",
                "Final answer: 42",
            ],
        },
        fail=fail,
    )
    for generator in instance.generators + instance.trained_generators[binding.policy.snapshot_id]:
        generator.transport = transport
    return instance, binding, transport, entry


def test_concurrent_base_and_trained_owners_use_separate_real_transport_routes(tmp_path):
    instance, binding, transport, entry = routed_runtime(tmp_path)
    base, trained = policy_arms()
    original_sandbox = instance.sandbox
    observed_initial = []

    class RecordingSandbox:
        async def run(self, initial, handle, **kwargs):
            observed_initial.append(initial)
            return await original_sandbox.run(initial, handle, **kwargs)

    instance.sandbox = RecordingSandbox()

    async def run():
        try:
            finals = await asyncio.gather(
                *(instance.generate(entry, arm, "synthetic") for arm in (base, trained))
            )
            assert [final.text for final in finals] == [r"\boxed{17}", r"\boxed{42}"]
            for arm in (base, trained):
                instance.validate_candidate(instance.journal, entry, arm, "synthetic")
                outputs = instance.journal.model_outputs(("synthetic", arm.arm_id, entry.task_id))
                assert all(
                    row.policy_id
                    == (
                        binding.policy.snapshot_id if arm.policy_id else instance.policy.snapshot_id
                    )
                    for row in outputs
                )
                assert all(
                    row.adapter_name == (binding.adapter_name if arm.policy_id else None)
                    for row in outputs
                )
                if arm.policy_id:
                    assert all(row.participant == "owner" for row in outputs)
                    assert finals[1].intervention_counts["peer_model_calls"] == 0
                    assert finals[1].completion_tokens == sum(
                        row.result["usage"]["output_tokens"] for row in outputs
                    )
            assert [payload.get("lora_path") for payload in transport.calls].count(
                binding.adapter_name
            ) == 3
            assert [payload.get("lora_path") for payload in transport.calls].count(None) == 1
            assert str(binding.checkpoint_directory) not in json.dumps(observed_initial)
            trained_initial = next(
                value
                for value in observed_initial
                if value["arm"]["condition_id"] == trained.arm_id
            )
            assert trained_initial["inference_state"]["forward_adapter_active"]
            assert trained_initial["inference_state"]["optimizer_steps"] == 16
        finally:
            await instance.close()

    asyncio.run(run())


def test_failed_trained_route_is_not_retried_or_replaced_by_a_base_answer(tmp_path):
    instance, binding, transport, entry = routed_runtime(tmp_path, fail=True)
    _, trained = policy_arms()

    async def run():
        try:
            with pytest.raises(EvaluationSGLangGenerationError):
                await instance.generate(entry, trained, "synthetic")
            assert len(transport.calls) == 1
            assert transport.calls[0]["lora_path"] == binding.adapter_name
            assert not instance.journal.model_outputs(("synthetic", trained.arm_id, entry.task_id))
        finally:
            await instance.close()

    asyncio.run(run())
