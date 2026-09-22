"""Active conditions expose one owner; historical topology records stay readable."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from skillev_private.evaluation import integrity_runtime

from skillev.evaluation.direct_baseline import DirectGenerationRequest
from skillev.evaluation.step0_integrity import (
    AgentTopology,
    InferenceArm,
    SkillMode,
    ToolCallMode,
    decode_integrity_arm,
    load_integrity_arm,
)
from skillev.rollout import GenerationPhase
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.evaluation.test_owner_output_provenance import entry
from tests.evaluation.test_step0_architecture import _Generator, _profile
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer, clean_client


@pytest.mark.parametrize(
    "filename",
    [
        "step0_integrity_v1.yaml",
        "step0_integrity_v2.yaml",
        "step0_integrity_native_tools.yaml",
        "step0_integrity_native_interaction.yaml",
        "step0_integrity_owner_native_thinking.yaml",
    ],
)
def test_registered_active_profiles_have_only_one_owner(filename):
    arm = load_integrity_arm(Path("configs/evaluation") / filename)
    arm.validate_live_topology()
    assert arm.agent_topology is AgentTopology.SINGLE
    if filename == "step0_integrity_native_interaction.yaml":
        settings = yaml.safe_load(
            Path("configs/evaluation/step0_native_interaction_repairs.yaml").read_text()
        )
        assert arm.arm_id == settings["condition_id"]


@pytest.mark.parametrize("trained", [False, True])
def test_old_multi_agent_arm_decodes_but_cannot_start_a_live_attempt(tmp_path, trained):
    arm = InferenceArm("historical", agent_topology=AgentTopology.MULTI_AGENT)
    if trained:
        arm = replace(arm, optimizer_steps=16, policy_id="synthetic-trained")
    assert decode_integrity_arm(arm.to_value()) == arm
    instance = runtime(tmp_path, entry(), [])
    try:
        with pytest.raises(ValueError):
            asyncio.run(instance.generate(entry(), arm, "synthetic"))
        assert not instance.journal.model_outputs(("synthetic", arm.arm_id, "case"))
        assert (
            instance.journal.connection.execute("SELECT count(*) FROM episode_attempts").fetchone()[
                0
            ]
            == 0
        )
    finally:
        instance.journal.close()


def test_old_topology_is_rejected_before_contacting_model_services(monkeypatch, tmp_path):
    def unexpected_service_call(*args, **kwargs):
        pytest.fail("retired topology contacted an inference service")

    monkeypatch.setattr(integrity_runtime, "_server", unexpected_service_call)
    config = {
        "transport_attempts": 1,
        "arms": [InferenceArm("old", agent_topology=AgentTopology.MULTI_AGENT).to_value()],
    }
    with pytest.raises(ValueError):
        integrity_runtime.PrivateIntegrityRuntime(config, SimpleNamespace(), tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mode", list(ToolCallMode))
@pytest.mark.parametrize("skills", [SkillMode.OFF, SkillMode.GENERIC_TEXT])
def test_owner_tools_and_rendered_requests_do_not_offer_consultants(mode, skills):
    generator = _Generator([(GenerationPhase.ACTION, "42")], tokenizer=CleanTokenizer())
    arm = InferenceArm("owner-only", skill_mode=skills, tool_call_mode=mode)
    client = clean_client(generator, arm)
    client.generation.call_limit = 3
    client.generation.output_token_limit = 4096
    result = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case", ({"role": "user", "content": "Compute the synthetic integer."},), _profile()
            )
        )
    )
    assert result.text == r"\boxed{42}"
    assert client.counts.model_calls == 1
    assert client.counts.peer_model_calls == client.counts.peer_messages == 0
    assert all(call.participant == "owner" for call in client.generation.calls)
    for messages, _ in generator.tokenizer.messages:
        rendered = "\n".join(row["content"] for row in messages)
        assert not any(
            word in rendered.casefold() for word in ("solver", "researcher", "peer", "consultant")
        )
