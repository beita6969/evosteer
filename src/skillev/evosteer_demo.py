"""Offline synthetic integration exercise, with a real tiny HF/PEFT actor.

The executor's outputs and resource counts are synthetic. This fixture checks
the training machinery and is never a benchmark or a paper result.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from skillev.contracts.evosteer import EvoTask
from skillev.evolution.validated_admission import AdmissionLedger, SkillEntry
from skillev.evosteer_application import (
    EvoSteerApplication,
    EvoSteerConfig,
    ResetReceipt,
    SessionRequest,
    TaskBinding,
    TaskSession,
)
from skillev.orchestration.graph import NodeExecutionRequest, NodeExecutionResult, RoleSpec
from skillev.policy.evosteer import CausalLMOrchestrator
from skillev.rollout.evosteer_risk import ExecutionRiskPolicy
from skillev.runtime import BudgetVector
from skillev.training.evosteer import EvoOptimizerConfig

EOS_TEXT = "<|endoftext|>"
UNKNOWN_TEXT = "[UNK]"


class RepairExecutor:
    """Deterministic feedback world; counts below are declared synthetic units."""

    frozen_identity = "synthetic-repair-executor@1"

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        if request.role.role_id == "verifier":
            output = "correction"
        else:
            output = (
                "fixed"
                if request.skills or any(m["body"] == "correction" for m in request.messages)
                else "draft"
            )
        return NodeExecutionResult(
            output,
            BudgetVector(input_tokens=10, output_tokens=2, model_calls=1, agent_turns=1),
            {"synthetic": True},
        )


def build_tiny_policy(*, seed: int = 14) -> CausalLMOrchestrator:
    """Construct random weights and a byte tokenizer entirely in memory."""
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import (  # type: ignore[import-untyped]
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
    )
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {
        EOS_TEXT: 0,
        UNKNOWN_TEXT: 1,
        **{
            char: index + 2
            for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        },
    }
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=UNKNOWN_TEXT))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token=EOS_TEXT,
        unk_token=UNKNOWN_TEXT,
        pad_token=EOS_TEXT,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model: Any = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_embd=16,
                n_layer=1,
                n_head=2,
                n_positions=1024,
                eos_token_id=0,
                bos_token_id=0,
                pad_token_id=0,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["c_proj"],
                task_type="CAUSAL_LM",
                bias="none",
                fan_in_fan_out=True,
            ),
        )
    return CausalLMOrchestrator(
        model,
        tokenizer,
        reference_id=f"synthetic-random-gpt2@1/seed-{seed}",
        encoding_dim=16,
        context_window=768,
        max_action_tokens=256,
        context_mode="debug_head_tail",
        model_pin=f"synthetic-random-gpt2@1/seed-{seed}",
    )


def build_smoke_application(
    *, seed: int = 14
) -> tuple[EvoSteerApplication, tuple[TaskBinding, ...]]:
    policy = build_tiny_policy(seed=seed)
    maximum = BudgetVector(
        input_tokens=2048,
        output_tokens=128,
        model_calls=1,
        agent_turns=1,
        wall_time_milliseconds=1000,
    )
    config = EvoSteerConfig(
        task_families=("synthetic",),
        roles=(
            RoleSpec("solver", "Solve the synthetic task using available feedback.", maximum),
            RoleSpec("verifier", "Check the synthetic draft and return a correction.", maximum),
        ),
        optimizer=EvoOptimizerConfig(actor_learning_rate=0.005),
        max_nodes=2,
        max_actions=6,
        current_rollouts=1,
        reference_rollouts=1,
        seed=seed,
        total_token_cap=16_384,
    )
    ledger = AdmissionLedger()
    body = "Check the draft against feedback before returning the result."
    ledger.propose(
        SkillEntry(
            "synthetic-check-refine",
            "synthetic",
            "sha256:" + hashlib.sha256(body.encode()).hexdigest(),
            body=body,
        ),
        author_window_id="synthetic-fixture@1",
    )

    def session(request: SessionRequest) -> TaskSession:
        return TaskSession(
            RepairExecutor(),
            lambda output: float(output == "fixed"),
            reset_receipt=ResetReceipt(
                request.task.identity,
                request.task.reset_id,
                request.task.environment_config_id,
                request.task.identity,
                str(uuid.uuid4()),
                request.seed,
            ),
            risk_assessor=ExecutionRiskPolicy(
                RepairExecutor.frozen_identity,
                request.task.environment_config_id,
                scope="text_only",
                capability_id="synthetic-in-memory-executor@1",
            ),
        )

    binding = TaskBinding(
        EvoTask(
            "synthetic-repair-1",
            "synthetic",
            "Produce a checked result.",
            environment_config_id="synthetic-repair-world@1",
        ),
        RepairExecutor.frozen_identity,
        session,
        replay_safe=True,
    )
    return EvoSteerApplication(policy, config, admission=ledger), (binding,)
