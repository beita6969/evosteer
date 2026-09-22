from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from skillev.contracts import (
    InitialContext,
    SuccessRule,
    TerminalReward,
    TrajectoryRecord,
    TrajectoryStep,
    build_trajectory_record,
    stable_hash,
)
from skillev.policy import (
    QwenBackboneConfig,
    QwenPolicyBackbone,
    encode_rollout_prompt,
    qwen_tokenizer_artifact_identity,
)
from skillev.scoring import (
    assembled_context_hash,
    render_forward_prefix,
    render_hindsight_prefix,
)


@dataclass(frozen=True, slots=True)
class ScoringCase:
    initial_text: str
    record: TrajectoryRecord


ScoringCaseFactory = Callable[..., ScoringCase]


@pytest.fixture(scope="session")
def scoring_tiny_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build an offline random causal LM with enough context for two rendered steps."""

    model_path = tmp_path_factory.mktemp("tiny-scoring-backbone")
    ordinary_tokens = (
        "query",
        "skill",
        "omega",
        "reasonone",
        "thoughtone",
        "reasontwo",
        "thoughttwo",
        "observeone",
        "resultone",
        "observetwo",
        "resulttwo",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "empty",
        "changed",
        "legacy",
        "Step",
        "Reasoning",
        "Action",
        "Observation",
        "1",
        "2",
        "#",
        ":",
    )
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        **{token: index + 4 for index, token in enumerate(ordinary_tokens)},
    }
    tokenizer_backend = Tokenizer(
        WordLevel(vocab=vocabulary, unk_token="[UNK]"),
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} "
        "{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %} assistant{% endif %}"
    )
    tokenizer.save_pretrained(model_path)

    config = GPT2Config(
        vocab_size=len(vocabulary),
        n_positions=512,
        n_ctx=512,
        n_embd=24,
        n_layer=1,
        n_head=2,
        bos_token_id=vocabulary["[BOS]"],
        eos_token_id=vocabulary["[EOS]"],
        pad_token_id=vocabulary["[PAD]"],
        use_cache=True,
    )
    with torch.random.fork_rng():
        torch.manual_seed(20260720)
        GPT2LMHeadModel(config).save_pretrained(model_path)
    return model_path


@pytest.fixture
def scoring_backbone_config(scoring_tiny_model_path: Path) -> QwenBackboneConfig:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(scoring_tiny_model_path)
    return QwenBackboneConfig(
        base_model_path=str(scoring_tiny_model_path),
        revision="local-pinned",
        tokenizer_id="tiny-scoring-tokenizer@1",
        tokenizer_content_hash=qwen_tokenizer_artifact_identity(
            tokenizer=tokenizer,
            tokenizer_id="tiny-scoring-tokenizer@1",
            revision="local-pinned",
        ).content_hash,
        hidden_size=24,
        eos_token_ids=(int(GPT2Config.from_pretrained(scoring_tiny_model_path).eos_token_id),),
        torch_dtype="float32",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.35,
        lora_target_modules=("c_attn",),
        z_hidden_width=8,
        device="cpu",
    )


@pytest.fixture
def scoring_backbone(
    scoring_backbone_config: QwenBackboneConfig,
) -> QwenPolicyBackbone:
    """Exercise the production Qwen policy class."""

    return QwenPolicyBackbone(scoring_backbone_config)


@pytest.fixture
def make_scoring_case(
    scoring_backbone: QwenPolicyBackbone,
) -> ScoringCaseFactory:
    def factory(
        *,
        initial_text: str = "query skill omega\n",
        reasoning_texts: tuple[str, ...] = (
            "reasonone thoughtone",
            "reasontwo thoughttwo",
        ),
        action_texts: tuple[str, ...] = (
            "alpha beta gamma",
            "delta epsilon zeta",
        ),
        observation_texts: tuple[str, ...] = (
            "observeone resultone",
            "observetwo resulttwo",
        ),
        reward_value: float = 0.7,
        epsilon_min: float = 0.05,
    ) -> ScoringCase:
        if not (len(reasoning_texts) == len(action_texts) == len(observation_texts)):
            raise ValueError("step fixture fields must have the same length")

        placeholder_steps_list: list[TrajectoryStep] = []
        for index, (reasoning_text, action_text, observation_text) in enumerate(
            zip(reasoning_texts, action_texts, observation_texts, strict=True),
            start=1,
        ):
            action_token_ids = tuple(scoring_backbone.tokenizer.encode(action_text))
            placeholder_steps_list.append(
                TrajectoryStep(
                    index=index,
                    reasoning_text=reasoning_text,
                    action_text=action_text,
                    action_token_ids=action_token_ids,
                    action_token_count=len(action_token_ids),
                    observation_text=observation_text,
                    observation_status="success",
                    invoked_skill_ids=(),
                    forward_prefix_hash=stable_hash(
                        {"fixture": "uncommitted-forward", "step": index}
                    ),
                    hindsight_prefix_hash=stable_hash(
                        {"fixture": "uncommitted-hindsight", "step": index}
                    ),
                )
            )
        placeholder_steps = tuple(placeholder_steps_list)
        steps = tuple(
            replace(
                step,
                forward_prefix_hash=render_forward_prefix(
                    initial_text,
                    placeholder_steps,
                    step.index,
                ).prefix_hash,
                hindsight_prefix_hash=render_hindsight_prefix(
                    initial_text,
                    placeholder_steps,
                    step.index,
                ).prefix_hash,
            )
            for step in placeholder_steps
        )
        reward = TerminalReward(
            value=reward_value,
            success=reward_value >= 0.5,
            success_rule=SuccessRule.R_AT_THRESHOLD,
            success_threshold=0.5,
            native_metric_name="tiny-score",
            native_payload={"score": reward_value},
            environment_id="tiny-environment",
            verifier_version="tiny-verifier-v1",
        )
        context = InitialContext(
            query="query",
            retrieved_skill_ids=("skill-alpha",),
            active_skill_ids=("skill-alpha",),
            meta={
                "environment_id": "tiny-environment",
                "fixture": "phase-three-scoring",
                "task_family": "tiny-scoring",
            },
            assembler_version="ttb-test-assembler-v1",
            assembled_hash=assembled_context_hash(initial_text),
            assembled_token_count=len(
                encode_rollout_prompt(scoring_backbone.tokenizer, initial_text)
            ),
        )
        record = build_trajectory_record(
            tokenizer=scoring_backbone.tokenizer,
            trajectory_id="trajectory-scoring-two-step",
            environment_id="tiny-environment",
            task_family="tiny-scoring",
            initial_context=context,
            steps=steps,
            horizon=len(steps),
            reward=reward,
            shifted_reward=reward.value + epsilon_min,
            epsilon_min=epsilon_min,
            tokenizer_id=scoring_backbone.tokenizer.tokenizer_id,
            decoding_snapshot_id="tiny-decode-v1",
            created_at="2026-07-20T00:00:00Z",
        )
        return ScoringCase(initial_text=initial_text, record=record)

    return factory


@pytest.fixture
def scoring_case(make_scoring_case: ScoringCaseFactory) -> ScoringCase:
    return make_scoring_case()
