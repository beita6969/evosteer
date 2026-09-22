"""Declared balanced-domain training condition, independent of deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import cast

import yaml

from skillev.application_config import ApplicationConfig
from skillev.calibration import CalibrationConfig
from skillev.contracts import JsonValue, normalize_json
from skillev.contracts.action_wire import NATIVE_TOOL_WIRES
from skillev.contracts.skill_exposure import AUTONOMOUS_CATALOG_EXPOSURES
from skillev.diagnostics import DiagnosticsConfig
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.external_judge_policy import (
    DIRECT_HEALTHBENCH_PROFILE,
    GATEWAY_HEALTHBENCH_PROFILE,
)
from skillev.evaluation.healthbench_luna_profile import LEGACY_TRAINING_JUDGE, PROFILE_ID
from skillev.evolution.cold_start_config import ColdStartConfig
from skillev.policy import QwenMultimodalBackboneConfig
from skillev.policy.interface import INPUT_WINDOW_VERSION, ModelInputWindow
from skillev.rollout import RolloutBudgetProfile
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.task_semantic_guidance import (
    LEGACY_TASK_SEMANTICS,
    validate_task_semantic_guidance,
)
from skillev.training import (
    OptimizerConfig,
    PolicyRolloutConfig,
    TTBMethodConfig,
    conservative_rollout_maximum,
)
from skillev_private.benchmarks.protocol_v13_seven_training import (
    SEVEN_TRAINING_DOMAINS,
    TRAJECTORIES_PER_QUESTION,
    domain_schedule_condition,
)

from .bayesian_training_setup import build_application_config


@dataclass(frozen=True, slots=True)
class BayesianFormalConfig:
    domains: tuple[str, ...] = tuple(d.value for d in SEVEN_TRAINING_DOMAINS)
    format: str = "skillev-bayesian-formal-training@3"
    phase_context: bool = False
    reasoning_tool_catalog: bool = False
    token_budget_notice: bool = False
    action_wire: str = "structured-action-json@3"
    skill_exposure: str = "full-inline"
    initial_skill_profile: str = "public-advisory@2"
    # Absent on historical SFT/teaching/zero-coverage-extension experiments.
    learning_protocol: str | None = None
    cold_start: ColdStartConfig | None = None
    # Missing on historical configurations means the original local judge.
    healthbench_judge: str = LEGACY_TRAINING_JUDGE
    format_review_from_step: int | None = None
    public_action_semantics: bool = False
    task_semantic_guidance: str = LEGACY_TASK_SEMANTICS
    model: str = "Qwen3.5-9B"
    base_dtype: str = "bfloat16"
    lora_rank: int = 4
    lora_alpha: int = 8
    seed: int = 0
    steps: int = 250
    closure_steps: int = 1
    maximum_cycles: int = 2
    checkpoint_every: int = 10
    max_turns: int = 20
    static_max_turns: int = 8
    max_reasoning_tokens: int = 1024
    reasoning_tokens_by_domain: tuple[tuple[str, int], ...] = ()
    max_action_tokens: int = 2048
    action_tokens_by_domain: tuple[tuple[str, int], ...] = ()
    max_input_tokens: int = 65_536
    input_window: str = INPUT_WINDOW_VERSION
    adapter_learning_rate: float = 0.0001
    z_learning_rate: float = 0.0001
    weight_decay: float = 0.0
    gradient_clipping: bool = False
    extra_kl: float = 0.0
    ttb_beta: float = 1.0
    epsilon: float = 0.1
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    window: int = 50
    rho: float = 0.05
    k: float = 1.0
    reasoning_native_thinking: bool = True
    thinking_off_domains: tuple[str, ...] = ()
    hotpot_deliberation: bool = True
    phi_calls_per_cycle: int = 64
    performance_profile: str = "configs/training/protocol13_turn_latency.yaml"
    planning_hours: float = 72.0
    target_steps_per_hour: float = 4.2

    def __post_init__(self) -> None:
        available = tuple(d.value for d in SEVEN_TRAINING_DOMAINS)
        domains = tuple(self.domains)
        if not domains or tuple(d for d in available if d in domains) != domains:
            raise ValueError("active domains must be a nonempty canonical subset")
        object.__setattr__(self, "domains", domains)
        if self.initial_skill_profile not in {
            "public-advisory@2",
            "public-procedure-advice@3",
            "public-native-procedures@4",
            "public-method-cards@5",
        }:
            raise ValueError("unknown initial skill profile")
        if self.initial_skill_profile != "public-advisory@2" and (
            self.skill_exposure not in AUTONOMOUS_CATALOG_EXPOSURES
            or not self.format.endswith("@6")
        ):
            raise ValueError(
                "new advisory library requires an explicit autonomous catalog candidate"
            )
        if self.learning_protocol is not None:
            from .autonomous_ttb import require_autonomous_config

            require_autonomous_config(self)
        if self.healthbench_judge not in {
            LEGACY_TRAINING_JUDGE,
            DIRECT_HEALTHBENCH_PROFILE,
            GATEWAY_HEALTHBENCH_PROFILE,
            PROFILE_ID,
        }:
            raise ValueError("unsupported HealthBench scoring condition")
        if self.format_review_from_step is not None:
            from skillev.evaluation.format_content_review import format_review_condition

            format_review_condition(self.format_review_from_step)
            if self.format_review_from_step > self.steps:
                raise ValueError("format review starts beyond the declared training plan")
        if isinstance(self.cold_start, dict):
            object.__setattr__(self, "cold_start", ColdStartConfig.from_value(self.cold_start))
        if self.cold_start is not None:
            if not isinstance(self.cold_start, ColdStartConfig):
                raise TypeError("cold_start must be an explicit method-extension configuration")
            if self.skill_exposure != "catalog-then-read@1":
                raise ValueError("cold start requires optional catalog reads, not a call quota")
        if type(self.token_budget_notice) is not bool or (
            self.token_budget_notice and not self.phase_context
        ):
            raise ValueError("token budget notice requires phase context")
        if type(self.reasoning_tool_catalog) is not bool or (
            self.reasoning_tool_catalog
            and (not self.phase_context or self.action_wire not in NATIVE_TOOL_WIRES)
        ):
            raise ValueError("reasoning tool catalog requires native phase context")
        if self.format not in {
            "skillev-bayesian-formal-training@2",
            "skillev-bayesian-formal-training@3",
            "skillev-bayesian-formal-training@4",
            "skillev-bayesian-formal-training@5",
            "skillev-bayesian-formal-training@6",
        }:
            raise ValueError("unsupported formal training condition")
        if not self.format.endswith(("@4", "@5", "@6")) and (
            self.phase_context
            or self.action_wire != "structured-action-json@3"
            or self.skill_exposure != "full-inline"
        ):
            raise ValueError("interface candidates require formal condition version 4")
        validate_task_semantic_guidance(self.task_semantic_guidance)
        if self.task_semantic_guidance != LEGACY_TASK_SEMANTICS and not self.format.endswith("@6"):
            raise ValueError("shared task semantics require formal candidate version 6")
        if self.task_semantic_guidance != LEGACY_TASK_SEMANTICS and (
            self.action_wire in NATIVE_TOOL_WIRES and not self.public_action_semantics
        ):
            raise ValueError("shared native task guidance requires public action semantics")
        if self.public_action_semantics and not self.format.endswith(("@5", "@6")):
            raise ValueError("public semantics require formal candidate version 5")
        if not isinstance(self.thinking_off_domains, tuple | list) or any(
            type(v) is not str for v in self.thinking_off_domains
        ):
            raise TypeError("thinking-off domains must be a sequence of domain names")
        domains = tuple(self.thinking_off_domains)
        if len(set(domains)) != len(domains) or not set(domains) <= {
            d.value for d in SEVEN_TRAINING_DOMAINS
        }:
            raise ValueError(
                "thinking-off domains must be unique members of the seven-domain schedule"
            )
        object.__setattr__(self, "thinking_off_domains", tuple(sorted(domains)))
        for field in ("reasoning_tokens_by_domain", "action_tokens_by_domain"):
            limits = tuple(tuple(item) for item in getattr(self, field))
            if any(
                len(item) != 2
                or item[0] not in {d.value for d in SEVEN_TRAINING_DOMAINS}
                or type(item[1]) is not int
                or item[1] < 1
                for item in limits
            ) or len({item[0] for item in limits}) != len(limits):
                raise ValueError("domain budgets need unique domains and positive token caps")
            if limits and not self.format.endswith("@6"):
                raise ValueError("domain budgets require an explicit current candidate")
            object.__setattr__(self, field, tuple(sorted(limits)))
        if self.format.endswith("@2") and domains:
            raise ValueError("legacy formal condition cannot declare mixed thinking")
        if self.model != "Qwen3.5-9B" or self.base_dtype != "bfloat16":
            raise ValueError("formal training requires the declared single Qwen BF16 owner")
        if self.seed != 0 or type(self.seed) is not int:
            raise ValueError("this formal schedule uses the single declared seed zero")
        for name in (
            "lora_rank",
            "lora_alpha",
            "steps",
            "closure_steps",
            "maximum_cycles",
            "checkpoint_every",
            "max_turns",
            "static_max_turns",
            "max_reasoning_tokens",
            "max_action_tokens",
            "max_input_tokens",
            "window",
            "phi_calls_per_cycle",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.steps > 250 or not 1 <= self.closure_steps < self.steps:
            raise ValueError("formal schedule needs search steps and a positive closure tail")
        if self.static_max_turns > self.max_turns:
            raise ValueError("static domain horizon exceeds the global cap")
        if self.weight_decay != 0 or self.gradient_clipping is not False or self.extra_kl != 0:
            raise ValueError("declared TTB has no decay, clipping or added KL objective")
        if self.reasoning_native_thinking is not True or type(self.hotpot_deliberation) is not bool:
            raise ValueError("formal owner reasoning must actually enable native thinking")
        if self.planning_hours <= 0 or self.target_steps_per_hour <= 0:
            raise ValueError("planning estimates must be positive, not hard deadlines")
        # Existing scientific configuration types remain the numerical authority.
        _ = self.sampling_config, self.run_plan
        TTBMethodConfig(self.epsilon, self.ttb_beta)
        OptimizerConfig(self.adapter_learning_rate, self.z_learning_rate, self.weight_decay)
        CalibrationConfig(self.prior_alpha, self.prior_beta, self.k)
        DiagnosticsConfig(window_size=self.window, stagnation_rho=self.rho)

    @property
    def run_plan(self) -> ExactAttemptRunPlan:
        return ExactAttemptRunPlan(
            self.steps - self.closure_steps, self.closure_steps, self.maximum_cycles
        )

    @property
    def batch_size(self) -> int:
        return len(self.domains) * TRAJECTORIES_PER_QUESTION

    @property
    def scheduled_domains(self) -> tuple[Protocol13Benchmark, ...]:
        return tuple(d for d in SEVEN_TRAINING_DOMAINS if d.value in self.domains)

    @property
    def condition(self) -> str:
        suffix = "+hotpot-evidence-deliberation@1" if self.hotpot_deliberation else ""
        if self.healthbench_judge != LEGACY_TRAINING_JUDGE:
            suffix += "+" + self.healthbench_judge
        if self.format_review_from_step is not None:
            from skillev.evaluation.format_content_review import FORMAT_REVIEW_PROFILE

            suffix += f"+{FORMAT_REVIEW_PROFILE}-from{self.format_review_from_step}"
        if not self.format.endswith("@2"):
            suffix += "+domain-thinking@1-off=" + ",".join(self.thinking_off_domains)
        if self.format.endswith(("@4", "@5", "@6")):
            suffix += (
                f"+interface@1-phase={int(self.phase_context)}"
                f"-wire={self.action_wire}-skills={self.skill_exposure}"
                + (
                    f"-cold-start={self.cold_start.version}"
                    f"-sources={self.cold_start.min_source_questions}"
                    f"-batches={self.cold_start.min_batches}"
                    f"-new-skills={self.cold_start.max_new_skills}"
                    f"-witness-edges={self.cold_start.max_evidence_edges}"
                    if self.cold_start
                    else ""
                )
            )
        if self.public_action_semantics:
            suffix += "+public-action-semantics@1"
        if self.token_budget_notice:
            suffix += "+token-budget-notice@1"
        if self.reasoning_tool_catalog:
            suffix += "+reasoning-tool-catalog@1"
        if self.task_semantic_guidance != LEGACY_TASK_SEMANTICS:
            suffix += "+" + self.task_semantic_guidance
        if self.reasoning_tokens_by_domain:
            suffix += "+domain-reasoning-budgets@1=" + ",".join(
                f"{domain}:{tokens}" for domain, tokens in self.reasoning_tokens_by_domain
            )
            suffix += f"-static{self.static_max_turns}-interactive{self.max_turns}"
        if self.initial_skill_profile != "public-advisory@2":
            suffix += "+initial-skills=" + self.initial_skill_profile
        if self.learning_protocol is not None:
            suffix += "+" + self.learning_protocol
        if self.action_tokens_by_domain:
            suffix += "+domain-action-budgets@1=" + ",".join(
                f"{domain}:{tokens}" for domain, tokens in self.action_tokens_by_domain
            )
        return (
            domain_schedule_condition(self.scheduled_domains)
            + "+domain-horizons-native-reasoning@2"
            + "+completion-wire@2+"
            + self.input_window
            + suffix
        )

    @property
    def task_budget(self) -> RolloutBudgetProfile:
        return RolloutBudgetProfile(
            "formal-interactive-budget@2",
            self.max_turns,
            self.max_reasoning_tokens,
            self.max_action_tokens,
        )

    @property
    def static_task_budget(self) -> RolloutBudgetProfile:
        return RolloutBudgetProfile(
            "formal-static-budget@2",
            self.static_max_turns,
            self.max_reasoning_tokens,
            self.max_action_tokens,
        )

    @property
    def maximum_reasoning_tokens(self) -> int:
        """Global reservation envelope, not every task's generation allowance."""
        return max((self.max_reasoning_tokens, *(v for _, v in self.reasoning_tokens_by_domain)))

    @property
    def maximum_action_tokens(self) -> int:
        """Reservation envelope only; task-specific decoding keeps its own cap."""
        return max((self.max_action_tokens, *(v for _, v in self.action_tokens_by_domain)))

    @property
    def domain_task_budgets(self) -> dict[str, RolloutBudgetProfile]:
        reasoning, action = (
            dict(self.reasoning_tokens_by_domain),
            dict(self.action_tokens_by_domain),
        )
        return {
            domain: replace(
                self.task_budget if domain == "alfworld" else self.static_task_budget,
                profile_id=(
                    "formal-domain-phase-budget@2:"
                    if self.action_tokens_by_domain
                    else "formal-domain-reasoning-budget@1:"
                )
                + domain,
                max_reasoning_tokens=reasoning.get(domain, self.max_reasoning_tokens),
                max_action_tokens=action.get(domain, self.max_action_tokens),
            )
            for domain in sorted(reasoning.keys() | action.keys())
        }

    @property
    def sampling_config(self) -> PolicyRolloutConfig:
        """Inference controls only; no TrainerConfig, optimizer or model construction."""
        return PolicyRolloutConfig(
            base_seed=self.seed,
            max_turns=self.max_turns,
            max_reasoning_tokens=self.maximum_reasoning_tokens,
            max_action_tokens=self.maximum_action_tokens,
            reasoning_native_thinking=self.reasoning_native_thinking,
            per_rollout_maximum=conservative_rollout_maximum(
                max_turns=self.max_turns,
                max_reasoning_tokens=self.maximum_reasoning_tokens,
                max_action_tokens=self.maximum_action_tokens,
                max_model_input_tokens=self.max_input_tokens,
                max_tool_wall_time_milliseconds=120_000,
            ),
            format="skillev-policy-rollout@9"
            if self.format.endswith("@6")
            else "skillev-policy-rollout@8"
            if self.format.endswith(("@5", "@6"))
            else "skillev-policy-rollout@7"
            if self.format.endswith("@4")
            else "skillev-policy-rollout@6"
            if self.format.endswith("@3")
            else "skillev-policy-rollout@5",
            reasoning_by_domain=tuple(sorted(self.reasoning_modes.items()))
            if not self.format.endswith("@2")
            else (),
            phase_context=self.phase_context,
            reasoning_tool_catalog=self.reasoning_tool_catalog,
            token_budget_notice=self.token_budget_notice,
            action_wire=self.action_wire,
            skill_exposure=self.skill_exposure,
            public_action_semantics=self.public_action_semantics,
            task_semantic_guidance=self.task_semantic_guidance,
            hotpot_deliberation=self.hotpot_deliberation if self.format.endswith("@6") else False,
            input_window=ModelInputWindow(self.max_input_tokens, self.input_window),
        )

    def application_config(self, run_id: str) -> ApplicationConfig:
        base, _ = build_application_config(
            run_id=run_id,
            steps=self.steps,
            run_plan=self.run_plan,
            batch_size=self.batch_size,
            checkpoint_every=self.checkpoint_every,
            max_turns=self.max_turns,
            max_reasoning_tokens=self.maximum_reasoning_tokens,
            max_action_tokens=self.maximum_action_tokens,
            max_input_tokens=self.max_input_tokens,
        )
        return replace(
            base,
            maximum_h0_tokens=self.max_input_tokens,
            trainer=replace(
                base.trainer,
                rollout=self.sampling_config,
                method=TTBMethodConfig(self.epsilon, self.ttb_beta),
                optimizer=OptimizerConfig(
                    self.adapter_learning_rate, self.z_learning_rate, self.weight_decay
                ),
            ),
            calibration=CalibrationConfig(self.prior_alpha, self.prior_beta, self.k),
            diagnostics=DiagnosticsConfig(window_size=self.window, stagnation_rho=self.rho),
            evolution=replace(
                base.evolution, k=self.k, entropy_window=self.window, cold_start=self.cold_start
            ),
        )

    def require_backbone(self, backbone: QwenMultimodalBackboneConfig) -> None:
        if (
            backbone.lora_rank != self.lora_rank
            or backbone.lora_alpha != self.lora_alpha
            or backbone.lora_dropout != 0
            or backbone.torch_dtype != self.base_dtype
        ):
            raise ValueError("initial model binding differs from the declared LoRA/BF16 condition")

    @property
    def reasoning_modes(self) -> dict[str, bool]:
        return {d: d not in self.thinking_off_domains for d in self.domains}

    def to_value(self) -> dict[str, JsonValue]:
        value = cast(dict[str, JsonValue], normalize_json(asdict(self)))
        if self.domains == tuple(d.value for d in SEVEN_TRAINING_DOMAINS):
            value.pop("domains")  # Preserve historical configuration serialization.
        if self.healthbench_judge == LEGACY_TRAINING_JUDGE:
            value.pop("healthbench_judge")
        if self.format_review_from_step is None:
            value.pop("format_review_from_step")
        if self.cold_start is None:
            value.pop("cold_start")
        if self.initial_skill_profile == "public-advisory@2":
            value.pop("initial_skill_profile")
        if self.learning_protocol is None:
            value.pop("learning_protocol")
        if not self.token_budget_notice:
            value.pop("token_budget_notice")
        if not self.reasoning_tool_catalog:
            value.pop("reasoning_tool_catalog")
        if not self.reasoning_tokens_by_domain:
            value.pop("reasoning_tokens_by_domain")
        if not self.action_tokens_by_domain:
            value.pop("action_tokens_by_domain")
        if self.format.endswith("@2"):
            value.pop("thinking_off_domains")
        if not self.format.endswith(("@4", "@5", "@6")):
            for field in ("phase_context", "action_wire", "skill_exposure"):
                value.pop(field)
        if not self.format.endswith(("@5", "@6")):
            value.pop("public_action_semantics")
        if not self.format.endswith("@6"):
            value.pop("task_semantic_guidance")
        return value

    def expanded_value(self) -> dict[str, JsonValue]:
        """Resolved controls without adding a new null field to historical runs."""
        value = cast(dict[str, JsonValue], normalize_json(asdict(self)))
        if self.domains == tuple(d.value for d in SEVEN_TRAINING_DOMAINS):
            value.pop("domains")
        if self.learning_protocol is None:
            value.pop("learning_protocol")
        if self.format_review_from_step is None:
            value.pop("format_review_from_step")
        return value

    def schedule_summary(self) -> dict[str, JsonValue]:
        return {
            "domains": list(self.domains),
            "questions_per_step": len(self.domains),
            "trajectories_per_question": TRAJECTORIES_PER_QUESTION,
            "batch_size": self.batch_size,
            "question_occurrences": self.steps * len(self.domains),
            "trajectories": self.steps * self.batch_size,
            "condition": self.condition,
            "healthbench_judge": self.healthbench_judge,
            "cold_start": self.cold_start.to_value() if self.cold_start else None,
            **({"learning_protocol": self.learning_protocol} if self.learning_protocol else {}),
            "static_max_turns": self.static_max_turns,
            "interactive_max_turns": self.max_turns,
            "input_window": self.input_window,
            "reasoning_native_thinking_by_domain": normalize_json(self.reasoning_modes),
            "action_tokens_by_domain": normalize_json(
                {
                    d.value: dict(self.action_tokens_by_domain).get(d.value, self.max_action_tokens)
                    for d in self.scheduled_domains
                }
            ),
            "reasoning_tokens_by_domain": normalize_json(
                {
                    d.value: dict(self.reasoning_tokens_by_domain).get(
                        d.value, self.max_reasoning_tokens
                    )
                    for d in self.scheduled_domains
                }
            ),
        }

    @classmethod
    def load(cls, path: Path) -> BayesianFormalConfig:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = {field.name for field in fields(cls)}
        if isinstance(value, dict) and "domains" not in value:
            expected.remove("domains")
        if isinstance(value, dict) and "initial_skill_profile" not in value:
            expected.remove("initial_skill_profile")
        if isinstance(value, dict) and "learning_protocol" not in value:
            expected.remove("learning_protocol")
        if isinstance(value, dict) and "healthbench_judge" not in value:
            expected.remove("healthbench_judge")
        if isinstance(value, dict) and "format_review_from_step" not in value:
            expected.remove("format_review_from_step")
        if isinstance(value, dict) and "cold_start" not in value:
            expected.remove("cold_start")
        if isinstance(value, dict) and "token_budget_notice" not in value:
            expected.remove("token_budget_notice")
        if isinstance(value, dict) and "reasoning_tool_catalog" not in value:
            expected.remove("reasoning_tool_catalog")
        if isinstance(value, dict) and "action_tokens_by_domain" not in value:
            expected.remove("action_tokens_by_domain")
        if isinstance(value, dict) and "reasoning_tokens_by_domain" not in value:
            expected.remove("reasoning_tokens_by_domain")
        if isinstance(value, dict) and value.get("format") == "skillev-bayesian-formal-training@2":
            expected.remove("thinking_off_domains")
        if isinstance(value, dict) and value.get("format") not in {
            "skillev-bayesian-formal-training@4",
            "skillev-bayesian-formal-training@5",
            "skillev-bayesian-formal-training@6",
        }:
            expected -= {"phase_context", "action_wire", "skill_exposure"}
        if isinstance(value, dict) and value.get("format") not in {
            "skillev-bayesian-formal-training@5",
            "skillev-bayesian-formal-training@6",
        }:
            expected.remove("public_action_semantics")
        if isinstance(value, dict) and value.get("format") != "skillev-bayesian-formal-training@6":
            expected.remove("task_semantic_guidance")
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("formal configuration must explicitly declare every control")
        return cls(**value)
