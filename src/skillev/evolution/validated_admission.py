"""Reference-paired skill admission with a persistent run-level test budget.

The alpha-spending sum is bounded by alpha. An unconditional family-wise error
claim would additionally require valid independent/conditionally valid paired
evidence and a suitable treatment of adaptive candidate selection. The default
rejects repeated task identities within a comparison; callers must supply true
source identities rather than inventing new IDs for repeated problems.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

from skillev.runtime.skills import SkillDocument, model_visible_skill_content

from .paired_trials import ComparisonContext, PairedTrialOutcome, _hash, _text

FORMAT = "evosteer-validated-admission@2"


class SkillStatus(StrEnum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    alpha: float = 0.05
    max_validated_per_family: int = 3
    max_candidates_per_family: int = 1
    success_threshold: float = 0.5
    allow_repeated_tasks: bool = False
    max_library_size: int = 60
    max_library_per_family: int = 12

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if (
            isinstance(self.success_threshold, bool)
            or not math.isfinite(self.success_threshold)
            or not 0 <= self.success_threshold <= 1
        ):
            raise ValueError("success threshold must be in [0, 1]")
        for field in (
            "max_validated_per_family",
            "max_candidates_per_family",
            "max_library_size",
            "max_library_per_family",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"{field} must be a positive integer")
        if type(self.allow_repeated_tasks) is not bool:
            raise TypeError("allow_repeated_tasks must be boolean")


@dataclass(frozen=True, slots=True)
class SkillEntry:
    skill_id: str
    family: str
    content_hash: str
    document: SkillDocument | None = None
    body: str = ""

    def __post_init__(self) -> None:
        for field in ("skill_id", "family", "content_hash"):
            _text(getattr(self, field), field)
        if not isinstance(self.body, str):
            raise TypeError("skill body must be text")
        if self.document is not None:
            if not isinstance(self.document, SkillDocument):
                raise TypeError("document must be SkillDocument")
            if (
                self.document.manifest.skill_id != self.skill_id
                or self.document.manifest.content_hash != self.content_hash
            ):
                raise ValueError("entry identity differs from its immutable SkillDocument")
            rendered = model_visible_skill_content(self.document)
            if self.body and self.body != rendered:
                raise ValueError("skill body differs from the supplied SkillDocument")
            object.__setattr__(self, "body", rendered)

    @classmethod
    def from_document(cls, document: SkillDocument, family: str) -> SkillEntry:
        return cls(document.manifest.skill_id, family, document.manifest.content_hash, document)

    def to_value(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "family": self.family,
            "content_hash": self.content_hash,
            "document": None if self.document is None else self.document.to_value(),
            "body": self.body,
        }

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> SkillEntry:
        doc = value["document"]
        return cls(
            value["skill_id"],
            value["family"],
            value["content_hash"],
            None if doc is None else SkillDocument.from_value(doc),
            value["body"],
        )


@dataclass(frozen=True, slots=True)
class MenuSkill:
    entry: SkillEntry
    status: SkillStatus
    admission_basis: str = "candidate-proposal"

    def __post_init__(self) -> None:
        if not isinstance(self.entry, SkillEntry) or not isinstance(self.status, SkillStatus):
            raise TypeError("menu skill requires typed entry and lifecycle status")
        bases = ("candidate-proposal", "configured-seed", "paired-sign-test")
        if self.admission_basis not in bases:
            raise ValueError("unknown skill admission basis")

    @property
    def skill_id(self) -> str:
        return self.entry.skill_id

    @property
    def family(self) -> str:
        return self.entry.family

    @property
    def body(self) -> str:
        return self.entry.body

    @property
    def content_hash(self) -> str:
        return self.entry.content_hash

    def to_value(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_value(),
            "status": self.status.value,
            "admission_basis": self.admission_basis,
        }


@dataclass(frozen=True, slots=True)
class SkillMenuSnapshot:
    batch_id: str
    menu_id: str
    entries: tuple[MenuSkill, ...]

    def __post_init__(self) -> None:
        _text(self.batch_id, "batch_id")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(item, MenuSkill) for item in self.entries
        ):
            raise TypeError("menu entries must be an immutable tuple of MenuSkill")
        ids = tuple(item.skill_id for item in self.entries)
        if ids != tuple(sorted(set(ids))) or any(
            item.status is SkillStatus.RETIRED for item in self.entries
        ):
            raise ValueError("menu requires unique sorted available skills")
        if self.menu_id != _hash([item.to_value() for item in self.entries]):
            raise ValueError("menu identity differs from its immutable contents")

    def for_family(self, family: str) -> tuple[MenuSkill, ...]:
        """Candidates remain selectable before their admission test succeeds."""
        return tuple(item for item in self.entries if item.family == family)

    def skill_ids(self, family: str | None = None) -> tuple[str, ...]:
        return tuple(
            item.skill_id for item in self.entries if family is None or item.family == family
        )

    @property
    def visible_skill_ids(self) -> tuple[str, ...]:
        return self.skill_ids()

    @property
    def bodies(self) -> dict[str, str]:
        return {item.skill_id: item.body for item in self.entries}

    def background_id(self, excluded_skill_ids: tuple[str, ...] = ()) -> str:
        return _hash(
            [item.to_value() for item in self.entries if item.skill_id not in excluded_skill_ids]
        )

    def to_value(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "menu_id": self.menu_id,
            "entries": [item.to_value() for item in self.entries],
        }


@dataclass(frozen=True, slots=True)
class ComparisonState:
    comparison_id: str
    comparison_index: int
    skill_id: str
    content_hash: str
    context: ComparisonContext
    reevaluation: bool = False
    wins: int = 0
    losses: int = 0
    ties: int = 0
    observation_index: int = 0
    last_tested_discordants: int = 0
    alpha_spent: float = 0.0
    closed: bool = False

    @property
    def effect_size(self) -> float:
        total = self.wins + self.losses + self.ties
        return (self.wins - self.losses) / total if total else 0.0

    def to_value(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    comparison_id: str
    skill_id: str
    action: str
    status_before: SkillStatus
    status_after: SkillStatus
    wins: int
    losses: int
    ties: int
    observation_index: int
    p_positive: float | None
    p_negative: float | None
    alpha_each_direction: float
    effect_size: float


def exact_binomial_sign_p(discordant: int, favorable: int) -> float:
    """P[Bin(discordant, 1/2) >= favorable], without asymptotic approximations.

    Small tests use an exact integer numerator. Larger tests sum binomial PMFs
    by recurrence in the numerically favorable tail; underflow to zero is
    possible only below the representable positive floating-point range.
    """
    if type(discordant) is not int or discordant < 0 or type(favorable) is not int:
        raise ValueError("binomial counts must be integers, with nonnegative n")
    if not 0 <= favorable <= discordant:
        raise ValueError("favorable count is outside [0, n]")
    if favorable == 0:
        return 1.0
    if discordant <= 2048:
        numerator = sum(math.comb(discordant, k) for k in range(favorable, discordant + 1))
        return numerator / (1 << discordant)
    if favorable <= discordant // 2:
        return 1.0 - exact_binomial_sign_p(discordant, discordant - favorable + 1)
    log_first = (
        math.lgamma(discordant + 1)
        - math.lgamma(favorable + 1)
        - math.lgamma(discordant - favorable + 1)
        - discordant * math.log(2.0)
    )
    relative = 1.0
    terms = [relative]
    for k in range(favorable, discordant):
        relative *= (discordant - k) / (k + 1)
        terms.append(relative)
        if relative == 0.0:
            break
    return min(1.0, math.exp(log_first + math.log(math.fsum(terms))))


class AdmissionLedger:
    """Single-owner deterministic reducer; serialize between committed batches."""

    def __init__(self, config: AdmissionConfig | None = None) -> None:
        self._config = config or AdmissionConfig()
        if not isinstance(self._config, AdmissionConfig):
            raise TypeError("ledger requires AdmissionConfig")
        self._skills: dict[str, MenuSkill] = {}
        self._comparisons: dict[str, ComparisonState] = {}
        self._author_windows: set[str] = set()
        self._pairs: dict[str, str] = {}
        self._tasks: dict[str, set[str]] = {}
        self._menus: dict[str, SkillMenuSnapshot] = {}
        self._events: list[dict[str, Any]] = []

    @property
    def config(self) -> AdmissionConfig:
        """The run's statistical contract cannot be replaced after observations."""
        return self._config

    @property
    def alpha_spent(self) -> float:
        return math.fsum(item.alpha_spent for item in self._comparisons.values())

    @property
    def comparisons(self) -> tuple[ComparisonState, ...]:
        return tuple(self._comparisons.values())

    @property
    def skills(self) -> tuple[MenuSkill, ...]:
        return tuple(self._skills[key] for key in sorted(self._skills))

    def status(self, skill_id: str) -> SkillStatus:
        return self._skills[skill_id].status

    def comparison(self, comparison_id: str) -> ComparisonState:
        return self._comparisons[comparison_id]

    def has_task(self, comparison_id: str, task_id: str) -> bool:
        """Whether a complete eligible pair for this source was already consumed."""
        return task_id in self._tasks[comparison_id]

    def paired_observation_count(self, skill_id: str) -> int:
        """Whole eligible pairs across registered contexts, including ties."""
        if skill_id not in self._skills:
            raise KeyError(skill_id)
        return sum(
            item.wins + item.losses + item.ties
            for item in self._comparisons.values()
            if item.skill_id == skill_id
        )

    def trial_priority(
        self, family: str, *, reevaluation_due: bool = False
    ) -> tuple[MenuSkill, ...]:
        """Order scarce trials by fewest paired observations, with retirement access.

        A caller-scheduled reevaluation round gives validated skills priority
        even while a stalled candidate occupies its slot. No unreported effect
        threshold, expiry, or capacity-based retirement is introduced: evidence
        must still pass the negative sign-test boundary to retire a skill.
        """
        _text(family, "family")
        if type(reevaluation_due) is not bool:
            raise TypeError("reevaluation_due must be boolean")
        validated = tuple(
            item
            for item in self.skills
            if item.family == family and item.status is SkillStatus.VALIDATED
        )
        candidates = tuple(
            item
            for item in self.skills
            if item.family == family and item.status is SkillStatus.CANDIDATE
        )
        pool = validated if reevaluation_due and validated else candidates
        return tuple(
            sorted(
                pool, key=lambda item: (self.paired_observation_count(item.skill_id), item.skill_id)
            )
        )

    def can_propose(self, family: str) -> bool:
        """Check capacity before paying for an author call; retirement retains history.

        This read-only check does not reserve a slot or prevalidate a future
        proposal's content identity or its one-proposal author window.
        """
        _text(family, "family")
        family_skills = tuple(item for item in self._skills.values() if item.family == family)
        return (
            len(self._skills) < self.config.max_library_size
            and len(family_skills) < self.config.max_library_per_family
            and sum(item.status is SkillStatus.CANDIDATE for item in family_skills)
            < self.config.max_candidates_per_family
        )

    def _install(self, entry: SkillEntry, status: SkillStatus) -> None:
        if not isinstance(entry, SkillEntry):
            raise TypeError("skill must be SkillEntry")
        if entry.skill_id in self._skills:
            raise ValueError("skill ID is already registered")
        if any(
            item.entry.content_hash == entry.content_hash and item.family == entry.family
            for item in self._skills.values()
        ):
            raise ValueError("duplicate skill content in the same family")
        # Retired entries are immutable library history and still consume these
        # total slots. Neither retirement nor a new author window resets them.
        if len(self._skills) >= self.config.max_library_size:
            raise ValueError("total library capacity is full")
        if (
            sum(item.family == entry.family for item in self._skills.values())
            >= self.config.max_library_per_family
        ):
            raise ValueError("total library capacity for family is full")
        cap = (
            self.config.max_candidates_per_family
            if status is SkillStatus.CANDIDATE
            else self.config.max_validated_per_family
        )
        count = sum(
            item.family == entry.family and item.status is status for item in self._skills.values()
        )
        if count >= cap:
            raise ValueError(f"{status.value} capacity for family is full")
        basis = "configured-seed" if status is SkillStatus.VALIDATED else "candidate-proposal"
        self._skills[entry.skill_id] = MenuSkill(entry, status, basis)

    def add_seed(self, entry: SkillEntry) -> None:
        """Install an explicitly trusted seed, without fabricating test evidence."""
        self._install(entry, SkillStatus.VALIDATED)
        self._events.append({"kind": "seed", "entry": entry.to_value()})

    def propose(self, entry: SkillEntry, author_window_id: str) -> None:
        _text(author_window_id, "author_window_id")
        if author_window_id in self._author_windows:
            raise ValueError("an author window may contribute only one proposal")
        self._install(entry, SkillStatus.CANDIDATE)
        self._author_windows.add(author_window_id)
        self._events.append(
            {
                "kind": "proposal",
                "entry": entry.to_value(),
                "author_window_id": author_window_id,
            }
        )

    def register_comparison(
        self,
        skill_id: str,
        context: ComparisonContext,
        reevaluation: bool = False,
    ) -> str:
        if not isinstance(context, ComparisonContext) or type(reevaluation) is not bool:
            raise TypeError("comparison requires typed context and reevaluation flag")
        skill = self._skills[skill_id]
        expected = SkillStatus.VALIDATED if reevaluation else SkillStatus.CANDIDATE
        if skill.status is not expected:
            raise ValueError(f"comparison requires a {expected.value} skill")
        if context.task_family != skill.family:
            raise ValueError("comparison family differs from candidate family")
        if any(
            item.skill_id == skill_id
            and item.context == context
            and item.reevaluation == reevaluation
            and not item.closed
            for item in self._comparisons.values()
        ):
            raise ValueError("skill already has an open comparison for this context")
        index = len(self._comparisons) + 1
        comparison_id = _hash(
            {
                "index": index,
                "skill_id": skill_id,
                "content_hash": skill.entry.content_hash,
                "context": context.to_value(),
                "reevaluation": reevaluation,
            }
        )
        self._comparisons[comparison_id] = ComparisonState(
            comparison_id,
            index,
            skill_id,
            skill.entry.content_hash,
            context,
            reevaluation,
        )
        self._tasks[comparison_id] = set()
        self._events.append(
            {
                "kind": "register",
                "skill_id": skill_id,
                "context": context.to_value(),
                "reevaluation": reevaluation,
                "comparison_id": comparison_id,
            }
        )
        return comparison_id

    def close_comparison(self, comparison_id: str, reason: str) -> None:
        """End an unresolved comparison before changing its frozen conditions.

        This does not admit or retire a skill and never refunds spent alpha. A
        replacement comparison receives a fresh global index, including when
        changing only the value-head or background skill snapshot.
        """
        self._close_comparison(comparison_id, reason, kind="close")

    def supersede_comparison(self, comparison_id: str, reason: str) -> None:
        """Close on a context change without transferring evidence to its successor.

        Frequent value-head or menu changes consequently reduce statistical power;
        accumulated evidence and the spent run budget remain in the audit history.
        """
        self._close_comparison(comparison_id, reason, kind="supersede")

    def _close_comparison(self, comparison_id: str, reason: str, *, kind: str) -> None:
        _text(reason, "reason")
        state = self._comparisons[comparison_id]
        if state.closed:
            raise ValueError("comparison is already closed")
        self._comparisons[comparison_id] = replace(state, closed=True)
        self._events.append({"kind": kind, "comparison_id": comparison_id, "reason": reason})

    def freeze_menu(self, batch_id: str, family: str | None = None) -> SkillMenuSnapshot:
        _text(batch_id, "batch_id")
        if batch_id not in self._menus:
            entries = tuple(item for item in self.skills if item.status is not SkillStatus.RETIRED)
            menu = SkillMenuSnapshot(
                batch_id,
                _hash([item.to_value() for item in entries]),
                entries,
            )
            self._menus[batch_id] = menu
            self._events.append({"kind": "freeze", "batch_id": batch_id, "menu_id": menu.menu_id})
        frozen = self._menus[batch_id]
        if family is None:
            return frozen
        _text(family, "family")
        entries = frozen.for_family(family)
        return SkillMenuSnapshot(batch_id, _hash([item.to_value() for item in entries]), entries)

    def record_pair(
        self, pair: PairedTrialOutcome, *, validate: bool = True
    ) -> AdmissionDecision | None:
        """Accumulate a complete pair, optionally treating it as a validation round.

        Batch callers should use validate=False for each pair and then invoke
        validate_comparison once per comparison at their validation boundary.
        """
        if not isinstance(pair, PairedTrialOutcome):
            raise TypeError("record_pair requires a complete paired outcome object")
        if type(validate) is not bool:
            raise TypeError("validate must be boolean")
        spec = pair.positive.spec
        state = self._comparisons[spec.comparison_id]
        if (
            spec.skill_id != state.skill_id
            or spec.initial_condition.comparison_context != state.context
        ):
            raise ValueError("pair differs from registered skill or frozen comparison context")
        if not pair.eligible:
            return None
        digest = _hash(pair.to_value())
        if pair.pair_id in self._pairs:
            if self._pairs[pair.pair_id] != digest:
                raise ValueError("pair ID was reused with different outcomes or conditions")
            return None
        task_id = spec.initial_condition.task_id
        if not self.config.allow_repeated_tasks and task_id in self._tasks[state.comparison_id]:
            raise ValueError("repeated task is not independent paired evidence")
        assert pair.positive.reward is not None
        assert pair.negative.reward is not None
        positive = pair.positive.reward >= self.config.success_threshold
        negative = pair.negative.reward >= self.config.success_threshold
        win, loss = positive > negative, positive < negative
        next_state = replace(
            state,
            wins=state.wins + win,
            losses=state.losses + loss,
            ties=state.ties + (positive == negative),
        )
        decision = self._evaluate_comparison(next_state, validate=validate)
        self._pairs[pair.pair_id] = digest
        self._tasks[state.comparison_id].add(task_id)
        self._events.append({"kind": "pair", "outcome": pair.to_value(), "validate": validate})
        return decision

    def validate_comparison(self, comparison_id: str) -> AdmissionDecision | None:
        """Test new discordants, or close a retired skill's remaining comparison.

        Retirement is absorbing across concurrent comparison contexts. Closing a
        residual comparison consumes no look or alpha, even when it contains
        positive or only tied evidence collected before the retirement decision.
        """
        state = self._comparisons[comparison_id]
        if state.closed:
            return None
        if (
            self._skills[state.skill_id].status is not SkillStatus.RETIRED
            and state.wins + state.losses <= state.last_tested_discordants
        ):
            return None
        decision = self._evaluate_comparison(state, validate=True)
        self._events.append({"kind": "validate", "comparison_id": comparison_id})
        return decision

    def _evaluate_comparison(self, state: ComparisonState, *, validate: bool) -> AdmissionDecision:
        next_state = state
        skill = self._skills[state.skill_id]
        status_after = skill.status
        p_positive = p_negative = None
        alpha_each = 0.0
        action = "comparison_closed" if state.closed else "defer"
        if validate and skill.status is SkillStatus.RETIRED:
            action = "comparison_closed"
            next_state = replace(next_state, closed=True)
        elif (
            validate
            and state.wins + state.losses > state.last_tested_discordants
            and not state.closed
        ):
            look = state.observation_index + 1
            index = state.comparison_index
            level = self.config.alpha / (index * (index + 1) * look * (look + 1))
            alpha_each = level / 2
            n = next_state.wins + next_state.losses
            p_positive = exact_binomial_sign_p(n, next_state.wins)
            p_negative = exact_binomial_sign_p(n, next_state.losses)
            next_state = replace(
                next_state,
                observation_index=look,
                last_tested_discordants=n,
                alpha_spent=math.fsum((state.alpha_spent, level)),
            )
            if p_negative <= alpha_each:
                action, status_after = "retire", SkillStatus.RETIRED
                next_state = replace(next_state, closed=True)
            elif p_positive <= alpha_each:
                validated = sum(
                    item.family == skill.family and item.status is SkillStatus.VALIDATED
                    for item in self._skills.values()
                )
                if skill.status is SkillStatus.VALIDATED:
                    action = "retain"
                    next_state = replace(next_state, closed=True)
                elif validated < self.config.max_validated_per_family:
                    action, status_after = "promote", SkillStatus.VALIDATED
                    next_state = replace(next_state, closed=True)
                else:
                    action = "defer_capacity"
        decision = AdmissionDecision(
            state.comparison_id,
            state.skill_id,
            action,
            skill.status,
            status_after,
            next_state.wins,
            next_state.losses,
            next_state.ties,
            next_state.observation_index,
            p_positive,
            p_negative,
            alpha_each,
            next_state.effect_size,
        )
        basis = "paired-sign-test" if action == "promote" else skill.admission_basis
        self._skills[state.skill_id] = MenuSkill(skill.entry, status_after, basis)
        self._comparisons[state.comparison_id] = next_state
        return decision

    def _state_value(self) -> dict[str, Any]:
        return {
            "skills": [item.to_value() for item in self.skills],
            "comparisons": [item.to_value() for item in self.comparisons],
            "alpha_spent": self.alpha_spent,
            "menus": [item.to_value() for item in self._menus.values()],
        }

    def to_value(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "config": asdict(self.config),
            "events": copy.deepcopy(self._events),
            "state": self._state_value(),
        }

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> AdmissionLedger:
        if set(value) != {"format", "config", "events", "state"} or value["format"] != FORMAT:
            raise ValueError("unsupported admission ledger representation")
        ledger = cls(AdmissionConfig(**value["config"]))
        for event in value["events"]:
            kind = event["kind"]
            if kind == "seed":
                ledger.add_seed(SkillEntry.from_value(event["entry"]))
            elif kind == "proposal":
                ledger.propose(SkillEntry.from_value(event["entry"]), event["author_window_id"])
            elif kind == "register":
                identity = ledger.register_comparison(
                    event["skill_id"],
                    ComparisonContext.from_value(event["context"]),
                    event["reevaluation"],
                )
                if identity != event["comparison_id"]:
                    raise ValueError("persisted comparison identity differs from registration")
            elif kind == "freeze":
                if ledger.freeze_menu(event["batch_id"]).menu_id != event["menu_id"]:
                    raise ValueError("persisted menu identity differs from frozen content")
            elif kind == "close":
                ledger.close_comparison(event["comparison_id"], event["reason"])
            elif kind == "supersede":
                ledger.supersede_comparison(event["comparison_id"], event["reason"])
            elif kind == "pair":
                if (
                    ledger.record_pair(
                        PairedTrialOutcome.from_value(event["outcome"]), validate=event["validate"]
                    )
                    is None
                ):
                    raise ValueError("persisted ledger contains incomplete or repeated evidence")
            elif kind == "validate":
                if ledger.validate_comparison(event["comparison_id"]) is None:
                    raise ValueError(
                        "persisted validation neither tested evidence nor closed retirement"
                    )
            else:
                raise ValueError("unknown admission ledger event")
        if ledger.to_value() != value:
            raise ValueError("persisted admission state differs from its source events")
        return ledger


__all__ = [
    "AdmissionConfig",
    "AdmissionDecision",
    "AdmissionLedger",
    "ComparisonContext",
    "ComparisonState",
    "MenuSkill",
    "SkillEntry",
    "SkillMenuSnapshot",
    "SkillStatus",
    "exact_binomial_sign_p",
]
