"""Private native-transition diagnostics, never an actor action policy."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EnvironmentTransitionReceipt:
    command: str
    observation: str
    action_valid: bool | None
    previous_native_score: float | None
    native_score: float | None
    terminal: bool
    simulator_moves: int | None
    observation_profile: str
    delivered: bool = True
    acknowledged: bool = True
    previous_simulator_moves: int | None = None
    native_diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def category(self) -> str:
        if not self.delivered:
            return "framework-command-not-delivered"
        if not self.acknowledged:
            return "execution-acknowledgement-unresolved"
        if self.native_score is not None and self.native_score < 0 and self.terminal:
            return "native-negative-terminal"
        if self.action_valid is False:
            return "native-command-rejected"
        if self.native_score is None or self.previous_native_score is None:
            return "acknowledged-progress-unknown"
        if self.native_score > self.previous_native_score:
            return "acknowledged-native-score-increased"
        # Score stability is not proof that no useful task action occurred.
        return "acknowledged-no-native-score-increase"
