# mypy: ignore-errors
"""Python-3.8-compatible, private-only readout of the running native simulator.

No action, goal evaluation, score recalculation, reset or object-tree query is
performed. A diagnostic build additionally exposes the FIRST taskFailure stack
and tick phase; uninstrumented deployments explicitly report its absence.
"""


class ScienceWorldNativeDiagnostics:
    def __init__(self, env):
        self.env = env
        self.instrumented = None
        self.previous = None
        self.first_failure = None

    def snapshot(self, info):
        value = {"native_score": info.get("score"), "simulator_moves": info.get("moves")}
        try:
            interface = self.env.server.agentInterface().get()
            goals = interface.task().goalSequence()
            current = goals.getCurrentSubgoal()
            value.update(
                failed=bool(goals.isFailed()),
                goal_index=int(goals.curSubgoalIdx()),
                goal_type=(current.get().getClass().getName() if current.isDefined() else None),
                parser_pending=bool(interface.inputParser().isInAmbiguousState()),
            )
            if self.instrumented is None:
                self.instrumented = any(
                    method.getName() == "diagnosticFailureBranch"
                    for method in goals.getClass().getMethods()
                )
            if self.instrumented:
                value.update(
                    failure_branch=goals.diagnosticFailureBranch() or None,
                    failure_goal_type=goals.diagnosticFailureGoalType() or None,
                    failure_goal_index=int(goals.diagnosticFailureGoalIndex()),
                    failure_tick_phase=goals.diagnosticFailureTickPhase() or None,
                    failure_simulator_move=int(goals.diagnosticFailureMove()),
                    resolved_action=interface.diagnosticResolvedAction() or None,
                    diagnostic_status="instrumented-read-only@1",
                )
            else:
                value.update(
                    diagnostic_status="native-state-only; failure branch unavailable",
                    failure_branch=None,
                    resolved_action=None,
                )
        except Exception as error:
            # Auxiliary diagnostics must not change a successfully executed action
            # into a new attempt. Their own failure is retained, never hidden.
            value.update(diagnostic_status="unavailable", diagnostic_error=str(error))
        return value

    def reset(self, info):
        self.previous = self.snapshot(info)
        self.first_failure = None

    def transition(self, command, info):
        before, after = self.previous, self.snapshot(info)
        was_failed = before.get("failed") if before is not None else None
        is_failed = after.get("failed")
        first = (
            is_failed and not was_failed
            if isinstance(was_failed, bool) and isinstance(is_failed, bool)
            else None
        )
        if first and self.first_failure is None:
            self.first_failure = {"command": command, **after}
        self.previous = after
        return {
            "profile": "scienceworld-private-transition@1",
            "command": command,
            "before": before,
            "after": after,
            "first_failure_this_step": first,
            "first_failure": self.first_failure,
        }
