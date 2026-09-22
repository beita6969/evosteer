"""Versioned public reset goal extraction, also imported by the Python 3.8 worker.

No game metadata, annotations, hidden predicates, or successful action sequences
are inputs. A missing or ambiguous goal is a preparation error, not a task loss.
"""

LEGACY_GOAL_BINDING = "catalog-instruction@1"
RESET_GOAL_BINDING = "reset-public-goal@2"
GOAL_BINDINGS = (LEGACY_GOAL_BINDING, RESET_GOAL_BINDING)
_MARKER = "Your task is to:"


def require_goal_binding(value: str) -> str:
    if value not in GOAL_BINDINGS:
        raise ValueError("unsupported ALFWorld public goal binding")
    return value


def reset_public_goal(observation: str) -> str:
    if not isinstance(observation, str) or observation.count(_MARKER) != 1:
        raise ValueError("ALFWorld reset needs exactly one public task goal")
    goals = [
        line.strip()[len(_MARKER) :].strip()
        for line in observation.splitlines()
        if line.strip().startswith(_MARKER)
    ]
    if len(goals) != 1 or not goals[0] or "\x00" in goals[0]:
        raise ValueError("ALFWorld public task goal is missing or ambiguous")
    return goals[0]


def bound_instruction(observation: str, annotation: str, binding: str) -> str:
    require_goal_binding(binding)
    return reset_public_goal(observation) if binding == RESET_GOAL_BINDING else annotation
