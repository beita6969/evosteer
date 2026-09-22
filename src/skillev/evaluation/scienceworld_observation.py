"""Versioned views of native public descriptions and simulator resource usage.

No object-tree access, inferred contents, selectable-object catalogue or command
rewriting. The original observation remains alongside this whitespace-only view.
"""

from __future__ import annotations

CLOCK_OBSERVATION_PROFILE = "public-state-clock@3"


def public_simulator_clock(moves: int | None, limit: int) -> str:
    """Expose the returned move counter, not private task progress or a stop policy.

    ScienceWorldEnv.step uses ``moves > envStepLimit``, independently of the
    owner's tool-call count. A wait can cross that limit in a single invocation.
    """
    if moves is None:
        raise RuntimeError("the clock observation profile requires native moves")
    return (
        "simulator_clock:\n"
        f"Current native moves: {moves}; native move limit: {limit}. "
        f"The native environment ends when moves > {limit}. "
        "This is separate from the remaining owner action calls: some commands use "
        "no moves, while waiting can advance multiple moves in one call. "
        "The counter reports elapsed simulator time, not task progress."
    )


def containment_layout(text: str) -> str:
    """Indent literal parenthetical contents, without parsing or resolving names."""
    lines = []
    for line in text.splitlines():
        if "(containing " in line:
            indent = line[: len(line) - len(line.lstrip())] + "    "
            lines.append(line.replace("(containing ", "\n" + indent + "(containing "))
    return "\n".join(lines)


def public_containment_view(current_look: str, inventory: str) -> str:
    parts = [
        f"{name}:\n{layout}"
        for name, text in (("current_look", current_look), ("inventory", inventory))
        if (layout := containment_layout(text))
    ]
    if not parts:
        return ""
    return (
        "Public containment layout (the same visible text, with extra line breaks): "
        "'(containing ...)' describes contents of the preceding container. Naming a "
        "container does not name or select an object inside it. This view adds no "
        "objects, observations or actions.\n" + "\n".join(parts)
    )
