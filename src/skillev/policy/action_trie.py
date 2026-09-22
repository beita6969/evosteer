"""Finite legal graph actions as an exact token-level grammar.

The finite menu bounds graph node slots/roles/skills, not language-model logits.
Every token is normalized only over continuations admitted by this trie.
"""

from __future__ import annotations


class ActionTrie:
    def __init__(self, paths: tuple[tuple[int, ...], ...]) -> None:
        if not isinstance(paths, tuple) or not paths:
            raise ValueError("legal paths must be a nonempty immutable tuple")
        if any(
            not isinstance(path, tuple)
            or not path
            or any(type(t) is not int or t < 0 for t in path)
            for path in paths
        ):
            raise ValueError("invalid legal token path")
        if len(set(paths)) != len(paths):
            raise ValueError("legal paths must be nonempty and unique")
        self.paths = paths
        self._children: dict[tuple[int, ...], set[int]] = {}
        self._leaves = set(paths)
        for path in paths:
            for i, token in enumerate(path):
                prefix = path[:i]
                if prefix in self._leaves:
                    raise ValueError("action paths must be prefix-free; append a terminator")
                self._children.setdefault(prefix, set()).add(token)

    def allowed(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        self._validate_prefix(prefix)
        if prefix in self._leaves:
            return ()
        try:
            return tuple(sorted(self._children[prefix]))
        except KeyError as error:
            raise ValueError("token prefix is outside the legal action grammar") from error

    def masks(self, path: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        self._validate_prefix(path)
        if path not in self._leaves:
            raise ValueError("action path is not an admitted complete action")
        return tuple(self.allowed(path[:i]) for i in range(len(path)))

    def complete(self, prefix: tuple[int, ...]) -> bool:
        self._validate_prefix(prefix)
        return prefix in self._leaves

    @staticmethod
    def _validate_prefix(prefix: tuple[int, ...]) -> None:
        if not isinstance(prefix, tuple) or any(type(t) is not int or t < 0 for t in prefix):
            raise ValueError("token prefix must be an immutable tuple of nonnegative integers")
