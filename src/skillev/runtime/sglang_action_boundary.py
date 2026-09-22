"""Post-sampling, request-local JSON root termination in SGLang schedulers.

This hook runs after sampling and before the next decode iteration. It never
touches logits, grammar masks or RNG. Untagged requests use upstream behavior.
"""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any, cast

from skillev.rollout.action_root_boundary import (
    ACTION_JSON_ROOT_BOUNDARY_VERSION,
    ActionRootScanner,
)

ACTION_BOUNDARY_PARAMETER = "skillev_action_boundary"


class _ServerDecoder:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def decode(self, ids: tuple[int, ...]) -> str:
        return cast(
            str,
            self.tokenizer.decode(
                list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
        )

    def new_decode_stream(self) -> Any:
        from skillev.policy.tokenizer import incremental_token_decoder

        return incremental_token_decoder(self.tokenizer)


def install_action_root_boundary() -> None:
    """Install in the parent AND every spawned scheduler, not tokenizer threads."""
    upstream = importlib.import_module("sglang.srt.managers.schedule_batch")
    original = upstream.Req._check_str_based_finish
    if getattr(original, "_skillev_action_root", False):
        return

    @wraps(original)
    def check(self: Any, new_accepted_len: int = 1) -> bool:
        if original(self, new_accepted_len):
            return True
        params = self.sampling_params.custom_params
        if (
            not isinstance(params, dict)
            or params.get(ACTION_BOUNDARY_PARAMETER) != ACTION_JSON_ROOT_BOUNDARY_VERSION
        ):
            return False
        scanner = getattr(self, "_skillev_root_scanner", None)
        if scanner is None or len(scanner.ids) > len(self.output_ids):
            scanner = ActionRootScanner(_ServerDecoder(self.tokenizer))
            self._skillev_root_scanner = scanner
        # Accepted speculative tokens are checked individually at real token
        # boundaries; finished_len also excludes any already-computed suffix.
        for token_id in self.output_ids[len(scanner.ids) :]:
            if scanner.push(token_id):
                self.finished_reason = upstream.FINISH_MATCHED_STR(
                    matched=ACTION_JSON_ROOT_BOUNDARY_VERSION
                )
                self.finished_len = len(scanner.ids)
                return True
        return False

    check._skillev_action_root = True  # type: ignore[attr-defined]
    upstream.Req._check_str_based_finish = check
