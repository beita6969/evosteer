"""Frozen text generation served by an SGLang endpoint of the same base checkpoint.

Executor and author nodes only need ``frozen_text``: the adapter-free base model
generating a completion. Serving that base model from SGLang is numerically the
same frozen executor as the in-process adapter-disabled path, but decodes with
batched CUDA kernels instead of a one-token Python loop. The chat template,
thinking switch, seed and token accounting mirror
``CausalLMOrchestrator.frozen_text``. ``frozen_generation`` additionally reports
whether decoding stopped at ``max_new_tokens``; ``frozen_text`` keeps the plain
(text, input tokens, output tokens) contract that authors and tool nodes use.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any

from skillev.contracts.canonical import stable_hash

from .evosteer import ContextWindowExceededError


class SGLangFrozenText:
    # Each call is an independent HTTP request; executors may run it off-loop.
    thread_safe_generation = True

    def __init__(
        self,
        url: str,
        tokenizer: Any,
        *,
        reference_id: str,
        context_window: int,
        timeout_seconds: float = 600.0,
    ) -> None:
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError("SGLang frozen text requires an http(s) endpoint")
        if type(context_window) is not int or context_window < 1:
            raise ValueError("SGLang context window must be a positive integer")
        self.url = url.rstrip("/")
        self.tokenizer = tokenizer
        self.reference_id = reference_id
        self.context_window = context_window
        self.timeout_seconds = timeout_seconds
        self.eos_token_id = tokenizer.eos_token_id
        self.configuration_id = str(
            stable_hash(
                {
                    "reference": reference_id,
                    "context_window": context_window,
                    "chat_template": getattr(tokenizer, "chat_template", None),
                    "backend": "sglang-frozen-text@1",
                }
            )
        )

    def _prompt_ids(self, text: str) -> tuple[int, ...]:
        messages = [{"role": "user", "content": text}]
        if getattr(self.tokenizer, "chat_template", None):
            raw = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        else:
            raw = self.tokenizer.encode(text, add_special_tokens=False)
        return tuple(int(token) for token in raw)

    def frozen_prompt_tokens(self, text: str) -> int:
        """Token length of a node/author prompt exactly as ``frozen_text`` encodes it."""
        return len(self._prompt_ids(text))

    def _post(self, body: dict[str, Any], attempts: int = 4) -> dict[str, Any]:
        """POST /generate; transient failures and aborted replies are retried.

        The request is idempotent (same input IDs and seed), so a retry cannot
        select a different outcome than the one the seed asks for.
        """
        request = urllib.request.Request(
            f"{self.url}/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    reply = json.loads(response.read().decode("utf-8"))
                finish = reply.get("meta_info", {}).get("finish_reason") or {}
                if isinstance(finish, dict) and finish.get("type") == "abort":
                    raise ConnectionError(f"SGLang aborted the request: {finish}")
                return reply
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == attempts - 1:
                    raise
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == attempts - 1:
                    raise
            time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")

    def frozen_text(
        self,
        text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        input_limit: int | None = None,
    ) -> tuple[str, int, int]:
        output, prompt_tokens, generated, _ = self.frozen_generation(
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            input_limit=input_limit,
        )
        return output, prompt_tokens, generated

    def frozen_generation(
        self,
        text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        input_limit: int | None = None,
    ) -> tuple[str, int, int, bool]:
        """Same request as ``frozen_text``, plus whether the output cap cut it.

        SGLang reports ``finish_reason.type == "length"`` for a cut; a stop token
        sampled in the final slot is a natural finish, not a truncation. Only a
        reply without a finish reason falls back to the token count.
        """
        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 1
            or isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or not math.isfinite(temperature)
            or temperature <= 0
            or type(seed) is not int
            or not 0 <= seed < 2**64
            or (input_limit is not None and (type(input_limit) is not int or input_limit < 1))
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError("positive executor generation limits required")
        prompt = self._prompt_ids(text)
        if len(prompt) + max_new_tokens > self.context_window:
            raise ContextWindowExceededError(
                prompt_tokens=len(prompt),
                reserved_tokens=max_new_tokens,
                context_window=self.context_window,
                operation="SGLang frozen node/author generation",
            )
        if input_limit is not None and len(prompt) > input_limit:
            raise ValueError("node/author prompt exceeds its declared input envelope")
        body = {
            "input_ids": list(prompt),
            "sampling_params": {
                "temperature": float(temperature),
                "max_new_tokens": max_new_tokens,
                # Plain tempered softmax, as in the in-process frozen_text;
                # never the checkpoint's generation-config top-k/top-p.
                "top_k": -1,
                "top_p": 1.0,
                "min_p": 0.0,
                # SGLang stores request seeds as int64.
                "sampling_seed": seed % 2**63,
            },
        }
        reply = self._post(body)
        meta = reply.get("meta_info", {})
        generated = meta.get("completion_tokens")
        if type(generated) is not int or generated < 0:
            raise ValueError("SGLang reply lacks completion token accounting")
        finish = meta.get("finish_reason")
        kind = finish.get("type") if isinstance(finish, dict) else finish
        if isinstance(kind, str) and kind:
            truncated = kind == "length"
        else:
            truncated = generated >= max_new_tokens
        return str(reply.get("text", "")), len(prompt), generated, truncated


__all__ = ["SGLangFrozenText"]
