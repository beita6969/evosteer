from __future__ import annotations

import http.client
import json
import math
import re
import re._parser as _regex_parser
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from skillev.contracts.canonical import stable_hash

from .evosteer import ContextWindowExceededError

OUTAGE_SECONDS = 900.0
_BACKOFF_SECONDS = (2.0, 30.0)
_TRANSIENT = (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException)
STOP_REGEX_MAX_CHARS = 256


def _regex_bound(items: Any) -> float:
    total = 0.0
    for op, value in items:
        if op in (_regex_parser.LITERAL, _regex_parser.IN, _regex_parser.ANY):
            total += 1
        elif op is _regex_parser.SUBPATTERN:
            total += _regex_bound(value[3])
        elif op is _regex_parser.BRANCH:
            total += max(_regex_bound(branch) for branch in value[1])
        elif op in (_regex_parser.MAX_REPEAT, _regex_parser.MIN_REPEAT):
            _, high, inner = value
            if high == _regex_parser.MAXREPEAT:
                return math.inf
            total += high * _regex_bound(inner) if high else 0
        elif op is not _regex_parser.AT:
            return math.inf
    return total


def check_stop_regex(patterns: Any) -> tuple[str, ...]:
    if not isinstance(patterns, tuple) or any(type(p) is not str or not p for p in patterns):
        raise ValueError("stop regexes must be a tuple of nonempty patterns")
    for pattern in patterns:
        try:
            parsed = _regex_parser.parse(pattern)
        except re.error as error:
            raise ValueError(f"invalid stop regex {pattern!r}: {error}") from None
        if _regex_bound(parsed) > STOP_REGEX_MAX_CHARS:
            raise ValueError(
                f"stop regex {pattern!r} can match more than {STOP_REGEX_MAX_CHARS} characters "
                "(or uses a backreference): SGLang would decode the whole output at every step"
            )
    return patterns


class SGLangFrozenText:
    thread_safe_generation = True
    supports_stop_strings = True
    supports_stop_regex = True

    def __init__(
        self,
        url: str,
        tokenizer: Any,
        *,
        reference_id: str,
        context_window: int,
        timeout_seconds: float = 600.0,
        thinking: bool = False,
        outage_seconds: float = OUTAGE_SECONDS,
    ) -> None:
        parts = url.split(",") if isinstance(url, str) else []
        urls = tuple(part.strip().rstrip("/") for part in parts)
        if (
            not urls
            or any(not item.startswith(("http://", "https://")) for item in urls)
            or len(set(urls)) != len(urls)
        ):
            raise ValueError(
                "SGLang frozen text requires an http(s) endpoint, or a comma-separated list "
                "of distinct replica endpoints"
            )
        if type(context_window) is not int or context_window < 1:
            raise ValueError("SGLang context window must be a positive integer")
        if (
            isinstance(outage_seconds, bool)
            or not isinstance(outage_seconds, int | float)
            or not 0 <= outage_seconds < math.inf
        ):
            raise ValueError("outage_seconds must be a nonnegative number")
        self.urls = urls
        self.tokenizer = tokenizer
        self.reference_id = reference_id
        self.context_window = context_window
        self.timeout_seconds = timeout_seconds
        self.outage_seconds = float(outage_seconds)
        self._lock = threading.Lock()
        self._in_flight = [0] * len(urls)
        self._failures = [0] * len(urls)
        self._down_until = [0.0] * len(urls)
        self._turn = 0
        if type(thinking) is not bool:
            raise TypeError("thinking must be boolean")
        self.thinking = thinking
        self.eos_token_id = tokenizer.eos_token_id
        self.configuration_id = str(
            stable_hash(
                {
                    "reference": reference_id,
                    "context_window": context_window,
                    "chat_template": getattr(tokenizer, "chat_template", None),
                    "backend": "sglang-frozen-text@1",
                    **({"thinking": True} if thinking else {}),
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
                enable_thinking=self.thinking,
                return_dict=False,
            )
        else:
            raw = self.tokenizer.encode(text, add_special_tokens=False)
        return tuple(int(token) for token in raw)

    def frozen_prompt_tokens(self, text: str) -> int:
        return len(self._prompt_ids(text))

    def _acquire(self) -> tuple[int, float]:
        with self._lock:
            now = time.monotonic()
            count = len(self.urls)
            order = [(self._turn + step) % count for step in range(count)]
            self._turn = (self._turn + 1) % count
            ready = [index for index in order if self._down_until[index] <= now]
            if ready:
                index = min(ready, key=lambda item: self._in_flight[item])
            else:
                index = min(order, key=lambda item: self._down_until[item])
            self._in_flight[index] += 1
            return index, max(0.0, self._down_until[index] - now)

    def _release(self, index: int, *, failed: bool) -> None:
        with self._lock:
            self._in_flight[index] -= 1
            if failed:
                self._failures[index] += 1
                low, high = _BACKOFF_SECONDS
                backoff = min(high, low * 2 ** (self._failures[index] - 1))
                self._down_until[index] = time.monotonic() + backoff
            else:
                self._failures[index] = 0
                self._down_until[index] = 0.0

    def _send(self, url: str, data: bytes) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{url}/generate", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            reply = json.loads(response.read().decode("utf-8"))
        finish = reply.get("meta_info", {}).get("finish_reason") or {}
        if isinstance(finish, dict) and finish.get("type") == "abort":
            raise ConnectionError(f"SGLang aborted the request: {finish}")
        return reply

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        first_failure: float | None = None
        while True:
            index, wait = self._acquire()
            try:
                if wait:
                    time.sleep(wait)
                reply = self._send(self.urls[index], data)
            except urllib.error.HTTPError as error:
                self._release(index, failed=error.code >= 500)
                if error.code < 500:
                    raise
                failure: BaseException = error
            except _TRANSIENT as error:
                self._release(index, failed=True)
                failure = error
            except BaseException:
                self._release(index, failed=False)
                raise
            else:
                self._release(index, failed=False)
                return reply
            now = time.monotonic()
            first_failure = now if first_failure is None else first_failure
            if now - first_failure >= self.outage_seconds:
                failure.add_note(
                    f"SGLang replicas {', '.join(self.urls)} failed for "
                    f"{now - first_failure:.0f} s (limit {self.outage_seconds:.0f} s)"
                )
                raise failure

    def frozen_text(
        self,
        text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        input_limit: int | None = None,
        stop: tuple[str, ...] = (),
        stop_regex: tuple[str, ...] = (),
    ) -> tuple[str, int, int]:
        output, prompt_tokens, generated, _ = self.frozen_generation(
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            input_limit=input_limit,
            stop=stop,
            stop_regex=stop_regex,
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
        stop: tuple[str, ...] = (),
        stop_regex: tuple[str, ...] = (),
    ) -> tuple[str, int, int, bool]:
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
        if not isinstance(stop, tuple) or any(type(item) is not str or not item for item in stop):
            raise ValueError("stop strings must be a tuple of nonempty strings")
        check_stop_regex(stop_regex)
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
                "top_k": -1,
                "top_p": 1.0,
                "min_p": 0.0,
                "sampling_seed": seed % 2**63,
            },
        }
        if stop:
            body["sampling_params"]["stop"] = list(stop)
            body["sampling_params"]["no_stop_trim"] = True
        if stop_regex:
            body["sampling_params"]["stop_regex"] = list(stop_regex)
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
        output = str(reply.get("text", ""))
        if self.thinking:
            head, closed, answer = output.rpartition("</think>")
            if closed:
                output = answer.lstrip("\n")
        matched = finish.get("matched") if isinstance(finish, dict) else None
        if stop and kind == "stop" and matched in stop and not output.endswith(matched):
            output += matched
        return output, len(prompt), generated, truncated


__all__ = ["OUTAGE_SECONDS", "STOP_REGEX_MAX_CHARS", "SGLangFrozenText", "check_stop_regex"]
