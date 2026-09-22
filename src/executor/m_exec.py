

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

from openai import APIError, OpenAI

from src.executor.openai_request_policy import (
    ContextBudgetExceeded,
    OpenAIRequestPolicy,
    PermanentRequestRejected,
    TransientRequestExhausted,
)

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a capable AI assistant. "
    "Execute the given instruction accurately and completely. "
    "Be concise but thorough. Do not add unnecessary caveats or disclaimers."
)


class MExec:


    def __init__(
        self,
        api_base: str,
        model_name: str = "gpt-oss-120b",
        api_key: str = "",
        default_temperature: float = 0.1,
        default_max_tokens: int = 8192,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_timeout: float = 60.0,
        request_policy: OpenAIRequestPolicy | None = None,
        fail_closed: bool = False,
    ):
        self._api_base = api_base
        self._api_key = api_key or os.environ.get("MEXEC_API_KEY") or os.environ.get("SGLANG_API_KEY", "EMPTY")
        self._thread_local = threading.local()
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        self.request_policy = request_policy
        self.fail_closed = fail_closed


        self._total_calls = 0
        self._total_errors = 0


        MExec._instance = self

    @property
    def client(self) -> OpenAI:

        c = getattr(self._thread_local, "client", None)
        if c is None:
            c = OpenAI(
                base_url=self._api_base,
                api_key=self._api_key,
                timeout=self.request_timeout,
                max_retries=0,
            )
            self._thread_local.client = c
        return c

    def execute(
        self,
        instruction: str,
        context: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "",
    ) -> str:

        messages = self._build_messages(instruction, context)
        self._total_calls += 1
        requested_max_tokens = (
            max_tokens if max_tokens is not None else self.default_max_tokens
        )

        if self.request_policy is not None:
            try:
                resp = self.request_policy.create_chat_completion(
                    self.client,
                    request_kind=f"m_exec:{task_type or 'unknown'}",
                    model=self.model_name,
                    messages=messages,
                    tools=None,
                    max_tokens=requested_max_tokens,
                    temperature=(
                        temperature
                        if temperature is not None
                        else self.default_temperature
                    ),
                    enable_thinking=False,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                )
                return self._response_text(resp, task_type)
            except ContextBudgetExceeded:
                self._total_errors += 1
                raise
            except (PermanentRequestRejected, TransientRequestExhausted):
                self._total_errors += 1
                if self.fail_closed:
                    raise
                return "[EXECUTION_ERROR] request_failed"

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.default_temperature,
                    max_tokens=requested_max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return self._response_text(resp, task_type, len(instruction))

            except APIError as e:
                self._total_errors += 1
                logger.warning(
                    "[MExec] API error: attempt=%s error_class=%s status=%s",
                    attempt + 1,
                    type(e).__name__,
                    getattr(e, "status_code", None),
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                self._total_errors += 1
                logger.error("[MExec] Unexpected error: error_class=%s", type(e).__name__)
                if self.fail_closed:
                    raise
                return f"[EXECUTION_ERROR] {type(e).__name__}"

        return "[EXECUTION_ERROR] Max retries exceeded"

    def execute_batch(
        self,
        instructions: list[str],
        contexts: Optional[list[str]] = None,
        **kwargs,
    ) -> list[str]:

        results = []
        for i, instr in enumerate(instructions):
            ctx = contexts[i] if contexts else ""
            results.append(self.execute(instr, context=ctx, **kwargs))
        return results

    def test_connectivity(self) -> bool:

        try:
            result = self.execute("What is 2+2? Answer with just the number.", max_tokens=512)
            return "4" in result
        except Exception:
            return False

    @property
    def stats(self) -> dict:

        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(1, self._total_calls),
        }


    def _build_messages(self, instruction: str, context: str) -> list[dict]:

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if context:

            messages.append({
                "role": "user",
                "content": f"Here is some relevant context:\n\n{context}"
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I have read the context and will use it to answer."
            })

        messages.append({"role": "user", "content": instruction})
        return messages

    @staticmethod
    def _strip_thinking(text: str) -> str:

        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _response_text(
        self,
        response: Any,
        task_type: str,
        instruction_chars: int = 0,
    ) -> str:
        msg = response.choices[0].message
        result = self._strip_thinking(msg.content or "")
        if not result.strip():
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if reasoning:
                logger.warning(
                    "[MExec] content empty, using reasoning_content (%s chars) as fallback",
                    len(reasoning),
                )
                result = reasoning
        logger.debug(
            "[MExec] %s | instruction_chars=%s | result_chars=%s",
            task_type or "unknown",
            instruction_chars,
            len(result),
        )
        return result
