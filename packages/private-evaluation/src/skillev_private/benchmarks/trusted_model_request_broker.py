"""Case-private Unix HTTP proxy leasing GPU capacity only during rubric requests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from skillev.contracts import canonical_json, normalize_json
from skillev.runtime.request_journal import DurableRequestJournal, UnknownRequestOutcomeError
from skillev.training import AsyncResourceLimiter


class TrustedModelRequestBroker:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        limiter: AsyncResourceLimiter,
        timeout: float,
        request_journal: DurableRequestJournal | None = None,
        task_id: str | None = None,
    ) -> None:
        self.endpoint, self.model, self.limiter, self.timeout = endpoint, model, limiter, timeout
        if request_journal is not None and not task_id:
            raise ValueError("durable grading requires the fixed trajectory task identity")
        self.request_journal, self.task_id = request_journal, task_id
        self._failed = False
        self._quota = asyncio.Semaphore(2)
        self._tasks: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> TrustedModelRequestBroker:
        if self.request_journal is not None:
            self.request_journal.require_resolved_prefix(("judge", str(self.task_id)))
        self._directory = TemporaryDirectory(prefix="skillev-judge-")
        self.socket_path = str(Path(self._directory.name) / "request.sock")
        self._server = await asyncio.start_unix_server(self._connected, self.socket_path)
        return self

    def _connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._handle(reader, writer))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def __aexit__(self, *_: object) -> None:
        self._server.close()
        await self._server.wait_closed()
        # Do not release a GPU lease while its non-cancellable HTTP thread is running.
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._directory.cleanup()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, response = 502, b'{"error":"trusted judge transport failed"}'
        try:
            async with asyncio.timeout(self.timeout + 30):
                header = await reader.readuntil(b"\r\n\r\n")
                lines = header.decode("ascii").split("\r\n")
                if lines[0] != "POST /v1/chat/completions HTTP/1.1":
                    raise ValueError("unsupported judge request path")
                headers = dict(line.split(":", 1) for line in lines[1:] if line)
                length = next(
                    int(v.strip()) for k, v in headers.items() if k.lower() == "content-length"
                )
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("judge payload exceeds the private bound")
                body = await reader.readexactly(length)
                payload = json.loads(body)
                if payload.get("model") != self.model or payload.get("stream", False):
                    raise ValueError("judge request must target the frozen base route")
            async with self._quota:
                async with self.limiter.lease(role="judge"):
                    status, response = await self._forward_drained(body)
        except UnknownRequestOutcomeError:
            response = b'{"error":"prior judge dispatch outcome is unknown; no retry"}'
        except asyncio.CancelledError:
            writer.close()
            await writer.wait_closed()
            raise
        except (OSError, ValueError, TimeoutError, StopIteration, asyncio.IncompleteReadError):
            pass
        try:
            writer.write(
                (
                    f"HTTP/1.1 {status} Result\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(response)}\r\nConnection: close\r\n\r\n"
                ).encode()
                + response
            )
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _forward_drained(self, body: bytes) -> tuple[int, bytes]:
        """Use the actor's ownership rule: a cancelled HTTP thread still owns its slot."""
        task = asyncio.create_task(asyncio.to_thread(self._persisted_forward, body))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if task.done() and not task.cancelled():
                task.exception()  # consume an underlying transport error; retain cancellation
            raise

    def _persisted_forward(self, body: bytes) -> tuple[int, bytes]:
        if self._failed:
            raise UnknownRequestOutcomeError("this grading case already has a transport failure")
        if self.request_journal is None:
            return self._forward(body)
        payload = normalize_json(json.loads(body))
        if not isinstance(payload, dict):
            raise ValueError("judge request must be an object")

        def send() -> tuple[int, str]:
            status, response = self._forward(body)
            return status, response.decode("utf-8")

        # The trusted worker already distinguishes parse-attempt prompts. Exact
        # body identity is independent of concurrent rubric arrival order. It
        # remains private (including rubric text), not a model-visible request ID.
        try:
            status, response = self.request_journal.request(
                identity=("judge", str(self.task_id), canonical_json(payload)),
                endpoint=self.endpoint,
                payload=payload,
                send=send,
            )
        except Exception:
            # Set before returning an HTTP failure to the worker. Its scorer's
            # parse-repair path must not create a new model sample after a lost
            # response. Requests already in flight still drain normally.
            self._failed = True
            raise
        if not isinstance(response, str):
            raise ValueError("stored judge response is not HTTP text")
        return status, response.encode("utf-8")

    def _forward(self, body: bytes) -> tuple[int, bytes]:
        request = Request(  # noqa: S310 - validated HTTP(S) deployment endpoint
            self.endpoint.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - deployment-pinned private endpoint
                return response.status, response.read(8 * 1024 * 1024)
        except HTTPError as error:
            return error.code, error.read(8 * 1024 * 1024)
