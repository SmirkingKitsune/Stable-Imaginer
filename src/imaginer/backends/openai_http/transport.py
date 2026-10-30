"""One httpx client per engine, with timeouts, retries and typed errors.

Retry policy:
* Connection failures before the request was sent (refused, DNS, connect/pool timeout)
  are always retried. Nothing reached the engine, so even a POST is safe to repeat.
* HTTP 503 is retried for every method: the engine refused the request without running
  it (llama-server answers 503 while its model is still loading).
* 502/504, dropped connections and read timeouts are retried only for idempotent
  requests, because the engine may already have started working on them.

Every failure surfaces as ``EngineUnavailableError`` (503), ``EngineTimeoutError`` (504)
or ``EngineError`` (502), never as a raw httpx exception.
"""

import asyncio
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from imaginer.core.config import RetryConfig, TimeoutConfig
from imaginer.core.logging import get_logger
from imaginer.domain.errors import (
    EngineError,
    EngineTimeoutError,
    EngineUnavailableError,
    ImaginerError,
)

log = get_logger(__name__)

_IDEMPOTENT = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_MAX_BODY_IN_ERROR = 500


class HttpTransport:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        timeouts: TimeoutConfig,
        retry: RetryConfig,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self._retry = retry
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=timeouts.connect,
                read=timeouts.read,
                write=timeouts.write,
                pool=timeouts.pool,
            ),
        )

    @property
    def client(self) -> httpx.AsyncClient:
        """The underlying client, for passing requests through as they are (the link proxy)."""
        return self._client

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        attempts: int | None = None,
        accept_status: frozenset[int] = frozenset(),
    ) -> httpx.Response:
        """Send a request; return the response if it is 2xx (or in ``accept_status``)."""
        method = method.upper()
        max_attempts = attempts or self._retry.attempts
        extra: dict[str, Any] = {}
        if timeout_s is not None:
            extra["timeout"] = timeout_s
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.request(
                    method, path, json=json, params=params, **extra
                )
            except httpx.HTTPError as exc:
                error = self._map_transport_error(exc, path)
                if attempt < max_attempts and self._retryable_exception(exc, method):
                    await self._backoff(attempt, path, error)
                    continue
                raise error from exc
            if response.is_success or response.status_code in accept_status:
                return response
            error = _status_error(self.name, path, response.status_code, response.text)
            if attempt < max_attempts and self._retryable_status(response.status_code, method):
                await self._backoff(attempt, path, error)
                continue
            raise error
        raise AssertionError("unreachable")  # pragma: no cover

    async def get_json(self, path: str, **kwargs: Any) -> Any:
        return (await self.request("GET", path, **kwargs)).json()

    async def post_json(self, path: str, body: Any, **kwargs: Any) -> Any:
        return (await self.request("POST", path, json=body, **kwargs)).json()

    @asynccontextmanager
    async def stream(
        self, method: str, path: str, *, json: Any = None
    ) -> AsyncIterator[httpx.Response]:
        """Open a streaming response. Retries only until response headers arrive.

        After that, a broken stream raises ``EngineUnavailableError``/``EngineTimeoutError``
        and is never replayed: the engine has already produced output.
        """
        method = method.upper()
        for attempt in range(1, self._retry.attempts + 1):
            request = self._client.build_request(method, path, json=json)
            try:
                response = await self._client.send(request, stream=True)
            except httpx.HTTPError as exc:
                error = self._map_transport_error(exc, path)
                if attempt < self._retry.attempts and isinstance(exc, _NOT_SENT):
                    await self._backoff(attempt, path, error)
                    continue
                raise error from exc
            try:
                if not response.is_success:
                    body = (await response.aread()).decode(errors="replace")
                    error = _status_error(self.name, path, response.status_code, body)
                    if attempt < self._retry.attempts and response.status_code == 503:
                        await self._backoff(attempt, path, error)
                        continue
                    raise error
                try:
                    yield response
                except httpx.HTTPError as exc:
                    raise self._map_transport_error(exc, path) from exc
                return
            finally:
                await response.aclose()

    # --- policy ---------------------------------------------------------------

    @staticmethod
    def _retryable_exception(exc: httpx.HTTPError, method: str) -> bool:
        if isinstance(exc, _NOT_SENT):
            return True
        return method in _IDEMPOTENT and isinstance(exc, httpx.TransportError)

    @staticmethod
    def _retryable_status(status: int, method: str) -> bool:
        return status == 503 or (status in (502, 504) and method in _IDEMPOTENT)

    async def _backoff(self, attempt: int, path: str, error: ImaginerError) -> None:
        ceiling = min(self._retry.backoff_max_s, self._retry.backoff_initial_s * 2 ** (attempt - 1))
        delay = ceiling / 2 + random.uniform(0, ceiling / 2)  # equal jitter
        log.info(
            "engine.retry",
            backend=self.name,
            path=path,
            attempt=attempt,
            delay_s=round(delay, 3),
            error=error.detail,
        )
        await asyncio.sleep(delay)

    def _map_transport_error(self, exc: httpx.HTTPError, path: str) -> ImaginerError:
        where = f"{self.name} ({self.base_url}{path})"
        if isinstance(exc, httpx.ConnectTimeout | httpx.ConnectError | httpx.PoolTimeout):
            return EngineUnavailableError(f"{where} is unreachable: {exc!s}", backend=self.name)
        if isinstance(exc, httpx.TimeoutException):
            return EngineTimeoutError(f"{where} timed out: {exc!s}", backend=self.name)
        return EngineUnavailableError(f"{where} failed: {exc!r}", backend=self.name)


def _status_error(backend: str, path: str, status: int, body: str) -> ImaginerError:
    snippet = body.strip()[:_MAX_BODY_IN_ERROR]
    detail = f"{backend} returned HTTP {status} for {path}: {snippet}"
    if status == 503:
        return EngineUnavailableError(detail, backend=backend, engine_status=status)
    if status == 504:
        return EngineTimeoutError(detail, backend=backend, engine_status=status)
    return EngineError(detail, backend=backend, engine_status=status)
