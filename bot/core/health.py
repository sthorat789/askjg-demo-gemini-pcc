import asyncio
import json
import os
from typing import Callable, Optional

from loguru import logger


class HealthState:
    """Shared process health state for liveness/readiness probes."""

    def __init__(self):
        self._ready = False
        self._active_session = False
        self._client_connected = False
        self._transport: Optional[str] = None
        self._sample_rate: Optional[int] = None
        self._last_reason: Optional[str] = None
        self._last_error: Optional[str] = None

    def mark_session_started(self, transport: str, sample_rate: int):
        self._ready = True
        self._active_session = True
        self._client_connected = False
        self._transport = transport
        self._sample_rate = sample_rate
        self._last_error = None

    def mark_client_connected(self):
        self._client_connected = True

    def mark_client_disconnected(self):
        self._client_connected = False

    def mark_session_finished(self, reason: Optional[str] = None):
        self._active_session = False
        self._client_connected = False
        self._last_reason = reason or self._last_reason

    def mark_error(self, error: Exception | str):
        self._last_error = str(error)

    def mark_not_ready(self):
        self._ready = False

    def payload(self) -> dict:
        return {
            "status": "ok" if not self._last_error else "degraded",
            "ready": self._ready and not self._last_error,
            "active_session": self._active_session,
            "client_connected": self._client_connected,
            "transport": self._transport,
            "sample_rate": self._sample_rate,
            "last_reason": self._last_reason,
            "last_error": self._last_error,
        }


class HealthServer:
    """Minimal HTTP server for liveness/readiness checks."""

    def __init__(self, port: int, payload_provider: Callable[[], dict]):
        self._port = port
        self._payload_provider = payload_provider
        self._server: Optional[asyncio.base_events.Server] = None

    async def start(self):
        if self._server:
            return
        self._server = await asyncio.start_server(
            self._handle_client, host="0.0.0.0", port=self._port
        )
        logger.info(f"Health server listening on 0.0.0.0:{self._port}")

    async def stop(self):
        if not self._server:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        try:
            request_line = await reader.readline()
            parts = request_line.decode("utf-8", errors="ignore").strip().split()
            path = parts[1] if len(parts) >= 2 else "/healthz"

            while True:
                header = await reader.readline()
                if header in (b"", b"\r\n", b"\n"):
                    break

            payload = self._payload_provider()
            ready = payload.get("ready", False)

            if path in ("/ready", "/readyz"):
                response_body = json.dumps(payload).encode("utf-8")
                status_line = (
                    "HTTP/1.1 200 OK\r\n" if ready else "HTTP/1.1 503 Service Unavailable\r\n"
                )
            elif path in ("/health", "/healthz", "/"):
                response_body = json.dumps(payload).encode("utf-8")
                status_line = "HTTP/1.1 200 OK\r\n"
            else:
                response_body = b'{"error":"not-found"}'
                status_line = "HTTP/1.1 404 Not Found\r\n"

            headers = (
                f"{status_line}"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(response_body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("utf-8")
            writer.write(headers + response_body)
            await writer.drain()
        except Exception as exc:
            logger.exception(f"Health server request failed: {exc}")
        finally:
            writer.close()
            await writer.wait_closed()


_HEALTH_PORT = int(os.getenv("PORT", "8080"))
_HEALTH_SERVER: Optional[HealthServer] = None
_HEALTH_LOCK = asyncio.Lock()
HEALTH_STATE = HealthState()


async def ensure_health_server():
    global _HEALTH_SERVER

    async with _HEALTH_LOCK:
        if _HEALTH_SERVER is None:
            _HEALTH_SERVER = HealthServer(_HEALTH_PORT, HEALTH_STATE.payload)
            await _HEALTH_SERVER.start()
