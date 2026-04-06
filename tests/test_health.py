import asyncio
import json
import tomllib
from pathlib import Path

import pytest

from bot.core.health import HealthServer, HealthState


@pytest.mark.asyncio
async def test_readiness_endpoint_tracks_runtime_state(port: int):
    state = HealthState()
    server = HealthServer(port, state.payload)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        raw = await reader.read()
        assert b"503 Service Unavailable" in raw

        state.mark_session_started("daily", 16000)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        raw = await reader.read()

        assert b"200 OK" in raw
        payload = json.loads(raw.split(b"\r\n\r\n", 1)[1])
        assert payload["ready"] is True
        assert payload["transport"] == "daily"
        assert payload["sample_rate"] == 16000
    finally:
        await server.stop()


def test_deploy_config_uses_real_defaults():
    config_path = Path(__file__).resolve().parents[1] / "pcc-deploy.toml"
    config = tomllib.loads(config_path.read_text())

    assert config["image"] == "ghcr.io/sthorat789/askjg-demo-gemini-pcc:latest"
    assert config["secret_set"] == "askjg-demo-gemini-pcc-credentials"
    assert config["ports"]["health"] == 8080
