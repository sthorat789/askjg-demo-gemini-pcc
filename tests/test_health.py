import asyncio
import json
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bot"))

from core.health import HealthServer, HealthState  # noqa: E402


@pytest.mark.asyncio
async def test_readyz_tracks_runtime_state(unused_tcp_port: int):
    state = HealthState()
    server = HealthServer(unused_tcp_port, state.payload)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        writer.write(b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        raw = await reader.read()
        assert b"503 Service Unavailable" in raw

        state.mark_session_started("daily", 16000)
        reader, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
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
    assert config["secret_set"] == "askjg-demo-gemini-pcc-secrets"
    assert config["ports"]["health"] == 8080
