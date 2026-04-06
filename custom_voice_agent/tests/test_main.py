import asyncio
import json
from unittest.mock import patch

import pytest

from custom_voice_agent.main import HealthServer, _build_runtime_config


def _base_env():
    return {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "GOOGLE_API_KEY": "secret-key",
        "GEMINI_MODEL": "preview",
        "PORT": "8080",
        "SILERO_VAD_MODEL_PATH": "/models/silero_vad.onnx",
    }


class TestRuntimeConfig:
    def test_runtime_config_uses_validated_environment(self):
        with patch.dict("os.environ", _base_env(), clear=True):
            config = _build_runtime_config("voice-room", "Maya")

        assert config.livekit_url == "wss://example.livekit.cloud"
        assert config.health_port == 8080
        assert config.gemini_config.api_key == "secret-key"

    def test_runtime_config_rejects_invalid_vad_confidence(self):
        env = _base_env()
        env["VAD_CONFIDENCE"] = "1.5"

        with patch.dict("os.environ", env, clear=True), pytest.raises(ValueError):
            _build_runtime_config("voice-room", "Maya")


class TestHealthServer:
    @pytest.mark.asyncio
    async def test_readyz_returns_503_until_ready(self):
        state = {"ready": False}

        def payload():
            return {"status": "ok", "ready": state["ready"]}

        server = HealthServer(18080, payload)
        await server.start()

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 18080)
            writer.write(b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            raw = await reader.read()
            assert b"503 Service Unavailable" in raw

            state["ready"] = True
            reader, writer = await asyncio.open_connection("127.0.0.1", 18080)
            writer.write(b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            raw = await reader.read()
            assert b"200 OK" in raw
            assert json.loads(raw.split(b"\r\n\r\n", 1)[1])["ready"] is True
        finally:
            await server.stop()
