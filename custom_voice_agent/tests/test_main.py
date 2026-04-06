import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_voice_agent.agent import VoiceAgent
from custom_voice_agent.llm.gemini_live import GeminiLiveConfig, GeminiLiveSession
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
        assert config.vad_params.confidence == 0.75
        assert config.vad_params.start_secs == 0.2
        assert config.vad_params.stop_secs == 0.2
        assert config.vad_params.min_volume == 0.6
        assert config.gemini_config.max_tokens == 8192
        assert config.gemini_config.voice_id == "Aoede"
        assert config.gemini_config.enable_affective_dialog is True
        assert config.gemini_config.proactive_audio is True
        assert config.gemini_config.thinking_budget_tokens == 0
        assert config.idle_timeout_secs == 120
        assert config.max_duration_secs == 840

    def test_runtime_config_rejects_invalid_vad_confidence(self):
        env = _base_env()
        env["VAD_CONFIDENCE"] = "1.5"

        with patch.dict("os.environ", env, clear=True), pytest.raises(ValueError):
            _build_runtime_config("voice-room", "Maya")

    def test_runtime_config_preview_matches_bot_thinking_budget(self):
        env = _base_env()
        env["THINKING_BUDGET_TOKENS"] = "32"

        with patch.dict("os.environ", env, clear=True):
            config = _build_runtime_config("voice-room", "Maya")

        assert config.gemini_config.proactive_audio is True
        assert config.gemini_config.thinking_budget_tokens == 32

    def test_runtime_config_ga_matches_bot_turn_defaults(self):
        env = {
            "LIVEKIT_URL": "wss://example.livekit.cloud",
            "GOOGLE_VERTEX_CREDENTIALS": '{"type":"service_account"}',
            "GOOGLE_CLOUD_PROJECT_ID": "demo-project",
            "GEMINI_MODEL": "ga",
            "PORT": "8080",
            "SILERO_VAD_MODEL_PATH": "/models/silero_vad.onnx",
        }

        with patch.dict("os.environ", env, clear=True):
            config = _build_runtime_config("voice-room", "Maya")

        assert config.gemini_config.model == "google/gemini-live-2.5-flash-native-audio"
        assert config.gemini_config.enable_affective_dialog is True
        assert config.gemini_config.proactive_audio is False
        assert config.gemini_config.thinking_budget_tokens == 0


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


class TestGeminiTurnSettings:
    def test_gemini_session_config_matches_preview_bot_settings(self):
        with patch.multiple(
            "custom_voice_agent.llm.gemini_live",
            LiveConnectConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            SpeechConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            VoiceConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            PrebuiltVoiceConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            Content=lambda **kwargs: SimpleNamespace(**kwargs),
            Part=lambda **kwargs: SimpleNamespace(**kwargs),
            ProactivityConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            ThinkingConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            session = GeminiLiveSession(
                GeminiLiveConfig(
                    api_key="secret-key",
                    system_instruction="hi",
                    voice_id="Aoede",
                    max_tokens=8192,
                    enable_affective_dialog=True,
                    proactive_audio=True,
                    thinking_budget_tokens=64,
                    api_version="v1alpha",
                )
            )

            live_config = session._build_live_connect_config()

            assert live_config.response_modalities == ["AUDIO"]
            assert live_config.generation_config.max_output_tokens == 8192
            assert live_config.enable_affective_dialog is True
            assert live_config.proactivity.proactive_audio is True
            assert live_config.thinking_config.include_thoughts is False
            assert live_config.thinking_config.thinking_budget == 64
            assert (
                live_config.speech_config.voice_config.prebuilt_voice_config.voice_name
                == "Aoede"
            )

    def test_gemini_session_config_matches_ga_bot_settings(self):
        with patch.multiple(
            "custom_voice_agent.llm.gemini_live",
            LiveConnectConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            SpeechConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            VoiceConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            PrebuiltVoiceConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            Content=lambda **kwargs: SimpleNamespace(**kwargs),
            Part=lambda **kwargs: SimpleNamespace(**kwargs),
            ProactivityConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            ThinkingConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            session = GeminiLiveSession(
                GeminiLiveConfig(
                    credentials='{"type":"service_account"}',
                    project_id="demo-project",
                    system_instruction="hi",
                    voice_id="Aoede",
                    max_tokens=8192,
                    enable_affective_dialog=True,
                )
            )

            live_config = session._build_live_connect_config()

            assert live_config.enable_affective_dialog is True
            assert not hasattr(live_config, "proactivity")
            assert not hasattr(live_config, "thinking_config")


class TestAgentTurnSignals:
    @pytest.mark.asyncio
    async def test_user_speaking_sends_activity_start_and_barge_in(self):
        fake_vad = MagicMock()
        fake_vad.num_frames_required = 512
        fake_vad.is_closed = False
        fake_vad.close = MagicMock()

        with patch("custom_voice_agent.agent.SileroVAD", return_value=fake_vad):
            agent = VoiceAgent(room=MagicMock(), gemini_config=GeminiLiveConfig(api_key="secret"))

        agent._gemini.send_activity_start = AsyncMock()
        agent._lk_output.cancel_and_clear = AsyncMock()
        agent._bot_speaking = True

        await agent._on_user_started_speaking()

        agent._gemini.send_activity_start.assert_awaited_once()
        agent._lk_output.cancel_and_clear.assert_awaited_once()
        assert agent._user_speaking is True
        assert agent._bot_speaking is False

    @pytest.mark.asyncio
    async def test_user_stop_sends_activity_end(self):
        fake_vad = MagicMock()
        fake_vad.num_frames_required = 512
        fake_vad.is_closed = False
        fake_vad.close = MagicMock()

        with patch("custom_voice_agent.agent.SileroVAD", return_value=fake_vad):
            agent = VoiceAgent(room=MagicMock(), gemini_config=GeminiLiveConfig(api_key="secret"))

        agent._gemini.send_activity_end = AsyncMock()
        agent._user_speaking = True

        await agent._on_user_stopped_speaking()

        agent._gemini.send_activity_end.assert_awaited_once()
        assert agent._user_speaking is False

    @pytest.mark.asyncio
    async def test_turn_complete_signals_output_end(self):
        fake_vad = MagicMock()
        fake_vad.num_frames_required = 512
        fake_vad.is_closed = False
        fake_vad.close = MagicMock()

        with patch("custom_voice_agent.agent.SileroVAD", return_value=fake_vad):
            agent = VoiceAgent(room=MagicMock(), gemini_config=GeminiLiveConfig(api_key="secret"))

        agent._lk_output.signal_end_of_response = AsyncMock()
        agent._bot_speaking = True

        await agent._handle_turn_complete()

        agent._lk_output.signal_end_of_response.assert_awaited_once()
        assert agent._bot_speaking is False
