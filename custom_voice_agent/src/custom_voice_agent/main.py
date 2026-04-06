#
# Entry point for the custom voice agent with LiveKit + Gemini.
#
# This is the equivalent of bot/bot.py but without any Pipecat dependency.
# It sets up a LiveKit room, configures the agent, and runs the event loop.
#
# Usage:
#   python -m custom_voice_agent.main
#
# Required environment variables:
#   LIVEKIT_URL       — LiveKit server URL (e.g., wss://your-server.livekit.cloud)
#   LIVEKIT_API_KEY   — LiveKit API key
#   LIVEKIT_API_SECRET — LiveKit API secret
#
# Plus one of:
#   GOOGLE_API_KEY              — For Google AI API (preview model)
#   GOOGLE_VERTEX_CREDENTIALS   — For Vertex AI (GA model)
#   GOOGLE_CLOUD_PROJECT_ID     — Required for Vertex AI
#

import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from typing import Callable, Optional

try:
    from livekit import rtc
except ImportError:
    raise ImportError(
        "livekit package is required. Install with: pip install livekit"
    )

try:
    from livekit import api as livekit_api
except ImportError:
    livekit_api = None  # Token generation requires: pip install livekit-api

from custom_voice_agent.agent import VoiceAgent
from custom_voice_agent.frames import EndedReason
from custom_voice_agent.llm.gemini_live import GeminiLiveConfig
from custom_voice_agent.vad.silero_vad import VADParams

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DEFAULT_HEALTH_PORT = 8080


def _load_env():
    """Load environment variables from .env file if available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass


@dataclass
class RuntimeConfig:
    livekit_url: str
    room_name: str
    bot_name: str
    health_port: int
    max_duration_secs: float
    idle_timeout_secs: float
    vad_params: VADParams
    gemini_config: GeminiLiveConfig
    livekit_token: Optional[str] = None


class HealthServer:
    """Minimal HTTP server for liveness/readiness checks."""

    def __init__(self, port: int, payload_provider: Callable[[], dict]):
        self._port = port
        self._payload_provider = payload_provider
        self._server: Optional[asyncio.base_events.Server] = None

    async def start(self):
        self._server = await asyncio.start_server(
            self._handle_client, host="0.0.0.0", port=self._port
        )
        logger.info("Health server listening on 0.0.0.0:%s", self._port)

    async def stop(self):
        if not self._server:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        logger.info("Health server stopped")

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
        except Exception:
            logger.exception("Health server request failed")
        finally:
            writer.close()
            await writer.wait_closed()


def _build_gemini_config() -> GeminiLiveConfig:
    """Build Gemini Live configuration from environment variables.

    Supports the same model selection as the Pipecat bot:
    - GEMINI_MODEL=preview → Google AI API with preview model
    - GEMINI_MODEL=ga → Vertex AI with GA model (default)
    """
    model_type = os.getenv("GEMINI_MODEL", "ga").lower()

    # Load system prompt
    system_prompt = _load_system_prompt()
    thinking_budget = int(os.getenv("THINKING_BUDGET_TOKENS", "0"))

    if model_type == "preview":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for preview model")

        return GeminiLiveConfig(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            api_key=api_key,
            system_instruction=system_prompt,
            voice_id="Aoede",
            max_tokens=8192,
            enable_affective_dialog=True,
            proactive_audio=True,
            thinking_budget_tokens=thinking_budget,
            api_version="v1alpha",
            connect_timeout_secs=_get_positive_float("GEMINI_CONNECT_TIMEOUT_SECS", 20.0),
            send_timeout_secs=_get_positive_float("GEMINI_SEND_TIMEOUT_SECS", 10.0),
            close_timeout_secs=_get_positive_float("GEMINI_CLOSE_TIMEOUT_SECS", 5.0),
            connect_retries=_get_positive_int("GEMINI_CONNECT_RETRIES", 3),
            retry_backoff_secs=_get_positive_float("GEMINI_RETRY_BACKOFF_SECS", 1.0),
        )
    else:
        credentials = os.getenv("GOOGLE_VERTEX_CREDENTIALS")
        credentials_path = os.getenv("GOOGLE_VERTEX_CREDENTIALS_PATH")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not credentials and not credentials_path:
            raise ValueError(
                "GOOGLE_VERTEX_CREDENTIALS or GOOGLE_VERTEX_CREDENTIALS_PATH required for GA model"
            )
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT_ID required for GA model")

        return GeminiLiveConfig(
            model="google/gemini-live-2.5-flash-native-audio",
            credentials=credentials,
            credentials_path=credentials_path,
            project_id=project_id,
            location=location,
            system_instruction=system_prompt,
            voice_id="Aoede",
            max_tokens=8192,
            enable_affective_dialog=True,
            connect_timeout_secs=_get_positive_float("GEMINI_CONNECT_TIMEOUT_SECS", 20.0),
            send_timeout_secs=_get_positive_float("GEMINI_SEND_TIMEOUT_SECS", 10.0),
            close_timeout_secs=_get_positive_float("GEMINI_CLOSE_TIMEOUT_SECS", 5.0),
            connect_retries=_get_positive_int("GEMINI_CONNECT_RETRIES", 3),
            retry_backoff_secs=_get_positive_float("GEMINI_RETRY_BACKOFF_SECS", 1.0),
        )


def _load_system_prompt() -> str:
    """Load system prompt from the prompts directory."""
    from pathlib import Path

    candidate_roots = tuple(Path(__file__).resolve().parents) + (Path.cwd(),)
    for candidate_root in candidate_roots:
        prompt_path = candidate_root / "bot" / "prompts" / "demo_system_prompt.md"
        if prompt_path.exists():
            return prompt_path.read_text()

    logger.warning("System prompt not found, using default")
    return (
        "You are a friendly voice AI assistant. "
        "Keep responses brief and conversational."
    )


async def generate_livekit_token(
    room_name: str,
    participant_name: str = "bot",
) -> str:
    """Generate a LiveKit access token for the bot.

    Args:
        room_name: Name of the LiveKit room to join.
        participant_name: Identity of the bot participant.

    Returns:
        JWT access token string.
    """
    if livekit_api is None:
        raise ImportError(
            "livekit-api package is required for token generation. "
            "Install with: pip install livekit-api\n"
            "Alternatively, provide a pre-generated token via LIVEKIT_TOKEN env var."
        )

    lk_api_key = os.getenv("LIVEKIT_API_KEY")
    lk_api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not lk_api_key or not lk_api_secret:
        raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required")

    token = livekit_api.AccessToken(lk_api_key, lk_api_secret)
    token.with_identity(participant_name)
    token.with_name(participant_name)
    token.with_grants(
        livekit_api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_publish_data=True,
            can_subscribe=True,
        )
    )

    return token.to_jwt()


def _get_positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _get_positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _build_runtime_config(room_name: str, bot_name: str) -> RuntimeConfig:
    livekit_url = os.getenv("LIVEKIT_URL")
    if not livekit_url:
        raise ValueError("LIVEKIT_URL is required")

    vad_params = VADParams(
        confidence=float(os.getenv("VAD_CONFIDENCE", "0.75")),
        start_secs=_get_positive_float("VAD_START_SECS", 0.2),
        stop_secs=_get_positive_float("VAD_STOP_SECS", 0.2),
        min_volume=float(os.getenv("VAD_MIN_VOLUME", "0.6")),
    )
    if not 0.0 <= vad_params.confidence <= 1.0:
        raise ValueError("VAD_CONFIDENCE must be between 0 and 1")
    if not 0.0 <= vad_params.min_volume <= 1.0:
        raise ValueError("VAD_MIN_VOLUME must be between 0 and 1")

    config = RuntimeConfig(
        livekit_url=livekit_url,
        room_name=room_name,
        bot_name=bot_name,
        health_port=int(os.getenv("PORT", str(DEFAULT_HEALTH_PORT))),
        max_duration_secs=_get_positive_float("MAX_CALL_DURATION_SECS", 840.0),
        idle_timeout_secs=_get_positive_float("USER_IDLE_TIMEOUT_SECS", 120.0),
        vad_params=vad_params,
        gemini_config=_build_gemini_config(),
        livekit_token=os.getenv("LIVEKIT_TOKEN"),
    )
    if config.health_port <= 0:
        raise ValueError("PORT must be > 0")
    return config


def _describe_runtime_config(config: RuntimeConfig) -> dict:
    return {
        "room_name": config.room_name,
        "bot_name": config.bot_name,
        "health_port": config.health_port,
        "gemini_model": config.gemini_config.model,
        "gemini_auth": "api_key"
        if config.gemini_config.api_key
        else "vertex_credentials",
        "max_duration_secs": config.max_duration_secs,
        "idle_timeout_secs": config.idle_timeout_secs,
        "livekit_token_source": "env" if config.livekit_token else "generated",
        "vad_model_path": os.getenv("SILERO_VAD_MODEL_PATH", "bundled"),
    }


async def run_agent(
    room_name: str = "voice-agent",
    bot_name: str = "Maya",
):
    """Run the voice agent in a LiveKit room.

    This is the main entry point. It:
    1. Creates a LiveKit room connection
    2. Configures the voice agent
    3. Runs until the call ends or is interrupted

    Args:
        room_name: LiveKit room to join.
        bot_name: Name the bot introduces itself as.
    """
    config = _build_runtime_config(room_name, bot_name)
    logger.info("Runtime config: %s", json.dumps(_describe_runtime_config(config)))

    room = rtc.Room()
    agent_done = asyncio.Event()
    agent: Optional[VoiceAgent] = None
    room_connected = False

    def health_payload() -> dict:
        agent_health = agent.health_snapshot() if agent else None
        ready = room_connected and agent is not None and agent.is_running
        return {
            "status": "ok",
            "ready": ready,
            "room_connected": room_connected,
            "room_name": getattr(room, "name", config.room_name),
            "agent": agent_health,
        }

    health_server = HealthServer(config.health_port, health_payload)
    await health_server.start()

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info("Participant connected: %s", participant.identity)

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info("Participant disconnected: %s", participant.identity)

    @room.on("disconnected")
    def on_disconnected():
        nonlocal room_connected
        room_connected = False
        logger.info("Disconnected from LiveKit room")
        if agent:
            agent.request_stop(reason=EndedReason.CONNECTION_TIMED_OUT)
        agent_done.set()

    try:
        token = config.livekit_token
        if not token:
            token = await generate_livekit_token(config.room_name, participant_name=config.bot_name)

        logger.info("Connecting to LiveKit room: %s", config.room_name)
        await room.connect(config.livekit_url, token)
        room_connected = True
        logger.info("Connected to room: %s", room.name)

        agent = VoiceAgent(
            room=room,
            gemini_config=config.gemini_config,
            bot_name=config.bot_name,
            vad_params=config.vad_params,
            max_call_duration_secs=config.max_duration_secs,
            user_idle_timeout_secs=config.idle_timeout_secs,
        )

        async def on_call_ended(reason: str):
            logger.info(
                "Call ended: %s",
                json.dumps({"session_id": agent.session_id, "reason": reason}),
            )
            agent_done.set()

        agent.on_call_ended = on_call_ended
        await agent.start()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: agent.request_stop(reason=EndedReason.CUSTOMER_ENDED_CALL),
            )

        await agent_done.wait()
    finally:
        if agent:
            await agent.stop(reason=agent.ended_reason or EndedReason.CUSTOMER_ENDED_CALL)
        if room_connected:
            await room.disconnect()
        await health_server.stop()
        logger.info("Agent session complete")


def main():
    """CLI entry point."""
    _load_env()

    room_name = os.getenv("LIVEKIT_ROOM", "voice-agent")
    bot_name = os.getenv("BOT_NAME", "Maya")

    logger.info(f"Custom Voice Agent — {bot_name}")
    logger.info(f"Room: {room_name}")
    logger.info("Transport: LiveKit | LLM: Gemini Live | VAD: Silero")
    logger.info("No Pipecat dependency — custom implementation")
    logger.info("=" * 60)

    asyncio.run(run_agent(room_name=room_name, bot_name=bot_name))


if __name__ == "__main__":
    main()
