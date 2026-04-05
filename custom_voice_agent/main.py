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
import logging
import os
import signal
import sys
from typing import Optional

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


def _load_env():
    """Load environment variables from .env file if available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass


def _build_gemini_config() -> GeminiLiveConfig:
    """Build Gemini Live configuration from environment variables.

    Supports the same model selection as the Pipecat bot:
    - GEMINI_MODEL=preview → Google AI API with preview model
    - GEMINI_MODEL=ga → Vertex AI with GA model (default)
    """
    model_type = os.getenv("GEMINI_MODEL", "ga").lower()

    # Load system prompt
    system_prompt = _load_system_prompt()

    if model_type == "preview":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for preview model")

        return GeminiLiveConfig(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            api_key=api_key,
            system_instruction=system_prompt,
            voice_id="Aoede",
            api_version="v1alpha",
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
        )


def _load_system_prompt() -> str:
    """Load system prompt from the prompts directory."""
    from pathlib import Path

    # Try the bot/prompts directory (shared with Pipecat bot)
    prompt_path = Path(__file__).parent.parent / "bot" / "prompts" / "demo_system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text()

    # Fallback: check current directory
    prompt_path = Path("bot/prompts/demo_system_prompt.md")
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
        )
    )

    return token.to_jwt()


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
    livekit_url = os.getenv("LIVEKIT_URL")
    if not livekit_url:
        raise ValueError("LIVEKIT_URL is required")

    # Use pre-generated token or generate one
    token = os.getenv("LIVEKIT_TOKEN")
    if not token:
        token = await generate_livekit_token(room_name, participant_name=bot_name)

    # Create and connect to LiveKit room
    room = rtc.Room()

    # Track connection state
    agent_done = asyncio.Event()
    agent: Optional[VoiceAgent] = None

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant disconnected: {participant.identity}")
        if agent:
            asyncio.create_task(
                agent.stop(reason=EndedReason.CUSTOMER_ENDED_CALL)
            )

    @room.on("disconnected")
    def on_disconnected():
        logger.info("Disconnected from LiveKit room")
        agent_done.set()

    # Connect to room
    logger.info(f"Connecting to LiveKit room: {room_name}")
    await room.connect(livekit_url, token)
    logger.info(f"Connected to room: {room.name}")

    # Build configurations
    gemini_config = _build_gemini_config()
    vad_params = VADParams(
        confidence=float(os.getenv("VAD_CONFIDENCE", "0.75")),
        start_secs=float(os.getenv("VAD_START_SECS", "0.2")),
        stop_secs=float(os.getenv("VAD_STOP_SECS", "0.2")),
        min_volume=float(os.getenv("VAD_MIN_VOLUME", "0.6")),
    )

    max_duration = float(os.getenv("MAX_CALL_DURATION_SECS", "840"))
    idle_timeout = float(os.getenv("USER_IDLE_TIMEOUT_SECS", "120"))

    # Create the agent
    agent = VoiceAgent(
        room=room,
        gemini_config=gemini_config,
        bot_name=bot_name,
        vad_params=vad_params,
        max_call_duration_secs=max_duration,
        user_idle_timeout_secs=idle_timeout,
    )

    # Register call-ended callback
    async def on_call_ended(reason: str):
        logger.info(f"Call ended: {reason}")
        agent_done.set()

    agent.on_call_ended = on_call_ended

    # Start the agent
    await agent.start()

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(
                agent.stop(reason=EndedReason.CUSTOMER_ENDED_CALL)
            ),
        )

    # Wait until the call ends
    await agent_done.wait()

    # Cleanup
    await room.disconnect()
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
