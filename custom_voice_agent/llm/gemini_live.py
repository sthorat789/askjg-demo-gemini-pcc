#
# Gemini Live LLM session — direct google-genai usage, no Pipecat.
#
# Manages a bidirectional WebSocket session with Gemini Live API:
# - Sends user audio via send_realtime_input()
# - Receives bot audio/text via async iteration on session.receive()
# - Handles turn boundaries (turn_complete)
# - Supports both Google AI API (preview) and Vertex AI (GA) models
#
# Interruption behavior:
# Gemini handles barge-in implicitly — sending new user audio while
# it's generating causes it to stop and process the new input.
# If client-side VAD is used, send ActivityStart/ActivityEnd signals.
#

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Coroutine, Optional

try:
    from google import genai
    from google.genai import types
    from google.genai.types import (
        Content,
        GenerationConfig,
        HttpOptions,
        LiveClientRealtimeInput,
        LiveConnectConfig,
        Part,
        PrebuiltVoiceConfig,
        SpeechConfig,
        VoiceConfig,
    )
except ImportError:
    raise ImportError(
        "google-genai package is required. Install with: pip install google-genai"
    )

logger = logging.getLogger(__name__)


@dataclass
class GeminiLiveConfig:
    """Configuration for Gemini Live session.

    Supports both Google AI API (api_key) and Vertex AI (credentials).
    """

    # Model selection
    model: str = "gemini-2.5-flash-native-audio-preview-12-2025"

    # Authentication: Google AI API
    api_key: Optional[str] = None

    # Authentication: Vertex AI
    credentials: Optional[str] = None  # JSON string
    credentials_path: Optional[str] = None  # File path
    project_id: Optional[str] = None
    location: str = "us-central1"

    # Model configuration
    system_instruction: str = ""
    voice_id: str = "Aoede"
    max_tokens: int = 8192
    enable_affective_dialog: bool = True

    # API version (needed for some features)
    api_version: Optional[str] = None

    # Operational safeguards
    connect_timeout_secs: float = 20.0
    send_timeout_secs: float = 10.0
    close_timeout_secs: float = 5.0
    connect_retries: int = 3
    retry_backoff_secs: float = 1.0


class GeminiLiveSession:
    """Manages a bidirectional streaming session with Gemini Live API.

    This wraps google-genai's AsyncSession to provide:
    - Audio input forwarding to Gemini
    - Background task for receiving Gemini responses
    - Callbacks for audio output and text transcripts
    - Turn boundary detection

    Usage:
        session = GeminiLiveSession(config)
        session.on_audio_output = agent.handle_gemini_audio
        session.on_text_output = agent.handle_gemini_text
        session.on_turn_complete = agent.handle_turn_complete

        await session.connect()
        await session.send_audio(audio_bytes)  # Forward user audio
        ...
        await session.disconnect()
    """

    def __init__(self, config: GeminiLiveConfig, *, session_id: str = "unknown"):
        """Initialize the Gemini Live session.

        Args:
            config: Session configuration including auth and model settings.
        """
        self._config = config
        self._client: Optional[genai.Client] = None
        self._session = None  # genai AsyncSession
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = False
        self._session_id = session_id
        self._last_error: Optional[str] = None
        self._connect_lock = asyncio.Lock()
        self._disconnect_lock = asyncio.Lock()
        self.on_failure: Optional[Callable[[str], Coroutine]] = None

        # Callbacks (set by the agent)
        self.on_audio_output: Optional[Callable[[bytes, int], Coroutine]] = None
        self.on_text_output: Optional[Callable[[str], Coroutine]] = None
        self.on_turn_complete: Optional[Callable[[], Coroutine]] = None
        self.on_interrupted: Optional[Callable[[], Coroutine]] = None

    @property
    def _log_prefix(self) -> str:
        return f"[session_id={self._session_id}]"

    async def connect(self):
        """Establish WebSocket connection to Gemini Live API."""
        async with self._connect_lock:
            if self._connected and self._session is not None:
                return

            config = self._config
            self._last_error = None

            # Create client based on auth method
            if config.api_key:
                http_options = None
                if config.api_version:
                    http_options = HttpOptions(api_version=config.api_version)
                self._client = genai.Client(api_key=config.api_key, http_options=http_options)
                logger.info(
                    "%s Connecting to Gemini Live via Google AI API (model=%s)",
                    self._log_prefix,
                    config.model,
                )

            elif config.credentials or config.credentials_path:
                import json
                from google.oauth2 import service_account

                scopes = ["https://www.googleapis.com/auth/cloud-platform"]

                if config.credentials:
                    creds_dict = json.loads(config.credentials)
                    credentials = service_account.Credentials.from_service_account_info(
                        creds_dict, scopes=scopes
                    )
                else:
                    credentials = service_account.Credentials.from_service_account_file(
                        config.credentials_path, scopes=scopes
                    )

                self._client = genai.Client(
                    vertexai=True,
                    project=config.project_id,
                    location=config.location,
                    credentials=credentials,
                )
                logger.info(
                    "%s Connecting to Gemini Live via Vertex AI (model=%s, project=%s)",
                    self._log_prefix,
                    config.model,
                    config.project_id,
                )
            else:
                raise ValueError("Either api_key or credentials must be provided")

            live_config = LiveConnectConfig(
                response_modalities=["AUDIO"],
                generation_config=GenerationConfig(
                    max_output_tokens=config.max_tokens,
                ),
                speech_config=SpeechConfig(
                    voice_config=VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(
                            voice_name=config.voice_id,
                        )
                    )
                ),
            )

            if config.system_instruction:
                live_config.system_instruction = Content(
                    parts=[Part(text=config.system_instruction)]
                )

            last_error: Optional[Exception] = None
            for attempt in range(1, config.connect_retries + 1):
                try:
                    async with asyncio.timeout(config.connect_timeout_secs):
                        self._session = await self._client.aio.live.connect(
                            model=config.model,
                            config=live_config,
                        )
                    self._connected = True
                    self._receive_task = asyncio.create_task(self._receive_loop())
                    logger.info("%s Gemini Live session connected", self._log_prefix)
                    return
                except Exception as exc:
                    last_error = exc
                    self._session = None
                    self._connected = False
                    if attempt >= config.connect_retries:
                        break
                    delay = config.retry_backoff_secs * attempt
                    logger.warning(
                        "%s Gemini connect attempt %s/%s failed: %s",
                        self._log_prefix,
                        attempt,
                        config.connect_retries,
                        exc,
                    )
                    await asyncio.sleep(delay)

            self._last_error = str(last_error) if last_error else "Gemini connect failed"
            raise RuntimeError(self._last_error) from last_error

    async def disconnect(self):
        """Close the WebSocket connection."""
        async with self._disconnect_lock:
            self._connected = False

            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                if self._receive_task is not asyncio.current_task():
                    try:
                        await self._receive_task
                    except asyncio.CancelledError:
                        pass
            self._receive_task = None

            if self._session:
                try:
                    async with asyncio.timeout(self._config.close_timeout_secs):
                        await self._session.close()
                except Exception:
                    logger.debug("%s Error closing Gemini session", self._log_prefix, exc_info=True)
                self._session = None

            logger.info("%s Gemini Live session disconnected", self._log_prefix)

    async def send_audio(self, audio_bytes: bytes, sample_rate: int = 16000):
        """Send user audio to Gemini Live.

        Args:
            audio_bytes: Raw PCM int16 audio data.
            sample_rate: Sample rate of the audio (default: 16000).
        """
        if not self._connected or not self._session:
            return

        session = self._session
        await self._run_session_call(
            "send audio",
            session.send_realtime_input(
                audio=types.Blob(
                    data=audio_bytes,
                    mime_type=f"audio/pcm;rate={sample_rate}",
                )
            ),
        )

    async def send_text(self, text: str, role: str = "user"):
        """Send a text message to Gemini Live.

        Useful for the initial greeting trigger.

        Args:
            text: Text content to send.
            role: Message role ("user" or "model").
        """
        if not self._connected or not self._session:
            return

        session = self._session
        await self._run_session_call(
            "send text",
            session.send_client_content(
                turns=Content(role=role, parts=[Part(text=text)]),
                turn_complete=True,
            ),
        )

    async def send_activity_start(self):
        """Signal that the user started speaking (for client-side VAD).

        When using client-side VAD (not Gemini's server-side VAD),
        this tells Gemini that the user is speaking so it can
        stop generating and prepare for new input.
        """
        if not self._connected or not self._session:
            return

        session = self._session
        await self._run_session_call(
            "send activity_start",
            session.send_realtime_input(activity_start=types.ActivityStart()),
        )

    async def send_activity_end(self):
        """Signal that the user stopped speaking (for client-side VAD)."""
        if not self._connected or not self._session:
            return

        session = self._session
        await self._run_session_call(
            "send activity_end",
            session.send_realtime_input(activity_end=types.ActivityEnd()),
        )

    async def _receive_loop(self):
        """Background task: receive messages from Gemini Live API.

        Processes:
        - Audio chunks (bot speech) → on_audio_output callback
        - Text chunks (transcript) → on_text_output callback
        - Turn complete → on_turn_complete callback
        - Interrupted → on_interrupted callback
        """
        logger.info("%s Gemini receive loop started", self._log_prefix)
        try:
            async for message in self._session.receive():
                if not self._connected:
                    break

                server_content = getattr(message, "server_content", None)
                if server_content is None:
                    continue

                # Check for interruption (Gemini was interrupted by user)
                interrupted = getattr(server_content, "interrupted", False)
                if interrupted:
                    logger.debug("Gemini signaled interruption")
                    if self.on_interrupted:
                        await self.on_interrupted()
                    continue

                # Process model turn (audio + text)
                model_turn = getattr(server_content, "model_turn", None)
                if model_turn and model_turn.parts:
                    for part in model_turn.parts:
                        # Audio output from Gemini
                        inline_data = getattr(part, "inline_data", None)
                        if inline_data and inline_data.data:
                            if self.on_audio_output:
                                # Gemini outputs 24kHz PCM audio
                                await self.on_audio_output(
                                    inline_data.data, 24000
                                )

                        # Text output (transcript of bot speech)
                        text = getattr(part, "text", None)
                        if text and self.on_text_output:
                            await self.on_text_output(text)

                # Turn complete — bot finished speaking
                turn_complete = getattr(server_content, "turn_complete", False)
                if turn_complete:
                    logger.debug("Gemini turn complete")
                    if self.on_turn_complete:
                        await self.on_turn_complete()

        except asyncio.CancelledError:
            logger.info("%s Gemini receive loop cancelled", self._log_prefix)
        except Exception as exc:
            logger.exception("%s Error in Gemini receive loop", self._log_prefix)
            await self._handle_failure(f"Gemini receive loop failed: {exc}")

    @property
    def is_connected(self) -> bool:
        """Whether the session is connected."""
        return self._connected

    def health_snapshot(self) -> dict:
        return {
            "connected": self._connected,
            "receive_loop_running": self._receive_task is not None
            and not self._receive_task.done(),
            "last_error": self._last_error,
        }

    async def _run_session_call(self, action: str, operation):
        try:
            async with asyncio.timeout(self._config.send_timeout_secs):
                await operation
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("%s Failed to %s", self._log_prefix, action)
            await self._handle_failure(f"Gemini {action} failed: {exc}")

    async def _handle_failure(self, message: str):
        if self._last_error == message:
            return

        self._last_error = message
        self._connected = False
        if self.on_failure:
            await self.on_failure(message)

        if self._session:
            asyncio.create_task(self.disconnect())
