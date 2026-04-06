#
# VoiceAgent — the central orchestrator for the custom voice pipeline.
#
# This replaces Pipecat's Pipeline + PipelineTask with a purpose-built
# event-driven agent that coordinates:
#   - LiveKit audio input → VAD → Gemini
#   - Gemini audio output → LiveKit (with barge-in support)
#   - Session timer (max call duration)
#   - User idle detection
#
# The agent manages all state transitions and ensures that interruption
# (barge-in) is handled correctly at every stage.
#

import asyncio
import logging
import time
import uuid
from typing import Callable, Coroutine, Optional

from custom_voice_agent.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndedReason,
    InputAudioFrame,
    InterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from custom_voice_agent.llm.gemini_live import GeminiLiveConfig, GeminiLiveSession
from custom_voice_agent.transport.livekit_input import LiveKitInput
from custom_voice_agent.transport.livekit_output import LiveKitOutput
from custom_voice_agent.vad.silero_vad import SileroVAD, VADEvent, VADParams

try:
    from livekit import rtc
except ImportError:
    raise ImportError("livekit package is required. Install with: pip install livekit")

logger = logging.getLogger(__name__)


class VoiceAgent:
    """Custom voice agent with Gemini Live + LiveKit + Silero VAD.

    This is the equivalent of Pipecat's Pipeline + PipelineTask, but
    purpose-built for the Gemini + LiveKit use case.

    Architecture:
        LiveKit (user audio) → VAD → Gemini Live → LiveKit (bot audio)

    Key behaviors:
    - VAD detects user speech with 4-state machine (QUIET/STARTING/SPEAKING/STOPPING)
    - When user speaks while bot is talking → BARGE-IN:
      1. Output audio queue cleared instantly (~40ms latency)
      2. Gemini stops generating (implicit, via new audio input)
      3. ActivityStart signal sent to Gemini
    - Session timer enforces max call duration
    - User idle timeout detects silence and ends call

    Usage:
        agent = VoiceAgent(room, gemini_config, ...)
        await agent.start()
        # ... agent runs until stopped ...
        await agent.stop()
    """

    def __init__(
        self,
        room: "rtc.Room",
        gemini_config: GeminiLiveConfig,
        *,
        bot_name: str = "Maya",
        vad_params: Optional[VADParams] = None,
        max_call_duration_secs: float = 840,
        user_idle_timeout_secs: float = 120,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
    ):
        """Initialize the voice agent.

        Args:
            room: Connected LiveKit room.
            gemini_config: Configuration for Gemini Live session.
            bot_name: Name the bot introduces itself as.
            vad_params: VAD parameters (defaults to project settings).
            max_call_duration_secs: Maximum call duration before auto-termination.
            user_idle_timeout_secs: User silence timeout before ending call.
            input_sample_rate: Sample rate for user audio (16000 for Gemini).
            output_sample_rate: Sample rate for bot audio (24000 from Gemini).
        """
        self._room = room
        self._bot_name = bot_name
        self._session_id = uuid.uuid4().hex
        self._max_duration = max_call_duration_secs
        self._idle_timeout = user_idle_timeout_secs
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._vad_params = vad_params or VADParams()

        # Core components
        self._vad = SileroVAD(
            sample_rate=input_sample_rate,
            params=self._vad_params,
        )
        self._gemini = GeminiLiveSession(gemini_config, session_id=self._session_id)
        self._lk_input = LiveKitInput(
            room,
            on_audio_frame=self._handle_audio_input,
            target_sample_rate=input_sample_rate,
            session_id=self._session_id,
        )
        self._lk_output = LiveKitOutput(
            room,
            sample_rate=output_sample_rate,
            livekit_sample_rate=48000,
            session_id=self._session_id,
        )

        # State tracking
        self._user_speaking = False
        self._bot_speaking = False
        self._running = False
        self._ended_reason: Optional[str] = None
        self._last_error: Optional[str] = None
        self._start_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._stop_task: Optional[asyncio.Task] = None
        self._call_ended_notified = False

        # Timers
        self._session_timer_task: Optional[asyncio.Task] = None
        self._idle_timer_task: Optional[asyncio.Task] = None
        self._last_user_activity: float = 0.0

        # VAD buffer: accumulates audio until we have enough for one inference
        self._vad_buffer = bytearray()
        self._vad_frame_bytes = self._vad.num_frames_required * 2  # 16-bit PCM

        # Event callbacks (for external listeners like transcript/reporting)
        self.on_bot_text: Optional[Callable[[str], Coroutine]] = None
        self.on_user_started_speaking: Optional[Callable[[], Coroutine]] = None
        self.on_user_stopped_speaking: Optional[Callable[[], Coroutine]] = None
        self.on_bot_started_speaking: Optional[Callable[[], Coroutine]] = None
        self.on_bot_stopped_speaking: Optional[Callable[[], Coroutine]] = None
        self.on_call_ended: Optional[Callable[[str], Coroutine]] = None

        # Wire up output events
        self._lk_output.add_event_callback(self._handle_output_event)

        # Wire up Gemini callbacks
        self._gemini.on_audio_output = self._handle_gemini_audio
        self._gemini.on_text_output = self._handle_gemini_text
        self._gemini.on_turn_complete = self._handle_turn_complete
        self._gemini.on_interrupted = self._handle_gemini_interrupted
        self._gemini.on_failure = self._handle_component_failure
        self._lk_input.on_failure = self._handle_component_failure
        self._lk_output.on_failure = self._handle_component_failure

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def _log_prefix(self) -> str:
        return f"[session_id={self._session_id}]"

    async def start(self):
        """Start the voice agent.

        This:
        1. Connects to Gemini Live
        2. Starts LiveKit input/output
        3. Starts session timer and idle detection
        4. Triggers the initial greeting
        """
        async with self._start_lock:
            if self._running:
                return

            logger.info(
                "%s Starting VoiceAgent '%s' (max_duration=%ss, idle_timeout=%ss)",
                self._log_prefix,
                self._bot_name,
                self._max_duration,
                self._idle_timeout,
            )

            self._running = True
            self._ended_reason = None
            self._last_error = None
            self._call_ended_notified = False
            self._last_user_activity = time.monotonic()
            self._vad_buffer.clear()
            self._user_speaking = False
            self._bot_speaking = False
            self._stop_task = None

            if self._vad.is_closed:
                self._vad = SileroVAD(
                    sample_rate=self._input_sample_rate,
                    params=self._vad_params,
                )

            try:
                await self._gemini.connect()
                await self._lk_input.start()
                await self._lk_output.start()

                self._session_timer_task = asyncio.create_task(self._session_timer())
                self._idle_timer_task = asyncio.create_task(self._idle_timer())

                await self._gemini.send_text(
                    f"Start the conversation. Introduce yourself as {self._bot_name}."
                )
            except Exception:
                self._running = False
                logger.exception("%s VoiceAgent failed to start", self._log_prefix)
                await self._shutdown_components()
                raise

            logger.info("%s VoiceAgent started — waiting for user", self._log_prefix)

    async def stop(self, reason: Optional[str] = None):
        """Stop the voice agent gracefully.

        Args:
            reason: Why the call ended (EndedReason constant).
        """
        async with self._stop_lock:
            if self._ended_reason is None:
                self._ended_reason = reason or EndedReason.CUSTOMER_ENDED_CALL
            elif reason and self._ended_reason == EndedReason.CUSTOMER_ENDED_CALL:
                self._ended_reason = reason

            if not self._running and self._call_ended_notified:
                return

            self._running = False
            logger.info(
                "%s Stopping VoiceAgent (reason=%s)",
                self._log_prefix,
                self._ended_reason,
            )

            await self._cancel_task(self._session_timer_task)
            await self._cancel_task(self._idle_timer_task)
            self._session_timer_task = None
            self._idle_timer_task = None

            await self._shutdown_components()

            if self.on_call_ended and not self._call_ended_notified:
                self._call_ended_notified = True
                await self.on_call_ended(self._ended_reason)

            logger.info("%s VoiceAgent stopped", self._log_prefix)

    @property
    def ended_reason(self) -> Optional[str]:
        """Why the call ended, or None if still running."""
        return self._ended_reason

    def request_stop(self, reason: Optional[str] = None):
        """Schedule a safe asynchronous stop without self-cancellation."""
        if self._stop_task and not self._stop_task.done():
            return self._stop_task

        self._stop_task = asyncio.create_task(self.stop(reason=reason))
        self._stop_task.add_done_callback(self._log_stop_result)
        return self._stop_task

    def health_snapshot(self) -> dict:
        return {
            "session_id": self._session_id,
            "running": self._running,
            "ended_reason": self._ended_reason,
            "user_speaking": self._user_speaking,
            "bot_speaking": self._bot_speaking,
            "last_error": self._last_error,
            "idle_for_secs": max(0.0, time.monotonic() - self._last_user_activity),
            "transport_input": self._lk_input.health_snapshot(),
            "transport_output": self._lk_output.health_snapshot(),
            "gemini": self._gemini.health_snapshot(),
        }

    # -----------------------------------------------------------------------
    # Audio input processing (LiveKit → VAD → Gemini)
    # -----------------------------------------------------------------------

    async def _handle_audio_input(self, frame: InputAudioFrame):
        """Process incoming audio from LiveKit.

        1. Accumulate audio into VAD-sized chunks
        2. Run VAD on each complete chunk
        3. Handle state transitions (barge-in, etc.)
        4. Forward all audio to Gemini
        """
        if not self._running:
            return

        # Update idle timer
        self._last_user_activity = time.monotonic()

        # Forward audio to Gemini immediately (Gemini needs continuous audio)
        await self._gemini.send_audio(frame.audio, frame.sample_rate)

        # Accumulate for VAD
        self._vad_buffer.extend(frame.audio)

        # Process complete VAD chunks
        while len(self._vad_buffer) >= self._vad_frame_bytes:
            vad_chunk = bytes(self._vad_buffer[: self._vad_frame_bytes])
            self._vad_buffer = self._vad_buffer[self._vad_frame_bytes :]

            # Run VAD (non-blocking, uses thread pool)
            event = await self._vad.analyze(vad_chunk)

            if event == VADEvent.SPEECH_STARTED:
                await self._on_user_started_speaking()
            elif event == VADEvent.SPEECH_STOPPED:
                await self._on_user_stopped_speaking()

    async def _on_user_started_speaking(self):
        """Handle user starting to speak."""
        if self._user_speaking:
            return  # Already speaking

        self._user_speaking = True
        logger.debug("User started speaking")

        # Notify Gemini of user activity (client-side VAD signal)
        await self._gemini.send_activity_start()

        # BARGE-IN: If bot is currently speaking, interrupt it
        if self._bot_speaking:
            logger.info("BARGE-IN: User interrupted bot — cancelling output")
            await self._lk_output.cancel_and_clear()
            self._bot_speaking = False

        # Emit event
        if self.on_user_started_speaking:
            await self.on_user_started_speaking()

    async def _on_user_stopped_speaking(self):
        """Handle user stopping speaking."""
        if not self._user_speaking:
            return  # Wasn't speaking

        self._user_speaking = False
        logger.debug("User stopped speaking")

        # Notify Gemini of turn boundary
        await self._gemini.send_activity_end()

        # Emit event
        if self.on_user_stopped_speaking:
            await self.on_user_stopped_speaking()

    # -----------------------------------------------------------------------
    # Gemini output processing
    # -----------------------------------------------------------------------

    async def _handle_gemini_audio(self, audio_bytes: bytes, sample_rate: int):
        """Handle audio output from Gemini.

        Chunks audio and queues it for LiveKit output.
        The output manager handles the actual streaming.
        """
        if not self._running:
            return

        self._bot_speaking = True
        await self._lk_output.queue_audio(audio_bytes)

    async def _handle_gemini_text(self, text: str):
        """Handle text transcript from Gemini (bot's speech as text)."""
        if not self._running:
            return
        if self.on_bot_text:
            await self.on_bot_text(text)

    async def _handle_turn_complete(self):
        """Handle Gemini turn completion (bot finished speaking)."""
        logger.debug("Gemini turn complete — signaling end of response")
        self._bot_speaking = False
        await self._lk_output.signal_end_of_response()

    async def _handle_gemini_interrupted(self):
        """Handle Gemini reporting it was interrupted.

        This is Gemini's server-side signal that it stopped generating
        because the user started speaking. We may have already handled
        this via client-side VAD, but this ensures consistency.
        """
        logger.debug("Gemini reported interruption")
        if self._bot_speaking:
            await self._lk_output.cancel_and_clear()
            self._bot_speaking = False

    # -----------------------------------------------------------------------
    # Output event handling
    # -----------------------------------------------------------------------

    async def _handle_output_event(self, frame):
        """Handle events from the output manager (bot speaking state)."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if self.on_bot_started_speaking:
                await self.on_bot_started_speaking()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self.on_bot_stopped_speaking:
                await self.on_bot_stopped_speaking()

    # -----------------------------------------------------------------------
    # Safeguard timers
    # -----------------------------------------------------------------------

    async def _session_timer(self):
        """Enforce maximum call duration."""
        try:
            await asyncio.sleep(self._max_duration)
            if self._running:
                logger.warning(
                    "%s Session max duration (%ss) reached, ending call",
                    self._log_prefix,
                    self._max_duration,
                )
                self.request_stop(reason=EndedReason.EXCEEDED_MAX_DURATION)
        except asyncio.CancelledError:
            pass

    async def _idle_timer(self):
        """Monitor for user inactivity and end call if idle too long."""
        try:
            while self._running:
                await asyncio.sleep(5)  # Check every 5 seconds

                idle_time = time.monotonic() - self._last_user_activity
                if idle_time >= self._idle_timeout:
                    logger.warning(
                        "%s User idle timeout (%ss) reached, ending call",
                        self._log_prefix,
                        self._idle_timeout,
                    )
                    self.request_stop(reason=EndedReason.SILENCE_TIMED_OUT)
                    return
        except asyncio.CancelledError:
            pass

    async def _handle_component_failure(self, message: str):
        self._last_error = message
        logger.error("%s %s", self._log_prefix, message)
        self.request_stop(reason=EndedReason.PIPELINE_ERROR)

    async def _shutdown_components(self):
        results = await asyncio.gather(
            self._lk_output.stop(),
            self._lk_input.stop(),
            self._gemini.disconnect(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.exception(
                    "%s Error while shutting down VoiceAgent component",
                    self._log_prefix,
                    exc_info=result,
                )

        try:
            self._vad.close()
        except Exception:
            logger.debug("%s Error while closing VAD", self._log_prefix, exc_info=True)

    async def _cancel_task(self, task: Optional[asyncio.Task]):
        if task and not task.done():
            task.cancel()
            if task is not asyncio.current_task():
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _log_stop_result(self, task: asyncio.Task):
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("%s VoiceAgent stop task failed", self._log_prefix)
