#
# LiveKit audio output with barge-in interruption support.
#
# Replicates Pipecat's MediaSender behavior:
# - Chunks audio into ≤40 ms pieces for responsive interruption
# - Streams chunks to LiveKit via rtc.AudioSource
# - On InterruptionFrame: cancels send task, clears queue instantly
#
# This is the most critical component for barge-in — it ensures
# that bot audio stops within ~40 ms of the user starting to speak.
#

import asyncio
import logging
from typing import Callable, Coroutine, Optional

import numpy as np

try:
    from livekit import rtc
except ImportError:
    raise ImportError(
        "livekit package is required. Install with: pip install livekit"
    )

from custom_voice_agent.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    OutputAudioFrame,
)

logger = logging.getLogger(__name__)

# Default chunk size: 40ms of audio at 24kHz
DEFAULT_CHUNK_MS = 40
DEFAULT_OUTPUT_SAMPLE_RATE = 24000


def _resample_linear(
    audio_int16: np.ndarray, src_rate: int, dst_rate: int
) -> np.ndarray:
    """Linear interpolation resampling for output audio."""
    if src_rate == dst_rate:
        return audio_int16

    ratio = dst_rate / src_rate
    src_len = len(audio_int16)
    dst_len = int(src_len * ratio)

    if dst_len == 0:
        return np.array([], dtype=np.int16)

    src_float = audio_int16.astype(np.float64)
    indices = np.linspace(0, src_len - 1, dst_len)
    resampled = np.interp(indices, np.arange(src_len), src_float)

    return np.clip(resampled, -32768, 32767).astype(np.int16)


class LiveKitOutput:
    """Manages bot audio output to a LiveKit room with instant barge-in support.

    Audio flow:
    1. Gemini returns audio chunks → queue_audio()
    2. Audio is split into ≤40ms chunks and queued
    3. Background _send_loop() streams chunks to LiveKit
    4. On interruption: cancel_and_clear() stops everything instantly

    The 40ms chunking is critical — it means at most 40ms of audio
    plays after an interruption is triggered. This matches Pipecat's
    default audio_out_10ms_chunks=4 setting.

    Usage:
        output = LiveKitOutput(room, sample_rate=24000)
        await output.start()

        # Stream audio from Gemini
        await output.queue_audio(audio_bytes)

        # Barge-in: stop instantly
        await output.cancel_and_clear()

        # Cleanup
        await output.stop()
    """

    def __init__(
        self,
        room: "rtc.Room",
        *,
        sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
        num_channels: int = 1,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        livekit_sample_rate: int = 48000,
        max_queue_chunks: int = 200,
        session_id: str = "unknown",
    ):
        """Initialize LiveKit audio output.

        Args:
            room: Connected LiveKit room.
            sample_rate: Sample rate of input audio from Gemini (24000).
            num_channels: Number of audio channels (1 for mono).
            chunk_ms: Chunk size in milliseconds for interruption responsiveness.
            livekit_sample_rate: Sample rate for LiveKit output (48000).
        """
        self._room = room
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._chunk_ms = chunk_ms
        self._livekit_sample_rate = livekit_sample_rate
        self._max_queue_chunks = max_queue_chunks
        self._session_id = session_id

        # Chunk size in bytes (16-bit PCM)
        samples_per_chunk = int(sample_rate * chunk_ms / 1000)
        self._chunk_bytes = samples_per_chunk * 2 * num_channels  # 2 bytes per sample

        # Audio queue and send task
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=max_queue_chunks)
        self._send_task: Optional[asyncio.Task] = None

        # LiveKit audio source and track
        self._audio_source: Optional[rtc.AudioSource] = None
        self._local_track: Optional[rtc.LocalAudioTrack] = None

        # State
        self._bot_speaking = False
        self._running = False
        self._dropped_chunks = 0
        self._sent_chunks = 0
        self._last_error: Optional[str] = None

        # Event callbacks (set by agent)
        self.on_bot_started_speaking: Optional[asyncio.Future] = None
        self.on_bot_stopped_speaking: Optional[asyncio.Future] = None
        self.on_failure: Optional[Callable[[str], Coroutine]] = None

        # Callback for state change events
        self._event_callbacks: list = []

    @property
    def _log_prefix(self) -> str:
        return f"[session_id={self._session_id}]"

    def add_event_callback(self, callback):
        """Register a callback for bot speaking state changes."""
        self._event_callbacks.append(callback)

    async def _emit_event(self, frame):
        """Emit a frame event to all registered callbacks."""
        for cb in self._event_callbacks:
            try:
                await cb(frame)
            except Exception:
                logger.exception("Error in event callback")

    async def start(self):
        """Start the audio output: create LiveKit track and start send loop."""
        if self._running:
            return

        self._running = True
        self._last_error = None

        # Create LiveKit audio source at the LiveKit-native sample rate
        self._audio_source = rtc.AudioSource(
            self._livekit_sample_rate, self._num_channels
        )
        self._local_track = rtc.LocalAudioTrack.create_audio_track(
            "bot-audio", self._audio_source
        )

        # Publish the track
        await self._room.local_participant.publish_track(
            self._local_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        # Start the send loop
        self._send_task = asyncio.create_task(self._send_loop())
        logger.info(
            f"LiveKit output started (rate={self._sample_rate}, "
            f"chunk={self._chunk_ms}ms, lk_rate={self._livekit_sample_rate})"
        )

    async def stop(self):
        """Stop the audio output and unpublish the track."""
        if not self._running and self._send_task is None:
            return

        self._running = False

        # Cancel send task
        await self._cancel_send_task()

        # Unpublish track
        if self._local_track:
            await self._room.local_participant.unpublish_track(self._local_track.sid)
            self._local_track = None
            self._audio_source = None

        self._drain_queue()
        logger.info("%s LiveKit output stopped", self._log_prefix)

    async def queue_audio(self, audio_bytes: bytes):
        """Queue audio for output, chunked for responsive interruption.

        Args:
            audio_bytes: Raw PCM int16 audio at self._sample_rate.
        """
        if not self._running or len(audio_bytes) == 0:
            return

        # Split into chunks
        offset = 0
        while offset < len(audio_bytes):
            end = min(offset + self._chunk_bytes, len(audio_bytes))
            chunk = audio_bytes[offset:end]
            self._enqueue_queue_item(chunk)
            offset = end

    async def cancel_and_clear(self):
        """BARGE-IN: Cancel current audio and clear the queue instantly.

        This is the critical interruption handler. It:
        1. Cancels the background send task (stops audio immediately)
        2. Drains all queued audio (discards buffered content)
        3. Restarts the send task (ready for next response)
        4. Emits BotStoppedSpeakingFrame

        Result: Audio stops within ~40ms (one chunk) of this call.
        """
        # Cancel the send task
        await self._cancel_send_task()

        # Drain the queue
        drained = self._drain_queue()

        if drained > 0:
            logger.debug("%s Barge-in: drained %s audio chunks", self._log_prefix, drained)

        # Update state
        was_speaking = self._bot_speaking
        self._bot_speaking = False

        # Restart send task
        if self._running:
            self._send_task = asyncio.create_task(self._send_loop())

        # Emit event if bot was speaking
        if was_speaking:
            await self._emit_event(BotStoppedSpeakingFrame())

    @property
    def is_speaking(self) -> bool:
        """Whether the bot is currently outputting audio."""
        return self._bot_speaking

    async def _send_loop(self):
        """Background task: stream audio chunks to LiveKit.

        Pulls chunks from the queue and sends them via the LiveKit audio source.
        If cancelled (by barge-in), exits immediately.
        """
        try:
            while self._running:
                # Wait for next chunk
                chunk = await self._queue.get()
                if chunk is None:
                    # Poison pill — end of current response
                    if self._bot_speaking:
                        self._bot_speaking = False
                        await self._emit_event(BotStoppedSpeakingFrame())
                    continue

                # Emit BotStartedSpeaking on first chunk
                if not self._bot_speaking:
                    self._bot_speaking = True
                    await self._emit_event(BotStartedSpeakingFrame())

                # Resample to LiveKit's native rate if needed
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                if self._sample_rate != self._livekit_sample_rate:
                    audio_int16 = _resample_linear(
                        audio_int16, self._sample_rate, self._livekit_sample_rate
                    )

                # Create LiveKit audio frame and send
                lk_frame = rtc.AudioFrame(
                    data=audio_int16.tobytes(),
                    sample_rate=self._livekit_sample_rate,
                    num_channels=self._num_channels,
                    samples_per_channel=len(audio_int16),
                )
                await self._audio_source.capture_frame(lk_frame)
                self._sent_chunks += 1

        except asyncio.CancelledError:
            # Barge-in cancellation — this is expected and intentional
            pass
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("%s Error in audio send loop", self._log_prefix)
            if self._running and self.on_failure:
                await self.on_failure(f"LiveKit output failed: {exc}")

    async def signal_end_of_response(self):
        """Signal that the current bot response audio is complete.

        This sends a poison pill (None) to the queue so the send loop
        knows to emit BotStoppedSpeakingFrame after all audio is played.
        """
        self._enqueue_queue_item(None)

    def health_snapshot(self) -> dict:
        """Return operational details for health/readiness reporting."""
        return {
            "running": self._running,
            "queue_depth": self._queue.qsize(),
            "queue_capacity": self._max_queue_chunks,
            "dropped_chunks": self._dropped_chunks,
            "sent_chunks": self._sent_chunks,
            "bot_speaking": self._bot_speaking,
            "last_error": self._last_error,
        }

    async def _cancel_send_task(self):
        task = self._send_task
        if task and not task.done():
            task.cancel()
            if task is not asyncio.current_task():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._send_task = None

    def _drain_queue(self) -> int:
        drained = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        return drained

    def _enqueue_queue_item(self, item: Optional[bytes]):
        dropped = 0
        while self._queue.full():
            try:
                self._queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break

        if dropped:
            self._dropped_chunks += dropped
            logger.warning(
                "%s Output queue full, dropped %s queued chunk(s)",
                self._log_prefix,
                dropped,
            )

        self._queue.put_nowait(item)
