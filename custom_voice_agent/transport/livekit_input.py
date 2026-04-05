#
# LiveKit audio input handler.
#
# Subscribes to remote audio tracks in a LiveKit room and feeds
# audio frames to the voice agent pipeline.
#
# Handles sample-rate conversion (48 kHz LiveKit → 16 kHz for VAD/Gemini).
#

import asyncio
import logging
import struct
from typing import Callable, Coroutine, Optional

import numpy as np

try:
    from livekit import rtc
except ImportError:
    raise ImportError(
        "livekit package is required. Install with: pip install livekit"
    )

from custom_voice_agent.frames import InputAudioFrame

logger = logging.getLogger(__name__)

# Target sample rate for VAD and Gemini
TARGET_SAMPLE_RATE = 16000


def _resample_linear(
    audio_int16: np.ndarray, src_rate: int, dst_rate: int
) -> np.ndarray:
    """Simple linear interpolation resampling.

    Good enough for speech audio. For production, consider using a
    proper resampler like soxr or scipy.signal.resample_poly.

    Args:
        audio_int16: Input samples as int16 numpy array.
        src_rate: Source sample rate.
        dst_rate: Target sample rate.

    Returns:
        Resampled int16 numpy array.
    """
    if src_rate == dst_rate:
        return audio_int16

    ratio = dst_rate / src_rate
    src_len = len(audio_int16)
    dst_len = int(src_len * ratio)

    if dst_len == 0:
        return np.array([], dtype=np.int16)

    # Linear interpolation
    src_float = audio_int16.astype(np.float64)
    indices = np.linspace(0, src_len - 1, dst_len)
    resampled = np.interp(indices, np.arange(src_len), src_float)

    return np.clip(resampled, -32768, 32767).astype(np.int16)


class LiveKitInput:
    """Receives audio from a LiveKit room and emits InputAudioFrame.

    Usage:
        lk_input = LiveKitInput(room, on_audio_frame=agent.handle_audio_input)
        await lk_input.start()
        ...
        await lk_input.stop()
    """

    def __init__(
        self,
        room: "rtc.Room",
        *,
        on_audio_frame: Callable[[InputAudioFrame], Coroutine],
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ):
        """Initialize LiveKit audio input.

        Args:
            room: Connected LiveKit room.
            on_audio_frame: Async callback for each audio frame (resampled).
            target_sample_rate: Output sample rate (default: 16000 for Gemini).
        """
        self._room = room
        self._on_audio_frame = on_audio_frame
        self._target_sample_rate = target_sample_rate
        self._audio_stream: Optional[rtc.AudioStream] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False

        # Register track subscription handler
        self._room.on("track_subscribed")(self._on_track_subscribed)

    def _on_track_subscribed(
        self,
        track: "rtc.Track",
        publication: "rtc.RemoteTrackPublication",
        participant: "rtc.RemoteParticipant",
    ):
        """Handle new track subscription — start receiving audio."""
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        logger.info(
            f"Subscribed to audio track from {participant.identity} "
            f"(track={track.sid})"
        )

        # Only handle one audio track at a time
        if self._audio_stream is not None:
            logger.warning("Already receiving audio, ignoring additional track")
            return

        self._audio_stream = rtc.AudioStream(track)
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        """Background task: receive audio frames from LiveKit and forward them."""
        logger.info("LiveKit audio receive loop started")
        try:
            async for event in self._audio_stream:
                if not self._running:
                    break

                frame = event.frame
                # LiveKit gives us rtc.AudioFrame with .data (bytes), .sample_rate, etc.
                src_rate = frame.sample_rate
                num_channels = frame.num_channels

                # Convert to numpy int16
                audio_int16 = np.frombuffer(frame.data, dtype=np.int16)

                # Mix to mono if stereo
                if num_channels > 1:
                    audio_int16 = audio_int16.reshape(-1, num_channels).mean(axis=1).astype(np.int16)

                # Resample to target rate
                if src_rate != self._target_sample_rate:
                    audio_int16 = _resample_linear(audio_int16, src_rate, self._target_sample_rate)

                # Emit as InputAudioFrame
                audio_frame = InputAudioFrame(
                    audio=audio_int16.tobytes(),
                    sample_rate=self._target_sample_rate,
                    num_channels=1,
                )
                await self._on_audio_frame(audio_frame)

        except asyncio.CancelledError:
            logger.info("LiveKit audio receive loop cancelled")
        except Exception:
            logger.exception("Error in LiveKit audio receive loop")

    async def start(self):
        """Start accepting audio input."""
        self._running = True
        logger.info(f"LiveKit input started (target_rate={self._target_sample_rate})")

    async def stop(self):
        """Stop receiving audio."""
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        self._audio_stream = None
        logger.info("LiveKit input stopped")
