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
        session_id: str = "unknown",
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
        self._track_sid: Optional[str] = None
        self._participant_identity: Optional[str] = None
        self._session_id = session_id
        self._last_error: Optional[str] = None
        self.on_failure: Optional[Callable[[str], Coroutine]] = None

        # Register track subscription handler
        self._room.on("track_subscribed")(self._on_track_subscribed)
        self._room.on("track_unsubscribed")(self._on_track_unsubscribed)

    @property
    def _log_prefix(self) -> str:
        return f"[session_id={self._session_id}]"

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
            "%s Subscribed to audio track from %s (track=%s)",
            self._log_prefix,
            participant.identity,
            track.sid,
        )

        asyncio.create_task(self._replace_audio_stream(track, participant.identity))

    def _on_track_unsubscribed(
        self,
        track: "rtc.Track",
        publication: "rtc.RemoteTrackPublication",
        participant: "rtc.RemoteParticipant",
    ):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        asyncio.create_task(self._clear_audio_stream(track.sid))

    async def _receive_loop(self, audio_stream: "rtc.AudioStream", track_sid: str):
        """Background task: receive audio frames from LiveKit and forward them."""
        logger.info("%s LiveKit audio receive loop started (track=%s)", self._log_prefix, track_sid)
        try:
            async for event in audio_stream:
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
            logger.info("%s LiveKit audio receive loop cancelled", self._log_prefix)
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("%s Error in LiveKit audio receive loop", self._log_prefix)
            if self._running and self.on_failure:
                await self.on_failure(f"LiveKit input failed for track {track_sid}: {exc}")
        finally:
            if self._track_sid == track_sid:
                self._audio_stream = None
                self._receive_task = None

    async def start(self):
        """Start accepting audio input."""
        if self._running:
            return
        self._running = True
        self._last_error = None
        logger.info(
            "%s LiveKit input started (target_rate=%s)",
            self._log_prefix,
            self._target_sample_rate,
        )

    async def stop(self):
        """Stop receiving audio."""
        self._running = False
        await self._cancel_receive_task()
        self._track_sid = None
        self._participant_identity = None
        self._audio_stream = None
        logger.info("%s LiveKit input stopped", self._log_prefix)

    def health_snapshot(self) -> dict:
        return {
            "running": self._running,
            "track_sid": self._track_sid,
            "participant_identity": self._participant_identity,
            "receiving": self._receive_task is not None and not self._receive_task.done(),
            "last_error": self._last_error,
        }

    async def _replace_audio_stream(self, track: "rtc.Track", participant_identity: str):
        if self._track_sid == track.sid and self._receive_task and not self._receive_task.done():
            return

        await self._cancel_receive_task()
        if not self._running:
            return

        self._track_sid = track.sid
        self._participant_identity = participant_identity
        self._audio_stream = rtc.AudioStream(track)
        self._receive_task = asyncio.create_task(
            self._receive_loop(self._audio_stream, track.sid)
        )

    async def _clear_audio_stream(self, track_sid: str):
        if self._track_sid != track_sid:
            return
        logger.info("%s Audio track unsubscribed (track=%s)", self._log_prefix, track_sid)
        await self._cancel_receive_task()
        self._audio_stream = None
        self._track_sid = None
        self._participant_identity = None

    async def _cancel_receive_task(self):
        task = self._receive_task
        if task and not task.done():
            task.cancel()
            if task is not asyncio.current_task():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._receive_task = None
