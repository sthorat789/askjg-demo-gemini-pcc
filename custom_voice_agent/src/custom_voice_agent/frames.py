#
# Frame types for the custom voice agent pipeline.
#
# Inspired by Pipecat's frame system but implemented as simple dataclasses.
# SystemFrame types (interruptions, control) are given priority=0 so they
# are always processed before DataFrame types (audio, text) at priority=1.
#

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class FramePriority(IntEnum):
    """Frame processing priority.

    System/control frames are processed before data frames in queues.
    """

    SYSTEM = 0  # Interruptions, lifecycle events — processed first
    DATA = 1  # Audio, text — processed after system frames


# ---------------------------------------------------------------------------
# Base frame types
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """Base frame type. All frames carry a timestamp."""

    timestamp: float = field(default_factory=time.monotonic)
    priority: int = field(default=FramePriority.DATA, init=False)

    def __lt__(self, other: "Frame") -> bool:
        """Support PriorityQueue ordering: lower priority number = higher priority."""
        if not isinstance(other, Frame):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class SystemFrame(Frame):
    """Control/lifecycle frame — always processed before data frames."""

    def __post_init__(self):
        self.priority = FramePriority.SYSTEM


@dataclass
class DataFrame(Frame):
    """Data frame — audio, text, etc."""

    def __post_init__(self):
        self.priority = FramePriority.DATA


# ---------------------------------------------------------------------------
# Audio frames
# ---------------------------------------------------------------------------


@dataclass
class AudioFrame(DataFrame):
    """Raw PCM audio data."""

    audio: bytes = b""
    sample_rate: int = 16000
    num_channels: int = 1

    @property
    def num_samples(self) -> int:
        """Number of samples (per channel) in this frame."""
        bytes_per_sample = 2  # 16-bit PCM
        if len(self.audio) == 0:
            return 0
        return len(self.audio) // (bytes_per_sample * self.num_channels)

    @property
    def duration_ms(self) -> float:
        """Duration of this audio frame in milliseconds."""
        if self.sample_rate == 0 or self.num_samples == 0:
            return 0.0
        return (self.num_samples / self.sample_rate) * 1000.0


@dataclass
class InputAudioFrame(AudioFrame):
    """Audio received from the user (via LiveKit)."""

    pass


@dataclass
class OutputAudioFrame(AudioFrame):
    """Audio to send to the user (from Gemini)."""

    pass


# ---------------------------------------------------------------------------
# Text frames
# ---------------------------------------------------------------------------


@dataclass
class TextFrame(DataFrame):
    """Text content (e.g., bot transcript text from Gemini)."""

    text: str = ""
    role: str = "assistant"  # "user" or "assistant"


@dataclass
class TranscriptionFrame(DataFrame):
    """A transcription of speech."""

    text: str = ""
    role: str = "user"
    is_final: bool = False


# ---------------------------------------------------------------------------
# VAD / Speaking state frames
# ---------------------------------------------------------------------------


@dataclass
class UserStartedSpeakingFrame(SystemFrame):
    """VAD confirmed the user started speaking."""

    pass


@dataclass
class UserStoppedSpeakingFrame(SystemFrame):
    """VAD confirmed the user stopped speaking."""

    pass


@dataclass
class BotStartedSpeakingFrame(SystemFrame):
    """Bot audio output has begun."""

    pass


@dataclass
class BotStoppedSpeakingFrame(SystemFrame):
    """Bot audio output has stopped (naturally or via interruption)."""

    pass


# ---------------------------------------------------------------------------
# Interruption frame — the critical barge-in signal
# ---------------------------------------------------------------------------


@dataclass
class InterruptionFrame(SystemFrame):
    """User interrupted the bot (barge-in).

    When this frame is processed by the audio output manager:
    1. All queued audio is discarded immediately.
    2. The current audio send task is cancelled.
    3. A BotStoppedSpeakingFrame is emitted.

    Gemini handles interruption implicitly — new user audio
    causes it to stop generating and process the new input.
    """

    pass


# ---------------------------------------------------------------------------
# Lifecycle / control frames
# ---------------------------------------------------------------------------


@dataclass
class StartFrame(SystemFrame):
    """Pipeline has started."""

    pass


@dataclass
class EndFrame(SystemFrame):
    """Graceful pipeline termination."""

    reason: Optional[str] = None


@dataclass
class CancelFrame(SystemFrame):
    """Immediate pipeline termination."""

    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Metrics / events
# ---------------------------------------------------------------------------


@dataclass
class MetricsFrame(DataFrame):
    """Pipeline metrics (TTFB, processing time, etc.)."""

    name: str = ""
    value: Any = None


# ---------------------------------------------------------------------------
# Ended reason codes (Vapi-compatible, same as existing bot)
# ---------------------------------------------------------------------------


class EndedReason:
    """Standardized end-of-call reason codes."""

    CUSTOMER_ENDED_CALL = "customer-ended-call"
    ASSISTANT_ENDED_CALL = "assistant-ended-call"
    EXCEEDED_MAX_DURATION = "exceeded-max-duration"
    SILENCE_TIMED_OUT = "silence-timed-out"
    CONNECTION_TIMED_OUT = "connection-timed-out"
    PIPELINE_ERROR = "pipeline-error"
