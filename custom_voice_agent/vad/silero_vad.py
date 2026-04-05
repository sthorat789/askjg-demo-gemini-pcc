#
# Silero VAD with 4-state machine — no Pipecat dependency.
#
# Replicates Pipecat's SileroVADAnalyzer behavior:
# - Uses Silero ONNX model for voice confidence scoring
# - Implements QUIET → STARTING → SPEAKING → STOPPING state machine
# - Runs inference in a ThreadPoolExecutor (non-blocking)
# - Emits UserStartedSpeaking / UserStoppedSpeaking transitions
#
# Key parameters (matching the existing bot):
#   confidence=0.75, start_secs=0.2, stop_secs=0.2, min_volume=0.6
#

import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VAD state machine
# ---------------------------------------------------------------------------


class VADState(Enum):
    """Voice Activity Detection states."""

    QUIET = auto()  # No voice detected
    STARTING = auto()  # Voice detected, waiting for confirmation
    SPEAKING = auto()  # Confirmed speech
    STOPPING = auto()  # Silence detected, waiting for confirmation


class VADEvent(Enum):
    """Events emitted by the VAD state machine."""

    NONE = auto()
    SPEECH_STARTED = auto()
    SPEECH_STOPPED = auto()


@dataclass
class VADParams:
    """Parameters for Voice Activity Detection.

    Matches the existing bot configuration:
    - confidence=0.75 (speech threshold, stricter than default 0.7)
    - start_secs=0.2 (speech start confirmation delay)
    - stop_secs=0.2 (quick interruption detection, 4× faster than default 0.8)
    - min_volume=0.6 (minimum volume threshold)
    """

    confidence: float = 0.75
    start_secs: float = 0.2
    stop_secs: float = 0.2
    min_volume: float = 0.6


# ---------------------------------------------------------------------------
# Silero ONNX model wrapper
# ---------------------------------------------------------------------------


class _SileroModel:
    """Thin wrapper around the Silero VAD ONNX model.

    Supports 16 kHz and 8 kHz sample rates.
    Processes 512 samples (16 kHz) or 256 samples (8 kHz) at a time.
    Resets internal state every 5 seconds to prevent memory drift.
    """

    # Number of inference calls before state reset (~5 s at 16 kHz / 512 samples)
    _RESET_INTERVAL: int = 156  # (16000 / 512) * 5 ≈ 156

    def __init__(self, model_path: str):
        import onnxruntime  # Lazy import to keep startup fast

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = onnxruntime.InferenceSession(model_path, sess_options=opts)
        self._call_count = 0

        # Model internal state (h, c for LSTM)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset_state(self):
        """Reset LSTM state to prevent drift over long sessions."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._call_count = 0

    def __call__(self, audio_float32: np.ndarray, sample_rate: int) -> float:
        """Run inference and return voice confidence [0.0, 1.0].

        Args:
            audio_float32: Audio samples as float32, normalized to [-1, 1].
            sample_rate: Must be 16000 or 8000.

        Returns:
            Confidence score from 0.0 to 1.0.
        """
        self._call_count += 1
        if self._call_count >= self._RESET_INTERVAL:
            self.reset_state()

        # Prepare input tensors
        audio_tensor = audio_float32[np.newaxis, :]  # Shape: (1, N)
        sr_tensor = np.array([sample_rate], dtype=np.int64)

        inputs = {
            "input": audio_tensor,
            "sr": sr_tensor,
            "h": self._h,
            "c": self._c,
        }

        outputs = self._session.run(None, inputs)
        confidence = outputs[0].item()

        # Update internal state
        self._h = outputs[1]
        self._c = outputs[2]

        return confidence


# ---------------------------------------------------------------------------
# Main VAD analyzer
# ---------------------------------------------------------------------------


class SileroVAD:
    """Voice Activity Detection using Silero ONNX model.

    Implements the same 4-state machine as Pipecat's SileroVADAnalyzer:

        QUIET ──(voice detected)──► STARTING
        STARTING ──(confirmed 0.2s)──► SPEAKING  [emits SPEECH_STARTED]
        STARTING ──(voice drops)──► QUIET
        SPEAKING ──(silence detected)──► STOPPING
        STOPPING ──(confirmed 0.2s)──► QUIET  [emits SPEECH_STOPPED]
        STOPPING ──(voice resumes)──► SPEAKING

    Thread safety: inference runs in a ThreadPoolExecutor.
    """

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        params: Optional[VADParams] = None,
    ):
        """Initialize the VAD analyzer.

        Args:
            model_path: Path to silero_vad.onnx. If None, downloads automatically.
            sample_rate: Audio sample rate (8000 or 16000).
            params: VAD parameters. Defaults to project settings.
        """
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Sample rate must be 8000 or 16000, got {sample_rate}")

        self._sample_rate = sample_rate
        self._params = params or VADParams()

        # Resolve model path
        if model_path is None:
            model_path = self._get_default_model_path()
        self._model = _SileroModel(model_path)

        # State machine
        self._state = VADState.QUIET
        self._accumulator_frames = 0  # Frames accumulated in STARTING/STOPPING

        # How many consecutive frames needed for state confirmation
        # Each frame = num_frames_required samples at sample_rate
        frame_duration_secs = self.num_frames_required / self._sample_rate
        self._start_frames_needed = max(1, int(self._params.start_secs / frame_duration_secs))
        self._stop_frames_needed = max(1, int(self._params.stop_secs / frame_duration_secs))

        # Thread pool for non-blocking inference
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vad")

        logger.info(
            f"SileroVAD initialized: sample_rate={sample_rate}, "
            f"confidence={self._params.confidence}, "
            f"start_frames={self._start_frames_needed}, "
            f"stop_frames={self._stop_frames_needed}"
        )

    @property
    def num_frames_required(self) -> int:
        """Number of samples required per inference call."""
        return 512 if self._sample_rate == 16000 else 256

    @property
    def state(self) -> VADState:
        """Current VAD state."""
        return self._state

    @staticmethod
    def _get_default_model_path() -> str:
        """Get path to bundled silero_vad.onnx model.

        Downloads the model from the Silero GitHub repo if not present.
        """
        model_dir = Path(__file__).parent
        model_path = model_dir / "silero_vad.onnx"

        if not model_path.exists():
            import urllib.request

            url = (
                "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            )
            logger.info(f"Downloading Silero VAD model to {model_path}...")
            urllib.request.urlretrieve(url, str(model_path))
            logger.info("Silero VAD model downloaded successfully")

        return str(model_path)

    def _compute_volume(self, audio_int16: np.ndarray) -> float:
        """Compute normalized RMS volume [0.0, 1.0] from int16 audio."""
        if len(audio_int16) == 0:
            return 0.0
        rms = np.sqrt(np.mean(audio_int16.astype(np.float64) ** 2))
        # Normalize: int16 max = 32768
        return min(1.0, rms / 32768.0)

    def _run_inference(self, audio_bytes: bytes) -> tuple[float, float]:
        """Run Silero inference synchronously (called from thread pool).

        Returns:
            Tuple of (confidence, volume).
        """
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        volume = self._compute_volume(audio_int16)

        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        confidence = self._model(audio_float32, self._sample_rate)

        return confidence, volume

    async def analyze(self, audio_bytes: bytes) -> VADEvent:
        """Analyze an audio chunk and return any state transition event.

        This method:
        1. Runs Silero inference in a thread pool (non-blocking).
        2. Updates the 4-state machine.
        3. Returns SPEECH_STARTED, SPEECH_STOPPED, or NONE.

        Args:
            audio_bytes: Raw PCM int16 audio, exactly num_frames_required samples.

        Returns:
            VADEvent indicating any state transition.
        """
        loop = asyncio.get_running_loop()
        confidence, volume = await loop.run_in_executor(
            self._executor, self._run_inference, audio_bytes
        )

        # Determine if this frame is "speech"
        is_speech = confidence >= self._params.confidence and volume >= self._params.min_volume

        return self._update_state(is_speech)

    def _update_state(self, is_speech: bool) -> VADEvent:
        """Update the state machine and return any event.

        State transitions:
            QUIET + speech     → STARTING (accumulate)
            STARTING + speech  → SPEAKING (if enough frames) [emit SPEECH_STARTED]
            STARTING + silence → QUIET (reset)
            SPEAKING + silence → STOPPING (accumulate)
            STOPPING + silence → QUIET (if enough frames) [emit SPEECH_STOPPED]
            STOPPING + speech  → SPEAKING (cancel stop)
        """
        event = VADEvent.NONE

        if self._state == VADState.QUIET:
            if is_speech:
                self._state = VADState.STARTING
                self._accumulator_frames = 1

        elif self._state == VADState.STARTING:
            if is_speech:
                self._accumulator_frames += 1
                if self._accumulator_frames >= self._start_frames_needed:
                    self._state = VADState.SPEAKING
                    self._accumulator_frames = 0
                    event = VADEvent.SPEECH_STARTED
            else:
                # Voice dropped before confirmation
                self._state = VADState.QUIET
                self._accumulator_frames = 0

        elif self._state == VADState.SPEAKING:
            if not is_speech:
                self._state = VADState.STOPPING
                self._accumulator_frames = 1

        elif self._state == VADState.STOPPING:
            if not is_speech:
                self._accumulator_frames += 1
                if self._accumulator_frames >= self._stop_frames_needed:
                    self._state = VADState.QUIET
                    self._accumulator_frames = 0
                    event = VADEvent.SPEECH_STOPPED
            else:
                # Voice resumed before stop confirmation
                self._state = VADState.SPEAKING
                self._accumulator_frames = 0

        return event

    def reset(self):
        """Reset VAD state (e.g., on new conversation)."""
        self._state = VADState.QUIET
        self._accumulator_frames = 0
        self._model.reset_state()

    def close(self):
        """Shut down the thread pool."""
        self._executor.shutdown(wait=False)
