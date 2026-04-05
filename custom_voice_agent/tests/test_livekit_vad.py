#
# Tests for the SileroVADAnalyzer in livekit_ui_integration.py.
#
# Tests the process_chunk logic (volume gating, confidence+volume speech
# decision, start/stop frame thresholds, and callback firing) without
# requiring the ONNX model or external dependencies (google, livekit),
# by mocking unavailable modules before import.
#

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock external dependencies that aren't installed in the test environment
_mock_modules = [
    "google", "google.genai", "google.genai.types",
    "livekit", "livekit.api", "livekit.rtc",
]
_saved = {}
for mod in _mock_modules:
    if mod not in sys.modules:
        _saved[mod] = None
        sys.modules[mod] = MagicMock()

from custom_voice_agent.livekit_ui_integration import (  # noqa: E402
    BYTES_PER_CHUNK,
    CHUNK_SIZE,
    RATE_16K,
    VAD_CONFIDENCE_THRESHOLD,
    VAD_MIN_VOLUME,
    SileroVADAnalyzer,
    VADInterruptionCallback,
)


def _make_analyzer(model_confidence: float = 0.9) -> SileroVADAnalyzer:
    """Create a SileroVADAnalyzer with a fake model and callback.

    The fake model always returns `model_confidence`.
    The callback's on_user_speech_started is a MagicMock.
    """
    callback = MagicMock(spec=VADInterruptionCallback)
    analyzer = SileroVADAnalyzer.__new__(SileroVADAnalyzer)

    # Fake model that returns a fixed confidence
    analyzer.model = MagicMock(return_value=model_confidence)
    analyzer.callback = callback
    analyzer.is_speaking = False
    analyzer.speaking_frames = 0
    analyzer.silent_frames = 0

    frames_per_sec = RATE_16K / CHUNK_SIZE
    analyzer.start_frames_threshold = int(0.2 * frames_per_sec)  # 6 frames
    analyzer.stop_frames_threshold = int(0.2 * frames_per_sec)  # 6 frames

    return analyzer


def _make_loud_chunk() -> bytes:
    """Create a loud audio chunk with RMS volume above VAD_MIN_VOLUME (0.6)."""
    samples = np.full(CHUNK_SIZE, 30000, dtype=np.int16)
    return samples.tobytes()


def _make_very_loud_chunk() -> bytes:
    """Create a very loud chunk with RMS volume well above VAD_MIN_VOLUME."""
    samples = np.full(CHUNK_SIZE, 32000, dtype=np.int16)
    return samples.tobytes()


def _make_quiet_chunk() -> bytes:
    """Create a nearly-silent audio chunk (very low volume)."""
    samples = np.full(CHUNK_SIZE, 10, dtype=np.int16)
    return samples.tobytes()


def _make_medium_chunk() -> bytes:
    """Create a medium-volume chunk (above silence skip, below VAD_MIN_VOLUME)."""
    # RMS will be ~5000/32768 ≈ 0.15 → above 0.1*0.6=0.06 but below 0.6
    samples = np.full(CHUNK_SIZE, 5000, dtype=np.int16)
    return samples.tobytes()


class TestSileroVADAnalyzerVolumeGating:
    """Test that volume is computed as RMS and gates speech decisions."""

    def test_very_quiet_chunk_skips_inference(self):
        """Volume below 10% of VAD_MIN_VOLUME should skip model inference."""
        analyzer = _make_analyzer(model_confidence=0.99)
        chunk = _make_quiet_chunk()

        result = analyzer.process_chunk(chunk)

        assert result is False
        # Model should NOT be called for very quiet frames
        analyzer.model.assert_not_called()

    def test_medium_volume_runs_inference_but_not_speech(self):
        """Volume above skip threshold but below VAD_MIN_VOLUME should run
        inference but not count as speech even with high confidence."""
        analyzer = _make_analyzer(model_confidence=0.99)
        chunk = _make_medium_chunk()

        # Process enough frames to exceed start_frames_threshold
        for _ in range(10):
            result = analyzer.process_chunk(chunk)

        assert result is False
        # Model should be called (volume above skip threshold)
        assert analyzer.model.call_count == 10

    def test_loud_chunk_with_high_confidence_is_speech(self):
        """Loud volume + high confidence should count as speech."""
        analyzer = _make_analyzer(model_confidence=0.9)
        chunk = _make_very_loud_chunk()

        # Need start_frames_threshold consecutive speech frames
        for _ in range(analyzer.start_frames_threshold):
            result = analyzer.process_chunk(chunk)

        assert result is True


class TestSileroVADAnalyzerSpeechTransitions:
    """Test start/stop frame threshold logic and callback firing."""

    def test_speech_start_fires_callback(self):
        """Callback.on_user_speech_started should fire on QUIET→SPEAKING transition."""
        analyzer = _make_analyzer(model_confidence=0.9)
        chunk = _make_very_loud_chunk()

        for _ in range(analyzer.start_frames_threshold):
            analyzer.process_chunk(chunk)

        analyzer.callback.on_user_speech_started.assert_called_once()

    def test_speech_start_requires_enough_frames(self):
        """Callback should NOT fire before start_frames_threshold is reached."""
        analyzer = _make_analyzer(model_confidence=0.9)
        chunk = _make_very_loud_chunk()

        for _ in range(analyzer.start_frames_threshold - 1):
            analyzer.process_chunk(chunk)

        analyzer.callback.on_user_speech_started.assert_not_called()
        assert analyzer.is_speaking is False

    def test_speech_stop_after_enough_silent_frames(self):
        """Speaking should stop after stop_frames_threshold silent frames."""
        analyzer = _make_analyzer(model_confidence=0.9)
        loud_chunk = _make_very_loud_chunk()

        # Start speaking
        for _ in range(analyzer.start_frames_threshold):
            analyzer.process_chunk(loud_chunk)
        assert analyzer.is_speaking is True

        # Switch to silence (low confidence)
        analyzer.model.return_value = 0.1
        quiet_chunk = _make_very_loud_chunk()  # Volume still loud, but confidence low

        for _ in range(analyzer.stop_frames_threshold):
            analyzer.process_chunk(quiet_chunk)

        assert analyzer.is_speaking is False

    def test_brief_silence_does_not_stop_speech(self):
        """Brief silence (fewer than stop_frames_threshold) should not stop speech."""
        analyzer = _make_analyzer(model_confidence=0.9)
        loud_chunk = _make_very_loud_chunk()

        # Start speaking
        for _ in range(analyzer.start_frames_threshold):
            analyzer.process_chunk(loud_chunk)
        assert analyzer.is_speaking is True

        # Brief silence (1 frame less than threshold)
        analyzer.model.return_value = 0.1
        for _ in range(analyzer.stop_frames_threshold - 1):
            analyzer.process_chunk(loud_chunk)

        # Resume speech
        analyzer.model.return_value = 0.9
        analyzer.process_chunk(loud_chunk)

        assert analyzer.is_speaking is True

    def test_callback_fires_only_on_transition(self):
        """Callback should fire only once when transitioning to speaking,
        not on every subsequent speech frame."""
        analyzer = _make_analyzer(model_confidence=0.9)
        chunk = _make_very_loud_chunk()

        # Start speaking + continue for several more frames
        for _ in range(analyzer.start_frames_threshold + 5):
            analyzer.process_chunk(chunk)

        # Callback should only fire once
        analyzer.callback.on_user_speech_started.assert_called_once()


class TestSileroVADAnalyzerRMSVolume:
    """Test that RMS volume computation matches the existing SileroVAD."""

    def test_rms_volume_for_constant_signal(self):
        """For a constant int16 signal, RMS = abs(value)/32768."""
        value = 20000
        samples = np.full(CHUNK_SIZE, value, dtype=np.int16)
        audio_float32 = samples.astype(np.float32) / 32768.0
        expected_rms = float(np.sqrt(np.mean(np.square(audio_float32))))

        # Should be approximately value/32768
        assert abs(expected_rms - value / 32768.0) < 0.001

    def test_rms_volume_for_silence(self):
        """Silence should have zero RMS."""
        samples = np.zeros(CHUNK_SIZE, dtype=np.int16)
        audio_float32 = samples.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(audio_float32))))
        assert rms == 0.0
