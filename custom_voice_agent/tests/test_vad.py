#
# Tests for the Silero VAD state machine.
#
# Tests the 4-state machine logic without requiring the ONNX model,
# using direct _update_state() calls to simulate VAD decisions.
#

import pytest

from custom_voice_agent.vad.silero_vad import (
    SileroVAD,
    VADEvent,
    VADParams,
    VADState,
)


class TestVADStateMachine:
    """Test the 4-state VAD state machine transitions.

    Uses a minimal VAD config with small frame counts for fast testing.
    """

    def _make_vad(self, start_frames: int = 3, stop_frames: int = 3) -> SileroVAD:
        """Create a SileroVAD with controllable frame thresholds.

        We directly set _start_frames_needed and _stop_frames_needed
        instead of going through the constructor's time-based calculation.
        """
        # Use params that won't matter since we call _update_state directly
        params = VADParams(confidence=0.75, start_secs=0.2, stop_secs=0.2)
        vad = SileroVAD.__new__(SileroVAD)
        vad._sample_rate = 16000
        vad._params = params
        vad._state = VADState.QUIET
        vad._accumulator_frames = 0
        vad._start_frames_needed = start_frames
        vad._stop_frames_needed = stop_frames
        return vad

    def test_initial_state_is_quiet(self):
        vad = self._make_vad()
        assert vad.state == VADState.QUIET

    # --- QUIET → STARTING → SPEAKING ---

    def test_quiet_to_starting_on_speech(self):
        vad = self._make_vad()
        event = vad._update_state(is_speech=True)
        assert vad.state == VADState.STARTING
        assert event == VADEvent.NONE  # Not confirmed yet

    def test_starting_to_speaking_after_enough_frames(self):
        vad = self._make_vad(start_frames=3)

        # Frame 1: QUIET → STARTING
        vad._update_state(is_speech=True)
        assert vad.state == VADState.STARTING

        # Frame 2: Still STARTING
        vad._update_state(is_speech=True)
        assert vad.state == VADState.STARTING

        # Frame 3: STARTING → SPEAKING (confirmed)
        event = vad._update_state(is_speech=True)
        assert vad.state == VADState.SPEAKING
        assert event == VADEvent.SPEECH_STARTED

    def test_starting_to_quiet_on_silence(self):
        """If voice drops during STARTING, go back to QUIET."""
        vad = self._make_vad(start_frames=3)

        vad._update_state(is_speech=True)  # QUIET → STARTING
        assert vad.state == VADState.STARTING

        event = vad._update_state(is_speech=False)  # STARTING → QUIET
        assert vad.state == VADState.QUIET
        assert event == VADEvent.NONE

    # --- SPEAKING → STOPPING → QUIET ---

    def test_speaking_to_stopping_on_silence(self):
        vad = self._make_vad(start_frames=1, stop_frames=3)

        # Get to SPEAKING state (needs 2 frames: QUIET→STARTING, then STARTING→SPEAKING)
        vad._update_state(is_speech=True)  # QUIET → STARTING
        vad._update_state(is_speech=True)  # STARTING → SPEAKING
        assert vad.state == VADState.SPEAKING

        # Silence detected: SPEAKING → STOPPING
        event = vad._update_state(is_speech=False)
        assert vad.state == VADState.STOPPING
        assert event == VADEvent.NONE

    def test_stopping_to_quiet_after_enough_frames(self):
        vad = self._make_vad(start_frames=1, stop_frames=3)

        # Get to SPEAKING (2 frames)
        vad._update_state(is_speech=True)  # → STARTING
        vad._update_state(is_speech=True)  # → SPEAKING
        assert vad.state == VADState.SPEAKING

        # Start stopping
        vad._update_state(is_speech=False)  # Frame 1: SPEAKING → STOPPING
        vad._update_state(is_speech=False)  # Frame 2
        event = vad._update_state(is_speech=False)  # Frame 3: STOPPING → QUIET
        assert vad.state == VADState.QUIET
        assert event == VADEvent.SPEECH_STOPPED

    def test_stopping_to_speaking_on_voice_resume(self):
        """If voice resumes during STOPPING, go back to SPEAKING."""
        vad = self._make_vad(start_frames=1, stop_frames=3)

        # Get to SPEAKING → STOPPING (2 frames to speak, 1 to start stopping)
        vad._update_state(is_speech=True)   # → STARTING
        vad._update_state(is_speech=True)   # → SPEAKING
        assert vad.state == VADState.SPEAKING
        vad._update_state(is_speech=False)  # → STOPPING
        assert vad.state == VADState.STOPPING

        # Voice resumes: STOPPING → SPEAKING
        event = vad._update_state(is_speech=True)
        assert vad.state == VADState.SPEAKING
        assert event == VADEvent.NONE  # No new event, was already speaking

    # --- Full cycle ---

    def test_full_speech_cycle(self):
        """Test a complete speech → silence cycle."""
        vad = self._make_vad(start_frames=2, stop_frames=2)

        # User starts speaking
        vad._update_state(is_speech=True)   # QUIET → STARTING
        event = vad._update_state(is_speech=True)   # STARTING → SPEAKING
        assert event == VADEvent.SPEECH_STARTED

        # User speaks for a while (no events)
        for _ in range(5):
            event = vad._update_state(is_speech=True)
            assert event == VADEvent.NONE
            assert vad.state == VADState.SPEAKING

        # User stops speaking
        vad._update_state(is_speech=False)  # SPEAKING → STOPPING
        event = vad._update_state(is_speech=False)  # STOPPING → QUIET
        assert event == VADEvent.SPEECH_STOPPED
        assert vad.state == VADState.QUIET

    def test_intermittent_speech(self):
        """Test speech with brief pauses (shouldn't trigger stop)."""
        vad = self._make_vad(start_frames=1, stop_frames=3)

        # Start speaking (2 frames to reach SPEAKING: QUIET→STARTING→SPEAKING)
        vad._update_state(is_speech=True)   # QUIET → STARTING
        event = vad._update_state(is_speech=True)   # STARTING → SPEAKING
        assert event == VADEvent.SPEECH_STARTED

        # Brief pause (1 frame of silence — not enough to stop)
        vad._update_state(is_speech=False)
        assert vad.state == VADState.STOPPING

        # Resume before stop_frames threshold
        vad._update_state(is_speech=True)
        assert vad.state == VADState.SPEAKING

        # This shows the stop was cancelled — no SPEECH_STOPPED emitted

    def test_quiet_silence_no_events(self):
        """Continuous silence in QUIET state produces no events."""
        vad = self._make_vad()

        for _ in range(10):
            event = vad._update_state(is_speech=False)
            assert event == VADEvent.NONE
            assert vad.state == VADState.QUIET


class TestVADParams:
    """Test VAD parameter defaults match the existing bot."""

    def test_default_params(self):
        params = VADParams()
        assert params.confidence == 0.75
        assert params.start_secs == 0.2
        assert params.stop_secs == 0.2
        assert params.min_volume == 0.6
