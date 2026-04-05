#
# Tests for the frame system.
#

import asyncio

import pytest

from custom_voice_agent.frames import (
    AudioFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    DataFrame,
    EndFrame,
    EndedReason,
    FramePriority,
    InputAudioFrame,
    InterruptionFrame,
    OutputAudioFrame,
    StartFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)


class TestFramePriority:
    """Test that system frames always sort before data frames."""

    def test_system_frame_has_system_priority(self):
        frame = InterruptionFrame()
        assert frame.priority == FramePriority.SYSTEM

    def test_data_frame_has_data_priority(self):
        frame = AudioFrame(audio=b"\x00" * 100)
        assert frame.priority == FramePriority.DATA

    def test_system_before_data_in_sort(self):
        data = AudioFrame(audio=b"\x00" * 100)
        system = InterruptionFrame()
        # System frame should sort before data frame
        assert system < data

    def test_priority_queue_ordering(self):
        """System frames should be dequeued before data frames."""
        q = asyncio.PriorityQueue()

        # Add frames in wrong order: data first, then system
        data1 = AudioFrame(audio=b"\x00" * 100)
        data2 = OutputAudioFrame(audio=b"\x00" * 200)
        interrupt = InterruptionFrame()
        cancel = CancelFrame()

        # Put data frames first
        q.put_nowait(data1)
        q.put_nowait(data2)
        # Then system frames
        q.put_nowait(interrupt)
        q.put_nowait(cancel)

        # System frames should come out first
        first = q.get_nowait()
        second = q.get_nowait()
        assert isinstance(first, SystemFrame)
        assert isinstance(second, SystemFrame)

        third = q.get_nowait()
        fourth = q.get_nowait()
        assert isinstance(third, DataFrame)
        assert isinstance(fourth, DataFrame)


class TestAudioFrame:
    """Test audio frame properties."""

    def test_num_samples_mono_16bit(self):
        # 100 bytes of 16-bit mono audio = 50 samples
        frame = AudioFrame(audio=b"\x00" * 100, sample_rate=16000, num_channels=1)
        assert frame.num_samples == 50

    def test_num_samples_empty(self):
        frame = AudioFrame(audio=b"", sample_rate=16000, num_channels=1)
        assert frame.num_samples == 0

    def test_duration_ms(self):
        # 16000 samples/sec, 1600 samples = 100ms
        audio = b"\x00" * (1600 * 2)  # 1600 samples × 2 bytes
        frame = AudioFrame(audio=audio, sample_rate=16000, num_channels=1)
        assert abs(frame.duration_ms - 100.0) < 0.01


class TestFrameTypes:
    """Test that all frame types instantiate correctly."""

    def test_interruption_frame_is_system(self):
        f = InterruptionFrame()
        assert f.priority == FramePriority.SYSTEM

    def test_start_frame_is_system(self):
        f = StartFrame()
        assert f.priority == FramePriority.SYSTEM

    def test_end_frame_has_reason(self):
        f = EndFrame(reason="test-reason")
        assert f.reason == "test-reason"
        assert f.priority == FramePriority.SYSTEM

    def test_cancel_frame_has_reason(self):
        f = CancelFrame(reason="timeout")
        assert f.reason == "timeout"

    def test_user_speaking_frames_are_system(self):
        assert UserStartedSpeakingFrame().priority == FramePriority.SYSTEM
        assert UserStoppedSpeakingFrame().priority == FramePriority.SYSTEM

    def test_bot_speaking_frames_are_system(self):
        assert BotStartedSpeakingFrame().priority == FramePriority.SYSTEM
        assert BotStoppedSpeakingFrame().priority == FramePriority.SYSTEM

    def test_text_frame(self):
        f = TextFrame(text="Hello", role="assistant")
        assert f.text == "Hello"
        assert f.role == "assistant"
        assert f.priority == FramePriority.DATA

    def test_transcription_frame(self):
        f = TranscriptionFrame(text="Hi there", role="user", is_final=True)
        assert f.text == "Hi there"
        assert f.is_final is True


class TestEndedReason:
    """Test EndedReason constants."""

    def test_reason_strings(self):
        assert EndedReason.CUSTOMER_ENDED_CALL == "customer-ended-call"
        assert EndedReason.ASSISTANT_ENDED_CALL == "assistant-ended-call"
        assert EndedReason.EXCEEDED_MAX_DURATION == "exceeded-max-duration"
        assert EndedReason.SILENCE_TIMED_OUT == "silence-timed-out"
        assert EndedReason.CONNECTION_TIMED_OUT == "connection-timed-out"
        assert EndedReason.PIPELINE_ERROR == "pipeline-error"
