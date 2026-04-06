#
# Tests for the LiveKit audio output with barge-in support.
#
# Uses a mock LiveKit room to test the critical interruption behavior.
#

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from custom_voice_agent.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame
from custom_voice_agent.transport.livekit_output import LiveKitOutput


class MockAudioSource:
    """Mock LiveKit AudioSource for testing."""

    def __init__(self, sample_rate, num_channels):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.frames_captured = []
        self.capture_frame = AsyncMock(side_effect=self._capture)

    async def _capture(self, frame):
        self.frames_captured.append(frame)


class MockRoom:
    """Mock LiveKit Room for testing."""

    def __init__(self):
        self.local_participant = MagicMock()
        self.local_participant.publish_track = AsyncMock()
        self.local_participant.unpublish_track = AsyncMock()


@pytest.fixture
def mock_room():
    return MockRoom()


class TestLiveKitOutputBargein:
    """Test the critical barge-in (cancel_and_clear) behavior."""

    @pytest.mark.asyncio
    async def test_cancel_and_clear_drains_queue(self, mock_room):
        """Barge-in should drain all queued audio."""
        with patch("custom_voice_agent.transport.livekit_output.rtc") as mock_rtc:
            mock_source = MockAudioSource(48000, 1)
            mock_rtc.AudioSource.return_value = mock_source
            mock_rtc.LocalAudioTrack.create_audio_track.return_value = MagicMock(sid="track1")
            mock_rtc.TrackPublishOptions.return_value = MagicMock()
            mock_rtc.TrackSource.SOURCE_MICROPHONE = 0

            output = LiveKitOutput(mock_room, sample_rate=24000, chunk_ms=40)
            await output.start()

            # Queue some audio
            audio = b"\x00" * (24000 * 2)  # 1 second of audio
            await output.queue_audio(audio)

            # Verify queue has items
            assert not output._queue.empty()

            # Barge-in!
            await output.cancel_and_clear()

            # Queue should be empty
            assert output._queue.empty()

            await output.stop()

    @pytest.mark.asyncio
    async def test_cancel_emits_bot_stopped_speaking(self, mock_room):
        """Barge-in should emit BotStoppedSpeakingFrame if bot was speaking."""
        with patch("custom_voice_agent.transport.livekit_output.rtc") as mock_rtc:
            mock_source = MockAudioSource(48000, 1)
            mock_rtc.AudioSource.return_value = mock_source
            mock_rtc.LocalAudioTrack.create_audio_track.return_value = MagicMock(sid="track1")
            mock_rtc.TrackPublishOptions.return_value = MagicMock()
            mock_rtc.TrackSource.SOURCE_MICROPHONE = 0

            output = LiveKitOutput(mock_room, sample_rate=24000, chunk_ms=40)

            events = []
            output.add_event_callback(lambda f: events.append(type(f).__name__) or asyncio.sleep(0))

            await output.start()

            # Simulate bot speaking
            output._bot_speaking = True

            # Barge-in
            await output.cancel_and_clear()

            # Should have emitted BotStoppedSpeakingFrame
            assert "BotStoppedSpeakingFrame" in events
            assert not output.is_speaking

            await output.stop()

    @pytest.mark.asyncio
    async def test_audio_chunking(self, mock_room):
        """Audio should be split into chunk_ms-sized pieces."""
        with patch("custom_voice_agent.transport.livekit_output.rtc") as mock_rtc:
            mock_source = MockAudioSource(48000, 1)
            mock_rtc.AudioSource.return_value = mock_source
            mock_rtc.LocalAudioTrack.create_audio_track.return_value = MagicMock(sid="track1")
            mock_rtc.TrackPublishOptions.return_value = MagicMock()
            mock_rtc.TrackSource.SOURCE_MICROPHONE = 0

            chunk_ms = 40
            sample_rate = 24000
            output = LiveKitOutput(mock_room, sample_rate=sample_rate, chunk_ms=chunk_ms)
            await output.start()

            # 100ms of audio = 2.5 chunks of 40ms
            samples = int(sample_rate * 0.1)  # 2400 samples
            audio = b"\x00" * (samples * 2)  # 16-bit
            await output.queue_audio(audio)

            # Expected chunks: ceil(4800 bytes / (960 * 2 bytes)) = 3 chunks
            expected_chunk_bytes = int(sample_rate * chunk_ms / 1000) * 2
            expected_chunks = -(-len(audio) // expected_chunk_bytes)  # ceil division

            # Count items in queue
            queue_items = 0
            while not output._queue.empty():
                output._queue.get_nowait()
                queue_items += 1

            assert queue_items == expected_chunks

            await output.stop()

    @pytest.mark.asyncio
    async def test_signal_end_of_response(self, mock_room):
        """signal_end_of_response should put None (poison pill) in queue."""
        with patch("custom_voice_agent.transport.livekit_output.rtc") as mock_rtc:
            mock_source = MockAudioSource(48000, 1)
            mock_rtc.AudioSource.return_value = mock_source
            mock_rtc.LocalAudioTrack.create_audio_track.return_value = MagicMock(sid="track1")
            mock_rtc.TrackPublishOptions.return_value = MagicMock()
            mock_rtc.TrackSource.SOURCE_MICROPHONE = 0

            output = LiveKitOutput(mock_room, sample_rate=24000)
            await output.start()

            await output.signal_end_of_response()

            item = output._queue.get_nowait()
            assert item is None

            await output.stop()

    @pytest.mark.asyncio
    async def test_queue_audio_drops_oldest_chunks_when_full(self, mock_room):
        """Bounded output queues should drop old audio instead of growing forever."""
        with patch("custom_voice_agent.transport.livekit_output.rtc") as mock_rtc:
            mock_source = MockAudioSource(48000, 1)
            mock_rtc.AudioSource.return_value = mock_source
            mock_rtc.LocalAudioTrack.create_audio_track.return_value = MagicMock(sid="track1")
            mock_rtc.TrackPublishOptions.return_value = MagicMock()
            mock_rtc.TrackSource.SOURCE_MICROPHONE = 0

            output = LiveKitOutput(
                mock_room,
                sample_rate=24000,
                chunk_ms=40,
                max_queue_chunks=2,
            )
            await output.start()

            audio = b"\x00" * int(24000 * 0.2 * 2)  # 200 ms => 5 chunks at 40 ms
            await output.queue_audio(audio)

            assert output._queue.qsize() == 2
            assert output.health_snapshot()["dropped_chunks"] == 3

            await output.stop()
