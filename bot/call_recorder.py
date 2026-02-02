"""Chunked call recording with GCS upload for Pipecat voice bots.

This module provides memory-efficient call recording that uploads audio
chunks periodically during the call instead of buffering the entire
conversation in memory.
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

from loguru import logger

from pipecat.audio.utils import interleave_stereo_audio
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.transports.base_transport import BaseTransport


# Chunking configuration
SAMPLE_RATE = 24000  # Hz - matches typical Pipecat audio
CHUNK_DURATION_SECS = 30  # Upload every 30 seconds
BUFFER_SIZE = SAMPLE_RATE * 2 * CHUNK_DURATION_SECS  # ~1.44 MB per chunk


class RecordingMode(str, Enum):
    """Recording mode configuration."""

    DISABLED = "disabled"
    MONO = "mono"  # Mixed audio (user + bot in single channel)
    STEREO = "stereo"  # Separate channels (user=left, bot=right)
    BOTH = "both"  # Alias for STEREO (stereo file with separate channels)


def create_wav_header(total_audio_bytes: int, sample_rate: int, num_channels: int) -> bytes:
    """Create a 44-byte WAV header for raw PCM data."""
    import struct

    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    header = bytearray()
    header.extend(b"RIFF")
    header.extend(struct.pack("<I", 36 + total_audio_bytes))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend(struct.pack("<I", 16))
    header.extend(struct.pack("<H", 1))
    header.extend(struct.pack("<H", num_channels))
    header.extend(struct.pack("<I", sample_rate))
    header.extend(struct.pack("<I", byte_rate))
    header.extend(struct.pack("<H", block_align))
    header.extend(struct.pack("<H", bits_per_sample))
    header.extend(b"data")
    header.extend(struct.pack("<I", total_audio_bytes))

    return bytes(header)


class GCSUploader:
    """Async uploader for Google Cloud Storage."""

    def __init__(self, bucket_name: str):
        self._bucket_name = bucket_name
        self._storage = None
        self._session = None

    async def _ensure_client(self):
        """Lazily initialize GCS client."""
        if self._storage is None:
            import aiohttp
            from gcloud.aio.storage import Storage

            self._session = aiohttp.ClientSession()
            from core.gcs import get_gcs_credentials

            self._storage = Storage(
                service_file=get_gcs_credentials(),
                session=self._session,
            )

    async def upload_audio(self, blob_name: str, audio_data: bytes, max_retries: int = 2) -> bool:
        """Upload audio to GCS with retry logic."""
        await self._ensure_client()

        for attempt in range(max_retries):
            try:
                await self._storage.upload(
                    self._bucket_name,
                    blob_name,
                    audio_data,
                    content_type="audio/wav",
                    timeout=60,
                )
                logger.info(f"Uploaded: gs://{self._bucket_name}/{blob_name}")
                return True
            except Exception as e:
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

        logger.error(f"Failed to upload {blob_name} after {max_retries} attempts")
        return False

    async def compose(self, dest_name: str, source_names: list[str], max_retries: int = 2) -> bool:
        """Compose multiple objects into one."""
        await self._ensure_client()

        for attempt in range(max_retries):
            try:
                await self._storage.compose(
                    self._bucket_name,
                    dest_name,
                    source_names,
                    content_type="audio/wav",
                    timeout=120,
                )
                logger.debug(f"Composed {len(source_names)} objects -> {dest_name}")
                return True
            except Exception as e:
                logger.warning(f"Compose attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

        logger.error(f"Failed to compose {dest_name} after {max_retries} attempts")
        return False

    async def delete(self, blob_name: str, max_retries: int = 2) -> bool:
        """Delete an object."""
        await self._ensure_client()

        for attempt in range(max_retries):
            try:
                await self._storage.delete(
                    self._bucket_name,
                    blob_name,
                    timeout=30,
                )
                logger.debug(f"Deleted {blob_name}")
                return True
            except Exception as e:
                logger.warning(f"Delete attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

        logger.error(f"Failed to delete {blob_name} after {max_retries} attempts")
        return False

    async def close(self):
        """Close the GCS client and session."""
        if self._storage:
            await self._storage.close()
        if self._session:
            await self._session.close()


class CallRecorder:
    """Records call audio in chunks and uploads to GCS periodically."""

    def __init__(
        self,
        mode: RecordingMode,
        session_id: str,
        bucket_name: str,
    ):
        self._mode = mode
        self._session_id = session_id
        # Flat file naming: YYYYMMDDHHmmss format for chronological sorting
        self._timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        self._uploader = GCSUploader(bucket_name) if mode != RecordingMode.DISABLED else None

        self._chunk_counter = 0
        self._chunk_files: list[str] = []
        self._lock = asyncio.Lock()
        self._total_audio_bytes = 0
        self._final_recording: Optional[str] = None
        self._sample_rate: int = SAMPLE_RATE
        self._is_running: bool = False

        num_channels = 2 if mode in (RecordingMode.STEREO, RecordingMode.BOTH) else 1
        self._processor = AudioBufferProcessor(
            sample_rate=SAMPLE_RATE,
            num_channels=num_channels,
            buffer_size=BUFFER_SIZE,
        )

        if mode == RecordingMode.MONO:
            self._processor.add_event_handler("on_audio_data", self._on_chunk_mono)
        elif mode in (RecordingMode.STEREO, RecordingMode.BOTH):
            self._processor.add_event_handler("on_track_audio_data", self._on_chunk_tracks)

    @property
    def processor(self) -> AudioBufferProcessor:
        """Get the AudioBufferProcessor for pipeline integration."""
        return self._processor

    @property
    def is_running(self) -> bool:
        """Check if recording is currently active."""
        return self._is_running

    @property
    def recording_urls(self) -> dict[str, Any]:
        """Get recording info after upload completes."""
        if not self._uploader:
            return {}

        bucket = self._uploader._bucket_name

        def gcs_url(blob_name: str) -> str:
            return f"https://storage.googleapis.com/{bucket}/{blob_name}"

        if self._final_recording:
            url = gcs_url(self._final_recording)
            if "stereo" in self._final_recording:
                return {"stereo": url}
            else:
                return {"mono": url}

        if not self._chunk_files:
            return {}

        # Sort chunks by chunk number for consistent ordering
        sorted_chunks = sorted(self._chunk_files, key=self._get_chunk_num)
        return {
            "chunks": [gcs_url(f) for f in sorted_chunks],
            "chunk_count": len(sorted_chunks),
        }

    def _get_chunk_name(self, chunk_num: int, suffix: str) -> str:
        # Flat naming: {timestamp}_{session_id}_chunk_{NNN}_{suffix}.pcm
        return f"recordings/{self._timestamp}_{self._session_id}_chunk_{chunk_num:03d}_{suffix}.pcm"

    def _get_header_name(self, suffix: str) -> str:
        return f"recordings/{self._timestamp}_{self._session_id}_header_{suffix}.bin"

    def _get_intermediate_name(self, level: int, index: int) -> str:
        return f"recordings/{self._timestamp}_{self._session_id}_intermediate_L{level}_{index:03d}.wav"

    def _get_final_recording_name(self) -> str:
        suffix = "stereo" if self._mode in (RecordingMode.STEREO, RecordingMode.BOTH) else "mono"
        return f"recordings/{self._timestamp}_{self._session_id}_{suffix}.wav"

    def _get_chunk_num(self, filename: str) -> int:
        """Extract chunk number from filename for sorting."""
        # Filename format: recordings/{timestamp}_{session_id}_chunk_{NNN}_{suffix}.pcm
        parts = filename.split("_chunk_")
        if len(parts) == 2:
            num_part = parts[1].split("_")[0]  # Get "001" from "001_stereo.pcm"
            try:
                return int(num_part)
            except ValueError:
                return 0
        return 0

    async def _compose_chunks(self) -> Optional[str]:
        """Compose all PCM chunk files into a single WAV recording."""
        if len(self._chunk_files) == 0:
            return None

        # Sort chunks by chunk number to ensure correct audio order
        # (chunks may have been appended out of order due to variable upload times)
        sorted_chunks = sorted(self._chunk_files, key=self._get_chunk_num)

        num_channels = 2 if self._mode in (RecordingMode.STEREO, RecordingMode.BOTH) else 1
        suffix = "stereo" if num_channels == 2 else "mono"

        header_data = create_wav_header(self._total_audio_bytes, self._sample_rate, num_channels)
        header_blob = self._get_header_name(suffix)
        header_ok = await self._uploader.upload_audio(header_blob, header_data)
        if not header_ok:
            logger.error("Failed to upload WAV header")
            return None

        final_name = self._get_final_recording_name()

        if len(sorted_chunks) <= 31:
            all_blobs = [header_blob] + sorted_chunks
            success = await self._uploader.compose(final_name, all_blobs)
            if not success:
                logger.error("Compose failed")
                await self._uploader.delete(header_blob)
                return None

            for blob in all_blobs:
                await self._uploader.delete(blob)

            logger.info(f"Composed {len(sorted_chunks)} chunks into {final_name}")
            return final_name

        # Tree-like merging for >31 chunks
        # Track intermediate files for cleanup on failure
        intermediate_files: list[str] = []
        current_level = sorted_chunks.copy()
        level = 0

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 32):
                batch = current_level[i : i + 32]
                if len(batch) == 1:
                    next_level.append(batch[0])
                else:
                    dest_name = self._get_intermediate_name(level, len(next_level))
                    success = await self._uploader.compose(dest_name, batch)
                    if not success:
                        logger.error(f"Compose failed at level {level}")
                        # Clean up header and any intermediate files we created
                        await self._uploader.delete(header_blob)
                        for intermediate in intermediate_files:
                            await self._uploader.delete(intermediate)
                        return None
                    intermediate_files.append(dest_name)
                    next_level.append(dest_name)
                    for blob in batch:
                        await self._uploader.delete(blob)
            current_level = next_level
            level += 1

        merged_pcm = current_level[0]
        success = await self._uploader.compose(final_name, [header_blob, merged_pcm])
        if not success:
            logger.error("Final compose (header + PCM) failed")
            # Clean up header and the final merged PCM
            await self._uploader.delete(header_blob)
            await self._uploader.delete(merged_pcm)
            return None

        await self._uploader.delete(header_blob)
        await self._uploader.delete(merged_pcm)

        logger.info(f"Composed {len(sorted_chunks)} chunks into {final_name}")
        return final_name

    async def start(self):
        """Start recording."""
        if self._mode == RecordingMode.DISABLED:
            return
        logger.info(f"Starting chunked recording (mode={self._mode.value}, chunk={CHUNK_DURATION_SECS}s)")
        self._is_running = True
        await self._processor.start_recording()

    async def stop(self):
        """Stop recording and compose chunks into single file."""
        if self._mode == RecordingMode.DISABLED:
            return
        if not self._is_running:
            logger.debug("Recording already stopped, skipping")
            return

        self._is_running = False

        # Trigger final chunk callback (stop_recording flushes the buffer)
        await self._processor.stop_recording()

        # Wait for all event handler tasks to complete (including uploads)
        # cleanup() waits for all async tasks spawned by event handlers
        await self._processor.cleanup()

        logger.info(f"Stopping recording ({len(self._chunk_files)} chunks uploaded)")

        if self._chunk_files and self._uploader:
            final_blob = await self._compose_chunks()
            if final_blob:
                self._final_recording = final_blob
            else:
                logger.warning("Composition failed, falling back to chunk URLs")

        logger.info(f"Recording complete: {self._final_recording or f'{len(self._chunk_files)} chunks'}")

        if self._uploader:
            await self._uploader.close()

    async def _on_chunk_mono(self, _, audio: bytes, sample_rate: int, num_channels: int):
        """Handle mono chunk."""
        if len(audio) == 0:
            return

        async with self._lock:
            self._chunk_counter += 1
            chunk_num = self._chunk_counter

        # Note: Setting _sample_rate outside lock is technically a race condition,
        # but harmless since sample rate is constant for the duration of a call.
        self._sample_rate = sample_rate
        blob_name = self._get_chunk_name(chunk_num, "mono")

        success = await self._uploader.upload_audio(blob_name, audio)
        if success:
            async with self._lock:
                self._chunk_files.append(blob_name)
                self._total_audio_bytes += len(audio)

    async def _on_chunk_tracks(
        self, _, user_audio: bytes, bot_audio: bytes, sample_rate: int, num_channels: int
    ):
        """Handle stereo chunk with separate user/bot tracks."""
        if len(user_audio) == 0 and len(bot_audio) == 0:
            return

        async with self._lock:
            self._chunk_counter += 1
            chunk_num = self._chunk_counter

        # Note: Setting _sample_rate outside lock is technically a race condition,
        # but harmless since sample rate is constant for the duration of a call.
        self._sample_rate = sample_rate
        stereo_audio = interleave_stereo_audio(user_audio, bot_audio)
        blob_name = self._get_chunk_name(chunk_num, "stereo")

        success = await self._uploader.upload_audio(blob_name, stereo_audio)
        if success:
            async with self._lock:
                self._chunk_files.append(blob_name)
                self._total_audio_bytes += len(stereo_audio)


def get_session_id_from_transport(transport: BaseTransport) -> str:
    """Extract session ID from Daily transport or generate unique ID."""
    if hasattr(transport, "_room_url") and transport._room_url:
        return urlparse(transport._room_url).path.strip("/")
    return f"local-{uuid.uuid4().hex[:12]}"
