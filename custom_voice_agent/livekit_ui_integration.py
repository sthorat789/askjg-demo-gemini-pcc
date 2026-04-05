#
# ADK LiveKit UI Integration — Alternative backend using Google ADK.
#
# This module provides an alternative voice agent implementation that uses:
# - Google ADK (Agent Development Kit) for agent orchestration
# - PyTorch-based Silero VAD (instead of ONNX Runtime)
# - LiveKit Data Channel for broadcasting UI state to frontends
# - Gemini Live API for real-time audio conversation
#
# It complements the existing custom_voice_agent architecture by adding
# real-time UI state synchronization via LiveKit's data channel, so a
# React/Next.js frontend can display live barge-in, speaking, and idle
# states without polling.
#
# Usage:
#   python -m custom_voice_agent.livekit_ui_integration
#
# Required environment variables:
#   GEMINI_API_KEY     — Google AI API key for Gemini
#   LIVEKIT_URL        — LiveKit server URL
#   LIVEKIT_API_KEY    — LiveKit API key
#   LIVEKIT_API_SECRET — LiveKit API secret
#

import asyncio
import json
import logging
import os
from typing import Optional

import numpy as np
import torch
from google import genai
from google.genai import types
from livekit import api, rtc

# Google ADK Imports
from google.adk.agents import Agent
from google.adk.callbacks import BaseCallback
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & PARAMETERS (100% Pipecat Parity)
# ==============================================================================
RATE_16K = 16000
CHUNK_SIZE = 512  # 32ms chunks strictly required for Silero timing parity
BYTES_PER_CHUNK = CHUNK_SIZE * 2  # 16-bit PCM = 2 bytes per sample

# The precise VAD parameters driving the ultra-fast barge-in
VAD_CONFIDENCE_THRESHOLD = 0.75
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6


# ==============================================================================
# ADK CALLBACK LOGIC (Interruption Handling & UI Sync)
# ==============================================================================
class VADInterruptionCallback(BaseCallback):
    """ADK callback for handling VAD interruptions and broadcasting UI state.

    Coordinates between client-side VAD events and the ADK agent lifecycle,
    broadcasting state changes to connected frontends via LiveKit data channel.
    """

    def __init__(self, agent_reference: "ADKLiveKitAgent"):
        self.flush_playback_queue = False
        self.barge_in_active = False
        self.agent_ref = agent_reference

    def on_user_speech_started(self):
        """Triggered when client-side VAD detects user speech onset."""
        logger.info("User speaking — halting execution & triggering 200ms barge-in")
        self.flush_playback_queue = True
        self.barge_in_active = True

        # Broadcast to UI: Show "User Speaking" and "Agent Interrupted" visuals
        asyncio.create_task(self.agent_ref.broadcast_ui_state({
            "status": "barge_in_active",
            "message": "User interrupted the agent."
        }))

    def on_server_interruption_acknowledged(self):
        """Triggered when Gemini's server-side VAD acknowledges the interruption."""
        logger.info("Gemini server-side VAD acknowledged interruption")
        self.barge_in_active = False
        self.flush_playback_queue = True

        asyncio.create_task(self.agent_ref.broadcast_ui_state({
            "status": "listening",
            "message": "Agent is listening..."
        }))

    def on_turn_complete(self):
        """Triggered when the agent's turn is complete."""
        self.barge_in_active = False
        asyncio.create_task(self.agent_ref.broadcast_ui_state({
            "status": "idle",
            "message": "Waiting for user input."
        }))


# ==============================================================================
# CLIENT-SIDE VAD (SILERO via PyTorch)
# ==============================================================================
class SileroVADAnalyzer:
    """Voice Activity Detection using Silero VAD via PyTorch.

    This is a PyTorch-based alternative to the ONNX-based SileroVAD in
    vad/silero_vad.py. It uses the same confidence/timing parameters
    for identical barge-in behavior.

    Uses the same 4-state logic as the existing VAD but in a simplified
    single-method interface suitable for the ADK integration pattern.
    """

    def __init__(self, callback: VADInterruptionCallback):
        logger.info("Loading Silero VAD model (PyTorch)...")
        self.callback = callback
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.is_speaking = False
        self.speaking_frames = 0
        self.silent_frames = 0

        frames_per_sec = RATE_16K / CHUNK_SIZE
        self.start_frames_threshold = int(VAD_START_SECS * frames_per_sec)
        self.stop_frames_threshold = int(VAD_STOP_SECS * frames_per_sec)

    def process_chunk(self, audio_chunk: bytes) -> bool:
        """Analyze a single audio chunk and return whether the user is speaking.

        Args:
            audio_chunk: Raw PCM int16 audio, exactly BYTES_PER_CHUNK bytes.

        Returns:
            True if the user is currently speaking, False otherwise.
        """
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        volume = np.max(np.abs(audio_int16)) / 32768.0

        # Skip inference for very quiet frames (below 10% of the minimum
        # volume threshold) to save CPU.  At this level the signal is
        # indistinguishable from background noise.
        if volume < VAD_MIN_VOLUME * 0.1:
            confidence = 0.0
        else:
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            tensor = torch.from_numpy(audio_float32)
            confidence = self.model(tensor, RATE_16K).item()

        was_speaking = self.is_speaking

        if confidence >= VAD_CONFIDENCE_THRESHOLD:
            self.speaking_frames += 1
            self.silent_frames = 0
            if self.speaking_frames >= self.start_frames_threshold:
                self.is_speaking = True
        else:
            self.silent_frames += 1
            self.speaking_frames = 0
            if self.silent_frames >= self.stop_frames_threshold:
                self.is_speaking = False

        # Fire ADK Event when speech-start transition occurs
        if self.is_speaking and not was_speaking:
            self.callback.on_user_speech_started()

        return self.is_speaking


# ==============================================================================
# LIVEKIT & ADK AGENT ARCHITECTURE
# ==============================================================================
class ADKLiveKitAgent:
    """Voice agent combining Google ADK, Gemini Live, LiveKit, and Silero VAD.

    This agent:
    1. Joins a LiveKit room and publishes an audio track for bot output
    2. Subscribes to the user's audio track
    3. Runs client-side Silero VAD on user audio for fast barge-in detection
    4. Streams user audio to Gemini Live API for conversation
    5. Broadcasts UI state updates via LiveKit Data Channel so frontends
       can display real-time speaking/listening/barge-in indicators

    Architecture:
        User (LiveKit) → VAD → Gemini Live → Audio Output (LiveKit)
                           ↓                     ↓
                      Data Channel ← UI State Broadcasts
    """

    def __init__(self, room_name: str = "gemini-adk-room"):
        self.room_name = room_name
        self.room = rtc.Room()
        self.audio_source = rtc.AudioSource(24000, 1)
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.output_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.is_running = False

        # ADK Architecture Initialization
        self.adk_agent = Agent(
            name="conversational_ai",
            model="gemini-2.0-flash-exp",
            description="A highly responsive AI using dual VAD and fast barge-in.",
            instruction=(
                "You are a helpful, conversational AI assistant connected via LiveKit. "
                "You can be interrupted naturally mid-sentence and handle it gracefully. "
                "You know when to stay quiet (like when someone is talking to another "
                "person nearby). Filter out background noise and focus on the person "
                "speaking to you."
            )
        )
        self.session_service = InMemorySessionService()
        self.interruption_callback = VADInterruptionCallback(agent_reference=self)

        self.runner = Runner(
            agent=self.adk_agent,
            app_name="adk_live_voice",
            session_service=self.session_service,
            callbacks=[self.interruption_callback]
        )

        self.client = genai.Client()
        self.vad: Optional[SileroVADAnalyzer] = None

    def _init_vad(self):
        """Initialize VAD lazily so torch.hub.load runs after event loop starts."""
        if self.vad is None:
            self.vad = SileroVADAnalyzer(self.interruption_callback)

    def generate_token(self) -> str:
        """Generate a LiveKit access token for the agent participant.

        Returns:
            JWT access token string.

        Raises:
            ValueError: If LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not set.
        """
        lk_api_key = os.getenv("LIVEKIT_API_KEY")
        lk_api_secret = os.getenv("LIVEKIT_API_SECRET")
        if not lk_api_key or not lk_api_secret:
            raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required")

        token = api.AccessToken(lk_api_key, lk_api_secret) \
            .with_identity("adk-agent") \
            .with_name("ADK Gemini Live") \
            .with_grants(api.VideoGrants(room_join=True, room=self.room_name))
        return token.to_jwt()

    async def broadcast_ui_state(self, payload: dict):
        """Send JSON state updates to connected frontends via LiveKit Data Channel.

        Args:
            payload: Dictionary with 'status' and 'message' keys to broadcast.
        """
        try:
            if self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                data = json.dumps(payload).encode('utf-8')
                await self.room.local_participant.publish_data(data, reliable=True)
        except Exception:
            logger.debug("Failed to broadcast UI state", exc_info=True)

    async def _process_user_track(self, track: rtc.RemoteAudioTrack):
        """Process incoming user audio: resample, chunk, and queue for VAD + Gemini.

        Args:
            track: The remote audio track from the user participant.
        """
        logger.info(f"Subscribed to user audio track: {track.sid}")
        audio_stream = rtc.AudioStream(track)
        pcm_buffer = bytearray()

        async for frame in audio_stream:
            if not self.is_running:
                break
            raw_data = frame.data.tobytes()
            audio_np = np.frombuffer(raw_data, dtype=np.int16)

            # Mono normalization for WebRTC stereo inputs
            if frame.num_channels > 1:
                audio_np = audio_np.reshape(-1, frame.num_channels)[:, 0]

            # Downsample to 16kHz
            if frame.sample_rate == 48000:
                raw_data = audio_np[::3].tobytes()
            elif frame.sample_rate == 16000:
                raw_data = audio_np.tobytes()
            else:
                # General case: linear interpolation resampling
                ratio = RATE_16K / frame.sample_rate
                dst_len = int(len(audio_np) * ratio)
                if dst_len > 0:
                    indices = np.linspace(0, len(audio_np) - 1, dst_len)
                    resampled = np.interp(indices, np.arange(len(audio_np)),
                                          audio_np.astype(np.float64))
                    raw_data = np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()
                else:
                    continue

            pcm_buffer.extend(raw_data)

            # Ensure exact 512-sample chunks for accurate VAD timing
            while len(pcm_buffer) >= BYTES_PER_CHUNK:
                chunk = bytes(pcm_buffer[:BYTES_PER_CHUNK])
                pcm_buffer = pcm_buffer[BYTES_PER_CHUNK:]
                self.input_queue.put_nowait(chunk)

    async def _publish_agent_audio(self):
        """Background task: stream bot audio to LiveKit with barge-in support."""
        was_speaking = False
        while self.is_running:
            if self.interruption_callback.flush_playback_queue:
                # Barge-in: drain the output queue immediately
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self.interruption_callback.flush_playback_queue = False
                was_speaking = False
                await asyncio.sleep(0.01)
                continue

            try:
                pcm_data = await asyncio.wait_for(self.output_queue.get(), timeout=0.1)

                # Notify UI that agent is actively outputting audio
                if not was_speaking:
                    asyncio.create_task(self.broadcast_ui_state({
                        "status": "speaking",
                        "message": "Agent is speaking."
                    }))
                    was_speaking = True

                frame = rtc.AudioFrame(
                    data=pcm_data, sample_rate=24000, num_channels=1,
                    samples_per_channel=len(pcm_data) // 2
                )
                await self.audio_source.capture_frame(frame)
            except asyncio.TimeoutError:
                if was_speaking:
                    was_speaking = False

    async def _send_audio_to_gemini(self, session):
        """Background task: forward VAD-filtered user audio to Gemini Live.

        Only sends audio chunks where the client-side VAD detects speech,
        reducing noise and improving Gemini's response quality.
        """
        while self.is_running:
            audio_chunk = await self.input_queue.get()
            is_speaking = self.vad.process_chunk(audio_chunk)

            if is_speaking:
                await session.send(input=types.LiveClientRealtimeInput(
                    media_chunks=[types.Blob(
                        data=audio_chunk,
                        mime_type="audio/pcm;rate=16000"
                    )]
                ))

    async def _receive_from_gemini(self, session):
        """Background task: receive audio/text from Gemini and queue for output."""
        async for response in session.receive():
            server_content = response.server_content

            if server_content is not None:
                if server_content.interrupted:
                    self.interruption_callback.on_server_interruption_acknowledged()

                if server_content.turn_complete:
                    self.interruption_callback.on_turn_complete()

                model_turn = server_content.model_turn
                if model_turn is not None:
                    # Ignore incoming data if barge-in is active
                    if self.interruption_callback.barge_in_active:
                        continue

                    for part in model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            self.output_queue.put_nowait(part.inline_data.data)

    async def start(self):
        """Start the ADK agent: connect to LiveKit, initialize VAD, and run."""
        self.is_running = True
        self._init_vad()

        token = self.generate_token()
        logger.info(f"Connecting to LiveKit Room: {self.room_name}...")

        livekit_url = os.getenv("LIVEKIT_URL")
        if not livekit_url:
            raise ValueError("LIVEKIT_URL environment variable is required")

        await self.room.connect(livekit_url, token)

        track = rtc.LocalAudioTrack.create_audio_track(
            "adk-agent-mic", self.audio_source
        )
        await self.room.local_participant.publish_track(track)

        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            pub: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.create_task(self._process_user_track(track))

        config = types.LiveConnectConfig(
            response_modalities=[types.LiveModality.AUDIO],
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=self.adk_agent.instruction)]
            ),
            voice_name="Aoede"  # Gemini's default female voice
        )

        async with self.client.aio.live.connect(
            model=self.adk_agent.model, config=config
        ) as session:
            logger.info("ADK Agent is active and listening in LiveKit room!")
            self.session_service.create_session(
                user_id="livekit_user",
                session_id="voice_session_01"
            )

            tasks = [
                asyncio.create_task(self._publish_agent_audio()),
                asyncio.create_task(self._send_audio_to_gemini(session)),
                asyncio.create_task(self._receive_from_gemini(session))
            ]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass
            finally:
                self.is_running = False
                await self.room.disconnect()


async def main():
    """Entry point for the ADK LiveKit UI integration agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    required = ["GEMINI_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    missing = [e for e in required if not os.getenv(e)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return

    agent = ADKLiveKitAgent(
        room_name=os.getenv("LIVEKIT_ROOM", "gemini-adk-room")
    )
    task = asyncio.create_task(agent.start())
    try:
        await asyncio.Event().wait()  # Run forever until interrupted
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
