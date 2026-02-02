#
# Voice AI Demo Bot - Gemini Live Native Audio
#
# A demo assistant that showcases Gemini Live native audio capabilities
# and encourages visitors to book discovery calls.
#
# Deployed on Pipecat Cloud with Daily WebRTC or Twilio telephony transport.
# Supports local development via pipecat runner with SmallWebRTC or Twilio.
#

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

# Conditional Pipecat Cloud session argument imports
try:
    from pipecatcloud.agent import DailySessionArguments, WebSocketSessionArguments

    _PIPECAT_CLOUD = True
except ImportError:
    DailySessionArguments = None
    WebSocketSessionArguments = None
    _PIPECAT_CLOUD = False

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import CancelFrame, EndFrame, LLMRunFrame, TranscriptionMessage
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)

from pipelines import create_pipeline
from call_recorder import CallRecorder, RecordingMode
from core.prompts import load_system_prompt_async
from end_of_call_reporter import EndedReason, EndOfCallReporter
from transport_context import TransportContext, build_transport_context

load_dotenv(override=True)


def create_vad_analyzer() -> SileroVADAnalyzer:
    """Create VAD analyzer with project-wide settings.

    All VAD parameters are explicit for consistency across transports.
    Adjust these values to tune interruption sensitivity and speech detection.
    """
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.75, # Speech confidence threshold (default: 0.7)
            start_secs=0.2,  # Speech start detection delay (default: 0.2)
            stop_secs=0.2,   # Quick interruption detection (default: 0.8)
            min_volume=0.6,  # Minimum volume threshold (default: 0.6)
        )
    )


# Connection timeout: kill bot if no client connects within this time
CONNECTION_TIMEOUT_SECS = int(os.getenv("CONNECTION_TIMEOUT_SECS", "60"))

# Daily transport parameters for WebRTC
DAILY_PARAMS = DailyParams(
    audio_in_enabled=True,
    audio_out_enabled=True,
    camera_out_enabled=False,
    vad_analyzer=create_vad_analyzer(),
    transcription_enabled=False,  # Native audio, no STT needed
)


async def run_bot(
    transport: BaseTransport,
    context: TransportContext,
    handle_sigint: bool = True,
    enable_rtvi: bool = True,
    sample_rate: int = 16000,
):
    """Run the demo bot with the configured transport.

    Args:
        transport: The transport to use for audio I/O
        context: Transport context with call metadata for reporting
        handle_sigint: Whether to handle SIGINT for graceful shutdown
        enable_rtvi: Whether to enable RTVI protocol for voice-ui-kit
        sample_rate: Audio sample rate (16000 for Daily/WebRTC)
    """
    bot_name = os.getenv("BOT_NAME", "Maya")
    logger.info(
        f"Starting {bot_name} demo bot (transport={context.transport}, "
        f"sample_rate={sample_rate}Hz, RTVI={enable_rtvi})"
    )

    # Track client connection for timeout watchdog
    client_connected = asyncio.Event()

    # Initialize recording if enabled
    recording_mode = RecordingMode(os.getenv("RECORDING_MODE", "disabled"))
    bucket_name = os.getenv("GCS_BUCKET_NAME", "")
    prompt_source = os.getenv("PROMPT_SOURCE", "local")
    recorder = None

    if recording_mode != RecordingMode.DISABLED and bucket_name:
        recorder = CallRecorder(
            mode=recording_mode,
            session_id=context.session_id,
            bucket_name=bucket_name,
        )
        logger.info(f"Recording enabled: mode={recording_mode.value}, bucket={bucket_name}")
    elif recording_mode != RecordingMode.DISABLED:
        logger.warning("RECORDING_MODE is set but GCS_BUCKET_NAME is missing, recording disabled")

    # Load system prompt from configured source (local by default)
    prompt_info = await load_system_prompt_async(bucket_name, source=prompt_source)
    logger.info(f"Loaded system prompt: source={prompt_info.source}, hash={prompt_info.hash[:12]}...")

    # Create pipeline using factory
    pipeline_config = create_pipeline(
        transport=transport,
        sample_rate=sample_rate,
        enable_rtvi=enable_rtvi,
        bot_name=bot_name,
        recorder=recorder,
        system_prompt=prompt_info.text,
    )

    # Initialize end-of-call reporter with transport context
    reporter = EndOfCallReporter(
        context=context,
        webhook_url=os.getenv("WEBHOOK_URL"),
        api_key=os.getenv("WEBHOOK_API_KEY"),
        auth_type=os.getenv("WEBHOOK_AUTH_TYPE", "header"),
        assistant_name=bot_name,
        prompt_info=prompt_info,
    )
    if reporter.enabled:
        auth_type = os.getenv("WEBHOOK_AUTH_TYPE", "header")
        summary_status = "yes" if reporter.summary_enabled else "no"
        logger.info(f"End-of-call reporting enabled (auth_type={auth_type}, summary={summary_status})")

    task = PipelineTask(
        pipeline_config.pipeline,
        params=pipeline_config.task_params,
        observers=pipeline_config.observers,  # CRITICAL: Pass observers!
    )

    # Get RTVI processor from pipeline if available (for event handler)
    rtvi = None
    for processor in pipeline_config.pipeline.processors:
        if hasattr(processor, "set_bot_ready"):  # RTVIProcessor has this method
            rtvi = processor
            break

    # RTVI client ready handler (web transports only)
    if rtvi:

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            logger.info("Pipecat client ready, sending bot-ready")
            await rtvi.set_bot_ready()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        client_connected.set()  # Cancel timeout watchdog
        logger.info(f"Client connected to {bot_name} demo")
        # Mark call start time (aligns with recording start)
        reporter.set_started_at()
        # Start recording if enabled
        if recorder:
            await recorder.start()
        # Kick off the LLM if needed
        if pipeline_config.needs_llm_run_frame:
            await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected from {bot_name} demo")
        # Mark call end time for reporting
        reporter.set_ended_at()

        # Cancel task with customer-ended-call reason
        # Recording stop and report will be handled in on_pipeline_finished
        # NOTE: We don't stop recording here because cleanup() would cancel
        # the AudioBufferProcessor's tasks, blocking the CancelFrame from
        # propagating through the pipeline.
        await task.cancel(reason=EndedReason.CUSTOMER_ENDED_CALL)

    # Register transcript event handler (TranscriptProcessor is a factory, not in the pipeline)
    transcript_processor = pipeline_config.transcript_processor
    if transcript_processor:

        @transcript_processor.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            """Log transcriptions and accumulate for end-of-call report."""
            # Accumulate for report
            await reporter.handle_transcript_update(processor, frame)

            # Also log for debugging
            for msg in frame.messages:
                if isinstance(msg, TranscriptionMessage):
                    timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                    logger.info(f"Transcript: {timestamp}{msg.role}: {msg.content}")

    @task.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(task, frame):
        """Handle pipeline termination and send end-of-call report.

        This handler is called for ALL termination scenarios:
        - Client disconnect (CancelFrame with customer-ended-call)
        - Session timeout (EndFrame with exceeded-max-duration)
        - User idle timeout (EndFrame with silence-timed-out)
        - Bot-initiated end (EndFrame with assistant-ended-call)
        - Connection timeout (CancelFrame with connection-timed-out)
        """
        # Extract reason from frame
        reason = EndedReason.CUSTOMER_ENDED_CALL  # Default
        if isinstance(frame, EndFrame):
            reason = frame.reason or EndedReason.CUSTOMER_ENDED_CALL
        elif isinstance(frame, CancelFrame):
            reason = frame.reason or EndedReason.CUSTOMER_ENDED_CALL

        logger.info(f"Pipeline finished with reason: {reason}")
        reporter.set_ended_reason(reason)

        # Ensure ended_at is set (may not be if pipeline ended before disconnect)
        if reporter._ended_at is None:
            reporter.set_ended_at()

        # Stop recording and get URLs for the report
        # This is done here (after pipeline shutdown) rather than in on_client_disconnected
        # because cleanup() cancels the processor's tasks, which would block CancelFrame propagation.
        if recorder and recorder.is_running:
            await recorder.stop()
            reporter.set_recording_urls(recorder.recording_urls)

        # Send end-of-call report and cleanup
        await reporter.send_report()
        await reporter.close()

    # Watchdog: terminate if no client connects within timeout
    async def wait_for_client_connection():
        try:
            await asyncio.wait_for(client_connected.wait(), timeout=CONNECTION_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            logger.warning(f"No client connected within {CONNECTION_TIMEOUT_SECS}s, shutting down")
            await task.cancel(reason=EndedReason.CONNECTION_TIMED_OUT)

    runner = PipelineRunner(handle_sigint=handle_sigint)
    asyncio.create_task(wait_for_client_connection())
    await runner.run(task)


# Pipecat Cloud entry point
async def bot(args):
    """Pipecat Cloud entry point for Daily WebRTC or Twilio telephony.

    Args:
        args: Either DailySessionArguments (WebRTC) or WebSocketSessionArguments (telephony)
    """
    bot_name = os.getenv("BOT_NAME", "Maya")

    # Detect transport type based on args
    if WebSocketSessionArguments and isinstance(args, WebSocketSessionArguments):
        # Twilio telephony via WebSocket
        provider, call_data = await parse_telephony_websocket(args.websocket)
        logger.info(f"Bot started with telephony provider={provider}, call_id={call_data.get('call_id')}")

        if provider != "twilio":
            raise ValueError(f"Unsupported telephony provider: {provider}")

        # Build transport context for Twilio (includes phone numbers)
        transport_ctx = build_transport_context(
            transport_type="twilio",
            call_data=call_data,
        )

        # Configure Twilio serializer (no credentials needed with auto_hang_up=False)
        serializer = TwilioFrameSerializer(
            stream_sid=call_data["stream_id"],
            params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
        )

        # Create FastAPI WebSocket transport with same VAD settings as Daily
        transport = FastAPIWebsocketTransport(
            websocket=args.websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                serializer=serializer,
                add_wav_header=False,
                vad_analyzer=create_vad_analyzer(),
            ),
        )

        # In cloud, don't handle SIGINT (container manages lifecycle)
        # RTVI disabled for telephony (no Voice UI Kit)
        await run_bot(
            transport,
            transport_ctx,
            handle_sigint=False,
            enable_rtvi=False,
            sample_rate=16000,
        )

    else:
        # Daily WebRTC (default)
        logger.info(f"Bot started with room_url={args.room_url}")

        # Build transport context for Daily
        transport_ctx = build_transport_context(
            transport_type="daily",
            room_url=args.room_url,
        )

        transport = DailyTransport(
            args.room_url,
            args.token,
            bot_name,
            DAILY_PARAMS,
        )

        # In cloud, don't handle SIGINT (container manages lifecycle)
        await run_bot(
            transport,
            transport_ctx,
            handle_sigint=False,
            enable_rtvi=True,
            sample_rate=16000,
        )


# Local development entry point (using pipecat runner with SmallWebRTC or Twilio)
if __name__ == "__main__":
    from pipecat.runner.run import main as runner_main
    from pipecat.runner.types import RunnerArguments
    from pipecat.runner.utils import create_transport
    from pipecat.transports.base_transport import TransportParams

    bot_name = os.getenv("BOT_NAME", "Maya")

    # Transport params for local development (SmallWebRTC)
    def get_webrtc_params():
        return TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=create_vad_analyzer(),
        )

    # Transport params for Twilio telephony (use with ngrok for local testing)
    # Run: python bot/bot.py -t twilio -x <subdomain>.ngrok.io
    def get_telephony_params():
        return FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=create_vad_analyzer(),
            # serializer is set automatically by the runner
        )

    transport_params = {
        "webrtc": get_webrtc_params,
        "twilio": get_telephony_params,
    }

    async def local_bot(runner_args: RunnerArguments):
        """Local development bot entry point."""
        logger.info("Starting local development bot")

        # Check if this is a telephony connection
        if hasattr(runner_args, "websocket") and runner_args.websocket:
            # Telephony (Twilio) - manually create transport to avoid credential requirement
            provider, call_data = await parse_telephony_websocket(runner_args.websocket)
            logger.info(f"Local telephony: provider={provider}, call_id={call_data.get('call_id')}")

            if provider != "twilio":
                raise ValueError(f"Unsupported telephony provider: {provider}")

            # Build transport context for Twilio (includes phone numbers)
            transport_ctx = build_transport_context("twilio", call_data=call_data)

            # Create serializer with auto_hang_up=False (no credentials needed)
            serializer = TwilioFrameSerializer(
                stream_sid=call_data["stream_id"],
                params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
            )

            transport = FastAPIWebsocketTransport(
                websocket=runner_args.websocket,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    serializer=serializer,
                    add_wav_header=False,
                    vad_analyzer=create_vad_analyzer(),
                ),
            )
            enable_rtvi = False
        else:
            # WebRTC - use create_transport as before
            transport = await create_transport(runner_args, transport_params)
            transport_ctx = build_transport_context(transport_type="web")
            enable_rtvi = True

        logger.info(
            f"Transport context: transport={transport_ctx.transport}, "
            f"call_id={transport_ctx.call_id}, session_id={transport_ctx.session_id}"
        )

        await run_bot(
            transport,
            transport_ctx,
            handle_sigint=runner_args.handle_sigint,
            enable_rtvi=enable_rtvi,
            sample_rate=16000,
        )

    # Set bot for runner and start
    bot = local_bot
    runner_main()
