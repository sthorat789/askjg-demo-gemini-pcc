#
# Gemini Live Pipeline: Native Audio Voice-to-Voice
#
# Uses Gemini Live's native audio model for speech-to-speech inference.
# No separate STT/TTS - the model handles audio directly.
# Uses server-side VAD from Gemini for turn detection.
#

import os
from typing import Optional

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIProcessor,
)
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport

# Gemini Live services (native audio)
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    InputParams,
)
from pipecat.services.google.gemini_live.llm_vertex import GeminiLiveVertexLLMService
from google.genai.types import (
    HttpOptions,
    ProactivityConfig,
    ThinkingConfig,
)

from pipelines.base import PipelineConfig
from core.observers import GeminiLiveRTVIObserver
from call_recorder import CallRecorder
from end_of_call_reporter import EndedReason
from processors.session_timer import SessionTimerProcessor


# ============================================================================
# Function Calling: End Conversation
# ============================================================================


async def end_conversation(params: FunctionCallParams):
    """LLM-callable function to gracefully end the conversation.

    The LLM should say goodbye BEFORE calling this function.
    """
    logger.info("end_conversation function called by LLM")
    await params.result_callback({"success": True})
    # Push EndTaskFrame upstream for graceful termination
    await params.llm.push_frame(
        EndTaskFrame(reason=EndedReason.ASSISTANT_ENDED_CALL),
        FrameDirection.UPSTREAM,
    )


# Define the function schema for Gemini
END_CONVERSATION_FUNCTION = FunctionSchema(
    name="end_conversation",
    description=(
        "Gracefully end the conversation and hang up. "
        "ONLY call this AFTER the user says goodbye AND you have said goodbye back. "
        "NEVER call at the start of a conversation. "
        "NEVER call unless the user explicitly indicates they want to end the call."
    ),
    properties={},
    required=[],
)


def create_llm_ga(
    system_prompt: str,
    credentials: Optional[str] = None,
    credentials_path: Optional[str] = None,
    project_id: Optional[str] = None,
    location: str = "us-central1",
) -> GeminiLiveVertexLLMService:
    """Create Gemini Live LLM service via Vertex AI (GA model).

    Uses the GA model for production stability.
    Function calling is disabled - call termination handled by safeguards.

    Args:
        system_prompt: The system instruction for the model
        credentials: Google Vertex credentials as JSON string
        credentials_path: Path to credentials file (alternative to credentials)
        project_id: Google Cloud project ID
        location: Google Cloud region
    """
    logger.info(f"Using Gemini Live GA model via Vertex AI (location={location})")

    return GeminiLiveVertexLLMService(
        credentials=credentials,
        credentials_path=credentials_path,
        project_id=project_id,
        location=location,
        model="google/gemini-live-2.5-flash-native-audio",  # GA model
        voice_id="Aoede",  # Warm, professional voice
        system_instruction=system_prompt,
        # No tools - function calling disabled for GA model
        params=InputParams(
            max_tokens=8192,
            # Enable affective dialog for emotional responsiveness
            enable_affective_dialog=True,
        ),
    )


def create_llm_preview(
    system_prompt: str,
    api_key: str,
    tools: Optional[ToolsSchema] = None,
) -> GeminiLiveLLMService:
    """Create Gemini Live LLM service via Google AI API (preview model).

    Uses the December 2025 preview model with optional thinking mode.

    Args:
        system_prompt: The system instruction for the model
        api_key: Google AI API key
        tools: Optional tools schema for function calling

    Environment Variables:
        THINKING_BUDGET_TOKENS: Thinking mode token budget (default: 0 = disabled)
    """
    # Load thinking budget (0 = disabled, default for lower latency)
    thinking_budget = int(os.getenv("THINKING_BUDGET_TOKENS", "0"))
    if thinking_budget > 0:
        logger.info(f"Using preview model with thinking mode: budget={thinking_budget} tokens")
    else:
        logger.info("Using preview model (thinking mode disabled)")

    return GeminiLiveLLMService(
        api_key=api_key,
        model="gemini-2.5-flash-native-audio-preview-12-2025",  # Dec 2025 preview
        voice_id="Aoede",  # Warm, professional voice
        system_instruction=system_prompt,
        tools=tools,
        params=InputParams(
            max_tokens=8192,
            # Enable proactive audio for natural conversation flow
            proactivity=ProactivityConfig(proactive_audio=True),
            # Thinking mode: only enable if budget > 0
            thinking=ThinkingConfig(
                includeThoughts=False,
                thinkingBudget=thinking_budget,
            ) if thinking_budget > 0 else None,
            # Enable affective dialog for emotional responsiveness
            enable_affective_dialog=True,
        ),
        # Required for proactivity and thinking mode
        http_options=HttpOptions(api_version="v1alpha"),
    )


def create_pipeline(
    transport: BaseTransport,
    sample_rate: int = 16000,
    enable_rtvi: bool = True,
    bot_name: str = "Maya",
    recorder: Optional[CallRecorder] = None,
    system_prompt: Optional[str] = None,
) -> PipelineConfig:
    """Create Gemini Live voice-to-voice pipeline.

    Args:
        transport: The transport to use for audio I/O
        sample_rate: Audio sample rate (16000 for Daily/WebRTC)
        enable_rtvi: Whether to enable RTVI protocol for voice-ui-kit
        bot_name: Name the bot introduces itself as
        recorder: Optional CallRecorder for recording audio
        system_prompt: System prompt for the LLM (required)

    Returns:
        PipelineConfig with the constructed pipeline and configuration

    Note:
        Gemini Live uses server-side VAD, so client-side VAD should be disabled
        when creating the transport for this pipeline.

    Environment Variables:
        GEMINI_MODEL: Model selection - "ga" (default) or "preview"
        MAX_CALL_DURATION_SECS: Maximum call duration in seconds (default: 840)
        USER_IDLE_TIMEOUT_SECS: User idle timeout in seconds (default: 120)
    """
    if not system_prompt:
        raise ValueError("system_prompt is required")

    logger.info(f"Creating Gemini Live pipeline (native audio, sample_rate={sample_rate}Hz)")

    # =========================================================================
    # Configuration from environment
    # =========================================================================
    user_idle_timeout = float(os.getenv("USER_IDLE_TIMEOUT_SECS", "120"))
    logger.info(f"Call safeguards: user_idle_timeout={user_idle_timeout}s")

    # =========================================================================
    # Model Selection: GA (Vertex AI) or Preview (Google AI API)
    # =========================================================================
    model_type = os.getenv("GEMINI_MODEL", "ga").lower()

    if model_type == "preview":
        # Google AI API with preview model (supports thinking mode + function calling)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for preview model")
        tools = ToolsSchema(standard_tools=[END_CONVERSATION_FUNCTION])
        llm = create_llm_preview(
            system_prompt=system_prompt,
            api_key=api_key,
            tools=tools,
        )
        llm.register_function("end_conversation", end_conversation)
    else:
        # Vertex AI with GA model (default, production stable)
        # Function calling disabled - rely on safeguards for call termination
        credentials = os.getenv("GOOGLE_VERTEX_CREDENTIALS")
        credentials_path = os.getenv("GOOGLE_VERTEX_CREDENTIALS_PATH") if not credentials else None
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not credentials and not credentials_path:
            raise ValueError(
                "GOOGLE_VERTEX_CREDENTIALS or GOOGLE_VERTEX_CREDENTIALS_PATH is required for GA model"
            )
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT_ID is required for GA model")

        llm = create_llm_ga(
            system_prompt=system_prompt,
            credentials=credentials,
            credentials_path=credentials_path,
            project_id=project_id,
            location=location,
        )

    # =========================================================================
    # User Idle Detection
    # =========================================================================

    async def handle_user_idle(processor: UserIdleProcessor, retry_count: int) -> bool:
        """Handle user inactivity by ending the call.

        With Gemini Live native audio, we can't reliably inject prompts,
        so we just end the call immediately on idle timeout.
        """
        logger.warning(f"User idle timeout ({user_idle_timeout}s) reached, ending call")
        await processor.push_frame(
            EndTaskFrame(reason=EndedReason.SILENCE_TIMED_OUT),
            FrameDirection.UPSTREAM,
        )
        return False

    user_idle = UserIdleProcessor(callback=handle_user_idle, timeout=user_idle_timeout)

    # =========================================================================
    # Session Timer (max call duration)
    # =========================================================================
    # max_duration_secs from env var (default 840s = 14 min)
    session_timer = SessionTimerProcessor()

    # Initial context to trigger the greeting
    llm_context = LLMContext(
        [
            {
                "role": "user",
                "content": f"Start the conversation. Introduce yourself as {bot_name}.",
            },
        ],
    )
    context_aggregator = LLMContextAggregatorPair(llm_context)

    transcript = TranscriptProcessor()

    # RTVI processor for voice-ui-kit integration
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]), transport=transport) if enable_rtvi else None

    # =========================================================================
    # Pipeline: Voice-to-Voice Architecture with Safeguards
    # =========================================================================
    # Frame flow: AudioRawFrame -> LLM -> AudioRawFrame (no separate STT/TTS)
    pipeline = Pipeline(
        [
            p
            for p in [
                transport.input(),  # Audio from client
                session_timer,  # Track session duration (near start)
                rtvi,  # RTVI processor
                user_idle,  # Monitor user activity
                context_aggregator.user(),  # Context aggregation
                transcript.user(),  # Track user transcriptions
                llm,  # Gemini Live: Audio -> Audio
                transport.output(),  # Audio to client
                recorder.processor if recorder else None,  # Recording processor
                transcript.assistant(),  # Track assistant transcriptions
                context_aggregator.assistant(),
            ]
            if p is not None
        ]
    )

    # Custom RTVI observer for voice-to-voice (emits bot-llm-text from TTSTextFrame)
    observers = [GeminiLiveRTVIObserver(rtvi)] if rtvi else []

    task_params = PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
    )

    return PipelineConfig(
        pipeline=pipeline,
        observers=observers,
        task_params=task_params,
        rtvi_observer_class=GeminiLiveRTVIObserver,
        needs_llm_run_frame=True,  # Need to trigger initial greeting
        transcript_processor=transcript,
    )
