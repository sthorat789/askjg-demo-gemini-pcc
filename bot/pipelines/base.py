#
# Base types and interfaces for pipeline configurations
#

from dataclasses import dataclass
from typing import Any

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.frameworks.rtvi import RTVIObserver


@dataclass
class PipelineConfig:
    """Configuration returned by pipeline factory functions.

    Attributes:
        pipeline: The constructed Pipeline with all processors
        observers: List of observers (e.g., RTVIObserver)
        task_params: Parameters for PipelineTask
        rtvi_observer_class: Observer class to use (RTVIObserver for cascade,
                             VoiceToVoiceRTVIObserver for voice-to-voice models)
        needs_llm_run_frame: Whether to send LLMRunFrame on client connect
                             (True for cascade/text LLMs, False for voice-to-voice)
        transcript_processor: TranscriptProcessor instance for event handler registration
    """

    pipeline: Pipeline
    observers: list
    task_params: PipelineParams
    rtvi_observer_class: type = RTVIObserver
    needs_llm_run_frame: bool = True
    transcript_processor: Any = None
