#
# RTVI Observers for voice-to-voice pipelines
#
# Voice-to-voice models (Gemini Live, OpenAI Realtime) emit TTSTextFrame instead of
# LLMTextFrame. The base RTVIObserver expects LLMTextFrame for bot-llm-text,
# so these subclasses send bot-llm-text from TTSTextFrame for voice-ui-kit.
#

from pipecat.frames.frames import TTSTextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frameworks.rtvi import (
    RTVIBotLLMTextMessage,
    RTVIObserver,
    RTVITextMessageData,
)
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.utils.string import match_endofsentence


class VoiceToVoiceRTVIObserver(RTVIObserver):
    """Extended RTVIObserver that emits bot-llm-text from TTSTextFrame.

    Voice-to-voice models (Gemini Live, OpenAI Realtime) emit TTSTextFrame
    instead of LLMTextFrame. The base RTVIObserver expects LLMTextFrame for
    bot-llm-text, so this subclass sends bot-llm-text from TTSTextFrame for
    voice-ui-kit.
    """

    def __init__(self, rtvi, *, params=None, **kwargs):
        super().__init__(rtvi, params=params, **kwargs)
        self._bot_tts_transcription_buffer = ""

    async def on_push_frame(self, data: FramePushed):
        await super().on_push_frame(data)

        frame = data.frame
        src = data.source

        # For voice-to-voice models: emit bot-llm-text from TTSTextFrame
        # voice-ui-kit expects bot-llm-text for conversation panel display
        # Only process when source is BaseOutputTransport (ensures correct timing,
        # avoids duplicates from other pipeline positions)
        if isinstance(frame, TTSTextFrame) and isinstance(src, BaseOutputTransport):
            self._bot_tts_transcription_buffer += frame.text

            if match_endofsentence(self._bot_tts_transcription_buffer):
                if self._bot_tts_transcription_buffer.strip():
                    message = RTVIBotLLMTextMessage(
                        data=RTVITextMessageData(text=self._bot_tts_transcription_buffer)
                    )
                    await self.send_rtvi_message(message)
                self._bot_tts_transcription_buffer = ""


# Alias for Gemini Live pipeline
GeminiLiveRTVIObserver = VoiceToVoiceRTVIObserver
