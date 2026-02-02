"""End-of-call reporting for Pipecat voice bots.

This module sends end-of-call reports to a webhook endpoint,
including transcript, recording URLs, LLM-generated summary,
and transport-specific metadata (Daily room name, Twilio phone numbers, etc.).
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import httpx
from loguru import logger

from pipecat.frames.frames import TranscriptionMessage, TranscriptionUpdateFrame

from core import gemini
from transport_context import TransportContext

if TYPE_CHECKING:
    from core.prompts import LoadedPrompt


class EndedReason:
    """Standardized end-of-call reason codes (Vapi-compatible).

    These codes are used in the end-of-call report to indicate why
    the call ended. Using Vapi-compatible naming for industry consistency.
    """

    # Normal endings
    CUSTOMER_ENDED_CALL = "customer-ended-call"
    ASSISTANT_ENDED_CALL = "assistant-ended-call"

    # Timeout endings
    EXCEEDED_MAX_DURATION = "exceeded-max-duration"
    SILENCE_TIMED_OUT = "silence-timed-out"
    CONNECTION_TIMED_OUT = "connection-timed-out"

    # Error endings
    PIPELINE_ERROR = "pipeline-error"

    @classmethod
    def is_normal_ending(cls, reason: str) -> bool:
        """Check if the reason represents a normal (non-error) call ending."""
        return reason in {
            cls.CUSTOMER_ENDED_CALL,
            cls.ASSISTANT_ENDED_CALL,
            cls.EXCEEDED_MAX_DURATION,
            cls.SILENCE_TIMED_OUT,
        }

    @classmethod
    def is_timeout(cls, reason: str) -> bool:
        """Check if the reason represents a timeout condition."""
        return reason in {
            cls.EXCEEDED_MAX_DURATION,
            cls.SILENCE_TIMED_OUT,
            cls.CONNECTION_TIMED_OUT,
        }

    @classmethod
    def is_error(cls, reason: str) -> bool:
        """Check if the reason represents an error condition."""
        return reason in {
            cls.PIPELINE_ERROR,
            cls.CONNECTION_TIMED_OUT,
        }


# Maximum transcript length to send to LLM (~12k tokens)
MAX_TRANSCRIPT_CHARS = 50000

# Timeout for summary generation
SUMMARY_TIMEOUT_SECONDS = 15.0

# Timeout for transcript processing (includes up to 3 retries)
PROCESSING_TIMEOUT_SECONDS = 120.0


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class CallTranscript:
    """Accumulates transcript messages throughout a call."""

    session_id: str
    messages: list[dict] = field(default_factory=list)
    started_at: Optional[datetime] = None

    def add_message(self, msg: TranscriptionMessage):
        """Add a transcript message with timing info."""
        now = datetime.now()

        if self.started_at is None:
            self.started_at = now

        seconds_from_start = (now - self.started_at).total_seconds()

        # Map role to Pipecat format
        role = "user" if msg.role == "user" else "assistant"

        self.messages.append(
            {
                "role": role,
                "content": msg.content,
                "timestamp": seconds_from_start,
            }
        )

    def to_plain_text(
        self, assistant_name: str = "Maya", include_timestamps: bool = True
    ) -> str:
        """Generate plain text transcript.

        Args:
            assistant_name: Name to use for assistant in transcript
            include_timestamps: Whether to include timestamps (False for LLM summary)
        """
        lines = []
        for m in self.messages:
            speaker = "User" if m["role"] == "user" else assistant_name
            if include_timestamps:
                timestamp = _format_timestamp(m["timestamp"])
                lines.append(f"[{timestamp}] {speaker}: {m['content']}")
            else:
                lines.append(f"{speaker}: {m['content']}")
        return "\n".join(lines)


class EndOfCallReporter:
    """Sends end-of-call reports to webhook endpoint.

    This class accumulates transcript messages during a call and sends
    an end-of-call report when the call ends, including an optional
    LLM-generated summary.
    """

    def __init__(
        self,
        context: TransportContext,
        webhook_url: Optional[str] = None,
        api_key: Optional[str] = None,
        auth_type: str = "header",
        assistant_name: str = "Maya",
        prompt_info: Optional["LoadedPrompt"] = None,
    ):
        """Initialize the reporter.

        Args:
            context: Transport context with call metadata
            webhook_url: URL to POST end-of-call reports to
            api_key: API key for webhook authentication
            auth_type: Authentication type - "header" (X-API-Key header) or
                       "body" (envelope wrapper with auth in body)
            assistant_name: Name to use in transcript (default: Maya)
            prompt_info: System prompt metadata for observability
        """
        self.context = context
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.auth_type = auth_type
        self.assistant_name = assistant_name
        self.prompt_info = prompt_info
        self.transcript = CallTranscript(session_id=context.session_id)
        self.recording_urls: dict[str, str] = {}
        self.ended_reason: str = EndedReason.CUSTOMER_ENDED_CALL
        self.summary: Optional[str] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._started_at: Optional[datetime] = None
        self._ended_at: Optional[datetime] = None
        self._report_sent: bool = False

        # Transcript processing state
        self._processed_messages: Optional[list[dict]] = None
        self._processing_time_ms: Optional[int] = None

    @property
    def enabled(self) -> bool:
        """Check if reporting is enabled (webhook URL and API key configured)."""
        return bool(self.webhook_url and self.api_key)

    @property
    def summary_enabled(self) -> bool:
        """Check if summary generation is enabled via ENABLE_SUMMARY env var."""
        return os.getenv("ENABLE_SUMMARY", "true").lower() == "true"

    @property
    def processing_enabled(self) -> bool:
        """Check if transcript processing is enabled via ENABLE_TRANSCRIPT_POST_PROCESSING env var."""
        return os.getenv("ENABLE_TRANSCRIPT_POST_PROCESSING", "true").lower() == "true"

    async def handle_transcript_update(
        self, _processor, frame: TranscriptionUpdateFrame
    ):
        """Event handler for transcript updates.

        Register this with TranscriptProcessor:
            @transcript.event_handler("on_transcript_update")
            async def on_transcript_update(processor, frame):
                await reporter.handle_transcript_update(processor, frame)
        """
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                self.transcript.add_message(msg)

    def set_recording_urls(self, urls: dict):
        """Set recording URLs from CallRecorder.

        Args:
            urls: Dictionary with recording info.
        """
        self.recording_urls = urls

    def set_ended_reason(self, reason: str):
        """Set the call ended reason."""
        self.ended_reason = reason

    def set_started_at(self):
        """Mark the call as started (call on client connect)."""
        self._started_at = datetime.now()

    def set_ended_at(self):
        """Mark the call as ended (call on client disconnect)."""
        self._ended_at = datetime.now()

    async def generate_summary(self) -> Optional[str]:
        """Generate a summary of the call transcript using the LLM.

        Returns:
            The generated summary string, or None if generation failed or timed out.
        """
        if not self.summary_enabled:
            logger.debug("Summary generation not enabled")
            return None

        # Use plain text without timestamps for LLM
        transcript_text = self.transcript.to_plain_text(
            self.assistant_name, include_timestamps=False
        )
        if not transcript_text.strip():
            logger.debug("No transcript to summarize")
            return None

        # Truncate long transcripts to avoid token limits
        if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
            logger.warning(
                f"Transcript truncated from {len(transcript_text)} to {MAX_TRANSCRIPT_CHARS} chars"
            )
            transcript_text = transcript_text[-MAX_TRANSCRIPT_CHARS:]

        logger.info("Generating call summary...")

        try:
            # Apply timeout to prevent blocking if LLM is slow
            summary = await asyncio.wait_for(
                self._generate_summary_internal(transcript_text),
                timeout=SUMMARY_TIMEOUT_SECONDS,
            )
            return summary
        except asyncio.TimeoutError:
            logger.warning(f"Summary generation timed out after {SUMMARY_TIMEOUT_SECONDS}s")
            return None
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None

    async def _generate_summary_internal(self, transcript_text: str) -> Optional[str]:
        """Internal summary generation without timeout wrapper."""
        summary = await gemini.generate_summary(transcript_text)

        if summary:
            self.summary = summary.strip()
            logger.info(f"Summary generated ({len(self.summary)} chars)")
            return self.summary
        else:
            logger.warning("Summary generation returned empty")
            return None

    async def process_transcript(self) -> Optional[list[dict]]:
        """Process transcript for translation and STT error correction.

        Uses Gemini to:
        - Translate non-English messages to English
        - Fix speech-to-text errors (homophones, word boundaries)

        Returns:
            List of processed messages, or None if processing failed.
        """
        messages = self.transcript.messages
        if not messages:
            logger.debug("No transcript messages to process")
            return None

        # Check transcript length
        messages_json = json.dumps(messages)
        if len(messages_json) > MAX_TRANSCRIPT_CHARS:
            logger.warning(
                f"Transcript too long for processing ({len(messages_json)} chars), skipping"
            )
            return None

        logger.info("Processing transcript for translation and STT correction...")
        start_time = datetime.now()

        try:
            processed = await asyncio.wait_for(
                self._process_transcript_internal(messages),
                timeout=PROCESSING_TIMEOUT_SECONDS,
            )

            if processed:
                self._processing_time_ms = int(
                    (datetime.now() - start_time).total_seconds() * 1000
                )
                self._processed_messages = processed
                logger.info(
                    f"Transcript processed ({len(processed)} messages, "
                    f"{self._processing_time_ms}ms)"
                )
            return processed
        except asyncio.TimeoutError:
            logger.warning(
                f"Transcript processing timed out after {PROCESSING_TIMEOUT_SECONDS}s"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to process transcript: {e}")
            return None

    async def _process_transcript_internal(
        self, messages: list[dict]
    ) -> Optional[list[dict]]:
        """Internal transcript processing without timeout wrapper."""
        # Uses structured JSON output via core.gemini module
        return await gemini.process_transcript(messages)

    def _build_processed_text(self) -> str:
        """Build formatted text from processed messages."""
        if not self._processed_messages:
            return ""

        lines = []
        for m in self._processed_messages:
            speaker = "User" if m.get("role") == "user" else self.assistant_name
            timestamp = m.get("timestamp")
            if timestamp is not None:
                ts_str = _format_timestamp(timestamp)
                lines.append(f"[{ts_str}] {speaker}: {m.get('content', '')}")
            else:
                lines.append(f"{speaker}: {m.get('content', '')}")
        return "\n".join(lines)

    def _build_report_data(self) -> dict:
        """Build the core end-of-call report data."""
        now = datetime.now()
        started_at = self._started_at or now
        ended_at = self._ended_at or now

        data = {
            "type": "end-of-call-report",
            "call": {
                "id": self.context.call_id,
                "transport": self.context.transport,
                "direction": self.context.direction,
                "status": "ended",
                "endedReason": self.ended_reason,
                "startedAt": started_at.isoformat(),
                "endedAt": ended_at.isoformat(),
            },
            "transcript": {
                "text": self.transcript.to_plain_text(self.assistant_name),
                "messages": self.transcript.messages,
            },
        }

        # Include summary if generated
        if self.summary:
            data["summary"] = self.summary

        # Include processed transcript if available
        if self._processed_messages:
            data["processedTranscript"] = {
                "version": 1,
                "processedAt": datetime.now().isoformat(),
                "processingTimeMs": self._processing_time_ms,
                "model": "gemini-2.5-flash",
                "messages": self._processed_messages,
                "text": self._build_processed_text(),
            }

        # Only include recording if we have URLs
        if self.recording_urls:
            data["recording"] = self.recording_urls

        # Include prompt info for observability
        if self.prompt_info:
            data["prompt"] = {
                "source": self.prompt_info.source,
                "hash": self.prompt_info.hash,
                "text": self.prompt_info.text,
            }

        # Only include metadata with actual values
        metadata = {}
        if self.context.session_id:
            metadata["sessionId"] = self.context.session_id

        # Daily-specific metadata
        if self.context.daily_room_name:
            metadata["dailyRoomName"] = self.context.daily_room_name

        # Twilio-specific metadata
        if self.context.twilio_call_sid:
            metadata["twilioCallSid"] = self.context.twilio_call_sid
        if self.context.twilio_stream_sid:
            metadata["twilioStreamSid"] = self.context.twilio_stream_sid
        if self.context.twilio_direction:
            metadata["twilioDirection"] = self.context.twilio_direction
        if self.context.twilio_phone_from:
            metadata["twilioPhoneFrom"] = self.context.twilio_phone_from
        if self.context.twilio_phone_to:
            metadata["twilioPhoneTo"] = self.context.twilio_phone_to

        if metadata:
            data["metadata"] = metadata

        return data

    def build_payload(self) -> dict:
        """Build end-of-call payload with appropriate auth format.

        Returns:
            For auth_type="header": the report data directly
            For auth_type="body": envelope with {auth: {...}, data: {...}}
        """
        data = self._build_report_data()

        if self.auth_type == "body":
            # Wrap in envelope with auth in body
            envelope = {"data": data}
            if self.api_key:
                envelope["auth"] = {"type": "api_key", "key": self.api_key}
            return envelope

        # Default: return data directly (auth via header)
        return data

    async def send_report(self) -> bool:
        """Send end-of-call report to webhook.

        This method will:
        1. Generate a summary (if configured)
        2. Build the payload
        3. Send to webhook

        Returns:
            True if report was sent successfully, False otherwise
        """
        # Guard against duplicate reports
        if self._report_sent:
            logger.debug("Report already sent, skipping duplicate")
            return False

        if not self.enabled:
            logger.debug("End-of-call reporting disabled (no webhook URL or API key)")
            return False

        # Mark as sent immediately to prevent race conditions
        self._report_sent = True

        # Generate summary before building payload
        if self.summary_enabled:
            await self.generate_summary()

        # Process transcript (translation + STT correction)
        if self.processing_enabled:
            await self.process_transcript()

        payload = self.build_payload()

        # Extract call_id from payload (may be nested in envelope for body auth)
        if self.auth_type == "body":
            call_id = payload["data"]["call"]["id"]
        else:
            call_id = payload["call"]["id"]

        message_count = len(self.transcript.messages)
        has_summary = self.summary is not None
        has_processed = self._processed_messages is not None

        logger.info(
            f"Sending end-of-call report for {call_id} "
            f"({message_count} messages, summary={'yes' if has_summary else 'no'}, "
            f"processed={'yes' if has_processed else 'no'})"
        )

        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

            # Build headers based on auth type
            headers = {"Content-Type": "application/json"}
            if self.auth_type == "header" and self.api_key:
                headers["X-API-Key"] = self.api_key

            response = await self._http_client.post(
                self.webhook_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            logger.info(f"End-of-call report sent successfully: {response.status_code}")
            return True
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Webhook returned error: {e.response.status_code} - {e.response.text}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to send end-of-call report: {e}")
            return False

    async def close(self):
        """Cleanup HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
