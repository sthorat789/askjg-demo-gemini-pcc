"""Transport context for call reporting.

This module provides a dataclass to capture transport-specific metadata
(Daily WebRTC, Twilio telephony, local WebRTC) for end-of-call reports.
"""

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
import uuid


@dataclass
class TransportContext:
    """Context for call reporting based on transport type.

    Attributes:
        transport: Transport type - "web", "daily", or "twilio"
        direction: Call direction - "inbound" (default)
        session_id: Unique session ID for recording filenames
        call_id: Unique call identifier

        # Daily-specific
        daily_room_name: Daily.co room name

        # Twilio-specific
        twilio_call_sid: Twilio Call SID (CA...)
        twilio_stream_sid: Twilio Media Stream SID (MZ...)
        twilio_direction: Twilio call direction
        twilio_phone_from: Caller phone number (E.164 format)
        twilio_phone_to: Receiving phone number (E.164 format)
    """

    transport: str  # "web", "daily", "twilio"
    direction: str  # "inbound"
    session_id: str  # For recording filenames
    call_id: str  # Unique call identifier

    # Daily-specific metadata
    daily_room_name: Optional[str] = None

    # Twilio-specific metadata
    twilio_call_sid: Optional[str] = None
    twilio_stream_sid: Optional[str] = None
    twilio_direction: Optional[str] = None
    twilio_phone_from: Optional[str] = None
    twilio_phone_to: Optional[str] = None


def build_transport_context(
    transport_type: str,
    room_url: Optional[str] = None,
    call_data: Optional[dict] = None,
    direction: str = "inbound",
) -> TransportContext:
    """Build transport context from available data.

    Args:
        transport_type: "daily", "twilio", or "web"
        room_url: Daily room URL (for daily transport)
        call_data: Twilio call data from parse_telephony_websocket (for twilio transport)
        direction: Call direction, defaults to "inbound"

    Returns:
        TransportContext with appropriate IDs and metadata
    """
    if transport_type == "daily" and room_url:
        room_name = urlparse(room_url).path.strip("/")
        return TransportContext(
            transport="daily",
            direction=direction,
            session_id=room_name,
            call_id=room_name,
            daily_room_name=room_name,
        )

    if transport_type == "twilio" and call_data:
        call_sid = call_data.get("call_id", "")
        stream_sid = call_data.get("stream_id", "")
        body = call_data.get("body", {})

        # Use stream_sid for session_id, fallback to truncated call_sid
        session_id = stream_sid if stream_sid else call_sid[:16]

        return TransportContext(
            transport="twilio",
            direction=body.get("direction", direction),
            session_id=session_id,
            call_id=call_sid,
            twilio_call_sid=call_sid,
            twilio_stream_sid=stream_sid,
            twilio_direction=body.get("direction"),
            twilio_phone_from=body.get("phoneFrom"),
            twilio_phone_to=body.get("phoneTo"),
        )

    # Local WebRTC fallback
    unique_id = uuid.uuid4().hex[:12]
    return TransportContext(
        transport="web",
        direction=direction,
        session_id=f"web-{unique_id}",
        call_id=f"web-{unique_id}",
    )
