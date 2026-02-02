#
# Session Timer Processor
#
# Enforces a maximum session duration for calls.
# Ends the call immediately when the timeout is reached.
#

import asyncio
import os
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    EndTaskFrame,
    Frame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from end_of_call_reporter import EndedReason


class SessionTimerProcessor(FrameProcessor):
    """Enforces a maximum session duration.

    When the timer expires, it immediately ends the call.
    With Gemini Live native audio, we can't reliably inject goodbye messages,
    so we just terminate cleanly.

    Environment Variables:
        MAX_CALL_DURATION_SECS: Maximum call duration (default: 840 = 14 minutes)
    """

    def __init__(
        self,
        max_duration_secs: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the session timer.

        Args:
            max_duration_secs: Maximum session duration in seconds.
                              Defaults to MAX_CALL_DURATION_SECS env var or 840.
        """
        super().__init__(**kwargs)

        default_max = int(os.getenv("MAX_CALL_DURATION_SECS", "840"))
        self._max_duration = max_duration_secs if max_duration_secs is not None else default_max

        self._timer_task: Optional[asyncio.Task] = None
        self._session_started = False

        logger.info(f"SessionTimerProcessor: max_duration={self._max_duration}s")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Start timer when StartFrame is received
        if isinstance(frame, StartFrame) and not self._session_started:
            self._session_started = True
            self._start_timer()

        # Stop timer on end/cancel frames
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop_timer()

        await self.push_frame(frame, direction)

    def _start_timer(self):
        """Start the session timer."""
        if self._timer_task is None:
            logger.info(f"Session timer started: {self._max_duration}s max duration")
            self._timer_task = self.create_task(self._timer_handler())

    async def _stop_timer(self):
        """Stop the timer."""
        if self._timer_task:
            await self.cancel_task(self._timer_task)
            self._timer_task = None

    async def _timer_handler(self):
        """Handle session timeout."""
        await asyncio.sleep(self._max_duration)
        logger.warning(f"Session max duration ({self._max_duration}s) reached, ending call")
        await self.push_frame(
            EndTaskFrame(reason=EndedReason.EXCEEDED_MAX_DURATION),
            FrameDirection.UPSTREAM,
        )

    async def cleanup(self):
        await super().cleanup()
        await self._stop_timer()
