"""Camera processor for capturing frames from video streams."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
import threading
import time

import cv2
import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Result from frame capture."""

    frame: np.ndarray | None
    success: bool
    error: str | None = None
    timestamp: float = 0


class CameraProcessor:
    """Processor for capturing frames from IP cameras."""

    def __init__(
        self,
        stream_url: str,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        """Initialize the camera processor.

        Args:
            stream_url: URL of the video stream (RTSP, HTTP, etc.)
            username: Optional username for authentication
            password: Optional password for authentication
        """
        self._stream_url = stream_url
        self._username = username
        self._password = password
        self._authenticated_url = self._build_authenticated_url()
        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._last_frame: np.ndarray | None = None
        self._last_frame_time: float = 0

    def _build_authenticated_url(self) -> str:
        """Build URL with authentication credentials if provided."""
        if not self._username or not self._password:
            return self._stream_url

        # Check if URL already has credentials
        if "@" in self._stream_url:
            return self._stream_url

        # Insert credentials into URL
        if "://" in self._stream_url:
            protocol, rest = self._stream_url.split("://", 1)
            return f"{protocol}://{self._username}:{self._password}@{rest}"

        return self._stream_url

    def capture_frame(self) -> FrameResult:
        """Capture a single frame from the video stream.

        Returns:
            FrameResult with the captured frame or error information
        """
        with self._lock:
            try:
                # Create new capture for each frame to handle reconnection
                cap = cv2.VideoCapture(self._authenticated_url)

                if not cap.isOpened():
                    return FrameResult(
                        frame=None,
                        success=False,
                        error="Failed to open video stream",
                    )

                # Set buffer size to minimize latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Read a few frames to get the latest
                for _ in range(3):
                    ret, frame = cap.read()

                cap.release()

                if not ret or frame is None:
                    return FrameResult(
                        frame=None,
                        success=False,
                        error="Failed to read frame from stream",
                    )

                self._last_frame = frame
                self._last_frame_time = time.time()

                return FrameResult(
                    frame=frame,
                    success=True,
                    timestamp=self._last_frame_time,
                )

            except Exception as ex:
                _LOGGER.error("Error capturing frame: %s", ex)
                return FrameResult(
                    frame=None,
                    success=False,
                    error=str(ex),
                )

    def get_last_frame(self) -> FrameResult:
        """Get the last captured frame.

        Returns:
            FrameResult with the last captured frame
        """
        if self._last_frame is not None:
            return FrameResult(
                frame=self._last_frame.copy(),
                success=True,
                timestamp=self._last_frame_time,
            )
        return FrameResult(
            frame=None,
            success=False,
            error="No frame captured yet",
        )

    def test_connection(self) -> bool:
        """Test if the stream is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(self._authenticated_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                return False

            ret, _ = cap.read()
            cap.release()

            return ret
        except Exception as ex:
            _LOGGER.error("Connection test failed: %s", ex)
            return False

    @property
    def stream_url(self) -> str:
        """Return the stream URL (without credentials)."""
        return self._stream_url


class CameraProcessorPool:
    """Pool of camera processors for managing multiple cameras."""

    def __init__(self) -> None:
        """Initialize the camera processor pool."""
        self._processors: dict[str, CameraProcessor] = {}
        self._lock = threading.Lock()

    def get_processor(
        self,
        stream_url: str,
        username: str | None = None,
        password: str | None = None,
    ) -> CameraProcessor:
        """Get or create a camera processor for the given stream.

        Args:
            stream_url: URL of the video stream
            username: Optional username for authentication
            password: Optional password for authentication

        Returns:
            CameraProcessor instance
        """
        key = f"{stream_url}_{username or ''}"

        with self._lock:
            if key not in self._processors:
                self._processors[key] = CameraProcessor(
                    stream_url, username, password
                )
            return self._processors[key]

    def remove_processor(self, stream_url: str, username: str | None = None) -> None:
        """Remove a camera processor from the pool.

        Args:
            stream_url: URL of the video stream
            username: Optional username for authentication
        """
        key = f"{stream_url}_{username or ''}"

        with self._lock:
            if key in self._processors:
                del self._processors[key]
