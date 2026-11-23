"""
AI-Powered Precision Pest Management - Safe Video Stream Ingestion Interface
File: src/monitoring/video_stream.py

Provides robust, hardware-independent video frame ingestion supporting:
1. Local pre-recorded video files (.mp4, .avi, .mkv)
2. Synthetic animated pest streams for deterministic offline simulation
3. Injectable custom frame sequences for unit testing
4. Optional physical camera stream (disabled by default; strictly isolated)

Safety Constraint:
Physical camera hardware is never required to pass tests or run offline pipelines.
"""

import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np


class StreamSourceType(str, Enum):
    FILE = "FILE"
    SYNTHETIC = "SYNTHETIC"
    INJECTED = "INJECTED"
    CAMERA = "CAMERA"


class StreamStatus(str, Enum):
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    END_OF_STREAM = "END_OF_STREAM"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


class SafeVideoStream:
    """
    Robust video stream ingestion abstraction with automatic fallbacks,
    looping options, end-of-stream detection, and resource management.
    """

    def __init__(
        self,
        source: Optional[Union[int, str, Path]] = None,
        width: int = 640,
        height: int = 480,
        fps_target: int = 30,
        loop: bool = False,
        allow_camera: bool = False,
        fallback_synthetic: bool = True,
        injected_frames: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps_target = fps_target
        self.loop = loop
        self.allow_camera = allow_camera
        self.fallback_synthetic = fallback_synthetic
        self.is_fallback_active = False

        self.cap: Optional[cv2.VideoCapture] = None
        self.injected_frames: List[np.ndarray] = injected_frames or []
        self.injected_index = 0

        self.source_type = StreamSourceType.SYNTHETIC
        self.status = StreamStatus.INITIALIZING
        self.error_message = ""
        self.frame_count = 0
        self.source_path = str(source) if source is not None else "synthetic"
        self._last_frame_timestamp = time.time()

        self._initialize_stream(source)

    def _initialize_stream(self, source: Optional[Union[int, str, Path]]) -> None:
        """Initialize the requested stream source or activate safe synthetic fallback."""
        # Mode 1: Injected frame sequence
        if len(self.injected_frames) > 0:
            self.source_type = StreamSourceType.INJECTED
            self.status = StreamStatus.ACTIVE
            return

        # Mode 2: Explicit synthetic source
        if source is None or str(source).lower() in ("synthetic", "simulated", "mock"):
            self.source_type = StreamSourceType.SYNTHETIC
            self.status = StreamStatus.ACTIVE
            return

        # Mode 3: Video file source
        source_str = str(source)
        if not source_str.isdigit() and isinstance(source, (str, Path)):
            file_path = Path(source)
            if not file_path.exists():
                err_msg = f"Video file not found: {file_path}"
                if self.fallback_synthetic:
                    print(f"[SafeVideoStream] Warning: {err_msg}. Activating synthetic fallback.")
                    self.is_fallback_active = True
                    self.source_type = StreamSourceType.SYNTHETIC
                    self.status = StreamStatus.ACTIVE
                    return
                else:
                    self.status = StreamStatus.ERROR
                    self.error_message = err_msg
                    raise FileNotFoundError(err_msg)

            self.cap = cv2.VideoCapture(str(file_path))
            if self.cap.isOpened():
                self.source_type = StreamSourceType.FILE
                self.status = StreamStatus.ACTIVE
                # Detect video metadata if available
                file_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                file_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                file_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if file_fps > 0:
                    self.fps_target = int(file_fps)
                return
            else:
                err_msg = f"Failed to open video file codec: {file_path}"
                if self.fallback_synthetic:
                    print(f"[SafeVideoStream] Warning: {err_msg}. Activating synthetic fallback.")
                    self.source_type = StreamSourceType.SYNTHETIC
                    self.status = StreamStatus.ACTIVE
                    return
                else:
                    self.status = StreamStatus.ERROR
                    self.error_message = err_msg
                    raise IOError(err_msg)

        # Mode 4: Camera device index
        if str(source).isdigit() or isinstance(source, int):
            if not self.allow_camera:
                print("[SafeVideoStream] Hardware camera disabled by policy. Using synthetic feed.")
                self.source_type = StreamSourceType.SYNTHETIC
                self.status = StreamStatus.ACTIVE
                return

            device_idx = int(source)
            try:
                self.cap = cv2.VideoCapture(device_idx)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.source_type = StreamSourceType.CAMERA
                    self.status = StreamStatus.ACTIVE
                    return
                else:
                    raise RuntimeError(f"Could not open camera device {device_idx}")
            except Exception as e:
                if self.fallback_synthetic:
                    print(f"[SafeVideoStream] Camera device unavailable ({e}). Activating synthetic feed.")
                    self.source_type = StreamSourceType.SYNTHETIC
                    self.status = StreamStatus.ACTIVE
                    return
                else:
                    self.status = StreamStatus.ERROR
                    self.error_message = str(e)
                    raise

        # Catch-all fallback
        self.source_type = StreamSourceType.SYNTHETIC
        self.status = StreamStatus.ACTIVE

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Acquire the next video frame from the active stream.

        Returns:
            Tuple of:
            - success (bool): True if a valid frame was read.
            - frame (np.ndarray or None): BGR uint8 image frame.
            - metadata (dict): Stream state, frame index, timestamp, and source type.
        """
        now = time.time()
        self.frame_count += 1

        metadata: Dict[str, Any] = {
            "frame_id": self.frame_count,
            "timestamp": now,
            "source_type": self.source_type.value,
            "is_synthetic": self.source_type == StreamSourceType.SYNTHETIC,
            "is_fallback": self.is_fallback_active,
            "is_eos": False,
            "fps_target": self.fps_target,
            "status": self.status.value,
        }

        # Handle StreamStatus.ERROR or CLOSED
        if self.status in (StreamStatus.ERROR, StreamStatus.CLOSED):
            return False, None, metadata

        # Case 1: Injected Frames
        if self.source_type == StreamSourceType.INJECTED:
            if self.injected_index < len(self.injected_frames):
                frame = self.injected_frames[self.injected_index]
                self.injected_index += 1
                if frame is not None and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                    frame = cv2.resize(frame, (self.width, self.height))
                return True, frame, metadata
            elif self.loop and len(self.injected_frames) > 0:
                self.injected_index = 0
                frame = self.injected_frames[self.injected_index]
                self.injected_index += 1
                return True, frame, metadata
            else:
                self.status = StreamStatus.END_OF_STREAM
                metadata["is_eos"] = True
                metadata["status"] = self.status.value
                return False, None, metadata

        # Case 2: Video File or Physical Camera
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return True, frame, metadata
            else:
                # End of file reached
                if self.loop and self.source_type == StreamSourceType.FILE:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_loop, frame_loop = self.cap.read()
                    if ret_loop and frame_loop is not None:
                        if frame_loop.shape[1] != self.width or frame_loop.shape[0] != self.height:
                            frame_loop = cv2.resize(frame_loop, (self.width, self.height))
                        return True, frame_loop, metadata

                self.status = StreamStatus.END_OF_STREAM
                metadata["is_eos"] = True
                metadata["status"] = self.status.value
                return False, None, metadata

        # Case 3: Synthetic Animated Stream
        frame = self._generate_synthetic_frame(self.frame_count)
        return True, frame, metadata

    def _generate_synthetic_frame(self, frame_idx: int) -> np.ndarray:
        """Generate a simulated crop canopy frame with an animated target pest element."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Foliage green background
        frame[:, :] = (34, 110, 34)

        # Synthetic pest feature moving horizontally
        center_x = (frame_idx * 8 + 80) % (self.width - 100)
        center_y = int(self.height / 2 + 30 * np.sin(frame_idx * 0.2))

        # Simulated pest body (dark brown / gray ellipse)
        cv2.ellipse(frame, (center_x, center_y), (25, 12), 15, 0, 360, (20, 50, 70), -1)
        cv2.circle(frame, (center_x + 18, center_y + 4), 6, (10, 30, 45), -1)

        # Label synthetic marker
        cv2.putText(
            frame,
            f"SIMULATED STREAM [Frame: {frame_idx}]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 240, 220),
            2,
        )
        return frame

    def is_active(self) -> bool:
        """Check if stream is active and readable."""
        return self.status == StreamStatus.ACTIVE

    def release(self) -> None:
        """Release underlying camera or video capture handles."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status = StreamStatus.CLOSED

    def __enter__(self) -> "SafeVideoStream":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
