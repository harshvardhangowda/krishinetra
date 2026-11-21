"""
AI-Powered Precision Pest Management - Camera Capture Module
File: src/monitoring/camera.py

Hardware-agnostic camera interface supporting:
- Laptop / USB Webcams (cv2.VideoCapture)
- Pre-recorded video files (for testing/benchmarking)
- Mock / Synthetic video stream (fallback for automated testing)
- Prepared for future Raspberry Pi camera (Picamera2 / libcamera) integration
"""

import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class CameraStream:
    """Manages video frame acquisition with graceful failure handling and resource cleanup."""

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        width: int = 640,
        height: int = 480,
        fps_target: int = 30,
        fallback_synthetic: bool = True,
    ) -> None:
        """
        Initialize the camera stream.

        Args:
            source: Camera device index (int e.g., 0) or path to video file (str/Path).
            width: Desired capture frame width.
            height: Desired capture frame height.
            fps_target: Target frame rate.
            fallback_synthetic: Use generated frames if physical camera is unavailable.
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps_target = fps_target
        self.fallback_synthetic = fallback_synthetic
        self.is_synthetic = False
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._last_frame_time = time.time()

        self._initialize_stream()

    def _initialize_stream(self) -> None:
        """Attempt to open hardware camera or fallback gracefully."""
        try:
            if isinstance(self.source, (str, Path)) and not str(self.source).isdigit():
                # Video file source
                print(f"[CameraStream] Opening video source: {self.source}")
                self.cap = cv2.VideoCapture(str(self.source))
            else:
                # Device index
                device_idx = int(self.source)
                print(f"[CameraStream] Opening physical webcam device [{device_idx}]...")
                self.cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW if cv2.__file__ else cv2.CAP_ANY)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[CameraStream] Hardware camera initialized successfully ({actual_w}x{actual_h}).")
                self.is_synthetic = False
            else:
                raise RuntimeError("Could not open video capture device.")

        except Exception as e:
            if self.fallback_synthetic:
                print(f"[CameraStream] Warning: Physical camera unavailable ({e}). Activating synthetic feed.")
                self.is_synthetic = True
                self.cap = None
            else:
                raise e

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the stream.

        Returns:
            Tuple of (success: bool, frame: np.ndarray in BGR format or None)
        """
        self._frame_count += 1

        if not self.is_synthetic and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return True, frame
            else:
                # End of video file or device disconnect
                return False, None

        # Synthetic fallback frame (generates moving foliage pattern for testing)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (34, 110, 34)  # Foliage green

        # Draw a moving reference element to simulate motion
        offset = (self._frame_count * 5) % self.width
        cv2.circle(frame, (offset, self.height // 2), 25, (20, 80, 220), -1)
        cv2.putText(
            frame,
            f"SYNTHETIC CAMERA FEED (Frame: {self._frame_count})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Enforce target FPS pacing
        elapsed = time.time() - self._last_frame_time
        sleep_time = (1.0 / self.fps_target) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_frame_time = time.time()

        return True, frame

    def is_opened(self) -> bool:
        """Check if camera stream is active."""
        if self.is_synthetic:
            return True
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Release underlying camera hardware resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[CameraStream] Camera resources released cleanly.")
