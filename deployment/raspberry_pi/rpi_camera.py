"""
AI-Powered Precision Pest Management - Raspberry Pi Camera Module
File: deployment/raspberry_pi/rpi_camera.py

Handles frame acquisition on Raspberry Pi 4 supporting:
1. Native Picamera2 / libcamera (Pi Camera Module v2/v3/HQ)
2. Standard V4L2 / USB webcam via OpenCV VideoCapture
3. Synthetic test feed fallback for CI/CD and offline verification
"""

import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Optional Picamera2 import for Raspberry Pi OS
try:
    from picam2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class RPiCameraStream:
    """Acquires camera frames on Raspberry Pi 4 with automated hardware detection."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        use_picam2: bool = False,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.use_picam2 = use_picam2 and PICAMERA2_AVAILABLE

        self.picam2: Optional[Any] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_synthetic = False
        self._frame_count = 0

        self._init_camera()

    def _init_camera(self) -> None:
        """Initialize Picamera2, USB VideoCapture, or synthetic fallback."""
        if self.use_picam2:
            try:
                print("[RPiCamera] Initializing Picamera2 / libcamera pipeline...")
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                print("[RPiCamera] Picamera2 stream started.")
                return
            except Exception as e:
                print(f"[RPiCamera] Picamera2 initialization failed ({e}). Falling back to OpenCV...")
                self.picam2 = None

        try:
            device_idx = int(self.source) if str(self.source).isdigit() else self.source
            print(f"[RPiCamera] Opening VideoCapture [{device_idx}]...")
            self.cap = cv2.VideoCapture(device_idx)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                print(f"[RPiCamera] VideoCapture ready ({self.width}x{self.height} @ {self.fps}FPS).")
            else:
                raise RuntimeError("Could not open VideoCapture device.")
        except Exception as e:
            print(f"[RPiCamera] Hardware camera unavailable ({e}). Activating synthetic feed.")
            self.is_synthetic = True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture next BGR frame."""
        self._frame_count += 1

        if self.picam2 is not None:
            try:
                rgb_frame = self.picam2.capture_array()
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                return True, bgr_frame
            except Exception:
                return False, None

        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return True, frame
            return False, None

        # Synthetic fallback frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (34, 110, 34)
        offset = (self._frame_count * 5) % self.width
        cv2.circle(frame, (offset, self.height // 2), 25, (20, 80, 220), -1)
        cv2.putText(
            frame,
            f"RPI SYNTHETIC FEED (Frame {self._frame_count})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        time.sleep(1.0 / self.fps)
        return True, frame

    def is_opened(self) -> bool:
        return self.is_synthetic or (self.picam2 is not None) or (self.cap is not None and self.cap.isOpened())

    def release(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[RPiCamera] Camera stream released.")
