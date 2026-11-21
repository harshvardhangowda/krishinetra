"""
AI-Powered Precision Pest Management - Pest Centering Visual Servoing Controller
File: src/servoing/centering_controller.py

Computes visual error vectors between detected target centroids and the optical
center of the imaging plane. Emits bounded, safe rotational alignment commands.

Coordinates:
- Optical Center: (frame_width / 2, frame_height / 2)
- Normalized Horizontal Error e_x = (c_x - center_x) / center_x in [-1.0, 1.0]
  - e_x > 0: Target is to the RIGHT of optical axis -> ROTATE_RIGHT
  - e_x < 0: Target is to the LEFT of optical axis -> ROTATE_LEFT
  - |e_x| <= dead_zone_x: Target is HORIZONTALLY CENTERED -> STOP
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any, Tuple

from src.motion.motion_commands import (
    MotionDirection,
    MotionCommand,
)
from src.servoing.target_selector import SelectedTarget


class ServoingStatus(str, Enum):
    CENTERED = "CENTERED"
    ALIGNING = "ALIGNING"
    TARGET_LOST = "TARGET_LOST"
    TEMPORARILY_OCCLUDED = "TEMPORARILY_OCCLUDED"
    SAFETY_LOCKED = "SAFETY_LOCKED"
    ERROR = "ERROR"


@dataclass
class AlignmentResult:
    """Outcome of visual error calculation and command formulation."""
    status: ServoingStatus
    error_x: float # Normalized horizontal error [-1.0, +1.0]
    error_y: float # Normalized vertical error [-1.0, +1.0]
    is_centered: bool
    suggested_command: Optional[MotionCommand]
    target_info: Optional[Dict[str, Any]] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "error_x": round(self.error_x, 4),
            "error_y": round(self.error_y, 4),
            "is_centered": self.is_centered,
            "suggested_command": self.suggested_command.to_dict() if self.suggested_command else None,
            "target_info": self.target_info,
            "notes": self.notes,
            "telemetry_type": "SIMULATED",
            "hardware_actuators": "DISABLED",
        }


class PestCenteringController:
    """Proportional visual servoing controller for platform/gimbal alignment."""

    def __init__(
        self,
        dead_zone_x: float = 0.05,
        dead_zone_y: float = 0.05,
        k_p_angular: float = 0.80,
        min_angular_speed_rad_s: float = 0.08,
        max_angular_speed_rad_s: float = 0.80,
        max_linear_speed_m_s: float = 0.25,
        command_duration_sec: float = 0.20,
        allow_predicted_servoing: bool = False,
        max_predicted_servoing_frames: int = 0,
    ):
        self.dead_zone_x = dead_zone_x
        self.dead_zone_y = dead_zone_y
        self.k_p_angular = k_p_angular
        self.min_angular_speed = min_angular_speed_rad_s
        self.max_angular_speed = max_angular_speed_rad_s
        self.max_linear_speed = max_linear_speed_m_s
        self.command_duration = command_duration_sec
        self.allow_predicted_servoing = allow_predicted_servoing
        self.max_predicted_servoing_frames = max_predicted_servoing_frames

        # State tracking
        self.last_direction: Optional[MotionDirection] = None

    def compute_alignment(
        self,
        target: Optional[SelectedTarget],
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> AlignmentResult:
        """
        Evaluate target centroid relative to image center and generate alignment action.
        """
        # Case 1: Target Lost or None
        if target is None or not target.is_valid or getattr(target, "status", "") == "LOST":
            stop_cmd = MotionCommand.create(
                direction=MotionDirection.STOP,
                linear_speed_m_s=0.0,
                angular_speed_rad_s=0.0,
                duration_sec=self.command_duration,
                requested_by="visual_servoing_target_lost",
            )
            return AlignmentResult(
                status=ServoingStatus.TARGET_LOST,
                error_x=0.0,
                error_y=0.0,
                is_centered=False,
                suggested_command=stop_cmd,
                target_info=None,
                notes="No valid pest detection in frame. Commanded STOP.",
            )

        center_x = frame_width / 2.0
        center_y = frame_height / 2.0

        # Calculate normalized errors
        raw_error_x = (target.centroid_x - center_x) / center_x
        raw_error_y = (target.centroid_y - center_y) / center_y

        error_x = max(-1.0, min(1.0, raw_error_x))
        error_y = max(-1.0, min(1.0, raw_error_y))

        target_dict = target.to_dict()

        # Case 2: Target is temporarily occluded/predicted
        is_predicted = getattr(target, "is_predicted", False) or getattr(target, "status", "") == "TEMPORARILY_MISSING"
        consecutive_misses = getattr(target, "consecutive_misses", 0)

        if is_predicted:
            can_serve_predicted = (
                self.allow_predicted_servoing
                and consecutive_misses <= self.max_predicted_servoing_frames
            )
            if not can_serve_predicted:
                stop_cmd = MotionCommand.create(
                    direction=MotionDirection.STOP,
                    linear_speed_m_s=0.0,
                    angular_speed_rad_s=0.0,
                    duration_sec=self.command_duration,
                    requested_by="visual_servoing_occluded_hold",
                )
                self.last_direction = MotionDirection.STOP
                return AlignmentResult(
                    status=ServoingStatus.TEMPORARILY_OCCLUDED,
                    error_x=error_x,
                    error_y=error_y,
                    is_centered=abs(error_x) <= self.dead_zone_x,
                    suggested_command=stop_cmd,
                    target_info=target_dict,
                    notes=f"Target occluded/predicted (missed={consecutive_misses} frames). Commanded safe STOP.",
                )

        # Case 3: Target is within dead-zone (Centered)
        if abs(error_x) <= self.dead_zone_x:
            stop_cmd = MotionCommand.create(
                direction=MotionDirection.STOP,
                linear_speed_m_s=0.0,
                angular_speed_rad_s=0.0,
                duration_sec=self.command_duration,
                requested_by="visual_servoing_centered",
            )
            self.last_direction = MotionDirection.STOP
            return AlignmentResult(
                status=ServoingStatus.CENTERED,
                error_x=error_x,
                error_y=error_y,
                is_centered=True,
                suggested_command=stop_cmd,
                target_info=target_dict,
                notes=f"Target aligned within dead zone (|e_x|={abs(error_x):.3f} <= {self.dead_zone_x}).",
            )

        # Case 4: Target requires angular correction
        if error_x > self.dead_zone_x:
            # Target is to the right -> rotate right (CW)
            direction = MotionDirection.ROTATE_RIGHT
        else:
            # Target is to the left -> rotate left (CCW)
            direction = MotionDirection.ROTATE_LEFT

        # Proportional velocity with deadband and clamp
        raw_omega = self.k_p_angular * abs(error_x)
        omega = max(self.min_angular_speed, min(self.max_angular_speed, raw_omega))

        motion_cmd = MotionCommand.create(
            direction=direction,
            linear_speed_m_s=0.0,
            angular_speed_rad_s=round(omega, 3),
            duration_sec=self.command_duration,
            requested_by="visual_servoing_aligning",
        )
        self.last_direction = direction

        pred_tag = " [PREDICTED]" if is_predicted else ""
        return AlignmentResult(
            status=ServoingStatus.ALIGNING,
            error_x=error_x,
            error_y=error_y,
            is_centered=False,
            suggested_command=motion_cmd,
            target_info=target_dict,
            notes=f"Aligning {direction.value}: e_x={error_x:.3f}, omega={omega:.3f} rad/s{pred_tag}",
        )
