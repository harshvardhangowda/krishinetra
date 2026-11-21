"""
AI-Powered Precision Pest Management - Rover Motion Commands & Data Models
File: src/motion/motion_commands.py

Defines directional movement commands, drive configurations, and structured
motion packets for the Krishnetra mobile platform.
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List


class MotionDirection(str, Enum):
    """Allowed movement directions for rover navigation."""
    FORWARD = "FORWARD"
    REVERSE = "REVERSE"
    STRAFE_LEFT = "STRAFE_LEFT"
    STRAFE_RIGHT = "STRAFE_RIGHT"
    ROTATE_LEFT = "ROTATE_LEFT"
    ROTATE_RIGHT = "ROTATE_RIGHT"
    STOP = "STOP"


class DriveConfiguration(str, Enum):
    """Supported rover drivetrain configurations."""
    MECANUM = "MECANUM"
    DIFFERENTIAL = "DIFFERENTIAL"


class MotionSafetyState(str, Enum):
    """Rover motion controller safety states."""
    LOCKED = "LOCKED"
    STOPPED = "STOPPED"
    MOVING = "MOVING"
    E_STOPPED = "E_STOPPED"
    TIMEOUT_STOPPED = "TIMEOUT_STOPPED"


@dataclass
class MotionCommand:
    """Represents an autonomous or manual movement instruction."""
    command_id: str
    direction: MotionDirection
    linear_speed_m_s: float
    angular_speed_rad_s: float
    duration_sec: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    requested_by: str = "edge_ai_navigation"
    simulation_mode: bool = True

    @classmethod
    def create(
        cls,
        direction: MotionDirection,
        linear_speed_m_s: float = 0.20,
        angular_speed_rad_s: float = 0.0,
        duration_sec: float = 1.0,
        requested_by: str = "edge_ai_navigation",
    ) -> "MotionCommand":
        return cls(
            command_id=str(uuid.uuid4()),
            direction=direction,
            linear_speed_m_s=float(linear_speed_m_s),
            angular_speed_rad_s=float(angular_speed_rad_s),
            duration_sec=float(duration_sec),
            timestamp=datetime.now(timezone.utc).isoformat(),
            requested_by=requested_by,
            simulation_mode=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["direction"] = self.direction.value
        return d


@dataclass
class WheelVelocities:
    """Theoretical wheel linear speeds (m/s) computed by kinematics."""
    drive_type: DriveConfiguration
    timestamp: str
    # Mecanum 4-wheel velocities: front_left, front_right, rear_left, rear_right
    front_left_m_s: float = 0.0
    front_right_m_s: float = 0.0
    rear_left_m_s: float = 0.0
    rear_right_m_s: float = 0.0
    # Differential 2-wheel velocities
    left_m_s: float = 0.0
    right_m_s: float = 0.0
    simulation_mode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["drive_type"] = self.drive_type.value
        return d
