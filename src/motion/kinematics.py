"""
AI-Powered Precision Pest Management - Rover Kinematics Engine
File: src/motion/kinematics.py

Provides mathematical forward and inverse kinematics for:
1. 4-Wheel Mecanum Omnidirectional Drivetrain
2. 2-Wheel Differential Drivetrain

CRITICAL SAFETY:
This module performs theoretical mathematical vector calculations ONLY.
It does NOT drive physical motor hardware, PWM pins, or H-bridges.
"""

from typing import Tuple
from datetime import datetime, timezone
from src.motion.motion_commands import (
    MotionDirection,
    DriveConfiguration,
    WheelVelocities,
)


class MecanumKinematics:
    """
    Inverse and Forward Kinematics for a 4-wheel Mecanum rover.
    
    Coordinate convention (Right-Handed):
    - +X: Forward
    - +Y: Left (Strafing)
    - +Z / +Omega: Counter-Clockwise (Rotate Left)
    """

    def __init__(
        self,
        wheel_base_meters: float = 0.250,
        track_width_meters: float = 0.280,
        wheel_radius_meters: float = 0.040,
    ):
        self.wheel_base = wheel_base_meters
        self.track_width = track_width_meters
        self.wheel_radius = wheel_radius_meters
        # Geometric lever arm for rotational moments
        self.k_geometry = (self.wheel_base + self.track_width) / 2.0

    def compute_wheel_velocities(
        self,
        v_x: float,
        v_y: float,
        omega_z: float,
    ) -> WheelVelocities:
        """
        Inverse Kinematics: Map chassis velocities (vx, vy, omega) to 4 wheel speeds (m/s).
        
        Wheel layout:
        FL (Front-Left)   [ \\ ]   [ / ]  FR (Front-Right)
        RL (Rear-Left)    [ / ]   [ \\ ]  RR (Rear-Right)
        """
        v_fl = v_x - v_y - (self.k_geometry * omega_z)
        v_fr = v_x + v_y + (self.k_geometry * omega_z)
        v_rl = v_x + v_y - (self.k_geometry * omega_z)
        v_rr = v_x - v_y + (self.k_geometry * omega_z)

        return WheelVelocities(
            drive_type=DriveConfiguration.MECANUM,
            timestamp=datetime.now(timezone.utc).isoformat(),
            front_left_m_s=round(v_fl, 4),
            front_right_m_s=round(v_fr, 4),
            rear_left_m_s=round(v_rl, 4),
            rear_right_m_s=round(v_rr, 4),
            simulation_mode=True,
        )

    def compute_chassis_velocities(
        self,
        v_fl: float,
        v_fr: float,
        v_rl: float,
        v_rr: float,
    ) -> Tuple[float, float, float]:
        """
        Forward Kinematics: Map 4 wheel speeds (m/s) to chassis velocities (vx, vy, omega_z).
        """
        v_x = (v_fl + v_fr + v_rl + v_rr) / 4.0
        v_y = (-v_fl + v_fr + v_rl - v_rr) / 4.0
        omega_z = (-v_fl + v_fr - v_rl + v_rr) / (4.0 * self.k_geometry)
        return round(v_x, 4), round(v_y, 4), round(omega_z, 4)

    def direction_to_chassis_velocity(
        self,
        direction: MotionDirection,
        linear_speed: float,
        angular_speed: float,
    ) -> Tuple[float, float, float]:
        """Convert directional motion command to (vx, vy, omega_z)."""
        if direction == MotionDirection.STOP:
            return 0.0, 0.0, 0.0
        elif direction == MotionDirection.FORWARD:
            return abs(linear_speed), 0.0, 0.0
        elif direction == MotionDirection.REVERSE:
            return -abs(linear_speed), 0.0, 0.0
        elif direction == MotionDirection.STRAFE_LEFT:
            return 0.0, abs(linear_speed), 0.0
        elif direction == MotionDirection.STRAFE_RIGHT:
            return 0.0, -abs(linear_speed), 0.0
        elif direction == MotionDirection.ROTATE_LEFT:
            return 0.0, 0.0, abs(angular_speed)
        elif direction == MotionDirection.ROTATE_RIGHT:
            return 0.0, 0.0, -abs(angular_speed)
        else:
            return 0.0, 0.0, 0.0


class DifferentialKinematics:
    """
    Inverse and Forward Kinematics for a 2-wheel Differential Drive rover.
    Note: Strafing (v_y != 0) is physically not possible in differential drive.
    """

    def __init__(
        self,
        track_width_meters: float = 0.280,
        wheel_radius_meters: float = 0.040,
    ):
        self.track_width = track_width_meters
        self.wheel_radius = wheel_radius_meters

    def compute_wheel_velocities(
        self,
        v_x: float,
        omega_z: float,
    ) -> WheelVelocities:
        """
        Inverse Kinematics: Map chassis velocities (vx, omega) to left and right wheel speeds.
        """
        half_width = self.track_width / 2.0
        v_left = v_x - (half_width * omega_z)
        v_right = v_x + (half_width * omega_z)

        return WheelVelocities(
            drive_type=DriveConfiguration.DIFFERENTIAL,
            timestamp=datetime.now(timezone.utc).isoformat(),
            left_m_s=round(v_left, 4),
            right_m_s=round(v_right, 4),
            simulation_mode=True,
        )

    def compute_chassis_velocities(
        self,
        v_left: float,
        v_right: float,
    ) -> Tuple[float, float]:
        """
        Forward Kinematics: Map wheel speeds to chassis velocity (vx, omega_z).
        """
        v_x = (v_right + v_left) / 2.0
        omega_z = (v_right - v_left) / self.track_width
        return round(v_x, 4), round(omega_z, 4)

    def direction_to_chassis_velocity(
        self,
        direction: MotionDirection,
        linear_speed: float,
        angular_speed: float,
    ) -> Tuple[float, float]:
        """Convert directional motion command to (vx, omega_z). Strafing raises ValueError."""
        if direction in (MotionDirection.STRAFE_LEFT, MotionDirection.STRAFE_RIGHT):
            raise ValueError(f"Strafing is not supported by {DriveConfiguration.DIFFERENTIAL.value} drive.")

        if direction == MotionDirection.STOP:
            return 0.0, 0.0
        elif direction == MotionDirection.FORWARD:
            return abs(linear_speed), 0.0
        elif direction == MotionDirection.REVERSE:
            return -abs(linear_speed), 0.0
        elif direction == MotionDirection.ROTATE_LEFT:
            return 0.0, abs(angular_speed)
        elif direction == MotionDirection.ROTATE_RIGHT:
            return 0.0, -abs(angular_speed)
        else:
            return 0.0, 0.0
