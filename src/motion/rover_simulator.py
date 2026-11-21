"""
AI-Powered Precision Pest Management - Rover Motion & Odometry Simulator
File: src/motion/rover_simulator.py

Simulates rover kinematics, dead-reckoning odometry, and trajectory evolution
in software without connecting to physical motors or GPIO pins.

Outputs clearly labeled SIMULATED telemetry.
"""

import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from src.motion.motion_commands import (
    MotionDirection,
    DriveConfiguration,
    MotionCommand,
    WheelVelocities,
)
from src.motion.motion_controller import SafeMotionController


@dataclass
class RoverPose:
    """Represents simulated planar pose in 2D coordinates."""
    x_meters: float = 0.0
    y_meters: float = 0.0
    heading_radians: float = 0.0 # 0.0 along +X axis

    @property
    def heading_degrees(self) -> float:
        return math.degrees(self.heading_radians) % 360.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_meters": round(self.x_meters, 4),
            "y_meters": round(self.y_meters, 4),
            "heading_radians": round(self.heading_radians, 4),
            "heading_degrees": round(self.heading_degrees, 2),
        }


class RoverSimulator:
    """Simulates rover movement, odometry dead-reckoning, and telemetry."""

    def __init__(
        self,
        controller: Optional[SafeMotionController] = None,
        initial_pose: Optional[RoverPose] = None,
    ):
        self.controller = controller or SafeMotionController()
        self.pose = initial_pose or RoverPose()
        self.trajectory_history: List[Dict[str, Any]] = []
        self.last_sim_time = time.time()

        # Record initial position
        self._record_trajectory_point(self.last_sim_time, 0.0, 0.0, 0.0)

    def step(self, dt_seconds: float, current_time: Optional[float] = None) -> RoverPose:
        """
        Advance simulation by dt_seconds based on current controller state.
        
        Args:
            dt_seconds: Time step in seconds.
            current_time: Optional epoch timestamp for reproducible testing.
        """
        now = current_time if current_time is not None else time.time()

        # Update controller watchdog
        self.controller.update(current_time=now)

        # Get active wheel velocities
        wheel_vel = self.controller.active_wheel_velocities
        v_x = 0.0
        v_y = 0.0
        omega = 0.0

        if wheel_vel is not None and self.controller.state.value == "MOVING":
            if self.controller.drive_config == DriveConfiguration.MECANUM:
                v_x, v_y, omega = self.controller.mecanum_kinematics.compute_chassis_velocities(
                    wheel_vel.front_left_m_s,
                    wheel_vel.front_right_m_s,
                    wheel_vel.rear_left_m_s,
                    wheel_vel.rear_right_m_s,
                )
            else:
                v_x, omega = self.controller.diff_kinematics.compute_chassis_velocities(
                    wheel_vel.left_m_s,
                    wheel_vel.right_m_s,
                )

        # Integrate pose in world frame
        theta = self.pose.heading_radians
        # Delta in robot body frame
        dx_body = v_x * dt_seconds
        dy_body = v_y * dt_seconds
        dtheta = omega * dt_seconds

        # Rotate to world frame
        dx_world = (dx_body * math.cos(theta)) - (dy_body * math.sin(theta))
        dy_world = (dx_body * math.sin(theta)) + (dy_body * math.cos(theta))

        self.pose.x_meters += dx_world
        self.pose.y_meters += dy_world
        self.pose.heading_radians = (self.pose.heading_radians + dtheta) % (2.0 * math.pi)

        self._record_trajectory_point(now, v_x, v_y, omega)
        self.last_sim_time = now

        return self.pose

    def get_telemetry(self) -> Dict[str, Any]:
        """Generate structured simulation telemetry packet."""
        return {
            "telemetry_type": "SIMULATED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pose": self.pose.to_dict(),
            "controller_status": self.controller.get_status_telemetry(),
            "trajectory_points_count": len(self.trajectory_history),
            "hardware_actuators": "DISABLED",
        }

    def _record_trajectory_point(self, timestamp: float, vx: float, vy: float, omega: float) -> None:
        point = {
            "timestamp": timestamp,
            "x": round(self.pose.x_meters, 4),
            "y": round(self.pose.y_meters, 4),
            "heading_rad": round(self.pose.heading_radians, 4),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "omega": round(omega, 4),
            "safety_state": self.controller.state.value,
        }
        self.trajectory_history.append(point)
