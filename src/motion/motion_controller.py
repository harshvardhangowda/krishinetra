"""
AI-Powered Precision Pest Management - Safe Rover Motion Controller
File: src/motion/motion_controller.py

Coordinates rover locomotion commands under strict safety interlocks.
Enforces:
1. Default state: LOCKED and STOPPED.
2. Mandatory software simulation mode.
3. Speed limit enforcement (linear and angular ceilings).
4. Command timeout watchdog (auto-stop upon duration expiry).
5. Emergency-stop (E-STOP) handling.
6. Fail-safe state transitions upon communication or heartbeat interruption.
7. Physical actuator isolation (zero motor/PWM driver calls).
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any

from src.motion.motion_commands import (
    MotionDirection,
    DriveConfiguration,
    MotionSafetyState,
    MotionCommand,
    WheelVelocities,
)
from src.motion.kinematics import MecanumKinematics, DifferentialKinematics

logger = logging.getLogger("KrishnetraMotionController")


class SafeMotionController:
    """Central motion safety and command coordinator."""

    def __init__(
        self,
        drive_config: DriveConfiguration = DriveConfiguration.MECANUM,
        simulation_mode: bool = True,
        max_linear_speed_m_s: float = 0.50,
        max_angular_speed_rad_s: float = 1.20,
        command_timeout_sec: float = 1.50,
    ):
        if not simulation_mode:
            raise ValueError(
                "CRITICAL SAFETY VIOLATION: Physical actuator operation is prohibited in Phase 13. "
                "simulation_mode must be set to True."
            )

        self.drive_config = drive_config
        self.simulation_mode = simulation_mode
        self.max_linear_speed = max_linear_speed_m_s
        self.max_angular_speed = max_angular_speed_rad_s
        self.command_timeout_sec = command_timeout_sec

        # Default state: STRICTLY LOCKED and STOPPED
        self.state: MotionSafetyState = MotionSafetyState.LOCKED
        self.current_command: Optional[MotionCommand] = None
        self.last_command_time: float = 0.0
        self.active_wheel_velocities: Optional[WheelVelocities] = None

        # Kinematics models
        self.mecanum_kinematics = MecanumKinematics()
        self.diff_kinematics = DifferentialKinematics()

        logger.info(
            f"[MotionController] Initialized in {self.drive_config.value} mode. "
            f"State: {self.state.value} | Simulation: {self.simulation_mode}"
        )

    def unlock_for_simulation(self, authorization_token: str = "SIMULATION_SAFE") -> bool:
        """Unlock the controller for software motion simulation."""
        if not self.simulation_mode:
            return False
        if authorization_token == "SIMULATION_SAFE":
            self.state = MotionSafetyState.STOPPED
            logger.info("[MotionController] Unlocked for safe software simulation.")
            return True
        return False

    def lock(self, reason: str = "MANUAL_LOCK") -> None:
        """Lock controller into fail-safe state, stopping all movement."""
        self.state = MotionSafetyState.LOCKED
        self.stop(reason=reason)
        logger.warning(f"[MotionController] LOCKED. Reason: {reason}")

    def emergency_stop(self) -> WheelVelocities:
        """Immediate hard emergency stop (E-STOP). Reverts to LOCKED."""
        self.state = MotionSafetyState.E_STOPPED
        zero_vel = self._generate_zero_velocities()
        self.active_wheel_velocities = zero_vel
        self.current_command = None
        logger.critical("[MotionController] EMERGENCY STOP TRIGGERED. All motion halted.")
        return zero_vel

    def stop(self, reason: str = "NORMAL_STOP") -> WheelVelocities:
        """Normal controlled stop."""
        if self.state not in (MotionSafetyState.LOCKED, MotionSafetyState.E_STOPPED):
            self.state = MotionSafetyState.STOPPED
        zero_vel = self._generate_zero_velocities()
        self.active_wheel_velocities = zero_vel
        self.current_command = None
        return zero_vel

    def handle_heartbeat_loss(self) -> None:
        """Invoked when communication link / heartbeat is interrupted."""
        logger.error("[MotionController] Heartbeat loss detected. Forcing E-STOP & LOCK.")
        self.emergency_stop()
        self.state = MotionSafetyState.LOCKED

    def request_motion(
        self,
        cmd: MotionCommand,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], Optional[WheelVelocities]]:
        """
        Evaluate and dispatch a requested motion command.
        
        Returns:
            (is_accepted: bool, rejection_reason: Optional[str], wheel_velocities: Optional[WheelVelocities])
        """
        now = current_time if current_time is not None else time.time()

        # Rule 1: Check safety lock
        if self.state == MotionSafetyState.LOCKED:
            return False, "SAFETY_LOCKED: Controller is locked. Unlock required.", None
        if self.state == MotionSafetyState.E_STOPPED:
            return False, "E_STOP_ACTIVE: Controller in emergency stop. Clear and relock required.", None

        # Rule 2: Validate direction
        if not isinstance(cmd.direction, MotionDirection):
            return False, f"INVALID_DIRECTION: Unknown movement direction '{cmd.direction}'", None

        # Rule 3: Stop command handling
        if cmd.direction == MotionDirection.STOP:
            zero_vel = self.stop(reason="COMMAND_STOP")
            return True, None, zero_vel

        # Rule 4: Validate speeds and limits
        if cmd.linear_speed_m_s < 0.0 or cmd.linear_speed_m_s > self.max_linear_speed:
            return (
                False,
                f"SPEED_LIMIT_EXCEEDED: Linear speed {cmd.linear_speed_m_s:.2f} m/s exceeds max {self.max_linear_speed:.2f} m/s",
                None,
            )

        if cmd.angular_speed_rad_s < 0.0 or cmd.angular_speed_rad_s > self.max_angular_speed:
            return (
                False,
                f"SPEED_LIMIT_EXCEEDED: Angular speed {cmd.angular_speed_rad_s:.2f} rad/s exceeds max {self.max_angular_speed:.2f} rad/s",
                None,
            )

        if cmd.duration_sec <= 0.0:
            return False, f"INVALID_DURATION: Duration must be > 0 (got {cmd.duration_sec})", None

        # Rule 5: Drive configuration capability check (e.g. differential cannot strafe)
        if self.drive_config == DriveConfiguration.DIFFERENTIAL:
            if cmd.direction in (MotionDirection.STRAFE_LEFT, MotionDirection.STRAFE_RIGHT):
                return False, "UNSUPPORTED_MOTION: Differential drive cannot execute lateral strafing.", None

        # Rule 6: Compute theoretical kinematics
        try:
            if self.drive_config == DriveConfiguration.MECANUM:
                vx, vy, omega = self.mecanum_kinematics.direction_to_chassis_velocity(
                    cmd.direction, cmd.linear_speed_m_s, cmd.angular_speed_rad_s
                )
                wheel_vel = self.mecanum_kinematics.compute_wheel_velocities(vx, vy, omega)
            else:
                vx, omega = self.diff_kinematics.direction_to_chassis_velocity(
                    cmd.direction, cmd.linear_speed_m_s, cmd.angular_speed_rad_s
                )
                wheel_vel = self.diff_kinematics.compute_wheel_velocities(vx, omega)
        except Exception as e:
            return False, f"KINEMATICS_ERROR: {str(e)}", None

        # Update controller state
        self.state = MotionSafetyState.MOVING
        self.current_command = cmd
        self.last_command_time = now
        self.active_wheel_velocities = wheel_vel

        return True, None, wheel_vel

    def update(self, current_time: Optional[float] = None) -> MotionSafetyState:
        """Periodic watchdog check to enforce command duration and timeouts."""
        now = current_time if current_time is not None else time.time()

        if self.state == MotionSafetyState.MOVING and self.current_command is not None:
            elapsed = now - self.last_command_time
            # Check if duration expired or watchdog timeout exceeded
            effective_timeout = min(self.current_command.duration_sec, self.command_timeout_sec)
            if elapsed >= effective_timeout:
                logger.info(f"[MotionController] Motion command duration elapsed ({elapsed:.2f}s). Halting motion.")
                self.stop(reason="DURATION_ELAPSED")
                self.state = MotionSafetyState.TIMEOUT_STOPPED

        return self.state

    def get_status_telemetry(self) -> Dict[str, Any]:
        """Generate structured status telemetry."""
        return {
            "telemetry_type": "SIMULATED",
            "drive_configuration": self.drive_config.value,
            "safety_state": self.state.value,
            "simulation_mode": self.simulation_mode,
            "active_command": self.current_command.to_dict() if self.current_command else None,
            "wheel_velocities": self.active_wheel_velocities.to_dict() if self.active_wheel_velocities else None,
            "limits": {
                "max_linear_speed_m_s": self.max_linear_speed,
                "max_angular_speed_rad_s": self.max_angular_speed,
                "command_timeout_sec": self.command_timeout_sec,
            },
            "hardware_actuators": "DISABLED",
        }

    def _generate_zero_velocities(self) -> WheelVelocities:
        if self.drive_config == DriveConfiguration.MECANUM:
            return self.mecanum_kinematics.compute_wheel_velocities(0.0, 0.0, 0.0)
        else:
            return self.diff_kinematics.compute_wheel_velocities(0.0, 0.0)
