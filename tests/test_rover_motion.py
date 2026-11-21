"""
Unit and Integration Tests for Phase 13: Rover Motion & Safe Simulation
File: tests/test_rover_motion.py

Validates:
1. Motion directions (Forward, Reverse, Strafe Left, Strafe Right, Rotate Left, Rotate Right, Stop)
2. Mecanum and Differential drive kinematics calculations
3. Safety lock enforcement (commands rejected when locked)
4. Speed limits enforcement (linear and angular speed ceilings)
5. Invalid command rejection (negative speeds, zero/negative duration, invalid direction)
6. Command timeout watchdog behavior
7. Emergency stop (E-STOP) handling
8. Heartbeat loss fail-safe stop
9. Differential drive rejection of lateral strafing
10. RoverSimulator dead-reckoning odometry and telemetry labeling
"""

import math
import unittest
from src.motion.motion_commands import (
    MotionDirection,
    DriveConfiguration,
    MotionSafetyState,
    MotionCommand,
    WheelVelocities,
)
from src.motion.kinematics import MecanumKinematics, DifferentialKinematics
from src.motion.motion_controller import SafeMotionController
from src.motion.rover_simulator import RoverPose, RoverSimulator


class TestRoverMotionArchitecture(unittest.TestCase):

    def setUp(self):
        self.mecanum_controller = SafeMotionController(
            drive_config=DriveConfiguration.MECANUM,
            simulation_mode=True,
            max_linear_speed_m_s=0.50,
            max_angular_speed_rad_s=1.20,
            command_timeout_sec=1.50,
        )
        self.diff_controller = SafeMotionController(
            drive_config=DriveConfiguration.DIFFERENTIAL,
            simulation_mode=True,
            max_linear_speed_m_s=0.50,
            max_angular_speed_rad_s=1.20,
            command_timeout_sec=1.50,
        )

    # 1. Default State: LOCKED & STOPPED
    def test_default_state_is_locked(self):
        self.assertEqual(self.mecanum_controller.state, MotionSafetyState.LOCKED)
        self.assertEqual(self.diff_controller.state, MotionSafetyState.LOCKED)

        # Commands must be rejected when locked
        cmd = MotionCommand.create(direction=MotionDirection.FORWARD, linear_speed_m_s=0.2)
        accepted, reason, vel = self.mecanum_controller.request_motion(cmd)
        self.assertFalse(accepted)
        self.assertIn("SAFETY_LOCKED", reason)
        self.assertIsNone(vel)

    # 2. Unlocking for Software Simulation
    def test_unlock_for_simulation(self):
        unlocked = self.mecanum_controller.unlock_for_simulation("SIMULATION_SAFE")
        self.assertTrue(unlocked)
        self.assertEqual(self.mecanum_controller.state, MotionSafetyState.STOPPED)

    # 3. Mecanum Kinematics - Forward, Reverse, Strafe, Rotate
    def test_mecanum_movement_directions(self):
        self.mecanum_controller.unlock_for_simulation()

        # Forward
        fwd_cmd = MotionCommand.create(MotionDirection.FORWARD, linear_speed_m_s=0.30)
        accepted, _, vel = self.mecanum_controller.request_motion(fwd_cmd)
        self.assertTrue(accepted)
        self.assertEqual(self.mecanum_controller.state, MotionSafetyState.MOVING)
        self.assertGreater(vel.front_left_m_s, 0.0)
        self.assertGreater(vel.front_right_m_s, 0.0)

        # Reverse
        rev_cmd = MotionCommand.create(MotionDirection.REVERSE, linear_speed_m_s=0.30)
        accepted, _, vel = self.mecanum_controller.request_motion(rev_cmd)
        self.assertTrue(accepted)
        self.assertLess(vel.front_left_m_s, 0.0)
        self.assertLess(vel.front_right_m_s, 0.0)

        # Strafe Left (FL < 0, FR > 0 in standard mecanum inverse kinematics)
        strafe_l_cmd = MotionCommand.create(MotionDirection.STRAFE_LEFT, linear_speed_m_s=0.30)
        accepted, _, vel = self.mecanum_controller.request_motion(strafe_l_cmd)
        self.assertTrue(accepted)
        self.assertLess(vel.front_left_m_s, 0.0)
        self.assertGreater(vel.front_right_m_s, 0.0)

        # Strafe Right
        strafe_r_cmd = MotionCommand.create(MotionDirection.STRAFE_RIGHT, linear_speed_m_s=0.30)
        accepted, _, vel = self.mecanum_controller.request_motion(strafe_r_cmd)
        self.assertTrue(accepted)
        self.assertGreater(vel.front_left_m_s, 0.0)
        self.assertLess(vel.front_right_m_s, 0.0)

        # Rotate Left
        rot_l_cmd = MotionCommand.create(MotionDirection.ROTATE_LEFT, angular_speed_rad_s=0.80)
        accepted, _, vel = self.mecanum_controller.request_motion(rot_l_cmd)
        self.assertTrue(accepted)
        self.assertLess(vel.front_left_m_s, 0.0)
        self.assertGreater(vel.front_right_m_s, 0.0)

        # Rotate Right
        rot_r_cmd = MotionCommand.create(MotionDirection.ROTATE_RIGHT, angular_speed_rad_s=0.80)
        accepted, _, vel = self.mecanum_controller.request_motion(rot_r_cmd)
        self.assertTrue(accepted)
        self.assertGreater(vel.front_left_m_s, 0.0)
        self.assertLess(vel.front_right_m_s, 0.0)

        # Stop
        stop_cmd = MotionCommand.create(MotionDirection.STOP)
        accepted, _, vel = self.mecanum_controller.request_motion(stop_cmd)
        self.assertTrue(accepted)
        self.assertEqual(self.mecanum_controller.state, MotionSafetyState.STOPPED)
        self.assertEqual(vel.front_left_m_s, 0.0)
        self.assertEqual(vel.front_right_m_s, 0.0)

    # 4. Differential Drive Kinematics & Strafing Rejection
    def test_differential_drive_and_strafing_rejection(self):
        self.diff_controller.unlock_for_simulation()

        # Forward
        fwd_cmd = MotionCommand.create(MotionDirection.FORWARD, linear_speed_m_s=0.25)
        accepted, _, vel = self.diff_controller.request_motion(fwd_cmd)
        self.assertTrue(accepted)
        self.assertAlmostEqual(vel.left_m_s, 0.25, places=3)
        self.assertAlmostEqual(vel.right_m_s, 0.25, places=3)

        # Rotate Left
        rot_cmd = MotionCommand.create(MotionDirection.ROTATE_LEFT, angular_speed_rad_s=0.50)
        accepted, _, vel = self.diff_controller.request_motion(rot_cmd)
        self.assertTrue(accepted)
        self.assertLess(vel.left_m_s, 0.0)
        self.assertGreater(vel.right_m_s, 0.0)

        # Strafing Left MUST be rejected in differential configuration
        strafe_cmd = MotionCommand.create(MotionDirection.STRAFE_LEFT, linear_speed_m_s=0.20)
        accepted, reason, vel = self.diff_controller.request_motion(strafe_cmd)
        self.assertFalse(accepted)
        self.assertIn("UNSUPPORTED_MOTION", reason)
        self.assertIsNone(vel)

    # 5. Speed Limit Enforcements
    def test_speed_limit_enforcement(self):
        self.mecanum_controller.unlock_for_simulation()

        # Exceed linear speed ceiling (max is 0.50)
        fast_cmd = MotionCommand.create(MotionDirection.FORWARD, linear_speed_m_s=0.85)
        accepted, reason, _ = self.mecanum_controller.request_motion(fast_cmd)
        self.assertFalse(accepted)
        self.assertIn("SPEED_LIMIT_EXCEEDED", reason)

        # Exceed angular speed ceiling (max is 1.20)
        spin_cmd = MotionCommand.create(MotionDirection.ROTATE_LEFT, angular_speed_rad_s=2.50)
        accepted, reason, _ = self.mecanum_controller.request_motion(spin_cmd)
        self.assertFalse(accepted)
        self.assertIn("SPEED_LIMIT_EXCEEDED", reason)

    # 6. Invalid Commands Rejection
    def test_invalid_command_rejection(self):
        self.mecanum_controller.unlock_for_simulation()

        # Negative linear speed
        neg_cmd = MotionCommand.create(MotionDirection.FORWARD, linear_speed_m_s=-0.20)
        accepted, reason, _ = self.mecanum_controller.request_motion(neg_cmd)
        self.assertFalse(accepted)
        self.assertIn("SPEED_LIMIT_EXCEEDED", reason)

        # Non-positive duration
        zero_dur_cmd = MotionCommand.create(MotionDirection.FORWARD, linear_speed_m_s=0.20, duration_sec=0.0)
        accepted, reason, _ = self.mecanum_controller.request_motion(zero_dur_cmd)
        self.assertFalse(accepted)
        self.assertIn("INVALID_DURATION", reason)

    # 7. Command Timeout Watchdog
    def test_command_timeout_watchdog(self):
        self.mecanum_controller.unlock_for_simulation()

        t0 = 100.0
        cmd = MotionCommand.create(MotionDirection.FORWARD, linear_speed_m_s=0.20, duration_sec=1.0)
        accepted, _, vel = self.mecanum_controller.request_motion(cmd, current_time=t0)
        self.assertTrue(accepted)
        self.assertEqual(self.mecanum_controller.state, MotionSafetyState.MOVING)

        # Step time forward by 0.5s (still within 1.0s duration)
        state = self.mecanum_controller.update(current_time=t0 + 0.5)
        self.assertEqual(state, MotionSafetyState.MOVING)

        # Step time past duration (1.1s later) -> must auto-stop
        state = self.mecanum_controller.update(current_time=t0 + 1.1)
        self.assertEqual(state, MotionSafetyState.TIMEOUT_STOPPED)
        self.assertEqual(self.mecanum_controller.active_wheel_velocities.front_left_m_s, 0.0)

    # 8. Emergency Stop (E-STOP) Handling
    def test_emergency_stop_handling(self):
        self.mecanum_controller.unlock_for_simulation()

        cmd = MotionCommand.create(MotionDirection.FORWARD, linear_speed_m_s=0.30)
        self.mecanum_controller.request_motion(cmd)
        self.assertEqual(self.mecanum_controller.state, MotionSafetyState.MOVING)

        # Trigger Emergency Stop
        zero_vel = self.mecanum_controller.emergency_stop()
        self.assertEqual(self.mecanum_controller.state, MotionSafetyState.E_STOPPED)
        self.assertEqual(zero_vel.front_left_m_s, 0.0)

        # Further commands must be rejected
        accepted, reason, _ = self.mecanum_controller.request_motion(cmd)
        self.assertFalse(accepted)
        self.assertIn("E_STOP_ACTIVE", reason)

    # 9. Heartbeat Loss Fail-Safe
    def test_heartbeat_loss_fail_safe(self):
        self.mecanum_controller.unlock_for_simulation()
        cmd = MotionCommand.create(MotionDirection.FORWARD, linear_speed_m_s=0.30)
        self.mecanum_controller.request_motion(cmd)
        self.assertEqual(self.mecanum_controller.state, MotionSafetyState.MOVING)

        # Simulate heartbeat loss
        self.mecanum_controller.handle_heartbeat_loss()
        self.assertEqual(self.mecanum_controller.state, MotionSafetyState.LOCKED)
        self.assertEqual(self.mecanum_controller.active_wheel_velocities.front_left_m_s, 0.0)

    # 10. Rover Simulator Odometry & Telemetry
    def test_rover_simulator_odometry_and_telemetry(self):
        controller = SafeMotionController(simulation_mode=True)
        controller.unlock_for_simulation()
        sim = RoverSimulator(controller=controller)

        # Initial pose
        self.assertEqual(sim.pose.x_meters, 0.0)
        self.assertEqual(sim.pose.y_meters, 0.0)

        # Move forward at 0.20 m/s for 1.0s
        t0 = 1000.0
        cmd = MotionCommand.create(MotionDirection.FORWARD, linear_speed_m_s=0.20, duration_sec=2.0)
        controller.request_motion(cmd, current_time=t0)

        # Step simulator forward by 1.0 second
        sim.step(dt_seconds=1.0, current_time=t0 + 1.0)
        self.assertAlmostEqual(sim.pose.x_meters, 0.20, places=2)
        self.assertAlmostEqual(sim.pose.y_meters, 0.0, places=2)

        # Telemetry packet inspection
        telemetry = sim.get_telemetry()
        self.assertEqual(telemetry["telemetry_type"], "SIMULATED")
        self.assertEqual(telemetry["hardware_actuators"], "DISABLED")
        self.assertIn("pose", telemetry)
        self.assertIn("controller_status", telemetry)
        self.assertGreaterEqual(telemetry["trajectory_points_count"], 2)


if __name__ == "__main__":
    unittest.main()
