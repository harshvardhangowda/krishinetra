"""
AI-Powered Precision Pest Management - Closed-Loop Visual Servoing Simulator
File: src/servoing/closed_loop_simulator.py

Simulates the complete closed-loop target alignment cycle in software:
1. Detect / Project target in camera frame.
2. Select target via TargetSelector.
3. Compute normalized offset & alignment command via PestCenteringController.
4. Execute simulated kinematic rotation via SafeMotionController & RoverSimulator.
5. Re-evaluate camera view and iterate until target is centered or lost.

STRICT SAFETY:
Physical actuators, relays, and motor drivers are PERMANENTLY DISABLED.
All telemetry packets are explicitly tagged as SIMULATED.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any

from src.motion.motion_commands import MotionDirection, MotionSafetyState
from src.motion.motion_controller import SafeMotionController
from src.motion.rover_simulator import RoverPose, RoverSimulator
from src.servoing.target_selector import TargetSelector, TargetSelectionPolicy, SelectedTarget
from src.servoing.centering_controller import PestCenteringController, ServoingStatus, AlignmentResult
from src.tracking.target_tracker import MultiFrameTracker, TrackedTarget


class SimulatedCameraEnvironment:
    """
    Simulates optical pinhole camera projection of a world-space pest target
    onto the 2D image plane based on rover pose.
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        horizontal_fov_degrees: float = 60.0,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fov_h_rad = math.radians(horizontal_fov_degrees)
        # Focal length in pixels: f = (W / 2) / tan(FOV / 2)
        self.focal_length_px = (frame_width / 2.0) / math.tan(self.fov_h_rad / 2.0)

    def project_pest_target(
        self,
        pest_world_x: float,
        pest_world_y: float,
        rover_pose: RoverPose,
        pest_class: str = "army worm",
        confidence: float = 0.88,
    ) -> List[Dict[str, Any]]:
        """
        Project world-frame pest coordinates (X, Y) into camera pixel coordinates.
        Returns empty list if pest is behind camera or outside field of view.
        """
        # Translation relative to rover
        dx = pest_world_x - rover_pose.x_meters
        dy = pest_world_y - rover_pose.y_meters
        theta = rover_pose.heading_radians

        # Transform to rover body / camera coordinates
        # d_forward: along camera optical axis
        # d_lateral: perpendicular (+left, -right)
        d_forward = (dx * math.cos(theta)) + (dy * math.sin(theta))
        d_lateral = (-dx * math.sin(theta)) + (dy * math.cos(theta))

        # Check if target is behind camera or too close
        if d_forward <= 0.05:
            return []

        # Angle from optical axis
        azimuth = math.atan2(d_lateral, d_forward)
        if abs(azimuth) > (self.fov_h_rad / 2.0):
            # Target is outside horizontal field of view
            return []

        # Pinhole projection:
        # In image coordinates, x increases to the right.
        # d_lateral > 0 is left, so pixel center_x - f * (d_lateral / d_forward)
        cx = (self.frame_width / 2.0) - (self.focal_length_px * (d_lateral / d_forward))
        cy = self.frame_height / 2.0 # Assume ground-level pest centered vertically

        # Projected apparent bounding box size inversely proportional to distance
        box_w = max(20.0, min(200.0, 60.0 / d_forward))
        box_h = max(20.0, min(200.0, 50.0 / d_forward))

        x1 = cx - (box_w / 2.0)
        y1 = cy - (box_h / 2.0)
        x2 = cx + (box_w / 2.0)
        y2 = cy + (box_h / 2.0)

        # Check if bounding box intersects frame
        if x2 < 0 or x1 > self.frame_width or y2 < 0 or y1 > self.frame_height:
            return []

        detection = {
            "class_name": pest_class,
            "class_id": 0,
            "confidence": confidence,
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        }
        return [detection]


class ClosedLoopServoingSimulator:
    """
    Executes iterative visual servoing loops connecting vision error to rover locomotion.
    Optionally integrates MultiFrameTracker for persistent track ID tracking and occlusion coasting.
    """

    def __init__(
        self,
        motion_controller: Optional[SafeMotionController] = None,
        rover_simulator: Optional[RoverSimulator] = None,
        target_selector: Optional[TargetSelector] = None,
        centering_controller: Optional[PestCenteringController] = None,
        camera_env: Optional[SimulatedCameraEnvironment] = None,
        tracker: Optional[MultiFrameTracker] = None,
    ):
        self.motion_controller = motion_controller or SafeMotionController(simulation_mode=True)
        self.rover_sim = rover_simulator or RoverSimulator(controller=self.motion_controller)
        self.target_selector = target_selector or TargetSelector(min_confidence=0.60)
        self.centering_controller = centering_controller or PestCenteringController()
        self.camera_env = camera_env or SimulatedCameraEnvironment()
        self.tracker = tracker

        self.iteration_history: List[Dict[str, Any]] = []

    def run_alignment_loop(
        self,
        pest_world_x: float,
        pest_world_y: float,
        max_iterations: int = 30,
        dt_seconds: float = 0.10,
        pest_class: str = "army worm",
        confidence: float = 0.88,
        occluded_iterations: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run the closed-loop visual servoing alignment simulation.
        
        Args:
            pest_world_x: Target world X in meters.
            pest_world_y: Target world Y in meters.
            max_iterations: Maximum loop iterations.
            dt_seconds: Timestep per iteration in seconds.
            pest_class: Target pest classification label.
            confidence: Simulated detection confidence score.
            occluded_iterations: Optional list of 1-indexed iteration numbers where target
                                 is completely obscured/undetected (simulates leaf occlusion).
                                 
        Terminates upon:
        - Target Centered within deadband
        - Target Lost (out of view or expired after prolonged occlusion)
        - Max iterations reached
        - Controller Safety Lock
        """
        self.iteration_history.clear()
        if self.tracker is not None:
            self.tracker.reset()

        start_time = time.time()
        occluded_set = set(occluded_iterations or [])

        # Unlock motion controller for safe simulation
        self.motion_controller.unlock_for_simulation("SIMULATION_SAFE")

        final_status = ServoingStatus.ALIGNING
        converged = False

        for i in range(max_iterations):
            iter_num = i + 1
            sim_time = start_time + (i * dt_seconds)

            # Step 1: Optical Projection / Target Detection (with simulated occlusion)
            if iter_num in occluded_set:
                raw_detections: List[Dict[str, Any]] = []
            else:
                raw_detections = self.camera_env.project_pest_target(
                    pest_world_x=pest_world_x,
                    pest_world_y=pest_world_y,
                    rover_pose=self.rover_sim.pose,
                    pest_class=pest_class,
                    confidence=confidence,
                )

            # Step 2: Temporal Tracking & Target Selection
            if self.tracker is not None:
                tracked_targets = self.tracker.update(raw_detections, frame_timestamp=sim_time)
                selected_target = self.target_selector.select_target(
                    tracked_targets,
                    frame_width=self.camera_env.frame_width,
                    frame_height=self.camera_env.frame_height,
                )
            else:
                selected_target = self.target_selector.select_target(
                    raw_detections,
                    frame_width=self.camera_env.frame_width,
                    frame_height=self.camera_env.frame_height,
                )

            # Step 3: Compute Visual Error & Propose Alignment Command
            alignment = self.centering_controller.compute_alignment(
                selected_target,
                frame_width=self.camera_env.frame_width,
                frame_height=self.camera_env.frame_height,
            )

            # Step 4: Dispatch Motion & Step Rover Simulation
            if alignment.suggested_command is not None:
                self.motion_controller.request_motion(
                    alignment.suggested_command,
                    current_time=sim_time,
                )

            # Step rover simulator physics
            self.rover_sim.step(dt_seconds=dt_seconds, current_time=sim_time)

            # Record iteration telemetry
            iteration_entry = {
                "iteration": iter_num,
                "timestamp_sim": round(sim_time - start_time, 2),
                "pose": self.rover_sim.pose.to_dict(),
                "error_x": round(alignment.error_x, 4),
                "error_y": round(alignment.error_y, 4),
                "status": alignment.status.value,
                "command": alignment.suggested_command.to_dict() if alignment.suggested_command else None,
                "target_detected": selected_target is not None,
                "track_id": getattr(selected_target, "track_id", None) if selected_target else None,
                "is_observed": getattr(selected_target, "is_observed", False) if selected_target else False,
                "is_predicted": getattr(selected_target, "is_predicted", False) if selected_target else False,
                "telemetry_type": "SIMULATED",
                "hardware_actuators": "DISABLED",
            }
            self.iteration_history.append(iteration_entry)

            # Step 5: Convergence & Termination Check
            if alignment.status == ServoingStatus.CENTERED:
                final_status = ServoingStatus.CENTERED
                converged = True
                self.motion_controller.stop(reason="SERVOING_CONVERGED")
                break
            elif alignment.status == ServoingStatus.TARGET_LOST:
                final_status = ServoingStatus.TARGET_LOST
                self.motion_controller.stop(reason="TARGET_LOST")
                break

        return {
            "telemetry_type": "SIMULATED",
            "converged": converged,
            "final_status": final_status.value,
            "total_iterations": len(self.iteration_history),
            "final_pose": self.rover_sim.pose.to_dict(),
            "final_error_x": self.iteration_history[-1]["error_x"] if self.iteration_history else 0.0,
            "hardware_actuators": "DISABLED",
            "iterations": self.iteration_history,
        }
