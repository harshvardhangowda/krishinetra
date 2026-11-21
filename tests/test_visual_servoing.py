"""
Unit and Integration Tests for Phase 14: Visual Servoing & Pest Centering
File: tests/test_visual_servoing.py

Validates:
1. Centered targets (within dead-zone -> STOP command & CENTERED status)
2. Left and right offset targets (proportional ROTATE_LEFT / ROTATE_RIGHT)
3. Missing detections (empty list -> TARGET_LOST & STOP)
4. Invalid bounding box filtering (malformed/inverted/out-of-bounds)
5. Multi-target policies (HIGHEST_CONFIDENCE, CLOSEST_TO_CENTER, LARGEST_BBOX_AREA)
6. Confidence filtering (< min_confidence ignored)
7. Closed-loop convergence (iterative alignment loop reaches center)
8. Iteration limits (avoids infinite loops)
9. Safety-lock enforcement during servoing
10. Real trained pest model pipeline compatibility
"""

import math
import unittest
from src.motion.motion_commands import MotionDirection, MotionSafetyState
from src.motion.motion_controller import SafeMotionController
from src.motion.rover_simulator import RoverPose, RoverSimulator
from src.servoing.target_selector import (
    TargetSelector,
    TargetSelectionPolicy,
    SelectedTarget,
)
from src.servoing.centering_controller import (
    PestCenteringController,
    ServoingStatus,
    AlignmentResult,
)
from src.servoing.closed_loop_simulator import (
    SimulatedCameraEnvironment,
    ClosedLoopServoingSimulator,
)
from src.detection.detector import PestDetector


class TestVisualServoingPipeline(unittest.TestCase):

    def setUp(self):
        self.selector = TargetSelector(min_confidence=0.60)
        self.centering_ctrl = PestCenteringController(
            dead_zone_x=0.05,
            dead_zone_y=0.05,
            k_p_angular=0.80,
            min_angular_speed_rad_s=0.08,
            max_angular_speed_rad_s=0.80,
        )

    # 1. Centered Target
    def test_centered_target_within_dead_zone(self):
        # Frame: 640x480, Center: (320, 240). Bbox centered around 320: [300, 220, 340, 260]
        detections = [
            {"class_name": "army worm", "confidence": 0.85, "bbox": [300, 220, 340, 260]}
        ]
        target = self.selector.select_target(detections, 640, 480)
        self.assertIsNotNone(target)
        self.assertEqual(target.centroid_x, 320.0)

        alignment = self.centering_ctrl.compute_alignment(target, 640, 480)
        self.assertEqual(alignment.status, ServoingStatus.CENTERED)
        self.assertTrue(alignment.is_centered)
        self.assertAlmostEqual(alignment.error_x, 0.0, places=3)
        self.assertEqual(alignment.suggested_command.direction, MotionDirection.STOP)

    # 2. Left and Right Offset Targets
    def test_offset_targets_left_and_right(self):
        # Target to the RIGHT (cx = 480 > 320, error_x = +0.50)
        right_det = [{"class_name": "corn borer", "confidence": 0.80, "bbox": [460, 220, 500, 260]}]
        target_r = self.selector.select_target(right_det, 640, 480)
        align_r = self.centering_ctrl.compute_alignment(target_r, 640, 480)

        self.assertEqual(align_r.status, ServoingStatus.ALIGNING)
        self.assertFalse(align_r.is_centered)
        self.assertGreater(align_r.error_x, 0.0)
        self.assertEqual(align_r.suggested_command.direction, MotionDirection.ROTATE_RIGHT)
        self.assertGreater(align_r.suggested_command.angular_speed_rad_s, 0.0)

        # Target to the LEFT (cx = 160 < 320, error_x = -0.50)
        left_det = [{"class_name": "corn borer", "confidence": 0.80, "bbox": [140, 220, 180, 260]}]
        target_l = self.selector.select_target(left_det, 640, 480)
        align_l = self.centering_ctrl.compute_alignment(target_l, 640, 480)

        self.assertEqual(align_l.status, ServoingStatus.ALIGNING)
        self.assertFalse(align_l.is_centered)
        self.assertLess(align_l.error_x, 0.0)
        self.assertEqual(align_l.suggested_command.direction, MotionDirection.ROTATE_LEFT)

    # 3. Missing Detections (Empty List)
    def test_missing_detections_returns_target_lost(self):
        target = self.selector.select_target([], 640, 480)
        self.assertIsNone(target)

        alignment = self.centering_ctrl.compute_alignment(None, 640, 480)
        self.assertEqual(alignment.status, ServoingStatus.TARGET_LOST)
        self.assertFalse(alignment.is_centered)
        self.assertEqual(alignment.suggested_command.direction, MotionDirection.STOP)

    # 4. Invalid Bounding Box Handling
    def test_invalid_bounding_boxes_are_filtered(self):
        invalid_detections = [
            {"class_name": "army worm", "confidence": 0.90, "bbox": [300, 200, 250, 250]}, # x2 <= x1 inverted
            {"class_name": "army worm", "confidence": 0.90, "bbox": [100, 300, 200, 200]}, # y2 <= y1 inverted
            {"class_name": "army worm", "confidence": 0.90, "bbox": [float("nan"), 10, 50, 60]}, # NaN
            {"class_name": "army worm", "confidence": 0.90, "bbox": "bad_format"}, # Not a list
            {"class_name": "army worm", "confidence": 0.90, "bbox": [800, 600, 900, 700]}, # Outside frame
        ]
        target = self.selector.select_target(invalid_detections, 640, 480)
        self.assertIsNone(target)

    # 5. Multi-Target Policies
    def test_multi_target_selection_policies(self):
        dets = [
            {"class_name": "target_low_conf", "confidence": 0.65, "bbox": [310, 230, 330, 250]}, # Centered, area=400, conf=0.65
            {"class_name": "target_high_conf", "confidence": 0.95, "bbox": [100, 100, 150, 150]}, # Offset, area=2500, conf=0.95
            {"class_name": "target_large_area", "confidence": 0.70, "bbox": [400, 200, 550, 350]}, # Offset, area=22500, conf=0.70
        ]

        # Policy 1: Highest confidence
        sel_conf = TargetSelector(min_confidence=0.60, policy=TargetSelectionPolicy.HIGHEST_CONFIDENCE)
        t_conf = sel_conf.select_target(dets, 640, 480)
        self.assertEqual(t_conf.class_name, "target_high_conf")

        # Policy 2: Closest to center
        sel_center = TargetSelector(min_confidence=0.60, policy=TargetSelectionPolicy.CLOSEST_TO_CENTER)
        t_center = sel_center.select_target(dets, 640, 480)
        self.assertEqual(t_center.class_name, "target_low_conf")

        # Policy 3: Largest bbox area
        sel_area = TargetSelector(min_confidence=0.60, policy=TargetSelectionPolicy.LARGEST_BBOX_AREA)
        t_area = sel_area.select_target(dets, 640, 480)
        self.assertEqual(t_area.class_name, "target_large_area")

    # 6. Confidence Filtering
    def test_confidence_filtering(self):
        low_conf_dets = [
            {"class_name": "army worm", "confidence": 0.42, "bbox": [300, 200, 350, 250]} # Below 0.60 threshold
        ]
        target = self.selector.select_target(low_conf_dets, 640, 480)
        self.assertIsNone(target)

    # 7. Closed-Loop Visual Servoing Convergence
    def test_closed_loop_alignment_convergence(self):
        # Rover at (0, 0, theta=0). Pest at (1.0m, -0.3m).
        # Pest is at offset angle = atan2(-0.3, 1.0) = -0.291 rad (~ -16.7 degrees) to the right.
        sim = ClosedLoopServoingSimulator()
        result = sim.run_alignment_loop(
            pest_world_x=1.0,
            pest_world_y=-0.30,
            max_iterations=35,
            dt_seconds=0.10,
        )

        self.assertEqual(result["telemetry_type"], "SIMULATED")
        self.assertEqual(result["hardware_actuators"], "DISABLED")
        self.assertTrue(result["converged"])
        self.assertEqual(result["final_status"], ServoingStatus.CENTERED.value)
        self.assertLessEqual(abs(result["final_error_x"]), 0.05)
        self.assertGreater(result["total_iterations"], 1)

    # 8. Iteration Limits (Guards against infinite loops)
    def test_iteration_limit_terminates_gracefully(self):
        sim = ClosedLoopServoingSimulator()
        # Set max iterations very small (e.g. 2 iterations)
        result = sim.run_alignment_loop(
            pest_world_x=1.0,
            pest_world_y=0.40,
            max_iterations=2,
            dt_seconds=0.10,
        )
        self.assertEqual(result["total_iterations"], 2)
        self.assertFalse(result["converged"])
        self.assertEqual(result["hardware_actuators"], "DISABLED")

    # 9. Safety Lock Enforcement during Servoing
    def test_safety_lock_rejection_during_servoing(self):
        controller = SafeMotionController(simulation_mode=True)
        # Controller starts in LOCKED state
        self.assertEqual(controller.state, MotionSafetyState.LOCKED)

        # Propose alignment command
        target = SelectedTarget(
            class_name="army worm",
            class_id=0,
            confidence=0.88,
            bbox=[450, 200, 500, 250],
            centroid_x=475.0,
            centroid_y=225.0,
            bbox_width=50.0,
            bbox_height=50.0,
            bbox_area=2500.0,
        )
        alignment = self.centering_ctrl.compute_alignment(target, 640, 480)
        self.assertIsNotNone(alignment.suggested_command)

        # Attempt to dispatch while locked
        accepted, reason, vel = controller.request_motion(alignment.suggested_command)
        self.assertFalse(accepted)
        self.assertIn("SAFETY_LOCKED", reason)
        self.assertIsNone(vel)

    # 10. Real Model Pipeline Compatibility Check
    def test_real_model_pipeline_compatibility(self):
        # Verify detector loading and output structure compatibility with TargetSelector
        detector = PestDetector(weights_path="models/pest_yolov8n_best.pt", device="cpu")
        results = detector.detect("tests/data/test_sample.jpg")

        self.assertIsInstance(results, list)
        # Even if 0 detections on test_sample, selector must process without exception
        selected = self.selector.select_target(results, 640, 480)
        # Should return None cleanly for 0 detections
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
