"""
Unit and Integration Tests for Phase 15: Multi-Frame Target Tracking & Occlusion Handling
File: tests/test_target_tracker.py

Validates:
1. Persistent track IDs across multiple frames
2. Moving target velocity estimation & coasting
3. Multiple targets simultaneous tracking with distinct IDs
4. Crossed trajectories and duplicate assignment prevention
5. Brief temporal occlusion handling and track recovery
6. Prolonged target loss and track expiration
7. Invalid bounding boxes and non-detection resilience
8. Confidence threshold filtering
9. Duplicate detection association prevention (1-to-1 matching)
10. Deterministic target selection with PERSISTENT_LOCK policy
11. Centering controller occlusion safety hold (commanded STOP during occlusion)
12. Closed-loop visual servoing with simulated occlusion recovery
"""

import unittest
from src.motion.motion_commands import MotionDirection, MotionSafetyState
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
from src.tracking.target_tracker import (
    TrackStatus,
    TrackedTarget,
    MultiFrameTracker,
    compute_iou,
    compute_centroid_distance,
)


class TestMultiFrameTargetTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = MultiFrameTracker(
            max_missed_frames=4,
            min_hits_to_confirm=1,
            min_detection_confidence=0.60,
            iou_threshold=0.20,
            max_centroid_distance_px=100.0,
            frame_width=640,
            frame_height=480,
        )

    # 1. Persistent Track ID Across Consecutive Frames
    def test_persistent_id_single_target(self):
        # Frame 1: Detection at (100, 100) -> [80, 80, 120, 120]
        det_f1 = [{"class_name": "army worm", "confidence": 0.85, "bbox": [80, 80, 120, 120]}]
        tracks_f1 = self.tracker.update(det_f1)
        self.assertEqual(len(tracks_f1), 1)
        initial_id = tracks_f1[0].track_id
        self.assertEqual(tracks_f1[0].status, TrackStatus.TRACKED)
        self.assertTrue(tracks_f1[0].is_observed)
        self.assertFalse(tracks_f1[0].is_predicted)

        # Frame 2: Small movement to [84, 82, 124, 122]
        det_f2 = [{"class_name": "army worm", "confidence": 0.88, "bbox": [84, 82, 124, 122]}]
        tracks_f2 = self.tracker.update(det_f2)
        self.assertEqual(len(tracks_f2), 1)
        self.assertEqual(tracks_f2[0].track_id, initial_id)
        self.assertEqual(tracks_f2[0].age_frames, 2)
        self.assertEqual(tracks_f2[0].total_hits, 2)
        self.assertEqual(tracks_f2[0].consecutive_misses, 0)

        # Frame 3: Small movement to [88, 85, 128, 125]
        det_f3 = [{"class_name": "army worm", "confidence": 0.86, "bbox": [88, 85, 128, 125]}]
        tracks_f3 = self.tracker.update(det_f3)
        self.assertEqual(tracks_f3[0].track_id, initial_id)
        self.assertEqual(tracks_f3[0].total_hits, 3)

    # 2. Moving Target Velocity Estimation
    def test_moving_target_velocity_estimation(self):
        # Frame 1: Center at 100.0
        self.tracker.update([{"class_name": "corn borer", "confidence": 0.90, "bbox": [80, 80, 120, 120]}])
        # Frame 2: Center moves +10px to 110.0
        self.tracker.update([{"class_name": "corn borer", "confidence": 0.90, "bbox": [90, 80, 130, 120]}])
        # Frame 3: Center moves +10px to 120.0
        tracks = self.tracker.update([{"class_name": "corn borer", "confidence": 0.90, "bbox": [100, 80, 140, 120]}])

        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0].centroid_x, 120.0, places=1)
        # Velocity should approximate +10.0 px/frame
        self.assertGreater(tracks[0].velocity_x, 8.0)
        self.assertLessEqual(tracks[0].velocity_x, 12.0)
        self.assertAlmostEqual(tracks[0].velocity_y, 0.0, places=1)

    # 3. Multiple Targets with Distinct Persistent IDs
    def test_multiple_targets_distinct_ids(self):
        # Frame 1: Target A at (100, 100), Target B at (400, 300)
        dets_f1 = [
            {"class_name": "army worm", "confidence": 0.85, "bbox": [80, 80, 120, 120]},
            {"class_name": "blister beetle", "confidence": 0.78, "bbox": [380, 280, 420, 320]},
        ]
        tracks_f1 = self.tracker.update(dets_f1)
        self.assertEqual(len(tracks_f1), 2)
        ids = {t.track_id for t in tracks_f1}
        self.assertEqual(len(ids), 2)

        # Frame 2: Slight displacement for both
        dets_f2 = [
            {"class_name": "army worm", "confidence": 0.87, "bbox": [85, 82, 125, 122]},
            {"class_name": "blister beetle", "confidence": 0.80, "bbox": [382, 285, 422, 325]},
        ]
        tracks_f2 = self.tracker.update(dets_f2)
        self.assertEqual(len(tracks_f2), 2)
        ids_f2 = {t.track_id for t in tracks_f2}
        self.assertEqual(ids, ids_f2)

    # 4. Crossed Trajectories & Duplicate Prevention
    def test_crossed_trajectories_and_duplicate_prevention(self):
        # Target 1 (army worm) moving right from cx=200 to cx=240
        # Target 2 (corn borer) moving left from cx=280 to cx=240
        t1_f1 = {"class_name": "army worm", "confidence": 0.85, "bbox": [180, 200, 220, 240]} # cx=200
        t2_f1 = {"class_name": "corn borer", "confidence": 0.85, "bbox": [260, 200, 300, 240]} # cx=280
        tracks_f1 = self.tracker.update([t1_f1, t2_f1])
        self.assertEqual(len(tracks_f1), 2)
        id1 = next(t.track_id for t in tracks_f1 if t.class_name == "army worm")
        id2 = next(t.track_id for t in tracks_f1 if t.class_name == "corn borer")

        # Crossing frame: both close to center cx ~ 240
        t1_f2 = {"class_name": "army worm", "confidence": 0.85, "bbox": [220, 200, 260, 240]} # cx=240
        t2_f2 = {"class_name": "corn borer", "confidence": 0.85, "bbox": [230, 200, 270, 240]} # cx=250
        tracks_f2 = self.tracker.update([t1_f2, t2_f2])
        self.assertEqual(len(tracks_f2), 2)

        # Post-crossing frame: Target 1 at cx=270, Target 2 at cx=210
        t1_f3 = {"class_name": "army worm", "confidence": 0.85, "bbox": [250, 200, 290, 240]} # cx=270
        t2_f3 = {"class_name": "corn borer", "confidence": 0.85, "bbox": [190, 200, 230, 240]} # cx=210
        tracks_f3 = self.tracker.update([t1_f3, t2_f3])
        self.assertEqual(len(tracks_f3), 2)

        # Verify class and identity preserved without duplicate assignment
        res_t1 = next(t for t in tracks_f3 if t.class_name == "army worm")
        res_t2 = next(t for t in tracks_f3 if t.class_name == "corn borer")
        self.assertEqual(res_t1.track_id, id1)
        self.assertEqual(res_t2.track_id, id2)

    # 5. Brief Temporal Occlusion Handling and Reacquisition
    def test_brief_occlusion_and_recovery(self):
        # Frame 1 & 2: Observed target
        self.tracker.update([{"class_name": "army worm", "confidence": 0.90, "bbox": [200, 200, 250, 250]}])
        t_f2 = self.tracker.update([{"class_name": "army worm", "confidence": 0.90, "bbox": [205, 200, 255, 250]}])
        track_id = t_f2[0].track_id

        # Frame 3: Missing (leaf occlusion 1)
        t_f3 = self.tracker.update([])
        self.assertEqual(len(t_f3), 1)
        self.assertEqual(t_f3[0].track_id, track_id)
        self.assertEqual(t_f3[0].status, TrackStatus.TEMPORARILY_MISSING)
        self.assertEqual(t_f3[0].consecutive_misses, 1)
        self.assertFalse(t_f3[0].is_observed)
        self.assertTrue(t_f3[0].is_predicted)

        # Frame 4: Missing (leaf occlusion 2)
        t_f4 = self.tracker.update([])
        self.assertEqual(t_f4[0].status, TrackStatus.TEMPORARILY_MISSING)
        self.assertEqual(t_f4[0].consecutive_misses, 2)

        # Frame 5: Reappearance near predicted position
        t_f5 = self.tracker.update([{"class_name": "army worm", "confidence": 0.92, "bbox": [215, 200, 265, 250]}])
        self.assertEqual(len(t_f5), 1)
        self.assertEqual(t_f5[0].track_id, track_id) # Same track ID preserved!
        self.assertEqual(t_f5[0].status, TrackStatus.TRACKED)
        self.assertEqual(t_f5[0].consecutive_misses, 0)
        self.assertTrue(t_f5[0].is_observed)
        self.assertFalse(t_f5[0].is_predicted)

    # 6. Prolonged Loss and Track Expiration
    def test_prolonged_loss_and_expiration(self):
        # Tracker has max_missed_frames=4
        self.tracker.update([{"class_name": "army worm", "confidence": 0.85, "bbox": [100, 100, 150, 150]}])

        # Miss 4 frames: should stay TEMPORARILY_MISSING
        for i in range(1, 5):
            tracks = self.tracker.update([])
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].consecutive_misses, i)
            self.assertEqual(tracks[0].status, TrackStatus.TEMPORARILY_MISSING)

        # Miss 5th frame: exceeds max_missed_frames (4) -> pruned
        tracks_f6 = self.tracker.update([])
        self.assertEqual(len(tracks_f6), 0)
        self.assertEqual(len(self.tracker.get_active_tracks()), 0)

    # 7. Invalid Bounding Box Resilience
    def test_invalid_bounding_boxes_resilience(self):
        # Valid track first
        self.tracker.update([{"class_name": "army worm", "confidence": 0.85, "bbox": [100, 100, 150, 150]}])

        # Next frame with corrupt detections
        corrupt = [
            {"class_name": "army worm", "confidence": 0.90, "bbox": [200, 200, 100, 100]}, # x2 < x1
            {"class_name": "army worm", "confidence": 0.90, "bbox": [float("nan"), 10, 20, 30]}, # NaN
            {"class_name": "army worm", "confidence": 0.90, "bbox": "invalid_string"},
        ]
        # Must not raise an exception, treats as unmatched
        tracks = self.tracker.update(corrupt)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].status, TrackStatus.TEMPORARILY_MISSING)

    # 8. Confidence Filtering
    def test_confidence_filtering(self):
        # Below min_detection_confidence (0.60)
        low_conf = [{"class_name": "army worm", "confidence": 0.40, "bbox": [100, 100, 150, 150]}]
        tracks = self.tracker.update(low_conf)
        self.assertEqual(len(tracks), 0)

        # Above threshold
        high_conf = [{"class_name": "army worm", "confidence": 0.75, "bbox": [100, 100, 150, 150]}]
        tracks_high = self.tracker.update(high_conf)
        self.assertEqual(len(tracks_high), 1)

    # 9. Duplicate Association Prevention (1-to-1 Matching)
    def test_duplicate_association_prevention(self):
        # 1 active track
        self.tracker.update([{"class_name": "army worm", "confidence": 0.85, "bbox": [100, 100, 140, 140]}])

        # Next frame: 2 distinct detections close to the same spot
        dets = [
            {"class_name": "army worm", "confidence": 0.90, "bbox": [102, 102, 142, 142]},
            {"class_name": "army worm", "confidence": 0.88, "bbox": [106, 106, 146, 146]},
        ]
        tracks = self.tracker.update(dets)
        # One detection matches existing track; the other spawns a new distinct track
        self.assertEqual(len(tracks), 2)
        self.assertNotEqual(tracks[0].track_id, tracks[1].track_id)

    # 10. Deterministic Target Selection with PERSISTENT_LOCK Policy
    def test_deterministic_persistent_lock_policy(self):
        selector = TargetSelector(
            min_confidence=0.60,
            policy=TargetSelectionPolicy.PERSISTENT_LOCK,
            fallback_policy=TargetSelectionPolicy.HIGHEST_CONFIDENCE,
        )

        # Frame 1: Target 1 (conf=0.80) and Target 2 (conf=0.75)
        # Fallback picks Target 1 (conf=0.80) and locks onto its track_id
        tracks_f1 = [
            TrackedTarget(
                track_id=101, class_name="army worm", class_id=0, confidence=0.80,
                bbox=[100, 100, 150, 150], centroid_x=125.0, centroid_y=125.0,
                bbox_width=50.0, bbox_height=50.0, bbox_area=2500.0,
            ),
            TrackedTarget(
                track_id=102, class_name="corn borer", class_id=1, confidence=0.75,
                bbox=[400, 200, 450, 250], centroid_x=425.0, centroid_y=225.0,
                bbox_width=50.0, bbox_height=50.0, bbox_area=2500.0,
            ),
        ]
        sel_f1 = selector.select_target(tracks_f1)
        self.assertEqual(sel_f1.track_id, 101)

        # Frame 2: Target 2 gets higher confidence (0.95 vs 0.82)
        tracks_f2 = [
            TrackedTarget(
                track_id=101, class_name="army worm", class_id=0, confidence=0.82,
                bbox=[105, 100, 155, 150], centroid_x=130.0, centroid_y=125.0,
                bbox_width=50.0, bbox_height=50.0, bbox_area=2500.0,
            ),
            TrackedTarget(
                track_id=102, class_name="corn borer", class_id=1, confidence=0.95,
                bbox=[402, 200, 452, 250], centroid_x=427.0, centroid_y=225.0,
                bbox_width=50.0, bbox_height=50.0, bbox_area=2500.0,
            ),
        ]
        sel_f2 = selector.select_target(tracks_f2)
        # Should remain locked onto Target 101 (prevent erratic target hopping)
        self.assertEqual(sel_f2.track_id, 101)

        # Frame 3: Target 101 is lost/expired
        tracks_f3 = [
            TrackedTarget(
                track_id=102, class_name="corn borer", class_id=1, confidence=0.95,
                bbox=[405, 200, 455, 250], centroid_x=430.0, centroid_y=225.0,
                bbox_width=50.0, bbox_height=50.0, bbox_area=2500.0,
            ),
        ]
        sel_f3 = selector.select_target(tracks_f3)
        # Lock switches safely to Target 102
        self.assertEqual(sel_f3.track_id, 102)

    # 11. Centering Controller Occlusion Safety Hold
    def test_centering_controller_occlusion_safety_hold(self):
        controller = PestCenteringController(
            allow_predicted_servoing=False, # Safe default: zero physical motion during occlusion
        )
        occluded_target = SelectedTarget(
            class_name="army worm",
            class_id=0,
            confidence=0.75,
            bbox=[450, 200, 500, 250],
            centroid_x=475.0, # Offset to the right
            centroid_y=225.0,
            bbox_width=50.0,
            bbox_height=50.0,
            bbox_area=2500.0,
            track_id=42,
            status="TEMPORARILY_MISSING",
            is_observed=False,
            is_predicted=True,
            consecutive_misses=2,
        )

        alignment = controller.compute_alignment(occluded_target, 640, 480)
        self.assertEqual(alignment.status, ServoingStatus.TEMPORARILY_OCCLUDED)
        self.assertEqual(alignment.suggested_command.direction, MotionDirection.STOP)
        self.assertEqual(alignment.suggested_command.angular_speed_rad_s, 0.0)
        self.assertIn("Target occluded/predicted", alignment.notes)

    # 12. Closed-Loop Servoing with Simulated Occlusion Recovery
    def test_closed_loop_servoing_with_occlusion_recovery(self):
        tracker = MultiFrameTracker(max_missed_frames=5)
        sim = ClosedLoopServoingSimulator(tracker=tracker)

        # Pest at world (1.0m, -0.25m). Temporarily occluded at iterations 3 and 4.
        result = sim.run_alignment_loop(
            pest_world_x=1.0,
            pest_world_y=-0.25,
            max_iterations=30,
            dt_seconds=0.10,
            occluded_iterations=[3, 4],
        )

        self.assertEqual(result["telemetry_type"], "SIMULATED")
        self.assertEqual(result["hardware_actuators"], "DISABLED")
        self.assertTrue(result["converged"])
        self.assertEqual(result["final_status"], ServoingStatus.CENTERED.value)

        # Inspect iteration telemetry
        iterations = result["iterations"]
        # Iteration 3 & 4 should reflect occlusion hold
        iter_3 = next(it for it in iterations if it["iteration"] == 3)
        self.assertEqual(iter_3["status"], ServoingStatus.TEMPORARILY_OCCLUDED.value)
        self.assertEqual(iter_3["command"]["direction"], MotionDirection.STOP.value)
        self.assertTrue(iter_3["is_predicted"])

        # Track ID should be preserved before, during, and after occlusion
        track_ids = [it["track_id"] for it in iterations if it["track_id"] is not None]
        self.assertTrue(len(track_ids) > 0)
        self.assertEqual(len(set(track_ids)), 1) # Same single track ID throughout


if __name__ == "__main__":
    unittest.main()
