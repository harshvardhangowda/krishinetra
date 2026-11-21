"""
Unit and Integration Tests for Phase 9:
- Scientific Acoustic Profiles
- Simulation-Only Decision Layer
- Bioassay Experiment Models and CV Tracker
"""

import unittest
from src.decision.simulation_engine import SimulationDecisionLayer
from src.research.cv_measurement import BioassayArenaTracker
from src.research.experiment_models import BioassayTrialRecord, TrackingDataPoint


class TestPhase9ResearchAndSimulation(unittest.TestCase):

    def test_simulation_layer_army_worm(self):
        sim = SimulationDecisionLayer()
        detection_packet = {
            "is_baseline_model": False,
            "detections": [
                {"class_name": "army worm", "confidence": 0.88, "bbox": [100, 100, 200, 200]}
            ]
        }
        report = sim.simulate_detection_event(detection_packet)

        self.assertTrue(report["simulation_mode"])
        self.assertEqual(report["deterrence_status"], "DETERRENCE NOT ACTIVATED")
        self.assertEqual(report["hardware_action"], "DISABLED")
        self.assertEqual(report["pest_detected"], "army worm")
        self.assertIn("25.0 - 45.0 kHz", report["recommended_treatment"])
        self.assertIn("Strong", report["evidence_level"])
        self.assertTrue(report["safety_interlock_engaged"])

    def test_simulation_layer_corn_borer(self):
        sim = SimulationDecisionLayer()
        detection_packet = {
            "is_baseline_model": False,
            "detections": [
                {"class_name": "corn borer", "confidence": 0.82, "bbox": [50, 50, 150, 150]}
            ]
        }
        report = sim.simulate_detection_event(detection_packet)

        self.assertEqual(report["deterrence_status"], "DETERRENCE NOT ACTIVATED")
        self.assertEqual(report["hardware_action"], "DISABLED")
        self.assertEqual(report["pest_detected"], "corn borer")
        self.assertIn("25.0 - 50.0 kHz", report["recommended_treatment"])
        self.assertIn("Moderate", report["evidence_level"])

    def test_simulation_layer_blister_beetle(self):
        sim = SimulationDecisionLayer()
        detection_packet = {
            "is_baseline_model": False,
            "detections": [
                {"class_name": "blister beetle", "confidence": 0.91, "bbox": [80, 80, 180, 180]}
            ]
        }
        report = sim.simulate_detection_event(detection_packet)

        self.assertEqual(report["deterrence_status"], "DETERRENCE NOT ACTIVATED")
        self.assertEqual(report["hardware_action"], "DISABLED")
        self.assertEqual(report["pest_detected"], "blister beetle")
        self.assertIn("NO SUFFICIENT EVIDENCE FOUND", report["recommended_treatment"])
        self.assertIn("Insufficient", report["evidence_level"])

    def test_simulation_layer_beneficial_insect(self):
        sim = SimulationDecisionLayer()
        detection_packet = {
            "is_baseline_model": False,
            "detections": [
                {"class_name": "honey bee", "confidence": 0.95, "bbox": [10, 10, 60, 60]}
            ]
        }
        report = sim.simulate_detection_event(detection_packet)

        self.assertEqual(report["deterrence_status"], "DETERRENCE NOT ACTIVATED")
        self.assertEqual(report["hardware_action"], "DISABLED")
        self.assertEqual(report["pest_detected"], "None")
        self.assertEqual(report["recommended_treatment"], "NO_ACTION")

    def test_cv_arena_tracker_metrics(self):
        tracker = BioassayArenaTracker(
            arena_width_mm=1000.0,
            arena_height_mm=400.0,
            pixel_to_mm_ratio=1.0,
            treated_zone_x_bounds=(0.0, 450.0),
            control_zone_x_bounds=(550.0, 1000.0),
        )

        # Frame 1: Center neutral
        p1 = tracker.update_frame("TRIAL_001", 0, 0.0, [490, 190, 510, 210], is_stimulus_active=False)
        self.assertEqual(p1.current_zone, "ZONE_NEUTRAL")

        # Frame 2: Move into Control Zone (x=700)
        p2 = tracker.update_frame("TRIAL_001", 1, 1000.0, [690, 190, 710, 210], is_stimulus_active=True)
        self.assertEqual(p2.current_zone, "ZONE_CONTROL")

        # Frame 3: Stay in Control Zone (x=750)
        p3 = tracker.update_frame("TRIAL_001", 2, 5000.0, [740, 190, 760, 210], is_stimulus_active=True)

        record = BioassayTrialRecord(
            trial_id="TRIAL_001",
            experiment_id="EXP_001",
            pest_class="army_worm",
            ip102_class_id=24,
            insect_stage="adult_moth",
            specimen_id="SPEC_01",
            treatment_type="SWEPT_PULSE_US",
            frequency_hz=35000.0,
            exposure_duration_sec=5.0,
        )

        summary = tracker.compute_trial_summary(record)
        self.assertEqual(summary.avoidance_preference_index, 1.0)
        self.assertEqual(summary.result_classification, "REPELLED")


if __name__ == "__main__":
    unittest.main()
