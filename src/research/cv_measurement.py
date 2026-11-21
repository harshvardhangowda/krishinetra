"""
AI-Powered Precision Pest Management - CV Behavioral Measurement System
File: src/research/cv_measurement.py

Provides an OpenCV & Computer Vision-based behavioral tracking analysis engine
for controlled laboratory bioassay trials without physical actuation.

Measures:
1. Centroid trajectory & distance travelled (mm)
2. Instantaneous & mean velocity (mm/s)
3. Zone occupancy duration (Treated vs Control Zone)
4. Zone transition events (Entries / Exits)
5. Response latency from stimulus onset
6. Avoidance Preference Index (API)
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from src.research.experiment_models import BioassayTrialRecord, TrackingDataPoint


class BioassayArenaTracker:
    """Tracks insect position in a calibrated dual-choice chamber using bounding box / centroid."""

    def __init__(
        self,
        arena_width_mm: float = 1000.0,
        arena_height_mm: float = 400.0,
        pixel_to_mm_ratio: float = 1.0,
        treated_zone_x_bounds: Tuple[float, float] = (0.0, 450.0),
        control_zone_x_bounds: Tuple[float, float] = (550.0, 1000.0),
    ):
        """
        Args:
            arena_width_mm: Width of physical arena in millimeters.
            arena_height_mm: Height of physical arena in millimeters.
            pixel_to_mm_ratio: Millimeters per image pixel.
            treated_zone_x_bounds: (x_min, x_max) in physical mm for treated stimulus zone.
            control_zone_x_bounds: (x_min, x_max) in physical mm for control silent zone.
        """
        self.arena_width_mm = arena_width_mm
        self.arena_height_mm = arena_height_mm
        self.scale = pixel_to_mm_ratio
        self.treated_bounds = treated_zone_x_bounds
        self.control_bounds = control_zone_x_bounds

        # Tracking state
        self.trajectory: List[TrackingDataPoint] = []
        self.current_zone: str = "ZONE_NEUTRAL"
        self.zone_durations: Dict[str, float] = {
            "ZONE_TREATED": 0.0,
            "ZONE_CONTROL": 0.0,
            "ZONE_NEUTRAL": 0.0,
        }
        self.entries_treated: int = 0
        self.exits_treated: int = 0
        self.total_distance_mm: float = 0.0
        self.stimulus_onset_ms: Optional[float] = None
        self.first_response_latency_sec: Optional[float] = None

    def classify_zone(self, x_mm: float) -> str:
        """Classify spatial location into Treated, Control, or Neutral zone."""
        if self.treated_bounds[0] <= x_mm <= self.treated_bounds[1]:
            return "ZONE_TREATED"
        elif self.control_bounds[0] <= x_mm <= self.control_bounds[1]:
            return "ZONE_CONTROL"
        else:
            return "ZONE_NEUTRAL"

    def update_frame(
        self,
        trial_id: str,
        frame_idx: int,
        timestamp_ms: float,
        bbox_xyxy: List[float],
        is_stimulus_active: bool = False,
    ) -> TrackingDataPoint:
        """
        Process a detected bounding box and update tracking state.

        Args:
            trial_id: Identifier for bioassay trial.
            frame_idx: Video frame index.
            timestamp_ms: Timestamp in milliseconds.
            bbox_xyxy: [x1, y1, x2, y2] in pixels.
            is_stimulus_active: Whether acoustic stimulus is actively on.
        """
        cx_px = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        cy_px = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0

        cx_mm = cx_px * self.scale
        cy_mm = cy_px * self.scale

        new_zone = self.classify_zone(cx_mm)
        velocity_mm_s = 0.0

        if self.trajectory:
            prev_point = self.trajectory[-1]
            dt_sec = max((timestamp_ms - prev_point.timestamp_ms) / 1000.0, 0.001)
            dx = cx_mm - prev_point.centroid_x
            dy = cy_mm - prev_point.centroid_y
            dist_step = math.sqrt(dx * dx + dy * dy)
            self.total_distance_mm += dist_step
            velocity_mm_s = dist_step / dt_sec

            # Accumulate time in previous zone
            self.zone_durations[self.current_zone] += dt_sec

            # Detect zone transitions
            if self.current_zone != new_zone:
                if new_zone == "ZONE_TREATED":
                    self.entries_treated += 1
                elif self.current_zone == "ZONE_TREATED":
                    self.exits_treated += 1

            # Detect response latency on stimulus onset
            if is_stimulus_active and not prev_point.is_stimulus_active:
                self.stimulus_onset_ms = timestamp_ms

            if self.stimulus_onset_ms is not None and self.first_response_latency_sec is None:
                # If velocity changes by > 50% or zone changes, record latency
                if velocity_mm_s > 15.0 or new_zone != self.current_zone:
                    self.first_response_latency_sec = round((timestamp_ms - self.stimulus_onset_ms) / 1000.0, 3)

        self.current_zone = new_zone

        point = TrackingDataPoint(
            trial_id=trial_id,
            frame_number=frame_idx,
            timestamp_ms=timestamp_ms,
            centroid_x=round(cx_mm, 2),
            centroid_y=round(cy_mm, 2),
            current_zone=new_zone,
            velocity_mm_per_sec=round(velocity_mm_s, 2),
            is_stimulus_active=is_stimulus_active,
        )
        self.trajectory.append(point)
        return point

    def compute_trial_summary(self, trial_record: BioassayTrialRecord) -> BioassayTrialRecord:
        """Compute aggregate behavioral metrics and update trial record."""
        trial_record.time_spent_treated_zone_sec = round(self.zone_durations["ZONE_TREATED"], 2)
        trial_record.time_spent_control_zone_sec = round(self.zone_durations["ZONE_CONTROL"], 2)
        trial_record.entries_to_treated_zone = self.entries_treated
        trial_record.exits_from_treated_zone = self.exits_treated
        trial_record.response_latency_sec = self.first_response_latency_sec

        # Calculate Avoidance Preference Index
        trial_record.calculate_avoidance_index()

        # Classify result
        api = trial_record.avoidance_preference_index or 0.0
        if api >= 0.35:
            trial_record.result_classification = "REPELLED"
        elif api <= -0.35:
            trial_record.result_classification = "ATTRACTED"
        else:
            trial_record.result_classification = "NEUTRAL"

        return trial_record
