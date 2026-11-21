"""
AI-Powered Precision Pest Management - Bioassay Experiment Data Models
File: src/research/experiment_models.py

Defines validated data models for laboratory bioassay trials, environmental logs,
and computer vision response metrics.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List


@dataclass
class BioassayTrialRecord:
    """Represents a single controlled laboratory bioassay trial."""
    trial_id: str
    experiment_id: str
    pest_class: str
    ip102_class_id: int
    insect_stage: str
    specimen_id: str
    treatment_type: str
    frequency_hz: Optional[float] = None
    frequency_sweep_min_hz: Optional[float] = None
    frequency_sweep_max_hz: Optional[float] = None
    pulse_repetition_rate_hz: Optional[float] = None
    amplitude_db_spl: Optional[float] = None
    distance_meters: float = 0.50
    exposure_duration_sec: float = 600.0
    temperature_c: float = 26.0
    humidity_percent: float = 65.0
    lux_level: float = 0.0
    baseline_activity_score: Optional[float] = None
    treatment_activity_score: Optional[float] = None
    time_spent_treated_zone_sec: float = 0.0
    time_spent_control_zone_sec: float = 0.0
    entries_to_treated_zone: int = 0
    exits_from_treated_zone: int = 0
    avoidance_preference_index: Optional[float] = None
    response_latency_sec: Optional[float] = None
    feeding_suppression_pct: Optional[float] = None
    mortality_observed: bool = False
    result_classification: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def calculate_avoidance_index(self) -> float:
        """
        Calculate Avoidance Preference Index (API):
        API = (time_control - time_treated) / (time_control + time_treated)
        Returns:
            Value between -1.0 (strong attraction) and +1.0 (strong repulsion).
        """
        total_time = self.time_spent_control_zone_sec + self.time_spent_treated_zone_sec
        if total_time <= 0.0:
            self.avoidance_preference_index = 0.0
        else:
            self.avoidance_preference_index = round(
                (self.time_spent_control_zone_sec - self.time_spent_treated_zone_sec) / total_time,
                4
            )
        return self.avoidance_preference_index

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


@dataclass
class TrackingDataPoint:
    """Represents a discrete temporal tracking point from CV trajectory extraction."""
    trial_id: str
    frame_number: int
    timestamp_ms: float
    centroid_x: float
    centroid_y: float
    current_zone: str
    velocity_mm_per_sec: float = 0.0
    is_stimulus_active: bool = False
