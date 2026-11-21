"""
AI-Powered Precision Pest Management - Multi-Frame Target Tracking & Occlusion Handling
File: src/tracking/target_tracker.py

Provides persistent track ID association, temporal occlusion coasting,
and multi-target state estimation across consecutive video/sensor frames.

Features:
- IoU & centroid-distance bipartite association
- Persistent track ID assignment
- State machine: NEW -> TRACKED -> TEMPORARILY_MISSING -> LOST
- Velocity estimation & linear coasting during occlusion
- Zero physical hardware actuation: simulation-only telemetry
"""

import math
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

from src.servoing.target_selector import SelectedTarget


class TrackStatus(str, Enum):
    NEW = "NEW"                                 # Newly detected candidate in current frame
    TRACKED = "TRACKED"                         # Confirmed and actively matched to an observed detection
    TEMPORARILY_MISSING = "TEMPORARILY_MISSING" # Occluded or missed by detector for <= max_missed_frames
    LOST = "LOST"                               # Missed for > max_missed_frames; expired and pruned


@dataclass
class TrackedTarget:
    """Represents a temporally tracked pest target across multiple frames."""
    track_id: int
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]                  # [x1, y1, x2, y2] in pixels
    centroid_x: float
    centroid_y: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    status: TrackStatus = TrackStatus.TRACKED
    age_frames: int = 1                # Total lifespan in frames
    total_hits: int = 1                # Number of frames with actual detection match
    consecutive_misses: int = 0        # Number of frames missed consecutively
    is_observed: bool = True           # True if matched with an actual detection in current frame
    is_predicted: bool = False         # True if position is estimated/predicted during occlusion
    velocity_x: float = 0.0            # Pixels per frame
    velocity_y: float = 0.0            # Pixels per frame
    last_observed_timestamp: float = 0.0
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "bbox_width": round(self.bbox_width, 2),
            "bbox_height": round(self.bbox_height, 2),
            "bbox_area": round(self.bbox_area, 2),
            "status": self.status.value,
            "age_frames": self.age_frames,
            "total_hits": self.total_hits,
            "consecutive_misses": self.consecutive_misses,
            "is_observed": self.is_observed,
            "is_predicted": self.is_predicted,
            "velocity_x": round(self.velocity_x, 3),
            "velocity_y": round(self.velocity_y, 3),
            "last_observed_timestamp": round(self.last_observed_timestamp, 3),
            "is_valid": self.is_valid,
            "telemetry_type": "SIMULATED",
            "hardware_actuators": "DISABLED",
        }

    def to_selected_target(self) -> SelectedTarget:
        """Converts to SelectedTarget for drop-in compatibility with visual servoing."""
        return SelectedTarget(
            class_name=self.class_name,
            class_id=self.class_id,
            confidence=self.confidence,
            bbox=list(self.bbox),
            centroid_x=self.centroid_x,
            centroid_y=self.centroid_y,
            bbox_width=self.bbox_width,
            bbox_height=self.bbox_height,
            bbox_area=self.bbox_area,
            is_valid=self.is_valid,
            track_id=self.track_id,
            status=self.status.value,
            is_observed=self.is_observed,
            is_predicted=self.is_predicted,
            consecutive_misses=self.consecutive_misses,
        )


def compute_iou(bbox_a: List[float], bbox_b: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2].
    Returns 0.0 if either box has non-positive area or there is no overlap.
    """
    x1_a, y1_a, x2_a, y2_a = bbox_a
    x1_b, y1_b, x2_b, y2_b = bbox_b

    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    intersection = inter_w * inter_h

    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)

    union = area_a + area_b - intersection
    if union <= 1e-6:
        return 0.0

    return float(intersection / union)


def compute_centroid_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


class MultiFrameTracker:
    """
    Multi-object tracker associating pest detections across frames using
    spatial overlap (IoU), centroid proximity, and class consistency.
    Maintains track identity through temporary occlusions without fabricating detections.
    """

    def __init__(
        self,
        max_missed_frames: int = 5,
        min_hits_to_confirm: int = 1,
        min_detection_confidence: float = 0.60,
        iou_threshold: float = 0.25,
        max_centroid_distance_px: float = 120.0,
        weight_iou: float = 0.60,
        weight_distance: float = 0.30,
        weight_confidence: float = 0.10,
        velocity_damping: float = 0.80,
        confidence_decay: float = 0.85,
        velocity_smoothing: float = 0.40,
        strict_class_matching: bool = True,
        frame_width: int = 640,
        frame_height: int = 480,
    ):
        self.max_missed_frames = max_missed_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.min_detection_confidence = min_detection_confidence
        self.iou_threshold = iou_threshold
        self.max_centroid_distance_px = max_centroid_distance_px
        self.weight_iou = weight_iou
        self.weight_distance = weight_distance
        self.weight_confidence = weight_confidence
        self.velocity_damping = velocity_damping
        self.confidence_decay = confidence_decay
        self.velocity_smoothing = velocity_smoothing
        self.strict_class_matching = strict_class_matching
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.tracks: Dict[int, TrackedTarget] = {}
        self.next_track_id: int = 1
        self.frame_index: int = 0
        self.frame_diagonal: float = math.hypot(frame_width, frame_height)

    def reset(self) -> None:
        """Clear all tracks and reset the ID sequence."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_index = 0

    def validate_bbox(self, bbox: Any) -> bool:
        """Verify that a bounding box is valid, non-degenerate and finite."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        try:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except (ValueError, TypeError):
            return False

        if any(math.isnan(v) for v in (x1, y1, x2, y2)):
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        if x2 <= 0 or y2 <= 0 or x1 >= self.frame_width or y1 >= self.frame_height:
            return False
        return True

    def _compute_affinity_score(
        self,
        track: TrackedTarget,
        detection: Dict[str, Any],
    ) -> float:
        """
        Compute association affinity score in [0.0, 1.0] between a track and a detection.
        Returns 0.0 if association criteria are violated.
        """
        det_class = str(detection.get("class_name", ""))
        if self.strict_class_matching and det_class and track.class_name != det_class:
            return 0.0

        det_bbox = [float(v) for v in detection["bbox"]]
        det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
        det_cy = (det_bbox[1] + det_bbox[3]) / 2.0

        iou = compute_iou(track.bbox, det_bbox)
        dist = compute_centroid_distance((track.centroid_x, track.centroid_y), (det_cx, det_cy))

        # Check association bounds
        if iou < self.iou_threshold and dist > self.max_centroid_distance_px:
            return 0.0

        # Distance affinity: 1.0 at 0 distance, linearly decaying to 0.0 at max_centroid_distance_px
        dist_affinity = max(0.0, 1.0 - (dist / self.max_centroid_distance_px))
        conf = float(detection.get("confidence", 0.0))

        score = (
            (self.weight_iou * iou)
            + (self.weight_distance * dist_affinity)
            + (self.weight_confidence * conf)
        )
        return max(0.0, min(1.0, score))

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_timestamp: Optional[float] = None,
    ) -> List[TrackedTarget]:
        """
        Update the tracker state with a new frame of detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'confidence', 'class_name'.
            frame_timestamp: Current time in seconds.
            
        Returns:
            List of active TrackedTarget instances (both TRACKED and TEMPORARILY_MISSING).
        """
        self.frame_index += 1
        now = frame_timestamp if frame_timestamp is not None else time.time()

        # Step 1: Filter and validate raw detections
        valid_detections: List[Dict[str, Any]] = []
        for d in detections:
            conf = float(d.get("confidence", 0.0))
            if conf < self.min_detection_confidence:
                continue
            raw_bbox = d.get("bbox")
            if not self.validate_bbox(raw_bbox):
                continue
            valid_detections.append(d)

        # Step 2: Global greedy bipartite matching (prevent duplicate assignments)
        active_track_ids = list(self.tracks.keys())
        candidate_matches: List[Tuple[float, int, int]] = [] # (score, track_id, det_idx)

        for track_id in active_track_ids:
            track = self.tracks[track_id]
            for det_idx, det in enumerate(valid_detections):
                score = self._compute_affinity_score(track, det)
                if score > 0.0:
                    candidate_matches.append((score, track_id, det_idx))

        # Sort matches by score descending
        candidate_matches.sort(key=lambda m: m[0], reverse=True)

        assigned_track_ids: set = set()
        assigned_det_indices: set = set()

        for score, track_id, det_idx in candidate_matches:
            if track_id in assigned_track_ids or det_idx in assigned_det_indices:
                continue

            # Assign detection to track
            assigned_track_ids.add(track_id)
            assigned_det_indices.add(det_idx)
            self._update_matched_track(self.tracks[track_id], valid_detections[det_idx], now)

        # Step 3: Handle unmatched tracks (coasting / temporal occlusion / expiration)
        expired_track_ids: List[int] = []
        for track_id in active_track_ids:
            if track_id not in assigned_track_ids:
                track = self.tracks[track_id]
                self._update_unmatched_track(track)
                if track.status == TrackStatus.LOST:
                    expired_track_ids.append(track_id)

        # Prune expired tracks
        for track_id in expired_track_ids:
            del self.tracks[track_id]

        # Step 4: Spawn new tracks for unmatched valid detections
        for det_idx, det in enumerate(valid_detections):
            if det_idx not in assigned_det_indices:
                self._spawn_new_track(det, now)

        return self.get_active_tracks()

    def _update_matched_track(
        self,
        track: TrackedTarget,
        detection: Dict[str, Any],
        timestamp: float,
    ) -> None:
        """Update a track that successfully matched an observed detection."""
        det_bbox = [float(v) for v in detection["bbox"]]
        new_cx = (det_bbox[0] + det_bbox[2]) / 2.0
        new_cy = (det_bbox[1] + det_bbox[3]) / 2.0
        new_w = det_bbox[2] - det_bbox[0]
        new_h = det_bbox[3] - det_bbox[1]

        # Calculate instantaneous velocity (pixel displacement)
        inst_vx = new_cx - track.centroid_x
        inst_vy = new_cy - track.centroid_y

        # Exponential moving average velocity update
        if track.total_hits > 1:
            track.velocity_x = (self.velocity_smoothing * inst_vx) + ((1.0 - self.velocity_smoothing) * track.velocity_x)
            track.velocity_y = (self.velocity_smoothing * inst_vy) + ((1.0 - self.velocity_smoothing) * track.velocity_y)
        else:
            track.velocity_x = inst_vx
            track.velocity_y = inst_vy

        # Update observed geometry and state
        track.bbox = det_bbox
        track.centroid_x = new_cx
        track.centroid_y = new_cy
        track.bbox_width = new_w
        track.bbox_height = new_h
        track.bbox_area = new_w * new_h
        track.confidence = float(detection.get("confidence", track.confidence))
        track.class_name = str(detection.get("class_name", track.class_name))
        track.class_id = int(detection.get("class_id", track.class_id))
        track.age_frames += 1
        track.total_hits += 1
        track.consecutive_misses = 0
        track.is_observed = True
        track.is_predicted = False
        track.status = TrackStatus.TRACKED
        track.last_observed_timestamp = timestamp
        track.is_valid = True

    def _update_unmatched_track(self, track: TrackedTarget) -> None:
        """
        Handle a track that was not matched in the current frame.
        Applies linear coasting and confidence decay during temporary occlusion.
        Never fabricates detections.
        """
        track.consecutive_misses += 1
        track.age_frames += 1
        track.is_observed = False

        if track.consecutive_misses <= self.max_missed_frames:
            # Mark as TEMPORARILY_MISSING and coast position
            track.status = TrackStatus.TEMPORARILY_MISSING
            track.is_predicted = True

            # Damp velocity
            track.velocity_x *= self.velocity_damping
            track.velocity_y *= self.velocity_damping

            # Coast centroid
            pred_cx = track.centroid_x + track.velocity_x
            pred_cy = track.centroid_y + track.velocity_y

            # Clamp centroid to frame bounds
            pred_cx = max(0.0, min(float(self.frame_width), pred_cx))
            pred_cy = max(0.0, min(float(self.frame_height), pred_cy))

            half_w = track.bbox_width / 2.0
            half_h = track.bbox_height / 2.0

            track.centroid_x = pred_cx
            track.centroid_y = pred_cy
            track.bbox = [
                max(0.0, pred_cx - half_w),
                max(0.0, pred_cy - half_h),
                min(float(self.frame_width), pred_cx + half_w),
                min(float(self.frame_height), pred_cy + half_h),
            ]
            # Decay confidence
            track.confidence *= self.confidence_decay
        else:
            # Exceeded occlusion window -> mark as LOST
            track.status = TrackStatus.LOST
            track.is_predicted = False
            track.is_valid = False

    def _spawn_new_track(self, detection: Dict[str, Any], timestamp: float) -> TrackedTarget:
        """Create and initialize a new track for an unassigned valid detection."""
        det_bbox = [float(v) for v in detection["bbox"]]
        cx = (det_bbox[0] + det_bbox[2]) / 2.0
        cy = (det_bbox[1] + det_bbox[3]) / 2.0
        bw = det_bbox[2] - det_bbox[0]
        bh = det_bbox[3] - det_bbox[1]

        track_id = self.next_track_id
        self.next_track_id += 1

        new_track = TrackedTarget(
            track_id=track_id,
            class_name=str(detection.get("class_name", "pest")),
            class_id=int(detection.get("class_id", -1)),
            confidence=float(detection.get("confidence", 0.0)),
            bbox=det_bbox,
            centroid_x=cx,
            centroid_y=cy,
            bbox_width=bw,
            bbox_height=bh,
            bbox_area=bw * bh,
            status=TrackStatus.TRACKED if self.min_hits_to_confirm <= 1 else TrackStatus.NEW,
            age_frames=1,
            total_hits=1,
            consecutive_misses=0,
            is_observed=True,
            is_predicted=False,
            velocity_x=0.0,
            velocity_y=0.0,
            last_observed_timestamp=timestamp,
            is_valid=True,
        )
        self.tracks[track_id] = new_track
        return new_track

    def get_active_tracks(self) -> List[TrackedTarget]:
        """Return all active tracks (both TRACKED and TEMPORARILY_MISSING)."""
        return [t for t in self.tracks.values() if t.status in (TrackStatus.TRACKED, TrackStatus.TEMPORARILY_MISSING)]

    def get_confirmed_tracks(self) -> List[TrackedTarget]:
        """Return only tracks with total_hits >= min_hits_to_confirm."""
        return [
            t for t in self.tracks.values()
            if t.total_hits >= self.min_hits_to_confirm
            and t.status in (TrackStatus.TRACKED, TrackStatus.TEMPORARILY_MISSING)
        ]

    def get_track_by_id(self, track_id: int) -> Optional[TrackedTarget]:
        """Look up an active track by its persistent ID."""
        return self.tracks.get(track_id)
