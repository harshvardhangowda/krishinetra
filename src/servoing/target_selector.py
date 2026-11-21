"""
AI-Powered Precision Pest Management - Visual Servoing Target Selector
File: src/servoing/target_selector.py

Filters, validates, and prioritizes pest detections for closed-loop centering.
Supports configurable multi-target policies:
- HIGHEST_CONFIDENCE: Prioritize target with highest YOLO detection confidence.
- CLOSEST_TO_CENTER: Prioritize target nearest to the optical center.
- LARGEST_BBOX_AREA: Prioritize target with largest apparent image footprint.
"""

import math
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple


class TargetSelectionPolicy(str, Enum):
    HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"
    CLOSEST_TO_CENTER = "CLOSEST_TO_CENTER"
    LARGEST_BBOX_AREA = "LARGEST_BBOX_AREA"
    PERSISTENT_LOCK = "PERSISTENT_LOCK"


@dataclass
class SelectedTarget:
    """Represents a validated, prioritized target candidate for visual centering."""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float] # [x1, y1, x2, y2] in pixels
    centroid_x: float
    centroid_y: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    is_valid: bool = True
    track_id: Optional[int] = None
    status: str = "TRACKED"
    is_observed: bool = True
    is_predicted: bool = False
    consecutive_misses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TargetSelector:
    """Evaluates detections or tracked targets and selects a single prioritized tracking target."""

    def __init__(
        self,
        min_confidence: float = 0.60,
        policy: TargetSelectionPolicy = TargetSelectionPolicy.HIGHEST_CONFIDENCE,
        fallback_policy: TargetSelectionPolicy = TargetSelectionPolicy.HIGHEST_CONFIDENCE,
    ):
        self.min_confidence = min_confidence
        self.policy = policy
        self.fallback_policy = fallback_policy
        self.locked_track_id: Optional[int] = None

    def lock_track(self, track_id: int) -> None:
        """Explicitly lock the selector to a persistent track ID."""
        self.locked_track_id = track_id

    def clear_target_lock(self) -> None:
        """Clear active persistent track ID lock."""
        self.locked_track_id = None

    def validate_bbox(
        self,
        bbox: Any,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """
        Check if bounding box coordinates are well-formed and within image bounds.
        """
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False

        try:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except (ValueError, TypeError):
            return False

        # Non-NaN, positive area, valid bounds
        if any(math.isnan(v) for v in (x1, y1, x2, y2)):
            return False

        if x2 <= x1 or y2 <= y1:
            return False

        # Must have at least some overlap with the image frame
        if x2 <= 0 or y2 <= 0 or x1 >= frame_width or y1 >= frame_height:
            return False

        return True

    def select_target(
        self,
        detections: List[Any],
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> Optional[SelectedTarget]:
        """
        Select the optimal target from a list of detection packets or tracked targets.
        
        Args:
            detections: List of detection dicts or TrackedTarget/SelectedTarget objects.
            frame_width: Image width in pixels.
            frame_height: Image height in pixels.
            
        Returns:
            SelectedTarget or None if no valid candidate exists.
        """
        if not detections:
            return None

        center_x = frame_width / 2.0
        center_y = frame_height / 2.0

        candidates: List[SelectedTarget] = []

        for d in detections:
            # Handle TrackedTarget or SelectedTarget objects
            if hasattr(d, "to_selected_target"):
                cand = d.to_selected_target()
                if cand.confidence >= self.min_confidence and cand.is_valid:
                    candidates.append(cand)
                continue
            elif isinstance(d, SelectedTarget):
                if d.confidence >= self.min_confidence and d.is_valid:
                    candidates.append(d)
                continue

            # Dict-based detection
            if not isinstance(d, dict):
                continue

            conf = float(d.get("confidence", 0.0))
            if conf < self.min_confidence:
                continue

            raw_bbox = d.get("bbox")
            if not self.validate_bbox(raw_bbox, frame_width, frame_height):
                continue

            x1, y1, x2, y2 = [float(v) for v in raw_bbox]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            candidates.append(
                SelectedTarget(
                    class_name=str(d.get("class_name", "pest")),
                    class_id=int(d.get("class_id", -1)),
                    confidence=conf,
                    bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    centroid_x=round(cx, 2),
                    centroid_y=round(cy, 2),
                    bbox_width=round(bw, 2),
                    bbox_height=round(bh, 2),
                    bbox_area=round(area, 2),
                    is_valid=bool(d.get("is_valid", True)),
                    track_id=d.get("track_id"),
                    status=str(d.get("status", "TRACKED")),
                    is_observed=bool(d.get("is_observed", True)),
                    is_predicted=bool(d.get("is_predicted", False)),
                    consecutive_misses=int(d.get("consecutive_misses", 0)),
                )
            )

        if not candidates:
            return None

        # Policy: PERSISTENT_LOCK
        if self.policy == TargetSelectionPolicy.PERSISTENT_LOCK:
            if self.locked_track_id is not None:
                # Look for existing locked target among candidates
                locked_cand = next(
                    (c for c in candidates if c.track_id == self.locked_track_id and c.is_valid),
                    None,
                )
                if locked_cand is not None:
                    return locked_cand

            # If not locked or locked target lost, acquire using fallback policy
            chosen = self._apply_policy(candidates, self.fallback_policy, center_x, center_y)
            if chosen is not None and chosen.track_id is not None:
                self.locked_track_id = chosen.track_id
            return chosen

        return self._apply_policy(candidates, self.policy, center_x, center_y)

    def _apply_policy(
        self,
        candidates: List[SelectedTarget],
        policy: TargetSelectionPolicy,
        center_x: float,
        center_y: float,
    ) -> Optional[SelectedTarget]:
        """Apply a ranking policy to a list of candidates."""
        if not candidates:
            return None

        if policy == TargetSelectionPolicy.HIGHEST_CONFIDENCE:
            return max(candidates, key=lambda c: c.confidence)
        elif policy == TargetSelectionPolicy.CLOSEST_TO_CENTER:
            return min(
                candidates,
                key=lambda c: (c.centroid_x - center_x) ** 2 + (c.centroid_y - center_y) ** 2,
            )
        elif policy == TargetSelectionPolicy.LARGEST_BBOX_AREA:
            return max(candidates, key=lambda c: c.bbox_area)
        else:
            return candidates[0]
