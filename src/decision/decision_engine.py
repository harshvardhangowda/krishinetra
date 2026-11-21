"""
AI-Powered Precision Pest Management - Decision Engine
File: src/decision/decision_engine.py

Separates computer vision inference from hardware actuation.
Evaluates detection packets against:
1. Model validity (blocks baseline COCO models from triggering actions)
2. Confidence thresholds (calibrated to prevent false positives)
3. Target pest whitelist (Armyworm, Blister Beetle, Corn Borer)
4. Beneficial insect safeguards (Honeybees, Ladybugs, Earthworms)
5. Strict Safety Interlocks (physical hardware actuators disabled by default)

Safety Standard: Default state is always NO ACTION.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml


class DecisionEngine:
    """Evaluates detections and determines targeting actions with fail-safe defaults."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        min_confidence_threshold: float = 0.60,
        hardware_safety_lock: bool = True,
    ) -> None:
        """
        Initialize the decision engine.

        Args:
            config_path: Path to configs/target_pests.yaml.
            min_confidence_threshold: Minimum confidence to consider a target candidate.
            hardware_safety_lock: Master safety switch preventing physical actuator signals.
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.hardware_safety_lock = hardware_safety_lock

        self.target_whitelist: Set[str] = set()
        self.beneficial_safelist: Set[str] = set()

        # Load configuration if provided
        if config_path is None:
            default_cfg = Path(__file__).resolve().parent.parent.parent / "configs" / "target_pests.yaml"
            if default_cfg.exists():
                config_path = str(default_cfg)

        if config_path and Path(config_path).exists():
            self._load_config(Path(config_path))
        else:
            # Fallback default target lists
            self.target_whitelist = {"army worm", "blister beetle", "corn borer"}
            self.beneficial_safelist = {"honey bee", "ladybug", "earthworm"}

    def _load_config(self, cfg_path: Path) -> None:
        """Load target pest and beneficial insect lists from YAML."""
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for p in cfg.get("target_classes", []):
            if isinstance(p, dict) and "name" in p:
                self.target_whitelist.add(p["name"].strip().lower())
            elif isinstance(p, str):
                self.target_whitelist.add(p.strip().lower())

        for b in cfg.get("beneficial_classes", []):
            if isinstance(b, dict) and "name" in b:
                self.beneficial_safelist.add(b["name"].strip().lower())
            elif isinstance(b, str):
                self.beneficial_safelist.add(b.strip().lower())

        print(f"[DecisionEngine] Target Whitelist: {sorted(list(self.target_whitelist))}")
        print(f"[DecisionEngine] Beneficial Safelist: {sorted(list(self.beneficial_safelist))}")
        print(f"[DecisionEngine] Hardware Safety Lock: {'ENGAGED (Actuators Disabled)' if self.hardware_safety_lock else 'DISENGAGED'}")

    def evaluate(
        self,
        detection_packet: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate detections from a frame and return an action decision.

        Returns:
            Dict containing decision status, reason, target details, and actuator command.
        """
        # Rule 1: Baseline COCO models MUST NEVER trigger actions
        if detection_packet.get("is_baseline_model", False):
            return {
                "decision": "NO_ACTION",
                "reason": "BASELINE_COCO_MODEL_OBSERVATION_ONLY",
                "target_pest": None,
                "confidence": 0.0,
                "actuator_command": None,
                "safety_interlock_engaged": self.hardware_safety_lock,
            }

        detections = detection_packet.get("detections", [])
        if not detections:
            return {
                "decision": "NO_ACTION",
                "reason": "NO_DETECTIONS_IN_FRAME",
                "target_pest": None,
                "confidence": 0.0,
                "actuator_command": None,
                "safety_interlock_engaged": self.hardware_safety_lock,
            }

        # Check all detections in frame
        highest_target_conf = 0.0
        selected_target = None

        for det in detections:
            cls_name = det["class_name"].strip().lower()
            conf = float(det["confidence"])

            # Rule 2: Beneficial insect safeguard
            if cls_name in self.beneficial_safelist:
                return {
                    "decision": "NO_ACTION",
                    "reason": f"BENEFICIAL_INSECT_DETECTED ({cls_name})",
                    "target_pest": None,
                    "confidence": conf,
                    "actuator_command": None,
                    "safety_interlock_engaged": self.hardware_safety_lock,
                }

            # Rule 3: Target pest whitelist match
            if cls_name in self.target_whitelist:
                if conf >= self.min_confidence_threshold and conf > highest_target_conf:
                    highest_target_conf = conf
                    selected_target = det

        # Rule 4: Action selection based on confidence and target check
        if selected_target is not None:
            # Candidate for action (Deterrence / Targeting)
            action_status = "TARGET_CANDIDATE" if not self.hardware_safety_lock else "TARGET_OBSERVED_SAFETY_LOCKED"
            return {
                "decision": action_status,
                "reason": f"TARGET_PEST_IDENTIFIED ({selected_target['class_name']})",
                "target_pest": selected_target["class_name"],
                "confidence": round(highest_target_conf, 4),
                "bbox": selected_target["bbox"],
                "actuator_command": "TRIGGER_DETERRENCE" if not self.hardware_safety_lock else "NO_ACTION_SAFETY_LOCKED",
                "safety_interlock_engaged": self.hardware_safety_lock,
            }

        # Default fallback: Low confidence or non-target insect
        return {
            "decision": "NO_ACTION",
            "reason": "NON_TARGET_OR_LOW_CONFIDENCE",
            "target_pest": None,
            "confidence": 0.0,
            "actuator_command": None,
            "safety_interlock_engaged": self.hardware_safety_lock,
        }
