"""
AI-Powered Precision Pest Management - Safe Simulation-Only Decision Layer
File: src/decision/simulation_engine.py

Demonstrates the precision pest deterrence pipeline in a completely safe,
simulation-only mode.

STRICT SAFETY RULES:
- Physical hardware actuation is PERMANENTLY DISABLED.
- No GPIO, PWM, relay, speaker, or motor calls are executed.
- Output explicitly declares "DETERRENCE NOT ACTIVATED".
- Recommends treatment frequencies ONLY when supported by scientific literature.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.decision.decision_engine import DecisionEngine


class SimulationDecisionLayer:
    """Safe simulation layer that computes theoretical deterrence recommendations."""

    def __init__(
        self,
        acoustic_profile_path: Optional[str] = None,
        target_config_path: Optional[str] = None,
    ):
        """
        Initialize the simulation engine with scientific acoustic profiles.
        """
        self.decision_engine = DecisionEngine(
            config_path=target_config_path,
            min_confidence_threshold=0.60,
            hardware_safety_lock=True,  # Safety lock strictly engaged
        )

        self.acoustic_profiles: Dict[str, Any] = {}
        if acoustic_profile_path is None:
            default_profile = Path(__file__).resolve().parent.parent / "research" / "pest_acoustic_profile.yaml"
            if default_profile.exists():
                acoustic_profile_path = str(default_profile)

        if acoustic_profile_path and Path(acoustic_profile_path).exists():
            with open(acoustic_profile_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                self.acoustic_profiles = loaded.get("pests", {})

    def simulate_detection_event(
        self,
        detection_packet: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a CV detection packet and return a simulation-only recommendation report.

        Args:
            detection_packet: Standard detection packet from YOLO inference.

        Returns:
            Dictionary containing structured simulation output.
        """
        # 1. Query the base decision engine
        decision_result = self.decision_engine.evaluate(detection_packet)

        target_pest = decision_result.get("target_pest")
        confidence = decision_result.get("confidence", 0.0)

        # 2. Map pest name to scientific acoustic profile
        normalized_name = target_pest.replace(" ", "_").lower() if target_pest else ""
        pest_profile = self.acoustic_profiles.get(normalized_name, {})

        treatment_recommendation = "NO_ACTION"
        scientific_justification = "No target pest identified or confidence below threshold."
        evidence_level = "N/A"

        if target_pest:
            if normalized_name == "army_worm":
                treatment_recommendation = (
                    "Frequency-modulated swept ultrasound (25.0 - 45.0 kHz, 85 dB SPL @ 0.5m, 15 Hz PRF)"
                )
                scientific_justification = (
                    "Targets tympanal organ sensory cells A1/A2, triggering evasive diving behavior (Fullard 1987)."
                )
                evidence_level = "Strong (Adult Tympanal Evasion)"

            elif normalized_name == "corn_borer":
                treatment_recommendation = (
                    "Pulsed ultrasound bursts (25.0 - 50.0 kHz, 85 dB SPL @ 0.5m, 20 Hz PRF)"
                )
                scientific_justification = (
                    "Triggers abdominal tympanal evasion and flight disruption (Belton & Kempster 1962)."
                )
                evidence_level = "Moderate (Laboratory Flight Chambers)"

            elif normalized_name == "blister_beetle":
                treatment_recommendation = "NO SUFFICIENT EVIDENCE FOUND (Acoustic deterrence NOT recommended)"
                scientific_justification = (
                    "Meloidae lack tympanal hearing organs; defense is chemical (cantharidin). Acoustic deterrence ineffective."
                )
                evidence_level = "Insufficient / Non-Existent"

            else:
                treatment_recommendation = "NO SUFFICIENT EVIDENCE FOUND"
                scientific_justification = f"No scientific acoustic profile available for {target_pest}."
                evidence_level = "Insufficient"

        # 3. Format structured simulation output
        simulation_report = {
            "simulation_mode": True,
            "deterrence_status": "DETERRENCE NOT ACTIVATED",
            "hardware_action": "DISABLED",
            "physical_actuator_state": "DISCONNECTED / LOCKED",
            "pest_detected": target_pest if target_pest else "None",
            "confidence": round(confidence, 4),
            "recommended_treatment": treatment_recommendation,
            "scientific_justification": scientific_justification,
            "evidence_level": evidence_level,
            "decision_engine_status": decision_result["decision"],
            "safety_interlock_engaged": True,
        }

        return simulation_report

    def format_cli_output(self, sim_report: Dict[str, Any]) -> str:
        """Format the report into clean human-readable CLI text."""
        lines = [
            "==================================================",
            "   KRISHNETRA SIMULATION-ONLY DECISION LAYER",
            "==================================================",
            f"DETERRENCE STATUS      : {sim_report['deterrence_status']}",
            f"HARDWARE ACTION        : {sim_report['hardware_action']}",
            f"PHYSICAL ACTUATORS     : {sim_report['physical_actuator_state']}",
            "--------------------------------------------------",
            f"Pest detected          : {sim_report['pest_detected']}",
            f"Confidence             : {sim_report['confidence']:.2f}",
            f"Recommended treatment  : {sim_report['recommended_treatment']}",
            f"Evidence Level         : {sim_report['evidence_level']}",
            f"Scientific Rationale   : {sim_report['scientific_justification']}",
            "==================================================",
        ]
        return "\n".join(lines)
