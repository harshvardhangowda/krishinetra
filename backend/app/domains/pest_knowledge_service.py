"""
KrishiNetra Backend - Pest Knowledge & Agronomic Enrichment Service
File: backend/app/domains/pest_knowledge_service.py

Loads verified entomological and acoustic data from:
1. configs/target_pests.yaml
2. src/research/pest_acoustic_profile.yaml

Enriches raw computer vision detections with scientific taxonomy, host crop index,
verified non-chemical IPM recommendations, and transparent heuristic severity estimates.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from backend.app.contracts.prediction import (
    PestKnowledge,
    NonChemicalIPM,
    HeuristicEstimate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class PestKnowledgeService:
    """Service providing verified scientific metadata and deterministic agronomic heuristics."""

    def __init__(self) -> None:
        self.target_pests_cfg = self._load_target_pests_cfg()
        self.acoustic_profiles = self._load_acoustic_profiles()

    def _load_target_pests_cfg(self) -> Dict[str, Any]:
        cfg_path = PROJECT_ROOT / "configs" / "target_pests.yaml"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[PestKnowledgeService] Warning: Failed to parse target_pests.yaml: {e}")
        return {}

    def _load_acoustic_profiles(self) -> Dict[str, Any]:
        profile_path = PROJECT_ROOT / "src" / "research" / "pest_acoustic_profile.yaml"
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    return data.get("pests", {})
            except Exception as e:
                print(f"[PestKnowledgeService] Warning: Failed to parse pest_acoustic_profile.yaml: {e}")
        return {}

    def get_knowledge(self, class_id: int, class_name: str) -> PestKnowledge:
        """
        Enrich a detection with peer-reviewed entomological knowledge.
        Maps model class_id (0: army worm, 1: blister beetle, 2: corn borer) to scientific data.
        """
        normalized_name = class_name.lower().replace("-", " ").replace("_", " ").strip()

        # Class 0: Army Worm (Spodoptera frugiperda / litura)
        if class_id == 0 or "army" in normalized_name:
            profile = self.acoustic_profiles.get("army_worm", {})
            return PestKnowledge(
                scientific_name="Spodoptera frugiperda",
                taxa=profile.get("taxa", "Spodoptera frugiperda / Spodoptera litura (Family: Noctuidae)"),
                family="Noctuidae",
                target_crops=["Maize", "Cotton", "Soybean", "Sorghum", "Rice"],
                common_names=["Fall Armyworm", "Spodoptera litura", "Armyworm caterpillar"],
                non_chemical_ipm=NonChemicalIPM(
                    cultural_mechanical=(
                        "Install pheromone traps (5/acre) for population monitoring; "
                        "hand-pick egg masses and early-instar larvae during morning scouting; "
                        "apply dry fine sand/sawdust into crop whorls."
                    ),
                    biological_control=(
                        "Release Trichogramma chilonis egg parasitoids (50,000/acre); "
                        "foliar spray of Bacillus thuringiensis (Bt) or 5% Neem Seed Kernel Extract (NSKE)."
                    ),
                    bioacoustic_feasibility="FEASIBLE (adult moths respond to pulsed ultrasound with evasive maneuvers)",
                    recommended_stimulus=profile.get("recommended_stimulus", "Frequency-modulated swept pulses (25 - 45 kHz, 80-95 dB SPL @ 0.5m)"),
                    hearing_organ=profile.get("hearing_organ", "Metathoracic tympanal organs (A1 and A2 sensory cells)"),
                    frequency_range_khz=profile.get("peak_sensitivity_khz", [20.0, 45.0]),
                ),
            )

        # Class 1: Blister Beetle (Mylabris phalerata / Epicauta spp.)
        elif class_id == 1 or "blister" in normalized_name:
            profile = self.acoustic_profiles.get("blister_beetle", {})
            return PestKnowledge(
                scientific_name="Mylabris phalerata",
                taxa=profile.get("taxa", "Mylabris phalerata / Epicauta spp. (Family: Meloidae)"),
                family="Meloidae",
                target_crops=["Pigeonpea (Tur)", "Mung bean", "Urad", "Cowpea", "Cotton"],
                common_names=["Blister beetle", "Red-headed blister beetle", "Cantharidin beetle"],
                non_chemical_ipm=NonChemicalIPM(
                    cultural_mechanical=(
                        "Manual collection with protective gloves and eyewear into kerosene-water mixture. "
                        "CAUTION: Beetles synthesize toxic vesicant cantharidin which causes severe skin blistering. "
                        "Install yellow sticky or light traps around field borders."
                    ),
                    biological_control=(
                        "Foliar application of 5% Neem Seed Kernel Extract (NSKE) as antifeedant; "
                        "erect physical barrier netting around pulse crops."
                    ),
                    bioacoustic_feasibility=(
                        "NOT FEASIBLE (Blister beetles completely lack tympanal hearing organs; "
                        "acoustic deterrence is scientifically unsupported and not recommended)"
                    ),
                    recommended_stimulus=None,
                    hearing_organ=profile.get("hearing_organ", "NONE (Acoustic tympanal organs are absent in Meloidae)"),
                    frequency_range_khz=None,
                ),
            )

        # Class 2: Corn Borer (Chilo partellus / Ostrinia nubilalis)
        elif class_id == 2 or "borer" in normalized_name:
            profile = self.acoustic_profiles.get("corn_borer", {})
            return PestKnowledge(
                scientific_name="Chilo partellus",
                taxa=profile.get("taxa", "Ostrinia nubilalis / Chilo partellus (Family: Crambidae / Pyralidae)"),
                family="Crambidae / Pyralidae",
                target_crops=["Maize", "Sorghum (Jowar)", "Sweet corn", "Sugarcane"],
                common_names=["Spotted stem borer", "European corn borer", "Stalk borer"],
                non_chemical_ipm=NonChemicalIPM(
                    cultural_mechanical=(
                        "Destroy crop stubbles and residues post-harvest to kill diapausing larvae; "
                        "intercrop maize with non-host legumes (Cowpea/Desmodium); "
                        "deploy light traps (1/acre) to disrupt mating flights."
                    ),
                    biological_control=(
                        "Release Trichogramma chilonis at 50,000/ha; "
                        "apply Bacillus thuringiensis kurstaki (Btk) during early pre-boring stage."
                    ),
                    bioacoustic_feasibility="FEASIBLE (tympanal sensitivity to bat echolocation ultrasonic mimic pulses)",
                    recommended_stimulus=profile.get("recommended_stimulus", "Bat echolocation mimic pulses (30 - 55 kHz, pulsed at 10-25 Hz repetition rate)"),
                    hearing_organ=profile.get("hearing_organ", "Abdominal tympanal organs (sensilla scolopophora)"),
                    frequency_range_khz=profile.get("peak_sensitivity_khz", [25.0, 50.0]),
                ),
            )

        # Fallback for unrecognized classes
        return PestKnowledge(
            scientific_name=f"Unclassified Pest ({class_name})",
            taxa="Insecta",
            family="Unspecified",
            target_crops=["Field Crops"],
            common_names=[class_name],
            non_chemical_ipm=NonChemicalIPM(
                cultural_mechanical="Implement standard sanitation and physical sticky traps.",
                biological_control="Deploy general beneficial predators (ladybugs, lacewings).",
                bioacoustic_feasibility="UNVERIFIED (Requires biological hearing organ evaluation)",
                recommended_stimulus=None,
                hearing_organ=None,
                frequency_range_khz=None,
            ),
        )

    def compute_heuristic_estimates(
        self,
        box_pixels: Dict[str, float],
        image_width: int,
        image_height: int,
        confidence: float,
        total_pests: int,
    ) -> HeuristicEstimate:
        """
        Calculates transparent, deterministic heuristic estimates.
        Uses bounding box pixel dimensions and pest count to gauge localized canopy infestation.
        """
        w = max(1.0, box_pixels["x2"] - box_pixels["x1"])
        h = max(1.0, box_pixels["y2"] - box_pixels["y1"])
        box_area = w * h
        total_image_area = max(1.0, float(image_width * image_height))
        canopy_coverage_pct = round(min(100.0, (box_area / total_image_area) * 100.0), 2)

        # Severity categorization rules:
        # LOW: single pest covering < 4% frame area
        # MEDIUM: 2-3 pests or covering 4% - 12% frame area
        # HIGH: >3 pests or covering > 12% frame area (or high pest density)
        if canopy_coverage_pct >= 12.0 or total_pests >= 4:
            severity = "HIGH"
        elif canopy_coverage_pct >= 4.0 or total_pests >= 2:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return HeuristicEstimate(
            severity=severity,
            canopy_coverage_pct=canopy_coverage_pct,
            estimation_basis=(
                f"Calculated from bounding box pixel area ({canopy_coverage_pct}% of frame) "
                f"and total frame pest density ({total_pests} detections). "
                "Deterministic agronomic heuristic, not neural network output."
            ),
        )

    def compute_overall_severity(self, heuristic_severities: List[str]) -> str:
        """Determines peak severity level across all detections in the image."""
        if not heuristic_severities:
            return "NONE"
        if "HIGH" in heuristic_severities:
            return "HIGH"
        if "MEDIUM" in heuristic_severities:
            return "MEDIUM"
        if "LOW" in heuristic_severities:
            return "LOW"
        return "NONE"


pest_knowledge_service = PestKnowledgeService()
