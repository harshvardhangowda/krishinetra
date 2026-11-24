"""
KrishiNetra Backend - Prediction API Contracts & Schemas
File: backend/app/contracts/prediction.py

Strictly defines the frontend-compatible prediction schema, separating:
1. Pure neural network outputs (class, confidence, bounding boxes).
2. Curated entomological knowledge base data (scientific names, crops, IPM).
3. Transparent agronomic heuristic estimates (severity, canopy coverage).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBoxPixels(BaseModel):
    """Bounding box pixel coordinates relative to original image."""
    x1: float = Field(..., description="Top-left X coordinate in pixels")
    y1: float = Field(..., description="Top-left Y coordinate in pixels")
    x2: float = Field(..., description="Bottom-right X coordinate in pixels")
    y2: float = Field(..., description="Bottom-right Y coordinate in pixels")


class BoundingBoxNormalized(BaseModel):
    """Normalized bounding box coordinates (0.0 to 1.0) for responsive web canvas."""
    xc: float = Field(..., description="Center X normalized (0.0 - 1.0)")
    yc: float = Field(..., description="Center Y normalized (0.0 - 1.0)")
    w: float = Field(..., description="Box width normalized (0.0 - 1.0)")
    h: float = Field(..., description="Box height normalized (0.0 - 1.0)")


class ModelPrediction(BaseModel):
    """Direct, unaugmented output from the trained YOLOv8 / ONNX neural network."""
    class_id: int = Field(..., description="Zero-indexed model class identifier")
    class_name: str = Field(..., description="Model class label")
    confidence: float = Field(..., description="Model prediction confidence score (0.0 - 1.0)")
    box_pixels: BoundingBoxPixels
    box_normalized: BoundingBoxNormalized


class NonChemicalIPM(BaseModel):
    """Verified non-chemical Integrated Pest Management (IPM) guidelines."""
    cultural_mechanical: str = Field(..., description="Mechanical & cultural intervention strategy")
    biological_control: str = Field(..., description="Biological control agent recommendations")
    bioacoustic_feasibility: str = Field(..., description="Entomological feasibility of acoustic deterrence")
    recommended_stimulus: Optional[str] = Field(None, description="Bioacoustic waveform profile if feasible")
    hearing_organ: Optional[str] = Field(None, description="Anatomical tympanal organ status")
    frequency_range_khz: Optional[List[float]] = Field(None, description="Acoustic sensitivity band in kHz")


class PestKnowledge(BaseModel):
    """Curated, peer-reviewed agronomic and taxonomic data from knowledge base."""
    scientific_name: str = Field(..., description="Primary binomial scientific nomenclature")
    taxa: str = Field(..., description="Taxonomical classification")
    family: str = Field(..., description="Entomological family")
    target_crops: List[str] = Field(default_factory=list, description="Host agricultural crops")
    common_names: List[str] = Field(default_factory=list, description="Common and regional names")
    non_chemical_ipm: NonChemicalIPM


class HeuristicEstimate(BaseModel):
    """Deterministic, transparent agronomic heuristic calculations (NOT neural network weights)."""
    severity: str = Field(..., description="Infestation severity category: LOW, MEDIUM, HIGH")
    canopy_coverage_pct: float = Field(..., description="Estimated percentage of frame covered by pest")
    estimation_basis: str = Field(..., description="Explanation of how heuristic was derived")


class EnrichedDetection(BaseModel):
    """Comprehensive single-pest detection entity with clear source attribution."""
    model_prediction: ModelPrediction
    knowledge_base: PestKnowledge
    heuristic_estimates: HeuristicEstimate


class ImageMetadata(BaseModel):
    """Technical details of the processed image file."""
    width: int = Field(..., description="Original image width in pixels")
    height: int = Field(..., description="Original image height in pixels")
    format: str = Field(..., description="Decoded image format (JPEG, PNG, WEBP)")
    file_size_bytes: int = Field(..., description="Total size of image in bytes")


class PredictionResponse(BaseModel):
    """Root response schema for POST /api/v1/predict."""
    success: bool = Field(True, description="Indicates whether inference completed successfully")
    inference_time_ms: float = Field(..., description="Model and post-processing latency in milliseconds")
    total_detections: int = Field(..., description="Total number of supported pests detected")
    overall_severity: str = Field(..., description="Aggregate severity rating: NONE, LOW, MEDIUM, HIGH")
    image_metadata: ImageMetadata
    detections: List[EnrichedDetection] = Field(
        default_factory=list,
        description="List of enriched detections; empty array if no supported pests detected"
    )
    mode: str = Field(..., description="'authenticated' or 'demo_unauthenticated'")
    hardware_safeguard: str = Field(
        "READ_ONLY_ACTIVE",
        description="Confirms all hardware actuators and acoustic transmitters remain safely disengaged"
    )
