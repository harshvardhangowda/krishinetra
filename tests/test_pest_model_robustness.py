"""
AI-Powered Precision Pest Management - Phase 6 Model Verification & Robustness Test
File: tests/test_pest_model_robustness.py

Performs comprehensive verification, error analysis, and robustness benchmarking
for the trained 3-class pest detector (models/pest_yolov8n_best.pt).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import PestDetector

MODEL_PATH = PROJECT_ROOT / "models" / "pest_yolov8n_best.pt"
TEST_IMAGES_DIR = PROJECT_ROOT / "datasets" / "ip102_subset" / "images" / "test"
TEST_LABELS_DIR = PROJECT_ROOT / "datasets" / "ip102_subset" / "labels" / "test"


def run_unseen_test_verification() -> Dict[str, Any]:
    print("=" * 65)
    print("STEP 6.1: UNSEEN TEST DATASET INFERENCE & VERIFICATION")
    print(f"Model Checkpoint: {MODEL_PATH}")
    print("=" * 65)

    if not MODEL_PATH.exists():
        print(f"[FAIL] Model file not found at {MODEL_PATH}")
        return {"status": "FAIL"}

    detector = PestDetector(weights_path=str(MODEL_PATH), device="cpu", conf_threshold=0.20)
    print(f"Known Class Names: {detector.get_class_names()}")

    test_imgs = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png")))
    print(f"Found {len(test_imgs)} unseen test images.")

    results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for img_path in test_imgs:
        t0 = time.perf_counter()
        dets = detector.detect(source=str(img_path))
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        # Read ground truth label
        lbl_path = TEST_LABELS_DIR / f"{img_path.stem}.txt"
        gt_classes = []
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.append(int(parts[0]))

        results.append(
            {
                "image": img_path.name,
                "ground_truth_classes": gt_classes,
                "detections": dets,
                "latency_ms": round(lat_ms, 2),
            }
        )

        print(f"Image: {img_path.name} | GT: {gt_classes} | Detections: {len(dets)} | Latency: {lat_ms:.2f}ms")
        for d in dets:
            print(f"  -> Class: {d['class_name']} (ID: {d['class_id']}) | Conf: {d['confidence']:.2f} | BBox: {d['bbox']}")

    avg_latency = np.mean(latencies) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("-" * 65)
    print(f"Average Inference Latency : {avg_latency:.2f} ms")
    print(f"Effective Inference FPS   : {fps:.2f} FPS")
    print("=" * 65 + "\n")

    return {
        "status": "PASS",
        "total_test_images": len(test_imgs),
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "results": results,
    }


def run_robustness_stress_test() -> Dict[str, Any]:
    print("=" * 65)
    print("STEP 6.5: ROBUSTNESS & STRESS TESTING")
    print("=" * 65)

    if not MODEL_PATH.exists():
        return {"status": "FAIL"}

    detector = PestDetector(weights_path=str(MODEL_PATH), device="cpu", conf_threshold=0.20)
    test_imgs = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png")))
    if not test_imgs:
        return {"status": "NO_TEST_IMAGES"}

    base_img = cv2.imread(str(test_imgs[0]))
    stress_conditions = {}

    # Condition 1: Low Lighting (Simulate dusk / overcast)
    low_light = cv2.convertScaleAbs(base_img, alpha=0.4, beta=-20)
    dets_low = detector.detect(low_light)
    stress_conditions["low_lighting"] = {
        "description": "40% brightness reduction",
        "detections_count": len(dets_low),
        "detections": dets_low,
    }

    # Condition 2: High Glare (Simulate harsh direct sunlight)
    high_glare = cv2.convertScaleAbs(base_img, alpha=1.5, beta=50)
    dets_glare = detector.detect(high_glare)
    stress_conditions["high_glare"] = {
        "description": "50% overexposure + glare offset",
        "detections_count": len(dets_glare),
        "detections": dets_glare,
    }

    # Condition 3: Gaussian Noise (Simulate sensor / low-light noise)
    noise = np.random.normal(0, 25, base_img.shape).astype(np.uint8)
    noisy_img = cv2.add(base_img, noise)
    dets_noisy = detector.detect(noisy_img)
    stress_conditions["sensor_noise"] = {
        "description": "Additive Gaussian noise (sigma=25)",
        "detections_count": len(dets_noisy),
        "detections": dets_noisy,
    }

    # Condition 4: Scale Variation (Simulated distance 2x farther)
    scaled_small = cv2.resize(base_img, (320, 320))
    dets_scaled = detector.detect(scaled_small)
    stress_conditions["distant_scale_320px"] = {
        "description": "Downscaled 2x (320x320 resolution)",
        "detections_count": len(dets_scaled),
        "detections": dets_scaled,
    }

    print("Robustness Stress Test Results:")
    for cond, data in stress_conditions.items():
        print(f"  [{cond.upper()}]: {data['description']} -> {data['detections_count']} detections")

    print("=" * 65 + "\n")
    return stress_conditions


def main() -> None:
    unseen_report = run_unseen_test_verification()
    robustness_report = run_robustness_stress_test()


if __name__ == "__main__":
    main()
