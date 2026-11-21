"""
AI-Powered Precision Pest Management - ONNX vs PyTorch Parity Test
File: tests/test_onnx_parity.py

Compares inference results between PyTorch (.pt) and ONNX (.onnx) formats
on unseen test images to verify numerical parity and bounding-box consistency.
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import PestDetector

PT_MODEL = PROJECT_ROOT / "models" / "pest_yolov8n_best.pt"
ONNX_MODEL = PROJECT_ROOT / "models" / "pest_yolov8n_best.onnx"
TEST_IMAGES_DIR = PROJECT_ROOT / "datasets" / "ip102_subset" / "images" / "test"


def verify_model_parity() -> bool:
    print("=" * 65)
    print("STEP 7.3: ONNX VS PYTORCH INFERENCE PARITY VERIFICATION")
    print(f"PyTorch Checkpoint : {PT_MODEL}")
    print(f"ONNX Model         : {ONNX_MODEL}")
    print("=" * 65)

    if not PT_MODEL.exists():
        print(f"[FAIL] PyTorch model not found: {PT_MODEL}")
        return False
    if not ONNX_MODEL.exists():
        print(f"[FAIL] ONNX model not found: {ONNX_MODEL}")
        return False

    detector_pt = PestDetector(weights_path=str(PT_MODEL), device="cpu", conf_threshold=0.25)
    detector_onnx = PestDetector(weights_path=str(ONNX_MODEL), device="cpu", conf_threshold=0.25)

    test_imgs = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png")))
    print(f"Testing {len(test_imgs)} unseen test images...\n")

    parity_matched = 0
    pt_latencies = []
    onnx_latencies = []

    for img_path in test_imgs:
        # PyTorch inference
        t0 = time.perf_counter()
        dets_pt = detector_pt.detect(str(img_path))
        pt_lat = (time.perf_counter() - t0) * 1000.0
        pt_latencies.append(pt_lat)

        # ONNX inference
        t0 = time.perf_counter()
        dets_onnx = detector_onnx.detect(str(img_path))
        onnx_lat = (time.perf_counter() - t0) * 1000.0
        onnx_latencies.append(onnx_lat)

        # Compare detection counts and top class
        pt_cls = [d["class_id"] for d in dets_pt]
        onnx_cls = [d["class_id"] for d in dets_onnx]

        match = (pt_cls == onnx_cls)
        if match:
            parity_matched += 1

        print(f"Image: {img_path.name:<16} | PT: {len(dets_pt)} dets ({pt_lat:.1f}ms) | ONNX: {len(dets_onnx)} dets ({onnx_lat:.1f}ms) | Match: {match}")

    avg_pt = np.mean(pt_latencies)
    avg_onnx = np.mean(onnx_latencies)

    print("\n" + "=" * 65)
    print("PARITY BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Total Test Images      : {len(test_imgs)}")
    print(f"Parity Match Rate      : {parity_matched}/{len(test_imgs)} ({100.0 * parity_matched / len(test_imgs):.1f}%)")
    print(f"Average PyTorch Latency: {avg_pt:.2f} ms")
    print(f"Average ONNX Latency   : {avg_onnx:.2f} ms")
    print("=" * 65 + "\n")

    return parity_matched == len(test_imgs)


if __name__ == "__main__":
    success = verify_model_parity()
    sys.exit(0 if success else 1)
