"""
AI-Powered Precision Pest Management - Milestone 1 & 2 Verification
File: tests/test_first_yolo.py

This test program validates the AI environment:
1. Confirms Python, PyTorch, CUDA, Ultralytics, and OpenCV dependencies.
2. Loads a pretrained YOLO model (yolov8n.pt).
3. Runs inference on a test image.
4. Prints structured detections (Class, Confidence, Bounding Box).
5. Saves the resulting annotated image with bounding boxes.

IMPORTANT: This baseline test proves that the computer-vision environment works.
It is NOT yet trained on agricultural pest datasets (e.g., IP102).
"""

import sys
from pathlib import Path

# Add project root to path for modular imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_env_check() -> None:
    """Print system and framework diagnostics."""
    import cv2
    import numpy as np
    import torch
    import ultralytics

    print("=" * 60)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    print(f"Python Version      : {sys.version.split()[0]}")
    print(f"PyTorch Version     : {torch.__version__}")
    print(f"CUDA Available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"Ultralytics Version : {ultralytics.__version__}")
    print(f"OpenCV Version      : {cv2.__version__}")
    print(f"NumPy Version       : {np.__version__}")
    print("=" * 60)


def create_or_get_test_image(output_path: Path) -> Path:
    """
    Ensure a test image exists for inference.
    If an external sample cannot be fetched, generate a synthetic RGB image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    # Try downloading standard Ultralytics sample or create synthetic
    try:
        import urllib.request

        sample_url = "https://ultralytics.com/images/bus.jpg"
        print(f"[Test Setup] Fetching sample verification image from {sample_url}...")
        urllib.request.urlretrieve(sample_url, output_path)
        print(f"[Test Setup] Sample image saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"[Test Setup] Download failed ({e}), creating synthetic image...")
        import cv2
        import numpy as np

        # Create a synthetic image (640x640 green foliage background)
        img = np.full((640, 640, 3), (34, 139, 34), dtype=np.uint8)  # Forest green
        cv2.putText(
            img,
            "Krishnetra Pest Detector Test",
            (40, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(str(output_path), img)
        return output_path


def main() -> None:
    print("\n[AI-Powered Precision Pest Management] Starting First YOLO Test...\n")
    run_env_check()

    from src.detection.detector import PestDetector

    # Step 1: Load pretrained YOLO model
    print("\n[Step 1] Loading Pretrained YOLO Model...")
    detector = PestDetector(weights_path="yolov8n.pt", conf_threshold=0.25)

    # Step 2: Load test image
    print("\n[Step 2] Preparing Test Image...")
    test_image_path = PROJECT_ROOT / "tests" / "data" / "test_sample.jpg"
    resolved_path = create_or_get_test_image(test_image_path)
    print(f"Test image path: {resolved_path}")

    # Step 3: Run inference
    print("\n[Step 3] Running Inference...")
    detections = detector.detect(
        source=str(resolved_path),
        save_annotated=True,
        output_dir=str(PROJECT_ROOT / "runs" / "detect"),
    )

    # Step 4: Print detections
    print("\n" + "=" * 60)
    print("DETECTION PIPELINE RESULTS")
    print("=" * 60)
    print("Image")
    print("  ↓")
    print("YOLO")
    print("  ↓")

    if not detections:
        print("No objects detected above confidence threshold.")
    else:
        for idx, det in enumerate(detections, 1):
            print(f"[Detection #{idx}]")
            print(f"  Object detected : True")
            print(f"  Class           : {det['class_name']} (ID: {det['class_id']})")
            print(f"  Confidence      : {det['confidence']:.2f}")
            print(f"  Bounding box    : {det['bbox']}")
            print("  ---")

    # Step 5: Verify saved output
    print("=" * 60)
    output_dir = PROJECT_ROOT / "runs" / "detect"
    print(f"[Step 5] Annotated detections saved in: {output_dir}")
    print("\n[STATUS] Milestone 1 & 2 verification complete: AI environment is functional!")
    print("[NOTE] This pretrained baseline model validates the environment only.")
    print("       Custom agricultural pest training (IP102) will follow in Milestone 3 & 4.\n")


if __name__ == "__main__":
    main()
