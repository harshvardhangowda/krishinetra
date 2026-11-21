"""
AI-Powered Precision Pest Management - YOLO Training Module
File: src/detection/train_pest_model.py

Trains a lightweight YOLOv8n object detection model on the 3-class target pest
dataset (army worm, blister beetle, corn borer) with reproducible seeds and
automated metric extraction.
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def train_pest_detector(
    data_yaml: str = "datasets/ip102_subset.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    seed: int = 42,
    project_dir: str = "runs/detect",
    name: str = "pest_yolov8n_exp1",
) -> Path:
    """
    Execute YOLO training and export best checkpoint to models/.
    """
    data_path = PROJECT_ROOT / data_yaml
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("KRISHNETRA PEST YOLO TRAINING PIPELINE")
    print(f"Data YAML   : {data_path}")
    print(f"Base Model  : {base_model}")
    print(f"Epochs      : {epochs} | Image Size: {imgsz} | Batch: {batch}")
    print(f"Device      : {device} | Seed: {seed}")
    print("=" * 65)

    model = YOLO(base_model)

    # Train model
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        workers=0,  # Safe for Windows multi-processing
        project=str(PROJECT_ROOT / project_dir),
        name=name,
        exist_ok=True,
        verbose=True,
    )

    # Locate and copy best weights
    exp_dir = PROJECT_ROOT / project_dir / name
    best_pt = exp_dir / "weights" / "best.pt"
    dest_best_pt = models_dir / "pest_yolov8n_best.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, dest_best_pt)
        print(f"\n[Training] Best model saved to: {dest_best_pt}")
    else:
        # Fallback to last.pt if best.pt wasn't created
        last_pt = exp_dir / "weights" / "last.pt"
        if last_pt.exists():
            shutil.copy2(last_pt, dest_best_pt)
            print(f"\n[Training] Model weights saved to: {dest_best_pt}")

    # Evaluate on test split
    print("\n[Evaluation] Validating model on test split...")
    val_metrics = model.val(data=str(data_path), split="test", project=str(PROJECT_ROOT / project_dir), name=f"{name}_val")

    print("\n" + "=" * 65)
    print("TRAINING & VALIDATION SUMMARY")
    print("=" * 65)
    print(f"Precision (P)    : {val_metrics.results_dict.get('metrics/precision(B)', 0.0):.4f}")
    print(f"Recall (R)       : {val_metrics.results_dict.get('metrics/recall(B)', 0.0):.4f}")
    print(f"mAP@50           : {val_metrics.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"mAP@50-95        : {val_metrics.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print(f"Artifacts Path   : {exp_dir}")
    print("=" * 65 + "\n")

    return dest_best_pt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3-Class Pest YOLO Model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or '0')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_pest_detector(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
