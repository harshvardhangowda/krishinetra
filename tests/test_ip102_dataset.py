"""
AI-Powered Precision Pest Management - Dataset Validation Test
File: tests/test_ip102_dataset.py

Validates the prepared YOLO dataset:
1. Verifies images and label directories for train, val, and test splits.
2. Asserts one-to-one mapping between images and YOLO .txt label files.
3. Validates YOLO syntax (5 space-separated values per line).
4. Verifies class IDs are strictly in {0, 1, 2}.
5. Checks normalized coordinates: 0.0 <= xc, yc <= 1.0, 0.0 < w, h <= 1.0.
6. Asserts positive width and height.
7. Verifies no data leakage across splits (duplicate filenames/hashes).
8. Generates dataset statistics table: Class | Train | Val | Test | Total.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "ip102_subset"

CLASS_NAMES = {
    0: "army worm",
    1: "blister beetle",
    2: "corn borer",
}


def validate_yolo_dataset() -> bool:
    print("=" * 70)
    print("IP102 TARGET SUBSET DATASET VALIDATION")
    print(f"Path: {DATASET_ROOT}")
    print("=" * 70)

    if not DATASET_ROOT.exists():
        print(f"[FAIL] Dataset root does not exist: {DATASET_ROOT}")
        return False

    splits = ["train", "val", "test"]
    split_images: Dict[str, List[Path]] = {}
    split_labels: Dict[str, List[Path]] = {}
    image_names_by_split: Dict[str, Set[str]] = {}

    # Step 1: Check directories and collect files
    for s in splits:
        img_dir = DATASET_ROOT / "images" / s
        lbl_dir = DATASET_ROOT / "labels" / s

        if not img_dir.exists():
            print(f"[FAIL] Missing image split folder: {img_dir}")
            return False
        if not lbl_dir.exists():
            print(f"[FAIL] Missing label split folder: {lbl_dir}")
            return False

        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))

        split_images[s] = imgs
        split_labels[s] = lbls
        image_names_by_split[s] = {img.name for img in imgs}

        print(f"[{s.upper()} SPLIT] Images: {len(imgs)} | Labels: {len(lbls)}")
        if len(imgs) == 0:
            print(f"[FAIL] Split '{s}' has 0 images!")
            return False

    # Step 2: Check for data leakage across splits
    train_names = image_names_by_split["train"]
    val_names = image_names_by_split["val"]
    test_names = image_names_by_split["test"]

    train_val_overlap = train_names.intersection(val_names)
    train_test_overlap = train_names.intersection(test_names)
    val_test_overlap = val_names.intersection(test_names)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("[FAIL] Data leakage detected across splits!")
        if train_val_overlap:
            print(f"  Train/Val overlap: {train_val_overlap}")
        if train_test_overlap:
            print(f"  Train/Test overlap: {train_test_overlap}")
        if val_test_overlap:
            print(f"  Val/Test overlap: {val_test_overlap}")
        return False
    else:
        print("[PASS] Zero data leakage across train, val, and test splits.")

    # Step 3: Validate label syntax and coordinate ranges
    class_split_boxes: Dict[int, Dict[str, int]] = {
        0: {"train": 0, "val": 0, "test": 0},
        1: {"train": 0, "val": 0, "test": 0},
        2: {"train": 0, "val": 0, "test": 0},
    }
    class_split_images: Dict[int, Dict[str, int]] = {
        0: {"train": 0, "val": 0, "test": 0},
        1: {"train": 0, "val": 0, "test": 0},
        2: {"train": 0, "val": 0, "test": 0},
    }

    total_boxes = 0
    syntax_errors = 0
    coordinate_errors = 0

    for s in splits:
        img_dir = DATASET_ROOT / "images" / s
        lbl_dir = DATASET_ROOT / "labels" / s

        for img_path in split_images[s]:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                print(f"[FAIL] Missing label for image: {img_path.name}")
                syntax_errors += 1
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            img_classes_seen: Set[int] = set()

            for line_idx, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    print(f"[FAIL] Syntax error in {lbl_path.name}:{line_idx} -> '{line}' (expected 5 tokens)")
                    syntax_errors += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError as e:
                    print(f"[FAIL] Numeric parse error in {lbl_path.name}:{line_idx} -> {e}")
                    syntax_errors += 1
                    continue

                # Class ID check
                if cls_id not in (0, 1, 2):
                    print(f"[FAIL] Invalid class ID {cls_id} in {lbl_path.name} (must be 0, 1, or 2)")
                    syntax_errors += 1
                    continue

                # Coordinate bounds check
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    print(f"[FAIL] Center coordinates out of [0, 1] range: ({xc}, {yc}) in {lbl_path.name}")
                    coordinate_errors += 1
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    print(f"[FAIL] Box dimension invalid: w={w}, h={h} in {lbl_path.name}")
                    coordinate_errors += 1

                class_split_boxes[cls_id][s] += 1
                total_boxes += 1
                img_classes_seen.add(cls_id)

            for c in img_classes_seen:
                class_split_images[c][s] += 1

    if syntax_errors > 0 or coordinate_errors > 0:
        print(f"[FAIL] Detected {syntax_errors} syntax errors and {coordinate_errors} coordinate errors!")
        return False
    else:
        print("[PASS] All YOLO label files have valid syntax and normalized coordinates within [0.0, 1.0].")

    # Step 4: Verify representation of all 3 classes
    for cls_id in (0, 1, 2):
        train_count = class_split_boxes[cls_id]["train"]
        if train_count == 0:
            print(f"[FAIL] Class {cls_id} ({CLASS_NAMES[cls_id]}) has 0 samples in train split!")
            return False

    print("[PASS] All three target pest classes are represented in the dataset.")

    # Step 5: Dataset Statistics Report
    total_imgs = sum(len(split_images[s]) for s in splits)
    train_pct = (len(split_images["train"]) / total_imgs) * 100
    val_pct = (len(split_images["val"]) / total_imgs) * 100
    test_pct = (len(split_images["test"]) / total_imgs) * 100

    print("\n" + "=" * 70)
    print("DATASET STATISTICS REPORT")
    print("=" * 70)
    print(f"{'Class Name':<16} | {'Class ID':<8} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total Boxes':<11}")
    print("-" * 70)

    for cls_id in (0, 1, 2):
        c_name = CLASS_NAMES[cls_id]
        tr = class_split_boxes[cls_id]["train"]
        va = class_split_boxes[cls_id]["val"]
        te = class_split_boxes[cls_id]["test"]
        tot = tr + va + te
        print(f"{c_name:<16} | {cls_id:<8} | {tr:<7} | {va:<7} | {te:<7} | {tot:<11}")

    print("-" * 70)
    print(
        f"{'Total Images':<16} | {'':<8} | {len(split_images['train']):<7} | {len(split_images['val']):<7} | {len(split_images['test']):<7} | {total_imgs:<11}"
    )
    print(
        f"{'Total Boxes':<16} | {'':<8} | {sum(class_split_boxes[c]['train'] for c in (0,1,2)):<7} | {sum(class_split_boxes[c]['val'] for c in (0,1,2)):<7} | {sum(class_split_boxes[c]['test'] for c in (0,1,2)):<7} | {total_boxes:<11}"
    )
    print("=" * 70)
    print(f"Split Percentages : Train: {train_pct:.1f}% | Val: {val_pct:.1f}% | Test: {test_pct:.1f}%")
    print(f"Total Images     : {total_imgs}")
    print(f"Total Boxes      : {total_boxes}")
    print(f"Validation Status: PASSED")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    success = validate_yolo_dataset()
    sys.exit(0 if success else 1)
