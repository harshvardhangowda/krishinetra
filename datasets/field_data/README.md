# Field Data Collection Foundation - Krishnetra Aavishkar

This directory is designated for real-world field imagery captured from the prototype camera or Raspberry Pi 4.

## Directory Layout
- `images/`: High-resolution raw RGB images captured in agricultural fields.
- `labels/`: Normalized YOLO format text annotations (`<class_id> <xc> <yc> <w> <h>`).

## Class Schema
- `0`: army worm
- `1`: blister beetle
- `2`: corn borer

## Domain Adaptation Protocol
1. Capture images across various lighting conditions (direct sunlight, shade, golden hour).
2. Capture pests at variable rover-to-crop distances (15cm to 100cm).
3. Annotate using standard bounding-box tools (CVAT / LabelImg / Roboflow).
4. Combine with the baseline IP102 dataset for domain-adaptation fine-tuning (Milestone 7).

*Current Status: Field data collection protocol initialized. Real field images pending field trials.*
