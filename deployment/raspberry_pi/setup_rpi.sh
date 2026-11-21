#!/bin/bash
# ==============================================================================
# Krishnetra Precision Pest Management - Raspberry Pi 4 Environment Setup Script
# ==============================================================================
set -e

echo "=== [1/4] Updating system packages ==="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-opencv libcamera-dev libcamera-apps

echo "=== [2/4] Setting up Python virtual environment ==="
python3 -m venv .venv_rpi
source .venv_rpi/bin/activate

echo "=== [3/4] Installing Python edge dependencies ==="
pip install --upgrade pip
pip install -r deployment/raspberry_pi/requirements_rpi.txt

echo "=== [4/4] Verifying model assets ==="
if [ ! -f "models/pest_yolov8n_best.onnx" ]; then
    echo "Warning: models/pest_yolov8n_best.onnx not found. Ensure ONNX checkpoint is transferred."
else
    echo "Model checkpoint verified: models/pest_yolov8n_best.onnx"
fi

echo "=== Setup complete! Run with: ==="
echo "source .venv_rpi/bin/activate"
echo "python deployment/raspberry_pi/rpi_edge_monitor.py --model models/pest_yolov8n_best.onnx --source 0"
