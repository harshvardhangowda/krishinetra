# Raspberry Pi 4 Edge Deployment Guide: Krishnetra Pest Detector

This directory contains the edge inference runtime and deployment configuration for running the custom 3-class pest detection model on a Raspberry Pi 4.

---

## 1. Hardware Requirements
* **Single-Board Computer:** Raspberry Pi 4 Model B (4GB or 8GB recommended).
* **Operating System:** Raspberry Pi OS 64-bit (Debian Bullseye or Bookworm).
* **Camera Sensor:** Raspberry Pi Camera Module v2 / v3 / HQ Camera via CSI, or standard USB webcam via V4L2.
* **Storage:** $\ge 16\text{ GB}$ MicroSD (Class 10 / A1).

---

## 2. Directory Layout on Raspberry Pi
```
krishnetra-rpi/
├── deployment/
│   └── raspberry_pi/
│       ├── rpi_edge_monitor.py      # Main edge inference execution script
│       ├── rpi_camera.py            # Hardware camera interface (Picamera2 / OpenCV)
│       ├── db_client.py             # PostgreSQL telemetry client with local fallback
│       ├── requirements_rpi.txt     # Lightweight ARM64 Python dependencies
│       └── setup_rpi.sh             # Automated setup script
├── models/
│   └── pest_yolov8n_best.onnx       # Optimized ONNX model (11.7 MB)
├── configs/
│   └── target_pests.yaml            # Target pest whitelist configuration
└── logs/                            # Local JSONL/CSV failover telemetry
```

---

## 3. Installation & Setup
```bash
# Clone or transfer repository to Raspberry Pi
git clone <repo_url> krishnetra-rpi
cd krishnetra-rpi

# Run automated setup
chmod +x deployment/raspberry_pi/setup_rpi.sh
./deployment/raspberry_pi/setup_rpi.sh
```

---

## 4. Environment Variables (PostgreSQL Telemetry)
Configure PostgreSQL connection variables before running (or leave unset to default to local fallback):
```bash
export PG_HOST="192.168.1.100"       # Central telemetry server IP
export PG_PORT="5432"
export PG_DATABASE="krishnetra_pest_db"
export PG_USER="krishnetra_user"
export PG_PASSWORD="secure_password"
```

---

## 5. Execution Commands
```bash
# Activate virtual environment
source .venv_rpi/bin/activate

# 1. Headless Background Execution with CSI Pi Camera
python deployment/raspberry_pi/rpi_edge_monitor.py --model models/pest_yolov8n_best.onnx --picam2 --headless

# 2. Standard USB Webcam Mode with Local Fallback
python deployment/raspberry_pi/rpi_edge_monitor.py --model models/pest_yolov8n_best.onnx --source 0 --headless

# 3. Interactive Desktop Mode (with HDMI / X11 display)
python deployment/raspberry_pi/rpi_edge_monitor.py --model models/pest_yolov8n_best.onnx --source 0 --gui
```

---

## 6. Safety Guarantee
* The edge monitoring software is **observation and telemetry only**.
* Physical actuators (motors, relays, deterrence speakers) remain disconnected and disabled in the decision layer (`NO_ACTION` state enforced).
