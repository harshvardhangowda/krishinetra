# Krishnetra Aavishkar: AI-Powered Precision Pest Management

Precision agriculture prototype utilizing computer vision to detect and identify harmful agricultural pests and selectively trigger non-chemical deterrence mechanisms.

CAMERA
  ↓
Raspberry Pi 4 / Edge AI
  ↓
YOLO Pest Detection Model
  ↓
Decision Engine (Confidence & Target Pest Verification)
  ├── Beneficial / Non-Target Insect → NO ACTION
  └── Target Harmful Pest → TARGET / DETER
        ↓
    MQTT / Wi-Fi
        ↓
    Microcontroller (ESP32)
        ↓
    Mecanum Rover + Targeted Non-Chemical Deterrence (Sound/Frequency)
        ↓
    Monitoring & Dashboard
```

---

## Current Development Milestone

- [x] **Milestone 1:** Python + YOLO environment working
- [x] **Milestone 2:** Pretrained YOLO detects objects (baseline environment test)
- [x] **Milestone 3:** Target pest selection (Armyworm, Blister Beetle, Corn Borer)
- [x] **Milestone 4:** IP102 dataset preparation & YOLO format conversion verified
- [x] **Milestone 5:** 3-Class pest detection model trained & evaluated (YOLOv8n)
- [x] **Milestone 6:** Real-world error analysis & field dataset foundation
- [x] **Milestone 7:** Raspberry Pi 4 edge deployment package & ONNX export
- [x] **Milestone 8:** PostgreSQL analytics & Krishnetra Streamlit monitoring dashboard
- [x] **Phase 9:** Scientific pest acoustic research & controlled deterrence bioassay design
- [x] **Phase 10:** ESP32 communication interface architecture & safe command protocol (Simulation Only)
- [x] **Phase 11:** ESP32 firmware architecture & safe device simulation (No Actuation)
- [x] **Phase 12:** ESP32 Physical Bench Setup, Hardware-in-the-Loop Validation & Safe Sensor Integration (Actuators Disabled)
- [x] **Phase 13:** Autonomous Mobile Rover Architecture & Safe Motion Simulation (Motors Disabled)
- [x] **Phase 14:** Computer Vision Visual Servoing, Pest Centering & Closed-Loop Alignment Simulation (Motors & Actuators Disabled)
- [x] **Phase 15:** Multi-Frame Target Tracking, Track ID Persistence & Temporal Occlusion Handling (Motors & Actuators Disabled)
- [x] **Phase 16:** Edge AI Benchmarking, Stream Ingestion & Pipeline Profiling (Motors & Actuators Disabled)
- [x] **Phase 17:** Real-Time Edge Monitoring Dashboard Integration & Embedded Benchmark Protocol (Motors & Actuators Disabled)
- [ ] **Milestone 18+:** Physical camera bench calibration, embedded power profiling, and real-time field trials

---

## Phase 17: Real-Time Edge Monitoring Dashboard Integration & Embedded Benchmark Protocol

> [!IMPORTANT]
> **SAFETY & BENCHMARKING POLICY:**
> * **Hardware Actuation:** Strictly **DISABLED**. All motors, relays, speakers, ultrasonic emitters, and PWM outputs remain permanently locked and non-operational.
> * **Zero Physical Actuator Commands:** Dashboard and benchmark scripts operate in read-only observation and software simulation modes.
> * **Host Environment Honesty:** Measurements on Windows/AMD64 host platforms are transparently labeled as desktop verification runs. **Desktop metrics must never be claimed as physical Raspberry Pi 4 edge throughput.**
> * **Temperature Telemetry Isolation:** SoC temperature reads are attempted strictly via Linux/RPi sysfs (`/sys/class/thermal/thermal_zone0/temp`) or `vcgencmd measure_temp`. On non-Pi hosts, the field is reported as `"UNAVAILABLE (Non-Raspberry Pi Host)"` without fabrication.
> * **Opt-In Video Ingestion:** Safe video streams operate on local recorded clips or synthetic canopy generators with automatic end-of-stream detection and deterministic resource teardown. Physical cameras are never required.

### 1. Dashboard Architecture (`src/dashboard/app.py`)

The Streamlit analytics suite is structured into three dedicated operational views:

```
+-----------------------------------------------------------------------------------------------------------------+
|                                KRISHNETRA STREAMLIT EDGE MONITORING DASHBOARD                                   |
|                                                                                                                 |
|  [ Tab 1: Pest Telemetry Analytics ]                                                                            |
|  ├── Real-time PostgreSQL database telemetry query with automatic local CSV/JSONL failover                      |
|  ├── Dynamic KPI metric cards: Total Observations, Harmful Targets, Beneficial Safe, System Health             |
|  └── Visual analytics: Hourly observation distribution, species breakdown donut, confidence scatter timeline   |
|                                                                                                                 |
|  [ Tab 2: Edge AI Pipeline Profiler & Benchmark ]                                                               |
|  ├── Dynamic Model Discovery: Auto-detects PyTorch (.pt) and ONNX (.onnx) model checkpoints                     |
|  ├── Live Pipeline Profiler Runner: Configurable frame count (5-100), warmup frames, and stream sources        |
|  ├── Real-Time KPI Cards: Observed FPS, End-to-End Latency, YOLO Inference Latency, Tracking/Servoing Latency   |
|  ├── Stage-by-Stage Latency Breakdown: Plotly horizontal stacked waterfall chart (Ingest, Preprocess,          |
|  │   YOLO Inference, Multi-Frame Tracking, Visual Servoing)                                                    |
|  ├── PyTorch vs ONNX Comparison Suite: Side-by-side throughput delta and latency speedup calculation            |
|  └── Host Metric Attribution: CPU & RSS Memory consumption with prominent Non-Pi desktop disclaimer             |
|                                                                                                                 |
|  [ Tab 3: Safe Video Ingestion & Stream Inspector ]                                                             |
|  ├── SafeVideoStream Inspector: Select pre-recorded MP4/AVI or synthetic agricultural canopy                    |
|  ├── Non-blocking single-frame inspection with resolution, channel, and FPS metadata readout                    |
|  └── Automatic EOS, frame drop handling, and safe resource release                                             |
+-----------------------------------------------------------------------------------------------------------------+
```

### 2. Embedded Raspberry Pi Benchmark Protocol (`deployment/raspberry_pi/benchmark_edge.py`)

A dedicated, standalone CLI edge benchmarking utility designed for execution directly on edge microcomputers (Raspberry Pi 4 Model B):

* **Platform Detection:** Inspects `/proc/cpuinfo` and `/proc/device-tree/model` to verify hardware platform authenticity (`is_raspberry_pi()`).
* **Thermal Monitoring:** Captures SoC core temperature via `/sys/class/thermal/thermal_zone0/temp` or `vcgencmd measure_temp`.
* **Isolated Warm-up:** Warm-up iterations are strictly segregated from timed benchmark frames to eliminate cold-start cache distortions.
* **CLI Options:**
  ```powershell
  # Synthetic Canopy Benchmark (Host or Raspberry Pi)
  python deployment/raspberry_pi/benchmark_edge.py --model models/pest_yolov8n_best.onnx --source synthetic --frames 30 --warmup 5

  # Pre-recorded Field Clip Benchmark
  python deployment/raspberry_pi/benchmark_edge.py --model models/pest_yolov8n_best.onnx --source tests/data/sample_pest_clip.mp4 --frames 30

  # Comparative PyTorch vs ONNX Edge Benchmark
  python deployment/raspberry_pi/benchmark_edge.py --compare --frames 20 --warmup 3 --output edge_benchmark_results.csv
  ```

### 3. Empirical Host Benchmark Comparison (PyTorch vs ONNX Runtime)

*Measured on Host CPU (640x480 resolution, 20 frames, 3 warm-up frames excluded, synthetic stream):*

| Pipeline Stage | PyTorch (`pest_yolov8n_best.pt`) | ONNX Runtime (`pest_yolov8n_best.onnx`) | Speedup Ratio |
| :--- | :---: | :---: | :---: |
| **Ingestion** | $0.00\text{ ms}$ | $0.00\text{ ms}$ | — |
| **Preprocessing** | $<0.01\text{ ms}$ | $<0.01\text{ ms}$ | — |
| **YOLO Inference** | $64.88\text{ ms}$ (p95: $92.74\text{ ms}$) | **$53.64\text{ ms}$** (p95: $58.12\text{ ms}$) | **$1.21\times$** |
| **Multi-Frame Tracking** | $0.02\text{ ms}$ | $0.01\text{ ms}$ | $2.00\times$ |
| **Visual Servoing** | $0.07\text{ ms}$ | $0.07\text{ ms}$ | $1.00\times$ |
| **Total End-to-End Latency** | **$66.02\text{ ms}$** | **$54.72\text{ ms}$** | **$1.21\times$** |
| **Observed Throughput (FPS)** | **$15.15\text{ FPS}$** | **$18.28\text{ FPS}$** | **$1.21\times$** |
| **Process Memory (RSS)** | $477.2\text{ MB}$ | $480.8\text{ MB}$ | — |
| **SoC Temperature** | UNAVAILABLE (Non-RPi) | UNAVAILABLE (Non-RPi) | — |

### 4. Running Phase 17 Tests & Launching Dashboard

```powershell
# Run Phase 17 Dashboard & Edge Benchmark Integration Tests (10 tests)
python -m unittest tests/test_dashboard_profiler_integration.py -v

# Run Full Test Suite Across All 17 Phases (98 tests)
python -m unittest discover tests -v

# Launch the Streamlit Monitoring & Benchmark Dashboard
streamlit run src/dashboard/app.py
```

> [!IMPORTANT]
> **SAFETY & BENCHMARKING POLICY:**
> * **Hardware Actuation:** Strictly **DISABLED**. All motors, relays, speakers, ultrasonic emitters, and PWM outputs remain permanently locked and non-operational.
> * **Input Source Isolation:** Testing utilizes local pre-recorded video files (`.mp4`), injectable frame arrays, or synthetic feeds. Physical camera hardware is never required.
> * **Integrity of Measurements:** Desktop profiling metrics are benchmarked on the host environment for relative algorithm comparison (PyTorch vs ONNX). **Desktop FPS does not represent physical Raspberry Pi 4 edge throughput.** Latency measurements do not represent pest detection accuracy.
> * **Telemetry Labeling:** All benchmark outputs and telemetry packets are tagged with `"telemetry_type": "SIMULATED"` and `"hardware_actuators": "DISABLED"`.

### 1. End-to-End Pipeline Profiling Architecture

```
+-------------------------------------------------------------------------------------------------------+
|                                    SAFE VIDEO INGESTION STAGE                                         |
|                                                                                                       |
|   [ SafeVideoStream ] ---> Pre-recorded Video (MP4) | Synthetic Canopy | Injected Test Frames         |
|   (Automatic EOS detection, frame looping, and strict physical camera isolation)                      |
+-------------------------------------------------------------------------------------------------------+
                                                   |
                                            (t_ingest ~ 1.0 ms)
                                                   v
+-------------------------------------------------------------------------------------------------------+
|                                        PREPROCESSING STAGE                                            |
|                                                                                                       |
|   - Spatial validation & color format alignment (640x480 RGB / BGR)                                    |
+-------------------------------------------------------------------------------------------------------+
                                                   |
                                           (t_preprocess < 0.1 ms)
                                                   v
+-------------------------------------------------------------------------------------------------------+
|                                     YOLO INFERENCE STAGE                                              |
|                                                                                                       |
|   [ PestDetector ] ---> PyTorch (pest_yolov8n_best.pt) vs ONNX Runtime (pest_yolov8n_best.onnx)       |
+-------------------------------------------------------------------------------------------------------+
                                                   |
                                         (t_inference: 54 - 67 ms)
                                                   v
+-------------------------------------------------------------------------------------------------------+
|                                   MULTI-FRAME TRACKING STAGE                                          |
|                                                                                                       |
|   [ MultiFrameTracker ] ---> Greedy 1-to-1 Bipartite IoU & Centroid Proximity Association             |
+-------------------------------------------------------------------------------------------------------+
                                                   |
                                           (t_tracking ~ 0.02 ms)
                                                   v
+-------------------------------------------------------------------------------------------------------+
|                               VISUAL SERVOING & DECISION STAGE                                        |
|                                                                                                       |
|   [ TargetSelector (PERSISTENT_LOCK) ] + [ PestCenteringController (Safe Occlusion Hold) ]            |
+-------------------------------------------------------------------------------------------------------+
                                                   |
                                           (t_servoing ~ 0.08 ms)
                                                   v
                                     [ Total E2E Latency: 55 - 68 ms ]
```

### 2. Empirical Benchmark Results (PyTorch vs ONNX Runtime)

*Measured on Host CPU (640x480 resolution, 15 frames, 3 warm-up frames excluded, `tests/data/sample_pest_clip.mp4`):*

| Pipeline Stage | PyTorch (`pest_yolov8n_best.pt`) | ONNX Runtime (`pest_yolov8n_best.onnx`) | Speedup Ratio |
| :--- | :---: | :---: | :---: |
| **Frame Ingestion** | $1.07\text{ ms}$ (median: $1.03\text{ ms}$) | $1.00\text{ ms}$ (median: $0.96\text{ ms}$) | $1.07\times$ |
| **Preprocessing** | $<0.01\text{ ms}$ | $<0.01\text{ ms}$ | $1.00\times$ |
| **Model Inference** | $66.84\text{ ms}$ (p95: $96.65\text{ ms}$) | **$54.13\text{ ms}$** (p95: $58.65\text{ ms}$) | **$1.23\times$** |
| **Multi-Frame Tracking** | $0.02\text{ ms}$ | $0.01\text{ ms}$ | $2.00\times$ |
| **Visual Servoing** | $0.08\text{ ms}$ | $0.07\text{ ms}$ | $1.14\times$ |
| **Total End-to-End Latency** | **$68.00\text{ ms}$** (p95: $97.77\text{ ms}$) | **$55.21\text{ ms}$** (p95: $59.74\text{ ms}$) | **$1.23\times$** |
| **Observed Throughput (FPS)** | **$14.71\text{ FPS}$** | **$18.11\text{ FPS}$** | **$1.23\times$** |
| **Process Memory (RSS)** | $406.9\text{ MB}$ | $480.8\text{ MB}$ | — |

*Key Takeaway:* ONNX Runtime provides a **$1.23\times$ speedup** over PyTorch CPU inference with significantly tighter latency variance ($58.65\text{ ms}$ p95 vs $96.65\text{ ms}$ p95). Tracking and visual servoing contribute $<0.1\text{ ms}$ ($<0.2\%$ of total frame time), verifying zero algorithmic bottleneck.

### 3. Reproducible Benchmark Commands

```powershell
# Run Phase 16 Pipeline Profiling Unit & Integration Tests (12 tests)
python -m unittest tests/test_pipeline_profiling.py -v

# Run Full Regression Suite (88 tests across all 16 phases)
python -m unittest discover tests -v
```

---

## Phase 15: Multi-Frame Target Tracking, Track ID Persistence & Temporal Occlusion Handling

> [!IMPORTANT]
> **SAFETY & ACTUATION POLICY:**
> * **Hardware Actuation:** Strictly **DISABLED**. All motors, relays, speakers, ultrasonic emitters, and PWM outputs remain permanently locked and non-operational.
> * **Simulation & Odometry:** Closed-loop servoing and multi-frame tracking operate exclusively in **software simulation**.
> * **Zero Fabricated Detections:** When a pest is occluded or missed, detections are never fabricated. Predicted positions are explicitly flagged with `"is_predicted": true` and `"is_observed": false`.
> * **Safe Occlusion Hold:** During occlusion, the default servoing policy commands an immediate `STOP` (`TEMPORARILY_OCCLUDED`) to prevent erratic phantom movement.

### 1. Multi-Frame Tracking & Occlusion Architecture

```
+-------------------------------------------------------------------------------------------------+
|                                 DETECTION & VALIDATION STAGE                                    |
|                                                                                                 |
|   [ Incoming Frame Detections ] ---> [ Filter (conf >= 0.60, valid non-degenerate bbox) ]        |
+-------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                               BIPARTITE ASSOCIATION ENGINE                                     |
|                                                                                                 |
|   - Compute IoU Overlap: inter_area / union_area                                                |
|   - Centroid Proximity: Euclidean distance affinity max(0, 1 - d / d_max)                       |
|   - Strict Class Matching (Armyworm != Corn Borer)                                              |
|   - Global Greedy 1-to-1 Matching (prevents duplicate track or detection assignment)            |
+-------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                                TRACK LIFECYCLE STATE MACHINE                                    |
|                                                                                                 |
|    [ Match Observed ]   --->   TRACKED (is_observed=True, consecutive_misses=0, smooth EMA v)   |
|                                     |                                                           |
|    [ Missed <= 5 frames ] ---> TEMPORARILY_MISSING (is_predicted=True, velocity coasting)       |
|                                     |                                                           |
|    [ Missed > 5 frames ]  ---> LOST (is_valid=False, safely pruned from active tracks)          |
+-------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                          TARGET SELECTOR WITH PERSISTENT LOCK                                   |
|                                                                                                 |
|   - Policy: PERSISTENT_LOCK                                                                     |
|   - Locks onto track_id; maintains lock across frames (eliminates multi-pest target hopping)    |
|   - Re-acquires via HIGHEST_CONFIDENCE fallback only when locked track is LOST                  |
+-------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                     PEST CENTERING CONTROLLER & VISUAL SERVOING                                 |
|                                                                                                 |
|   - If target is TEMPORARILY_MISSING (predicted) & allow_predicted_servoing=False:              |
|       ---> Status: TEMPORARILY_OCCLUDED, Suggested Command: STOP (Safe Hold)                   |
|   - If target is TRACKED (observed):                                                            |
|       ---> Proportional ROTATE_RIGHT / ROTATE_LEFT or CENTERED / STOP                           |
|   - If target is LOST:                                                                          |
|       ---> Status: TARGET_LOST, Suggested Command: STOP                                         |
+-------------------------------------------------------------------------------------------------+
```

### 2. Association & Tracking Formulation

#### Bounding Box IoU
$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$

#### Centroid Distance Affinity
$$\text{Affinity}(c_1, c_2) = \max\left(0, 1.0 - \frac{\|c_1 - c_2\|_2}{d_{\max}}\right)$$

#### Combined Affinity Score
$$\text{Score} = w_{\text{iou}} \cdot \text{IoU} + w_{\text{dist}} \cdot \text{Affinity} + w_{\text{conf}} \cdot \text{conf}$$
Matching is evaluated globally with a strict 1-to-1 greedy bipartite assignment to ensure no target is assigned more than once per frame.

#### Velocity Estimation & Temporal Coasting
For active tracks, velocity is filtered via exponential moving average:
$$\mathbf{v}_t = \beta \cdot (\mathbf{c}_t - \mathbf{c}_{t-1}) + (1 - \beta) \cdot \mathbf{v}_{t-1}$$
During temporary occlusion ($1 \le \text{misses} \le \text{max\_missed\_frames}$):
$$\mathbf{c}_{t+\Delta t} = \mathbf{c}_t + \mathbf{v} \cdot \alpha, \quad \text{conf}_{t+\Delta t} = \text{conf}_t \cdot \gamma$$
where damping $\alpha = 0.80$ and decay $\gamma = 0.85$. If occlusion exceeds $\text{max\_missed\_frames} = 5$, the track transitions to `LOST` and is pruned.

### 3. Running Multi-Frame Tracking Tests

```powershell
# Run Phase 15 Tracking Unit & Integration Tests (12 tests)
python -m unittest tests/test_target_tracker.py -v

# Run Full Regression Suite (76 tests across all 15 phases)
python -m unittest discover tests -v
```

---

## Phase 14: Computer Vision Visual Servoing, Pest Centering & Closed-Loop Alignment Simulation

> [!IMPORTANT]
> **SAFETY & ACTUATION POLICY:**
> * **Actuation Hardware:** Permanently **DISABLED**. Zero motors, relays, PWM pins, or acoustic emitters are enabled.
> * **Closed-Loop Mode:** Visual servoing and camera projection are executed in **software simulation** mode.
> * **Telemetry Labeling:** All output packets are explicitly marked with `"telemetry_type": "SIMULATED"`.

### 1. Visual Servoing Architecture

```
+---------------------------------------------------------------------------------------+
|                               IMAGE ACQUISITION & YOLO                                |
|                                                                                       |
|   [ Input Frame / Camera Projection ] ---> [ PestDetector (yolov8n_best.pt) ]         |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                 TARGET SELECTOR                                       |
|                                                                                       |
|   - Bounding Box Integrity Validation (x1 < x2, y1 < y2, finite, in-bounds)           |
|   - Confidence Cutoff (conf >= 0.60)                                                  |
|   - Multi-Target Policy (HIGHEST_CONFIDENCE, CLOSEST_TO_CENTER, LARGEST_BBOX_AREA)    |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                           PEST CENTERING CONTROLLER                                   |
|                                                                                       |
|   - Optical Center: (W/2, H/2) = (320, 240)                                           |
|   - Normalized Error: e_x = (c_x - W/2) / (W/2) in [-1.0, +1.0]                       |
|   - Dead Zone Check: |e_x| <= 0.05 (+/- 5% of half-width) ---> Status: CENTERED, STOP |
|   - Proportional Servoing: omega_z = clamp(K_p * |e_x|, omega_min, omega_max)         |
|   - e_x > +0.05 ---> ROTATE_RIGHT | e_x < -0.05 ---> ROTATE_LEFT                      |
|   - Target Lost / Missing ---> Status: TARGET_LOST, STOP                              |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                      CLOSED-LOOP SERVOING SIMULATOR (HIL)                             |
|                                                                                       |
|   - Pinhole Camera Model: Projects ground coordinates (X_p, Y_p) to (u, v) pixels     |
|   - Dispatches MotionCommand to SafeMotionController                                  |
|   - Updates RoverSimulator Planar Pose (X_r, Y_r, theta_r)                            |
|   - Iterates until Centered, Lost, or Max Iterations Reached                          |
|   * PHYSICAL ACTUATOR OUTPUTS: STRICTLY DISABLED                                      |
+---------------------------------------------------------------------------------------+
```

### 2. Normalized Error & Alignment Control Equations

Given image width $W$ and height $H$, target centroid $(c_x, c_y)$ is normalized relative to optical center $(W/2, H/2)$:
$$e_x = \frac{c_x - \frac{W}{2}}{\frac{W}{2}} \in [-1.0, +1.0], \quad e_y = \frac{c_y - \frac{H}{2}}{\frac{H}{2}} \in [-1.0, +1.0]$$

* **Horizontally Centered Condition:**
  $$|e_x| \le \text{dead\_zone\_x} \implies \text{Command: } \text{STOP}, \quad \text{Status: } \text{CENTERED}$$
* **Proportional Rotational Correction:**
  $$\omega_z = \text{clamp}\left( K_{p,\text{angular}} \cdot |e_x|, \; \omega_{\text{min}}, \; \omega_{\text{max}} \right)$$
  * If $e_x > \text{dead\_zone\_x}$: `ROTATE_RIGHT` (target to the right, platform turns clockwise).
  * If $e_x < -\text{dead\_zone\_x}$: `ROTATE_LEFT` (target to the left, platform turns counter-clockwise).

### 3. Closed-Loop Simulation Telemetry Format

Each iteration logs a verified telemetry packet:
```json
{
  "iteration": 8,
  "timestamp_sim": 0.80,
  "pose": {"x_meters": 0.0, "y_meters": 0.0, "heading_radians": -0.285, "heading_degrees": 343.67},
  "error_x": 0.0124,
  "error_y": 0.0,
  "status": "CENTERED",
  "command": {
    "direction": "STOP",
    "linear_speed_m_s": 0.0,
    "angular_speed_rad_s": 0.0,
    "duration_sec": 0.20
  },
  "target_detected": true,
  "telemetry_type": "SIMULATED",
  "hardware_actuators": "DISABLED"
}
```

### 4. Running Visual Servoing Tests

```powershell
python -m unittest tests/test_visual_servoing.py -v
```

---

## Phase 13: Autonomous Mobile Rover Architecture & Safe Motion Simulation

> [!IMPORTANT]
> **SAFETY NOTICE: PHYSICAL MOTORS & ACTUATION REMAIN STRICTLY DISABLED**
> * No motors, H-bridges, relays, or PWM signals are driven.
> * Kinematics calculations and odometry dead-reckoning are executed in a **pure software simulation** mode.
> * All generated telemetry packets are explicitly tagged with `"telemetry_type": "SIMULATED"`.

### 1. Motion Command Architecture

Supported directional commands:
* `FORWARD`: Linear translation along +X axis ($v_x > 0$).
* `REVERSE`: Linear translation along -X axis ($v_x < 0$).
* `STRAFE_LEFT`: Lateral translation along +Y axis ($v_y > 0$, Mecanum only).
* `STRAFE_RIGHT`: Lateral translation along -Y axis ($v_y < 0$, Mecanum only).
* `ROTATE_LEFT`: Counter-clockwise angular rotation ($\omega_z > 0$).
* `ROTATE_RIGHT`: Clockwise angular rotation ($\omega_z < 0$).
* `STOP`: Controlled immediate velocity ramp to zero.

### 2. Drivetrain Kinematics

#### 4-Wheel Mecanum Drive
Inverse kinematics maps chassis vector $(v_x, v_y, \omega_z)$ to 4 independent wheel surface speeds:
$$\begin{aligned}
v_{FL} &= v_x - v_y - k \cdot \omega_z \\
v_{FR} &= v_x + v_y + k \cdot \omega_z \\
v_{RL} &= v_x + v_y - k \cdot \omega_z \\
v_{RR} &= v_x - v_y + k \cdot \omega_z
\end{aligned}$$
where $k = \frac{\text{wheelbase} + \text{track\_width}}{2} = \frac{0.250\text{ m} + 0.280\text{ m}}{2} = 0.265\text{ m}$.

#### 2-Wheel Differential Drive
Inverse kinematics maps chassis vector $(v_x, \omega_z)$ to left and right wheel speeds:
$$\begin{aligned}
v_L &= v_x - \frac{W}{2} \cdot \omega_z \\
v_R &= v_x + \frac{W}{2} \cdot \omega_z
\end{aligned}$$
*Note: Lateral strafing commands are strictly rejected with an explicit error in differential configuration.*

### 3. Safe Motion Controller State Machine

```
      +-----------------------------------------+
      |                 LOCKED                  | <---+ (Emergency Stop / Heartbeat Loss)
      |   (Default State, All Movement Blocked) |     |
      +-----------------------------------------+     |
                           |                          |
               unlock_for_simulation()                |
                           v                          |
      +-----------------------------------------+     |
      |                 STOPPED                 |     |
      |          (Zero Wheel Velocity)          |     |
      +-----------------------------------------+     |
                 |                     ^              |
           request_motion()       STOP / Timeout      |
                 v                     |              |
      +-----------------------------------------+     |
      |                 MOVING                  | ----+
      |       (Simulated Wheel Speeds)          |
      +-----------------------------------------+
```

### 4. Safety Constraints & Enforcement
1. **Default Locked State:** Rover initializes in `LOCKED` mode; motion requests are rejected until unlocked.
2. **Speed Limits:** Hard ceiling enforced at $v_{\text{max}} = 0.50\text{ m/s}$ and $\omega_{\text{max}} = 1.20\text{ rad/s}$.
3. **Command Timeout Watchdog:** Movement automatically halts (`TIMEOUT_STOPPED`) if duration expires or no new command is received within $1.5\text{ s}$.
4. **Heartbeat Loss Protection:** Link loss immediately engages emergency stop and reverts to `LOCKED`.
5. **Emergency Stop (E-STOP):** Instantaneous transition to `E_STOPPED` and `LOCKED`; subsequent motion commands are rejected.

### 5. Running Motion Simulation Tests

```powershell
python -m unittest tests/test_rover_motion.py -v
```

---

## Phase 12: ESP32 Physical Bench Setup, Hardware-in-the-Loop Validation & Safe Sensor Integration

> [!IMPORTANT]
> **SAFETY & ACTUATION POLICY:**
> * **Actuation Hardware:** Permanently **DISABLED**. No motors, relays, speakers, ultrasonic transducers, or GPIO drivers are enabled.
> * **Hardware Readiness:** Physical hardware audit categorized components into AVAILABLE, NOT AVAILABLE, and NOT VERIFIED. Hardware-in-the-Loop (HIL) testing was executed via automated software bus simulation without claiming physical hardware bench testing.

### 1. Hardware Availability Audit

| Component | Status | Details / Audit Finding |
|---|---|---|
| **ESP32 Development Board** | `NOT AVAILABLE` | No CP2102/CH340 USB-UART bridges detected on host ports (only virtual Bluetooth COM ports). |
| **USB Data Cable** | `NOT VERIFIED` | Host USB controller verified; physical micro-USB/Type-C cable not connected to MCU. |
| **Raspberry Pi 4** | `NOT AVAILABLE` | Host execution environment is Windows edge development workstation; deployment package verified. |
| **RPi Power Supply (5V/3A)** | `NOT VERIFIED` | External bench power supply not physically attached. |
| **Camera (UVC / USB)** | `AVAILABLE` | USB2.0 HD UVC WebCam detected and verified (`USB\VID_2B7E&PID_B888`). |
| **Wi-Fi Network** | `AVAILABLE` | Host local area network active. |
| **Sensors (DHT22, INA219, BH1750)** | `NOT AVAILABLE` | Physical I2C/1-Wire sensors not connected; simulated via read-only HAL. |
| **Breadboard & Jumper Wires** | `NOT VERIFIED` | Bench prototyping accessories pending lab setup. |
| **Multimeter** | `NOT VERIFIED` | External bench test equipment. |

### 2. Architecture & Data Flow

```
+---------------------------------------------------------------------------------------+
|                          EDGE AI PIPELINE (Raspberry Pi 4 / Host)                     |
|                                                                                       |
|   [ UVC Camera Feed ] ---> [ YOLOv8n ONNX Detector ] ---> [ Detection Packet ]        |
|                                                                 |                     |
|                                                                 v                     |
|                                                         [ Decision Engine ]           |
|                                                                 |                     |
|                                                                 v                     |
|                                                     [ Safety Interlock Layer ]        |
|                                                     (Master State: LOCKED)            |
|                                                                 |                     |
|                                                                 v                     |
|                                                      [ Safe Command Packet ]          |
+---------------------------------------------------------------------------------------+
                                           |
                        [ MQTT over Wi-Fi / Serial Fallback ]
                        (Topics: krishnetra/edge/commands, ack, heartbeat)
                                           |
+---------------------------------------------------------------------------------------+
|                       ESP32 MICROCONTROLLER & SENSOR HAL                              |
|                                                                                       |
|   [ MQTT Subscriber ] ---> [ Command Validator ] ---> [ ACK / SAFETY_LOCKED Response ]|
|                                    |                                                  |
|                                    v                                                  |
|   [ Heartbeat Watchdog ] <---> [ PostgreSQL Telemetry / JSONL Fallback Logger ]       |
|                                    ^                                                  |
|                                    |                                                  |
|   [ Safe Sensor Manager HAL ] -----+ (Telemetry JSON: Temperature, Humidity, Lux, V) |
|   (Read-Only: DHT22, INA219, BH1750)                                                  |
|                                                                                       |
|   * PHYSICAL ACTUATION STATUS: PERMANENTLY DISABLED                                   |
+---------------------------------------------------------------------------------------+
```

### 3. Safe Sensor Abstraction Layer

Implemented in `firmware/esp32/include/sensors/` and `src/sensors/sensor_manager.py`:
* **Read-only channels:** Ambient Temperature (`DHT22_TEMP_01`), Relative Humidity (`DHT22_HUM_01`), Battery Voltage (`INA219_VBAT_01`), Ambient Light (`BH1750_LUX_01`).
* **Telemetry Structure:**
  ```json
  {
    "source": "esp32_sensor_hal",
    "timestamp": "2026-10-14T17:15:00.000Z",
    "sensor_count": 4,
    "readings": [
      {"sensor_id": "DHT22_TEMP_01", "sensor_type": "TEMPERATURE", "value": 26.5, "unit": "C", "status": "OK"},
      {"sensor_id": "DHT22_HUM_01", "sensor_type": "HUMIDITY", "value": 62.0, "unit": "%", "status": "OK"},
      {"sensor_id": "INA219_VBAT_01", "sensor_type": "BATTERY_VOLTAGE", "value": 12.4, "unit": "V", "status": "OK"},
      {"sensor_id": "BH1750_LUX_01", "sensor_type": "AMBIENT_LIGHT", "value": 450.0, "unit": "LUX", "status": "OK"}
    ],
    "actuation_state": "DISABLED_READ_ONLY"
  }
  ```
* **Strict Safeguard:** Sensor readings are strictly read-only telemetry. They do not trigger actuators or bypass the safety interlock.

### 4. Hardware Safety Checklist (Pre-Actuation Requisites)

Before any physical actuation testing in future phases, the following bench checklist is mandatory:
- [ ] **Dual Power Supply Isolation:** Logic (ESP32/RPi at 5V/3.3V) powered on a separate isolated rail from motor/transducer power (12V).
- [ ] **Common Ground:** Common ground tied with low-impedance connection between logic and power stages.
- [ ] **Current Limiting & Fusing:** Fast-blow fuse installed on 12V battery rail (max 3A).
- [ ] **Emergency Hardware Disconnect:** Physical red E-stop toggle switch in series with actuator supply.
- [ ] **Optocoupler / Driver Isolation:** Gate drive isolation for MOSFETs / H-bridge drivers.
- [ ] **Software Safety Lock:** Safety interlock verified in `LOCKED` state prior to boot.
- [ ] **Hardware Watchdog:** Watchdog timer set to max 5.0s timeout to force `NO_ACTION` on packet drop.
- [ ] **Manual Shutdown Procedure:** Physical kill switch procedure rehearsed before power-on.

### 5. Hardware-in-the-Loop (HIL) Testing Procedure

Run the automated HIL and sensor regression suite:
```powershell
python -m unittest tests/test_hil_simulation.py -v
python -m unittest tests/test_sensor_integration.py -v
```

---

## Phase 11: ESP32 Firmware Architecture & Safe Device Simulation

> [!IMPORTANT]
> **SAFETY & HARDWARE STATUS:**
> * **Actuation Hardware:** Permanently **DISABLED**. No motors, relays, speakers, ultrasonic transducers, or GPIO drivers are enabled.
> * **Hardware Validation Status:** Firmware is implemented using the **Arduino-ESP32 C++** framework and verified via automated Python host simulation. Physical board bench testing is pending microcontroller provisioning.

### 1. Firmware Directory Layout (`firmware/esp32/`)

```
firmware/esp32/
├── platformio.ini               # PlatformIO build configuration for ESP32 Dev Module
├── config_secrets.h.example     # Credentials template (WIFI_SSID, MQTT_BROKER)
├── include/
│   ├── config.h                 # Network topics, timeouts, and safety flags
│   ├── safety_state_machine.h   # 5-state safety state machine header
│   ├── command_parser.h         # JSON command parser and validator header
│   ├── heartbeat_watchdog.h     # Link liveness and watchdog monitor header
│   ├── ack_reporter.h           # JSON ACK and telemetry reporter header
│   ├── wifi_manager.h           # Wi-Fi connection manager header
│   └── mqtt_manager.h           # MQTT pub/sub message router header
├── src/
│   ├── main.cpp                 # Application entry point (setup / loop)
│   ├── safety_state_machine.cpp # State transition and lock enforcement
│   ├── command_parser.cpp       # ArduinoJson validation implementation
│   ├── heartbeat_watchdog.cpp   # Watchdog timeout detector implementation
│   ├── ack_reporter.cpp         # Structured response serializer implementation
│   ├── wifi_manager.cpp         # Auto-reconnect Wi-Fi logic
│   └── mqtt_manager.cpp         # Topic subscriptions & dispatch logic
└── sim/
    └── firmware_simulator.py    # Host-side Python simulator mirroring C++ firmware
```

### 2. Firmware Safety State Machine

```
      +-------------------+
      |      BOOTING      |
      +-------------------+
                |
                v
      +-------------------+           Wi-Fi / MQTT Up
      |   DISCONNECTED    | ----------------------------------> +--------------------+
      +-------------------+                                     |  CONNECTED_LOCKED  |
                ^                                               +--------------------+
                |                   Link Drop / Outage                    |
                +---------------------------------------------------------+
                                                                          |
                                                              Watchdog Timeout (>5.0s)
                                                                          |
                                                                          v
                                                                +--------------------+
                                                                | COMMUNICATION_LOST |
                                                                +--------------------+
```

* **`BOOTING`:** Hardware initialization; outputs locked.
* **`DISCONNECTED`:** Network disconnected; default `NO_ACTION`.
* **`CONNECTED_LOCKED`:** Connected to MQTT broker; actuation permanently locked.
* **`COMMUNICATION_LOST`:** Heartbeat ping timeout; forces `NO_ACTION`.
* **`ERROR_LOCKED`:** Exception / fault state; requires manual reset.

### 3. Firmware Configuration & Building

#### Prerequisites (When flashing to physical hardware):
* [PlatformIO Core](https://platformio.org/) or [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/).
* ESP32 DevKit V1 (or equivalent ESP32 board).

#### Build & Flash Instructions:
```bash
# 1. Navigate to firmware directory
cd firmware/esp32

# 2. Configure credentials
cp config_secrets.h.example include/config_secrets.h
# Edit include/config_secrets.h with Wi-Fi & MQTT broker details

# 3. Build firmware
pio run -e esp32dev

# 4. Flash to connected ESP32 board
pio run -e esp32dev --target upload
```

#### Running Host-Side Simulation Tests (No Hardware Required):
```powershell
python -m unittest tests/test_firmware_simulation.py -v
```

---

## Phase 10: ESP32 Communication Architecture & Safe Command Protocol

> [!IMPORTANT]
> **SAFETY NOTICE: PHYSICAL ACTUATION IS PERMANENTLY DISABLED**
> All physical actuators (speakers, ultrasonic transducers, motors, relays, GPIO outputs) remain disconnected and disabled. Phase 10 implements software abstractions, simulation layers, safety interlocks, and telemetry logging only.

### 1. Communication Flow Architecture

```
+-----------------------------------------------------------------------------+
|                           EDGE AI PIPELINE (Raspberry Pi 4)                 |
|                                                                             |
|  [ Camera Feed ] ---> [ YOLOv8n ONNX Engine ] ---> [ Detection Packet ]     |
|                                                                |            |
|                                                                v            |
|                                                    [ Decision Engine ]      |
|                                                                |            |
|                                                                v            |
|                                                   [ Safety Interlock Layer ]|
|                                                   (Master State: LOCKED)    |
|                                                                |            |
|                                                                v            |
|                                                    [ Safe Command Packet ]  |
+-----------------------------------------------------------------------------+
                                       |
                   [ MQTT over Wi-Fi / Serial Fallback ]
                                       |
+-----------------------------------------------------------------------------+
|                        MOCK MICROCONTROLLER (ESP32)                         |
|                                                                             |
|  [ MQTT Subscriber ] ---> [ Command Validator ] ---> [ ACK / Safety Locked] |
|                                |                                            |
|                                v                                            |
|  [ Heartbeat Monitor ] <---> [ Telemetry Logger (PostgreSQL / JSONL) ]      |
|                                                                             |
|  * PHYSICAL HARDWARE DRIVER: DISABLED / PERMANENTLY LOCKED                  |
+-----------------------------------------------------------------------------+
```

### 2. Command Protocol Schema (JSON)

Commands emitted by Edge AI to the ESP32 follow strict validation:

```json
{
  "command_id": "b3c79a21-7291-49fa-9481-22fb75e4c029",
  "timestamp": "2026-10-07T14:00:00.000Z",
  "source": "raspberry_pi",
  "model_version": "pest_yolov8n_best",
  "pest_class": "army worm",
  "confidence": 0.89,
  "decision": "NO_ACTION",
  "action": "NONE",
  "safety_state": "LOCKED"
}
```

*Allowed Simulated Actions:* `NONE`, `SIMULATED_DETERRENCE`, `SIMULATED_STOP`. (Any direct actuator command is rejected).

### 3. ESP32 Acknowledgement Protocol (ACK)

```json
{
  "command_id": "b3c79a21-7291-49fa-9481-22fb75e4c029",
  "device": "esp32",
  "status": "ACK",
  "timestamp": "2026-10-07T14:00:00.050Z",
  "safety_state": "LOCKED",
  "reason": "COMMAND_ACCEPTED_SIMULATION_ONLY"
}
```

*Possible Statuses:* `ACK`, `REJECTED`, `TIMEOUT`, `INVALID_COMMAND`, `SAFETY_LOCKED`.

### 4. Heartbeat & Liveness Monitoring

* **Ping (`raspberry_pi` $\rightarrow$ `esp32`):** Sent every $2.0\text{ s}$ on topic `krishnetra/esp32/heartbeat`.
* **Timeout Handling:** If no ACK is received within $5.0\text{ s}$, state shifts to `COMMUNICATION_LOST` and forces `NO_ACTION`.

### 5. MQTT Topic Architecture

* `krishnetra/edge/commands`: Edge AI command broadcast.
* `krishnetra/esp32/ack`: Microcontroller command execution response.
* `krishnetra/esp32/heartbeat`: Bidirectional keepalive ping-pong.
* `krishnetra/system/status`: Global node telemetry and safety status.

### 6. Safety Interlock Policy (12 Core Rules)
1. Default state is `LOCKED`.
2. Default action is `NO_ACTION`.
3. Physical actuator control commands are strictly rejected.
4. Unknown or unapproved actions are rejected.
5. Malformed JSON is rejected.
6. Missing `command_id` is rejected.
7. Missing `timestamp` is rejected.
8. Missing `safety_state` is rejected.
9. Link communication timeout automatically resolves to `NO_ACTION`.
10. Low model confidence ($< 0.60$) cannot trigger any action.
11. Mock mode operates entirely in memory with no GPIO access.
12. Zero physical GPIO/actuator libraries are loaded for actuation.

---

## Project Structure

```
krishnetra-aavishkar/
│
├── configs/
│   ├── detection_config.yaml      # Model hyperparameters & inference thresholds
│   └── target_pests.yaml          # Target pest whitelist & beneficial insect safelist
│
├── datasets/                      # Raw and processed datasets (e.g., IP102 subsets)
│
├── models/                        # Trained model checkpoints (.pt)
│
├── src/
│   ├── __init__.py
│   └── detection/
│       ├── __init__.py
│       └── detector.py            # Modular YOLO PestDetector class
│
├── tests/
│   ├── data/                      # Test assets
│   └── test_first_yolo.py         # AI environment and inference verification test
│
├── requirements.txt               # Strict AI prototype dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## Getting Started

### 1. Environment Setup
```powershell
# Create Python virtual environment
py -3.14 -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install initial AI dependencies
pip install -r requirements.txt
```

### 2. Run First Working Test
```powershell
python tests/test_first_yolo.py
```

The script will:
1. Verify Python, PyTorch, CUDA, Ultralytics, and OpenCV.
2. Load the baseline YOLO model.
3. Run inference on a test image.
4. Output detected objects with bounding boxes and confidence scores.
5. Save the annotated output in `runs/detect/`.

### 3. Launch Analytics & Telemetry Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run src/dashboard/app.py
```
Access the interactive dashboard locally at `http://localhost:8501`.
Features real-time PostgreSQL analytics with automatic local JSONL/CSV failover, KPI metric cards, pest distribution bar/donut charts, and live detection timelines.

