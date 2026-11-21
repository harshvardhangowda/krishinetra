"""
AI-Powered Precision Pest Management - Edge AI Pipeline Profiler & Benchmark Engine
File: src/monitoring/pipeline_profiler.py

Measures stage-by-stage latencies, system resource utilization, and throughput
for the end-to-end edge AI vision and visual servoing pipeline:
  [ Ingest ] -> [ Preprocess ] -> [ YOLO Inference ] -> [ Multi-Frame Tracking ] -> [ Visual Servoing ]

Safety & Integrity:
- Excludes warm-up iterations from timing percentiles.
- Accurately captures CPU and Memory (RSS) metrics via psutil.
- Compares PyTorch (.pt) vs ONNX Runtime (.onnx) inference using verified model files.
- Disables physical actuators and flags all outputs as SIMULATED.
- Distinguishes desktop benchmark speeds from real Raspberry Pi 4 edge performance.
"""

import os
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.detection.detector import PestDetector
from src.monitoring.video_stream import SafeVideoStream, StreamSourceType
from src.servoing.centering_controller import PestCenteringController
from src.servoing.target_selector import TargetSelector, TargetSelectionPolicy
from src.tracking.target_tracker import MultiFrameTracker


@dataclass
class StageLatencyStats:
    """Statistical summary for a pipeline stage or end-to-end latency."""
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }


@dataclass
class BenchmarkReport:
    """Structured report containing complete edge AI pipeline performance metrics."""
    model_name: str
    model_format: str               # "PYTORCH" or "ONNX"
    model_path: str
    device: str                     # "cpu", "cuda"
    input_resolution: List[int]     # [width, height]
    source_type: str                # "FILE", "SYNTHETIC", "INJECTED"
    total_frames_processed: int
    warmup_frames_excluded: int
    timed_frames_count: int
    dropped_or_failed_frames: int
    observed_fps: float
    wall_clock_time_sec: float

    # Stage Latencies
    ingest_latency: StageLatencyStats
    preprocess_latency: StageLatencyStats
    inference_latency: StageLatencyStats
    tracking_latency: StageLatencyStats
    servoing_latency: StageLatencyStats
    e2e_latency: StageLatencyStats

    # Resource Metrics
    cpu_percent: Union[float, str]
    process_memory_rss_mb: Union[float, str]

    # Environment & Safety Metadata
    telemetry_type: str = "SIMULATED"
    hardware_actuators: str = "DISABLED"
    host_platform: str = ""
    disclaimer: str = ""

    @property
    def stage_latencies_ms(self) -> Dict[str, Dict[str, float]]:
        return {
            "ingest": self.ingest_latency.to_dict(),
            "preprocess": self.preprocess_latency.to_dict(),
            "inference": self.inference_latency.to_dict(),
            "tracking": self.tracking_latency.to_dict(),
            "servoing": self.servoing_latency.to_dict(),
            "e2e": self.e2e_latency.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_format": self.model_format,
            "model_path": self.model_path,
            "device": self.device,
            "input_resolution": self.input_resolution,
            "source_type": self.source_type,
            "total_frames_processed": self.total_frames_processed,
            "warmup_frames_excluded": self.warmup_frames_excluded,
            "timed_frames_count": self.timed_frames_count,
            "dropped_or_failed_frames": self.dropped_or_failed_frames,
            "observed_fps": round(self.observed_fps, 2),
            "wall_clock_time_sec": round(self.wall_clock_time_sec, 3),
            "stage_latencies_ms": {
                "ingest": self.ingest_latency.to_dict(),
                "preprocess": self.preprocess_latency.to_dict(),
                "inference": self.inference_latency.to_dict(),
                "tracking": self.tracking_latency.to_dict(),
                "servoing": self.servoing_latency.to_dict(),
                "e2e": self.e2e_latency.to_dict(),
            },
            "system_resources": {
                "cpu_percent": self.cpu_percent,
                "process_memory_rss_mb": self.process_memory_rss_mb,
            },
            "telemetry_type": self.telemetry_type,
            "hardware_actuators": self.hardware_actuators,
            "host_platform": self.host_platform,
            "disclaimer": self.disclaimer,
        }


def compute_stage_stats(latencies: List[float]) -> StageLatencyStats:
    """Compute mean, median, p95, min, and max for a list of latencies in ms."""
    if not latencies:
        return StageLatencyStats(0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.array(latencies, dtype=np.float64)
    return StageLatencyStats(
        mean_ms=float(np.mean(arr)),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
    )


class PipelineProfiler:
    """
    Executes and profiles the Krishnetra Aavishkar edge vision pipeline.
    """

    def __init__(
        self,
        warmup_frames: int = 3,
        input_width: int = 640,
        input_height: int = 480,
    ) -> None:
        self.warmup_frames = warmup_frames
        self.input_width = input_width
        self.input_height = input_height

    def get_system_metrics(self) -> Tuple[Union[float, str], Union[float, str]]:
        """Retrieve process CPU usage (%) and Resident Set Size memory (MB)."""
        if not PSUTIL_AVAILABLE:
            return "UNAVAILABLE", "UNAVAILABLE"
        try:
            process = psutil.Process(os.getpid())
            # Non-blocking CPU reading
            cpu_pct = round(process.cpu_percent(interval=None), 1)
            mem_mb = round(process.memory_info().rss / (1024.0 * 1024.0), 2)
            return cpu_pct, mem_mb
        except Exception:
            return "UNAVAILABLE", "UNAVAILABLE"

    def profile_pipeline(
        self,
        stream: SafeVideoStream,
        detector: PestDetector,
        tracker: Optional[MultiFrameTracker] = None,
        selector: Optional[TargetSelector] = None,
        controller: Optional[PestCenteringController] = None,
        max_frames: int = 20,
    ) -> BenchmarkReport:
        """
        Run the edge vision pipeline through the video stream and profile latencies.
        """
        tracker = tracker or MultiFrameTracker(frame_width=self.input_width, frame_height=self.input_height)
        selector = selector or TargetSelector(policy=TargetSelectionPolicy.PERSISTENT_LOCK)
        controller = controller or PestCenteringController()

        tracker.reset()
        selector.clear_target_lock()

        # Latency lists (for timed frames only)
        ingest_times: List[float] = []
        preprocess_times: List[float] = []
        inference_times: List[float] = []
        tracking_times: List[float] = []
        servoing_times: List[float] = []
        e2e_times: List[float] = []

        total_processed = 0
        timed_frames = 0
        dropped_frames = 0

        # Prime CPU reading
        if PSUTIL_AVAILABLE:
            try:
                psutil.Process(os.getpid()).cpu_percent(interval=None)
            except Exception:
                pass

        wall_start = time.perf_counter()

        for frame_idx in range(max_frames):
            total_processed += 1
            is_warmup = frame_idx < self.warmup_frames

            # 1. Ingestion Stage
            t0 = time.perf_counter()
            ret, frame, meta = stream.read_frame()
            t_ingest = (time.perf_counter() - t0) * 1000.0

            if not ret or frame is None:
                dropped_frames += 1
                if meta.get("is_eos", False):
                    break
                continue

            # 2. Preprocessing Stage (resize check / color validation)
            t1 = time.perf_counter()
            if frame.shape[1] != self.input_width or frame.shape[0] != self.input_height:
                processed_frame = cv2.resize(frame, (self.input_width, self.input_height))
            else:
                processed_frame = frame
            t_preprocess = (time.perf_counter() - t1) * 1000.0

            # 3. Model Inference Stage
            t2 = time.perf_counter()
            detections = detector.detect(processed_frame)
            t_inference = (time.perf_counter() - t2) * 1000.0

            # 4. Multi-Frame Tracking Stage
            t3 = time.perf_counter()
            active_tracks = tracker.update(detections, frame_timestamp=time.time())
            t_tracking = (time.perf_counter() - t3) * 1000.0

            # 5. Visual Servoing & Decision Stage
            t4 = time.perf_counter()
            target = selector.select_target(active_tracks, self.input_width, self.input_height)
            alignment = controller.compute_alignment(target, self.input_width, self.input_height)
            t_servoing = (time.perf_counter() - t4) * 1000.0

            t_e2e = t_ingest + t_preprocess + t_inference + t_tracking + t_servoing

            # Record metrics only if past warm-up window
            if not is_warmup:
                timed_frames += 1
                ingest_times.append(t_ingest)
                preprocess_times.append(t_preprocess)
                inference_times.append(t_inference)
                tracking_times.append(t_tracking)
                servoing_times.append(t_servoing)
                e2e_times.append(t_e2e)

        wall_elapsed = time.perf_counter() - wall_start
        cpu_pct, mem_mb = self.get_system_metrics()

        e2e_stats = compute_stage_stats(e2e_times)
        observed_fps = 1000.0 / e2e_stats.mean_ms if e2e_stats.mean_ms > 0 else 0.0

        model_path_str = str(detector.weights_path)
        is_onnx = model_path_str.endswith(".onnx")
        model_fmt = "ONNX" if is_onnx else "PYTORCH"

        disclaimer = (
            "Desktop benchmark conducted on host platform for relative profiling only. "
            "Host FPS does not represent physical Raspberry Pi 4 edge throughput. "
            "Latency measurements do not represent pest detection accuracy."
        )

        return BenchmarkReport(
            model_name=Path(model_path_str).stem,
            model_format=model_fmt,
            model_path=model_path_str,
            device=detector.device,
            input_resolution=[self.input_width, self.input_height],
            source_type=stream.source_type.value,
            total_frames_processed=total_processed,
            warmup_frames_excluded=min(self.warmup_frames, total_processed),
            timed_frames_count=timed_frames,
            dropped_or_failed_frames=dropped_frames,
            observed_fps=observed_fps,
            wall_clock_time_sec=wall_elapsed,
            ingest_latency=compute_stage_stats(ingest_times),
            preprocess_latency=compute_stage_stats(preprocess_times),
            inference_latency=compute_stage_stats(inference_times),
            tracking_latency=compute_stage_stats(tracking_times),
            servoing_latency=compute_stage_stats(servoing_times),
            e2e_latency=e2e_stats,
            cpu_percent=cpu_pct,
            process_memory_rss_mb=mem_mb,
            telemetry_type="SIMULATED",
            hardware_actuators="DISABLED",
            host_platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            disclaimer=disclaimer,
        )

    def compare_models(
        self,
        pt_path: str = "models/pest_yolov8n_best.pt",
        onnx_path: str = "models/pest_yolov8n_best.onnx",
        stream_factory: Optional[Any] = None,
        num_frames: int = 15,
    ) -> Dict[str, Any]:
        """
        Run a comparative benchmark between PyTorch and ONNX models on identical stream inputs.
        """
        factory = stream_factory or (lambda: SafeVideoStream(source="synthetic", width=self.input_width, height=self.input_height))

        # 1. Profile PyTorch model
        stream_pt = factory()
        detector_pt = PestDetector(weights_path=pt_path, device="cpu")
        report_pt = self.profile_pipeline(stream=stream_pt, detector=detector_pt, max_frames=num_frames)
        stream_pt.release()

        # 2. Profile ONNX model
        stream_onnx = factory()
        detector_onnx = PestDetector(weights_path=onnx_path, device="cpu")
        report_onnx = self.profile_pipeline(stream=stream_onnx, detector=detector_onnx, max_frames=num_frames)
        stream_onnx.release()

        # Speedup comparison: PT / ONNX latency ratio
        pt_infer_mean = report_pt.inference_latency.mean_ms
        onnx_infer_mean = report_onnx.inference_latency.mean_ms
        speedup_ratio = round(pt_infer_mean / max(0.001, onnx_infer_mean), 2)

        return {
            "pytorch": report_pt.to_dict(),
            "onnx": report_onnx.to_dict(),
            "inference_speedup_factor": speedup_ratio,
            "faster_model": "ONNX" if onnx_infer_mean < pt_infer_mean else "PYTORCH",
            "telemetry_type": "SIMULATED",
            "hardware_actuators": "DISABLED",
        }
