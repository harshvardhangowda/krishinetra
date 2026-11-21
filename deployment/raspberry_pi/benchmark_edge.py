"""
AI-Powered Precision Pest Management - Embedded Edge AI Benchmark Protocol
File: deployment/raspberry_pi/benchmark_edge.py

Dedicated benchmark tool for Raspberry Pi 4 (ARM64) and edge embedded platforms.
Measures:
1. Warm-up vs timed inference latency
2. Stage-by-stage latencies (Ingest -> Preprocess -> Inference -> Tracking -> Servoing)
3. End-to-end throughput (Observed FPS)
4. System resource utilization (CPU %, RAM RSS, and SoC Temperature when on Raspberry Pi)
5. PyTorch vs ONNX Runtime performance comparison

Safety Constraint:
Actuators remain permanently DISABLED. Does not issue physical deterrence or movement.
Desktop runs are explicitly flagged with a disclaimer and never claimed as physical Pi data.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Set project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.detection.detector import PestDetector
from src.monitoring.video_stream import SafeVideoStream
from src.monitoring.pipeline_profiler import PipelineProfiler, BenchmarkReport


def is_raspberry_pi() -> bool:
    """Check if host platform is a physical Raspberry Pi device."""
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            content = cpuinfo.read_text(encoding="utf-8", errors="ignore")
            if "Raspberry Pi" in content or "BCM" in content:
                return True
        model = Path("/proc/device-tree/model")
        if model.exists():
            content = model.read_text(encoding="utf-8", errors="ignore")
            if "Raspberry Pi" in content:
                return True
    except Exception:
        pass
    return False


def get_soc_temperature() -> Union[float, str]:
    """
    Attempt to read SoC temperature on Raspberry Pi.
    Returns temperature in degrees Celsius (float) or 'UNAVAILABLE'.
    """
    if not is_raspberry_pi():
        return "UNAVAILABLE (Non-Raspberry Pi Host)"

    # Method 1: vcgencmd measure_temp
    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL).decode("utf-8")
        # Output format: temp=48.2'C
        if "temp=" in output:
            temp_str = output.split("=")[1].split("'")[0]
            return float(temp_str)
    except Exception:
        pass

    # Method 2: sysfs thermal zone
    try:
        tz_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if tz_path.exists():
            raw_temp = float(tz_path.read_text().strip())
            return round(raw_temp / 1000.0, 1)
    except Exception:
        pass

    return "UNAVAILABLE"


def run_edge_benchmark(
    model_path: str = "models/pest_yolov8n_best.onnx",
    source: str = "synthetic",
    num_frames: int = 30,
    warmup_frames: int = 5,
    width: int = 640,
    height: int = 480,
    save_report: Optional[str] = None,
) -> BenchmarkReport:
    """Execute edge pipeline benchmark and return structured report."""
    resolved_model = PROJECT_ROOT / model_path if not Path(model_path).is_absolute() else Path(model_path)
    if not resolved_model.exists():
        # Check for fallback
        alt_ext = ".pt" if resolved_model.suffix == ".onnx" else ".onnx"
        alt_model = resolved_model.with_suffix(alt_ext)
        if alt_model.exists():
            print(f"[BenchmarkEdge] Warning: {resolved_model.name} not found. Using fallback: {alt_model.name}")
            resolved_model = alt_model
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {resolved_model}")

    # Initialize video stream safely
    if source.lower() == "synthetic":
        stream = SafeVideoStream(source="synthetic", width=width, height=height)
    else:
        resolved_src = PROJECT_ROOT / source if not Path(source).is_absolute() else Path(source)
        stream = SafeVideoStream(source=str(resolved_src), loop=True, width=width, height=height, fallback_synthetic=True)

    print("=" * 70)
    print("KRISHNETRA EMBEDDED EDGE AI BENCHMARK PROTOCOL")
    print(f"Model Checkpoint : {resolved_model.name} ({resolved_model.suffix.upper()})")
    print(f"Input Resolution : {width}x{height}")
    print(f"Stream Source    : {stream.source_type.value} ({stream.source_path})")
    print(f"Total Frames     : {num_frames} ({warmup_frames} warm-up frames excluded)")
    print(f"Device Target    : CPU (Observation & Telemetry Only)")
    print(f"Hardware Safety  : ACTUATORS DISABLED")

    on_pi = is_raspberry_pi()
    soc_temp = get_soc_temperature()
    print(f"Platform Host    : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Raspberry Pi SBC : {'YES (Physical Pi Hardware)' if on_pi else 'NO (Desktop/Host Verification)'}")
    print(f"SoC Temperature  : {f'{soc_temp} C' if isinstance(soc_temp, (int, float)) else soc_temp}")

    if not on_pi:
        print("\n[NOTE] Running on non-Raspberry Pi host. Results do NOT reflect physical Pi 4 edge FPS.")
    print("=" * 70 + "\n")

    try:
        detector = PestDetector(weights_path=str(resolved_model), device="cpu")
        profiler = PipelineProfiler(warmup_frames=warmup_frames, input_width=width, input_height=height)

        print("[BenchmarkEdge] Profiling edge pipeline...")
        report = profiler.profile_pipeline(stream=stream, detector=detector, max_frames=num_frames)

        # Print latency summary
        stages = report.stage_latencies_ms
        print("\n" + "=" * 70)
        print("STAGE-BY-STAGE LATENCY & THROUGHPUT BREAKDOWN")
        print("=" * 70)
        print(f"{'Stage':<24} | {'Mean (ms)':<10} | {'Median (ms)':<11} | {'p95 (ms)':<10}")
        print("-" * 70)
        print(f"{'1. Ingestion':<24} | {stages['ingest']['mean_ms']:<10.2f} | {stages['ingest']['median_ms']:<11.2f} | {stages['ingest']['p95_ms']:<10.2f}")
        print(f"{'2. Preprocessing':<24} | {stages['preprocess']['mean_ms']:<10.2f} | {stages['preprocess']['median_ms']:<11.2f} | {stages['preprocess']['p95_ms']:<10.2f}")
        print(f"{'3. YOLO Inference':<24} | {stages['inference']['mean_ms']:<10.2f} | {stages['inference']['median_ms']:<11.2f} | {stages['inference']['p95_ms']:<10.2f}")
        print(f"{'4. Tracking (IoU)':<24} | {stages['tracking']['mean_ms']:<10.2f} | {stages['tracking']['median_ms']:<11.2f} | {stages['tracking']['p95_ms']:<10.2f}")
        print(f"{'5. Visual Servoing':<24} | {stages['servoing']['mean_ms']:<10.2f} | {stages['servoing']['median_ms']:<11.2f} | {stages['servoing']['p95_ms']:<10.2f}")
        print("-" * 70)
        print(f"{'TOTAL END-TO-END':<24} | {stages['e2e']['mean_ms']:<10.2f} | {stages['e2e']['median_ms']:<11.2f} | {stages['e2e']['p95_ms']:<10.2f}")
        print("=" * 70)
        print(f"Observed Throughput : {report.observed_fps:.2f} FPS")
        print(f"Timed Frames Count  : {report.timed_frames_count} frames")
        print(f"Memory (RSS)        : {report.process_memory_rss_mb} MB")
        print(f"CPU Utilization     : {report.cpu_percent}%")
        print("=" * 70 + "\n")

        if save_report:
            out_path = Path(save_report)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            report_dict = report.to_dict()
            report_dict["is_raspberry_pi"] = on_pi
            report_dict["soc_temperature_c"] = soc_temp
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2)
            print(f"[BenchmarkEdge] Report saved to: {out_path}")

        return report
    finally:
        stream.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Krishnetra Embedded Edge AI Benchmark Protocol")
    parser.add_argument("--model", type=str, default="models/pest_yolov8n_best.onnx", help="Path to model (.onnx or .pt)")
    parser.add_argument("--source", type=str, default="synthetic", help="Video source path or 'synthetic'")
    parser.add_argument("--frames", type=int, default=20, help="Total frames to benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up frames to exclude from aggregates")
    parser.add_argument("--width", type=int, default=640, help="Input frame width")
    parser.add_argument("--height", type=int, default=480, help="Input frame height")
    parser.add_argument("--compare", action="store_true", help="Run comparative benchmark between PyTorch and ONNX")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")

    args = parser.parse_args()

    if args.compare:
        profiler = PipelineProfiler(warmup_frames=args.warmup, input_width=args.width, input_height=args.height)
        pt_path = str(PROJECT_ROOT / "models" / "pest_yolov8n_best.pt")
        onnx_path = str(PROJECT_ROOT / "models" / "pest_yolov8n_best.onnx")
        stream_factory = lambda: SafeVideoStream(source=args.source, width=args.width, height=args.height, loop=True)
        comparison = profiler.compare_models(pt_path=pt_path, onnx_path=onnx_path, stream_factory=stream_factory, num_frames=args.frames)
        print(json.dumps(comparison, indent=2))
    else:
        run_edge_benchmark(
            model_path=args.model,
            source=args.source,
            num_frames=args.frames,
            warmup_frames=args.warmup,
            width=args.width,
            height=args.height,
            save_report=args.output,
        )


if __name__ == "__main__":
    main()
