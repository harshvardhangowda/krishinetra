"""
Unit and Integration Tests for Phase 17: Dashboard Profiler Integration & Embedded Edge Benchmark
File: tests/test_dashboard_profiler_integration.py

Validates:
1. DashboardDataSource model discovery (PyTorch and ONNX)
2. DashboardDataSource system status diagnostics
3. DashboardDataSource pipeline benchmark execution (Synthetic source)
4. DashboardDataSource pipeline benchmark execution (File source)
5. DashboardDataSource model comparison execution (PyTorch vs ONNX)
6. Missing model handling and error propagation
7. Raspberry Pi hardware environment detection (is_raspberry_pi)
8. SoC temperature reading and graceful fallback on non-Pi platforms
9. Embedded benchmark script execution (benchmark_edge.py)
10. Video stream resource cleanup and actuator safety isolation
"""

import unittest
from pathlib import Path

from src.dashboard.data_source import DashboardDataSource
from src.monitoring.video_stream import SafeVideoStream, StreamStatus
from deployment.raspberry_pi.benchmark_edge import (
    is_raspberry_pi,
    get_soc_temperature,
    run_edge_benchmark,
)


class TestDashboardProfilerIntegration(unittest.TestCase):

    def setUp(self):
        self.ds = DashboardDataSource()
        self.sample_video = "tests/data/sample_pest_clip.mp4"

    # 1. Model Checkpoint Discovery
    def test_model_discovery(self):
        models = self.ds.get_available_models()
        self.assertIn("pytorch_available", models)
        self.assertIn("onnx_available", models)
        self.assertTrue(models["pytorch_available"])
        self.assertTrue(models["onnx_available"])
        self.assertTrue(models["both_available"])

    # 2. System Status Contains Model Information
    def test_system_status_contains_models(self):
        status = self.ds.get_system_status()
        self.assertIn("models_available", status)
        self.assertIn("actuators", status)
        self.assertEqual(status["actuators"], "DISABLED")
        self.assertTrue(status["models_available"]["both_available"])

    # 3. Pipeline Benchmark Execution (Synthetic Source)
    def test_run_pipeline_benchmark_synthetic(self):
        report = self.ds.run_pipeline_benchmark(
            model_format="PYTORCH",
            source_type="SYNTHETIC",
            max_frames=5,
            warmup_frames=1,
            width=320,
            height=240,
        )
        self.assertEqual(report["model_format"], "PYTORCH")
        self.assertEqual(report["source_type"], "SYNTHETIC")
        self.assertEqual(report["total_frames_processed"], 5)
        self.assertEqual(report["timed_frames_count"], 4)
        self.assertGreater(report["observed_fps"], 0.0)
        self.assertEqual(report["hardware_actuators"], "DISABLED")
        self.assertEqual(report["telemetry_type"], "SIMULATED")

    # 4. Pipeline Benchmark Execution (File Source with ONNX)
    def test_run_pipeline_benchmark_file_onnx(self):
        report = self.ds.run_pipeline_benchmark(
            model_format="ONNX",
            source_type="FILE",
            source_path=self.sample_video,
            max_frames=5,
            warmup_frames=1,
            width=320,
            height=240,
        )
        self.assertEqual(report["model_format"], "ONNX")
        self.assertEqual(report["source_type"], "FILE")
        self.assertEqual(report["total_frames_processed"], 5)
        self.assertGreater(report["observed_fps"], 0.0)
        self.assertGreater(report["stage_latencies_ms"]["inference"]["mean_ms"], 0.0)

    # 5. Model Comparison (PyTorch vs ONNX)
    def test_run_model_comparison(self):
        comparison = self.ds.run_model_comparison(
            source_type="SYNTHETIC",
            num_frames=5,
            warmup_frames=1,
            width=320,
            height=240,
        )
        self.assertIn("pytorch", comparison)
        self.assertIn("onnx", comparison)
        self.assertIn("inference_speedup_factor", comparison)
        self.assertIn(comparison["faster_model"], ("PYTORCH", "ONNX"))
        self.assertEqual(comparison["telemetry_type"], "SIMULATED")
        self.assertEqual(comparison["hardware_actuators"], "DISABLED")

    # 6. Missing Model Error Handling
    def test_missing_model_error_handling(self):
        # Override model check temporarily
        orig_onnx = Path("models/pest_yolov8n_best.onnx")
        with self.assertRaises(Exception):
            # Attempting non-existent model format
            self.ds.run_pipeline_benchmark(model_format="TENSORRT", source_type="SYNTHETIC")

    # 7. Raspberry Pi Hardware Detection
    def test_is_raspberry_pi_detection(self):
        # Host running on Windows/Mac/CI will detect as False
        res = is_raspberry_pi()
        self.assertIsInstance(res, bool)

    # 8. SoC Temperature Handling
    def test_soc_temperature_handling(self):
        temp = get_soc_temperature()
        if is_raspberry_pi():
            self.assertTrue(isinstance(temp, float) or temp == "UNAVAILABLE")
        else:
            self.assertIn("UNAVAILABLE", str(temp))

    # 9. Embedded Benchmark Script Execution
    def test_benchmark_edge_execution(self):
        report = run_edge_benchmark(
            model_path="models/pest_yolov8n_best.onnx",
            source="synthetic",
            num_frames=5,
            warmup_frames=1,
            width=320,
            height=240,
        )
        self.assertEqual(report.total_frames_processed, 5)
        self.assertEqual(report.warmup_frames_excluded, 1)
        self.assertGreater(report.observed_fps, 0.0)
        self.assertEqual(report.hardware_actuators, "DISABLED")

    # 10. Stream Cleanup and Resource Safety
    def test_stream_cleanup_and_safety(self):
        stream = SafeVideoStream(source="synthetic")
        self.assertTrue(stream.is_active())
        stream.release()
        self.assertEqual(stream.status, StreamStatus.CLOSED)
        # Calling read_frame on closed stream should return False cleanly
        ret, frame, meta = stream.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)


if __name__ == "__main__":
    unittest.main()
