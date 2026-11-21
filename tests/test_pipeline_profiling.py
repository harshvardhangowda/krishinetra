"""
Unit and Integration Tests for Phase 16: Edge AI Benchmarking & Pipeline Profiling
File: tests/test_pipeline_profiling.py

Validates:
1. Video source missing file handling (FileNotFoundError vs graceful synthetic fallback)
2. Video source end-of-stream (EOS) detection without infinite hangs
3. Video source auto-looping on pre-recorded files
4. Injectable frame source for automated tests (no camera required)
5. Physical camera isolation & safety policy enforcement
6. SafeVideoStream resource release and context manager cleanup
7. Stage latency calculations (mean, median, p95, min, max)
8. Warm-up frames exclusion from benchmark statistics
9. Full end-to-end edge AI pipeline profiling (all 5 stages)
10. PyTorch vs ONNX benchmark comparison with verified model weights
11. System resource utilization tracking (CPU % and Memory RSS MB)
12. Safety interlocks & explicit SIMULATED metadata tagging
"""

import unittest
from pathlib import Path
import numpy as np

from src.detection.detector import PestDetector
from src.monitoring.video_stream import (
    SafeVideoStream,
    StreamSourceType,
    StreamStatus,
)
from src.monitoring.pipeline_profiler import (
    PipelineProfiler,
    BenchmarkReport,
    compute_stage_stats,
)
from src.tracking.target_tracker import MultiFrameTracker


class TestPipelineProfiling(unittest.TestCase):

    def setUp(self):
        self.sample_video = "tests/data/sample_pest_clip.mp4"
        self.pt_weights = "models/pest_yolov8n_best.pt"
        self.onnx_weights = "models/pest_yolov8n_best.onnx"

    # 1. Missing File Handling
    def test_missing_file_handling(self):
        # Strict mode: raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            SafeVideoStream(source="non_existent_clip_123.mp4", fallback_synthetic=False)

        # Fallback mode: activates synthetic feed gracefully
        stream = SafeVideoStream(source="non_existent_clip_123.mp4", fallback_synthetic=True)
        self.assertEqual(stream.source_type, StreamSourceType.SYNTHETIC)
        self.assertEqual(stream.status, StreamStatus.ACTIVE)
        ret, frame, meta = stream.read_frame()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertTrue(meta["is_synthetic"])
        stream.release()

    # 2. End-of-Stream (EOS) Detection
    def test_end_of_stream_detection(self):
        # sample_pest_clip.mp4 has exactly 15 frames
        stream = SafeVideoStream(source=self.sample_video, loop=False, fallback_synthetic=False)
        self.assertEqual(stream.source_type, StreamSourceType.FILE)

        read_count = 0
        while True:
            ret, frame, meta = stream.read_frame()
            if not ret or frame is None:
                self.assertTrue(meta["is_eos"])
                self.assertEqual(stream.status, StreamStatus.END_OF_STREAM)
                break
            read_count += 1

        self.assertEqual(read_count, 15)
        stream.release()

    # 3. Video Looping Functionality
    def test_video_looping(self):
        # Read 25 frames from a 15-frame video with loop=True
        stream = SafeVideoStream(source=self.sample_video, loop=True, fallback_synthetic=False)
        self.assertEqual(stream.source_type, StreamSourceType.FILE)

        frames_read = 0
        for _ in range(25):
            ret, frame, meta = stream.read_frame()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertFalse(meta["is_eos"])
            frames_read += 1

        self.assertEqual(frames_read, 25)
        stream.release()

    # 4. Injectable Frame Sequence
    def test_injected_frame_stream(self):
        custom_frames = [np.full((240, 320, 3), fill_value=i * 20, dtype=np.uint8) for i in range(5)]
        stream = SafeVideoStream(injected_frames=custom_frames, loop=False)
        self.assertEqual(stream.source_type, StreamSourceType.INJECTED)

        retrieved = []
        while True:
            ret, frame, meta = stream.read_frame()
            if not ret:
                break
            retrieved.append(frame)

        self.assertEqual(len(retrieved), 5)
        self.assertEqual(stream.status, StreamStatus.END_OF_STREAM)
        stream.release()

    # 5. Camera Safety Policy (Disabled by Default)
    def test_camera_safety_policy(self):
        # When allow_camera is False, index 0 defaults to synthetic without touching hardware
        stream = SafeVideoStream(source=0, allow_camera=False)
        self.assertEqual(stream.source_type, StreamSourceType.SYNTHETIC)
        self.assertEqual(stream.status, StreamStatus.ACTIVE)
        ret, frame, _ = stream.read_frame()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        stream.release()

    # 6. Context Manager & Clean Resource Release
    def test_context_manager_release(self):
        with SafeVideoStream(source="synthetic") as s:
            self.assertTrue(s.is_active())
            ret, frame, _ = s.read_frame()
            self.assertTrue(ret)

        self.assertEqual(s.status, StreamStatus.CLOSED)

    # 7. Latency Metric Calculation (Mean, Median, P95)
    def test_latency_percentile_calculations(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_stage_stats(latencies)
        self.assertAlmostEqual(stats.mean_ms, 30.0, places=2)
        self.assertAlmostEqual(stats.median_ms, 30.0, places=2)
        self.assertAlmostEqual(stats.min_ms, 10.0, places=2)
        self.assertAlmostEqual(stats.max_ms, 50.0, places=2)
        self.assertGreater(stats.p95_ms, 45.0)

        # Empty list handling
        empty_stats = compute_stage_stats([])
        self.assertEqual(empty_stats.mean_ms, 0.0)

    # 8. Warm-Up Frames Exclusion
    def test_warmup_frames_exclusion(self):
        profiler = PipelineProfiler(warmup_frames=3, input_width=320, input_height=240)
        stream = SafeVideoStream(source="synthetic", width=320, height=240)
        detector = PestDetector(weights_path=self.pt_weights, device="cpu")

        report = profiler.profile_pipeline(
            stream=stream,
            detector=detector,
            max_frames=8,
        )
        stream.release()

        self.assertEqual(report.total_frames_processed, 8)
        self.assertEqual(report.warmup_frames_excluded, 3)
        self.assertEqual(report.timed_frames_count, 5)
        self.assertGreater(report.observed_fps, 0.0)

    # 9. End-to-End Pipeline Profiling
    def test_end_to_end_pipeline_profiling(self):
        profiler = PipelineProfiler(warmup_frames=2, input_width=320, input_height=240)
        stream = SafeVideoStream(source=self.sample_video, loop=True, width=320, height=240)
        detector = PestDetector(weights_path=self.pt_weights, device="cpu")

        report = profiler.profile_pipeline(
            stream=stream,
            detector=detector,
            max_frames=6,
        )
        stream.release()

        self.assertEqual(report.model_format, "PYTORCH")
        self.assertEqual(report.source_type, StreamSourceType.FILE.value)
        self.assertEqual(report.dropped_or_failed_frames, 0)
        self.assertGreater(report.observed_fps, 0.0)

        # Verify all stages are populated
        stages = [
            report.ingest_latency,
            report.preprocess_latency,
            report.inference_latency,
            report.tracking_latency,
            report.servoing_latency,
            report.e2e_latency,
        ]
        for s in stages:
            self.assertGreater(s.mean_ms, 0.0)
            self.assertGreaterEqual(s.max_ms, s.min_ms)

    # 10. PyTorch vs ONNX Comparative Benchmark
    def test_pytorch_vs_onnx_benchmark_comparison(self):
        profiler = PipelineProfiler(warmup_frames=2, input_width=320, input_height=240)

        comparison = profiler.compare_models(
            pt_path=self.pt_weights,
            onnx_path=self.onnx_weights,
            num_frames=6,
        )

        self.assertIn("pytorch", comparison)
        self.assertIn("onnx", comparison)
        self.assertEqual(comparison["pytorch"]["model_format"], "PYTORCH")
        self.assertEqual(comparison["onnx"]["model_format"], "ONNX")
        self.assertGreater(comparison["inference_speedup_factor"], 0.0)
        self.assertIn(comparison["faster_model"], ("PYTORCH", "ONNX"))
        self.assertEqual(comparison["telemetry_type"], "SIMULATED")
        self.assertEqual(comparison["hardware_actuators"], "DISABLED")

    # 11. System Resource Metrics
    def test_system_resource_metrics(self):
        profiler = PipelineProfiler()
        cpu_pct, mem_mb = profiler.get_system_metrics()
        # Either float values or UNAVAILABLE
        if isinstance(cpu_pct, float):
            self.assertGreaterEqual(cpu_pct, 0.0)
        else:
            self.assertEqual(cpu_pct, "UNAVAILABLE")

        if isinstance(mem_mb, float):
            self.assertGreater(mem_mb, 0.0)
        else:
            self.assertEqual(mem_mb, "UNAVAILABLE")

    # 12. Safety and Metadata Tagging
    def test_safety_and_metadata_tagging(self):
        profiler = PipelineProfiler(warmup_frames=1, input_width=320, input_height=240)
        stream = SafeVideoStream(source="synthetic")
        detector = PestDetector(weights_path=self.pt_weights, device="cpu")

        report = profiler.profile_pipeline(stream, detector, max_frames=3)
        stream.release()

        report_dict = report.to_dict()
        self.assertEqual(report_dict["telemetry_type"], "SIMULATED")
        self.assertEqual(report_dict["hardware_actuators"], "DISABLED")
        self.assertIn("Desktop benchmark", report_dict["disclaimer"])


if __name__ == "__main__":
    unittest.main()
