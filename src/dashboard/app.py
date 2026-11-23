"""
AI-Powered Precision Pest Management - Analytics & Monitoring Dashboard
File: src/dashboard/app.py

Interactive telemetry dashboard for Krishnetra precision agriculture system.
Displays real-time pest detection analytics, timelines, confidence distributions,
and system safety interlocks using PostgreSQL or local fallback logs.

Run with:
  streamlit run src/dashboard/app.py
"""

import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.data_source import DashboardDataSource

# Set Page Config
st.set_page_config(
    page_title="Krishnetra | Precision Pest Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #52b788;
        margin-bottom: 20px;
    }
    .safety-card {
        background-color: #f8f9fa;
        border-left: 5px solid #2d6a4f;
        padding: 12px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_data_source() -> DashboardDataSource:
    return DashboardDataSource()


def main() -> None:
    ds = get_data_source()
    status = ds.get_system_status()

    # Header & Branding
    st.markdown('<div class="main-header">🌾 KRISHNETRA | Precision Pest Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Autonomous Precision Agriculture & Non-Chemical Deterrence Telemetry</div>', unsafe_allow_html=True)

    # Safety & System Status Banner
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        if status["postgres_connected"]:
            st.success(f"🟢 PostgreSQL: Connected ({status['host']})")
        else:
            st.warning("🟠 PostgreSQL: Offline (Local Log Fallback Active)")
    with col_s2:
        st.info("🤖 Model: 3-Class Pest YOLOv8n")
    with col_s3:
        st.info("🛡️ Decision: NO_ACTION (Default)")
    with col_s4:
        st.error("🔒 Actuators: DISABLED (Observation-Only)")

    # Top-Level Dashboard Tabs
    tab_telemetry, tab_profiler, tab_stream = st.tabs([
        "📊 Pest Telemetry Analytics",
        "⚡ Edge AI Pipeline Profiler & Benchmark",
        "🎥 Safe Video Ingestion & Stream Inspector",
    ])

    # =========================================================================
    # TAB 1: TELEMETRY ANALYTICS (Preserved 100% existing functionality)
    # =========================================================================
    with tab_telemetry:
        # Sidebar Filters
        st.sidebar.header("🔍 Telemetry Filters")
        time_horizon = st.sidebar.selectbox(
            "Time Horizon",
            options=["all", "today", "24h", "7d", "30d"],
            format_func=lambda x: {
                "all": "All Recorded History",
                "today": "Today Only (00:00 - Now)",
                "24h": "Last 24 Hours",
                "7d": "Last 7 Days",
                "30d": "Last 30 Days",
            }[x],
            index=0,
        )

        available_pests = ["All", "army worm", "blister beetle", "corn borer"]
        selected_pests = st.sidebar.multiselect("Filter Pest Classes", options=available_pests, default=["All"])
        min_confidence = st.sidebar.slider("Minimum Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

        # Auto-refresh button
        if st.sidebar.button("🔄 Refresh Telemetry"):
            st.cache_data.clear()
            st.rerun()

        # Query Data
        metrics = ds.get_overview_metrics(time_horizon=time_horizon, pest_filter=selected_pests, min_confidence=min_confidence)
        records_df = ds.fetch_all_records(time_horizon=time_horizon, pest_filter=selected_pests, min_confidence=min_confidence)
        stats_df = ds.get_pest_statistics_table(time_horizon=time_horizon, min_confidence=min_confidence)

        # 1. Executive Metric Cards
        st.markdown("### 📊 System Telemetry Overview")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric(label="Total Detections", value=metrics["total_detections"])
        with m_col2:
            st.metric(label="Today's Detections", value=metrics["today_detections"])
        with m_col3:
            st.metric(label="Unique Pests", value=metrics["unique_pests"])
        with m_col4:
            st.metric(label="Avg Confidence", value=f"{metrics['avg_confidence'] * 100:.1f}%")
        with m_col5:
            st.metric(label="Avg Inference Latency", value=f"{metrics['avg_latency_ms']} ms")

        st.divider()

        # Handle Empty Dataset State
        if records_df.empty:
            st.info("📌 **NO DETECTION DATA AVAILABLE FOR SELECTED FILTERS**\n\nRun the real-time camera monitor (`python src/monitoring/live_monitor.py`) or Raspberry Pi edge detector to record events.")
        else:
            # 2. Charts Section
            st.markdown("### 📈 Visual Pest Analytics & Distribution")
            c1, c2 = st.columns(2)

            with c1:
                if not stats_df.empty:
                    fig_bar = px.bar(
                        stats_df,
                        x="Pest Class",
                        y="Detection Count",
                        color="Pest Class",
                        color_discrete_map={
                            "army worm": "#2d6a4f",
                            "blister beetle": "#d90429",
                            "corn borer": "#ffb703",
                        },
                        text="Detection Count",
                        title="Detection Count by Target Pest Class",
                    )
                    fig_bar.update_layout(showlegend=False, template="plotly_white")
                    st.plotly_chart(fig_bar, use_container_width=True)

            with c2:
                if "timestamp" in records_df.columns:
                    timeline_df = records_df.copy()
                    timeline_df["time_bin"] = timeline_df["timestamp"].dt.floor("h" if time_horizon in ("today", "24h") else "D")
                    grouped_timeline = timeline_df.groupby(["time_bin", "class_name"]).size().reset_index(name="Count")

                    fig_timeline = px.line(
                        grouped_timeline,
                        x="time_bin",
                        y="Count",
                        color="class_name",
                        markers=True,
                        color_discrete_map={
                            "army worm": "#2d6a4f",
                            "blister beetle": "#d90429",
                            "corn borer": "#ffb703",
                        },
                        title="Detection Activity Timeline",
                        labels={"time_bin": "Timestamp", "Count": "Detections"},
                    )
                    fig_timeline.update_layout(template="plotly_white")
                    st.plotly_chart(fig_timeline, use_container_width=True)

            # 3. Pest Breakdown Table
            st.markdown("### 🪲 Target Pest Statistical Summary")
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            # 4. Recent Detections Log Table
            st.markdown("### 📋 Recent Detection Events (Latest 100)")
            display_cols = [c for c in ["timestamp", "frame_id", "model_type", "class_name", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "decision", "inference_latency_ms"] if c in records_df.columns]
            st.dataframe(records_df[display_cols].head(100), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: EDGE AI PROFILER & BENCHMARK
    # =========================================================================
    with tab_profiler:
        st.markdown("### ⚡ Edge AI Pipeline Latency & Throughput Profiler")
        st.markdown(
            "Measures real-time stage latencies: `[ Ingestion ]` ➔ `[ Preprocess ]` ➔ "
            "`[ YOLO Inference ]` ➔ `[ Multi-Frame Tracking ]` ➔ `[ Visual Servoing ]`."
        )

        models_info = ds.get_available_models()

        # Benchmarking Control Panel
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            available_fmts = []
            if models_info["pytorch_available"]:
                available_fmts.append("PYTORCH (.pt)")
            if models_info["onnx_available"]:
                available_fmts.append("ONNX (.onnx)")
            if not available_fmts:
                available_fmts = ["PYTORCH (.pt)"]

            selected_fmt = st.selectbox("Model Format", options=available_fmts, index=0)
            model_fmt_code = "ONNX" if "ONNX" in selected_fmt else "PYTORCH"

        with p_col2:
            stream_options = ["Synthetic Crop Canopy (Simulated)", "Pre-recorded Video Clip (tests/data/sample_pest_clip.mp4)"]
            selected_stream = st.selectbox("Video Source", options=stream_options, index=0)
            src_type = "FILE" if "Pre-recorded" in selected_stream else "SYNTHETIC"
            src_path = str(PROJECT_ROOT / "tests" / "data" / "sample_pest_clip.mp4") if src_type == "FILE" else None

        with p_col3:
            num_frames = st.slider("Benchmark Frames", min_value=5, max_value=30, value=15, step=5)

        with p_col4:
            warmup_frames = st.slider("Warm-up Frames (Excluded)", min_value=1, max_value=5, value=3, step=1)

        b_run1, b_run2 = st.columns(2)
        run_profiler = b_run1.button("🚀 Run Pipeline Profiler", use_container_width=True)
        run_comparison = b_run2.button("⚖️ Compare PyTorch vs ONNX Runtime", use_container_width=True, disabled=not models_info["both_available"])

        # Profiler Output Area
        if run_profiler:
            with st.spinner(f"Profiling edge pipeline using {model_fmt_code} ({num_frames} frames)..."):
                try:
                    report = ds.run_pipeline_benchmark(
                        model_format=model_fmt_code,
                        source_type=src_type,
                        source_path=src_path,
                        max_frames=num_frames,
                        warmup_frames=warmup_frames,
                    )
                    st.session_state["last_benchmark_report"] = report
                    st.success(f"✅ Profiling complete: {report['timed_frames_count']} timed frames evaluated ({report['warmup_frames_excluded']} warm-up excluded).")
                except Exception as e:
                    st.error(f"❌ Profiling failed: {e}")

        # Comparison Output Area
        if run_comparison:
            with st.spinner("Running side-by-side PyTorch vs ONNX benchmark..."):
                try:
                    comp = ds.run_model_comparison(
                        source_type=src_type,
                        source_path=src_path,
                        num_frames=num_frames,
                        warmup_frames=warmup_frames,
                    )
                    st.session_state["last_model_comparison"] = comp
                    st.success("✅ Comparative benchmark completed successfully.")
                except Exception as e:
                    st.error(f"❌ Model comparison failed: {e}")

        # Render Active Benchmark Report
        if "last_benchmark_report" in st.session_state:
            rep = st.session_state["last_benchmark_report"]
            stages = rep["stage_latencies_ms"]

            st.divider()
            st.markdown(f"#### ⏱️ Performance Metrics: `{rep['model_name']}` ({rep['model_format']})")

            # Metrics Row
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Observed FPS", f"{rep['observed_fps']:.1f}", help="Throughput = 1000 / E2E Mean Latency")
            k2.metric("E2E Latency", f"{stages['e2e']['mean_ms']:.1f} ms", f"p95: {stages['e2e']['p95_ms']:.1f} ms")
            k3.metric("YOLO Inference", f"{stages['inference']['mean_ms']:.1f} ms", f"p95: {stages['inference']['p95_ms']:.1f} ms")
            k4.metric("Tracking Latency", f"{stages['tracking']['mean_ms']:.2f} ms")
            k5.metric("Servoing Latency", f"{stages['servoing']['mean_ms']:.2f} ms")
            k6.metric("Process RAM", f"{rep['system_resources']['process_memory_rss_mb']} MB")

            # Stage Latency Horizontal Bar Chart
            stage_names = ["Ingest", "Preprocess", "YOLO Inference", "Multi-Frame Tracking", "Visual Servoing"]
            stage_means = [
                stages["ingest"]["mean_ms"],
                stages["preprocess"]["mean_ms"],
                stages["inference"]["mean_ms"],
                stages["tracking"]["mean_ms"],
                stages["servoing"]["mean_ms"],
            ]
            fig_stages = px.bar(
                x=stage_means,
                y=stage_names,
                orientation="h",
                labels={"x": "Mean Latency (ms)", "y": "Pipeline Stage"},
                title=f"Pipeline Stage Latency Breakdown (Total E2E: {stages['e2e']['mean_ms']:.1f} ms)",
                color=stage_names,
                color_discrete_sequence=["#52b788", "#74c69d", "#2d6a4f", "#1b4332", "#081c15"],
                text=[f"{v:.2f} ms" for v in stage_means],
            )
            fig_stages.update_layout(showlegend=False, template="plotly_white", height=320)
            st.plotly_chart(fig_stages, use_container_width=True)

            with st.expander("🔍 View Raw Benchmark Telemetry JSON"):
                st.json(rep)

        # Render Model Comparison Section
        if "last_model_comparison" in st.session_state:
            comp = st.session_state["last_model_comparison"]
            pt = comp["pytorch"]
            ox = comp["onnx"]

            st.divider()
            st.markdown("#### ⚖️ PyTorch vs ONNX Runtime Comparison")

            c_pt, c_ratio, c_ox = st.columns([2, 1, 2])
            with c_pt:
                st.markdown("**PyTorch (`.pt`)**")
                st.write(f"- Inference Mean: **{pt['stage_latencies_ms']['inference']['mean_ms']:.1f} ms**")
                st.write(f"- Inference p95: **{pt['stage_latencies_ms']['inference']['p95_ms']:.1f} ms**")
                st.write(f"- Total E2E: **{pt['stage_latencies_ms']['e2e']['mean_ms']:.1f} ms**")
                st.write(f"- Observed FPS: **{pt['observed_fps']:.1f} FPS**")
            with c_ratio:
                st.metric(
                    label="ONNX Speedup Factor",
                    value=f"{comp['inference_speedup_factor']:.2f}x",
                    delta=f"Faster: {comp['faster_model']}",
                )
            with c_ox:
                st.markdown("**ONNX Runtime (`.onnx`)**")
                st.write(f"- Inference Mean: **{ox['stage_latencies_ms']['inference']['mean_ms']:.1f} ms**")
                st.write(f"- Inference p95: **{ox['stage_latencies_ms']['inference']['p95_ms']:.1f} ms**")
                st.write(f"- Total E2E: **{ox['stage_latencies_ms']['e2e']['mean_ms']:.1f} ms**")
                st.write(f"- Observed FPS: **{ox['observed_fps']:.1f} FPS**")

        st.caption(
            "⚠️ **DISCLAIMER:** Desktop host metrics reflect relative algorithm efficiency on CPU. "
            "Host FPS does not represent physical Raspberry Pi 4 edge throughput. "
            "Hardware actuators remain permanently DISABLED."
        )

    # =========================================================================
    # TAB 3: SAFE VIDEO STREAM INSPECTOR
    # =========================================================================
    with tab_stream:
        st.markdown("### 🎥 Safe Video Ingestion & Stream Inspector")
        st.markdown("Inspect input frames from pre-recorded videos or synthetic generators without physical hardware.")

        v_source = st.radio(
            "Stream Source Selection",
            options=["Synthetic Canopy Generator", "Pre-recorded Video Clip (tests/data/sample_pest_clip.mp4)"],
            index=0,
            horizontal=True,
        )

        from src.monitoring.video_stream import SafeVideoStream
        selected_src = str(PROJECT_ROOT / "tests" / "data" / "sample_pest_clip.mp4") if "Pre-recorded" in v_source else "synthetic"

        if st.button("📸 Grab Frame Sample"):
            with SafeVideoStream(source=selected_src, width=640, height=480, loop=True) as stream:
                ret, frame_bgr, meta = stream.read_frame()
                if ret and frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Stream Source: {meta['source_type']} | Frame ID: {meta['frame_id']} | Synthetic: {meta['is_synthetic']}", use_column_width=True)
                    st.json(meta)
                else:
                    st.warning("Could not read frame from stream.")

        st.caption("Physical camera access is opt-in and disabled by default. Actuator outputs remain strictly locked.")

    # Footer
    st.divider()
    st.caption("Krishnetra Precision Pest Management System | College Project Prototype | Read-Only Telemetry Dashboard")


if __name__ == "__main__":
    main()
