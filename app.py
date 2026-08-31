"""
SIH26124: AI-Powered Mobile Urban Intelligence Platform
Streamlit Smart-City Decision Support Dashboard — Fleet Authentication & Bus Identity Integration
"""
import time
import tempfile
import json
from pathlib import Path
import streamlit as st
import cv2
import pandas as pd
from streamlit_folium import st_folium

from src.detection.vehicle_detector import VehicleDetector
from src.detection.road_damage_detector import RoadDamageDetector
from src.events.geo_tagger import GeoTagger
from src.events.generator import EventGenerator
from src.storage.db_manager import DatabaseManager
from src.video.processor import VideoProcessor
from src.maps.folium_map import create_event_map, render_folium_map
from src.analytics.traffic_metrics import (
    TrafficAnalytics,
    compute_traffic_metrics,
    VALID_VEHICLE_CLASSES
)
from src.analytics.road_health import (
    RoadHealthAnalyzer,
    PROTOTYPE_DISCLAIMER
)
from src.maps.map_generator import create_urban_map
from config import settings
from config.buses import (
    get_available_bus_ids,
    get_available_buses,
    get_bus_info,
    format_bus_display,
    authenticate_bus,
    UNKNOWN_BUS_LABEL
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
# Custom Styling for Clean Professional Look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #e2e8f0;
        margin-bottom: 10px;
    }
    .disclaimer-banner {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 14px;
        font-size: 13px;
        color: #1e3a8a;
    }
    .warning-banner {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 14px;
        font-size: 13px;
        color: #78350f;
    }
    .auth-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 24px;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Pipeline Initialization (Cached for CPU Efficiency)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Inference Engines & Storage...")
def load_pipeline():
    veh_det = VehicleDetector(model_name="yolov8n.pt", conf_threshold=0.35)
    dam_det = RoadDamageDetector(conf_threshold=0.30)
    db_mgr = DatabaseManager()
    geo_tag = GeoTagger()
    road_health = RoadHealthAnalyzer()
    return veh_det, dam_det, db_mgr, geo_tag, road_health

vehicle_detector, road_damage_detector, db_manager, geo_tagger, road_health_analyzer = load_pipeline()


# -----------------------------------------------------------------------------
# 3. Session State & Prototype Fleet Authentication Screen
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "current_bus_id" not in st.session_state:
    st.session_state["current_bus_id"] = None
if "data_scope" not in st.session_state:
    st.session_state["data_scope"] = "CURRENT BUS"


# Render Login Screen if not authenticated
if not st.session_state["authenticated"]:
    st.markdown("<h1 style='text-align: center;'>SIH26124 Fleet Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>AI-Powered Mobile Urban Sensing Platform</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-banner" style="text-align: center;">
        <b>⚠️ Prototype Fleet Authentication</b><br/>
        This is a hackathon prototype, NOT production authentication.
    </div>
    """, unsafe_allow_html=True)

    auth_col1, auth_col2, auth_col3 = st.columns([1, 2, 1])
    with auth_col2:
        st.markdown("### 🚌 BUS AUTHENTICATION")
        bus_ids = get_available_bus_ids()

        with st.form("prototype_bus_login_form"):
            selected_bus_id = st.selectbox(
                "Bus ID:",
                options=bus_ids,
                format_func=lambda b_id: format_bus_display(b_id)
            )
            entered_pin = st.text_input("Access PIN:", type="password", placeholder="Enter prototype PIN")
            login_btn = st.form_submit_button("LOGIN", type="primary", use_container_width=True)

            if login_btn:
                if authenticate_bus(selected_bus_id, entered_pin):
                    st.session_state["authenticated"] = True
                    st.session_state["current_bus_id"] = selected_bus_id
                    st.session_state["data_scope"] = "CURRENT BUS"
                    st.success(f"Authenticated as {selected_bus_id} successfully.")
                    st.rerun()
                else:
                    st.error("Authentication failed: Invalid Bus ID or Access PIN. Please try again.")

    st.stop()


# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Active Bus Information
# -----------------------------------------------------------------------------
current_bus_id = st.session_state["current_bus_id"]
current_bus_info = get_bus_info(current_bus_id)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/bus.png", width=64)
    st.title("SIH26124 Fleet Hub")
    st.caption("AI-Powered Urban Transit Sensing")
    st.markdown("---")

    # Active Bus Identity Card
    st.subheader("🚌 Active Bus Identity")
    if current_bus_info:
        st.markdown(f"""
        <div class="metric-card">
            <b>{current_bus_info['bus_id']}</b> — {current_bus_info['display_name']}<br/>
            <span style="font-size: 11px; color: #475569;">🛣️ <b>Route:</b> {current_bus_info['route']}</span><br/>
            <span style="font-size: 11px; color: #16a34a; font-weight: bold;">🟢 Authenticated / Active</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write(f"Bus ID: `{current_bus_id}`")

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["current_bus_id"] = None
        st.session_state["data_scope"] = "CURRENT BUS"
        st.rerun()

    st.markdown("---")
    st.subheader("📡 Data Scope Concept")
    scope_option = st.radio(
        "Select Data Selection Scope",
        ["CURRENT BUS", "FLEET VIEW"],
        index=0 if st.session_state.get("data_scope") == "CURRENT BUS" else 1,
        help="CURRENT BUS: Shows events captured by this authenticated vehicle. FLEET VIEW: Aggregates events across all vehicles in the fleet."
    )
    st.session_state["data_scope"] = scope_option

    st.markdown("---")
    st.subheader("⚙️ System Status")
    st.write(f"🟢 **Vehicle AI**: `{vehicle_detector.model_name}`")

    # Mode toggle for Road Damage
    simulate_toggle = st.toggle("Force Demo AI Mode", value=False, help="Forces heuristic road damage simulation mode for testing.")
    road_damage_detector.force_demo_mode = simulate_toggle
    road_damage_detector.detection_mode = "DEMO_SIMULATION" if simulate_toggle else "REAL_AI"

    if road_damage_detector.detection_mode == "REAL_AI":
        st.write(f"🟢 **Damage AI**: `REAL_AI` ({road_damage_detector.model_name})")
    else:
        st.write("🟠 **Damage AI**: `DEMO_SIMULATION` (Simulation Mode)")

    st.write("📍 **GPS Mode**: `SIMULATED GPS`")

    db_stats = db_manager.get_event_statistics()
    st.write(f"💾 **SQLite Events**: `{db_stats['total_events']}` records")
    st.markdown("---")

    st.subheader("🎯 Detection Thresholds")
    veh_conf = st.slider("Vehicle Confidence", 0.10, 0.90, 0.35, 0.05)
    vehicle_detector.conf_threshold = veh_conf
    dam_conf = st.slider("Damage Confidence", 0.10, 0.90, 0.30, 0.05)
    road_damage_detector.conf_threshold = dam_conf

    st.markdown("---")
    st.subheader("🛠️ Database Tools")
    if st.button("🗑️ Clear Database Events", use_container_width=True):
        db_manager.clear_events()
        st.success("Database cleared successfully.")
        st.rerun()


# -----------------------------------------------------------------------------
# 5. Header & Top Transparency Disclosure
# -----------------------------------------------------------------------------
st.title("🚌 AI-Powered Mobile Urban Intelligence Platform")
st.markdown("**Smart-City Decision Support Dashboard** — Smart India Hackathon (SIH26124)")

# Prominent Transparency & Active Scope Indicators
data_scope = st.session_state["data_scope"]
b1, b2, b3, b4 = st.columns([1.2, 1.2, 1.1, 1.1])

with b1:
    st.info(f"🚌 **Active Bus**: `{current_bus_id}`")

with b2:
    if data_scope == "CURRENT BUS":
        st.success(f"🎯 **Scope**: `CURRENT BUS ({current_bus_id})`")
    else:
        st.warning("🌐 **Scope**: `FLEET VIEW (All Buses)`")

with b3:
    if road_damage_detector.detection_mode == "REAL_AI":
        st.success(f"🟢 **Damage AI**: `REAL_AI`")
    else:
        st.warning("🟠 **Damage AI**: `DEMO_SIMULATION`")

with b4:
    st.info("📍 **GPS**: `SIMULATED GPS`")

st.markdown("""
<div class="disclaimer-banner">
    <b>ℹ️ Data Transparency & Methodology Notice:</b><br/>
    • <b>Bus Identity:</b> Events generated by this session are tagged with the active authenticated bus identity.<br/>
    • <b>GPS Telemetry:</b> Coordinates are interpolated along a predefined simulated transit corridor (Route-7B, New Delhi).<br/>
    • <b>Traffic Analytics:</b> Represents vehicle detection events captured by mobile transit edge cameras and deduplicated via spatial-temporal tracking. Observational proxy data only; not induction-loop calibrated.<br/>
    • <b>Traffic Density:</b> Heuristic classification based on observed detection rates (events/min).
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. Tabbed Interface for Clean Organization
# -----------------------------------------------------------------------------
tab_video, tab_traffic, tab_damage, tab_gis, tab_table, tab_export = st.tabs([
    "📹 Video Ingestion & Processing",
    "📊 Traffic Analytics",
    "⚠️ Road Damage Analytics",
    "🗺️ GIS Spatial Map",
    "📋 Event Explorer",
    "📥 Data Export & Logs"
])


# =============================================================================
# TAB 1: VIDEO INGESTION & PROCESSING
# =============================================================================
with tab_video:
    st.subheader("1. Ingest Road Video & Run Dual AI Pipeline")
    st.caption(f"Newly processed events will be tagged with active bus ID: **{current_bus_id}**")

    col_v1, col_v2 = st.columns([1.5, 1])

    with col_v1:
        input_mode = st.radio(
            "Select Video Feed Source",
            [
                "Real Road Video (real_road_sample.mp4)",
                "Backup Demo Video (backup_road_demo.mp4)",
                "Upload Custom Video (.mp4, .avi, .mov)"
            ],
            horizontal=False
        )

        selected_video_path = None
        if input_mode == "Real Road Video (real_road_sample.mp4)":
            p = settings.SAMPLE_DATA_DIR / "real_road_sample.mp4"
            if p.exists():
                selected_video_path = str(p)
                st.success(f"Loaded Real Video Feed (`{round(p.stat().st_size / (1024*1024), 2)} MB`)")
        elif input_mode == "Backup Demo Video (backup_road_demo.mp4)":
            p = settings.SAMPLE_DATA_DIR / "backup_road_demo.mp4"
            if p.exists():
                selected_video_path = str(p)
                st.info(f"Loaded Backup Video Feed (`{round(p.stat().st_size / 1024, 1)} KB`)")
        else:
            uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                selected_video_path = tfile.name
                st.success("Custom video uploaded successfully.")

    with col_v2:
        st.markdown("**Processing Parameters**")
        max_frames_slider = st.slider("Max Frames to Process (0 = Entire Video)", 0, 600, 60, 15)
        max_frames_val = None if max_frames_slider == 0 else max_frames_slider
        frame_skip = st.selectbox("Frame Skip Rate", [1, 2, 3, 5], index=0, help="Skip frames to accelerate processing on CPU.")
        reset_db_on_run = st.checkbox("Clear Database Before Run", value=False)

    st.markdown("---")

    if selected_video_path is not None:
        if st.button("🚀 Process Video, Geotag & Store Events", type="primary", use_container_width=True):
            if reset_db_on_run:
                db_manager.clear_events()

            processor = VideoProcessor(
                vehicle_detector=vehicle_detector,
                road_damage_detector=road_damage_detector,
                db_manager=db_manager,
                geo_tagger=geo_tagger
            )

            progress_bar = st.progress(0, text="Initializing processing pipeline...")
            metrics_placeholder = st.empty()
            output_save_path = settings.SAMPLE_DATA_DIR / "annotated_real_road_sample.mp4"

            def ui_progress(processed, total, current_fps, current_v, current_d):
                pct = min(int((processed / max(total, 1)) * 100), 100)
                progress_bar.progress(
                    pct,
                    text=f"Frame {processed}/{total} ({pct}%) | Live FPS: {current_fps:.1f} | Veh Detections: {current_v} | Damage Detections: {current_d}"
                )

            start_proc_time = time.perf_counter()
            results = processor.process_video(
                input_path=selected_video_path,
                output_path=str(output_save_path),
                frame_skip=frame_skip,
                max_frames=max_frames_val,
                save_to_db=True,
                bus_id=current_bus_id,
                progress_callback=ui_progress
            )
            progress_bar.progress(100, text="Pipeline Execution Complete!")

            st.success(f"✅ Successfully processed {results['processed_frames_count']} frames in {results['total_processing_time_sec']}s ({results['complete_pipeline_fps']} FPS). Generated {results['total_generated_events']} new deduplicated events tagged to bus **{current_bus_id}**.")

            # Performance Metrics Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Frames", results["processed_frames_count"])
            with m2:
                st.metric("Pipeline Speed", f"{results['complete_pipeline_fps']} FPS")
            with m3:
                st.metric("Model Inference Speed", f"{results['model_inference_fps']} FPS")
            with m4:
                st.metric("Deduplicated Events", results["total_generated_events"], delta=f"-{results['total_duplicates_filtered']} dupes")

            # Video Preview
            if output_save_path.exists() and output_save_path.stat().st_size > 0:
                st.markdown("#### 🎬 Annotated Output Video")
                st.video(str(output_save_path))
                st.caption(f"Saved annotated file: `{output_save_path}` ({round(output_save_path.stat().st_size / 1024, 1)} KB)")


# =============================================================================
# TAB 2: TRAFFIC ANALYTICS (CHECKPOINT 5)
# =============================================================================
with tab_traffic:
    st.subheader("2. Traffic Volume & Density Analytics")
    if data_scope == "CURRENT BUS":
        st.caption(f"📊 Displaying traffic analytics filtered to **Current Bus ({current_bus_id})**")
        traffic_events = db_manager.filter_events(bus_id=current_bus_id, limit=100000)
    else:
        st.caption("🌐 Displaying aggregate fleet-wide traffic analytics (**Fleet View**)")
        traffic_events = db_manager.get_events(limit=100000)

    # Fetch traffic summary based on active scope
    traffic_summary = compute_traffic_metrics(events=traffic_events)
    veh_counts = traffic_summary["vehicle_counts"]
    temp_dist = traffic_summary["temporal_distribution"]
    density_info = traffic_summary["traffic_density_classification"]

    # Top KPI Row
    tk1, tk2, tk3, tk4 = st.columns(4)
    with tk1:
        st.metric("Total Vehicle Events", veh_counts["total_vehicle_count"], help="Valid vehicle classes: car, bus, truck, motorcycle")
    with tk2:
        st.metric("Dominant Vehicle Class", (veh_counts["dominant_vehicle_class"] or "None").upper())
    with tk3:
        st.metric("Traffic Density Level", density_info["density_level"], help="Heuristic density classification based on observed rate")
    with tk4:
        st.metric("Observed Traffic Rate", f"{density_info['observed_rate_per_minute']} ev/min")

    st.markdown("---")

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("#### 🚗 Vehicle Breakdown by Class")
        class_counts = veh_counts["by_class"]
        if class_counts:
            df_classes = pd.DataFrame([
                {"Vehicle Class": k.capitalize(), "Event Count": v, "Share (%)": f"{veh_counts['class_percentage_shares'].get(k, 0.0):.1f}%"}
                for k, v in class_counts.items()
            ])
            st.dataframe(df_classes, use_container_width=True, hide_index=True)
            st.bar_chart(pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"]).set_index("Class"))
        else:
            st.info("No valid vehicle events detected in current dataset.")

    with tc2:
        st.markdown("#### ⏱️ Temporal Event Distribution")
        st.write(f"• **Observation Duration**: `{temp_dist['duration_seconds']} seconds`")
        st.write(f"• **Rate per Hour**: `{temp_dist['events_per_hour']} events/hr`")
        st.write(f"• **Peak Bucket Count**: `{temp_dist['peak_count']} events`")

        if temp_dist["time_buckets"]:
            df_time = pd.DataFrame([
                {"Time Interval": b["timestamp"].split("T")[-1][:8], "Count": b["count"]}
                for b in temp_dist["time_buckets"]
            ])
            st.line_chart(df_time.set_index("Time Interval"))
        else:
            st.caption("Temporal series requires positive duration vehicle records.")


# =============================================================================
# TAB 3: ROAD DAMAGE ANALYTICS
# =============================================================================
with tab_damage:
    st.subheader("3. Road Surface Condition & Damage Analytics")

    if data_scope == "CURRENT BUS":
        st.caption(f"⚠️ Displaying road damage analytics for **Current Bus ({current_bus_id})**")
        dam_events_scope = db_manager.filter_events(event_type="ROAD_DAMAGE", bus_id=current_bus_id, limit=100000)
        all_health_events_scope = db_manager.filter_events(bus_id=current_bus_id, limit=100000)
    else:
        st.caption("🌐 Displaying aggregate fleet-wide road damage analytics (**Fleet View**)")
        dam_events_scope = db_manager.filter_events(event_type="ROAD_DAMAGE", limit=100000)
        all_health_events_scope = db_manager.get_events(limit=100000)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("#### 📊 Damage Severity Breakdown")
        sev_counts = {"low": 0, "medium": 0, "high": 0}
        for e in dam_events_scope:
            s = e.severity.lower()
            if s in sev_counts:
                sev_counts[s] += 1

        filtered_sev = {k.capitalize(): v for k, v in sev_counts.items() if v > 0}
        if filtered_sev:
            st.bar_chart(pd.DataFrame(list(filtered_sev.items()), columns=["Severity", "Count"]).set_index("Severity"))
        else:
            st.info("No road damage records currently stored for this scope.")

    with dc2:
        st.markdown("#### 🤖 Damage AI Detection Modes")
        mode_counts = {}
        for e in dam_events_scope:
            m = e.detection_mode
            mode_counts[m] = mode_counts.get(m, 0) + 1

        df_modes = pd.DataFrame([{"Detection Mode": k, "Events": v} for k, v in mode_counts.items()])
        st.dataframe(df_modes, use_container_width=True, hide_index=True)
        st.caption("ℹ️ `REAL_AI`: Model inference via YOLO road damage weights. `DEMO_SIMULATION`: Heuristic simulation mode.")

    st.markdown("---")
    st.subheader("🛣️ Road Health & Maintenance Priority")

    # Mandated Prototype Disclaimer
    st.markdown(f"""
    <div class="disclaimer-banner">
        <b>⚠️ Decision-Support Disclaimer:</b><br/>
        {PROTOTYPE_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)

    st.caption("📍 **Decision-Support Pipeline Flow**: Road Damage Events ➔ Spatial/Segment Aggregation ➔ Road Health Score (0–100) ➔ Maintenance Priority Tier")

    if not all_health_events_scope:
        st.info("ℹ️ No road events currently recorded in this scope. Ingest and process a video feed in Tab 1 to compute Road Health & Maintenance Priority rankings.")
    else:
        # Filter to ensure valid vehicle events semantics
        valid_health_events = [
            e for e in all_health_events_scope
            if e.event_type == "ROAD_DAMAGE" or (e.event_type == "VEHICLE" and e.class_name.lower() in VALID_VEHICLE_CLASSES)
        ]

        health_report = road_health_analyzer.analyze_events(valid_health_events)

        # 1. Overall Network KPI Row
        rh1, rh2, rh3, rh4, rh5 = st.columns(5)
        with rh1:
            st.metric("Overall Health", f"{health_report.overall_network_health:.1f} / 100")
        with rh2:
            st.metric("Analyzed Segments", health_report.total_segments)
        with rh3:
            st.metric("Critical Segments", health_report.critical_segments_count)
        with rh4:
            st.metric("High Priority Segments", health_report.high_priority_segments_count)
        with rh5:
            st.metric("Total Damage Events", health_report.total_damage_events)

        # 2. Priority Distribution Overview
        st.markdown("#### 🎯 Maintenance Priority Distribution")
        dist_c1, dist_c2, dist_c3, dist_c4, dist_c5 = st.columns(5)
        with dist_c1:
            st.markdown(f"🔴 **CRITICAL**: `{health_report.critical_segments_count}`")
        with dist_c2:
            st.markdown(f"🟠 **HIGH**: `{health_report.high_priority_segments_count}`")
        with dist_c3:
            st.markdown(f"🟡 **MEDIUM**: `{health_report.medium_priority_segments_count}`")
        with dist_c4:
            st.markdown(f"🔵 **LOW**: `{health_report.low_priority_segments_count}`")
        with dist_c5:
            st.markdown(f"🟢 **NORMAL**: `{health_report.normal_segments_count}`")

        # 3. Segment Priority Ranking Table
        st.markdown("#### 📋 Segment Maintenance Priority Rankings")

        filter_col, _ = st.columns([1.5, 2.5])
        with filter_col:
            selected_tier = st.selectbox(
                "Filter by Priority Tier",
                ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"],
                index=0
            )

        segment_records = []
        for s in health_report.segments:
            if selected_tier != "ALL" and s.maintenance_priority.upper() != selected_tier:
                continue

            if s.traffic_exposure and s.traffic_exposure.get("is_measured") and s.traffic_exposure.get("vehicle_count", 0) > 0:
                traffic_str = f"{s.traffic_exposure['vehicle_count']} veh ({s.traffic_exposure['exposure_level']})"
            else:
                traffic_str = "Not Measured"

            sb = s.severity_breakdown
            sev_str = f"L:{sb.get('low', 0)} M:{sb.get('medium', 0)} H:{sb.get('high', 0)}"

            segment_records.append({
                "Segment ID": s.segment_id,
                "Segment Name": s.segment_name,
                "Health Score": f"{s.health_score:.1f}",
                "Priority Tier": s.maintenance_priority,
                "Priority Score": f"{s.priority_score:.1f}",
                "Damages": s.damage_count,
                "Dominant Severity": s.dominant_severity.upper(),
                "Severity (L/M/H)": sev_str,
                "Recurrence": s.recurrence_count,
                "Traffic Exposure": traffic_str,
                "Length": f"{s.length_meters:.0f} m"
            })

        if segment_records:
            df_segments = pd.DataFrame(segment_records)
            st.dataframe(df_segments, use_container_width=True, hide_index=True)
            st.caption(f"Displaying {len(segment_records)} segment records.")
        else:
            st.info(f"No segments match the priority filter '{selected_tier}'.")


# =============================================================================
# TAB 4: GIS SPATIAL MAP (SIMULATED GPS)
# =============================================================================
with tab_gis:
    st.subheader("4. GIS Spatial Map — SIMULATED GPS (Transit Corridor Route-7B)")
    st.markdown("""
    <div class="warning-banner">
        ⚠️ <b>Simulated Geographic Data:</b> Coordinates are interpolated along simulated bus route waypoints in New Delhi for MVP demonstration.
    </div>
    """, unsafe_allow_html=True)

    if data_scope == "CURRENT BUS":
        all_stored_events = db_manager.filter_events(bus_id=current_bus_id, limit=1000)
    else:
        all_stored_events = db_manager.get_events(limit=1000)

    map_col1, map_col2 = st.columns([3, 1])
    with map_col2:
        st.markdown("**Map Display Filters**")
        show_vehicles_on_map = st.checkbox("Overlay Vehicle Detections", value=False)
        sev_filter_map = st.multiselect("Severity Filter", ["high", "medium", "low"], default=["high", "medium", "low"])

        filtered_map_events = [
            e for e in all_stored_events
            if e.event_type.upper() == "ROAD_DAMAGE" and e.severity.lower() in sev_filter_map
        ]
        if show_vehicles_on_map:
            filtered_map_events += [e for e in all_stored_events if e.event_type.upper() == "VEHICLE"]

        st.write(f"Displaying **{len(filtered_map_events)}** event markers on map ({data_scope}).")

    with map_col1:
        urban_map = create_urban_map(
            events=filtered_map_events,
            waypoints=geo_tagger.waypoints,
            include_vehicles=show_vehicles_on_map
        )
        st_folium(urban_map, width="100%", height=500)


# =============================================================================
# TAB 5: EVENT EXPLORER TABLE
# =============================================================================
with tab_table:
    st.subheader("5. Structured Events Table (SQLite Database)")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        f_type = st.selectbox("Filter Event Type", ["ALL", "ROAD_DAMAGE", "VEHICLE"])
    with f_col2:
        f_class = st.selectbox("Filter Class", ["ALL", "pothole", "car", "bus", "truck", "motorcycle", "person"])
    with f_col3:
        f_sev = st.selectbox("Filter Severity", ["ALL", "HIGH", "MEDIUM", "LOW", "NONE"])
    with f_col4:
        bus_filter_options = ["ALL (Fleet View)", "CURRENT BUS"] + get_available_bus_ids() + [UNKNOWN_BUS_LABEL]
        default_idx = 1 if data_scope == "CURRENT BUS" else 0
        f_bus = st.selectbox("Filter Bus ID", bus_filter_options, index=default_idx)

    type_arg = None if f_type == "ALL" else f_type
    class_arg = None if f_class == "ALL" else f_class
    sev_arg = None if f_sev == "ALL" else f_sev

    if f_bus == "ALL (Fleet View)":
        bus_arg = None
    elif f_bus == "CURRENT BUS":
        bus_arg = current_bus_id
    elif f_bus == UNKNOWN_BUS_LABEL:
        bus_arg = "UNKNOWN"
    else:
        bus_arg = f_bus

    events_list = db_manager.filter_events(
        event_type=type_arg,
        class_name=class_arg,
        severity=sev_arg,
        bus_id=bus_arg,
        limit=1000
    )

    if events_list:
        records = []
        for e in events_list:
            bus_tag = e.bus_id if e.bus_id else UNKNOWN_BUS_LABEL
            records.append({
                "Event ID": e.event_id,
                "Bus ID": bus_tag,
                "Type": e.event_type,
                "Class": e.class_name,
                "Confidence": f"{e.confidence * 100:.1f}%",
                "Severity": e.severity.upper(),
                "Latitude": f"{e.latitude:.5f}",
                "Longitude": f"{e.longitude:.5f}",
                "Timestamp": e.timestamp,
                "Detection Mode": e.detection_mode,
                "GPS Mode": e.gps_mode,
                "Frame": e.frame_index,
                "Source ID": e.source_id
            })
        df_all_events = pd.DataFrame(records)
        st.dataframe(df_all_events, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(records)} events from SQLite database.")
    else:
        st.info("No records match the current filter selection.")


# =============================================================================
# TAB 6: DATA EXPORT & AUDIT LOGS
# =============================================================================
with tab_export:
    st.subheader("6. Export Event Data & Analytics Reports")

    exp_c1, exp_c2, exp_c3 = st.columns(3)

    csv_path = db_manager.export_events_csv()
    json_path = db_manager.export_events_json()

    with exp_c1:
        st.markdown("#### 📄 SQLite Events (CSV)")
        with open(csv_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 Download Events CSV",
                data=f.read(),
                file_name="urban_events_sih26124.csv",
                mime="text/csv",
                use_container_width=True
            )

    with exp_c2:
        st.markdown("#### 📋 SQLite Events (JSON)")
        with open(json_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 Download Events JSON",
                data=f.read(),
                file_name="urban_events_sih26124.json",
                mime="application/json",
                use_container_width=True
            )

    with exp_c3:
        st.markdown("#### 📊 Traffic Report (JSON)")
        st.download_button(
            label="📥 Download Traffic Analytics JSON",
            data=json.dumps(traffic_summary, indent=2),
            file_name="traffic_analytics_report_sih26124.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### 🛣️ Road Health & Maintenance Priority Reports")

    all_export_events = db_manager.get_events(limit=100000)
    if all_export_events:
        valid_export_events = [
            e for e in all_export_events
            if e.event_type == "ROAD_DAMAGE" or (e.event_type == "VEHICLE" and e.class_name.lower() in VALID_VEHICLE_CLASSES)
        ]
        export_health_report = road_health_analyzer.analyze_events(valid_export_events)
        rh_json_path = road_health_analyzer.export_report_json(export_health_report)
        rh_csv_path = road_health_analyzer.export_report_csv(export_health_report)

        rh_c1, rh_c2 = st.columns(2)
        with rh_c1:
            with open(rh_json_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Download Road Health Report (JSON)",
                    data=f.read(),
                    file_name="road_health_report_sih26124.json",
                    mime="application/json",
                    use_container_width=True
                )
        with rh_c2:
            with open(rh_csv_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Download Segment Health Matrix (CSV)",
                    data=f.read(),
                    file_name="road_health_segments_sih26124.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("ℹ️ Road Health exports unavailable: No event records exist in the SQLite database. Ingest video data in Tab 1 to generate reports.")

    st.markdown("---")
    st.markdown("#### 📂 Storage System Information")
    st.write(f"• **Database Path**: `{db_manager.db_path}`")
    st.write(f"• **Active Bus**: `{current_bus_id}` ({format_bus_display(current_bus_id)})")
    st.write(f"• **Data Selection Scope**: `{data_scope}`")
    st.write(f"• **Sample Data Directory**: `{settings.SAMPLE_DATA_DIR}`")
    st.write(f"• **Events Data Directory**: `{settings.EVENTS_DATA_DIR}`")
    st.write(f"• **Models Directory**: `{settings.MODELS_DIR}`")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Smart India Hackathon 2026 (SIH26124) | Urban Transit Intelligence & Road Asset Management Platform | Fleet Authentication Active")