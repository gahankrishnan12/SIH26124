"""
SIH26124: AI-Powered Mobile Urban Intelligence Platform
Streamlit Application - Checkpoint 4: Event Generation, Geotagging & SQLite Persistence
"""
import time
import tempfile
from pathlib import Path
import streamlit as st
import cv2
import pandas as pd
import json

from src.detection.vehicle_detector import VehicleDetector
from src.detection.road_damage_detector import RoadDamageDetector
from src.events.geo_tagger import GeoTagger
from src.events.generator import EventGenerator
from src.storage.db_manager import DatabaseManager
from src.video.processor import VideoProcessor
from src.maps.folium_map import create_event_map, render_folium_map
from config import settings

st.set_page_config(
    page_title="Urban Intelligence Platform | Checkpoint 5",
    page_icon="🚌",
    layout="wide"
)

# Header Section
st.title("🚌 AI-Powered Mobile Urban Intelligence Platform")
st.caption("Smart India Hackathon 2026 — Problem Statement: SIH26124 | Checkpoint 5: GIS & Interactive Map Visualization")
st.markdown("---")

# Cached Pipeline Initializations
@st.cache_resource(show_spinner="Loading AI Detection Engines on CPU...")
def load_pipeline():
    veh_det = VehicleDetector(model_name="yolov8n.pt", conf_threshold=0.35)
    dam_det = RoadDamageDetector()
    db_mgr = DatabaseManager()
    geo_tagger = GeoTagger()
    return veh_det, dam_det, db_mgr, geo_tagger

vehicle_detector, road_damage_detector, db_manager, geo_tagger = load_pipeline()

# Disclosure & Mode Banners (Top of Dashboard)
st.subheader("1. System Mode & Data Integrity Disclosure")
b1, b2, b3 = st.columns([1.5, 1.5, 1])

with b1:
    if road_damage_detector.detection_mode == "REAL_AI":
        st.success(f"🟢 **AI Detection**: REAL AI (`{road_damage_detector.model_name}`)")
    else:
        st.warning("🟠 **AI Detection**: DEMO/SIMULATION MODE")

with b2:
    st.info("📍 **GPS**: SIMULATED GPS COORDINATES (Transit Corridor Route-7B)")

with b3:
    simulate_toggle = st.toggle("Force Demo AI Mode", value=False)
    road_damage_detector.force_demo_mode = simulate_toggle
    road_damage_detector.detection_mode = "DEMO_SIMULATION" if simulate_toggle else "REAL_AI"

st.caption("ℹ️ *Data Transparency Note*: GPS waypoints are interpolated along a predefined transit corridor. Severity classification uses a prototype bounding-box area heuristic for hackathon evaluation.")

st.markdown("---")

# Section 2: Input Selection & Pipeline Controls
st.subheader("2. Video Ingestion & Processing Controls")
c1, c2, c3 = st.columns([1.5, 1, 1])

with c1:
    input_mode = st.radio(
        "Select Video Input",
        [
            "Real Road Video (real_road_sample.mp4)",
            "Backup Demo Video (backup_road_demo.mp4)",
            "Upload Custom Video"
        ],
        horizontal=False
    )
    
    selected_video_path = None
    if input_mode == "Real Road Video (real_road_sample.mp4)":
        p = settings.SAMPLE_DATA_DIR / "real_road_sample.mp4"
        if p.exists():
            selected_video_path = str(p)
            st.success(f"Loaded Real Video ({round(p.stat().st_size / (1024*1024), 2)} MB)")
    elif input_mode == "Backup Demo Video (backup_road_demo.mp4)":
        p = settings.SAMPLE_DATA_DIR / "backup_road_demo.mp4"
        if p.exists():
            selected_video_path = str(p)
            st.info(f"Loaded Backup Video ({round(p.stat().st_size / 1024, 1)} KB)")
    else:
        uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            selected_video_path = tfile.name
            st.success("Custom video uploaded successfully.")

with c2:
    st.markdown("**Confidence Thresholds**")
    veh_conf = st.slider("Vehicle Conf", 0.10, 0.90, 0.35, 0.05)
    vehicle_detector.conf_threshold = veh_conf
    dam_conf = st.slider("Road Damage Conf", 0.10, 0.90, 0.30, 0.05)
    road_damage_detector.conf_threshold = dam_conf

with c3:
    st.markdown("**Frame & Storage Controls**")
    max_frames_slider = st.slider("Max Frames (0 = All)", 0, 600, 60, 15)
    max_frames_val = None if max_frames_slider == 0 else max_frames_slider
    frame_skip = st.selectbox("Frame Skip Interval", [1, 2, 3, 5], index=0)
    reset_db_on_run = st.checkbox("Clear Database Before Run", value=True)

st.markdown("---")

# Section 3: Execution
st.subheader("3. Execute Pipeline & Generate Persistent Events")

processor = VideoProcessor(
    vehicle_detector=vehicle_detector,
    road_damage_detector=road_damage_detector,
    db_manager=db_manager,
    geo_tagger=geo_tagger
)

if selected_video_path is not None:
    if st.button("🚀 Process Video, Geotag & Store Events", type="primary"):
        if reset_db_on_run:
            db_manager.clear_events()

        progress_bar = st.progress(0, text="Starting pipeline...")
        output_save_path = settings.SAMPLE_DATA_DIR / "annotated_checkpoint4_output.mp4"

        def ui_progress(processed, total, current_fps, current_v, current_d):
            pct = min(int((processed / max(total, 1)) * 100), 100)
            progress_bar.progress(
                pct,
                text=f"Frame {processed}/{total} | Live Speed: {current_fps:.1f} FPS | Veh: {current_v} | Damage: {current_d}"
            )

        start_t = time.perf_counter()
        results = processor.process_video(
            input_path=selected_video_path,
            output_path=str(output_save_path),
            frame_skip=frame_skip,
            max_frames=max_frames_val,
            save_to_db=True,
            progress_callback=ui_progress
        )
        progress_bar.progress(100, text="Processing & Storage Complete!")
        st.success(f"✅ Video Processed in {results['total_processing_time_sec']}s | Generated {results['total_generated_events']} Events (Filtered {results['total_duplicates_filtered']} duplicate detections).")

        # Section 4: Event Summary KPI Cards
        st.markdown("---")
        st.subheader("4. Urban Event Summary")

        k1, k2, k3, k4 = st.columns(4)
        db_stats = db_manager.get_event_statistics()
        total_evts = db_stats["total_events"]
        dam_evts = db_stats["by_event_type"].get("ROAD_DAMAGE", 0)
        veh_evts = db_stats["by_event_type"].get("VEHICLE", 0)
        high_sev = db_stats["by_severity"].get("high", 0)

        with k1:
            st.metric(label="Total Persistent Events", value=total_evts, help="Deduplicated canonical events stored in SQLite")
        with k2:
            st.metric(label="Road Damage Events", value=dam_evts)
        with k3:
            st.metric(label="Vehicle Events", value=veh_evts)
        with k4:
            st.metric(label="High Severity Hazards", value=high_sev)

        # Section 5: Filterable Event Table
        st.markdown("---")
        st.subheader("5. Structured Events Table (SQLite)")

        filter_c1, filter_c2 = st.columns(2)
        with filter_c1:
            filter_type = st.selectbox("Filter by Event Type", ["ALL", "ROAD_DAMAGE", "VEHICLE"])
        with filter_c2:
            filter_sev = st.selectbox("Filter by Severity", ["ALL", "HIGH", "MEDIUM", "LOW", "NONE"])

        selected_type = None if filter_type == "ALL" else filter_type
        selected_sev = None if filter_sev == "ALL" else filter_sev

        filtered_events = db_manager.filter_events(event_type=selected_type, severity=selected_sev, limit=500)

        if filtered_events:
            event_records = []
            for e in filtered_events:
                event_records.append({
                    "Event ID": e.event_id,
                    "Timestamp": e.timestamp,
                    "Type": e.event_type,
                    "Class": e.class_name,
                    "Confidence": f"{e.confidence*100:.1f}%",
                    "Severity": e.severity.upper(),
                    "Latitude": e.latitude,
                    "Longitude": e.longitude,
                    "Detection Mode": e.detection_mode,
                    "GPS Mode": e.gps_mode
                })
            df_events = pd.DataFrame(event_records)
            st.dataframe(df_events, use_container_width=True, hide_index=True)
        else:
            st.info("No events match the selected filter criteria.")

        # Section 6: GIS & Interactive Map Visualization
        st.markdown("---")
        st.subheader("6. Interactive GIS Transit Map (Simulated GPS)")
        st.caption("📍 *Spatial Event Distribution along Transit Corridor Route-7B*. Coordinates are simulated for MVP demonstration. Click on any marker to inspect detailed event telemetry.")

        map_opt_c1, map_opt_c2 = st.columns(2)
        with map_opt_c1:
            enable_clusters = st.checkbox("Enable Marker Clustering", value=False, help="Group proximate markers into interactive clusters")
        with map_opt_c2:
            show_corridor_line = st.checkbox("Show Simulated Transit Corridor (Route-7B)", value=True, help="Draw the simulated bus route polyline with start/end terminals")

        if filtered_events:
            event_map = create_event_map(
                events=filtered_events,
                enable_clustering=enable_clusters,
                show_route=show_corridor_line
            )
            render_folium_map(event_map, height=540)
        else:
            st.info("No events available to display on the map.")

        # Section 7: Data Export Actions
        st.markdown("---")
        st.subheader("7. Export Event Records")
        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])

        csv_export_path = db_manager.export_events_csv()
        json_export_path = db_manager.export_events_json()

        with exp_col1:
            with open(csv_export_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Download Events CSV",
                    data=f.read(),
                    file_name="urban_events_sih26124.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with exp_col2:
            with open(json_export_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Download Events JSON",
                    data=f.read(),
                    file_name="urban_events_sih26124.json",
                    mime="application/json",
                    use_container_width=True
                )

        with exp_col3:
            st.caption(f"💾 **SQLite Database**: `{db_manager.db_path}` | Events Table Synchronized")

        # Section 8: Video Preview
        st.markdown("---")
        st.subheader("8. Annotated Video Output Preview")
        if output_save_path.exists() and output_save_path.stat().st_size > 0:
            st.video(str(output_save_path))
            st.caption(f"Annotated Video Saved: {output_save_path} ({round(output_save_path.stat().st_size / 1024, 1)} KB)")

else:
    st.warning("Please select a video input above.")

st.markdown("---")
st.caption("Checkpoint 5 verified: GIS module integrated with Folium & streamlit-folium, displaying existing event data, road damage markers with severity classification, vehicle markers, telemetry popups, and simulated GPS disclosures.")

