"""
SIH26124: AI-Powered Mobile Urban Intelligence Platform
Smart City Fleet Operations & Urban Intelligence Command Center
"""
import time
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import cv2
import pandas as pd
from streamlit_folium import st_folium

from config import settings
from config.buses import (
    get_available_bus_ids,
    get_available_buses,
    get_bus_info,
    format_bus_display,
    authenticate_bus,
    UNKNOWN_BUS_LABEL
)
from src.detection.vehicle_detector import VehicleDetector
from src.detection.road_damage_detector import RoadDamageDetector
from src.events.schema import UrbanEvent
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

# -----------------------------------------------------------------------------
# 1. Page Configuration & Professional Operations Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SIH26124 Fleet Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Layout */
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Command Center Top Header */
    .command-header {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 18px 22px;
        color: #f8fafc;
        margin-bottom: 16px;
    }
    .command-title {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.3px;
        color: #ffffff;
        margin: 0;
    }
    .command-subtitle {
        font-size: 12px;
        color: #94a3b8;
        margin: 2px 0 0 0;
    }

    /* Professional Status Chips & Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .badge-active {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
    }
    .badge-scope-bus {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #2563eb;
    }
    .badge-scope-fleet {
        background-color: #78350f;
        color: #fcd34d;
        border: 1px solid #d97706;
    }
    .badge-registered {
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
    }

    /* Status Dot */
    .status-dot-active {
        display: inline-block;
        width: 7px;
        height: 7px;
        background-color: #22c55e;
        border-radius: 50%;
    }
    .status-dot-muted {
        display: inline-block;
        width: 7px;
        height: 7px;
        background-color: #64748b;
        border-radius: 50%;
    }

    /* Metric & KPI Cards */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Fleet Bus Grid Cards */
    .bus-grid-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
        height: 100%;
    }
    .bus-grid-card-active {
        background-color: #f8fafc;
        border: 2px solid #0f172a;
    }
    .bus-grid-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .bus-grid-route {
        font-size: 11px;
        color: #475569;
        margin-bottom: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .bus-grid-stat {
        font-size: 11px;
        color: #334155;
        display: flex;
        justify-content: space-between;
        padding: 2px 0;
        border-top: 1px solid #f1f5f9;
    }

    /* Architecture Flow Banner */
    .flow-banner {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 14px;
        font-size: 11px;
        color: #334155;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 4px;
    }
    .flow-step {
        display: inline-flex;
        align-items: center;
        padding: 2px 7px;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        font-weight: 600;
        color: #0f172a;
    }
    .flow-arrow {
        color: #94a3b8;
        font-weight: bold;
    }

    /* Information & Disclaimer Banners */
    .info-banner {
        background-color: #f0f9ff;
        border-left: 3px solid #0284c7;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-size: 12px;
        color: #0369a1;
        line-height: 1.45;
    }
    .warning-banner {
        background-color: #fffbeb;
        border-left: 3px solid #d97706;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-size: 12px;
        color: #92400e;
        line-height: 1.45;
    }
    .alert-banner {
        background-color: #fef2f2;
        border-left: 3px solid #dc2626;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-size: 12px;
        color: #991b1b;
        line-height: 1.45;
    }

    /* Authentication Box */
    .auth-panel {
        max-width: 440px;
        margin: 40px auto 20px auto;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 28px 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .auth-header {
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
    }

    /* Priority Pill Tags */
    .prio-pill {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 3px;
        font-size: 10.5px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .prio-critical { background-color: #fee2e2; color: #b91c1c; border: 1px solid #f87171; }
    .prio-high { background-color: #ffedd5; color: #c2410c; border: 1px solid #fb923c; }
    .prio-medium { background-color: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }
    .prio-low { background-color: #f0fdf4; color: #15803d; border: 1px solid #86efac; }
    .prio-normal { background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
</style>
""", unsafe_allow_html=True)


def get_or_create_browser_preview(source_video_path: Path) -> Path:
    """
    Creates a browser-compatible H.264/avc1 preview file with faststart
    from an OpenCV-generated annotated MP4 video, preserving the original file.
    """
    if not source_video_path.exists() or source_video_path.stat().st_size == 0:
        return source_video_path

    preview_path = source_video_path.parent / f"browser_preview_{source_video_path.name}"

    if preview_path.exists() and preview_path.stat().st_mtime >= source_video_path.stat().st_mtime:
        return preview_path

    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(source_video_path),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(preview_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and preview_path.exists() and preview_path.stat().st_size > 0:
            return preview_path
    except Exception:
        pass

    return source_video_path


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
# 3. Session State & Authentication Control
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "current_bus_id" not in st.session_state:
    st.session_state["current_bus_id"] = None
if "data_scope" not in st.session_state:
    st.session_state["data_scope"] = "CURRENT BUS"


# =============================================================================
# AUTHENTICATION SCREEN (ZERO-EMOJI PROFESSIONAL DESIGN)
# =============================================================================
if not st.session_state["authenticated"]:
    _, auth_center_col, _ = st.columns([1, 1.8, 1])

    with auth_center_col:
        st.markdown("""
        <div class="auth-panel">
            <div class="auth-header">
                <div style="font-size: 18px; font-weight: 700; color: #0f172a; letter-spacing: -0.2px;">
                    SIH26124 FLEET INTELLIGENCE
                </div>
                <div style="font-size: 12px; color: #64748b; margin-top: 3px;">
                    AI-Powered Mobile Urban Intelligence Platform
                </div>
            </div>
            <div style="font-size: 12px; font-weight: 700; color: #334155; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 0.5px;">
                BUS AUTHENTICATION
            </div>
        """, unsafe_allow_html=True)

        bus_ids = get_available_bus_ids()

        with st.form("fleet_bus_login_form", clear_on_submit=False):
            st.markdown("<label style='font-size: 11.5px; font-weight: 600; color: #475569;'>Bus ID</label>", unsafe_allow_html=True)
            selected_bus_id = st.selectbox(
                "Bus ID",
                options=bus_ids,
                format_func=lambda b_id: format_bus_display(b_id),
                label_visibility="collapsed"
            )

            st.markdown("<label style='font-size: 11.5px; font-weight: 600; color: #475569; margin-top: 8px;'>Access PIN</label>", unsafe_allow_html=True)
            entered_pin = st.text_input(
                "Access PIN",
                type="password",
                placeholder="Enter access PIN",
                label_visibility="collapsed"
            )

            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            login_btn = st.form_submit_button("LOGIN TO FLEET", type="primary", use_container_width=True)

            if login_btn:
                if authenticate_bus(selected_bus_id, entered_pin):
                    st.session_state["authenticated"] = True
                    st.session_state["current_bus_id"] = selected_bus_id
                    st.session_state["data_scope"] = "CURRENT BUS"
                    st.success(f"Authenticated as {selected_bus_id} successfully.")
                    st.rerun()
                else:
                    st.error("Authentication failed: Invalid Bus ID or Access PIN. Please verify credentials.")
    st.stop()


# -----------------------------------------------------------------------------
# 4. Authenticated Context & Data Scope Resolution
# -----------------------------------------------------------------------------
current_bus_id = st.session_state["current_bus_id"]
current_bus_info = get_bus_info(current_bus_id)
data_scope = st.session_state.get("data_scope", "CURRENT BUS")

# Fetch scope-resolved events from DatabaseManager
if data_scope == "CURRENT BUS":
    scoped_events: List[UrbanEvent] = db_manager.filter_events(bus_id=current_bus_id, limit=100000)
    scoped_damages: List[UrbanEvent] = db_manager.filter_events(event_type="ROAD_DAMAGE", bus_id=current_bus_id, limit=100000)
else:
    scoped_events = db_manager.get_events(limit=100000)
    scoped_damages = db_manager.filter_events(event_type="ROAD_DAMAGE", limit=100000)

db_stats = db_manager.get_event_statistics()


# -----------------------------------------------------------------------------
# 5. Sidebar Navigation & Fleet Operational Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div style="font-size: 15px; font-weight: 700; color: #0f172a; line-height: 1.2;">SIH26124</div>
        <div style="font-size: 11px; color: #64748b;">Fleet Intelligence Operations</div>
    </div>
    """, unsafe_allow_html=True)

    # Fleet Context Panel
    st.markdown("<div style='font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.4px;'>FLEET CONTEXT</div>", unsafe_allow_html=True)
    if current_bus_info:
        st.markdown(f"""
        <div class="kpi-card" style="background-color: #f8fafc; padding: 10px 12px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 13px; font-weight: 700; color: #0f172a;">{current_bus_info['bus_id']}</span>
                <span class="status-badge badge-active"><span class="status-dot-active"></span> ACTIVE</span>
            </div>
            <div style="font-size: 11px; color: #475569; margin-top: 2px;">{current_bus_info['display_name']}</div>
            <div style="font-size: 10.5px; color: #64748b; margin-top: 2px;">Route: {current_bus_info['route']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write(f"Current Bus: `{current_bus_id}`")

    # Data Selection Scope Switcher
    st.markdown("<div style='font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-top: 8px; margin-bottom: 4px;'>DATA SCOPE</div>", unsafe_allow_html=True)
    scope_option = st.radio(
        "Data Scope",
        options=["CURRENT BUS", "FLEET VIEW"],
        index=0 if data_scope == "CURRENT BUS" else 1,
        help="CURRENT BUS: Limits analytics to the authenticated vehicle. FLEET VIEW: Aggregates records across the entire fleet.",
        label_visibility="collapsed"
    )
    if scope_option != st.session_state["data_scope"]:
        st.session_state["data_scope"] = scope_option
        st.rerun()

    if st.button("Logout Session", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["current_bus_id"] = None
        st.session_state["data_scope"] = "CURRENT BUS"
        st.rerun()

    st.markdown("---")

    # System Status
    st.markdown("<div style='font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.4px;'>SYSTEM STATUS</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size: 11.5px; color: #334155; line-height: 1.8;">
        <div style="display: flex; justify-content: space-between;">
            <span>Vehicle AI:</span> <span style="font-weight: 600; color: #16a34a;">ACTIVE ({vehicle_detector.model_name})</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Damage AI:</span> <span style="font-weight: 600; color: #16a34a;">{road_damage_detector.detection_mode}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>GPS:</span> <span style="font-weight: 600; color: #0284c7;">SIMULATED</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>SQLite:</span> <span style="font-weight: 600; color: #0f172a;">CONNECTED ({db_stats['total_events']} events)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode toggle for Road Damage
    simulate_toggle = st.toggle("Force Demo Damage AI", value=False, help="Forces heuristic road damage simulation mode for testing.")
    road_damage_detector.force_demo_mode = simulate_toggle
    road_damage_detector.detection_mode = "DEMO_SIMULATION" if simulate_toggle else "REAL_AI"

    st.markdown("---")

    # Technical System Controls Expander
    with st.expander("System Controls & Confidence", expanded=False):
        st.markdown("**Inference Confidence Thresholds**")
        veh_conf = st.slider("Vehicle AI Confidence", 0.10, 0.90, float(vehicle_detector.conf_threshold), 0.05)
        vehicle_detector.conf_threshold = veh_conf

        dam_conf = st.slider("Damage AI Confidence", 0.10, 0.90, float(road_damage_detector.conf_threshold), 0.05)
        road_damage_detector.conf_threshold = dam_conf

        st.markdown("---")
        st.markdown("**Database Clearance**")
        st.caption("Destructive operation. Requires explicit confirmation.")
        confirm_clear = st.checkbox("I understand this removes stored event records", value=False)

        if st.button("Clear Database Records", disabled=not confirm_clear, use_container_width=True):
            db_manager.clear_events()
            st.success("Database records cleared successfully.")
            st.rerun()


# -----------------------------------------------------------------------------
# 6. Application Shell Header
# -----------------------------------------------------------------------------
route_label = current_bus_info['route'] if current_bus_info else "Simulated Route-7B Corridor"
scope_badge = f'<span class="status-badge badge-scope-bus">SCOPE: CURRENT BUS ({current_bus_id})</span>' if data_scope == "CURRENT BUS" else '<span class="status-badge badge-scope-fleet">SCOPE: FLEET VIEW (ALL BUSES)</span>'

st.markdown(f"""
<div class="command-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 class="command-title">SIH26124 FLEET INTELLIGENCE</h1>
            <div class="command-subtitle">AI-Powered Mobile Urban Intelligence Platform</div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            <span class="status-badge badge-active"><span class="status-dot-active"></span> {current_bus_id} &nbsp;|&nbsp; ACTIVE SESSION</span>
            {scope_badge}
        </div>
    </div>
    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #1e293b; font-size: 11.5px; color: #94a3b8; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <div>Route: {route_label}</div>
        <div>Telemetry: SIMULATED GPS (Route-7B) &nbsp;|&nbsp; AI Perception: YOLOv8 Dual Detection</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Navigation Structure (Clean Professional Text Tabs)
# -----------------------------------------------------------------------------
tab_overview, tab_video, tab_traffic, tab_damage, tab_gis, tab_table, tab_export = st.tabs([
    "Overview",
    "Video & AI",
    "Traffic Analytics",
    "Road Health",
    "GIS Map",
    "Event Explorer",
    "Reports"
])


# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    # 1. Operational Flow Banner
    st.markdown("""
    <div class="flow-banner">
        <span class="flow-step">BUS FLEET</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">MOBILE SENSING</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">AI PERCEPTION</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">STRUCTURED EVENTS</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">GIS TELEMETRY</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">TRAFFIC & ROAD HEALTH</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">PRIORITY ACTION</span>
    </div>
    """, unsafe_allow_html=True)

    # 2. Executive KPI Summary Cards
    traffic_metrics_scoped = compute_traffic_metrics(events=scoped_events)
    total_scoped_events = len(scoped_events)
    total_scoped_damages = len(scoped_damages)
    total_scoped_vehicles = traffic_metrics_scoped["vehicle_counts"]["total_vehicle_count"]
    high_critical_count = sum(1 for e in scoped_damages if e.severity.lower() == "high")

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">TOTAL EVENTS</div>
            <div class="kpi-value">{total_scoped_events}</div>
            <div class="kpi-sub">{data_scope} records stored in SQLite</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ROAD DAMAGE</div>
            <div class="kpi-value" style="color: #c2410c;">{total_scoped_damages}</div>
            <div class="kpi-sub">Potholes & surface distress detected</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">VALID VEHICLES</div>
            <div class="kpi-value" style="color: #1d4ed8;">{total_scoped_vehicles}</div>
            <div class="kpi-sub">Cars, buses, trucks, motorcycles</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        sev_color = "#b91c1c" if high_critical_count > 0 else "#15803d"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">HIGH / CRITICAL ISSUES</div>
            <div class="kpi-value" style="color: {sev_color};">{high_critical_count}</div>
            <div class="kpi-sub">Urgent maintenance priority alerts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Fleet Status Grid (All 5 Configured Buses)
    st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.4px;'>FLEET STATUS</div>", unsafe_allow_html=True)
    st.caption("Active monitoring units across urban transit corridors. The active session unit is highlighted.")

    fleet_cols = st.columns(5)
    all_buses = get_available_buses()
    by_bus_stats = db_stats.get("by_bus_id", {})

    for idx, bus in enumerate(all_buses):
        b_id = bus["bus_id"]
        is_active = (b_id == current_bus_id)
        bus_events_count = by_bus_stats.get(b_id, 0)
        bus_dam_count = sum(1 for e in db_manager.filter_events(event_type="ROAD_DAMAGE", bus_id=b_id, limit=10000))

        card_class = "bus-grid-card bus-grid-card-active" if is_active else "bus-grid-card"
        badge_html = '<span class="status-badge badge-active"><span class="status-dot-active"></span> ACTIVE SESSION</span>' if is_active else '<span class="status-badge badge-registered"><span class="status-dot-muted"></span> REGISTERED</span>'

        with fleet_cols[idx]:
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div class="bus-grid-title">{b_id}</div>
                    {badge_html}
                </div>
                <div class="bus-grid-route" title="{bus['route']}">{bus['route']}</div>
                <div class="bus-grid-stat">
                    <span>Events:</span>
                    <b>{bus_events_count}</b>
                </div>
                <div class="bus-grid-stat">
                    <span>Road Damage:</span>
                    <b>{bus_dam_count}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Intelligence Summary Split: Road Health vs Traffic
    ov_c1, ov_c2 = st.columns(2)

    with ov_c1:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.4px;'>ROAD INFRASTRUCTURE CONDITION</div>", unsafe_allow_html=True)
        if not scoped_events:
            st.info(f"NO EVENTS AVAILABLE: No event records are currently recorded for {data_scope}.")
        else:
            valid_health_events = [
                e for e in scoped_events
                if e.event_type == "ROAD_DAMAGE" or (e.event_type == "VEHICLE" and e.class_name.lower() in VALID_VEHICLE_CLASSES)
            ]
            health_report = road_health_analyzer.analyze_events(valid_health_events)

            rh_k1, rh_k2, rh_k3 = st.columns(3)
            with rh_k1:
                st.metric("Network Health", f"{health_report.overall_network_health:.1f} / 100")
            with rh_k2:
                st.metric("Critical Segments", health_report.critical_segments_count)
            with rh_k3:
                st.metric("High Priority Segments", health_report.high_priority_segments_count)

            urgent_segs = [s for s in health_report.segments if s.maintenance_priority in ("CRITICAL", "HIGH")]
            if urgent_segs:
                st.markdown("**Top Maintenance Alerts:**")
                for s in urgent_segs[:3]:
                    st.markdown(f"- **{s.segment_name}** — Priority: `{s.maintenance_priority}` (Score: `{s.priority_score:.1f}`, Damages: `{s.damage_count}`)")
            else:
                st.success("No critical road segment degradation identified in current scope.")

    with ov_c2:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.4px;'>MOBILITY & TRAFFIC FLOW</div>", unsafe_allow_html=True)
        t_density = traffic_metrics_scoped["traffic_density_classification"]
        t_counts = traffic_metrics_scoped["vehicle_counts"]

        tr_k1, tr_k2, tr_k3 = st.columns(3)
        with tr_k1:
            st.metric("Traffic Density", t_density["density_level"])
        with tr_k2:
            st.metric("Observed Rate", f"{t_density['observed_rate_per_minute']} ev/min")
        with tr_k3:
            dom_class = t_counts["dominant_vehicle_class"] or "None"
            st.metric("Dominant Class", dom_class.upper())

        if t_counts["by_class"]:
            df_mini_class = pd.DataFrame([
                {"Class": k.capitalize(), "Events": v, "Share": f"{t_counts['class_percentage_shares'].get(k, 0):.1f}%"}
                for k, v in t_counts["by_class"].items()
            ])
            st.dataframe(df_mini_class, use_container_width=True, hide_index=True)
        else:
            st.caption("NO VALID VEHICLE OBSERVATIONS: No valid vehicle observations are available for the selected scope.")

    st.markdown("---")

    # 5. Recent Events Stream
    st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.4px;'>RECENT STRUCTURED EVENTS</div>", unsafe_allow_html=True)
    if scoped_events:
        recent_records = []
        for e in scoped_events[:6]:
            recent_records.append({
                "Event ID": e.event_id,
                "Bus ID": e.bus_id if e.bus_id else UNKNOWN_BUS_LABEL,
                "Type": e.event_type,
                "Class": e.class_name,
                "Severity": e.severity.upper(),
                "Confidence": f"{e.confidence * 100:.1f}%",
                "Timestamp": e.timestamp
            })
        st.dataframe(pd.DataFrame(recent_records), use_container_width=True, hide_index=True)
    else:
        st.info("NO EVENTS AVAILABLE: No recent event records available.")


# =============================================================================
# TAB 2: VIDEO & AI INGESTION
# =============================================================================
with tab_video:
    st.markdown("<div style='font-size: 15px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 2px; letter-spacing: 0.4px;'>VIDEO INGESTION & AI PROCESSING</div>", unsafe_allow_html=True)
    st.caption(f"Newly processed events will be tagged with active bus ID: **{current_bus_id}**")

    col_v1, col_v2 = st.columns([1.5, 1])

    with col_v1:
        input_mode = st.radio(
            "Select Video Feed Source",
            [
                "Judge Demo Video (judge_demo_1.mp4)",
                "Real Road Video (real_road_sample.mp4)",
                "Backup Demo Video (backup_road_demo.mp4)",
                "Upload Custom Video (.mp4, .avi, .mov)"
            ],
            horizontal=False
        )

        selected_video_path = None
        if input_mode == "Judge Demo Video (judge_demo_1.mp4)":
            p = settings.SAMPLE_DATA_DIR / "judge_demo_1.mp4"
            if p.exists():
                selected_video_path = str(p)
                st.success(f"Loaded Judge Demo Video ({round(p.stat().st_size / (1024*1024), 2)} MB)")
            else:
                st.error(f"Video file not found: {p}")
        elif input_mode == "Real Road Video (real_road_sample.mp4)":
            p = settings.SAMPLE_DATA_DIR / "real_road_sample.mp4"
            if p.exists():
                selected_video_path = str(p)
                st.success(f"Loaded Real Video Feed ({round(p.stat().st_size / (1024*1024), 2)} MB)")
            else:
                st.error(f"Video file not found: {p}")
        elif input_mode == "Backup Demo Video (backup_road_demo.mp4)":
            p = settings.SAMPLE_DATA_DIR / "backup_road_demo.mp4"
            if p.exists():
                selected_video_path = str(p)
                st.info(f"Loaded Backup Video Feed ({round(p.stat().st_size / 1024, 1)} KB)")
            else:
                st.error(f"Backup video file not found: {p}")
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
        if input_mode == "Judge Demo Video (judge_demo_1.mp4)":
            output_save_path = settings.SAMPLE_DATA_DIR / "annotated_judge_demo_1.mp4"
        else:
            output_save_path = settings.SAMPLE_DATA_DIR / "annotated_real_road_sample.mp4"

        if st.button("PROCESS VIDEO", type="primary", use_container_width=True):
            if reset_db_on_run:
                db_manager.clear_events()

            processor = VideoProcessor(
                vehicle_detector=vehicle_detector,
                road_damage_detector=road_damage_detector,
                db_manager=db_manager,
                geo_tagger=geo_tagger
            )

            progress_bar = st.progress(0, text="Initializing processing pipeline...")

            def ui_progress(processed, total, current_fps, current_v, current_d):
                pct = min(int((processed / max(total, 1)) * 100), 100)
                progress_bar.progress(
                    pct,
                    text=f"Frame {processed}/{total} ({pct}%) | Speed: {current_fps:.1f} FPS | Vehicles: {current_v} | Hazards: {current_d}"
                )

            start_proc_time = time.perf_counter()
            try:
                results = processor.process_video(
                    input_path=selected_video_path,
                    output_path=str(output_save_path),
                    frame_skip=frame_skip,
                    max_frames=max_frames_val,
                    save_to_db=True,
                    bus_id=current_bus_id,
                    progress_callback=ui_progress
                )
                progress_bar.progress(100, text="Pipeline Execution Complete")

                st.success(
                    f"Processed {results['processed_frames_count']} frames in {results['total_processing_time_sec']}s "
                    f"({results['complete_pipeline_fps']} FPS). Generated {results['total_generated_events']} deduplicated events "
                    f"tagged to bus {current_bus_id}."
                )

                # Performance Metrics Cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Frames", results["processed_frames_count"])
                with m2:
                    st.metric("Pipeline Speed", f"{results['complete_pipeline_fps']} FPS")
                with m3:
                    st.metric("Inference Speed", f"{results['model_inference_fps']} FPS")
                with m4:
                    st.metric("Generated Events", results["total_generated_events"], delta=f"-{results['total_duplicates_filtered']} dupes")

            except Exception as e:
                st.error(f"Video processing error: {e}")

        # Video Preview
        if output_save_path.exists() and output_save_path.stat().st_size > 0:
            st.markdown("#### Annotated AI Output")
            preview_file = get_or_create_browser_preview(output_save_path)
            try:
                st.video(str(preview_file))
                st.caption(f"Saved annotated file: `{output_save_path}` ({round(output_save_path.stat().st_size / (1024*1024), 2)} MB) | Browser preview: `{preview_file.name}`")
            except Exception as vid_err:
                st.error(f"Error displaying video preview: {vid_err}")


# =============================================================================
# TAB 3: TRAFFIC ANALYTICS
# =============================================================================
with tab_traffic:
    st.markdown("<div style='font-size: 15px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 2px; letter-spacing: 0.4px;'>TRAFFIC ANALYTICS</div>", unsafe_allow_html=True)
    if data_scope == "CURRENT BUS":
        st.caption(f"Displaying traffic analytics filtered to **Current Bus ({current_bus_id})**")
    else:
        st.caption("Displaying aggregate fleet-wide traffic analytics (**Fleet View**)")

    traffic_summary = compute_traffic_metrics(events=scoped_events)
    veh_counts = traffic_summary["vehicle_counts"]
    temp_dist = traffic_summary["temporal_distribution"]
    density_info = traffic_summary["traffic_density_classification"]

    # Top KPI Row
    tk1, tk2, tk3, tk4 = st.columns(4)
    with tk1:
        st.metric("Valid Vehicle Events", veh_counts["total_vehicle_count"], help="Valid vehicle classes: car, bus, truck, motorcycle")
    with tk2:
        st.metric("Dominant Class", (veh_counts["dominant_vehicle_class"] or "None").upper())
    with tk3:
        st.metric("Traffic Density", density_info["density_level"], help="Heuristic density classification based on observed rate")
    with tk4:
        st.metric("Observed Rate", f"{density_info['observed_rate_per_minute']} ev/min")

    st.markdown("---")

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 6px;'>VEHICLE CLASS DISTRIBUTION</div>", unsafe_allow_html=True)
        class_counts = veh_counts["by_class"]
        if class_counts:
            df_classes = pd.DataFrame([
                {"Vehicle Class": k.capitalize(), "Event Count": v, "Share (%)": f"{veh_counts['class_percentage_shares'].get(k, 0.0):.1f}%"}
                for k, v in class_counts.items()
            ])
            st.dataframe(df_classes, use_container_width=True, hide_index=True)
            st.bar_chart(pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"]).set_index("Class"))
        else:
            st.info("NO VALID VEHICLE OBSERVATIONS: No valid vehicle observations are available for this scope.")

    with tc2:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 6px;'>TEMPORAL DISTRIBUTION</div>", unsafe_allow_html=True)
        st.write(f"• Observation Duration: `{temp_dist['duration_seconds']} seconds`")
        st.write(f"• Rate per Hour: `{temp_dist['events_per_hour']} events/hr`")
        st.write(f"• Peak Interval Count: `{temp_dist['peak_count']} events`")

        if temp_dist["time_buckets"]:
            df_time = pd.DataFrame([
                {"Time Interval": b["timestamp"].split("T")[-1][:8], "Count": b["count"]}
                for b in temp_dist["time_buckets"]
            ])
            st.line_chart(df_time.set_index("Time Interval"))
        else:
            st.caption("Temporal series requires positive duration vehicle records.")


# =============================================================================
# TAB 4: ROAD HEALTH
# =============================================================================
with tab_damage:
    st.markdown("<div style='font-size: 15px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 2px; letter-spacing: 0.4px;'>ROAD NETWORK HEALTH & MAINTENANCE PRIORITY</div>", unsafe_allow_html=True)
    if data_scope == "CURRENT BUS":
        st.caption(f"Displaying road health analytics for **Current Bus ({current_bus_id})**")
    else:
        st.caption("Displaying aggregate fleet-wide road health analytics (**Fleet View**)")

    # Prototype Disclaimer Banner
    st.markdown(f"""
    <div class="info-banner">
        <b>Decision-Support Notice:</b> {PROTOTYPE_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)

    if not scoped_events:
        st.info("NO ROAD HEALTH ASSESSMENT: Road health cannot be calculated until road-damage events are available.")
    else:
        valid_health_events = [
            e for e in scoped_events
            if e.event_type == "ROAD_DAMAGE" or (e.event_type == "VEHICLE" and e.class_name.lower() in VALID_VEHICLE_CLASSES)
        ]

        health_report = road_health_analyzer.analyze_events(valid_health_events)

        # 1. Overall Network KPI Row
        rh1, rh2, rh3, rh4, rh5 = st.columns(5)
        with rh1:
            st.metric("Network Health", f"{health_report.overall_network_health:.1f} / 100")
        with rh2:
            st.metric("Segments", health_report.total_segments)
        with rh3:
            st.metric("Critical", health_report.critical_segments_count)
        with rh4:
            st.metric("High Priority", health_report.high_priority_segments_count)
        with rh5:
            st.metric("Total Damage", health_report.total_damage_events)

        # 2. Priority Distribution Overview
        st.markdown("<div style='font-size: 12px; font-weight: 700; color: #334155; text-transform: uppercase; margin-top: 10px; margin-bottom: 6px;'>MAINTENANCE PRIORITY DISTRIBUTION</div>", unsafe_allow_html=True)
        dist_c1, dist_c2, dist_c3, dist_c4, dist_c5 = st.columns(5)
        with dist_c1:
            st.markdown(f"<span class='prio-pill prio-critical'>CRITICAL: {health_report.critical_segments_count}</span>", unsafe_allow_html=True)
        with dist_c2:
            st.markdown(f"<span class='prio-pill prio-high'>HIGH: {health_report.high_priority_segments_count}</span>", unsafe_allow_html=True)
        with dist_c3:
            st.markdown(f"<span class='prio-pill prio-medium'>MEDIUM: {health_report.medium_priority_segments_count}</span>", unsafe_allow_html=True)
        with dist_c4:
            st.markdown(f"<span class='prio-pill prio-low'>LOW: {health_report.low_priority_segments_count}</span>", unsafe_allow_html=True)
        with dist_c5:
            st.markdown(f"<span class='prio-pill prio-normal'>NORMAL: {health_report.normal_segments_count}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # 3. Segment Priority Ranking Table
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 6px;'>SEGMENT MAINTENANCE PRIORITIES</div>", unsafe_allow_html=True)

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
                "Segment": s.segment_name,
                "Health": f"{s.health_score:.1f}",
                "Priority": s.maintenance_priority,
                "Priority Score": f"{s.priority_score:.1f}",
                "Damage": s.damage_count,
                "Dominant Severity": s.dominant_severity.upper(),
                "Severity (L/M/H)": sev_str,
                "Recurrence": s.recurrence_count,
                "Traffic Exposure": traffic_str,
                "Length": f"{s.length_meters:.0f} m"
            })

        if segment_records:
            df_segments = pd.DataFrame(segment_records)
            st.dataframe(df_segments, use_container_width=True, hide_index=True)
            st.caption(f"Displaying {len(segment_records)} segment records in current view.")
        else:
            st.info(f"No segments match the priority filter '{selected_tier}'.")


# =============================================================================
# TAB 5: GIS SPATIAL MAP
# =============================================================================
with tab_gis:
    st.markdown("<div style='font-size: 15px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 2px; letter-spacing: 0.4px;'>GIS / SPATIAL INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-banner">
        <b>Data Status: SIMULATED GPS</b> — Coordinates are interpolated along simulated bus route waypoints in New Delhi for MVP demonstration.
    </div>
    """, unsafe_allow_html=True)

    map_col1, map_col2 = st.columns([3, 1])

    with map_col2:
        st.markdown("**Map Display Controls**")
        show_vehicles_on_map = st.checkbox("Overlay Vehicle Detections", value=False)
        sev_filter_map = st.multiselect("Severity Filter", ["high", "medium", "low"], default=["high", "medium", "low"])

        filtered_map_events = [
            e for e in scoped_events
            if e.event_type.upper() == "ROAD_DAMAGE" and e.severity.lower() in sev_filter_map
        ]
        if show_vehicles_on_map:
            filtered_map_events += [e for e in scoped_events if e.event_type.upper() == "VEHICLE"]

        st.write(f"Displaying **{len(filtered_map_events)}** event markers on map ({data_scope}).")

    with map_col1:
        gis_map = create_event_map(
            events=filtered_map_events,
            route_waypoints=geo_tagger.waypoints,
            show_route=True
        )
        render_folium_map(gis_map, height=520, key="fleet_command_gis_map")


# =============================================================================
# TAB 6: EVENT EXPLORER
# =============================================================================
with tab_table:
    st.markdown("<div style='font-size: 15px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 2px; letter-spacing: 0.4px;'>EVENT EXPLORER</div>", unsafe_allow_html=True)

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
        st.caption(f"Displaying {len(records)} structured events from SQLite storage.")
    else:
        st.info("NO EVENTS AVAILABLE: No records match the current filter selection.")


# =============================================================================
# TAB 7: REPORTS & EXPORTS
# =============================================================================
with tab_export:
    st.markdown("<div style='font-size: 15px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 2px; letter-spacing: 0.4px;'>REPORTS & EXPORTS</div>", unsafe_allow_html=True)

    exp_c1, exp_c2, exp_c3 = st.columns(3)

    csv_path = db_manager.export_events_csv()
    json_path = db_manager.export_events_json()

    with exp_c1:
        st.markdown("#### EVENT DATA (CSV)")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="Download Events CSV",
                    data=f.read(),
                    file_name="urban_events_sih26124.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"CSV export error: {e}")

    with exp_c2:
        st.markdown("#### EVENT DATA (JSON)")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="Download Events JSON",
                    data=f.read(),
                    file_name="urban_events_sih26124.json",
                    mime="application/json",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"JSON export error: {e}")

    with exp_c3:
        st.markdown("#### TRAFFIC REPORT (JSON)")
        st.download_button(
            label="Download Traffic Report JSON",
            data=json.dumps(traffic_summary, indent=2),
            file_name="traffic_analytics_report_sih26124.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 6px;'>ROAD HEALTH & ASSET MANAGEMENT REPORTS</div>", unsafe_allow_html=True)

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
            try:
                with open(rh_json_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="Download Road Health Report (JSON)",
                        data=f.read(),
                        file_name="road_health_report_sih26124.json",
                        mime="application/json",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Road Health JSON export error: {e}")

        with rh_c2:
            try:
                with open(rh_csv_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="Download Segment Health Matrix (CSV)",
                        data=f.read(),
                        file_name="road_health_segments_sih26124.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Road Health CSV export error: {e}")
    else:
        st.info("NO ROAD HEALTH ASSESSMENT: Road Health exports unavailable. No event records exist in the SQLite database.")

    st.markdown("---")
    st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; margin-bottom: 6px;'>STORAGE & METADATA</div>", unsafe_allow_html=True)
    st.write(f"• Database Path: `{db_manager.db_path}`")
    st.write(f"• Active Bus: `{current_bus_id}` ({format_bus_display(current_bus_id)})")
    st.write(f"• Data Scope: `{data_scope}`")
    st.write(f"• Sample Data Directory: `{settings.SAMPLE_DATA_DIR}`")
    st.write(f"• Events Data Directory: `{settings.EVENTS_DATA_DIR}`")
    st.write(f"• Models Directory: `{settings.MODELS_DIR}`")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Smart India Hackathon 2026 (SIH26124) | Urban Transit Intelligence & Road Asset Management Platform | Operations Command Center")
