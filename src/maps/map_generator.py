"""
SIH26124: GIS Map Generator Module (Checkpoint 6)
Creates interactive Folium map visualizations of simulated GPS transit corridors,
road damage hazards, and urban mobility events.
"""
from typing import List, Dict, Any, Optional, Tuple
import folium
from folium.plugins import MarkerCluster

from src.events.schema import UrbanEvent
from config import settings


SEVERITY_COLOR_MAP = {
    "high": "#DC2626",    # Crimson Red
    "medium": "#F59E0B",  # Amber Orange
    "low": "#3B82F6",     # Ocean Blue
    "none": "#6B7280"     # Slate Gray
}

SEVERITY_RADIUS_MAP = {
    "high": 9,
    "medium": 7,
    "low": 5,
    "none": 4
}


def create_urban_map(
    events: Optional[List[UrbanEvent]] = None,
    waypoints: Optional[List[Dict[str, Any]]] = None,
    center_lat: float = settings.DEFAULT_ORIGIN_LAT,
    center_lon: float = settings.DEFAULT_ORIGIN_LON,
    zoom_start: int = 14,
    include_vehicles: bool = False
) -> folium.Map:
    """
    Generate an interactive Folium map visualizing simulated transit corridor
    and geotagged road condition / vehicle events.

    NOTE: All coordinates represent simulated GPS telemetry.
    """
    events = events or []

    # Recalculate center if events are provided and have coordinates
    valid_coords = [(e.latitude, e.longitude) for e in events if e.latitude and e.longitude]
    if valid_coords:
        avg_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
        avg_lon = sum(c[1] for c in valid_coords) / len(valid_coords)
        map_center = [avg_lat, avg_lon]
    elif waypoints:
        map_center = [waypoints[0]["latitude"], waypoints[0]["longitude"]]
    else:
        map_center = [center_lat, center_lon]

    # Initialize Base Map
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # 1. Draw Transit Corridor Waypoints Route (Simulated)
    if waypoints and len(waypoints) > 1:
        route_coords = [[wp["latitude"], wp["longitude"]] for wp in waypoints]
        folium.PolyLine(
            locations=route_coords,
            color="#2563EB",
            weight=4,
            opacity=0.75,
            tooltip="Transit Corridor Route-7B (SIMULATED GPS)",
            dash_array="5, 10"
        ).add_to(m)

        # Route Start & End Markers
        start_wp = waypoints[0]
        end_wp = waypoints[-1]

        folium.Marker(
            location=[start_wp["latitude"], start_wp["longitude"]],
            popup="<b>Transit Route Origin</b><br/>GPS: SIMULATED",
            tooltip="Route Start (Simulated)",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)

        folium.Marker(
            location=[end_wp["latitude"], end_wp["longitude"]],
            popup="<b>Transit Route Terminus</b><br/>GPS: SIMULATED",
            tooltip="Route End (Simulated)",
            icon=folium.Icon(color="darkred", icon="stop")
        ).add_to(m)

    # 2. Add Road Damage & Hazard Event Markers
    damage_events = [e for e in events if e.event_type.upper() == "ROAD_DAMAGE"]
    vehicle_events = [e for e in events if e.event_type.upper() == "VEHICLE"] if include_vehicles else []

    # Road Damage Feature Group
    damage_group = folium.FeatureGroup(name="Road Hazards (Potholes/Damage)").add_to(m)

    for evt in damage_events:
        sev = evt.severity.lower()
        color = SEVERITY_COLOR_MAP.get(sev, "#6B7280")
        radius = SEVERITY_RADIUS_MAP.get(sev, 6)

        popup_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; min-width: 190px;">
            <h4 style="margin: 0 0 6px 0; color: {color};">⚠️ {evt.class_name.upper()}</h4>
            <b>Severity:</b> <span style="color: {color}; font-weight: bold;">{evt.severity.upper()}</span><br/>
            <b>Confidence:</b> {evt.confidence * 100:.1f}%<br/>
            <b>Event ID:</b> <code>{evt.event_id}</code><br/>
            <b>Mode:</b> {evt.detection_mode}<br/>
            <b>GPS:</b> {evt.gps_mode} <i>(Simulated)</i><br/>
            <b>Coordinates:</b> {evt.latitude:.5f}, {evt.longitude:.5f}<br/>
            <b>Frame:</b> {evt.frame_index}<br/>
            <b>Timestamp:</b> {evt.timestamp}
        </div>
        """

        folium.CircleMarker(
            location=[evt.latitude, evt.longitude],
            radius=radius,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Hazard: {evt.class_name} ({evt.severity.upper()}) - Simulated GPS"
        ).add_to(damage_group)

    # 3. Add Vehicle Events (Optional overlay)
    if vehicle_events:
        vehicle_group = folium.FeatureGroup(name="Vehicle Detection Events").add_to(m)
        for evt in vehicle_events:
            popup_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; min-width: 170px;">
                <h4 style="margin: 0 0 6px 0; color: #7C3AED;">🚗 {evt.class_name.upper()}</h4>
                <b>Confidence:</b> {evt.confidence * 100:.1f}%<br/>
                <b>Event ID:</b> <code>{evt.event_id}</code><br/>
                <b>GPS:</b> {evt.gps_mode} <i>(Simulated)</i><br/>
                <b>Coordinates:</b> {evt.latitude:.5f}, {evt.longitude:.5f}
            </div>
            """
            folium.CircleMarker(
                location=[evt.latitude, evt.longitude],
                radius=4,
                color="#7C3AED",
                weight=1,
                fill=True,
                fill_color="#7C3AED",
                fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"Vehicle: {evt.class_name}"
            ).add_to(vehicle_group)

    folium.LayerControl().add_to(m)
    return m
