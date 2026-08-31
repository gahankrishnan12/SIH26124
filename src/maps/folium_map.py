"""
SIH26124: GIS & Interactive Map Visualization Module
Provides reusable Folium map generation functions to plot urban mobility and road damage events.

IMPORTANT DATA INTEGRITY DISCLOSURE:
All GPS coordinates rendered by this module are SIMULATED along predefined transit corridors (e.g. Route-7B).
Every marker, popup, tooltip, and map overlay explicitly indicates that GPS coordinates are simulated.
"""
from typing import List, Dict, Any, Optional, Sequence, Union
import json
import folium
from folium.plugins import MarkerCluster
from branca.element import Element

from src.events.schema import UrbanEvent
from src.events.geo_tagger import GeoTagger
from config import settings


def get_marker_styling(event: Union[UrbanEvent, Dict[str, Any]]) -> Dict[str, str]:
    """
    Determine Folium marker color, icon name, and icon library prefix based on event type,
    class name, and damage severity.

    Args:
        event: UrbanEvent instance or dictionary representation.

    Returns:
        Dict with keys: 'color', 'icon', 'prefix', 'icon_color'
    """
    if isinstance(event, dict):
        event_type = str(event.get("event_type", "UNKNOWN")).upper()
        class_name = str(event.get("class_name", "unknown")).lower()
        severity = str(event.get("severity", "none")).lower()
    else:
        event_type = event.event_type.upper()
        class_name = event.class_name.lower()
        severity = event.severity.lower()

    # 1. Road Damage Markers
    if event_type == "ROAD_DAMAGE":
        if severity == "high":
            return {
                "color": "red",
                "icon": "exclamation-triangle",
                "prefix": "fa",
                "icon_color": "white"
            }
        elif severity == "medium":
            return {
                "color": "orange",
                "icon": "warning",
                "prefix": "fa",
                "icon_color": "white"
            }
        elif severity == "low":
            return {
                "color": "beige",
                "icon": "info-circle",
                "prefix": "fa",
                "icon_color": "black"
            }
        else:
            return {
                "color": "lightred",
                "icon": "road",
                "prefix": "fa",
                "icon_color": "white"
            }

    # 2. Vehicle Detection Markers
    elif event_type == "VEHICLE":
        if class_name in ("bus", "truck"):
            return {
                "color": "purple",
                "icon": "bus" if class_name == "bus" else "truck",
                "prefix": "fa",
                "icon_color": "white"
            }
        elif class_name in ("motorcycle", "bicycle"):
            return {
                "color": "cadetblue",
                "icon": "motorcycle",
                "prefix": "fa",
                "icon_color": "white"
            }
        elif class_name == "person":
            return {
                "color": "green",
                "icon": "user",
                "prefix": "fa",
                "icon_color": "white"
            }
        else:
            return {
                "color": "blue",
                "icon": "car",
                "prefix": "fa",
                "icon_color": "white"
            }

    # 3. Default fallback
    return {
        "color": "gray",
        "icon": "map-marker",
        "prefix": "fa",
        "icon_color": "white"
    }


def create_event_popup_html(event: Union[UrbanEvent, Dict[str, Any]]) -> str:
    """
    Generate rich HTML popup markup displaying all 9 required event metadata fields:
    1. Event marker context
    2. Road-damage / Vehicle marker type & class
    3. Severity information
    4. Latitude / Longitude coordinates
    5. Event type (ROAD_DAMAGE / VEHICLE)
    6. Model detection confidence
    7. ISO 8601 Timestamp
    8. Detection mode (REAL_AI / DEMO_SIMULATION)
    9. GPS mode (Explicitly SIMULATED)

    Args:
        event: UrbanEvent object or dict.

    Returns:
        HTML string suitable for folium.Popup.
    """
    if isinstance(event, dict):
        evt_id = str(event.get("event_id", "N/A"))
        evt_type = str(event.get("event_type", "UNKNOWN")).upper()
        cname = str(event.get("class_name", "unknown")).upper()
        conf = float(event.get("confidence", 0.0))
        sev = str(event.get("severity", "none")).upper()
        lat = float(event.get("latitude", 0.0))
        lon = float(event.get("longitude", 0.0))
        ts = str(event.get("timestamp", "N/A"))
        det_mode = str(event.get("detection_mode", "REAL_AI"))
        gps_mode = str(event.get("gps_mode", "SIMULATED"))
        raw_bus_id = event.get("bus_id")
        bus_label = str(raw_bus_id) if raw_bus_id else "UNKNOWN / LEGACY SOURCE"
    else:
        evt_id = event.event_id
        evt_type = event.event_type.upper()
        cname = event.class_name.upper()
        conf = float(event.confidence)
        sev = event.severity.upper()
        lat = float(event.latitude)
        lon = float(event.longitude)
        ts = event.timestamp
        det_mode = event.detection_mode
        gps_mode = event.gps_mode
        bus_label = str(event.bus_id) if event.bus_id else "UNKNOWN / LEGACY SOURCE"

    # Severity badge styling
    sev_bg = "#6c757d"
    if sev == "HIGH":
        sev_bg = "#dc3545"
    elif sev == "MEDIUM":
        sev_bg = "#fd7e14"
    elif sev == "LOW":
        sev_bg = "#ffc107; color: #212529"
    elif sev in ("NONE", "N/A"):
        sev_bg = "#0d6efd"

    # Detection mode badge styling
    det_mode_bg = "#198754" if det_mode == "REAL_AI" else "#fd7e14"

    conf_pct = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"

    html = f"""
    <div style="font-family: Arial, sans-serif; font-size: 12px; width: 270px; line-height: 1.4; color: #333;">
        <div style="background-color: #212529; color: #fff; padding: 6px 10px; border-radius: 4px 4px 0 0; font-weight: bold;">
            📍 {evt_type}: {cname}
        </div>
        <div style="border: 1px solid #dee2e6; border-top: none; padding: 8px 10px; border-radius: 0 0 4px 4px; background: #fafafa;">
            <div style="margin-bottom: 6px;">
                <strong>Event ID:</strong> <code style="background: #e9ecef; padding: 1px 4px; border-radius: 3px; font-size: 11px;">{evt_id}</code>
            </div>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 6px; font-size: 11px;">
                <tr>
                    <td style="padding: 2px 0; color: #666;"><strong>Event Type:</strong></td>
                    <td style="padding: 2px 0; text-align: right;">{evt_type}</td>
                </tr>
                <tr>
                    <td style="padding: 2px 0; color: #666;"><strong>Class:</strong></td>
                    <td style="padding: 2px 0; text-align: right;">{cname}</td>
                </tr>
                <tr>
                    <td style="padding: 2px 0; color: #666;"><strong>Severity:</strong></td>
                    <td style="padding: 2px 0; text-align: right;">
                        <span style="background: {sev_bg}; color: white; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: bold;">
                            {sev}
                        </span>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 2px 0; color: #666;"><strong>Confidence:</strong></td>
                    <td style="padding: 2px 0; text-align: right; font-weight: bold; color: #0d6efd;">{conf_pct}</td>
                </tr>
                <tr>
                    <td style="padding: 2px 0; color: #666;"><strong>Bus Identity:</strong></td>
                    <td style="padding: 2px 0; text-align: right;"><code style="background: #e9ecef; padding: 1px 4px; border-radius: 3px; font-size: 10px;">{bus_label}</code></td>
                </tr>
                <tr>
                    <td style="padding: 2px 0; color: #666;"><strong>Coordinates:</strong></td>
                    <td style="padding: 2px 0; text-align: right; font-family: monospace; font-size: 10px;">{lat:.6f}, {lon:.6f}</td>
                </tr>
                <tr>
                    <td style="padding: 2px 0; color: #666;"><strong>Timestamp:</strong></td>
                    <td style="padding: 2px 0; text-align: right; font-size: 10px;">{ts}</td>
                </tr>
                <tr>
                    <td style="padding: 2px 0; color: #666;"><strong>Detection Mode:</strong></td>
                    <td style="padding: 2px 0; text-align: right;">
                        <span style="background: {det_mode_bg}; color: white; padding: 1px 5px; border-radius: 3px; font-size: 10px;">
                            {det_mode}
                        </span>
                    </td>
                </tr>
            </table>
            <div style="background-color: #fff3cd; border: 1px solid #ffe69c; color: #664d03; padding: 4px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; text-align: center;">
                ⚠️ GPS MODE: {gps_mode} (Transit Route)
            </div>
        </div>
    </div>
    """
    return html.strip()



def create_event_tooltip(event: Union[UrbanEvent, Dict[str, Any]]) -> str:
    """
    Generate concise hover tooltip string for map markers.

    Args:
        event: UrbanEvent object or dict.

    Returns:
        Tooltip text string.
    """
    if isinstance(event, dict):
        evt_type = str(event.get("event_type", "UNKNOWN")).upper()
        cname = str(event.get("class_name", "unknown")).lower()
        sev = str(event.get("severity", "none")).upper()
        conf = float(event.get("confidence", 0.0))
        gps_mode = str(event.get("gps_mode", "SIMULATED")).upper()
    else:
        evt_type = event.event_type.upper()
        cname = event.class_name.lower()
        sev = event.severity.upper()
        conf = float(event.confidence)
        gps_mode = event.gps_mode.upper()

    conf_str = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"
    
    if evt_type == "ROAD_DAMAGE":
        return f"[{evt_type}: {cname} | Severity: {sev} | Conf: {conf_str} | GPS: {gps_mode}]"
    else:
        return f"[{evt_type}: {cname} | Conf: {conf_str} | GPS: {gps_mode}]"


def add_simulated_gps_watermark(folium_map: folium.Map, route_name: str = "Route-7B Transit Corridor") -> None:
    """
    Add a persistent disclosure watermark on the Folium map
    indicating that all coordinates are simulated for demonstration.

    Args:
        folium_map: folium.Map instance.
        route_name: Name of simulated transit corridor.
    """
    watermark_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.94);
        padding: 8px 12px;
        border: 2px solid #fd7e14;
        border-radius: 6px;
        font-family: Arial, sans-serif;
        font-size: 11px;
        color: #212529;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        max-width: 320px;
        pointer-events: auto;
    ">
        <div style="font-weight: bold; color: #d63384; margin-bottom: 2px;">
            ⚠️ GPS MODE: SIMULATED COORDINATES
        </div>
        <div>
            Points are interpolated along <strong>{route_name}</strong> for MVP demonstration.
        </div>
    </div>
    """
    folium_map.get_root().html.add_child(Element(watermark_html))


def add_route_corridor(
    folium_map: folium.Map,
    waypoints: List[Dict[str, Any]],
    route_name: str = "Simulated Route-7B Corridor",
    feature_group: Optional[folium.FeatureGroup] = None
) -> folium.FeatureGroup:
    """
    Add the simulated transit route polyline and terminus markers to the map.

    Args:
        folium_map: folium.Map instance.
        waypoints: List of waypoint dicts with 'latitude', 'longitude', 'segment_name'.
        route_name: Name of corridor.
        feature_group: Optional FeatureGroup to add into.

    Returns:
        The FeatureGroup containing route geometry.
    """
    if feature_group is None:
        feature_group = folium.FeatureGroup(name="🛣️ Simulated Transit Corridor", show=True)

    if not waypoints or len(waypoints) < 2:
        return feature_group

    coords = [[wp["latitude"], wp["longitude"]] for wp in waypoints]

    # Draw transit corridor line
    folium.PolyLine(
        locations=coords,
        color="#0d6efd",
        weight=4,
        opacity=0.75,
        dash_array="6, 6",
        tooltip=f"Simulated Bus Route: {route_name} (GPS Mode: SIMULATED)"
    ).add_to(feature_group)

    # Add Route Start Waypoint Marker
    start_wp = waypoints[0]
    folium.Marker(
        location=[start_wp["latitude"], start_wp["longitude"]],
        popup=folium.Popup(f"<b>Route Origin (Simulated)</b><br>{start_wp.get('segment_name', 'Start')}<br>Lat: {start_wp['latitude']}, Lon: {start_wp['longitude']}", max_width=250),
        tooltip="Route Start (Simulated)",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(feature_group)

    # Add Route End Waypoint Marker
    end_wp = waypoints[-1]
    folium.Marker(
        location=[end_wp["latitude"], end_wp["longitude"]],
        popup=folium.Popup(f"<b>Route Terminal (Simulated)</b><br>{end_wp.get('segment_name', 'Terminal')}<br>Lat: {end_wp['latitude']}, Lon: {end_wp['longitude']}", max_width=250),
        tooltip="Route Terminal (Simulated)",
        icon=folium.Icon(color="black", icon="flag-checkered", prefix="fa")
    ).add_to(feature_group)

    feature_group.add_to(folium_map)
    return feature_group


def create_event_map(
    events: Sequence[Union[UrbanEvent, Dict[str, Any]]],
    route_waypoints: Optional[List[Dict[str, Any]]] = None,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom_start: int = 14,
    enable_clustering: bool = False,
    show_route: bool = True
) -> folium.Map:
    """
    Build a complete, styled interactive Folium map displaying existing urban event data,
    differentiating road damage hazards and vehicle detections, with prominent simulated GPS indications.

    Args:
        events: List or sequence of UrbanEvent objects or dictionaries.
        route_waypoints: Optional list of simulated route waypoints. If None, loaded from GeoTagger.
        center_lat: Optional center latitude. Defaults to average event lat or corridor origin.
        center_lon: Optional center longitude. Defaults to average event lon or corridor origin.
        zoom_start: Initial zoom level (default 14).
        enable_clustering: If True, uses MarkerCluster for event markers.
        show_route: If True, draws the simulated transit route corridor.

    Returns:
        folium.Map instance ready for rendering.
    """
    # 1. Resolve waypoints
    if route_waypoints is None:
        try:
            tagger = GeoTagger()
            route_waypoints = tagger.waypoints
            route_name = tagger.route_name
        except Exception:
            route_waypoints = GeoTagger.DEFAULT_WAYPOINTS
            route_name = "Simulated Ring Road Transit Corridor"
    else:
        route_name = "Simulated Transit Corridor"

    # 2. Determine Map Center
    valid_lats = []
    valid_lons = []
    for evt in events:
        if isinstance(evt, dict):
            lat = evt.get("latitude")
            lon = evt.get("longitude")
        else:
            lat = evt.latitude
            lon = evt.longitude
        if lat is not None and lon is not None:
            valid_lats.append(float(lat))
            valid_lons.append(float(lon))

    if center_lat is None or center_lon is None:
        if valid_lats and valid_lons:
            map_center = [sum(valid_lats) / len(valid_lats), sum(valid_lons) / len(valid_lons)]
        elif route_waypoints:
            map_center = [route_waypoints[0]["latitude"], route_waypoints[0]["longitude"]]
        else:
            map_center = [settings.DEFAULT_ORIGIN_LAT, settings.DEFAULT_ORIGIN_LON]
    else:
        map_center = [center_lat, center_lon]

    # 3. Create Folium Map
    folium_map = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Add alternative base tile layer for user choice
    folium.TileLayer("CartoDB positron", name="CartoDB Light Map").add_to(folium_map)

    # 4. Add Simulated Transit Route PolyLine
    if show_route and route_waypoints:
        add_route_corridor(folium_map, route_waypoints, route_name=route_name)

    # 5. Create Feature Groups for Layers
    fg_damage = folium.FeatureGroup(name="🚨 Road Damage Hazards", show=True)
    fg_vehicles = folium.FeatureGroup(name="🚗 Vehicle Detections", show=True)

    damage_target = MarkerCluster(name="🚨 Clustered Road Damage").add_to(fg_damage) if enable_clustering else fg_damage
    vehicle_target = MarkerCluster(name="🚗 Clustered Vehicles").add_to(fg_vehicles) if enable_clustering else fg_vehicles

    # 6. Add Event Markers
    for evt in events:
        styling = get_marker_styling(evt)
        popup_html = create_event_popup_html(evt)
        tooltip_text = create_event_tooltip(evt)

        if isinstance(evt, dict):
            lat = float(evt.get("latitude", 0.0))
            lon = float(evt.get("longitude", 0.0))
            evt_type = str(evt.get("event_type", "UNKNOWN")).upper()
        else:
            lat = float(evt.latitude)
            lon = float(evt.longitude)
            evt_type = evt.event_type.upper()

        marker = folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=tooltip_text,
            icon=folium.Icon(
                color=styling["color"],
                icon=styling["icon"],
                prefix=styling.get("prefix", "fa"),
                icon_color=styling.get("icon_color", "white")
            )
        )

        if evt_type == "ROAD_DAMAGE":
            marker.add_to(damage_target)
        else:
            marker.add_to(vehicle_target)

    fg_damage.add_to(folium_map)
    fg_vehicles.add_to(folium_map)

    # 7. Add Simulated GPS Watermark & Disclosure Notice
    add_simulated_gps_watermark(folium_map, route_name=route_name)

    # 8. Add Layer Control for interactive toggling
    folium.LayerControl(position="topright", collapsed=False).add_to(folium_map)

    return folium_map


def render_folium_map(
    folium_map: folium.Map,
    width: Optional[int] = None,
    height: int = 520,
    key: Optional[str] = "sih_gis_map"
) -> Any:
    """
    Render a Folium map inside a Streamlit application using streamlit-folium.

    Args:
        folium_map: folium.Map instance to render.
        width: Pixel width (optional, responsive by default).
        height: Pixel height (default 520px).
        key: Streamlit component unique key.

    Returns:
        Output dict from st_folium or None.
    """
    try:
        from streamlit_folium import st_folium
        return st_folium(
            folium_map,
            width=width,
            height=height,
            key=key,
            returned_objects=["last_object_clicked"]
        )
    except Exception as e:
        import streamlit as st
        st.error(f"Error rendering interactive map: {e}")
        return None
