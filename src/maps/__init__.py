"""
SIH26124: Maps Package

Provides both GIS event mapping and urban map generation.
"""

from src.maps.folium_map import (
    create_event_map,
    render_folium_map,
    get_marker_styling,
    create_event_popup_html,
    create_event_tooltip,
    add_simulated_gps_watermark,
    add_route_corridor,
)

from src.maps.map_generator import (
    create_urban_map,
    SEVERITY_COLOR_MAP,
    SEVERITY_RADIUS_MAP,
)

__all__ = [
    "create_event_map",
    "render_folium_map",
    "get_marker_styling",
    "create_event_popup_html",
    "create_event_tooltip",
    "add_simulated_gps_watermark",
    "add_route_corridor",
    "create_urban_map",
    "SEVERITY_COLOR_MAP",
    "SEVERITY_RADIUS_MAP",
]
