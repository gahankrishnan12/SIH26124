"""
SIH26124: Maps Package
"""
from src.maps.folium_map import (
    create_event_map,
    render_folium_map,
    get_marker_styling,
    create_event_popup_html,
    create_event_tooltip,
    add_simulated_gps_watermark,
    add_route_corridor
)

__all__ = [
    "create_event_map",
    "render_folium_map",
    "get_marker_styling",
    "create_event_popup_html",
    "create_event_tooltip",
    "add_simulated_gps_watermark",
    "add_route_corridor"
]
