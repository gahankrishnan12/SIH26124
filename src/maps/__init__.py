"""
SIH26124: GIS Mapping Subsystem
Folium spatial mapping and simulated GPS corridor visualization.
"""
from src.maps.map_generator import (
    create_urban_map,
    SEVERITY_COLOR_MAP,
    SEVERITY_RADIUS_MAP
)

__all__ = [
    "create_urban_map",
    "SEVERITY_COLOR_MAP",
    "SEVERITY_RADIUS_MAP"
]
