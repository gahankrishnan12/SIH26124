"""
SIH26124: Configuration Settings
Centralized paths, parameters, and constants for the Urban Intelligence Platform.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
EVENTS_DATA_DIR = DATA_DIR / "events"
GPS_DATA_DIR = DATA_DIR / "gps"
SRC_DIR = BASE_DIR / "src"

# Database
DB_PATH = EVENTS_DATA_DIR / "urban_events.db"

# Vehicle Detection Target Classes (COCO Dataset IDs)
VEHICLE_CLASS_MAP = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# GPS Default Origin (Simulated Route: New Delhi Urban Transit Corridor)
DEFAULT_ORIGIN_LAT = 28.6139
DEFAULT_ORIGIN_LON = 77.2090
SAMPLE_GPS_ROUTE_PATH = GPS_DATA_DIR / "sample_bus_route.json"

# UI / Display Settings
APP_TITLE = "Mobile Urban Intelligence Platform"
APP_SUBTITLE = "AI-Powered Road Condition & Traffic Sensing Fleet (SIH26124)"
