"""
SIH26124: Prototype Fleet Bus Configuration & Authentication Module
Defines demo bus fleet identities and isolated prototype authentication logic.
"""
from typing import Dict, List, Optional, Any

# Prototype Demo Buses Configuration
# Note: Fictional demonstration routes and isolated prototype PINs for hackathon demo.
_DEMO_BUS_REGISTRY: Dict[str, Dict[str, str]] = {
    "BUS-001": {
        "bus_id": "BUS-001",
        "display_name": "City Transit Bus 001",
        "route": "Route-7B (Central Ring Road Corridor)",
        "pin": "1001"
    },
    "BUS-002": {
        "bus_id": "BUS-002",
        "display_name": "City Transit Bus 002",
        "route": "Route-12A (North Express Corridor)",
        "pin": "1002"
    },
    "BUS-003": {
        "bus_id": "BUS-003",
        "display_name": "City Transit Bus 003",
        "route": "Route-4M (Ring Metro Feeder)",
        "pin": "1003"
    },
    "BUS-004": {
        "bus_id": "BUS-004",
        "display_name": "City Transit Bus 004",
        "route": "Route-9C (South Corridor Shuttle)",
        "pin": "1004"
    },
    "BUS-005": {
        "bus_id": "BUS-005",
        "display_name": "City Transit Bus 005",
        "route": "Route-15E (East River Connector)",
        "pin": "1005"
    }
}

UNKNOWN_BUS_LABEL = "UNKNOWN / LEGACY SOURCE"


def get_available_bus_ids() -> List[str]:
    """Return sorted list of all configured demo bus IDs."""
    return sorted(list(_DEMO_BUS_REGISTRY.keys()))


def get_available_buses() -> List[Dict[str, str]]:
    """
    Return list of public bus profiles (excluding security PINs).
    """
    return [
        {
            "bus_id": bus["bus_id"],
            "display_name": bus["display_name"],
            "route": bus["route"]
        }
        for bus in _DEMO_BUS_REGISTRY.values()
    ]


def get_bus_info(bus_id: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Retrieve public bus profile for a given bus_id (excluding security PIN).
    Returns None if bus_id is not registered.
    """
    if not bus_id or bus_id not in _DEMO_BUS_REGISTRY:
        return None
    bus = _DEMO_BUS_REGISTRY[bus_id]
    return {
        "bus_id": bus["bus_id"],
        "display_name": bus["display_name"],
        "route": bus["route"]
    }


def format_bus_display(bus_id: Optional[str]) -> str:
    """
    Format a user-friendly label for a bus ID.
    If bus_id is None, empty, or unknown, returns UNKNOWN_BUS_LABEL.
    """
    if not bus_id:
        return UNKNOWN_BUS_LABEL
    info = get_bus_info(bus_id)
    if info:
        return f"{info['bus_id']} — {info['display_name']} ({info['route']})"
    return f"{bus_id} ({UNKNOWN_BUS_LABEL})"


def authenticate_bus(bus_id: str, pin: str) -> bool:
    """
    Validate prototype bus authentication against configured PIN.
    Isolated within this module so credentials are not exposed to app.py.
    """
    if not bus_id or not pin:
        return False
    clean_bus_id = bus_id.strip()
    clean_pin = pin.strip()
    if clean_bus_id not in _DEMO_BUS_REGISTRY:
        return False
    return _DEMO_BUS_REGISTRY[clean_bus_id]["pin"] == clean_pin
