# SIH26124: AI-Powered Mobile Urban Intelligence Platform

Smart India Hackathon 2026 — Problem Statement: **SIH26124**
> **AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet**

---

## 1. Project Overview

This platform turns public transit bus fleets into intelligent, distributed mobile edge-sensing units. By mounting camera sensors on transit vehicles, the platform continuously analyzes road conditions and traffic dynamics in real-time on CPU edge hardware.

### Core Capabilities:
1. **Dual AI Computer Vision**: Simultaneous vehicle detection (`yolov8n.pt`) and road damage hazard detection (`road_damage_yolov8.pt`) on video feeds.
2. **Geotagged Urban Events**: Real-time event structuring (`UrbanEvent`) with simulated GPS waypoint interpolation and spatial-temporal deduplication.
3. **SQLite Persistence & Analytics**: Local ACID-compliant SQLite event storage, CSV/JSON data exports, and transparent traffic analytics (volume, class shares, temporal trends, heuristic density).
4. **GIS Spatial Visualization**: Interactive Folium map displaying transit corridor waypoints and severity-coded road hazard markers.
5. **Decision Support Dashboard**: Streamlit web application providing fleet operators and municipal engineers with real-time controls, analytics, maps, and audit logs.

---

## 2. Architecture & Pipeline

```
                              Transit Bus Camera Feed
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                       Dual AI Detection Engine                         │
    │  ├── Vehicle Detection: YOLOv8n (car, bus, truck, motorcycle)          │
    │  └── Road Damage Detection: Fine-tuned YOLOv8 (pothole, damage)        │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                  Event Generator & Geotagger                           │
    │  ├── Simulated GPS Interpolation (Route-7B Transit Corridor)           │
    │  └── Prototype Spatial-Temporal Deduplication (Sliding Window)         │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                      SQLite Persistence Layer                          │
    │  ├── Table: events (event_id, type, class, conf, lat, lon, time, etc.) │
    │  └── Export APIs: CSV & JSON Serializers                               │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                   Traffic Analytics Subsystem                          │
    │  ├── Whitelisted Vehicle Counts (car, motorcycle, bus, truck)          │
    │  ├── Class Share Distributions & Dominant Class Identification         │
    │  ├── Time-Series Aggregations & Observation Rates                      │
    │  └── Heuristic Traffic Density Categorization                          │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │             Streamlit Smart-City Decision Support Dashboard            │
    │  ├── Real-Time Model Status & Transparency Disclosures                 │
    │  ├── Video Ingestion, Live Progress & Annotated HUD Player             │
    │  ├── Traffic & Road Damage KPI Summary Cards                           │
    │  ├── Interactive Folium GIS Map (Simulated GPS Corridors)              │
    │  ├── Filterable Structured Event Grid                                  │
    │  └── One-Click CSV / JSON Data Downloads                               │
    └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Transparency & Methodology Disclosures

| Component | Operational Status | Technical Details & Transparency Notes |
|---|---|---|
| **Vehicle AI Detection** | **Real AI** | Pretrained Ultralytics `yolov8n.pt` model running on CPU. |
| **Road Damage Detection** | **Dual-Mode** | **`REAL_AI`**: Fine-tuned YOLOv8 model (`road_damage_yolov8.pt`).<br>**`DEMO_SIMULATION`**: Transparent fallback simulation mode toggled via UI for testing. |
| **GPS Telemetry** | **Simulated** | Interpolated transit corridor coordinates along Route-7B (New Delhi origin `28.6139°N, 77.2090°E`). Explicitly labeled as `SIMULATED GPS`. |
| **Traffic Analytics** | **Observational Proxy** | Measures vehicle detection events captured by mobile transit edge sensors and deduplicated across frames. Observational proxy data only; not scientifically calibrated induction-loop or Doppler radar measurements. |
| **Traffic Density** | **Rule-Based Heuristic** | Categorized into `EMPTY` ($0$), `LOW` ($<5$ ev/min), `MODERATE` ($5-15$ ev/min), `HIGH` ($15-30$ ev/min), and `CONGESTED` ($\ge 30$ ev/min). |
| **Vehicle Whitelist** | **Enforced** | Strict vehicle class whitelist (`car`, `motorcycle`, `bus`, `truck`). Pedestrians (`person`) and road hazards are strictly isolated from vehicle counts. |
| **Severity Scoring** | **Prototype Heuristic** | Bounding box relative area heuristic ($<1.5\%$ Low, $1.5-4.0\%$ Medium, $\ge 4.0\%$ High). |

---

## 4. Models Used

* **Vehicle Detector**: `yolov8n.pt` (Ultralytics Nano, 6.25 MB, COCO pretrained).
* **Road Damage Detector**: `road_damage_yolov8.pt` (Fine-tuned YOLOv8, 21.48 MB, Roboflow/Smartathon Pothole dataset).
* **Target Classes**:
  * Vehicles: `car`, `motorcycle`, `bus`, `truck` (pedestrian `person` parsed separately for non-vehicle analytics).
  * Road Damage: `pothole`.

---

## 5. Measured CPU Performance Benchmarks (Empirical)

> **Execution Environment**: Windows 64-bit, Python 3.13, CPU-only execution (no GPU required).

| Benchmark Metric | Measured Empirical Value | Operational Context |
|---|---|---|
| **Vehicle Inference Latency** | **228.81 ms / frame** | `yolov8n.pt` on 768x432 CPU frames |
| **Damage Inference Latency** | **263.89 ms / frame** | `road_damage_yolov8.pt` on 768x432 CPU frames |
| **Combined Dual-Model Latency** | **492.70 ms / frame** | Dual YOLO models executed sequentially per frame |
| **Dual-Model Inference Throughput** | **2.03 FPS** | Dual AI inference speed on standard CPU |
| **End-to-End Pipeline Throughput** | **2.00 FPS** | Includes video I/O, geotagging, deduplication & SQLite insert |
| **Duplicate Suppression Ratio** | **96.1% (49/51 detections)** | Sliding window spatial-temporal deduplication efficiency |

---

## 6. Project Directory Structure

```text
SIH26124/
├── app.py                      # Main Streamlit Decision Support Dashboard
├── config/
│   └── settings.py             # Centralized paths, coordinates, class mappings
├── data/
│   ├── events/                 # SQLite database & exported CSV/JSON files
│   │   ├── urban_events.db
│   │   ├── events_export.csv
│   │   └── events_export.json
│   ├── gps/
│   │   └── sample_bus_route.json # Delhi Transit Corridor Route-7B waypoints
│   └── sample/                 # Test videos & road damage imagery
│       ├── real_road_sample.mp4
│       ├── backup_road_demo.mp4
│       ├── test_pothole_big.jpg
│       └── annotated_real_road_sample.mp4
├── docs/                       # Project documentation & specs
├── Models/
│   ├── yolov8n.pt              # Vehicle detection model weights
│   └── road_damage_yolov8.pt   # Road damage model weights
├── src/
│   ├── analytics/              # Checkpoint 5 Traffic Analytics
│   │   ├── __init__.py
│   │   └── traffic_metrics.py
│   ├── detection/              # Checkpoint 2 & 3 AI Detectors
│   │   ├── vehicle_detector.py
│   │   └── road_damage_detector.py
│   ├── events/                 # Checkpoint 4 Event Engine
│   │   ├── schema.py
│   │   ├── generator.py
│   │   └── geo_tagger.py
│   ├── maps/                   # Checkpoint 6 Folium GIS Subsystem
│   │   ├── __init__.py
│   │   └── map_generator.py
│   ├── storage/                # SQLite Database Manager
│   │   └── db_manager.py
│   └── video/                  # Video Processing Pipeline
│       └── processor.py
├── tests/                      # Comprehensive Unit Test Suite (CP1 - CP5)
│   ├── test_checkpoint1.py
│   ├── test_checkpoint2.py
│   ├── test_checkpoint3.py
│   ├── test_checkpoint4.py
│   └── test_checkpoint5_analytics.py
└── requirements.txt
```

---

## 7. Quick Start & Execution Guide

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Launch Decision Support Dashboard
```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501` in any modern web browser.*

### 3. Run Automated Test Suite
```bash
python -m unittest discover tests -v
```
**Test Suite Status**: **46/46 Tests Passing (100%)**

---

## 8. Development Roadmap Status

- [x] **Checkpoint 1**: Environment setup, folder hierarchy, video/GPS sample assets.
- [x] **Checkpoint 2**: YOLOv8 vehicle detection & measured CPU benchmark engine.
- [x] **Checkpoint 3**: Road damage detection with dual `REAL_AI` / `DEMO_SIMULATION` modes.
- [x] **Checkpoint 4**: Event generator, simulated GPS geotagging, deduplication & SQLite persistence.
- [x] **Checkpoint 5**: Transparent traffic analytics, class share distributions, density heuristics & pedestrian separation.
- [x] **Checkpoint 6**: Streamlit Smart-City Decision Support Dashboard with Folium GIS spatial maps.
- [x] **Checkpoint 7**: Final end-to-end integration, demo data generation, and verification.

---

## 9. Known Limitations

1. **Edge Camera Field-of-View**: Detections are constrained to the camera angle mounted on transit buses; vehicles occluded by heavy traffic may be missed.
2. **GPS Simulation**: Real-world transit buses require hardware NMEA GPS receivers; this prototype MVP uses simulated waypoints along Route-7B.
3. **Traffic Volume Calibration**: Measured traffic rates reflect observational edge detections rather than calibrated stationary radar or inductive loops.
