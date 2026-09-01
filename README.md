# SIH26124 Fleet Intelligence

## AI-Powered Mobile Urban Intelligence Platform

SIH26124 Fleet Intelligence is a prototype mobile urban sensing and decision-support platform designed to transform road observations collected from public-transport vehicles into structured urban intelligence.

The platform combines AI-based vehicle detection, road-damage detection, structured event generation, fleet/bus identity, traffic analytics, road-health assessment, GIS visualization, and maintenance-priority analysis within a unified Streamlit dashboard.

For the current prototype demonstration, recorded road footage is used as the sensing input because access to live bus-mounted camera feeds is not available.

---

## 1. Problem Statement

Urban authorities need continuous and geographically distributed information about traffic conditions and road infrastructure. Conventional monitoring systems often rely on fixed infrastructure and may provide limited spatial coverage.

Public-transport vehicles already travel through large portions of urban road networks and can serve as mobile sensing platforms.

SIH26124 explores a prototype architecture in which road video captured from mobile sensing units is processed using computer vision models to detect vehicles and road damage, convert detections into structured events, and transform those events into actionable traffic and infrastructure intelligence.

---

## 2. Proposed Solution

The system processes road video through an AI perception pipeline and converts individual detections into structured urban events.

These events are stored in SQLite and subsequently consumed by multiple analytical modules:

- Traffic analytics
- Road-health assessment
- Maintenance-priority ranking
- GIS-based spatial visualization
- Event exploration
- Data export

Each newly generated event can be associated with the authenticated sensing bus through a `bus_id`, allowing the platform to distinguish observations from different vehicles in the fleet.

---

## 3. System Architecture

---

text
                    BUS AUTHENTICATION
                           |
                           v
                    CURRENT BUS ID
                           |
                           v
                    VIDEO INGESTION
                           |
                           v
                 +----------------------+
                 |     AI PERCEPTION    |
                 |                      |
                 | Vehicle Detection    |
                 | Road Damage Detection|
                 +----------+-----------+
                            |
                            v
                    STRUCTURED EVENTS
                            |
                            v
                         SQLite
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
       Traffic Analytics  Road Health     GIS Map
             |              |              |
             |              v              |
             |      Maintenance Priority  |
             |                             |
             +--------------+--------------+
                            |
                            v
                     EVENT EXPLORER
                            |
                            v
                    REPORTS / EXPORTS

---

## 4. End-to-End Data Flow
Road Video
    |
    v
Frame Extraction
    |
    +----------------------------+
    |                            |
    v                            v
Vehicle YOLOv8             Road Damage YOLOv8
    |                            |
    v                            v
Vehicle Detections        Damage Detections
    |                            |
    +-------------+--------------+
                  |
                  v
           Event Generation
                  |
                  v
             UrbanEvent
                  |
                  +---- bus_id
                  |
                  +---- timestamp
                  |
                  +---- class
                  |
                  +---- confidence
                  |
                  +---- severity
                  |
                  +---- location
                  |
                  v
               SQLite
                  |
       +----------+----------+----------+
       |          |          |          |
       v          v          v          v
    Traffic    Road Health  GIS      Explorer
    Analytics  Assessment   Map      & Reports

---

## 5. Key Capabilities

5.1 AI-Based Vehicle Detection

The system uses a YOLOv8 Nano model for vehicle detection.

Supported vehicle classes include:

Car
Bus
Truck
Motorcycle

Pedestrian detections are retained in the raw event stream but excluded from valid vehicle traffic metrics.

5.2 Road-Damage Detection

A fine-tuned YOLOv8 model is used for road-damage detection.

The current demonstration includes pothole detection.

The model produces actual inference results including:

Bounding boxes
Class labels
Confidence scores

These detections are converted into structured ROAD_DAMAGE events.

5.3 Video Processing

The video-processing pipeline performs:

Video ingestion
Frame extraction
Vehicle inference
Road-damage inference
Detection deduplication
Event generation
Optional SQLite persistence
Annotated video generation
Processing-performance reporting

The application supports:

Built-in demonstration videos
Custom uploaded videos
MP4
AVI
MOV

## 6. Fleet and Bus Identity

The system includes prototype fleet authentication and bus identity tracking.

Demo buses are configured as:

BUS-001
BUS-002
BUS-003
BUS-004
BUS-005

After authentication, the active bus ID is propagated through the video-processing pipeline.

Bus Authentication
        |
        v
Session State
        |
        v
VideoProcessor
        |
        v
EventGenerator
        |
        v
UrbanEvent.bus_id
        |
        v
SQLite events.bus_id
        |
        v
Bus-Specific Analytics

This allows the dashboard to provide:

Current Bus view
Fleet View
Bus-specific event filtering
Bus identity in event records
Bus identity in GIS event metadata

## 7. Traffic Analytics

The traffic analytics module transforms detected vehicle events into operational traffic indicators.

Current analytics include:

Total valid vehicle events
Vehicle-class distribution
Dominant vehicle class
Observed event rate
Traffic-density classification
Temporal distribution

Valid traffic classes are:

car
bus
truck
motorcycle

person detections are intentionally excluded from vehicle traffic metrics.

## 8. Road Health and Maintenance Priority

Road-damage events are aggregated spatially to assess road-segment condition.

The Road Health module considers factors including:

Damage count
Damage severity
Damage recurrence
Segment-level condition
Maintenance priority

The resulting system provides:

Network health indicators
Segment health scores
Severity distribution
Critical/high/medium/low priority classification
Maintenance-priority ranking

The road-health calculation is a prototype decision-support heuristic and is not intended to represent an official government road-maintenance standard.

## 9. GIS Spatial Intelligence

Detected events are associated with spatial information and visualized through an interactive GIS interface.

The GIS module provides:

Event markers
Road-segment visualization
Event metadata
Bus identity
Detection information
Spatial context for road-damage events

GPS coordinates in the current prototype are simulated/interpolated along predefined transit routes for demonstration purposes.

## 10. Event Management

All detections are converted into structured UrbanEvent records.

An event can contain information such as:

Event ID
Event Type
Class Name
Confidence
Severity
Timestamp
Source ID
Bus ID
Latitude
Longitude
Detection Mode

Events are stored in SQLite and can be queried through the dashboard.

The platform supports:

Event filtering
Bus-specific filtering
Fleet-wide filtering
Legacy-event handling
CSV export
JSON export
## 11. Dashboard

The Streamlit dashboard provides a unified operational interface containing:

Overview
Video & AI
Traffic Analytics
Road Health
GIS Map
Event Explorer
Reports
Overview

Provides high-level operational indicators and fleet context.

Video & AI

Provides video selection, AI processing, processing metrics, and annotated output.

Traffic Analytics

Provides vehicle-class and traffic-flow analysis.

Road Health

Provides segment condition and maintenance-priority information.

GIS Map

Provides spatial visualization of detected events.

Event Explorer

Provides detailed inspection and filtering of structured events.

Reports

Provides downloadable analytical and event data.

## 12. Demonstration Video

The repository contains:

data/sample/judge_demo_1.mp4

This recorded road video is used for the hackathon demonstration.

The application processes the video through the actual AI inference pipeline and generates an annotated output containing model-generated detections.

The demonstration shows:

Road Video
    |
    v
Vehicle Detection
    |
    +---- Car
    +---- Motorcycle
    +---- Truck
    |
    v
Road Damage Detection
    |
    +---- Pothole
    |
    v
Structured Urban Events
    |
    v
Traffic Analytics
    |
    v
Road Health
    |
    v
GIS Visualization

The current prototype does not claim to have access to a live bus-mounted camera feed.

## 13. Technology Stack

Component	Technology
Programming Language	Python
Dashboard	Streamlit
Computer Vision	OpenCV
Vehicle Detection	YOLOv8 Nano
Road Damage Detection	Fine-tuned YOLOv8
Deep Learning Framework	PyTorch
Data Processing	Pandas, NumPy
Database	SQLite
GIS Visualization	Folium
Streamlit GIS Integration	Streamlit-Folium
Testing	Python unittest
Version Control	Git / GitHub

## 14. Models

Vehicle Detection Model
models/yolov8n.pt

A COCO-pretrained YOLOv8 Nano model used for vehicle detection.

Road Damage Detection Model
models/road_damage_yolov8.pt

A fine-tuned YOLOv8 model used for road-damage detection.

The prototype uses actual model inference for the demonstrated detections and does not manually draw or hardcode bounding boxes.

## 15. Repository Structure

SIH26124/
|
├── app.py
├── README.md
├── requirements.txt
|
├── config/
|   ├── buses.py
|   └── settings.py
|
├── models/
|   ├── pothole_peterhdd.pt
|   ├── pothole_samdutse.pt
|   ├── road_damage_yolov8.pt
|   └── yolov8n.pt
|
├── src/
|   ├── analytics/
|   ├── detection/
|   ├── events/
|   ├── maps/
|   ├── storage/
|   └── video/
|
├── data/
|   ├── events/
|   └── sample/
|       ├── judge_demo_1.mp4
|       ├── real_road_sample.mp4
|       └── backup_road_demo.mp4
|
├── tests/
|
└── docs/

## 16. Installation

Clone the Repository
git clone https://github.com/meghashree-23/SIH26124.git
cd SIH26124
Create a Virtual Environment

Windows:

python -m venv .venv

Activate it:

source .venv/Scripts/activate
Install Dependencies
pip install -r requirements.txt

## 17. Run the Application

Start the Streamlit dashboard:

streamlit run app.py

The application will open in a browser.

## 18. Demonstration Workflow

A complete demonstration can be performed using the following sequence:

1. Bus Authentication
        |
2. Select Road Video
        |
3. Run AI Processing
        |
4. View Annotated AI Output
        |
5. Inspect Generated Events
        |
6. View Traffic Analytics
        |
7. View Road Health
        |
8. Explore GIS Map
        |
9. Review Reports and Exports
19. Testing

Run the complete automated test suite:

python -m unittest discover tests -v

Additional validation commands:

python -m py_compile app.py
git diff --check

The project includes regression tests covering:

Project configuration
Vehicle detection
Road-damage detection
Video processing
Event generation
Deduplication
SQLite persistence
GIS mapping
Road Health
Traffic analytics
Fleet authentication
Bus identity
Database migration
Event filtering
Export functionality

## 20. Prototype Disclosures

Recorded Video Input

The current hackathon demonstration uses recorded road footage because access to live bus-mounted camera feeds is not currently available.

The video-processing architecture is designed to accept continuous video input.

Simulated GPS

GPS coordinates are simulated/interpolated along predefined transit routes for demonstration purposes.

Traffic Analytics

Traffic metrics are observational estimates derived from detected vehicle events. They are not calibrated against radar, induction loops, or other dedicated traffic-counting infrastructure.

Road Health

Road-health and maintenance-priority scores are prototype decision-support heuristics and are not official government road-maintenance standards.

Authentication

Bus authentication is implemented as prototype session-state authentication for demonstration purposes. It is not production-grade authentication.

## 21. Current Prototype Status

The prototype demonstrates an end-to-end mobile urban intelligence pipeline:

Mobile Road Video
        |
        v
AI Perception
        |
        v
Structured Urban Events
        |
        v
Bus Attribution
        |
        v
Persistent Event Store
        |
        +-------------------+
        |                   |
        v                   v
Traffic Intelligence   Infrastructure Intelligence
        |                   |
        v                   v
Traffic Analytics     Road Health
                            |
                            v
                    Maintenance Priority
                            |
                            v
                       GIS Mapping

The system demonstrates how mobile sensing data can be transformed from raw video observations into structured information for traffic monitoring and infrastructure decision support.
---

## 22. Project Status

Hackathon Prototype — Complete

The current implementation provides an integrated demonstration of:

AI-based road perception
Mobile sensing architecture
Bus identity
Structured event generation
Persistent event storage
Traffic analytics
Road-health assessment
Maintenance prioritization
GIS visualization
Operational dashboard
Data export
