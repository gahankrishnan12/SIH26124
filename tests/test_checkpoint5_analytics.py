"""
SIH26124: Checkpoint 5 Test Suite - Traffic Analytics
Tests vehicle counts, class distributions, temporal time-series, density heuristics,
and route/source-level aggregations using controlled event datasets and SQLite integration.
"""
import unittest
import os
import gc
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.events.schema import UrbanEvent
from src.storage.db_manager import DatabaseManager
from src.analytics.traffic_metrics import (
    TrafficAnalytics,
    TrafficMetricsCalculator,
    compute_traffic_metrics,
    compute_vehicle_counts,
    compute_temporal_counts,
    classify_traffic_density,
    compute_source_statistics,
    DEFAULT_DENSITY_THRESHOLDS,
    DATA_DISCLAIMER
)
from config import settings


class TestCheckpoint5Analytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = settings.DATA_DIR / "events" / "test_analytics_urban_events.db"
        cls.db = DatabaseManager(db_path=str(cls.test_db_path))

    @classmethod
    def tearDownClass(cls):
        del cls.db
        gc.collect()
        try:
            if cls.test_db_path.exists():
                cls.test_db_path.unlink()
        except PermissionError:
            pass

    def setUp(self):
        self.db.clear_events()
        self.base_time = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    def _create_synthetic_events(self):
        """Create a controlled, deterministic set of test events."""
        events = []
        
        # 5 Cars (BUS_DEMO_01)
        for i in range(5):
            t = (self.base_time + timedelta(seconds=i * 15)).isoformat()
            events.append(UrbanEvent.create(
                event_type="VEHICLE",
                class_name="car",
                confidence=0.80 + (i * 0.02),
                latitude=28.6139 + (i * 0.001),
                longitude=77.2090 + (i * 0.001),
                frame_index=i * 30,
                bbox=[100.0, 100.0, 200.0, 200.0],
                timestamp=t,
                source_id="BUS_DEMO_01",
                detection_mode="REAL_AI"
            ))

        # 3 Buses (BUS_DEMO_01)
        for i in range(3):
            t = (self.base_time + timedelta(seconds=20 + i * 20)).isoformat()
            events.append(UrbanEvent.create(
                event_type="VEHICLE",
                class_name="bus",
                confidence=0.90,
                latitude=28.6145 + (i * 0.001),
                longitude=77.2095 + (i * 0.001),
                frame_index=15 + (i * 30),
                bbox=[50.0, 50.0, 300.0, 300.0],
                timestamp=t,
                source_id="BUS_DEMO_01",
                detection_mode="REAL_AI"
            ))

        # 2 Motorcycles (BUS_DEMO_02)
        for i in range(2):
            t = (self.base_time + timedelta(seconds=40 + i * 10)).isoformat()
            events.append(UrbanEvent.create(
                event_type="VEHICLE",
                class_name="motorcycle",
                confidence=0.75,
                latitude=28.6200 + (i * 0.002),
                longitude=77.2150 + (i * 0.002),
                frame_index=50 + (i * 20),
                bbox=[200.0, 200.0, 250.0, 250.0],
                timestamp=t,
                source_id="BUS_DEMO_02",
                detection_mode="DEMO_SIMULATION"
            ))

        # 4 Road Damage Events (Should be ignored by vehicle analytics)
        for i in range(4):
            t = (self.base_time + timedelta(seconds=i * 10)).isoformat()
            events.append(UrbanEvent.create(
                event_type="ROAD_DAMAGE",
                class_name="pothole",
                confidence=0.85,
                latitude=28.6139,
                longitude=77.2090,
                frame_index=i * 10,
                bbox=[50.0, 50.0, 100.0, 100.0],
                severity="medium",
                timestamp=t,
                source_id="BUS_DEMO_01",
                detection_mode="REAL_AI"
            ))

        return events

    def test_01_vehicle_count_calculations(self):
        """Verify total vehicle counts and proper exclusion of road damage events."""
        events = self._create_synthetic_events()
        analytics = TrafficAnalytics(events=events)

        # Total events = 14 (10 vehicles + 4 potholes)
        self.assertEqual(len(analytics.all_events), 14)
        
        # Vehicle count must be exactly 10
        total_veh = analytics.get_total_vehicle_count()
        self.assertEqual(total_veh, 10)

        # Filter by source_id
        bus1_veh = analytics.get_total_vehicle_count(source_id="BUS_DEMO_01")
        self.assertEqual(bus1_veh, 8)  # 5 cars + 3 buses

        bus2_veh = analytics.get_total_vehicle_count(source_id="BUS_DEMO_02")
        self.assertEqual(bus2_veh, 2)  # 2 motorcycles

        # Filter by minimum confidence
        high_conf_veh = analytics.get_total_vehicle_count(min_confidence=0.85)
        # 2 cars (0.86, 0.88 >= 0.85) + 3 buses (0.90) = 5
        self.assertEqual(high_conf_veh, 5)

        # Filter by detection mode
        real_ai_veh = analytics.get_total_vehicle_count(detection_mode="REAL_AI")
        self.assertEqual(real_ai_veh, 8)
        sim_veh = analytics.get_total_vehicle_count(detection_mode="DEMO_SIMULATION")
        self.assertEqual(sim_veh, 2)

    def test_02_counts_by_vehicle_class_and_shares(self):
        """Verify vehicle class frequency distribution, percentages, and dominant class."""
        events = self._create_synthetic_events()
        analytics = TrafficAnalytics(events=events)

        class_counts = analytics.get_vehicle_counts_by_class()
        expected_counts = {"car": 5, "bus": 3, "motorcycle": 2}
        self.assertEqual(class_counts, expected_counts)

        dist = analytics.get_vehicle_class_distribution()
        self.assertEqual(dist["total_vehicle_events"], 10)
        self.assertEqual(dist["dominant_class"], "car")
        self.assertEqual(dist["percentages_by_class"]["car"], 50.0)
        self.assertEqual(dist["percentages_by_class"]["bus"], 30.0)
        self.assertEqual(dist["percentages_by_class"]["motorcycle"], 20.0)

    def test_03_temporal_aggregation_and_rates(self):
        """Verify temporal bucketing, observation duration, and event rate calculations."""
        events = self._create_synthetic_events()
        analytics = TrafficAnalytics(events=events)

        # Base time + max second offset is 60s (i=4 -> 4*15 = 60s)
        temporal = analytics.get_vehicle_counts_over_time(interval_seconds=30)
        self.assertEqual(temporal["total_events"], 10)
        self.assertEqual(temporal["interval_seconds"], 30)
        self.assertGreater(temporal["duration_seconds"], 0.0)
        self.assertGreater(temporal["events_per_minute"], 0.0)
        self.assertGreater(temporal["events_per_hour"], 0.0)
        self.assertGreater(len(temporal["time_buckets"]), 0)
        self.assertIsNotNone(temporal["peak_bucket"])
        self.assertGreater(temporal["peak_count"], 0)

        # Verify bucket contents
        first_bucket = temporal["time_buckets"][0]
        self.assertIn("timestamp", first_bucket)
        self.assertIn("count", first_bucket)
        self.assertIn("by_class", first_bucket)
        self.assertIn("min_frame", first_bucket)
        self.assertIn("max_frame", first_bucket)

    def test_04_traffic_density_classification(self):
        """Verify density heuristics across all threshold ranges."""
        # 1. EMPTY / NO DATA
        analytics_empty = TrafficAnalytics(events=[])
        d_empty = analytics_empty.classify_traffic_density()
        self.assertEqual(d_empty["density_level"], "EMPTY")
        self.assertEqual(d_empty["observed_rate_per_minute"], 0.0)
        self.assertFalse(d_empty["scientific_validation"])

        # 2. LOW density: 2 events over 60 seconds (rate = 2 ev/min < 5.0)
        events_low = [
            UrbanEvent.create("VEHICLE", "car", 0.9, 28.0, 77.0, 1, [0, 0, 10, 10], timestamp=self.base_time.isoformat()),
            UrbanEvent.create("VEHICLE", "car", 0.9, 28.0, 77.0, 2, [0, 0, 10, 10], timestamp=(self.base_time + timedelta(seconds=60)).isoformat())
        ]
        d_low = TrafficAnalytics(events=events_low).classify_traffic_density()
        self.assertEqual(d_low["density_level"], "LOW")
        self.assertEqual(d_low["observed_rate_per_minute"], 2.0)

        # 3. MODERATE density: 8 events over 60 seconds (rate = 8 ev/min in [5.0, 15.0))
        events_mod = [
            UrbanEvent.create("VEHICLE", "car", 0.9, 28.0, 77.0, i, [0, 0, 10, 10], timestamp=(self.base_time + timedelta(seconds=i * 8.5)).isoformat())
            for i in range(8)
        ]
        d_mod = TrafficAnalytics(events=events_mod).classify_traffic_density(time_window_seconds=60.0)
        self.assertEqual(d_mod["density_level"], "MODERATE")
        self.assertEqual(d_mod["observed_rate_per_minute"], 8.0)

        # 4. HIGH density: 20 events over 60 seconds (rate = 20 ev/min in [15.0, 30.0))
        events_high = [
            UrbanEvent.create("VEHICLE", "car", 0.9, 28.0, 77.0, i, [0, 0, 10, 10], timestamp=(self.base_time + timedelta(seconds=i * 3.0)).isoformat())
            for i in range(20)
        ]
        d_high = TrafficAnalytics(events=events_high).classify_traffic_density(time_window_seconds=60.0)
        self.assertEqual(d_high["density_level"], "HIGH")
        self.assertEqual(d_high["observed_rate_per_minute"], 20.0)

        # 5. CONGESTED density: 35 events over 60 seconds (rate = 35 ev/min >= 30.0)
        events_cong = [
            UrbanEvent.create("VEHICLE", "car", 0.9, 28.0, 77.0, i, [0, 0, 10, 10], timestamp=(self.base_time + timedelta(seconds=i * 1.5)).isoformat())
            for i in range(35)
        ]
        d_cong = TrafficAnalytics(events=events_cong).classify_traffic_density(time_window_seconds=60.0)
        self.assertEqual(d_cong["density_level"], "CONGESTED")
        self.assertEqual(d_cong["observed_rate_per_minute"], 35.0)

    def test_05_source_level_statistics(self):
        """Verify per-source metrics, spatial boundaries, and multi-source breakdowns."""
        events = self._create_synthetic_events()
        analytics = TrafficAnalytics(events=events)

        source_stats = analytics.get_source_level_statistics()
        self.assertIn("BUS_DEMO_01", source_stats)
        self.assertIn("BUS_DEMO_02", source_stats)

        # Validate BUS_DEMO_01
        s1 = source_stats["BUS_DEMO_01"]
        self.assertEqual(s1["source_id"], "BUS_DEMO_01")
        self.assertEqual(s1["total_vehicle_events"], 8)
        self.assertEqual(s1["total_events_recorded"], 12) # 8 vehicles + 4 potholes
        self.assertEqual(s1["vehicle_counts_by_class"]["car"], 5)
        self.assertEqual(s1["vehicle_counts_by_class"]["bus"], 3)
        self.assertIsNotNone(s1["geographic_bounds"])
        self.assertIn("min_latitude", s1["geographic_bounds"])
        self.assertIn("max_latitude", s1["geographic_bounds"])
        self.assertIn("center_latitude", s1["geographic_bounds"])
        self.assertGreater(s1["average_vehicle_confidence"], 0.0)
        self.assertEqual(s1["detection_modes"]["REAL_AI"], 12)

        # Validate BUS_DEMO_02
        s2 = source_stats["BUS_DEMO_02"]
        self.assertEqual(s2["source_id"], "BUS_DEMO_02")
        self.assertEqual(s2["total_vehicle_events"], 2)
        self.assertEqual(s2["vehicle_counts_by_class"]["motorcycle"], 2)
        self.assertEqual(s2["detection_modes"]["DEMO_SIMULATION"], 2)

    def test_06_database_manager_integration(self):
        """Verify analytics module loaded directly from SQLite database."""
        events = self._create_synthetic_events()
        self.db.insert_events(events)

        # Verify DB has all 14 records
        db_events = self.db.get_events(limit=100)
        self.assertEqual(len(db_events), 14)

        # Test analytics instantiated with db_manager
        analytics = TrafficAnalytics(db_manager=self.db)
        self.assertEqual(len(analytics.all_events), 14)
        self.assertEqual(analytics.get_total_vehicle_count(), 10)

        # Test compute_traffic_metrics wrapper with db_manager
        summary = compute_traffic_metrics(db_manager=self.db)
        self.assertEqual(summary["total_events_in_dataset"], 14)
        self.assertEqual(summary["vehicle_events_analyzed"], 10)
        self.assertEqual(summary["road_damage_events_excluded"], 4)
        self.assertEqual(summary["vehicle_counts"]["total_vehicle_count"], 10)
        self.assertEqual(summary["vehicle_counts"]["dominant_vehicle_class"], "car")
        self.assertIn("BUS_DEMO_01", summary["source_level_statistics"])
        self.assertIn("data_transparency", summary)
        self.assertFalse(summary["data_transparency"]["is_scientifically_calibrated"])

    def test_07_standalone_functional_api(self):
        """Verify standalone helper functions compute identically to class methods."""
        events = self._create_synthetic_events()

        counts_res = compute_vehicle_counts(events)
        self.assertEqual(counts_res["total_vehicle_events"], 10)
        self.assertEqual(counts_res["dominant_class"], "car")

        temp_res = compute_temporal_counts(events, interval_seconds=45)
        self.assertEqual(temp_res["total_events"], 10)
        self.assertEqual(temp_res["interval_seconds"], 45)

        dens_res = classify_traffic_density(events, time_window_seconds=60.0)
        self.assertIn("density_level", dens_res)

        src_res = compute_source_statistics(events)
        self.assertEqual(len(src_res), 2)

    def test_08_edge_cases_and_robustness(self):
        """Verify handling of empty lists, dict formats, and non-vehicle data."""
        # 1. Empty list
        empty_analytics = TrafficAnalytics(events=[])
        self.assertEqual(empty_analytics.get_total_vehicle_count(), 0)
        self.assertEqual(empty_analytics.get_vehicle_counts_by_class(), {})
        self.assertIsNone(empty_analytics.get_vehicle_class_distribution()["dominant_class"])
        self.assertEqual(empty_analytics.get_vehicle_counts_over_time()["total_events"], 0)
        self.assertEqual(empty_analytics.get_source_level_statistics(), {})

        # 2. Only Road Damage events
        pothole_only = [
            UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.9, 28.0, 77.0, 1, [0, 0, 10, 10]),
            UrbanEvent.create("ROAD_DAMAGE", "rutting", 0.8, 28.0, 77.0, 2, [0, 0, 10, 10])
        ]
        pothole_analytics = TrafficAnalytics(events=pothole_only)
        self.assertEqual(pothole_analytics.get_total_vehicle_count(), 0)
        self.assertEqual(pothole_analytics.get_summary()["vehicle_events_analyzed"], 0)
        self.assertEqual(pothole_analytics.get_summary()["road_damage_events_excluded"], 2)

        # 3. Dict input representation
        dict_events = [
            UrbanEvent.create("VEHICLE", "truck", 0.88, 28.5, 77.1, 5, [0, 0, 50, 50]).to_dict(),
            UrbanEvent.create("VEHICLE", "car", 0.92, 28.5, 77.1, 6, [0, 0, 50, 50]).to_dict()
        ]
        dict_analytics = TrafficAnalytics(events=dict_events)
        self.assertEqual(dict_analytics.get_total_vehicle_count(), 2)
        self.assertEqual(dict_analytics.get_vehicle_counts_by_class()["truck"], 1)

        # 4. Single event duration zero division handling
        single_event = [UrbanEvent.create("VEHICLE", "bus", 0.9, 28.0, 77.0, 1, [0, 0, 10, 10])]
        single_analytics = TrafficAnalytics(events=single_event)
        temp_single = single_analytics.get_vehicle_counts_over_time()
        self.assertEqual(temp_single["total_events"], 1)
        self.assertEqual(temp_single["duration_seconds"], 0.0)
        self.assertEqual(temp_single["events_per_minute"], 0.0)

    def test_09_transparency_and_disclaimers_present(self):
        """Verify strict adherence to non-scientific disclosure and transparency standards."""
        events = self._create_synthetic_events()
        analytics = TrafficAnalytics(events=events)
        summary = analytics.get_summary()

        transparency = summary["data_transparency"]
        self.assertIn("is_scientifically_calibrated", transparency)
        self.assertFalse(transparency["is_scientifically_calibrated"])
        self.assertIn("methodology_note", transparency)
        self.assertIn("disclaimer", transparency)
        self.assertIn("Observational proxy data only", transparency["disclaimer"])


    def test_10_car_counted_as_vehicle(self):
        """Verify car is recognized and counted as a vehicle."""
        evt = UrbanEvent.create("VEHICLE", "car", 0.90, 28.61, 77.21, 1, [0, 0, 10, 10])
        analytics = TrafficAnalytics(events=[evt])
        self.assertEqual(analytics.get_total_vehicle_count(), 1)
        self.assertEqual(analytics.get_vehicle_counts_by_class(), {"car": 1})

    def test_11_motorcycle_counted_as_vehicle(self):
        """Verify motorcycle is recognized and counted as a vehicle."""
        evt = UrbanEvent.create("VEHICLE", "motorcycle", 0.85, 28.61, 77.21, 1, [0, 0, 10, 10])
        analytics = TrafficAnalytics(events=[evt])
        self.assertEqual(analytics.get_total_vehicle_count(), 1)
        self.assertEqual(analytics.get_vehicle_counts_by_class(), {"motorcycle": 1})

    def test_12_bus_counted_as_vehicle(self):
        """Verify bus is recognized and counted as a vehicle."""
        evt = UrbanEvent.create("VEHICLE", "bus", 0.92, 28.61, 77.21, 1, [0, 0, 10, 10])
        analytics = TrafficAnalytics(events=[evt])
        self.assertEqual(analytics.get_total_vehicle_count(), 1)
        self.assertEqual(analytics.get_vehicle_counts_by_class(), {"bus": 1})

    def test_13_truck_counted_as_vehicle(self):
        """Verify truck is recognized and counted as a vehicle."""
        evt = UrbanEvent.create("VEHICLE", "truck", 0.88, 28.61, 77.21, 1, [0, 0, 10, 10])
        analytics = TrafficAnalytics(events=[evt])
        self.assertEqual(analytics.get_total_vehicle_count(), 1)
        self.assertEqual(analytics.get_vehicle_counts_by_class(), {"truck": 1})

    def test_14_person_not_counted_as_vehicle(self):
        """Verify person (pedestrian) with event_type='VEHICLE' is NOT counted as a vehicle."""
        evt = UrbanEvent.create(
            event_type="VEHICLE",
            class_name="person",
            confidence=0.95,
            latitude=28.6139,
            longitude=77.2090,
            frame_index=15,
            bbox=[100.0, 100.0, 150.0, 200.0],
            detection_mode="REAL_AI"
        )
        analytics = TrafficAnalytics(events=[evt])
        self.assertEqual(len(analytics.all_events), 1)
        self.assertEqual(analytics.get_total_vehicle_count(), 0)
        self.assertEqual(analytics.get_vehicle_counts_by_class(), {})
        self.assertEqual(analytics.get_vehicle_events(), [])
        
        summary = analytics.get_summary()
        self.assertEqual(summary["total_events_in_dataset"], 1)
        self.assertEqual(summary["vehicle_events_analyzed"], 0)
        self.assertEqual(summary["non_vehicle_events_excluded"], 1)
        self.assertEqual(summary["vehicle_counts"]["total_vehicle_count"], 0)

    def test_15_road_damage_not_counted_as_vehicle(self):
        """Verify ROAD_DAMAGE / pothole is NOT counted as a vehicle."""
        evt = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.90, 28.61, 77.21, 1, [0, 0, 10, 10], severity="high")
        analytics = TrafficAnalytics(events=[evt])
        self.assertEqual(analytics.get_total_vehicle_count(), 0)
        self.assertEqual(analytics.get_vehicle_counts_by_class(), {})

    def test_16_mixed_vehicle_and_person_counts(self):
        """Verify mixed dataset with cars, trucks, motorcycles, buses, and persons computes correct vehicle counts."""
        events = [
            UrbanEvent.create("VEHICLE", "car", 0.90, 28.61, 77.21, 1, [0, 0, 10, 10]),
            UrbanEvent.create("VEHICLE", "truck", 0.85, 28.61, 77.21, 2, [0, 0, 10, 10]),
            UrbanEvent.create("VEHICLE", "motorcycle", 0.80, 28.61, 77.21, 3, [0, 0, 10, 10]),
            UrbanEvent.create("VEHICLE", "bus", 0.95, 28.61, 77.21, 4, [0, 0, 10, 10]),
            UrbanEvent.create("VEHICLE", "person", 0.99, 28.61, 77.21, 5, [0, 0, 10, 10]),  # Non-vehicle
            UrbanEvent.create("VEHICLE", "person", 0.75, 28.61, 77.21, 6, [0, 0, 10, 10]),  # Non-vehicle
            UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.88, 28.61, 77.21, 7, [0, 0, 10, 10]) # Non-vehicle
        ]
        analytics = TrafficAnalytics(events=events)
        self.assertEqual(len(analytics.all_events), 7)
        self.assertEqual(analytics.get_total_vehicle_count(), 4)
        
        counts = analytics.get_vehicle_counts_by_class()
        self.assertEqual(counts, {"car": 1, "truck": 1, "motorcycle": 1, "bus": 1})
        self.assertNotIn("person", counts)
        self.assertNotIn("pothole", counts)

        dist = analytics.get_vehicle_class_distribution()
        self.assertEqual(dist["total_vehicle_events"], 4)
        self.assertEqual(dist["percentages_by_class"]["car"], 25.0)
        self.assertEqual(dist["percentages_by_class"]["truck"], 25.0)
        self.assertEqual(dist["percentages_by_class"]["motorcycle"], 25.0)
        self.assertEqual(dist["percentages_by_class"]["bus"], 25.0)

    def test_17_density_rate_excludes_person_events(self):
        """Verify traffic density rate calculation strictly excludes person records."""
        # 1 real car at t=0 and 1 real car at t=60 (rate = 2.0 ev/min -> LOW)
        # Plus 50 person events in between (which would incorrectly trigger CONGESTED if counted)
        events = [
            UrbanEvent.create("VEHICLE", "car", 0.9, 28.0, 77.0, 1, [0, 0, 10, 10], timestamp=self.base_time.isoformat()),
            UrbanEvent.create("VEHICLE", "car", 0.9, 28.0, 77.0, 100, [0, 0, 10, 10], timestamp=(self.base_time + timedelta(seconds=60)).isoformat())
        ]
        for i in range(50):
            events.append(UrbanEvent.create(
                "VEHICLE", "person", 0.95, 28.0, 77.0, 10 + i, [0, 0, 10, 10],
                timestamp=(self.base_time + timedelta(seconds=i + 1)).isoformat()
            ))

        analytics = TrafficAnalytics(events=events)
        self.assertEqual(len(analytics.all_events), 52)
        self.assertEqual(analytics.get_total_vehicle_count(), 2)

        density = analytics.classify_traffic_density()
        self.assertEqual(density["observed_events"], 2)
        self.assertEqual(density["observed_rate_per_minute"], 2.0)
        self.assertEqual(density["density_level"], "LOW")

        temporal = analytics.get_vehicle_counts_over_time(interval_seconds=60)
        self.assertEqual(temporal["total_events"], 2)
        self.assertEqual(temporal["events_per_minute"], 2.0)

    def test_18_source_statistics_exclude_person_events(self):
        """Verify per-source vehicle statistics completely exclude person records."""
        events = [
            UrbanEvent.create("VEHICLE", "bus", 0.90, 28.61, 77.21, 1, [0, 0, 10, 10], source_id="BUS_01"),
            UrbanEvent.create("VEHICLE", "car", 0.80, 28.62, 77.22, 2, [0, 0, 10, 10], source_id="BUS_01"),
            UrbanEvent.create("VEHICLE", "person", 0.99, 28.63, 77.23, 3, [0, 0, 10, 10], source_id="BUS_01"),
            UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.85, 28.64, 77.24, 4, [0, 0, 10, 10], source_id="BUS_01")
        ]
        analytics = TrafficAnalytics(events=events)
        stats = analytics.get_source_level_statistics()
        
        self.assertIn("BUS_01", stats)
        s = stats["BUS_01"]
        self.assertEqual(s["total_events_recorded"], 4)
        self.assertEqual(s["total_vehicle_events"], 2) # only bus and car
        self.assertEqual(s["vehicle_counts_by_class"], {"bus": 1, "car": 1})
        self.assertNotIn("person", s["vehicle_counts_by_class"])
        self.assertEqual(s["average_vehicle_confidence"], 0.85) # (0.90 + 0.80) / 2

    def test_19_existing_analytics_preserved_for_valid_classes(self):
        """Verify full summary calculation for valid vehicle classes is completely preserved."""
        events = self._create_synthetic_events()
        # Add 3 person events to synthetic set
        for i in range(3):
            events.append(UrbanEvent.create(
                "VEHICLE", "person", 0.95, 28.61, 77.20, 100 + i, [0, 0, 10, 10],
                timestamp=(self.base_time + timedelta(seconds=i * 5)).isoformat()
            ))

        analytics = TrafficAnalytics(events=events)
        # 14 base events (10 vehicles + 4 potholes) + 3 persons = 17 total
        self.assertEqual(len(analytics.all_events), 17)
        self.assertEqual(analytics.get_total_vehicle_count(), 10)
        self.assertEqual(analytics.get_vehicle_counts_by_class(), {"car": 5, "bus": 3, "motorcycle": 2})

        summary = analytics.get_summary()
        self.assertEqual(summary["total_events_in_dataset"], 17)
        self.assertEqual(summary["vehicle_events_analyzed"], 10)
        self.assertEqual(summary["road_damage_events_excluded"], 4)
        self.assertEqual(summary["non_vehicle_events_excluded"], 7) # 4 potholes + 3 persons


if __name__ == "__main__":
    unittest.main()

