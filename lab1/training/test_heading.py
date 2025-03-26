import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from detection import (
    calculate_heading_matrix,
    detect_vessel_anomalies,
    ANGLE_DIFF_ANOMALY,
)


class TestHeadingCalculations(unittest.TestCase):
    def test_calculate_heading_matrix(self):
        """Test the heading calculation with known values"""
        # Test case 1: Heading North (0°)
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 1.0, 0.0
        heading = calculate_heading_matrix(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(heading, 0.0, places=1)

        # Test case 2: Heading East (90°)
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, 1.0
        heading = calculate_heading_matrix(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(heading, 90.0, places=1)

        # Test case 3: Heading South (180°)
        lat1, lon1 = 1.0, 0.0
        lat2, lon2 = 0.0, 0.0
        heading = calculate_heading_matrix(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(heading, 180.0, places=1)

        # Test case 4: Heading West (270°)
        lat1, lon1 = 0.0, 1.0
        lat2, lon2 = 0.0, 0.0
        heading = calculate_heading_matrix(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(heading, 270.0, places=1)

        # Test case 5: Diagonal heading (45°)
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 1.0, 1.0
        heading = calculate_heading_matrix(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(heading, 45.0, places=1)

    def test_heading_array_calculation(self):
        """Test the heading calculation with arrays"""
        lat1 = np.array([0.0, 0.0, 1.0, 0.0])
        lon1 = np.array([0.0, 0.0, 0.0, 1.0])
        lat2 = np.array([1.0, 0.0, 0.0, 0.0])
        lon2 = np.array([0.0, 1.0, 0.0, 0.0])

        headings = calculate_heading_matrix(lat1, lon1, lat2, lon2)

        expected = np.array([0.0, 90.0, 180.0, 270.0])
        np.testing.assert_array_almost_equal(headings, expected, decimal=1)

    def test_heading_boundary_cases(self):
        """Test edge cases like the 0/360 degree boundary"""
        # Test heading just below 360°
        lat1, lon1 = np.array([0.0]), np.array([0.0])
        lat2, lon2 = np.array([0.1]), np.array([-0.01])
        heading = calculate_heading_matrix(lat1, lon1, lat2, lon2)
        self.assertTrue(heading > 350.0 and heading < 360.0)

        # Test heading just above 0°
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.1, 0.01
        heading = calculate_heading_matrix(lat1, lon1, lat2, lon2)
        self.assertTrue(heading > 0.0 and heading < 10.0)


class TestHeadingAnomalyDetection(unittest.TestCase):
    def create_test_vessel_data(self, headings_mismatch=False):
        """Create synthetic vessel data for testing"""
        now = datetime.now()
        data = {
            "MMSI": ["123456789"] * 5,
            "Ship type": ["Cargo"] * 5,
            "Timestamp": [now + timedelta(minutes=i) for i in range(5)],
            "Latitude": [0.0, 0.01, 0.02, 0.03, 0.04],
            "Longitude": [0.0, 0.01, 0.02, 0.03, 0.04],
            "Heading": [45.0, 45.0, 45.0, 45.0, 45.0],
        }

        # If testing heading mismatch, make some reported headings differ significantly
        # from the calculated ones (diagonal movement at ~45° but reported as other directions)
        if headings_mismatch:
            data["Heading"] = [
                45.0,
                45.0,
                45.0 + ANGLE_DIFF_ANOMALY + 10,
                45.0,
                45.0 + ANGLE_DIFF_ANOMALY + 15,
            ]

        return pd.DataFrame(data)

    def test_no_heading_anomalies(self):
        """Test that no heading anomalies are detected when headings match"""
        vessel_data = self.create_test_vessel_data(headings_mismatch=False)
        result = detect_vessel_anomalies(vessel_data)

        # If there are no speed anomalies and no heading anomalies,
        # detect_vessel_anomalies should return None
        self.assertIsNone(result)

    def test_heading_anomalies(self):
        """Test that heading anomalies are detected when headings don't match"""
        vessel_data = self.create_test_vessel_data(headings_mismatch=True)
        result = detect_vessel_anomalies(vessel_data)

        # Should detect anomalies and return a result tuple
        self.assertIsNotNone(result)
        mmsi, ship_type, point_count, max_speed, is_anomaly, anomaly_batches = result

        self.assertEqual(mmsi, "123456789")
        self.assertTrue(is_anomaly)
        self.assertTrue(len(anomaly_batches) > 0)

    def test_heading_edge_case(self):
        """Test the case where calculated heading is near 0° and reported is near 360°"""
        # Create a special dataset for the 0/360 edge case
        now = datetime.now()
        data = {
            "MMSI": ["123456789"] * 5,
            "Ship type": ["Cargo"] * 5,
            "Timestamp": [now + timedelta(minutes=i) for i in range(5)],
            "Latitude": [0.0, 0.01, 0.02, 0.03, 0.04],
            "Longitude": [0.0, 0.0001, 0.0002, 0.0003, 0.0004],  # Almost due north
            "Heading": [0.0, 0.0, 358.0, 0.0, 359.0],  # Mix of values near 0 and 360
        }
        vessel_data = pd.DataFrame(data)

        result = detect_vessel_anomalies(vessel_data)

        # Should not detect anomalies because 0° and 360° are the same direction
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
