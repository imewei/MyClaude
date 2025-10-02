"""Unit Tests for Multi-Modal Data Fusion Framework

Tests the DataFusionFramework's core functionality including:
- Weighted average fusion
- Bayesian fusion
- Robust fusion
- Outlier detection
- Uncertainty propagation
- Quality metrics

Version 1.0.0
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fusion import (
    DataFusionFramework,
    Measurement,
    FusionMethod,
    FusedProperty,
    PropertyType
)


class TestMeasurement(unittest.TestCase):
    """Test the Measurement dataclass."""

    def test_measurement_creation(self):
        """Test basic measurement creation."""
        m = Measurement(
            technique="DSC",
            property_name="Tg",
            value=105.0,
            uncertainty=0.5,
            units="°C"
        )
        self.assertEqual(m.technique, "DSC")
        self.assertEqual(m.property_name, "Tg")
        self.assertEqual(m.value, 105.0)
        self.assertEqual(m.uncertainty, 0.5)
        self.assertEqual(m.units, "°C")

    def test_measurement_weight(self):
        """Test inverse variance weight calculation."""
        m = Measurement(
            technique="DSC",
            property_name="Tg",
            value=105.0,
            uncertainty=2.0,
            units="°C"
        )
        # Weight should be 1/variance = 1/4 = 0.25
        self.assertEqual(m.get_weight(), 0.25)

    def test_measurement_zero_uncertainty(self):
        """Test handling of zero uncertainty."""
        m = Measurement(
            technique="DSC",
            property_name="Tg",
            value=105.0,
            uncertainty=0.0,
            units="°C"
        )
        # Should return 0 weight to avoid division by zero
        self.assertEqual(m.get_weight(), 0.0)


class TestWeightedAverageFusion(unittest.TestCase):
    """Test weighted average fusion method."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = DataFusionFramework()

    def test_single_measurement(self):
        """Test fusion with single measurement."""
        measurements = [
            Measurement("DSC", "Tg", 105.0, 0.5, "°C")
        ]

        result = self.fusion.fuse_measurements(measurements, method=FusionMethod.WEIGHTED_AVERAGE)

        self.assertEqual(result.fused_value, 105.0)
        self.assertEqual(result.uncertainty, 0.5)
        self.assertEqual(len(result.contributing_measurements), 1)

    def test_two_equal_measurements(self):
        """Test fusion of two measurements with equal values and uncertainties."""
        measurements = [
            Measurement("DSC", "Tg", 100.0, 1.0, "°C"),
            Measurement("DMA", "Tg", 100.0, 1.0, "°C")
        ]

        result = self.fusion.fuse_measurements(measurements, method=FusionMethod.WEIGHTED_AVERAGE)

        # Should give 100.0
        self.assertAlmostEqual(result.fused_value, 100.0, places=5)
        # Uncertainty should be reduced by sqrt(2)
        self.assertAlmostEqual(result.uncertainty, 1.0 / np.sqrt(2), places=5)

    def test_different_uncertainties(self):
        """Test that measurements with lower uncertainty get more weight."""
        measurements = [
            Measurement("High Precision", "value", 10.0, 0.1, "units"),
            Measurement("Low Precision", "value", 20.0, 10.0, "units")
        ]

        result = self.fusion.fuse_measurements(measurements, method=FusionMethod.WEIGHTED_AVERAGE)

        # Result should be close to the high-precision measurement
        self.assertLess(abs(result.fused_value - 10.0), abs(result.fused_value - 20.0))

    def test_three_measurements(self):
        """Test fusion of three measurements."""
        measurements = [
            Measurement("DSC", "Tg", 105.2, 0.5, "°C"),
            Measurement("DMA", "Tg", 107.8, 1.0, "°C"),
            Measurement("TMA", "Tg", 106.5, 1.5, "°C")
        ]

        result = self.fusion.fuse_measurements(measurements, method=FusionMethod.WEIGHTED_AVERAGE)

        # Fused value should be between min and max
        values = [m.value for m in measurements]
        self.assertGreaterEqual(result.fused_value, min(values))
        self.assertLessEqual(result.fused_value, max(values))

        # Uncertainty should be less than all individual uncertainties
        uncertainties = [m.uncertainty for m in measurements]
        self.assertLess(result.uncertainty, min(uncertainties))


class TestBayesianFusion(unittest.TestCase):
    """Test Bayesian fusion method."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = DataFusionFramework()

    def test_bayesian_identical_to_weighted(self):
        """Test that Bayesian fusion with flat prior equals weighted average."""
        measurements = [
            Measurement("DSC", "Tg", 105.0, 1.0, "°C"),
            Measurement("DMA", "Tg", 107.0, 1.0, "°C")
        ]

        result_weighted = self.fusion.fuse_measurements(measurements, method=FusionMethod.WEIGHTED_AVERAGE)
        result_bayesian = self.fusion.fuse_measurements(measurements, method=FusionMethod.BAYESIAN)

        # Should be nearly identical
        self.assertAlmostEqual(result_weighted.fused_value, result_bayesian.fused_value, places=5)
        self.assertAlmostEqual(result_weighted.uncertainty, result_bayesian.uncertainty, places=5)

    def test_bayesian_uncertainty_reduction(self):
        """Test that Bayesian fusion reduces uncertainty correctly."""
        measurements = [
            Measurement("Technique1", "value", 10.0, 2.0, "units"),
            Measurement("Technique2", "value", 10.0, 2.0, "units")
        ]

        result = self.fusion.fuse_measurements(measurements, method=FusionMethod.BAYESIAN)

        # Posterior precision = 1/4 + 1/4 = 1/2
        # Posterior std = sqrt(2) ≈ 1.414
        self.assertAlmostEqual(result.uncertainty, np.sqrt(2.0), places=3)


class TestRobustFusion(unittest.TestCase):
    """Test robust fusion method."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = DataFusionFramework()

    def test_robust_median(self):
        """Test that robust fusion returns median."""
        measurements = [
            Measurement("T1", "value", 10.0, 1.0, "units"),
            Measurement("T2", "value", 20.0, 1.0, "units"),
            Measurement("T3", "value", 30.0, 1.0, "units"),
        ]

        result = self.fusion.fuse_measurements(measurements, method=FusionMethod.ROBUST)

        # Should return median value (20.0)
        self.assertEqual(result.fused_value, 20.0)

    def test_robust_with_outlier(self):
        """Test that robust fusion handles outliers well."""
        measurements = [
            Measurement("T1", "value", 10.0, 1.0, "units"),
            Measurement("T2", "value", 10.2, 1.0, "units"),
            Measurement("T3", "value", 10.1, 1.0, "units"),
            Measurement("Outlier", "value", 100.0, 1.0, "units"),  # Outlier
        ]

        result_robust = self.fusion.fuse_measurements(measurements, method=FusionMethod.ROBUST)

        # Robust fusion should use median, which is naturally resistant to outliers
        # The median of [10.0, 10.1, 10.2, 100.0] is between 10.1 and 10.2
        self.assertGreater(result_robust.fused_value, 10.0)
        self.assertLess(result_robust.fused_value, 11.0)  # Should not be pulled by outlier


class TestOutlierDetection(unittest.TestCase):
    """Test outlier detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = DataFusionFramework()

    def test_no_outliers(self):
        """Test that normal measurements are not flagged as outliers."""
        measurements = [
            Measurement("T1", "value", 10.0, 1.0, "units"),
            Measurement("T2", "value", 11.0, 1.0, "units"),
            Measurement("T3", "value", 10.5, 1.0, "units"),
        ]

        result = self.fusion.fuse_measurements(measurements, method=FusionMethod.WEIGHTED_AVERAGE)

        # No warnings about outliers
        self.assertEqual(len(result.warnings), 0)

    def test_outlier_detected(self):
        """Test that outliers are detected and reported."""
        measurements = [
            Measurement("T1", "value", 10.0, 1.0, "units"),
            Measurement("T2", "value", 11.0, 1.0, "units"),
            Measurement("T3", "value", 10.5, 1.0, "units"),
            Measurement("Outlier", "value", 100.0, 1.0, "units"),
        ]

        result = self.fusion.fuse_measurements(measurements, method=FusionMethod.WEIGHTED_AVERAGE, outlier_threshold=3.0)

        # Should have warning about outlier
        self.assertGreater(len(result.warnings), 0)
        self.assertTrue(any("outlier" in w.lower() for w in result.warnings))

    def test_outlier_threshold(self):
        """Test different outlier thresholds."""
        measurements = [
            Measurement("T1", "value", 10.0, 1.0, "units"),
            Measurement("T2", "value", 11.0, 1.0, "units"),
            Measurement("T3", "value", 15.0, 1.0, "units"),  # Borderline
        ]

        # Strict threshold should flag it
        result_strict = self.fusion.fuse_measurements(measurements, outlier_threshold=2.0)

        # Lenient threshold should not
        result_lenient = self.fusion.fuse_measurements(measurements, outlier_threshold=5.0)

        # Strict should have more warnings
        self.assertGreaterEqual(len(result_strict.warnings), len(result_lenient.warnings))


class TestQualityMetrics(unittest.TestCase):
    """Test quality metrics calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = DataFusionFramework()

    def test_perfect_agreement(self):
        """Test quality metrics for perfect agreement."""
        measurements = [
            Measurement("T1", "value", 10.0, 0.1, "units"),
            Measurement("T2", "value", 10.0, 0.1, "units"),
            Measurement("T3", "value", 10.0, 0.1, "units"),
        ]

        result = self.fusion.fuse_measurements(measurements)

        # Agreement should be very high
        self.assertGreater(result.quality_metrics['agreement'], 0.95)
        # CV should be very low
        self.assertLess(result.quality_metrics['coefficient_of_variation'], 0.01)

    def test_poor_agreement(self):
        """Test quality metrics for poor agreement."""
        measurements = [
            Measurement("T1", "value", 10.0, 1.0, "units"),
            Measurement("T2", "value", 20.0, 1.0, "units"),
            Measurement("T3", "value", 30.0, 1.0, "units"),
        ]

        result = self.fusion.fuse_measurements(measurements)

        # Agreement should be lower
        self.assertLess(result.quality_metrics['agreement'], 0.8)
        # CV should be higher
        self.assertGreater(result.quality_metrics['coefficient_of_variation'], 0.2)

    def test_rmse_calculation(self):
        """Test RMSE quality metric."""
        measurements = [
            Measurement("T1", "value", 9.0, 1.0, "units"),
            Measurement("T2", "value", 11.0, 1.0, "units"),
        ]

        result = self.fusion.fuse_measurements(measurements)

        # Fused value should be 10.0, so RMSE should be 1.0
        self.assertAlmostEqual(result.quality_metrics['rmse'], 1.0, places=2)


class TestConfidenceIntervals(unittest.TestCase):
    """Test confidence interval calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = DataFusionFramework()

    def test_confidence_interval_width(self):
        """Test that confidence interval has correct width."""
        measurements = [
            Measurement("T1", "value", 10.0, 1.0, "units")
        ]

        result = self.fusion.fuse_measurements(measurements)

        ci_lower, ci_upper = result.confidence_interval

        # For 95% CI, width should be approximately 1.96 * 2 * uncertainty
        width = ci_upper - ci_lower
        expected_width = 2 * 1.96 * result.uncertainty
        self.assertAlmostEqual(width, expected_width, places=2)

    def test_confidence_interval_contains_value(self):
        """Test that fused value is within confidence interval."""
        measurements = [
            Measurement("T1", "value", 10.0, 1.0, "units"),
            Measurement("T2", "value", 12.0, 1.0, "units"),
        ]

        result = self.fusion.fuse_measurements(measurements)

        ci_lower, ci_upper = result.confidence_interval

        # Fused value should be within CI
        self.assertGreaterEqual(result.fused_value, ci_lower)
        self.assertLessEqual(result.fused_value, ci_upper)


class TestFusionHistory(unittest.TestCase):
    """Test fusion history tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = DataFusionFramework()

    def test_history_tracking(self):
        """Test that fusion history is tracked."""
        measurements = [
            Measurement("T1", "value", 10.0, 1.0, "units"),
            Measurement("T2", "value", 11.0, 1.0, "units"),
        ]

        # Perform multiple fusions
        self.fusion.fuse_measurements(measurements)
        self.fusion.fuse_measurements(measurements)

        summary = self.fusion.get_fusion_summary()

        # Should have 2 fusions in history
        self.assertEqual(summary['num_fusions'], 2)

    def test_fusion_summary_statistics(self):
        """Test fusion summary statistics."""
        measurements = [
            Measurement("T1", "Tg", 10.0, 1.0, "°C"),
            Measurement("T2", "Tg", 11.0, 1.0, "°C"),
        ]

        self.fusion.fuse_measurements(measurements)
        self.fusion.fuse_measurements(measurements)

        summary = self.fusion.get_fusion_summary("Tg")

        # Should have statistics
        self.assertIn('avg_measurements_per_fusion', summary)
        self.assertIn('avg_agreement', summary)
        self.assertIn('avg_uncertainty', summary)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = DataFusionFramework()

    def test_empty_measurements(self):
        """Test that empty measurement list raises error."""
        with self.assertRaises(ValueError):
            self.fusion.fuse_measurements([])

    def test_all_zero_uncertainties(self):
        """Test handling of all zero uncertainties."""
        measurements = [
            Measurement("T1", "value", 10.0, 0.0, "units"),
            Measurement("T2", "value", 11.0, 0.0, "units"),
        ]

        result = self.fusion.fuse_measurements(measurements)

        # Should still produce a result
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.fused_value, 10.0)
        self.assertLessEqual(result.fused_value, 11.0)

    def test_large_number_of_measurements(self):
        """Test fusion with many measurements."""
        measurements = [
            Measurement(f"T{i}", "value", 10.0 + np.random.normal(0, 0.1), 0.5, "units")
            for i in range(100)
        ]

        result = self.fusion.fuse_measurements(measurements)

        # Should have very small uncertainty with 100 measurements
        self.assertLess(result.uncertainty, 0.1)


def run_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMeasurement))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightedAverageFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestBayesianFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestOutlierDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestQualityMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceIntervals))
    suite.addTests(loader.loadTestsFromTestCase(TestFusionHistory))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
