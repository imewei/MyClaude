"""Multi-Modal Data Fusion Framework for Materials Characterization.

This module provides Bayesian data fusion capabilities for combining measurements
from multiple characterization techniques with uncertainty weighting. It enables:

- Weighted averaging of property measurements from different techniques
- Bayesian inference for property estimation
- Uncertainty propagation and confidence intervals
- Conflict detection and resolution
- Multi-technique property consensus

Version 1.0.0

Key Features:
- Uncertainty-weighted data fusion
- Bayesian property estimation
- Confidence interval calculation
- Outlier detection and robust fusion
- Technique reliability weighting
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from scipy import stats


class FusionMethod(Enum):
    """Method for fusing multiple measurements."""
    WEIGHTED_AVERAGE = "weighted_average"      # Inverse variance weighting
    BAYESIAN = "bayesian"                      # Bayesian inference
    ROBUST = "robust"                          # Outlier-resistant fusion
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"  # ML estimation


class PropertyType(Enum):
    """Type of property being fused."""
    SCALAR = "scalar"              # Single value (e.g., Tg, modulus)
    DISTRIBUTION = "distribution"  # Distribution (e.g., particle size dist)
    SPECTRUM = "spectrum"          # Spectral data (e.g., UV-Vis)
    IMAGE = "image"               # Imaging data (e.g., AFM)


@dataclass
class Measurement:
    """Single measurement from a technique.

    Attributes:
        technique: Name of the technique
        property_name: Property being measured
        value: Measured value(s)
        uncertainty: Measurement uncertainty (std dev)
        units: Measurement units
        metadata: Additional measurement metadata
        timestamp: When measurement was taken
    """
    technique: str
    property_name: str
    value: float
    uncertainty: float
    units: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_weight(self) -> float:
        """Calculate inverse variance weight for this measurement.

        Returns:
            Weight for fusion (inverse variance)
        """
        if self.uncertainty <= 0:
            return 0.0
        return 1.0 / (self.uncertainty ** 2)


@dataclass
class FusedProperty:
    """Result of fusing multiple measurements.

    Attributes:
        property_name: Name of the property
        fused_value: Fused property value
        uncertainty: Combined uncertainty
        confidence_interval: (lower, upper) bounds at 95% confidence
        contributing_measurements: List of measurements used
        fusion_method: Method used for fusion
        quality_metrics: Fusion quality indicators
        warnings: Any warnings about the fusion
        timestamp: When fusion was performed
    """
    property_name: str
    fused_value: float
    uncertainty: float
    confidence_interval: Tuple[float, float]
    contributing_measurements: List[Measurement]
    fusion_method: FusionMethod
    quality_metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'property_name': self.property_name,
            'fused_value': self.fused_value,
            'uncertainty': self.uncertainty,
            'confidence_interval': self.confidence_interval,
            'num_measurements': len(self.contributing_measurements),
            'techniques': [m.technique for m in self.contributing_measurements],
            'fusion_method': self.fusion_method.value,
            'quality_metrics': self.quality_metrics,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat()
        }


class DataFusionFramework:
    """Framework for multi-modal data fusion in materials characterization.

    This framework combines measurements from multiple techniques using:
    - Weighted averaging with inverse variance weighting
    - Bayesian inference for property estimation
    - Uncertainty propagation
    - Outlier detection and robust fusion
    - Technique reliability assessment
    """

    def __init__(self):
        """Initialize the data fusion framework."""
        self.fusion_history: List[FusedProperty] = []
        self.technique_reliability: Dict[str, float] = {}

        # Default reliability scores (can be updated based on validation history)
        self.default_reliability = {
            # High precision techniques (±1-3%)
            'DSC': 0.95,
            'XRD': 0.95,
            'NMR': 0.95,

            # Good precision techniques (±3-5%)
            'DMA': 0.90,
            'TGA': 0.90,
            'SAXS': 0.90,
            'Ellipsometry': 0.90,

            # Moderate precision techniques (±5-10%)
            'Tensile': 0.85,
            'AFM': 0.85,
            'DLS': 0.85,
            'XPS': 0.85,

            # Lower precision techniques (±10-20%)
            'TMA': 0.80,
            'Contact Angle': 0.75,
        }

    def fuse_measurements(self, measurements: List[Measurement],
                         method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE,
                         outlier_threshold: float = 3.0) -> FusedProperty:
        """Fuse multiple measurements of the same property.

        Args:
            measurements: List of measurements to fuse
            method: Fusion method to use
            outlier_threshold: Threshold for outlier detection (in std devs)

        Returns:
            FusedProperty with combined value and uncertainty
        """
        if not measurements:
            raise ValueError("No measurements provided for fusion")

        if len(measurements) == 1:
            # Single measurement - no fusion needed
            m = measurements[0]
            return FusedProperty(
                property_name=m.property_name,
                fused_value=m.value,
                uncertainty=m.uncertainty,
                confidence_interval=self._calculate_confidence_interval(m.value, m.uncertainty),
                contributing_measurements=measurements,
                fusion_method=method,
                quality_metrics={'num_measurements': 1, 'agreement': 1.0}
            )

        # Check for outliers
        clean_measurements, outliers = self._detect_outliers(measurements, outlier_threshold)

        warnings = []
        if outliers:
            warnings.append(f"Detected {len(outliers)} outlier(s): {[m.technique for m in outliers]}")

        # Perform fusion based on method
        if method == FusionMethod.WEIGHTED_AVERAGE:
            result = self._weighted_average_fusion(clean_measurements)
        elif method == FusionMethod.BAYESIAN:
            result = self._bayesian_fusion(clean_measurements)
        elif method == FusionMethod.ROBUST:
            result = self._robust_fusion(measurements)  # Use all measurements
        elif method == FusionMethod.MAXIMUM_LIKELIHOOD:
            result = self._ml_fusion(clean_measurements)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        result.warnings = warnings
        self.fusion_history.append(result)
        return result

    def _weighted_average_fusion(self, measurements: List[Measurement]) -> FusedProperty:
        """Fuse measurements using inverse variance weighted average.

        Args:
            measurements: List of measurements

        Returns:
            FusedProperty with weighted average
        """
        # Calculate weights (inverse variance)
        weights = np.array([m.get_weight() for m in measurements])
        values = np.array([m.value for m in measurements])

        # Normalize weights
        if weights.sum() == 0:
            # All uncertainties are zero or invalid - use equal weights
            weights = np.ones(len(measurements))
        weights = weights / weights.sum()

        # Weighted average
        fused_value = np.sum(weights * values)

        # Combined uncertainty (inverse variance formula)
        total_weight = np.sum([m.get_weight() for m in measurements])
        if total_weight > 0:
            combined_uncertainty = 1.0 / np.sqrt(total_weight)
        else:
            # Fallback: average of uncertainties
            combined_uncertainty = np.mean([m.uncertainty for m in measurements])

        # Confidence interval
        ci = self._calculate_confidence_interval(fused_value, combined_uncertainty)

        # Quality metrics
        quality = self._calculate_quality_metrics(measurements, fused_value)

        return FusedProperty(
            property_name=measurements[0].property_name,
            fused_value=fused_value,
            uncertainty=combined_uncertainty,
            confidence_interval=ci,
            contributing_measurements=measurements,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            quality_metrics=quality
        )

    def _bayesian_fusion(self, measurements: List[Measurement]) -> FusedProperty:
        """Fuse measurements using Bayesian inference.

        Assumes Gaussian likelihood for each measurement and flat prior.

        Args:
            measurements: List of measurements

        Returns:
            FusedProperty with Bayesian estimate
        """
        # With flat prior, Bayesian posterior mean equals weighted average
        # Posterior precision = sum of individual precisions

        precisions = np.array([1.0 / (m.uncertainty ** 2) if m.uncertainty > 0 else 0
                              for m in measurements])
        values = np.array([m.value for m in measurements])

        # Posterior precision and mean
        posterior_precision = np.sum(precisions)
        if posterior_precision > 0:
            posterior_mean = np.sum(precisions * values) / posterior_precision
            posterior_std = 1.0 / np.sqrt(posterior_precision)
        else:
            posterior_mean = np.mean(values)
            posterior_std = np.std(values)

        # Confidence interval (95% credible interval)
        ci = self._calculate_confidence_interval(posterior_mean, posterior_std)

        # Quality metrics
        quality = self._calculate_quality_metrics(measurements, posterior_mean)
        quality['posterior_precision'] = float(posterior_precision)

        return FusedProperty(
            property_name=measurements[0].property_name,
            fused_value=posterior_mean,
            uncertainty=posterior_std,
            confidence_interval=ci,
            contributing_measurements=measurements,
            fusion_method=FusionMethod.BAYESIAN,
            quality_metrics=quality
        )

    def _robust_fusion(self, measurements: List[Measurement]) -> FusedProperty:
        """Fuse measurements using robust statistics (median and MAD).

        Args:
            measurements: List of measurements

        Returns:
            FusedProperty with robust estimate
        """
        values = np.array([m.value for m in measurements])

        # Median as robust central estimate
        fused_value = float(np.median(values))

        # MAD (Median Absolute Deviation) as robust scale estimate
        mad = float(np.median(np.abs(values - fused_value)))
        # Convert MAD to std estimate (for normal distribution)
        robust_std = 1.4826 * mad

        # If MAD is zero, fall back to IQR
        if robust_std == 0:
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            robust_std = iqr / 1.349  # IQR to std for normal

        # If still zero, use min uncertainty
        if robust_std == 0:
            robust_std = min(m.uncertainty for m in measurements)

        # Confidence interval
        ci = self._calculate_confidence_interval(fused_value, robust_std)

        # Quality metrics
        quality = self._calculate_quality_metrics(measurements, fused_value)
        quality['mad'] = mad

        return FusedProperty(
            property_name=measurements[0].property_name,
            fused_value=fused_value,
            uncertainty=robust_std,
            confidence_interval=ci,
            contributing_measurements=measurements,
            fusion_method=FusionMethod.ROBUST,
            quality_metrics=quality
        )

    def _ml_fusion(self, measurements: List[Measurement]) -> FusedProperty:
        """Fuse measurements using maximum likelihood estimation.

        Assumes Gaussian likelihood - equivalent to weighted average.

        Args:
            measurements: List of measurements

        Returns:
            FusedProperty with ML estimate
        """
        # For Gaussian measurements, MLE is the weighted average
        return self._weighted_average_fusion(measurements)

    def _detect_outliers(self, measurements: List[Measurement],
                        threshold: float) -> Tuple[List[Measurement], List[Measurement]]:
        """Detect outliers using modified Z-score.

        Args:
            measurements: List of measurements
            threshold: Threshold in standard deviations

        Returns:
            Tuple of (clean_measurements, outliers)
        """
        if len(measurements) < 3:
            return measurements, []

        values = np.array([m.value for m in measurements])

        # Use median and MAD for robust outlier detection
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad == 0:
            # No dispersion - no outliers
            return measurements, []

        # Modified Z-score
        modified_z_scores = 0.6745 * (values - median) / mad

        clean = []
        outliers = []
        for i, m in enumerate(measurements):
            if abs(modified_z_scores[i]) < threshold:
                clean.append(m)
            else:
                outliers.append(m)

        # Keep at least 2 measurements
        if len(clean) < 2:
            return measurements, []

        return clean, outliers

    def _calculate_confidence_interval(self, value: float, uncertainty: float,
                                      confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval.

        Args:
            value: Central value
            uncertainty: Standard deviation
            confidence: Confidence level (default 95%)

        Returns:
            Tuple of (lower, upper) bounds
        """
        # For normal distribution
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * uncertainty
        return (value - margin, value + margin)

    def _calculate_quality_metrics(self, measurements: List[Measurement],
                                   fused_value: float) -> Dict[str, float]:
        """Calculate quality metrics for the fusion.

        Args:
            measurements: List of measurements
            fused_value: Fused value

        Returns:
            Dictionary of quality metrics
        """
        values = np.array([m.value for m in measurements])

        # Agreement metric: inverse of coefficient of variation
        mean_val = np.mean(values)
        std_val = np.std(values)
        if mean_val != 0:
            cv = std_val / abs(mean_val)
            agreement = 1.0 / (1.0 + cv)
        else:
            agreement = 1.0 if std_val == 0 else 0.5

        # Residuals
        residuals = values - fused_value
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        max_residual = float(np.max(np.abs(residuals)))

        # Weighted chi-squared (goodness of fit)
        chi_squared = 0.0
        for m in measurements:
            if m.uncertainty > 0:
                chi_squared += ((m.value - fused_value) / m.uncertainty) ** 2

        return {
            'num_measurements': len(measurements),
            'agreement': float(agreement),
            'coefficient_of_variation': float(cv) if mean_val != 0 else 0.0,
            'rmse': rmse,
            'max_residual': max_residual,
            'chi_squared': float(chi_squared),
            'reduced_chi_squared': float(chi_squared / len(measurements)) if len(measurements) > 1 else 0.0
        }

    def fuse_property_set(self, measurement_dict: Dict[str, List[Measurement]],
                         method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE) -> Dict[str, FusedProperty]:
        """Fuse multiple properties at once.

        Args:
            measurement_dict: Dictionary mapping property names to measurement lists
            method: Fusion method to use

        Returns:
            Dictionary mapping property names to fused properties
        """
        results = {}
        for property_name, measurements in measurement_dict.items():
            try:
                fused = self.fuse_measurements(measurements, method=method)
                results[property_name] = fused
            except Exception as e:
                print(f"Warning: Failed to fuse {property_name}: {e}")

        return results

    def get_fusion_summary(self, property_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics of fusion history.

        Args:
            property_name: Filter by property name (optional)

        Returns:
            Dictionary of summary statistics
        """
        history = self.fusion_history
        if property_name:
            history = [f for f in history if f.property_name == property_name]

        if not history:
            return {'num_fusions': 0}

        # Collect statistics
        num_measurements = [len(f.contributing_measurements) for f in history]
        agreements = [f.quality_metrics.get('agreement', 0) for f in history]
        uncertainties = [f.uncertainty for f in history]

        return {
            'num_fusions': len(history),
            'avg_measurements_per_fusion': float(np.mean(num_measurements)),
            'avg_agreement': float(np.mean(agreements)),
            'avg_uncertainty': float(np.mean(uncertainties)),
            'properties_fused': list(set(f.property_name for f in history))
        }

    def generate_fusion_report(self, fused_property: FusedProperty) -> str:
        """Generate a detailed report for a fused property.

        Args:
            fused_property: FusedProperty to report

        Returns:
            Formatted report string
        """
        lines = ["=" * 80]
        lines.append(f"DATA FUSION REPORT: {fused_property.property_name}")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {fused_property.timestamp.isoformat()}")
        lines.append("")

        lines.append("FUSED VALUE:")
        lines.append(f"  Value: {fused_property.fused_value:.4f} ± {fused_property.uncertainty:.4f}")
        lines.append(f"  95% CI: [{fused_property.confidence_interval[0]:.4f}, {fused_property.confidence_interval[1]:.4f}]")
        lines.append(f"  Method: {fused_property.fusion_method.value}")
        lines.append("")

        lines.append(f"CONTRIBUTING MEASUREMENTS ({len(fused_property.contributing_measurements)}):")
        for m in fused_property.contributing_measurements:
            lines.append(f"  {m.technique}: {m.value:.4f} ± {m.uncertainty:.4f}")
        lines.append("")

        lines.append("QUALITY METRICS:")
        for key, value in fused_property.quality_metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        lines.append("")

        if fused_property.warnings:
            lines.append("WARNINGS:")
            for warning in fused_property.warnings:
                lines.append(f"  ⚠ {warning}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Initialize fusion framework
    fusion = DataFusionFramework()

    # Create example measurements of glass transition temperature (Tg)
    measurements = [
        Measurement(technique="DSC", property_name="Tg", value=105.2, uncertainty=0.5, units="°C"),
        Measurement(technique="DMA", property_name="Tg", value=107.8, uncertainty=1.0, units="°C"),
        Measurement(technique="TMA", property_name="Tg", value=106.5, uncertainty=1.5, units="°C"),
    ]

    print("Example: Fusing Glass Transition Temperature Measurements")
    print("=" * 80)
    print("\nInput Measurements:")
    for m in measurements:
        print(f"  {m.technique}: {m.value:.1f} ± {m.uncertainty:.1f} {m.units}")

    # Fuse using weighted average
    print("\n\n1. Weighted Average Fusion:")
    print("-" * 80)
    result_wa = fusion.fuse_measurements(measurements, method=FusionMethod.WEIGHTED_AVERAGE)
    print(fusion.generate_fusion_report(result_wa))

    # Fuse using Bayesian inference
    print("\n\n2. Bayesian Fusion:")
    print("-" * 80)
    result_bayes = fusion.fuse_measurements(measurements, method=FusionMethod.BAYESIAN)
    print(fusion.generate_fusion_report(result_bayes))

    # Fuse using robust method
    print("\n\n3. Robust Fusion:")
    print("-" * 80)
    result_robust = fusion.fuse_measurements(measurements, method=FusionMethod.ROBUST)
    print(fusion.generate_fusion_report(result_robust))

    # Add an outlier and test outlier detection
    measurements_with_outlier = measurements + [
        Measurement(technique="Outlier", property_name="Tg", value=95.0, uncertainty=2.0, units="°C")
    ]

    print("\n\n4. Fusion with Outlier Detection:")
    print("-" * 80)
    result_outlier = fusion.fuse_measurements(measurements_with_outlier, method=FusionMethod.WEIGHTED_AVERAGE)
    print(fusion.generate_fusion_report(result_outlier))

    # Summary statistics
    print("\n\nFUSION SUMMARY:")
    print("=" * 80)
    summary = fusion.get_fusion_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
