"""Cross-Validation Framework for Materials Characterization Agents.

This framework provides centralized orchestration of cross-validation between
different characterization techniques. It ensures data consistency, identifies
discrepancies, and provides recommendations for resolving conflicts.

Version 1.0.0

Key Features:
- Automatic validation pair discovery
- Standardized validation interface
- Conflict resolution recommendations
- Validation result caching
- Statistical agreement metrics
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np


class ValidationStatus(Enum):
    """Status of cross-validation result."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class AgreementLevel(Enum):
    """Level of agreement between techniques."""
    STRONG = "strong"        # <5% difference
    MODERATE = "moderate"    # 5-15% difference
    WEAK = "weak"           # 15-30% difference
    NONE = "none"           # >30% difference


@dataclass
class ValidationPair:
    """Defines a pair of techniques that can cross-validate."""
    technique_1: str
    technique_2: str
    property_measured: str
    validation_method: Callable
    description: str
    expected_agreement: str = "good"
    tolerance_percent: float = 10.0

    def get_key(self) -> str:
        """Get unique key for this validation pair."""
        # Sort to ensure (A, B) == (B, A)
        t1, t2 = sorted([self.technique_1, self.technique_2])
        return f"{t1}::{t2}::{self.property_measured}"


@dataclass
class ValidationResult:
    """Result of a cross-validation."""
    pair: ValidationPair
    status: ValidationStatus
    agreement_level: AgreementLevel
    values: Dict[str, Any]
    differences: Dict[str, float]
    interpretation: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'validation_pair': {
                'technique_1': self.pair.technique_1,
                'technique_2': self.pair.technique_2,
                'property': self.pair.property_measured
            },
            'status': self.status.value,
            'agreement_level': self.agreement_level.value,
            'values': self.values,
            'differences': self.differences,
            'interpretation': self.interpretation,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class CrossValidationFramework:
    """Central orchestrator for cross-validation between agents.

    This framework:
    - Registers validation pairs from all agents
    - Executes cross-validations
    - Tracks validation history
    - Provides conflict resolution
    - Generates validation reports
    """

    def __init__(self):
        """Initialize the cross-validation framework."""
        self.validation_pairs: Dict[str, ValidationPair] = {}
        self.validation_history: List[ValidationResult] = []
        self.agent_registry: Dict[str, Any] = {}

        # Statistics tracking
        self.stats = {
            'total_validations': 0,
            'excellent_count': 0,
            'good_count': 0,
            'acceptable_count': 0,
            'poor_count': 0,
            'failed_count': 0
        }

    def register_validation_pair(self, pair: ValidationPair) -> None:
        """Register a new validation pair.

        Args:
            pair: ValidationPair defining the cross-validation
        """
        key = pair.get_key()
        if key in self.validation_pairs:
            print(f"Warning: Overwriting existing validation pair: {key}")
        self.validation_pairs[key] = pair

    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """Register an agent with the framework.

        Args:
            agent_name: Name of the agent
            agent_instance: Instance of the agent
        """
        self.agent_registry[agent_name] = agent_instance

    def get_validation_pair(self, technique_1: str, technique_2: str,
                           property_name: str) -> Optional[ValidationPair]:
        """Get a validation pair by technique names and property.

        Args:
            technique_1: First technique name
            technique_2: Second technique name
            property_name: Property being measured

        Returns:
            ValidationPair if found, None otherwise
        """
        # Try both orderings
        t1, t2 = sorted([technique_1, technique_2])
        key = f"{t1}::{t2}::{property_name}"
        return self.validation_pairs.get(key)

    def validate(self, technique_1: str, result_1: Dict[str, Any],
                technique_2: str, result_2: Dict[str, Any],
                property_name: str) -> Optional[ValidationResult]:
        """Execute cross-validation between two technique results.

        Args:
            technique_1: Name of first technique
            result_1: Result dictionary from first technique
            technique_2: Name of second technique
            result_2: Result dictionary from second technique
            property_name: Property to validate

        Returns:
            ValidationResult if validation pair exists, None otherwise
        """
        pair = self.get_validation_pair(technique_1, technique_2, property_name)
        if pair is None:
            return None

        # Execute the validation method
        try:
            validation_output = pair.validation_method(result_1, result_2)

            # Parse the validation output
            status = self._determine_status(validation_output)
            agreement = self._determine_agreement(validation_output)

            result = ValidationResult(
                pair=pair,
                status=status,
                agreement_level=agreement,
                values=validation_output.get('values', {}),
                differences=validation_output.get('differences', {}),
                interpretation=validation_output.get('interpretation', ''),
                recommendations=self._generate_recommendations(validation_output, status),
                metadata={'raw_output': validation_output}
            )

            # Update statistics
            self._update_statistics(status)
            self.validation_history.append(result)

            return result

        except Exception as e:
            print(f"Error during validation: {e}")
            return None

    def _determine_status(self, validation_output: Dict[str, Any]) -> ValidationStatus:
        """Determine validation status from output.

        Args:
            validation_output: Output from validation method

        Returns:
            ValidationStatus enum
        """
        agreement = validation_output.get('agreement', 'poor').lower()

        if agreement == 'excellent':
            return ValidationStatus.EXCELLENT
        elif agreement == 'good':
            return ValidationStatus.GOOD
        elif agreement == 'acceptable':
            return ValidationStatus.ACCEPTABLE
        elif agreement == 'poor':
            return ValidationStatus.POOR
        else:
            return ValidationStatus.FAILED

    def _determine_agreement(self, validation_output: Dict[str, Any]) -> AgreementLevel:
        """Determine agreement level from output.

        Args:
            validation_output: Output from validation method

        Returns:
            AgreementLevel enum
        """
        # Check for relative difference
        rel_diff = validation_output.get('relative_difference_percent',
                                        validation_output.get('difference', 100))

        if rel_diff < 5:
            return AgreementLevel.STRONG
        elif rel_diff < 15:
            return AgreementLevel.MODERATE
        elif rel_diff < 30:
            return AgreementLevel.WEAK
        else:
            return AgreementLevel.NONE

    def _generate_recommendations(self, validation_output: Dict[str, Any],
                                 status: ValidationStatus) -> List[str]:
        """Generate recommendations based on validation result.

        Args:
            validation_output: Output from validation method
            status: Validation status

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if status == ValidationStatus.EXCELLENT:
            recommendations.append("Results show excellent agreement. No action needed.")
        elif status == ValidationStatus.GOOD:
            recommendations.append("Results show good agreement. Continue with analysis.")
        elif status == ValidationStatus.ACCEPTABLE:
            recommendations.append("Results show acceptable agreement. Consider additional measurements for confirmation.")
        elif status == ValidationStatus.POOR:
            recommendations.append("Poor agreement detected. Review measurement conditions and sample preparation.")
            recommendations.append("Check for systematic errors or sample heterogeneity.")
        else:
            recommendations.append("Validation failed. Repeat measurements or consult domain expert.")

        # Add specific recommendations from validation output
        if 'recommendation' in validation_output:
            recommendations.append(validation_output['recommendation'])

        return recommendations

    def _update_statistics(self, status: ValidationStatus) -> None:
        """Update validation statistics.

        Args:
            status: ValidationStatus to record
        """
        self.stats['total_validations'] += 1

        if status == ValidationStatus.EXCELLENT:
            self.stats['excellent_count'] += 1
        elif status == ValidationStatus.GOOD:
            self.stats['good_count'] += 1
        elif status == ValidationStatus.ACCEPTABLE:
            self.stats['acceptable_count'] += 1
        elif status == ValidationStatus.POOR:
            self.stats['poor_count'] += 1
        else:
            self.stats['failed_count'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics.

        Returns:
            Dictionary of statistics
        """
        total = self.stats['total_validations']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'excellent_percent': (self.stats['excellent_count'] / total) * 100,
            'good_percent': (self.stats['good_count'] / total) * 100,
            'acceptable_percent': (self.stats['acceptable_count'] / total) * 100,
            'poor_percent': (self.stats['poor_count'] / total) * 100,
            'failed_percent': (self.stats['failed_count'] / total) * 100,
            'success_rate': ((self.stats['excellent_count'] + self.stats['good_count']) / total) * 100
        }

    def get_validation_pairs_for_technique(self, technique: str) -> List[ValidationPair]:
        """Get all validation pairs involving a specific technique.

        Args:
            technique: Technique name

        Returns:
            List of ValidationPair objects
        """
        return [
            pair for pair in self.validation_pairs.values()
            if technique.lower() in [pair.technique_1.lower(), pair.technique_2.lower()]
        ]

    def get_validation_history(self, technique: Optional[str] = None,
                              min_status: Optional[ValidationStatus] = None) -> List[ValidationResult]:
        """Get validation history with optional filtering.

        Args:
            technique: Filter by technique name (optional)
            min_status: Minimum validation status (optional)

        Returns:
            List of ValidationResult objects
        """
        results = self.validation_history

        if technique:
            results = [
                r for r in results
                if technique.lower() in [r.pair.technique_1.lower(), r.pair.technique_2.lower()]
            ]

        if min_status:
            status_order = {
                ValidationStatus.EXCELLENT: 4,
                ValidationStatus.GOOD: 3,
                ValidationStatus.ACCEPTABLE: 2,
                ValidationStatus.POOR: 1,
                ValidationStatus.FAILED: 0
            }
            min_value = status_order[min_status]
            results = [r for r in results if status_order[r.status] >= min_value]

        return results

    def generate_report(self, technique: Optional[str] = None) -> str:
        """Generate a validation report.

        Args:
            technique: Generate report for specific technique (optional)

        Returns:
            Formatted report string
        """
        stats = self.get_statistics()

        report = ["=" * 80]
        report.append("CROSS-VALIDATION FRAMEWORK REPORT")
        report.append("=" * 80)
        report.append("")

        if technique:
            report.append(f"Technique: {technique}")
            pairs = self.get_validation_pairs_for_technique(technique)
            report.append(f"Available validation pairs: {len(pairs)}")
            report.append("")

        report.append("Overall Statistics:")
        report.append(f"  Total validations: {stats['total_validations']}")
        report.append(f"  Excellent: {stats['excellent_count']} ({stats.get('excellent_percent', 0):.1f}%)")
        report.append(f"  Good: {stats['good_count']} ({stats.get('good_percent', 0):.1f}%)")
        report.append(f"  Acceptable: {stats['acceptable_count']} ({stats.get('acceptable_percent', 0):.1f}%)")
        report.append(f"  Poor: {stats['poor_count']} ({stats.get('poor_percent', 0):.1f}%)")
        report.append(f"  Failed: {stats['failed_count']} ({stats.get('failed_percent', 0):.1f}%)")
        report.append(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
        report.append("")

        # Recent validations
        recent = self.validation_history[-10:]
        if recent:
            report.append("Recent Validations:")
            for r in recent:
                report.append(f"  {r.pair.technique_1} ↔ {r.pair.technique_2} ({r.pair.property_measured}): {r.status.value}")

        report.append("=" * 80)
        return "\n".join(report)

    def list_registered_pairs(self) -> List[str]:
        """List all registered validation pairs.

        Returns:
            List of validation pair descriptions
        """
        return [
            f"{pair.technique_1} ↔ {pair.technique_2}: {pair.property_measured} ({pair.description})"
            for pair in self.validation_pairs.values()
        ]


# Global framework instance
_framework_instance = None


def get_framework() -> CrossValidationFramework:
    """Get the global cross-validation framework instance.

    Returns:
        CrossValidationFramework singleton
    """
    global _framework_instance
    if _framework_instance is None:
        _framework_instance = CrossValidationFramework()
    return _framework_instance


# Example validation method
def example_particle_size_validation(result_1: Dict[str, Any],
                                     result_2: Dict[str, Any]) -> Dict[str, Any]:
    """Example: Validate particle size between SAXS and DLS.

    Args:
        result_1: SAXS result
        result_2: DLS result

    Returns:
        Validation output dictionary
    """
    saxs_size = result_1.get('physical_properties', {}).get('particle_size_nm', 0)
    dls_size = result_2.get('hydrodynamic_diameter_nm', 0)

    difference = abs(saxs_size - dls_size)
    relative_diff = (difference / saxs_size * 100) if saxs_size > 0 else 100

    if relative_diff < 10:
        agreement = 'excellent'
    elif relative_diff < 20:
        agreement = 'good'
    elif relative_diff < 30:
        agreement = 'acceptable'
    else:
        agreement = 'poor'

    return {
        'values': {
            'saxs_size_nm': saxs_size,
            'dls_size_nm': dls_size
        },
        'differences': {
            'absolute_nm': difference,
            'relative_percent': relative_diff
        },
        'agreement': agreement,
        'interpretation': (
            f"SAXS ({saxs_size:.1f} nm) vs DLS ({dls_size:.1f} nm): "
            f"{relative_diff:.1f}% difference. "
            f"DLS includes hydrodynamic layer; SAXS measures core."
        ),
        'recommendation': (
            'Excellent agreement' if relative_diff < 10
            else 'Difference suggests solvation layer or aggregation'
        )
    }


if __name__ == "__main__":
    # Example usage
    framework = get_framework()

    # Register a validation pair
    pair = ValidationPair(
        technique_1="SAXS",
        technique_2="DLS",
        property_measured="particle_size",
        validation_method=example_particle_size_validation,
        description="Particle size comparison: SAXS (structure) vs DLS (hydrodynamic)",
        tolerance_percent=20.0
    )
    framework.register_validation_pair(pair)

    # Example validation
    saxs_result = {'physical_properties': {'particle_size_nm': 50.0}}
    dls_result = {'hydrodynamic_diameter_nm': 55.0}

    result = framework.validate("SAXS", saxs_result, "DLS", dls_result, "particle_size")

    if result:
        print(f"Validation Status: {result.status.value}")
        print(f"Agreement Level: {result.agreement_level.value}")
        print(f"Interpretation: {result.interpretation}")
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")

    # Generate report
    print("\n" + framework.generate_report())
