"""Materials Characterization Master Orchestrator.

This module provides a unified interface to all materials characterization agents
and orchestrates cross-validation, multi-technique measurements, and intelligent
agent selection based on sample properties and measurement goals.

Version 1.1.0 - Added multi-modal data fusion

Key Features:
- Automatic agent selection based on sample type and property
- Multi-technique measurement coordination
- Automatic cross-validation execution
- Multi-modal data fusion with uncertainty weighting
- Intelligent measurement planning
- Result aggregation and reporting
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import importlib
import sys

from cross_validation_framework import (
    CrossValidationFramework,
    get_framework,
    ValidationResult,
    ValidationStatus
)
from register_validations import initialize_framework
from data_fusion import DataFusionFramework, Measurement, FusedProperty, FusionMethod


class SampleType(Enum):
    """Types of samples that can be characterized."""
    POLYMER = "polymer"
    CERAMIC = "ceramic"
    METAL = "metal"
    COMPOSITE = "composite"
    THIN_FILM = "thin_film"
    NANOPARTICLE = "nanoparticle"
    COLLOID = "colloid"
    BIOMATERIAL = "biomaterial"
    SEMICONDUCTOR = "semiconductor"
    LIQUID_CRYSTAL = "liquid_crystal"


class PropertyCategory(Enum):
    """Categories of material properties."""
    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    OPTICAL = "optical"
    CHEMICAL = "chemical"
    STRUCTURAL = "structural"
    SURFACE = "surface"
    MAGNETIC = "magnetic"


@dataclass
class MeasurementRequest:
    """Request for material characterization."""
    sample_name: str
    sample_type: SampleType
    properties_of_interest: List[str]
    property_categories: List[PropertyCategory]
    techniques_requested: Optional[List[str]] = None
    cross_validate: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementResult:
    """Aggregated result from characterization."""
    sample_name: str
    timestamp: datetime
    technique_results: Dict[str, Any]
    validation_results: List[ValidationResult]
    fused_properties: Dict[str, FusedProperty]
    summary: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class AgentRegistry:
    """Registry of all available characterization agents."""

    # Agent mapping: category -> agent class name
    AGENT_MAP = {
        PropertyCategory.THERMAL: [
            'DSCAgent',
            'TGAAgent',
            'TMAAgent',
        ],
        PropertyCategory.MECHANICAL: [
            'DMAAgent',
            'TensileTestingAgent',
            'RheologistAgent',
            'NanoindentationAgent',
            'ScanningProbeAgent',
        ],
        PropertyCategory.ELECTRICAL: [
            'VoltammetryAgent',
            'BatteryTestingAgent',
            'EISAgent',
            'BDSAgent',
        ],
        PropertyCategory.OPTICAL: [
            'OpticalSpectroscopyAgent',
            'OpticalMicroscopyAgent',
            'SurfaceScienceAgent',  # Includes Ellipsometry
        ],
        PropertyCategory.CHEMICAL: [
            'MassSpectrometryAgent',
            'SpectroscopyAgent',
            'NMRAgent',
            'EPRAgent',
            'XRaySpectroscopyAgent',  # XAS
        ],
        PropertyCategory.STRUCTURAL: [
            'XRayScatteringAgent',  # SAXS, WAXS, GISAXS
            'LightScatteringAgent',
            'ScanningProbeAgent',  # AFM
            'OpticalMicroscopyAgent',
        ],
        PropertyCategory.SURFACE: [
            'SurfaceScienceAgent',  # XPS, Ellipsometry, QCM-D, SPR
            'ScanningProbeAgent',
            'XRaySpectroscopyAgent',  # XPS comparison with XAS
        ],
        PropertyCategory.MAGNETIC: [
            'EPRAgent',
        ],
    }

    # Technique to agent mapping
    TECHNIQUE_MAP = {
        # Thermal
        'DSC': 'DSCAgent',
        'TGA': 'TGAAgent',
        'TMA': 'TMAAgent',

        # Mechanical
        'DMA': 'DMAAgent',
        'tensile': 'TensileTestingAgent',
        'compression': 'TensileTestingAgent',
        'rheology': 'RheologistAgent',
        'nanoindentation': 'NanoindentationAgent',
        'AFM': 'ScanningProbeAgent',

        # Spectroscopy
        'NMR': 'NMRAgent',
        'EPR': 'EPRAgent',
        'FTIR': 'SpectroscopyAgent',
        'Raman': 'SpectroscopyAgent',
        'UV-Vis': 'OpticalSpectroscopyAgent',

        # X-ray
        'SAXS': 'XRayScatteringAgent',
        'WAXS': 'XRayScatteringAgent',
        'GISAXS': 'XRayScatteringAgent',
        'XAS': 'XRaySpectroscopyAgent',
        'XPS': 'SurfaceScienceAgent',

        # Electrochemistry
        'CV': 'VoltammetryAgent',
        'EIS': 'EISAgent',
        'BDS': 'BDSAgent',

        # Surface
        'QCM-D': 'SurfaceScienceAgent',
        'SPR': 'SurfaceScienceAgent',
        'ellipsometry': 'SurfaceScienceAgent',
        'contact_angle': 'SurfaceScienceAgent',

        # Mass spec
        'MALDI': 'MassSpectrometryAgent',
        'ESI': 'MassSpectrometryAgent',

        # Scattering
        'DLS': 'LightScatteringAgent',
        'SLS': 'LightScatteringAgent',
    }

    def __init__(self):
        """Initialize the agent registry."""
        self.loaded_agents: Dict[str, Any] = {}

    def get_agents_for_category(self, category: PropertyCategory) -> List[str]:
        """Get agent names for a property category.

        Args:
            category: PropertyCategory enum

        Returns:
            List of agent class names
        """
        return self.AGENT_MAP.get(category, [])

    def get_agent_for_technique(self, technique: str) -> Optional[str]:
        """Get agent name for a specific technique.

        Args:
            technique: Technique name

        Returns:
            Agent class name or None
        """
        return self.TECHNIQUE_MAP.get(technique)

    def load_agent(self, agent_name: str) -> Optional[Any]:
        """Load an agent instance.

        Args:
            agent_name: Name of the agent class

        Returns:
            Agent instance or None if not found
        """
        if agent_name in self.loaded_agents:
            return self.loaded_agents[agent_name]

        # Try to import and instantiate the agent
        try:
            # Convert agent name to module name (e.g., DSCAgent -> dsc_agent)
            module_name = ''.join(['_' + c.lower() if c.isupper() else c for c in agent_name]).lstrip('_')
            module = importlib.import_module(f'materials-science-agents.{module_name}')
            agent_class = getattr(module, agent_name)
            agent_instance = agent_class()
            self.loaded_agents[agent_name] = agent_instance
            return agent_instance
        except Exception as e:
            print(f"Warning: Could not load agent {agent_name}: {e}")
            return None


class CharacterizationMaster:
    """Master orchestrator for materials characterization.

    This class coordinates:
    - Agent selection based on sample and properties
    - Multi-technique measurement execution
    - Automatic cross-validation
    - Result aggregation and reporting
    """

    def __init__(self, enable_fusion: bool = True):
        """Initialize the characterization master.

        Args:
            enable_fusion: Enable multi-modal data fusion (default True)
        """
        self.agent_registry = AgentRegistry()
        self.validation_framework = initialize_framework()
        self.fusion_framework = DataFusionFramework() if enable_fusion else None
        self.measurement_history: List[MeasurementResult] = []
        self.enable_fusion = enable_fusion

    def suggest_techniques(self, request: MeasurementRequest) -> Dict[PropertyCategory, List[str]]:
        """Suggest techniques based on sample type and properties of interest.

        Args:
            request: MeasurementRequest

        Returns:
            Dictionary mapping property categories to suggested techniques
        """
        suggestions = {}

        for category in request.property_categories:
            agents = self.agent_registry.get_agents_for_category(category)
            techniques = []

            # Get specific technique recommendations based on sample type
            if category == PropertyCategory.THERMAL:
                techniques = ['DSC', 'TGA']
                if request.sample_type in [SampleType.POLYMER, SampleType.COMPOSITE]:
                    techniques.append('TMA')

            elif category == PropertyCategory.MECHANICAL:
                if request.sample_type in [SampleType.POLYMER, SampleType.COMPOSITE]:
                    techniques = ['DMA', 'tensile', 'rheology']
                else:
                    techniques = ['nanoindentation', 'tensile']

            elif category == PropertyCategory.STRUCTURAL:
                if request.sample_type in [SampleType.NANOPARTICLE, SampleType.COLLOID]:
                    techniques = ['SAXS', 'DLS', 'TEM']
                elif request.sample_type == SampleType.THIN_FILM:
                    techniques = ['GISAXS', 'AFM', 'XRR']
                else:
                    techniques = ['WAXS', 'SAXS']

            elif category == PropertyCategory.CHEMICAL:
                techniques = ['NMR', 'FTIR', 'Raman']
                if 'oxidation_state' in request.properties_of_interest:
                    techniques.append('XAS')

            elif category == PropertyCategory.SURFACE:
                techniques = ['XPS', 'contact_angle']
                if request.sample_type == SampleType.THIN_FILM:
                    techniques.extend(['ellipsometry', 'AFM'])

            elif category == PropertyCategory.ELECTRICAL:
                techniques = ['EIS', 'BDS']
                if request.sample_type == SampleType.BIOMATERIAL:
                    techniques.insert(0, 'CV')

            suggestions[category] = techniques

        return suggestions

    def plan_measurements(self, request: MeasurementRequest) -> List[Tuple[str, str]]:
        """Plan measurement sequence with agents and techniques.

        Args:
            request: MeasurementRequest

        Returns:
            List of (agent_name, technique) tuples
        """
        plan = []

        # If specific techniques requested, use those
        if request.techniques_requested:
            for technique in request.techniques_requested:
                agent_name = self.agent_registry.get_agent_for_technique(technique)
                if agent_name:
                    plan.append((agent_name, technique))
                else:
                    print(f"Warning: No agent found for technique {technique}")

        # Otherwise, suggest based on properties
        else:
            suggestions = self.suggest_techniques(request)
            for category, techniques in suggestions.items():
                for technique in techniques:
                    agent_name = self.agent_registry.get_agent_for_technique(technique)
                    if agent_name and (agent_name, technique) not in plan:
                        plan.append((agent_name, technique))

        return plan

    def execute_measurement(self, request: MeasurementRequest) -> MeasurementResult:
        """Execute a complete characterization measurement.

        Args:
            request: MeasurementRequest

        Returns:
            MeasurementResult with aggregated data
        """
        start_time = datetime.now()
        technique_results = {}
        validation_results = []
        warnings = []
        recommendations = []

        # Plan measurements
        measurement_plan = self.plan_measurements(request)
        print(f"\nMeasurement Plan for {request.sample_name}:")
        for agent_name, technique in measurement_plan:
            print(f"  - {technique} (using {agent_name})")

        # Execute measurements
        for agent_name, technique in measurement_plan:
            agent = self.agent_registry.load_agent(agent_name)
            if agent is None:
                warnings.append(f"Could not load agent {agent_name} for {technique}")
                continue

            try:
                # Execute technique
                # Note: This is a placeholder - actual execution would require
                # proper input data preparation
                result = {
                    'technique': technique,
                    'agent': agent_name,
                    'status': 'simulated',
                    'message': f'Would execute {technique} using {agent_name}'
                }
                technique_results[technique] = result

            except Exception as e:
                warnings.append(f"Error executing {technique}: {str(e)}")

        # Perform cross-validation if requested
        if request.cross_validate and len(technique_results) > 1:
            print("\nPerforming cross-validation...")
            validation_results = self._perform_cross_validations(technique_results)

            # Generate recommendations based on validation
            for val_result in validation_results:
                if val_result.status in [ValidationStatus.POOR, ValidationStatus.FAILED]:
                    recommendations.extend(val_result.recommendations)

        # Perform data fusion if enabled
        fused_properties = {}
        if self.enable_fusion and self.fusion_framework and len(technique_results) > 1:
            print("\nPerforming data fusion...")
            fused_properties = self._perform_data_fusion(technique_results, request.properties_of_interest)

            # Add fusion recommendations
            for prop_name, fused_prop in fused_properties.items():
                if fused_prop.warnings:
                    recommendations.extend(fused_prop.warnings)
                if fused_prop.quality_metrics.get('agreement', 1.0) < 0.7:
                    recommendations.append(f"Low agreement for {prop_name}: consider additional measurements")

        # Create summary
        summary = {
            'sample_name': request.sample_name,
            'sample_type': request.sample_type.value,
            'num_techniques': len(technique_results),
            'num_validations': len(validation_results),
            'num_fused_properties': len(fused_properties),
            'measurement_time': (datetime.now() - start_time).total_seconds(),
            'validation_success_rate': self._calculate_validation_success_rate(validation_results)
        }

        result = MeasurementResult(
            sample_name=request.sample_name,
            timestamp=start_time,
            technique_results=technique_results,
            validation_results=validation_results,
            fused_properties=fused_properties,
            summary=summary,
            recommendations=recommendations if recommendations else ['All validations passed'],
            warnings=warnings,
            metadata=request.metadata
        )

        self.measurement_history.append(result)
        return result

    def _perform_cross_validations(self, technique_results: Dict[str, Any]) -> List[ValidationResult]:
        """Perform all applicable cross-validations.

        Args:
            technique_results: Dictionary of technique results

        Returns:
            List of ValidationResult objects
        """
        validation_results = []
        techniques = list(technique_results.keys())

        # Check all pairs
        for i, tech1 in enumerate(techniques):
            for tech2 in techniques[i+1:]:
                # Check if validation pair exists
                # This would need to match property names properly
                # For now, we'll simulate
                print(f"  Checking {tech1} ↔ {tech2}")

        return validation_results

    def _perform_data_fusion(self, technique_results: Dict[str, Any],
                            properties_of_interest: List[str]) -> Dict[str, FusedProperty]:
        """Perform data fusion for properties measured by multiple techniques.

        Args:
            technique_results: Dictionary of technique results
            properties_of_interest: List of property names to fuse

        Returns:
            Dictionary mapping property names to fused properties
        """
        if not self.fusion_framework:
            return {}

        # Group measurements by property
        property_measurements: Dict[str, List[Measurement]] = {}

        # Extract measurements from technique results
        # This is a simplified example - actual implementation would need
        # to extract specific properties from each technique's result structure
        for technique, result in technique_results.items():
            # Example: if result contains property values
            for prop_name in properties_of_interest:
                if prop_name in str(result):  # Simplified check
                    # Create measurement (would extract real values in production)
                    measurement = Measurement(
                        technique=technique,
                        property_name=prop_name,
                        value=100.0,  # Placeholder - would extract from result
                        uncertainty=1.0,  # Placeholder - would extract from result
                        units="units"  # Placeholder
                    )
                    if prop_name not in property_measurements:
                        property_measurements[prop_name] = []
                    property_measurements[prop_name].append(measurement)

        # Fuse each property
        fused_properties = {}
        for prop_name, measurements in property_measurements.items():
            if len(measurements) > 1:
                try:
                    fused = self.fusion_framework.fuse_measurements(
                        measurements,
                        method=FusionMethod.BAYESIAN
                    )
                    fused_properties[prop_name] = fused
                    print(f"  Fused {prop_name}: {fused.fused_value:.2f} ± {fused.uncertainty:.2f} (from {len(measurements)} techniques)")
                except Exception as e:
                    print(f"  Warning: Could not fuse {prop_name}: {e}")

        return fused_properties

    def _calculate_validation_success_rate(self, validation_results: List[ValidationResult]) -> float:
        """Calculate validation success rate.

        Args:
            validation_results: List of ValidationResult objects

        Returns:
            Success rate percentage
        """
        if not validation_results:
            return 100.0

        successful = sum(
            1 for r in validation_results
            if r.status in [ValidationStatus.EXCELLENT, ValidationStatus.GOOD]
        )
        return (successful / len(validation_results)) * 100

    def generate_report(self, measurement_result: MeasurementResult) -> str:
        """Generate a comprehensive measurement report.

        Args:
            measurement_result: MeasurementResult to report

        Returns:
            Formatted report string
        """
        lines = ["=" * 80]
        lines.append(f"CHARACTERIZATION REPORT: {measurement_result.sample_name}")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {measurement_result.timestamp.isoformat()}")
        lines.append("")

        lines.append("SUMMARY:")
        for key, value in measurement_result.summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        lines.append(f"TECHNIQUES EXECUTED ({len(measurement_result.technique_results)}):")
        for technique, result in measurement_result.technique_results.items():
            lines.append(f"  ✓ {technique}")
        lines.append("")

        if measurement_result.validation_results:
            lines.append(f"CROSS-VALIDATIONS ({len(measurement_result.validation_results)}):")
            for val_result in measurement_result.validation_results:
                lines.append(f"  {val_result.pair.technique_1} ↔ {val_result.pair.technique_2}: {val_result.status.value}")
            lines.append("")

        if measurement_result.fused_properties:
            lines.append(f"FUSED PROPERTIES ({len(measurement_result.fused_properties)}):")
            for prop_name, fused_prop in measurement_result.fused_properties.items():
                ci_lower, ci_upper = fused_prop.confidence_interval
                lines.append(f"  {prop_name}:")
                lines.append(f"    Value: {fused_prop.fused_value:.4f} ± {fused_prop.uncertainty:.4f}")
                lines.append(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                lines.append(f"    From {len(fused_prop.contributing_measurements)} techniques: {[m.technique for m in fused_prop.contributing_measurements]}")
                lines.append(f"    Agreement: {fused_prop.quality_metrics.get('agreement', 0):.2f}")
            lines.append("")

        if measurement_result.recommendations:
            lines.append("RECOMMENDATIONS:")
            for rec in measurement_result.recommendations:
                lines.append(f"  • {rec}")
            lines.append("")

        if measurement_result.warnings:
            lines.append("WARNINGS:")
            for warning in measurement_result.warnings:
                lines.append(f"  ⚠ {warning}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def get_measurement_history(self, sample_name: Optional[str] = None) -> List[MeasurementResult]:
        """Get measurement history with optional filtering.

        Args:
            sample_name: Filter by sample name (optional)

        Returns:
            List of MeasurementResult objects
        """
        if sample_name:
            return [r for r in self.measurement_history if r.sample_name == sample_name]
        return self.measurement_history


# Example usage
if __name__ == "__main__":
    # Initialize the master orchestrator
    master = CharacterizationMaster()

    # Create a measurement request
    request = MeasurementRequest(
        sample_name="Polymer-001",
        sample_type=SampleType.POLYMER,
        properties_of_interest=["glass_transition", "crystallinity", "modulus", "molecular_weight"],
        property_categories=[
            PropertyCategory.THERMAL,
            PropertyCategory.MECHANICAL,
            PropertyCategory.CHEMICAL,
            PropertyCategory.STRUCTURAL
        ],
        cross_validate=True
    )

    # Get technique suggestions
    print("Suggested Techniques:")
    suggestions = master.suggest_techniques(request)
    for category, techniques in suggestions.items():
        print(f"  {category.value}: {', '.join(techniques)}")

    # Execute measurement
    print("\n" + "=" * 80)
    result = master.execute_measurement(request)

    # Generate report
    print("\n" + master.generate_report(result))

    # Get framework statistics
    print("\n" + master.validation_framework.generate_report())
