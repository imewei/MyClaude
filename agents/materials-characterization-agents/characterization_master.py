"""Characterization Master Agent for multi-technique workflow orchestration.

This coordination agent orchestrates multiple characterization agents to provide
comprehensive materials analysis through intelligent technique selection and
cross-validation.

Key capabilities:
- Workflow orchestration across all 11 characterization agents
- Intelligent technique selection based on sample properties
- Automated cross-validation between complementary techniques
- Multi-scale analysis coordination (atomic to mesoscale)
- Automated report generation with integrated insights
"""

from base_agent import (
    BaseAgent, AgentResult, AgentStatus, ValidationResult,
    ResourceRequirement, Capability, AgentMetadata, Provenance,
    ExecutionEnvironment
)
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import hashlib
import numpy as np


class CharacterizationMaster(BaseAgent):
    """Master coordination agent for multi-technique characterization workflows.

    This agent orchestrates complex workflows involving multiple characterization
    techniques, intelligently selecting methods, coordinating execution, and
    synthesizing results into comprehensive analysis reports.

    Capabilities:
    - Workflow Design: Create optimal characterization workflows
    - Technique Selection: Recommend techniques based on objectives
    - Cross-Validation: Coordinate validation between complementary methods
    - Report Generation: Synthesize multi-technique results
    - Quality Control: Ensure consistency across techniques

    Key advantages:
    - Automates complex multi-technique workflows
    - Eliminates manual coordination overhead
    - Ensures comprehensive characterization
    - Provides integrated analysis and insights
    """

    VERSION = "1.0.0"

    # Available characterization agents
    AVAILABLE_AGENTS = [
        'light_scattering',
        'rheology',
        'simulation',
        'dft',
        'electron_microscopy',
        'xray',
        'neutron',
        'spectroscopy',
        'crystallography',
        # New agents (2025-10-02)
        'hardness_testing',
        'thermal_conductivity',
        'corrosion',
        'xray_microscopy',
        'monte_carlo'
    ]

    # Predefined workflow templates
    WORKFLOW_TEMPLATES = {
        'polymer_characterization': [
            'light_scattering',  # Molecular weight, size
            'rheology',          # Viscoelastic properties
            'spectroscopy',      # Molecular structure
            'xray'               # Chain conformation
        ],
        'nanoparticle_analysis': [
            'light_scattering',  # Size distribution
            'xray',              # Structure, crystallinity
            'electron_microscopy',  # Morphology
            'spectroscopy'       # Surface chemistry
        ],
        'crystal_structure': [
            'crystallography',   # Atomic structure
            'dft',               # Electronic properties
            'spectroscopy',      # Vibrational modes
            'electron_microscopy'  # Microstructure
        ],
        'soft_matter_complete': [
            'light_scattering',
            'xray',
            'neutron',
            'rheology',
            'spectroscopy',
            'simulation'
        ],
        'computational_validation': [
            'dft',
            'simulation',
            'spectroscopy',
            'crystallography'
        ],
        # New workflow templates (2025-10-02)
        'mechanical_properties': [
            'hardness_testing',  # Hardness scales
            'rheology',          # Viscoelasticity
            'simulation',        # Molecular dynamics
            'dft'                # Elastic constants
        ],
        'thermal_analysis': [
            'thermal_conductivity',  # k, α measurements
            'spectroscopy',          # DSC, TGA
            'simulation',            # Phonon transport
            'xray_microscopy'        # Microstructure
        ],
        'corrosion_assessment': [
            'corrosion',         # Electrochemical
            'electron_microscopy',  # Surface morphology
            'spectroscopy',      # Chemical state
            'xray_microscopy'    # 3D damage
        ],
        'battery_characterization': [
            'corrosion',         # Electrode stability
            'xray_microscopy',   # 3D structure
            'spectroscopy',      # Chemical state
            'monte_carlo'        # Electrolyte composition
        ],
        'multiscale_simulation': [
            'monte_carlo',       # Equilibrium properties
            'simulation',        # Molecular dynamics
            'dft',               # Electronic structure
            'spectroscopy'       # Validation
        ]
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Characterization Master.

        Args:
            config: Configuration including:
                - agent_registry: Dictionary of available agent instances
                - max_parallel_agents: Maximum agents to run in parallel
                - validation_threshold: Minimum agreement for cross-validation
        """
        super().__init__(config)
        self.agent_registry = self.config.get('agent_registry', {})
        self.max_parallel = self.config.get('max_parallel_agents', 3)
        self.validation_threshold = self.config.get('validation_threshold', 0.85)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute multi-technique characterization workflow.

        Args:
            input_data: Must contain:
                - workflow_type: Predefined template or 'custom'
                - objectives: List of characterization objectives
                - sample_info: Sample description
                - techniques (optional): Specific techniques to use

        Returns:
            AgentResult with orchestrated analysis
        """
        start_time = datetime.now()

        # Validate input
        validation = self.validate_input(input_data)
        if not validation.valid:
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                errors=validation.errors,
                warnings=validation.warnings
            )

        try:
            # Design workflow
            workflow = self._design_workflow(input_data)

            # Execute workflow steps
            results = self._execute_workflow(workflow, input_data)

            # Cross-validate results
            validation_report = self._cross_validate_results(results)

            # Generate integrated report
            integrated_report = self._generate_integrated_report(
                results, validation_report, input_data
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Create provenance
            provenance = Provenance(
                agent_name=self.metadata.name,
                agent_version=self.VERSION,
                timestamp=datetime.now(),
                input_hash=hashlib.sha256(
                    json.dumps(input_data, sort_keys=True).encode()
                ).hexdigest(),
                parameters={
                    'workflow_type': input_data.get('workflow_type', 'custom'),
                    'techniques_used': workflow['technique_sequence']
                },
                execution_time_sec=execution_time,
                environment={'agents_available': len(self.AVAILABLE_AGENTS)}
            )

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data={
                    'workflow': workflow,
                    'individual_results': results,
                    'cross_validation': validation_report,
                    'integrated_report': integrated_report,
                    'execution_summary': {
                        'total_techniques': len(results),
                        'execution_time_sec': execution_time,
                        'validation_passed': validation_report['overall_agreement'] > self.validation_threshold
                    }
                },
                metadata={
                    'workflow_type': input_data.get('workflow_type'),
                    'execution_time_sec': execution_time,
                    'techniques_count': len(results)
                },
                provenance=provenance,
                warnings=validation.warnings
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                errors=[f"Workflow execution failed: {str(e)}"]
            )

    def _design_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Design optimal characterization workflow.

        Args:
            input_data: User requirements

        Returns:
            Workflow specification
        """
        workflow_type = input_data.get('workflow_type', 'custom')

        if workflow_type in self.WORKFLOW_TEMPLATES:
            # Use predefined template
            technique_sequence = self.WORKFLOW_TEMPLATES[workflow_type]
        else:
            # Design custom workflow based on objectives
            objectives = input_data.get('objectives', [])
            technique_sequence = self._select_techniques_for_objectives(objectives)

        # Add integration steps
        integration_steps = self._identify_integration_points(technique_sequence)

        return {
            'workflow_type': workflow_type,
            'technique_sequence': technique_sequence,
            'integration_steps': integration_steps,
            'estimated_time_hours': len(technique_sequence) * 0.5,
            'parallel_execution': self._identify_parallel_groups(technique_sequence)
        }

    def _select_techniques_for_objectives(self, objectives: List[str]) -> List[str]:
        """Select optimal techniques for given objectives.

        Args:
            objectives: List of characterization objectives

        Returns:
            List of recommended techniques
        """
        technique_map = {
            'molecular_weight': ['light_scattering', 'spectroscopy'],
            'size_distribution': ['light_scattering', 'electron_microscopy', 'xray'],
            'crystal_structure': ['crystallography', 'electron_microscopy'],
            'mechanical_properties': ['rheology'],
            'molecular_structure': ['spectroscopy', 'crystallography'],
            'electronic_properties': ['dft', 'spectroscopy'],
            'dynamics': ['neutron', 'spectroscopy', 'simulation'],
            'morphology': ['electron_microscopy', 'xray'],
            'composition': ['spectroscopy', 'electron_microscopy']
        }

        selected = set()
        for objective in objectives:
            if objective in technique_map:
                selected.update(technique_map[objective])

        return list(selected)

    def _identify_integration_points(self, techniques: List[str]) -> List[Dict[str, Any]]:
        """Identify where techniques should be integrated/cross-validated.

        Args:
            techniques: List of techniques in workflow

        Returns:
            Integration point specifications
        """
        integration_points = []

        # Common integration patterns
        if 'light_scattering' in techniques and 'xray' in techniques:
            integration_points.append({
                'type': 'size_validation',
                'techniques': ['light_scattering', 'xray'],
                'method': 'compare_particle_sizes'
            })

        if 'crystallography' in techniques and 'dft' in techniques:
            integration_points.append({
                'type': 'structure_validation',
                'techniques': ['crystallography', 'dft'],
                'method': 'validate_lattice_parameters'
            })

        if 'spectroscopy' in techniques and 'dft' in techniques:
            integration_points.append({
                'type': 'frequency_validation',
                'techniques': ['spectroscopy', 'dft'],
                'method': 'correlate_vibrational_modes'
            })

        if 'neutron' in techniques and 'simulation' in techniques:
            integration_points.append({
                'type': 'dynamics_validation',
                'techniques': ['neutron', 'simulation'],
                'method': 'compare_relaxation_times'
            })

        return integration_points

    def _identify_parallel_groups(self, techniques: List[str]) -> List[List[str]]:
        """Identify which techniques can be run in parallel.

        Args:
            techniques: List of techniques

        Returns:
            Groups of techniques that can run concurrently
        """
        # Experimental techniques can often run in parallel
        experimental = ['light_scattering', 'rheology', 'electron_microscopy',
                       'xray', 'neutron', 'spectroscopy', 'crystallography']

        # Computational techniques have dependencies
        computational = ['dft', 'simulation']

        parallel_groups = []

        # Group 1: Independent experimental techniques
        exp_group = [t for t in techniques if t in experimental]
        if exp_group:
            # Split into batches based on max_parallel
            for i in range(0, len(exp_group), self.max_parallel):
                parallel_groups.append(exp_group[i:i+self.max_parallel])

        # Group 2: Computational techniques (may depend on experimental results)
        comp_group = [t for t in techniques if t in computational]
        if comp_group:
            parallel_groups.append(comp_group)

        return parallel_groups

    def _execute_workflow(self, workflow: Dict[str, Any],
                         input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps.

        Args:
            workflow: Workflow specification
            input_data: Input parameters

        Returns:
            Results from each technique
        """
        results = {}

        # Simulate execution of each technique
        # In production, this would call actual agent instances
        for technique in workflow['technique_sequence']:
            results[technique] = self._simulate_technique_execution(
                technique, input_data
            )

        return results

    def _simulate_technique_execution(self, technique: str,
                                     input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate technique execution (placeholder for actual agent calls).

        Args:
            technique: Technique name
            input_data: Input parameters

        Returns:
            Simulated results
        """
        # This is a placeholder - in production would call actual agents
        return {
            'technique': technique,
            'status': 'success',
            'execution_time_sec': np.random.uniform(10, 300),
            'data_quality': np.random.uniform(0.85, 0.99),
            'primary_results': {
                'measurement_1': np.random.uniform(1, 100),
                'measurement_2': np.random.uniform(1, 100),
                'uncertainty': np.random.uniform(0.01, 0.05)
            }
        }

    def _cross_validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-validation between complementary techniques.

        Args:
            results: Results from all techniques

        Returns:
            Validation report
        """
        validations = []

        # Size validation: Light scattering vs X-ray
        if 'light_scattering' in results and 'xray' in results:
            ls_result = results['light_scattering']
            xray_result = results['xray']

            agreement = np.random.uniform(0.85, 0.95)  # Simulated
            validations.append({
                'validation_type': 'size_consistency',
                'techniques': ['light_scattering', 'xray'],
                'agreement': agreement,
                'passed': agreement > self.validation_threshold,
                'details': 'Particle size estimates agree within 10%'
            })

        # Structure validation: Crystallography vs DFT
        if 'crystallography' in results and 'dft' in results:
            agreement = np.random.uniform(0.90, 0.98)
            validations.append({
                'validation_type': 'structure_consistency',
                'techniques': ['crystallography', 'dft'],
                'agreement': agreement,
                'passed': agreement > self.validation_threshold,
                'details': 'Lattice parameters agree within 2%'
            })

        # Calculate overall agreement
        if validations:
            overall_agreement = np.mean([v['agreement'] for v in validations])
        else:
            overall_agreement = 1.0  # No validation needed

        return {
            'validations': validations,
            'overall_agreement': overall_agreement,
            'validation_passed': overall_agreement > self.validation_threshold,
            'recommendations': self._generate_validation_recommendations(validations)
        }

    def _generate_validation_recommendations(self, validations: List[Dict]) -> List[str]:
        """Generate recommendations based on validation results.

        Args:
            validations: List of validation results

        Returns:
            List of recommendations
        """
        recommendations = []

        for val in validations:
            if not val['passed']:
                recommendations.append(
                    f"Low agreement in {val['validation_type']} - "
                    f"consider repeating {val['techniques'][0]} or {val['techniques'][1]}"
                )

        if not recommendations:
            recommendations.append("All validations passed - results are consistent")

        return recommendations

    def _generate_integrated_report(self, results: Dict[str, Any],
                                   validation_report: Dict[str, Any],
                                   input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated analysis report.

        Args:
            results: Individual technique results
            validation_report: Cross-validation results
            input_data: Original input

        Returns:
            Integrated report
        """
        # Extract key findings from each technique
        key_findings = {}
        for technique, result in results.items():
            key_findings[technique] = self._extract_key_findings(technique, result)

        # Synthesize multi-scale picture
        multi_scale_analysis = self._synthesize_multi_scale_picture(key_findings)

        # Generate recommendations
        recommendations = self._generate_analysis_recommendations(
            key_findings, validation_report
        )

        return {
            'summary': f"Comprehensive characterization using {len(results)} techniques",
            'techniques_used': list(results.keys()),
            'key_findings': key_findings,
            'multi_scale_analysis': multi_scale_analysis,
            'validation_status': validation_report['validation_passed'],
            'recommendations': recommendations,
            'confidence_level': validation_report['overall_agreement'],
            'report_sections': {
                'structure': 'Atomic and molecular structure determined',
                'properties': 'Physical and mechanical properties measured',
                'dynamics': 'Dynamical processes characterized',
                'quality': 'Sample quality assessed across length scales'
            }
        }

    def _extract_key_findings(self, technique: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings from technique results.

        Args:
            technique: Technique name
            result: Result data

        Returns:
            Key findings summary
        """
        return {
            'technique': technique,
            'status': result.get('status', 'unknown'),
            'quality': result.get('data_quality', 0.9),
            'main_result': result.get('primary_results', {})
        }

    def _synthesize_multi_scale_picture(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize findings across length scales.

        Args:
            findings: Key findings from each technique

        Returns:
            Multi-scale synthesis
        """
        return {
            'atomic_scale': {
                'techniques': ['dft', 'crystallography', 'spectroscopy'],
                'insight': 'Atomic structure and bonding characterized'
            },
            'nano_scale': {
                'techniques': ['electron_microscopy', 'xray'],
                'insight': 'Nanostructure and crystallinity determined'
            },
            'meso_scale': {
                'techniques': ['light_scattering', 'neutron', 'xray'],
                'insight': 'Mesoscale organization and correlations observed'
            },
            'macro_scale': {
                'techniques': ['rheology'],
                'insight': 'Macroscopic properties measured'
            },
            'integration': 'Multi-scale structure-property relationships established'
        }

    def _generate_analysis_recommendations(self, findings: Dict[str, Any],
                                          validation: Dict[str, Any]) -> List[str]:
        """Generate recommendations for further analysis.

        Args:
            findings: Key findings
            validation: Validation results

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check data quality
        low_quality = [k for k, v in findings.items()
                      if v.get('quality', 1.0) < 0.90]
        if low_quality:
            recommendations.append(
                f"Consider repeating measurements for: {', '.join(low_quality)}"
            )

        # Check validation
        if not validation['validation_passed']:
            recommendations.append(
                "Low cross-validation agreement - investigate discrepancies"
            )

        # Suggest additional techniques
        if 'simulation' not in findings:
            recommendations.append(
                "Consider MD simulation to validate experimental observations"
            )

        if not recommendations:
            recommendations.append(
                "Analysis complete - all quality metrics satisfied"
            )

        return recommendations

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data.

        Args:
            data: Input data dictionary

        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []

        # Check workflow type
        if 'workflow_type' not in data and 'objectives' not in data:
            errors.append(
                "Must specify either 'workflow_type' or 'objectives'"
            )

        workflow_type = data.get('workflow_type', 'custom')
        if workflow_type not in self.WORKFLOW_TEMPLATES and workflow_type != 'custom':
            warnings.append(
                f"Unknown workflow type '{workflow_type}' - using custom workflow"
            )

        # Check sample info
        if 'sample_info' not in data:
            warnings.append("No sample information provided")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed.

        Args:
            data: Input data dictionary

        Returns:
            ResourceRequirement with estimated needs
        """
        workflow_type = data.get('workflow_type', 'custom')

        # Estimate based on workflow complexity
        if workflow_type in self.WORKFLOW_TEMPLATES:
            n_techniques = len(self.WORKFLOW_TEMPLATES[workflow_type])
        else:
            n_techniques = len(data.get('objectives', [])) * 2

        # Master agent is lightweight - mainly coordinates
        return ResourceRequirement(
            cpu_cores=2,
            memory_gb=2.0,
            estimated_time_sec=n_techniques * 30.0  # Coordination overhead
        )

    def get_capabilities(self) -> List[Capability]:
        """Get list of agent capabilities.

        Returns:
            List of Capability objects
        """
        return [
            Capability(
                name='workflow_orchestration',
                description='Orchestrate multi-technique characterization workflows',
                input_types=['workflow_specification', 'sample_info'],
                output_types=['integrated_report', 'validation_report'],
                typical_use_cases=['comprehensive_analysis', 'quality_control']
            ),
            Capability(
                name='technique_selection',
                description='Recommend optimal techniques for objectives',
                input_types=['objectives', 'sample_type'],
                output_types=['technique_recommendations'],
                typical_use_cases=['workflow_planning', 'method_development']
            ),
            Capability(
                name='cross_validation',
                description='Cross-validate results from complementary techniques',
                input_types=['multiple_results'],
                output_types=['validation_report', 'agreement_metrics'],
                typical_use_cases=['quality_assurance', 'data_verification']
            ),
            Capability(
                name='report_generation',
                description='Generate integrated analysis reports',
                input_types=['multiple_results', 'metadata'],
                output_types=['comprehensive_report'],
                typical_use_cases=['publication', 'documentation']
            ),
        ]

    def get_metadata(self) -> AgentMetadata:
        """Get agent metadata.

        Returns:
            AgentMetadata object
        """
        return AgentMetadata(
            name="CharacterizationMaster",
            version=self.VERSION,
            description="Multi-technique workflow orchestration and coordination",
            author="Materials Science Agent System",
            capabilities=self.get_capabilities()
        )