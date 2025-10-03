"""
Comprehensive tests for CharacterizationMaster.

Test categories:
1. Initialization
2. Input validation
3. Workflow design
4. Technique selection
5. Cross-validation
6. Report generation
7. Resource estimation
"""

import pytest
import uuid
from characterization_master import CharacterizationMaster


class TestCharacterizationMasterBasics:
    """Test basic agent functionality."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = CharacterizationMaster()
        assert agent.VERSION == "1.0.0"
        assert agent.metadata.name == "CharacterizationMaster"

    def test_available_agents(self):
        """Test that all expected agents are available."""
        agent = CharacterizationMaster()
        agents = agent.AVAILABLE_AGENTS
        assert 'light_scattering' in agents
        assert 'crystallography' in agents
        assert 'spectroscopy' in agents
        assert len(agents) == 9

    def test_workflow_templates(self):
        """Test that predefined workflow templates exist."""
        agent = CharacterizationMaster()
        templates = agent.WORKFLOW_TEMPLATES
        assert 'polymer_characterization' in templates
        assert 'nanoparticle_analysis' in templates
        assert 'crystal_structure' in templates


class TestCharacterizationMasterValidation:
    """Test input validation."""

    def test_validation_missing_workflow_and_objectives(self):
        """Test validation with missing required fields."""
        agent = CharacterizationMaster()
        result = agent.validate_input({})
        assert not result.valid
        assert len(result.errors) > 0

    def test_validation_with_workflow_type(self):
        """Test validation with valid workflow type."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'polymer_characterization',
            'sample_info': {'name': 'PS-100k'}
        }
        result = agent.validate_input(input_data)
        assert result.valid

    def test_validation_with_objectives(self):
        """Test validation with objectives."""
        agent = CharacterizationMaster()
        input_data = {
            'objectives': ['molecular_weight', 'size_distribution'],
            'sample_info': {'type': 'polymer'}
        }
        result = agent.validate_input(input_data)
        assert result.valid

    def test_validation_unknown_workflow_warning(self):
        """Test warning for unknown workflow type."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'unknown_workflow',
            'sample_info': {}
        }
        result = agent.validate_input(input_data)
        assert result.valid  # Still valid, just warning
        assert len(result.warnings) > 0


class TestCharacterizationMasterWorkflowDesign:
    """Test workflow design functionality."""

    def test_design_workflow_predefined(self):
        """Test workflow design with predefined template."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'polymer_characterization',
            'sample_info': {}
        }
        workflow = agent._design_workflow(input_data)

        assert workflow['workflow_type'] == 'polymer_characterization'
        assert 'technique_sequence' in workflow
        assert len(workflow['technique_sequence']) > 0
        assert 'light_scattering' in workflow['technique_sequence']

    def test_design_workflow_custom(self):
        """Test custom workflow design."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'custom',
            'objectives': ['molecular_weight', 'crystal_structure']
        }
        workflow = agent._design_workflow(input_data)

        assert 'technique_sequence' in workflow
        assert 'integration_steps' in workflow

    def test_workflow_has_integration_steps(self):
        """Test that workflows include integration steps."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'crystal_structure',
            'sample_info': {}
        }
        workflow = agent._design_workflow(input_data)

        assert 'integration_steps' in workflow
        # Crystal structure workflow should have DFT-crystallography integration
        if 'crystallography' in workflow['technique_sequence'] and 'dft' in workflow['technique_sequence']:
            assert len(workflow['integration_steps']) > 0

    def test_workflow_parallel_execution(self):
        """Test parallel execution grouping."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'soft_matter_complete',
            'sample_info': {}
        }
        workflow = agent._design_workflow(input_data)

        assert 'parallel_execution' in workflow
        assert isinstance(workflow['parallel_execution'], list)


class TestCharacterizationMasterTechniqueSelection:
    """Test intelligent technique selection."""

    def test_select_techniques_molecular_weight(self):
        """Test technique selection for molecular weight."""
        agent = CharacterizationMaster()
        objectives = ['molecular_weight']
        techniques = agent._select_techniques_for_objectives(objectives)

        assert 'light_scattering' in techniques or 'spectroscopy' in techniques

    def test_select_techniques_size_distribution(self):
        """Test technique selection for size distribution."""
        agent = CharacterizationMaster()
        objectives = ['size_distribution']
        techniques = agent._select_techniques_for_objectives(objectives)

        # Should recommend multiple techniques for size
        assert len(techniques) >= 2
        assert any(t in techniques for t in ['light_scattering', 'electron_microscopy', 'xray'])

    def test_select_techniques_multiple_objectives(self):
        """Test technique selection for multiple objectives."""
        agent = CharacterizationMaster()
        objectives = ['molecular_weight', 'mechanical_properties', 'morphology']
        techniques = agent._select_techniques_for_objectives(objectives)

        # Should cover all objectives
        assert 'rheology' in techniques  # mechanical
        assert any(t in techniques for t in ['light_scattering', 'spectroscopy'])  # molecular weight
        assert any(t in techniques for t in ['electron_microscopy', 'xray'])  # morphology


class TestCharacterizationMasterExecution:
    """Test workflow execution."""

    def test_execute_polymer_workflow(self):
        """Test execution of polymer characterization workflow."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'polymer_characterization',
            'sample_info': {'name': 'Polystyrene', 'molecular_weight': '100kDa'}
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'workflow' in result.data
        assert 'individual_results' in result.data
        assert 'integrated_report' in result.data

    def test_execute_nanoparticle_workflow(self):
        """Test execution of nanoparticle analysis workflow."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'nanoparticle_analysis',
            'sample_info': {'type': 'gold_nanoparticles'}
        }
        result = agent.execute(input_data)

        assert result.success
        assert result.data['execution_summary']['total_techniques'] > 0

    def test_execute_custom_workflow(self):
        """Test execution of custom workflow."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'custom',
            'objectives': ['molecular_structure', 'electronic_properties'],
            'sample_info': {}
        }
        result = agent.execute(input_data)

        assert result.success


class TestCharacterizationMasterCrossValidation:
    """Test cross-validation functionality."""

    def test_cross_validate_size_methods(self):
        """Test cross-validation between size measurement methods."""
        agent = CharacterizationMaster()
        results = {
            'light_scattering': {
                'technique': 'light_scattering',
                'status': 'success',
                'primary_results': {'size_nm': 50}
            },
            'xray': {
                'technique': 'xray',
                'status': 'success',
                'primary_results': {'size_nm': 48}
            }
        }

        validation = agent._cross_validate_results(results)

        assert 'validations' in validation
        assert 'overall_agreement' in validation
        assert validation['overall_agreement'] >= 0

    def test_cross_validate_structure_methods(self):
        """Test cross-validation between structure methods."""
        agent = CharacterizationMaster()
        results = {
            'crystallography': {
                'technique': 'crystallography',
                'status': 'success',
                'primary_results': {'lattice_a': 4.05}
            },
            'dft': {
                'technique': 'dft',
                'status': 'success',
                'primary_results': {'lattice_a': 4.03}
            }
        }

        validation = agent._cross_validate_results(results)

        assert validation['overall_agreement'] > 0
        # Should have structure validation
        assert any('structure' in v['validation_type'] for v in validation['validations'])

    def test_validation_passed_threshold(self):
        """Test that validation pass/fail respects threshold."""
        agent = CharacterizationMaster()
        results = {
            'light_scattering': {'technique': 'light_scattering', 'status': 'success'},
            'xray': {'technique': 'xray', 'status': 'success'}
        }

        validation = agent._cross_validate_results(results)

        # Validation should pass if agreement > threshold
        if validation['overall_agreement'] > agent.validation_threshold:
            assert validation['validation_passed']
        else:
            assert not validation['validation_passed']


class TestCharacterizationMasterReportGeneration:
    """Test integrated report generation."""

    def test_generate_report_structure(self):
        """Test that report has required structure."""
        agent = CharacterizationMaster()
        results = {
            'light_scattering': {'technique': 'light_scattering', 'status': 'success',
                                'data_quality': 0.95, 'primary_results': {}},
            'rheology': {'technique': 'rheology', 'status': 'success',
                        'data_quality': 0.92, 'primary_results': {}}
        }
        validation = {'validations': [], 'overall_agreement': 0.95, 'validation_passed': True}

        report = agent._generate_integrated_report(results, validation, {})

        assert 'summary' in report
        assert 'techniques_used' in report
        assert 'key_findings' in report
        assert 'multi_scale_analysis' in report
        assert 'recommendations' in report

    def test_report_includes_all_techniques(self):
        """Test that report covers all executed techniques."""
        agent = CharacterizationMaster()
        results = {
            'spectroscopy': {'technique': 'spectroscopy', 'status': 'success',
                           'data_quality': 0.96, 'primary_results': {}},
            'crystallography': {'technique': 'crystallography', 'status': 'success',
                              'data_quality': 0.94, 'primary_results': {}},
            'dft': {'technique': 'dft', 'status': 'success',
                   'data_quality': 0.97, 'primary_results': {}}
        }
        validation = {'validations': [], 'overall_agreement': 0.96, 'validation_passed': True}

        report = agent._generate_integrated_report(results, validation, {})

        assert len(report['techniques_used']) == 3
        assert 'spectroscopy' in report['techniques_used']
        assert 'crystallography' in report['techniques_used']
        assert 'dft' in report['techniques_used']

    def test_report_multi_scale_synthesis(self):
        """Test multi-scale analysis synthesis."""
        agent = CharacterizationMaster()
        findings = {
            'dft': {'technique': 'dft', 'quality': 0.95},
            'light_scattering': {'technique': 'light_scattering', 'quality': 0.93}
        }

        synthesis = agent._synthesize_multi_scale_picture(findings)

        assert 'atomic_scale' in synthesis
        assert 'meso_scale' in synthesis
        assert 'integration' in synthesis


class TestCharacterizationMasterResourceEstimation:
    """Test resource estimation."""

    def test_estimate_resources_simple_workflow(self):
        """Test resource estimation for simple workflow."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'polymer_characterization'
        }
        resources = agent.estimate_resources(input_data)

        assert resources.estimated_time_sec > 0
        assert resources.cpu_cores >= 1
        assert resources.memory_gb > 0

    def test_estimate_resources_complex_workflow(self):
        """Test resource estimation for complex workflow."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'soft_matter_complete'
        }
        resources = agent.estimate_resources(input_data)

        # Complex workflow should require more time
        assert resources.estimated_time_sec > 100


class TestCharacterizationMasterCapabilities:
    """Test capability reporting."""

    def test_get_capabilities(self):
        """Test capability listing."""
        agent = CharacterizationMaster()
        capabilities = agent.get_capabilities()

        assert len(capabilities) >= 4
        cap_names = [c.name for c in capabilities]
        assert 'workflow_orchestration' in cap_names
        assert 'cross_validation' in cap_names

    def test_capabilities_have_descriptions(self):
        """Test that all capabilities have descriptions."""
        agent = CharacterizationMaster()
        capabilities = agent.get_capabilities()

        for cap in capabilities:
            assert len(cap.description) > 0
            assert len(cap.input_types) > 0
            assert len(cap.output_types) > 0


class TestCharacterizationMasterProvenance:
    """Test provenance tracking."""

    def test_provenance_tracking(self):
        """Test that execution metadata is tracked."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'polymer_characterization',
            'sample_info': {},
            'test_id': str(uuid.uuid4())
        }
        result = agent.execute(input_data)

        assert result.success
        assert result.provenance is not None
        assert result.provenance.agent_version == '1.0.0'

    def test_metadata_includes_workflow_info(self):
        """Test that metadata includes workflow information."""
        agent = CharacterizationMaster()
        input_data = {
            'workflow_type': 'nanoparticle_analysis',
            'sample_info': {}
        }
        result = agent.execute(input_data)

        assert result.metadata['workflow_type'] == 'nanoparticle_analysis'
        assert 'techniques_count' in result.metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])