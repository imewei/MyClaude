"""
Tests for MaterialsInformaticsAgent.

Test categories:
1. Initialization
2. Input validation
3. Property prediction
4. Structure prediction
5. Active learning
6. Optimization
7. Screening
"""

import pytest
import uuid
from materials_informatics_agent import MaterialsInformaticsAgent


class TestMaterialsInformaticsBasics:
    """Test basic agent functionality."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = MaterialsInformaticsAgent()
        assert agent.VERSION == "1.0.0"
        assert agent.metadata.name == "MaterialsInformaticsAgent"

    def test_supported_tasks(self):
        """Test that all expected tasks are supported."""
        agent = MaterialsInformaticsAgent()
        tasks = agent.SUPPORTED_TASKS
        assert 'property_prediction' in tasks
        assert 'active_learning' in tasks
        assert 'optimization' in tasks
        assert len(tasks) == 7

    def test_predictable_properties(self):
        """Test that common properties are listed."""
        agent = MaterialsInformaticsAgent()
        props = agent.PREDICTABLE_PROPERTIES
        assert 'band_gap' in props
        assert 'formation_energy' in props


class TestMaterialsInformaticsValidation:
    """Test input validation."""

    def test_validation_missing_task(self):
        """Test validation with missing task."""
        agent = MaterialsInformaticsAgent()
        result = agent.validate_input({})
        assert not result.valid

    def test_validation_invalid_task(self):
        """Test validation with invalid task."""
        agent = MaterialsInformaticsAgent()
        result = agent.validate_input({'task': 'invalid_ml_task'})
        assert not result.valid

    def test_validation_property_prediction_valid(self):
        """Test validation for valid property prediction."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'property_prediction',
            'structures': [{'composition': 'NaCl'}]
        }
        result = agent.validate_input(input_data)
        assert result.valid


class TestMaterialsInformaticsPropertyPrediction:
    """Test property prediction functionality."""

    def test_execute_property_prediction(self):
        """Test property prediction execution."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'property_prediction',
            'structures': [{'id': 1}, {'id': 2}],
            'target_properties': ['band_gap', 'formation_energy']
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'predictions' in result.data
        assert len(result.data['predictions']) == 2
        assert 'model_performance' in result.data

    def test_predictions_have_uncertainty(self):
        """Test that predictions include uncertainty estimates."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'property_prediction',
            'structures': [{'id': 1}],
            'target_properties': ['band_gap']
        }
        result = agent.execute(input_data)

        pred = result.data['predictions'][0]
        assert 'predictions' in pred
        if 'band_gap_ev' in pred['predictions']:
            assert 'uncertainty' in pred['predictions']['band_gap_ev']
            assert 'confidence' in pred['predictions']['band_gap_ev']


class TestMaterialsInformaticsStructurePrediction:
    """Test structure prediction functionality."""

    def test_execute_structure_prediction(self):
        """Test structure prediction execution."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'structure_prediction',
            'composition': 'LiFePO4',
            'n_candidates': 5
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'candidates' in result.data
        assert len(result.data['candidates']) == 5
        assert 'most_stable' in result.data

    def test_candidates_have_stability(self):
        """Test that candidates include stability metrics."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'structure_prediction',
            'composition': 'TiO2'
        }
        result = agent.execute(input_data)

        for candidate in result.data['candidates']:
            assert 'formation_energy_ev_atom' in candidate
            assert 'stability_score' in candidate


class TestMaterialsInformaticsActiveLearning:
    """Test active learning functionality."""

    def test_execute_active_learning(self):
        """Test active learning execution."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'active_learning',
            'candidate_pool': list(range(100)),
            'n_select': 5,
            'strategy': 'uncertainty'
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'selected_experiments' in result.data
        assert len(result.data['selected_experiments']) == 5

    def test_selected_have_scores(self):
        """Test that selected experiments have selection scores."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'active_learning',
            'n_select': 3
        }
        result = agent.execute(input_data)

        for exp in result.data['selected_experiments']:
            assert 'selection_score' in exp
            assert 'expected_information_gain' in exp


class TestMaterialsInformaticsOptimization:
    """Test optimization functionality."""

    def test_execute_optimization(self):
        """Test Bayesian optimization execution."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'optimization',
            'target_properties': {'ionic_conductivity': 'maximize'},
            'n_iterations': 10
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'optimization_history' in result.data
        assert 'optimal_solution' in result.data
        assert 'convergence' in result.data

    def test_optimization_converges(self):
        """Test that optimization shows convergence."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'optimization',
            'n_iterations': 20
        }
        result = agent.execute(input_data)

        history = result.data['optimization_history']
        assert len(history) == 20
        # Best value should improve or stay constant
        best_values = [h['best_so_far'] for h in history]
        for i in range(1, len(best_values)):
            assert best_values[i] <= best_values[i-1]


class TestMaterialsInformaticsScreening:
    """Test high-throughput screening functionality."""

    def test_execute_screening(self):
        """Test screening execution."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'screening',
            'library_size': 1000,
            'top_n': 20
        }
        result = agent.execute(input_data)

        assert result.success
        assert 'top_candidates' in result.data
        assert 'screening_statistics' in result.data

    def test_screening_ranks_candidates(self):
        """Test that screening ranks candidates."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'screening',
            'top_n': 10
        }
        result = agent.execute(input_data)

        top = result.data['top_candidates']
        assert len(top) == 10
        # Should be ranked
        for i, candidate in enumerate(top):
            assert candidate['rank'] == i + 1


class TestMaterialsInformaticsResourceEstimation:
    """Test resource estimation."""

    def test_estimate_property_prediction_resources(self):
        """Test resource estimation for property prediction."""
        agent = MaterialsInformaticsAgent()
        input_data = {'task': 'property_prediction'}
        resources = agent.estimate_resources(input_data)

        assert resources.estimated_time_sec > 0
        assert resources.memory_gb > 0

    def test_estimate_screening_resources_scales(self):
        """Test that screening resources scale with library size."""
        agent = MaterialsInformaticsAgent()

        small_library = agent.estimate_resources({
            'task': 'screening',
            'library_size': 1000
        })

        large_library = agent.estimate_resources({
            'task': 'screening',
            'library_size': 10000
        })

        # Larger library should require more time
        assert large_library.estimated_time_sec > small_library.estimated_time_sec


class TestMaterialsInformaticsCapabilities:
    """Test capability reporting."""

    def test_get_capabilities(self):
        """Test capability listing."""
        agent = MaterialsInformaticsAgent()
        capabilities = agent.get_capabilities()

        assert len(capabilities) >= 5
        cap_names = [c.name for c in capabilities]
        assert 'property_prediction' in cap_names
        assert 'active_learning' in cap_names

    def test_capabilities_have_use_cases(self):
        """Test that capabilities have use cases."""
        agent = MaterialsInformaticsAgent()
        capabilities = agent.get_capabilities()

        for cap in capabilities:
            assert len(cap.typical_use_cases) > 0


class TestMaterialsInformaticsProvenance:
    """Test provenance tracking."""

    def test_provenance_tracking(self):
        """Test that execution metadata is tracked."""
        agent = MaterialsInformaticsAgent()
        input_data = {
            'task': 'property_prediction',
            'test_id': str(uuid.uuid4())
        }
        result = agent.execute(input_data)

        assert result.success
        assert result.provenance is not None
        assert result.provenance.agent_version == '1.0.0'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])