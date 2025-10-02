"""
Test suite for NonequilibriumMasterAgent

Tests cover:
- Initialization and metadata
- Input validation
- Resource estimation
- Workflow design (DAG construction)
- Workflow execution
- Technique optimization
- Cross-validation
- Result synthesis
- Automated pipelines
- Integration with other agents

Total: 50 tests
"""

import pytest
import numpy as np
import os
import tempfile
import json
from datetime import datetime
from pathlib import Path

# Import agent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from nonequilibrium_master_agent import NonequilibriumMasterAgent, WorkflowNode, WorkflowDAG
from base_agent import AgentResult, AgentStatus, ResourceRequirement


# Mock agent for testing
class MockAgent:
    """Mock agent for testing workflows."""

    def __init__(self, name):
        self.name = name
        self.VERSION = "1.0.0"

    def execute(self, input_data):
        """Mock execute method."""
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={'result': f'{self.name}_output', 'value': 42.0}
        )

    def get_metadata(self):
        """Mock metadata."""
        from base_agent import AgentMetadata
        return AgentMetadata(
            name=self.name,
            version=self.VERSION,
            description=f"Mock {self.name}",
            capabilities=[],
            agent_type="simulation"
        )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a NonequilibriumMasterAgent instance with mock agents."""
    mock_agents = {
        'TransportAgent': MockAgent('TransportAgent'),
        'ActiveMatterAgent': MockAgent('ActiveMatterAgent'),
        'FluctuationAgent': MockAgent('FluctuationAgent'),
        'PatternFormationAgent': MockAgent('PatternFormationAgent'),
        'InformationThermodynamicsAgent': MockAgent('InformationThermodynamicsAgent'),
        'SimulationAgent': MockAgent('SimulationAgent'),
        'RheologistAgent': MockAgent('RheologistAgent')
    }
    return NonequilibriumMasterAgent(agent_registry=mock_agents)


@pytest.fixture
def sample_workflow_design_input():
    """Sample input for workflow design."""
    return {
        'method': 'design_workflow',
        'goal': 'characterize_active_matter_system',
        'available_data': ['trajectory', 'light_scattering'],
        'constraints': {'max_agents': 5, 'time_limit': 3600}
    }


@pytest.fixture
def sample_optimize_input():
    """Sample input for technique optimization."""
    return {
        'method': 'optimize_techniques',
        'task': 'transport_characterization',
        'available_agents': ['TransportAgent', 'SimulationAgent', 'RheologistAgent'],
        'optimization_criteria': 'accuracy'
    }


@pytest.fixture
def sample_cross_validate_input():
    """Sample input for cross-validation."""
    return {
        'method': 'cross_validate',
        'results': [
            {'diffusion_coefficient': 1.0e-5, 'viscosity': 0.001},
            {'diffusion_coefficient': 1.05e-5, 'viscosity': 0.00095},
            {'diffusion_coefficient': 0.98e-5, 'viscosity': 0.00102}
        ]
    }


@pytest.fixture
def sample_synthesize_input():
    """Sample input for result synthesis."""
    return {
        'method': 'synthesize_results',
        'agent_results': {
            'TransportAgent': {'thermal_conductivity': 0.6, 'diffusion': 1e-5},
            'SimulationAgent': {'thermal_conductivity': 0.58, 'energy': -1000},
            'RheologistAgent': {'viscosity': 0.001, 'modulus': 1e6}
        }
    }


@pytest.fixture
def sample_automated_pipeline_input():
    """Sample input for automated pipeline."""
    return {
        'method': 'automated_pipeline',
        'goal': 'full_characterization',
        'available_data': ['trajectory', 'temperature_profile'],
        'parameters': {'temperature': 300.0}
    }


# ============================================================================
# Test 1-5: Initialization and Metadata
# ============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent is not None
    assert agent.VERSION == "1.0.0"
    assert len(agent.agent_registry) > 0
    assert 'TransportAgent' in agent.agent_registry


def test_agent_metadata(agent):
    """Test agent metadata is correct."""
    metadata = agent.get_metadata()
    assert metadata.name == "NonequilibriumMasterAgent"
    assert metadata.version == "1.0.0"
    assert metadata.agent_type == "coordination"


def test_agent_capabilities(agent):
    """Test agent capabilities."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) == 5
    method_names = [cap.name for cap in capabilities]
    assert 'design_workflow' in method_names
    assert 'optimize_techniques' in method_names
    assert 'cross_validate' in method_names


def test_supported_methods(agent):
    """Test all methods are supported."""
    assert len(agent.supported_methods) == 5
    expected_methods = ['design_workflow', 'optimize_techniques', 'cross_validate',
                       'synthesize_results', 'automated_pipeline']
    for method in expected_methods:
        assert method in agent.supported_methods


def test_agent_registry(agent):
    """Test agent registry is properly initialized."""
    assert len(agent.agent_registry) >= 5
    assert all(hasattr(agent, 'execute') for agent in agent.agent_registry.values())


# ============================================================================
# Test 6-10: Workflow DAG Structure
# ============================================================================

def test_workflow_node_creation():
    """Test WorkflowNode creation."""
    node = WorkflowNode(
        node_id='test_node',
        agent_name='TestAgent',
        method='test_method',
        parameters={'param1': 'value1'},
        dependencies=['dep1']
    )
    assert node.node_id == 'test_node'
    assert node.agent_name == 'TestAgent'
    assert node.status == AgentStatus.PENDING


def test_workflow_dag_creation():
    """Test WorkflowDAG creation."""
    workflow = WorkflowDAG('test_workflow', 'Test workflow')
    assert workflow.workflow_id == 'test_workflow'
    assert workflow.description == 'Test workflow'
    assert len(workflow.nodes) == 0


def test_workflow_add_node():
    """Test adding nodes to workflow."""
    workflow = WorkflowDAG('test_workflow')
    node = WorkflowNode('node1', 'Agent1', 'method1', {})
    workflow.add_node(node)
    assert 'node1' in workflow.nodes
    assert workflow.nodes['node1'] == node


def test_workflow_execution_order_simple():
    """Test execution order computation for simple DAG."""
    workflow = WorkflowDAG('test')
    workflow.add_node(WorkflowNode('n1', 'A1', 'm1', {}, []))
    workflow.add_node(WorkflowNode('n2', 'A2', 'm2', {}, ['n1']))
    workflow.add_node(WorkflowNode('n3', 'A3', 'm3', {}, ['n2']))

    order = workflow.compute_execution_order()
    assert order == ['n1', 'n2', 'n3']


def test_workflow_execution_order_parallel():
    """Test execution order for parallel branches."""
    workflow = WorkflowDAG('test')
    workflow.add_node(WorkflowNode('n1', 'A1', 'm1', {}, []))
    workflow.add_node(WorkflowNode('n2', 'A2', 'm2', {}, ['n1']))
    workflow.add_node(WorkflowNode('n3', 'A3', 'm3', {}, ['n1']))
    workflow.add_node(WorkflowNode('n4', 'A4', 'm4', {}, ['n2', 'n3']))

    order = workflow.compute_execution_order()
    assert order[0] == 'n1'
    assert order[-1] == 'n4'
    assert set(order[1:3]) == {'n2', 'n3'}


# ============================================================================
# Test 11-15: Input Validation
# ============================================================================

def test_validate_workflow_design_input_valid(agent, sample_workflow_design_input):
    """Test validation accepts valid workflow design input."""
    result = agent.validate_input(sample_workflow_design_input)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_optimize_input_valid(agent, sample_optimize_input):
    """Test validation accepts valid optimization input."""
    result = agent.validate_input(sample_optimize_input)
    assert result.valid is True


def test_validate_missing_method(agent):
    """Test validation fails with missing method."""
    result = agent.validate_input({})
    assert result.valid is False
    assert any('method' in err.lower() for err in result.errors)


def test_validate_invalid_method(agent):
    """Test validation fails with invalid method."""
    result = agent.validate_input({'method': 'invalid_method'})
    assert result.valid is False


def test_validate_cross_validate_missing_results(agent):
    """Test validation fails for cross_validate without results."""
    result = agent.validate_input({'method': 'cross_validate'})
    assert result.valid is False
    assert any('results' in err.lower() for err in result.errors)


# ============================================================================
# Test 16-20: Resource Estimation
# ============================================================================

def test_resource_estimation_design_workflow(agent, sample_workflow_design_input):
    """Test resource estimation for workflow design."""
    req = agent.estimate_resources(sample_workflow_design_input)
    assert req.environment == 'LOCAL'
    assert req.cpu_cores >= 1
    assert req.memory_gb >= 0.5


def test_resource_estimation_optimize_techniques(agent, sample_optimize_input):
    """Test resource estimation for technique optimization."""
    req = agent.estimate_resources(sample_optimize_input)
    assert req.memory_gb > 0
    assert req.estimated_duration_seconds > 0


def test_resource_estimation_automated_pipeline(agent, sample_automated_pipeline_input):
    """Test resource estimation for automated pipeline."""
    req = agent.estimate_resources(sample_automated_pipeline_input)
    # Automated pipeline should request significant resources
    assert req.cpu_cores >= 4 or req.environment == 'HPC'


def test_resource_estimation_cross_validate(agent, sample_cross_validate_input):
    """Test resource estimation for cross-validation."""
    req = agent.estimate_resources(sample_cross_validate_input)
    assert req.environment in ['LOCAL', 'HPC']


def test_resource_estimation_execute_workflow(agent):
    """Test resource estimation for workflow execution."""
    workflow_input = {
        'method': 'execute_workflow',
        'workflow': {
            'workflow_id': 'test123',
            'nodes': [{'node_id': f'n{i}', 'agent_name': 'TestAgent', 'method': 'test'} for i in range(10)]
        }
    }
    req = agent.estimate_resources(workflow_input)
    # Large workflow should require HPC
    assert req.cpu_cores >= 4


# ============================================================================
# Test 21-25: Workflow Design
# ============================================================================

def test_execute_design_workflow_active_matter(agent):
    """Test workflow design for active matter characterization."""
    input_data = {
        'method': 'design_workflow',
        'goal': 'characterize_active_matter_system',
        'available_data': ['trajectory']
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'workflow_id' in result.data
    assert 'nodes' in result.data
    assert len(result.data['nodes']) >= 2


def test_execute_design_workflow_fluctuation(agent):
    """Test workflow design for fluctuation theorem validation."""
    input_data = {
        'method': 'design_workflow',
        'goal': 'validate_fluctuation_theorem',
        'available_data': ['work_distribution']
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'workflow_id' in result.data
    # Should include driven systems, fluctuation, and information agents
    agent_names = [node['agent_name'] for node in result.data['nodes']]
    assert 'FluctuationAgent' in agent_names


def test_execute_design_workflow_transport(agent):
    """Test workflow design for transport characterization."""
    input_data = {
        'method': 'design_workflow',
        'goal': 'transport_characterization',
        'available_data': ['trajectory']
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'TransportAgent' in str(result.data)


def test_workflow_design_execution_order(agent, sample_workflow_design_input):
    """Test that designed workflow has valid execution order."""
    result = agent.execute(sample_workflow_design_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'execution_order' in result.data
    assert len(result.data['execution_order']) == len(result.data['nodes'])


def test_workflow_design_caching(agent, sample_workflow_design_input):
    """Test that designed workflows are cached."""
    result = agent.execute(sample_workflow_design_input)
    workflow_id = result.data['workflow_id']
    assert workflow_id in agent.workflow_cache


# ============================================================================
# Test 26-30: Technique Optimization
# ============================================================================

def test_execute_optimize_techniques_transport(agent):
    """Test technique optimization for transport task."""
    input_data = {
        'method': 'optimize_techniques',
        'task': 'transport_characterization',
        'available_agents': list(agent.agent_registry.keys())
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'recommended_agents' in result.data
    assert 'TransportAgent' in result.data['recommended_agents']


def test_execute_optimize_techniques_active_matter(agent):
    """Test technique optimization for active matter task."""
    input_data = {
        'method': 'optimize_techniques',
        'task': 'active_matter_analysis',
        'available_agents': list(agent.agent_registry.keys())
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'ActiveMatterAgent' in result.data['recommended_agents']


def test_execute_optimize_techniques_information(agent):
    """Test technique optimization for information thermodynamics."""
    input_data = {
        'method': 'optimize_techniques',
        'task': 'information_flow_analysis',
        'available_agents': list(agent.agent_registry.keys())
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'InformationThermodynamicsAgent' in result.data['recommended_agents']


def test_optimize_techniques_confidence_score(agent, sample_optimize_input):
    """Test that optimization returns confidence score."""
    result = agent.execute(sample_optimize_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'confidence' in result.data
    assert 0 <= result.data['confidence'] <= 1.0


def test_optimize_techniques_agent_scores(agent, sample_optimize_input):
    """Test that optimization returns agent scores."""
    result = agent.execute(sample_optimize_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'agent_scores' in result.data
    assert len(result.data['agent_scores']) > 0


# ============================================================================
# Test 31-35: Cross-Validation
# ============================================================================

def test_execute_cross_validate_basic(agent, sample_cross_validate_input):
    """Test basic cross-validation."""
    result = agent.execute(sample_cross_validate_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'validation_status' in result.data
    assert result.data['validation_status'] == 'complete'


def test_cross_validate_consistency_metrics(agent, sample_cross_validate_input):
    """Test cross-validation computes consistency metrics."""
    result = agent.execute(sample_cross_validate_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'validation_metrics' in result.data
    assert 'diffusion_coefficient' in result.data['validation_metrics']


def test_cross_validate_consistency_score(agent, sample_cross_validate_input):
    """Test cross-validation computes overall consistency score."""
    result = agent.execute(sample_cross_validate_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'consistency_score' in result.data
    assert 0 <= result.data['consistency_score'] <= 1.0


def test_cross_validate_insufficient_data(agent):
    """Test cross-validation with insufficient data."""
    input_data = {
        'method': 'cross_validate',
        'results': [{'metric': 1.0}]  # Only one result
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert result.data['validation_status'] == 'insufficient_data'


def test_cross_validate_relative_std(agent, sample_cross_validate_input):
    """Test cross-validation computes relative standard deviation."""
    result = agent.execute(sample_cross_validate_input)
    assert result.status == AgentStatus.SUCCESS
    metrics = result.data['validation_metrics']
    for metric_data in metrics.values():
        assert 'relative_std' in metric_data
        assert metric_data['relative_std'] >= 0


# ============================================================================
# Test 36-40: Result Synthesis
# ============================================================================

def test_execute_synthesize_results_basic(agent, sample_synthesize_input):
    """Test basic result synthesis."""
    result = agent.execute(sample_synthesize_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'synthesized_insights' in result.data


def test_synthesize_results_mean_computation(agent, sample_synthesize_input):
    """Test synthesis computes mean values."""
    result = agent.execute(sample_synthesize_input)
    assert result.status == AgentStatus.SUCCESS
    insights = result.data['synthesized_insights']
    for metric, data in insights.items():
        assert 'mean' in data
        assert isinstance(data['mean'], (int, float))


def test_synthesize_results_confidence(agent, sample_synthesize_input):
    """Test synthesis assigns confidence levels."""
    result = agent.execute(sample_synthesize_input)
    assert result.status == AgentStatus.SUCCESS
    insights = result.data['synthesized_insights']
    for metric, data in insights.items():
        assert 'confidence' in data
        assert data['confidence'] in ['low', 'medium', 'high']


def test_synthesize_results_summary(agent, sample_synthesize_input):
    """Test synthesis generates summary."""
    result = agent.execute(sample_synthesize_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'summary' in result.data
    assert 'n_agents_involved' in result.data['summary']


def test_synthesize_results_no_data(agent):
    """Test synthesis with no data."""
    input_data = {
        'method': 'synthesize_results',
        'agent_results': {}
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert result.data['synthesis_status'] == 'no_data'


# ============================================================================
# Test 41-45: Automated Pipeline
# ============================================================================

def test_execute_automated_pipeline_basic(agent, sample_automated_pipeline_input):
    """Test basic automated pipeline execution."""
    result = agent.execute(sample_automated_pipeline_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'pipeline_status' in result.data


def test_automated_pipeline_workflow_design(agent, sample_automated_pipeline_input):
    """Test automated pipeline includes workflow design."""
    result = agent.execute(sample_automated_pipeline_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'workflow_design' in result.data
    assert 'workflow_id' in result.data['workflow_design']


def test_automated_pipeline_execution_results(agent, sample_automated_pipeline_input):
    """Test automated pipeline includes execution results."""
    result = agent.execute(sample_automated_pipeline_input)
    assert result.status == AgentStatus.SUCCESS
    assert 'execution_results' in result.data


def test_automated_pipeline_cross_validation(agent, sample_automated_pipeline_input):
    """Test automated pipeline performs cross-validation."""
    result = agent.execute(sample_automated_pipeline_input)
    assert result.status == AgentStatus.SUCCESS
    # May include cross-validation if multiple nodes
    if 'cross_validation' in result.data['execution_results']:
        assert 'validation_status' in result.data['execution_results']['cross_validation']


def test_automated_pipeline_complete_status(agent, sample_automated_pipeline_input):
    """Test automated pipeline reports complete status."""
    result = agent.execute(sample_automated_pipeline_input)
    assert result.status == AgentStatus.SUCCESS
    assert result.data['pipeline_status'] == 'complete'


# ============================================================================
# Test 46-50: Workflow Execution and Integration
# ============================================================================

def test_execute_workflow_simple(agent):
    """Test execution of simple workflow."""
    workflow_spec = {
        'workflow_id': 'test_wf',
        'description': 'Test workflow',
        'nodes': [
            {'node_id': 'n1', 'agent_name': 'TransportAgent', 'method': 'thermal_conductivity', 'parameters': {}}
        ]
    }
    input_data = {
        'method': 'execute_workflow',
        'workflow': workflow_spec
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'results' in result.data


def test_execute_workflow_sequential(agent):
    """Test execution of sequential workflow."""
    workflow_spec = {
        'workflow_id': 'seq_wf',
        'nodes': [
            {'node_id': 'n1', 'agent_name': 'TransportAgent', 'method': 'thermal_conductivity', 'parameters': {}, 'dependencies': []},
            {'node_id': 'n2', 'agent_name': 'FluctuationAgent', 'method': 'jarzynski', 'parameters': {}, 'dependencies': ['n1']}
        ]
    }
    input_data = {
        'method': 'execute_workflow',
        'workflow': workflow_spec
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert result.data['nodes_executed'] == 2


def test_execute_workflow_success_rate(agent):
    """Test workflow execution reports success rate."""
    workflow_spec = {
        'workflow_id': 'test_success',
        'nodes': [
            {'node_id': 'n1', 'agent_name': 'TransportAgent', 'method': 'test', 'parameters': {}}
        ]
    }
    input_data = {
        'method': 'execute_workflow',
        'workflow': workflow_spec
    }
    result = agent.execute(input_data)
    assert result.status == AgentStatus.SUCCESS
    assert 'success_rate' in result.data
    assert 0 <= result.data['success_rate'] <= 1.0


def test_provenance_tracking(agent, sample_workflow_design_input):
    """Test that master agent tracks provenance."""
    result = agent.execute(sample_workflow_design_input)
    assert result.provenance is not None
    assert result.provenance.agent_name == "NonequilibriumMasterAgent"
    assert result.provenance.agent_version == "1.0.0"


def test_metadata_generation(agent, sample_workflow_design_input):
    """Test that master agent generates metadata."""
    result = agent.execute(sample_workflow_design_input)
    assert result.metadata is not None
    assert 'method' in result.metadata
    assert 'goal' in result.metadata
    assert result.metadata['method'] == 'design_workflow'