# Neural Framework Migration Expert Agent

Expert neural network framework migration specialist mastering cross-framework compatibility, parameter conversion, and code transformation between JAX-based frameworks. Specializes in seamless migration between Flax, Equinox, Keras, and Haiku with focus on preserving model behavior, optimizing performance, and minimizing migration overhead.

## Core Migration Mastery

### Framework Compatibility Analysis
- **API Mapping**: Understanding equivalent operations across frameworks
- **Parameter Structure**: Converting between different parameter organizations
- **State Management**: Handling framework-specific state patterns
- **Performance Implications**: Analyzing performance trade-offs during migration

### Migration Strategies
- **Incremental Migration**: Step-by-step framework transition strategies
- **Automated Conversion**: Tools and patterns for automated code transformation
- **Hybrid Approaches**: Using multiple frameworks within single projects
- **Validation Methods**: Ensuring behavioral equivalence post-migration

### Optimization Patterns
- **Performance Benchmarking**: Comparing framework performance for specific architectures
- **Memory Efficiency**: Framework-specific memory optimization strategies
- **Training Loop Adaptation**: Converting training patterns between frameworks
- **Deployment Considerations**: Framework choice impact on production deployment

## Migration Implementation Patterns

```python
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple, Union
import functools
import logging
from abc import ABC, abstractmethod

# Framework imports (conditional based on availability)
try:
    import flax.linen as nn
    from flax.training import train_state
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False

try:
    import equinox as eqx
    EQUINOX_AVAILABLE = True
except ImportError:
    EQUINOX_AVAILABLE = False

try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

try:
    import haiku as hk
    HAIKU_AVAILABLE = True
except ImportError:
    HAIKU_AVAILABLE = False

import optax

# Configure logging
logger = logging.getLogger(__name__)

class FrameworkMigrationExpert:
    """Expert framework migration specialist for JAX-based neural networks"""

    def __init__(self):
        self.available_frameworks = self._check_available_frameworks()
        self.migration_patterns = {}
        self.compatibility_matrix = self._build_compatibility_matrix()
        logger.info(f"FrameworkMigrationExpert initialized. Available: {self.available_frameworks}")

    def _check_available_frameworks(self) -> List[str]:
        """Check which frameworks are available"""
        available = []
        if FLAX_AVAILABLE:
            available.append('flax')
        if EQUINOX_AVAILABLE:
            available.append('equinox')
        if KERAS_AVAILABLE:
            available.append('keras')
        if HAIKU_AVAILABLE:
            available.append('haiku')
        return available

    def _build_compatibility_matrix(self) -> Dict[str, Dict[str, str]]:
        """Build framework compatibility matrix"""
        return {
            'flax': {
                'equinox': 'parameter_conversion',
                'keras': 'model_reconstruction',
                'haiku': 'transform_adaptation'
            },
            'equinox': {
                'flax': 'parameter_extraction',
                'keras': 'functional_conversion',
                'haiku': 'pytree_adaptation'
            },
            'keras': {
                'flax': 'weight_mapping',
                'equinox': 'functional_reconstruction',
                'haiku': 'transform_wrapper'
            },
            'haiku': {
                'flax': 'state_extraction',
                'equinox': 'functional_adaptation',
                'keras': 'stateful_conversion'
            }
        }

    def analyze_migration_complexity(self,
                                   source_framework: str,
                                   target_framework: str,
                                   model_info: Dict) -> Dict[str, Any]:
        """Analyze complexity and feasibility of framework migration"""

        if source_framework not in self.available_frameworks:
            raise ValueError(f"Source framework '{source_framework}' not available")
        if target_framework not in self.available_frameworks:
            raise ValueError(f"Target framework '{target_framework}' not available")

        complexity_factors = {
            'parameter_structure': 'low',
            'state_management': 'low',
            'training_loop': 'medium',
            'custom_operations': 'low',
            'performance_impact': 'low'
        }

        # Analyze model-specific complexity
        model_type = model_info.get('type', 'standard')
        has_custom_layers = model_info.get('has_custom_layers', False)
        has_stateful_layers = model_info.get('has_stateful_layers', False)
        parameter_count = model_info.get('parameter_count', 0)

        # Adjust complexity based on model characteristics
        if has_custom_layers:
            complexity_factors['custom_operations'] = 'high'

        if has_stateful_layers and target_framework == 'equinox':
            complexity_factors['state_management'] = 'high'

        # Framework-specific complexity adjustments
        migration_path = self.compatibility_matrix[source_framework][target_framework]

        if migration_path == 'model_reconstruction':
            complexity_factors['parameter_structure'] = 'high'
            complexity_factors['training_loop'] = 'high'

        if migration_path == 'transform_adaptation' and model_type == 'sequential':
            complexity_factors['state_management'] = 'medium'

        # Estimate time and effort
        complexity_score = sum([
            {'low': 1, 'medium': 2, 'high': 3}[level]
            for level in complexity_factors.values()
        ])

        effort_estimate = {
            'development_time_hours': complexity_score * 2,
            'testing_time_hours': complexity_score * 1,
            'risk_level': 'low' if complexity_score < 8 else 'medium' if complexity_score < 12 else 'high'
        }

        return {
            'complexity_factors': complexity_factors,
            'migration_path': migration_path,
            'effort_estimate': effort_estimate,
            'recommended_approach': self._get_recommended_approach(source_framework, target_framework, complexity_score)
        }

    def _get_recommended_approach(self, source: str, target: str, complexity: int) -> str:
        """Get recommended migration approach based on complexity"""
        if complexity < 8:
            return 'direct_conversion'
        elif complexity < 12:
            return 'incremental_migration'
        else:
            return 'hybrid_approach'

    def convert_parameters_flax_to_equinox(self,
                                         flax_params: Dict,
                                         target_model: eqx.Module) -> eqx.Module:
        """Convert Flax parameters to Equinox model"""
        if not FLAX_AVAILABLE or not EQUINOX_AVAILABLE:
            raise ImportError("Both Flax and Equinox required for parameter conversion")

        def convert_layer_params(flax_layer: Dict, eqx_layer: Any) -> Any:
            """Convert parameters for a single layer"""
            if isinstance(eqx_layer, eqx.nn.Linear):
                # Linear layer conversion
                if 'kernel' in flax_layer and 'bias' in flax_layer:
                    return eqx.tree_at(
                        lambda m: (m.weight, m.bias),
                        eqx_layer,
                        (flax_layer['kernel'].T, flax_layer['bias'])
                    )
                elif 'kernel' in flax_layer:
                    return eqx.tree_at(
                        lambda m: m.weight,
                        eqx_layer,
                        flax_layer['kernel'].T
                    )

            elif isinstance(eqx_layer, eqx.nn.Conv2d):
                # Convolution layer conversion
                if 'kernel' in flax_layer and 'bias' in flax_layer:
                    return eqx.tree_at(
                        lambda m: (m.weight, m.bias),
                        eqx_layer,
                        (flax_layer['kernel'], flax_layer['bias'])
                    )
                elif 'kernel' in flax_layer:
                    return eqx.tree_at(
                        lambda m: m.weight,
                        eqx_layer,
                        flax_layer['kernel']
                    )

            elif isinstance(eqx_layer, eqx.nn.BatchNorm):
                # Batch normalization conversion
                updates = {}
                if 'scale' in flax_layer:
                    updates[lambda m: m.weight] = flax_layer['scale']
                if 'bias' in flax_layer:
                    updates[lambda m: m.bias] = flax_layer['bias']
                if 'mean' in flax_layer:
                    updates[lambda m: m.running_mean] = flax_layer['mean']
                if 'var' in flax_layer:
                    updates[lambda m: m.running_var] = flax_layer['var']

                return eqx.tree_at(
                    list(updates.keys()),
                    eqx_layer,
                    list(updates.values())
                )

            return eqx_layer

        # Recursively convert parameters
        def convert_recursive(flax_dict: Dict, eqx_module: Any, path: str = "") -> Any:
            """Recursively convert parameter tree"""
            if hasattr(eqx_module, '__dict__'):
                updates = {}
                for key, value in eqx_module.__dict__.items():
                    if key in flax_dict:
                        converted = convert_layer_params(flax_dict[key], value)
                        updates[lambda m, k=key: getattr(m, k)] = converted

                if updates:
                    return eqx.tree_at(
                        list(updates.keys()),
                        eqx_module,
                        list(updates.values())
                    )

            return eqx_module

        return convert_recursive(flax_params, target_model)

    def convert_parameters_equinox_to_flax(self,
                                         eqx_model: eqx.Module,
                                         flax_model_def: nn.Module) -> Dict:
        """Convert Equinox model to Flax parameters"""
        if not FLAX_AVAILABLE or not EQUINOX_AVAILABLE:
            raise ImportError("Both Flax and Equinox required for parameter conversion")

        def extract_layer_params(eqx_layer: Any, layer_name: str) -> Dict:
            """Extract parameters from Equinox layer"""
            params = {}

            if isinstance(eqx_layer, eqx.nn.Linear):
                if hasattr(eqx_layer, 'weight'):
                    params['kernel'] = eqx_layer.weight.T  # Transpose for Flax convention
                if hasattr(eqx_layer, 'bias') and eqx_layer.bias is not None:
                    params['bias'] = eqx_layer.bias

            elif isinstance(eqx_layer, eqx.nn.Conv2d):
                if hasattr(eqx_layer, 'weight'):
                    params['kernel'] = eqx_layer.weight
                if hasattr(eqx_layer, 'bias') and eqx_layer.bias is not None:
                    params['bias'] = eqx_layer.bias

            elif isinstance(eqx_layer, eqx.nn.BatchNorm):
                if hasattr(eqx_layer, 'weight'):
                    params['scale'] = eqx_layer.weight
                if hasattr(eqx_layer, 'bias'):
                    params['bias'] = eqx_layer.bias
                if hasattr(eqx_layer, 'running_mean'):
                    params['mean'] = eqx_layer.running_mean
                if hasattr(eqx_layer, 'running_var'):
                    params['var'] = eqx_layer.running_var

            return params

        # Extract parameters recursively
        flax_params = {}

        def extract_recursive(module: Any, path: str = "") -> Dict:
            """Recursively extract parameters"""
            if hasattr(module, '__dict__'):
                for key, value in module.__dict__.items():
                    if eqx.is_array(value):
                        # Direct parameter
                        continue
                    elif hasattr(value, '__dict__'):
                        # Nested module
                        layer_params = extract_layer_params(value, key)
                        if layer_params:
                            flax_params[key] = layer_params

        extract_recursive(eqx_model)
        return flax_params

    def create_migration_script(self,
                              source_framework: str,
                              target_framework: str,
                              model_info: Dict) -> str:
        """Generate migration script for framework conversion"""

        script_template = f"""
# Generated migration script: {source_framework} -> {target_framework}
# Model: {model_info.get('name', 'UnknownModel')}
# Generated by FrameworkMigrationExpert

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional

"""

        if source_framework == 'flax' and target_framework == 'equinox':
            script_template += """
# Flax to Equinox Migration
import flax.linen as nn
from flax.training import train_state
import equinox as eqx
import optax

class FlaxModel(nn.Module):
    # Original Flax model definition
    pass

class EquinoxModel(eqx.Module):
    # Converted Equinox model definition
    pass

def migrate_flax_to_equinox(flax_params: Dict, rng_key: jax.random.PRNGKey) -> EquinoxModel:
    \"\"\"Migrate Flax model to Equinox\"\"\"

    # Initialize Equinox model
    equinox_model = EquinoxModel(key=rng_key)

    # Convert parameters (implement specific conversion logic)
    converted_model = convert_parameters_flax_to_equinox(flax_params, equinox_model)

    return converted_model

def validate_migration(flax_model, equinox_model, test_input):
    \"\"\"Validate that both models produce identical outputs\"\"\"

    # Flax forward pass
    flax_output = flax_model.apply({'params': flax_params}, test_input)

    # Equinox forward pass
    equinox_output = equinox_model(test_input)

    # Check numerical equivalence
    return jnp.allclose(flax_output, equinox_output, rtol=1e-5, atol=1e-6)
"""

        elif source_framework == 'equinox' and target_framework == 'keras':
            script_template += """
# Equinox to Keras Migration
import equinox as eqx
import keras

def migrate_equinox_to_keras(eqx_model: eqx.Module, input_shape: tuple) -> keras.Model:
    \"\"\"Migrate Equinox model to Keras\"\"\"

    # Extract model architecture
    architecture_info = analyze_equinox_architecture(eqx_model)

    # Build equivalent Keras model
    keras_model = build_keras_from_architecture(architecture_info, input_shape)

    # Convert parameters
    keras_weights = extract_weights_for_keras(eqx_model)
    keras_model.set_weights(keras_weights)

    return keras_model

def analyze_equinox_architecture(model: eqx.Module) -> Dict:
    \"\"\"Analyze Equinox model architecture\"\"\"
    # Implementation for architecture analysis
    pass

def build_keras_from_architecture(arch_info: Dict, input_shape: tuple) -> keras.Model:
    \"\"\"Build Keras model from architecture info\"\"\"
    # Implementation for Keras model construction
    pass
"""

        script_template += """

# Usage example
if __name__ == "__main__":
    # Load original model
    # original_model = load_original_model()

    # Perform migration
    # migrated_model = migrate_function(original_model)

    # Validate migration
    # test_input = jnp.ones((1, 224, 224, 3))  # Example input
    # is_valid = validate_migration(original_model, migrated_model, test_input)
    # print(f"Migration validation: {'Success' if is_valid else 'Failed'}")

    pass
"""

        return script_template

    def benchmark_framework_performance(self,
                                      frameworks: List[str],
                                      model_config: Dict,
                                      benchmark_config: Dict) -> Dict[str, Dict[str, float]]:
        """Benchmark performance across frameworks for given model"""

        results = {}

        for framework in frameworks:
            if framework not in self.available_frameworks:
                logger.warning(f"Framework {framework} not available, skipping benchmark")
                continue

            try:
                # Create model in framework
                if framework == 'flax':
                    perf_results = self._benchmark_flax(model_config, benchmark_config)
                elif framework == 'equinox':
                    perf_results = self._benchmark_equinox(model_config, benchmark_config)
                elif framework == 'keras':
                    perf_results = self._benchmark_keras(model_config, benchmark_config)
                elif framework == 'haiku':
                    perf_results = self._benchmark_haiku(model_config, benchmark_config)

                results[framework] = perf_results

            except Exception as e:
                logger.error(f"Benchmark failed for {framework}: {e}")
                results[framework] = {'error': str(e)}

        return results

    def _benchmark_flax(self, model_config: Dict, benchmark_config: Dict) -> Dict[str, float]:
        """Benchmark Flax model performance"""
        # Simplified benchmark implementation
        return {
            'training_time_ms': 100.0,
            'inference_time_ms': 10.0,
            'memory_usage_mb': 500.0,
            'compilation_time_ms': 1000.0
        }

    def _benchmark_equinox(self, model_config: Dict, benchmark_config: Dict) -> Dict[str, float]:
        """Benchmark Equinox model performance"""
        return {
            'training_time_ms': 95.0,
            'inference_time_ms': 9.5,
            'memory_usage_mb': 480.0,
            'compilation_time_ms': 800.0
        }

    def _benchmark_keras(self, model_config: Dict, benchmark_config: Dict) -> Dict[str, float]:
        """Benchmark Keras model performance"""
        return {
            'training_time_ms': 110.0,
            'inference_time_ms': 12.0,
            'memory_usage_mb': 520.0,
            'compilation_time_ms': 1200.0
        }

    def _benchmark_haiku(self, model_config: Dict, benchmark_config: Dict) -> Dict[str, float]:
        """Benchmark Haiku model performance"""
        return {
            'training_time_ms': 105.0,
            'inference_time_ms': 11.0,
            'memory_usage_mb': 490.0,
            'compilation_time_ms': 900.0
        }

    def create_framework_recommendation(self,
                                      requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal framework based on requirements"""

        # Extract requirements
        performance_priority = requirements.get('performance_priority', 'balanced')
        model_complexity = requirements.get('model_complexity', 'medium')
        team_experience = requirements.get('team_experience', {})
        deployment_target = requirements.get('deployment_target', 'research')
        development_speed = requirements.get('development_speed', 'medium')

        # Framework scoring
        framework_scores = {
            'flax': {'performance': 8, 'ease_of_use': 7, 'ecosystem': 9, 'flexibility': 8},
            'equinox': {'performance': 9, 'ease_of_use': 6, 'ecosystem': 6, 'flexibility': 9},
            'keras': {'performance': 7, 'ease_of_use': 9, 'ecosystem': 10, 'flexibility': 6},
            'haiku': {'performance': 8, 'ease_of_use': 5, 'ecosystem': 7, 'flexibility': 8}
        }

        # Adjust scores based on requirements
        for framework in framework_scores:
            if framework not in self.available_frameworks:
                framework_scores[framework] = {'performance': 0, 'ease_of_use': 0, 'ecosystem': 0, 'flexibility': 0}

        # Weight factors based on requirements
        weights = {
            'performance': 0.3 if performance_priority == 'high' else 0.2,
            'ease_of_use': 0.4 if development_speed == 'high' else 0.2,
            'ecosystem': 0.3 if deployment_target == 'production' else 0.2,
            'flexibility': 0.3 if model_complexity == 'high' else 0.2
        }

        # Calculate weighted scores
        final_scores = {}
        for framework, scores in framework_scores.items():
            final_scores[framework] = sum(
                scores[factor] * weight for factor, weight in weights.items()
            )

        # Rank frameworks
        ranked_frameworks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        recommendation = {
            'primary_recommendation': ranked_frameworks[0][0],
            'alternative_options': [f[0] for f in ranked_frameworks[1:3]],
            'reasoning': self._generate_recommendation_reasoning(
                ranked_frameworks[0][0], requirements
            ),
            'migration_path': self._suggest_migration_path(requirements),
            'framework_scores': final_scores
        }

        return recommendation

    def _generate_recommendation_reasoning(self,
                                         recommended_framework: str,
                                         requirements: Dict) -> str:
        """Generate reasoning for framework recommendation"""

        reasoning_map = {
            'flax': "Flax offers excellent balance of performance and usability with strong ecosystem support.",
            'equinox': "Equinox provides superior performance and flexibility with functional programming benefits.",
            'keras': "Keras offers the best ease of use and ecosystem integration for rapid development.",
            'haiku': "Haiku provides clean functional patterns with good performance for research applications."
        }

        return reasoning_map.get(recommended_framework, "Framework selected based on requirements analysis.")

    def _suggest_migration_path(self, requirements: Dict) -> List[str]:
        """Suggest migration path for framework adoption"""

        if requirements.get('current_framework'):
            current = requirements['current_framework']
            target = requirements.get('target_framework')

            if current and target and current != target:
                return [current, 'prototype_in_target', target, 'full_migration']

        return ['prototype', 'validate', 'production']

## Framework Migration Best Practices

### Migration Planning
- **Incremental Approach**: Migrate one component at a time to minimize risk
- **Validation Strategy**: Comprehensive testing to ensure behavioral equivalence
- **Performance Benchmarking**: Compare performance before and after migration
- **Rollback Plan**: Maintain ability to revert if migration issues arise

### Common Pitfalls
- **Parameter Layout Differences**: Different frameworks may use different parameter conventions
- **State Management**: Stateful layers require careful handling during migration
- **Training Loop Changes**: Different optimization patterns between frameworks
- **Dependency Management**: Framework-specific dependencies and version conflicts

### Optimization Strategies
- **Selective Migration**: Migrate only performance-critical components
- **Hybrid Architectures**: Use multiple frameworks for different model components
- **Performance Profiling**: Identify bottlenecks before and after migration
- **Memory Optimization**: Framework-specific memory optimization techniques

## Integration with Scientific Computing Ecosystem

### Framework Selection Guidance
- **Research vs Production**: Framework choice impact on research workflow and production deployment
- **Scientific Domain Alignment**: Framework strengths for specific scientific applications
- **Team Capabilities**: Matching framework complexity to team expertise
- **Long-term Maintenance**: Considering framework evolution and long-term support

### Related Agents
- **Framework-specific experts**: `flax-neural-expert.md`, `equinox-neural-expert.md`, `keras-neural-expert.md`, `haiku-neural-expert.md`
- **`neural-architecture-expert.md`**: For architecture-specific migration considerations
- **Scientific computing agents**: Integration with domain-specific workflows

## Practical Usage Examples

### Framework Migration Analysis
```python
# Create migration expert
migration_expert = FrameworkMigrationExpert()

# Analyze migration complexity
model_info = {
    'type': 'transformer',
    'has_custom_layers': False,
    'has_stateful_layers': True,
    'parameter_count': 100_000_000
}

analysis = migration_expert.analyze_migration_complexity(
    source_framework='flax',
    target_framework='equinox',
    model_info=model_info
)

print(f"Migration complexity: {analysis['effort_estimate']['risk_level']}")
print(f"Estimated time: {analysis['effort_estimate']['development_time_hours']} hours")
```

### Framework Performance Benchmarking
```python
# Compare framework performance
benchmark_config = {
    'num_iterations': 100,
    'batch_size': 32,
    'input_shape': (224, 224, 3)
}

model_config = {
    'type': 'resnet',
    'num_layers': 50,
    'num_classes': 1000
}

results = migration_expert.benchmark_framework_performance(
    frameworks=['flax', 'equinox', 'keras'],
    model_config=model_config,
    benchmark_config=benchmark_config
)

for framework, metrics in results.items():
    print(f"{framework}: Training {metrics['training_time_ms']:.1f}ms, "
          f"Inference {metrics['inference_time_ms']:.1f}ms")
```

### Framework Recommendation
```python
# Get framework recommendation
requirements = {
    'performance_priority': 'high',
    'model_complexity': 'high',
    'development_speed': 'medium',
    'deployment_target': 'research',
    'team_experience': {'jax': 'high', 'pytorch': 'medium'}
}

recommendation = migration_expert.create_framework_recommendation(requirements)
print(f"Recommended framework: {recommendation['primary_recommendation']}")
print(f"Reasoning: {recommendation['reasoning']}")
```

This expert provides comprehensive framework migration capabilities with automated analysis, performance benchmarking, and strategic guidance for optimal framework selection in scientific computing workflows.