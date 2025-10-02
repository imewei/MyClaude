"""JSON Schema Definitions for Data Validation.

Provides JSON schemas for all standard data formats, enabling:
- Automatic validation
- API documentation generation
- Client library generation
- Type checking

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, Any

# Solver Input Schema
SOLVER_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SolverInput",
    "type": "object",
    "required": ["solver_type", "n_states", "n_controls", "initial_state", "time_horizon"],
    "properties": {
        "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {"type": "object"},
        "solver_type": {
            "type": "string",
            "enum": ["pmp", "collocation", "magnus", "jax_pmp", "rl_ppo", "rl_sac", "rl_td3",
                    "multi_objective", "robust", "stochastic"]
        },
        "problem_type": {
            "type": "string",
            "enum": ["lqr", "quantum_control", "trajectory_tracking", "energy_optimization",
                    "thermodynamic_process", "custom"]
        },
        "n_states": {"type": "integer", "minimum": 1},
        "n_controls": {"type": "integer", "minimum": 1},
        "initial_state": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1
        },
        "target_state": {
            "type": ["array", "null"],
            "items": {"type": "number"}
        },
        "time_horizon": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
        },
        "dynamics": {"type": "object"},
        "cost": {"type": "object"},
        "constraints": {"type": "object"},
        "solver_config": {"type": "object"}
    }
}

# Solver Output Schema
SOLVER_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SolverOutput",
    "type": "object",
    "required": ["success", "solver_type"],
    "properties": {
        "version": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {"type": "object"},
        "success": {"type": "boolean"},
        "solver_type": {"type": "string"},
        "optimal_control": {
            "type": ["array", "null"],
            "items": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "optimal_state": {
            "type": ["array", "null"],
            "items": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "optimal_cost": {"type": ["number", "null"]},
        "convergence": {"type": "object"},
        "computation_time": {"type": "number", "minimum": 0},
        "iterations": {"type": "integer", "minimum": 0},
        "error_message": {"type": ["string", "null"]}
    }
}

# Training Data Schema
TRAINING_DATA_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TrainingData",
    "type": "object",
    "required": ["problem_type", "states", "controls"],
    "properties": {
        "version": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {"type": "object"},
        "problem_type": {"type": "string"},
        "states": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "controls": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "values": {
            "type": ["array", "null"],
            "items": {"type": "number"}
        },
        "advantages": {
            "type": ["array", "null"],
            "items": {"type": "number"}
        },
        "rewards": {
            "type": ["array", "null"],
            "items": {"type": "number"}
        },
        "next_states": {
            "type": ["array", "null"],
            "items": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "dones": {
            "type": ["array", "null"],
            "items": {"type": "boolean"}
        },
        "n_samples": {"type": "integer", "minimum": 0},
        "n_states": {"type": "integer", "minimum": 0},
        "n_controls": {"type": "integer", "minimum": 0},
        "generation_method": {"type": "string"}
    }
}

# Optimization Result Schema
OPTIMIZATION_RESULT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "OptimizationResult",
    "type": "object",
    "required": ["success", "objective_values"],
    "properties": {
        "version": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {"type": "object"},
        "success": {"type": "boolean"},
        "objective_values": {
            "oneOf": [
                {"type": "number"},
                {
                    "type": "array",
                    "items": {"type": "number"}
                }
            ]
        },
        "optimal_parameters": {
            "type": "array",
            "items": {"type": "number"}
        },
        "pareto_front": {
            "type": ["array", "null"],
            "items": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "uncertainty_bounds": {"type": ["object", "null"]},
        "risk_metrics": {"type": "object"},
        "computation_time": {"type": "number", "minimum": 0},
        "n_evaluations": {"type": "integer", "minimum": 0},
        "convergence_history": {
            "type": "array",
            "items": {"type": "number"}
        }
    }
}

# HPC Job Spec Schema
HPC_JOB_SPEC_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "HPCJobSpec",
    "type": "object",
    "required": ["job_name", "job_type"],
    "properties": {
        "version": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {"type": "object"},
        "job_name": {"type": "string"},
        "job_type": {
            "type": "string",
            "enum": ["solver", "training", "parameter_sweep", "optimization"]
        },
        "input_data": {"type": "object"},
        "resources": {
            "type": "object",
            "properties": {
                "nodes": {"type": "integer", "minimum": 1},
                "cpus": {"type": "integer", "minimum": 1},
                "memory_gb": {"type": "number", "minimum": 0},
                "gpus": {"type": "integer", "minimum": 0},
                "time_hours": {"type": "number", "minimum": 0}
            }
        },
        "scheduler": {
            "type": "string",
            "enum": ["slurm", "pbs", "lsf", "dask"]
        },
        "priority": {
            "type": "string",
            "enum": ["low", "normal", "high", "urgent"]
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

# API Request/Response Schemas
API_REQUEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "APIRequest",
    "type": "object",
    "required": ["endpoint", "method"],
    "properties": {
        "version": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {"type": "object"},
        "endpoint": {"type": "string"},
        "method": {
            "type": "string",
            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]
        },
        "data": {"type": "object"},
        "headers": {"type": "object"},
        "timeout": {"type": "number", "minimum": 0}
    }
}

API_RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "APIResponse",
    "type": "object",
    "required": ["status_code", "success"],
    "properties": {
        "version": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {"type": "object"},
        "status_code": {"type": "integer", "minimum": 100, "maximum": 599},
        "success": {"type": "boolean"},
        "data": {"type": "object"},
        "error": {"type": ["string", "null"]},
        "execution_time": {"type": "number", "minimum": 0}
    }
}

# Schema registry
SCHEMA_REGISTRY = {
    "solver_input": SOLVER_INPUT_SCHEMA,
    "solver_output": SOLVER_OUTPUT_SCHEMA,
    "training_data": TRAINING_DATA_SCHEMA,
    "optimization_result": OPTIMIZATION_RESULT_SCHEMA,
    "hpc_job_spec": HPC_JOB_SPEC_SCHEMA,
    "api_request": API_REQUEST_SCHEMA,
    "api_response": API_RESPONSE_SCHEMA,
}


def get_schema(schema_name: str) -> Dict[str, Any]:
    """Get schema by name.

    Args:
        schema_name: Name of schema

    Returns:
        JSON schema dictionary

    Raises:
        ValueError: If schema not found
    """
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema: {schema_name}. Available: {list(SCHEMA_REGISTRY.keys())}")

    return SCHEMA_REGISTRY[schema_name]


def validate_against_schema(data: Dict[str, Any], schema_name: str) -> bool:
    """Validate data against schema.

    Args:
        data: Data to validate
        schema_name: Name of schema

    Returns:
        True if valid

    Raises:
        jsonschema.ValidationError: If validation fails
        ValueError: If schema not found
    """
    try:
        import jsonschema
    except ImportError:
        # Graceful degradation if jsonschema not installed
        print("Warning: jsonschema not installed, skipping validation")
        return True

    schema = get_schema(schema_name)
    jsonschema.validate(instance=data, schema=schema)
    return True


def generate_example(schema_name: str) -> Dict[str, Any]:
    """Generate example data from schema.

    Args:
        schema_name: Name of schema

    Returns:
        Example data dictionary
    """
    examples = {
        "solver_input": {
            "version": "1.0.0",
            "solver_type": "pmp",
            "problem_type": "lqr",
            "n_states": 2,
            "n_controls": 1,
            "initial_state": [1.0, 0.0],
            "target_state": [0.0, 0.0],
            "time_horizon": [0.0, 1.0],
            "dynamics": {},
            "cost": {"Q": [[1.0, 0.0], [0.0, 1.0]], "R": [[0.1]]},
            "constraints": {},
            "solver_config": {"max_iterations": 100, "tolerance": 1e-6}
        },
        "solver_output": {
            "version": "1.0.0",
            "success": True,
            "solver_type": "pmp",
            "optimal_control": [[0.5], [0.3], [0.1]],
            "optimal_state": [[1.0, 0.0], [0.7, 0.2], [0.3, 0.1]],
            "optimal_cost": 0.523,
            "convergence": {"residual": 1e-7},
            "computation_time": 0.156,
            "iterations": 12,
            "error_message": None
        }
    }

    return examples.get(schema_name, {})
