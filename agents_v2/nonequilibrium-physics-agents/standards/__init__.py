"""Data Standards and Formats Module.

This module provides standardized data formats, schemas, validation, and
serialization for all optimal control components. Enables seamless integration
between solvers, ML models, HPC jobs, and deployment infrastructure.

Author: Nonequilibrium Physics Agents
"""

from .data_formats import (
    StandardDataFormat,
    SolverInput,
    SolverOutput,
    TrainingData,
    OptimizationResult,
    convert_to_standard_format,
    validate_standard_format
)

from .schemas import (
    SOLVER_INPUT_SCHEMA,
    SOLVER_OUTPUT_SCHEMA,
    TRAINING_DATA_SCHEMA,
    get_schema,
    validate_against_schema
)

from .validation import (
    DataValidator,
    validate_solver_input,
    validate_solver_output,
    validate_training_data,
    ValidationError
)

from .serialization import (
    serialize,
    deserialize,
    save_to_file,
    load_from_file,
    SerializationFormat
)

__all__ = [
    # Data formats
    'StandardDataFormat',
    'SolverInput',
    'SolverOutput',
    'TrainingData',
    'OptimizationResult',
    'convert_to_standard_format',
    'validate_standard_format',

    # Schemas
    'SOLVER_INPUT_SCHEMA',
    'SOLVER_OUTPUT_SCHEMA',
    'TRAINING_DATA_SCHEMA',
    'get_schema',
    'validate_against_schema',

    # Validation
    'DataValidator',
    'validate_solver_input',
    'validate_solver_output',
    'validate_training_data',
    'ValidationError',

    # Serialization
    'serialize',
    'deserialize',
    'save_to_file',
    'load_from_file',
    'SerializationFormat',
]

__version__ = '1.0.0'
