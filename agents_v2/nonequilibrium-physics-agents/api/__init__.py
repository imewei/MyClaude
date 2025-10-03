"""API Services for Optimal Control.

This module provides REST and GraphQL APIs for:
- Solver execution
- Job submission and monitoring
- Result retrieval
- System health monitoring

Author: Nonequilibrium Physics Agents
"""

from .rest_api import (
    create_app,
    run_server,
    OptimalControlAPI,
)

__all__ = [
    'create_app',
    'run_server',
    'OptimalControlAPI',
]
