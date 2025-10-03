"""Pytest configuration for nonequilibrium physics agents.

This file configures pytest to properly handle imports and set up
the Python path for testing.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path IMMEDIATELY
project_root = Path(__file__).parent.absolute()
project_root_str = str(project_root)

# Remove any existing references and add at the front
sys.path = [p for p in sys.path if os.path.abspath(p) != project_root_str]
sys.path.insert(0, project_root_str)

# Also set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = project_root_str + os.pathsep + os.environ.get('PYTHONPATH', '')

# Register custom markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
