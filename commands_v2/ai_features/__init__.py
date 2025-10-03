#!/usr/bin/env python3
"""
AI Features Module
==================

Advanced AI-powered features for the Claude Code Command Executor Framework.

This module provides breakthrough code intelligence capabilities including:
- Semantic code understanding and analysis
- Predictive optimization and quality prediction
- Automated code generation and refactoring
- Learning-based agent system
- Context-aware recommendations
- Neural code search
- AI-powered code review

Author: Claude Code AI Team
Version: 1.0
Last Updated: 2025-09-29
"""

__version__ = "1.0.0"
__author__ = "Claude Code AI Team"

# Core modules
from . import understanding
from . import prediction
from . import generation
from . import agents
from . import context
from . import documentation
from . import review
from . import anomaly
from . import models
from . import search
from . import integration

__all__ = [
    "understanding",
    "prediction",
    "generation",
    "agents",
    "context",
    "documentation",
    "review",
    "anomaly",
    "models",
    "search",
    "integration",
]