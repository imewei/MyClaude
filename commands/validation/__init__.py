"""
Claude Code Command Executor Validation Framework

Comprehensive real-world validation framework for testing the command executor
system against production codebases.
"""

__version__ = "1.0.0"

from validation.executor import ValidationExecutor, ValidationResult
from validation.metrics.metrics_collector import MetricsCollector
from validation.metrics.quality_analyzer import QualityAnalyzer
from validation.benchmarks.baseline_collector import BaselineCollector
from validation.benchmarks.regression_detector import RegressionDetector
from validation.reports.report_generator import ReportGenerator

__all__ = [
    'ValidationExecutor',
    'ValidationResult',
    'MetricsCollector',
    'QualityAnalyzer',
    'BaselineCollector',
    'RegressionDetector',
    'ReportGenerator',
]