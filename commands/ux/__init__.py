"""
UX Enhancement System for Claude Code Command Executor.

Provides comprehensive user experience enhancements including:
- Progress tracking and visualization
- Error formatting and suggestions
- Command recommendations
- Enhanced CLI features
- Accessibility support
"""

__version__ = "1.0.0"

# Progress tracking
from .progress.progress_tracker import (
    ProgressTracker,
    ProgressStatus,
    ProgressItem,
    get_global_tracker
)

from .progress.step_tracker import (
    StepTracker,
    StepStatus,
    Step
)

from .progress.live_dashboard import (
    LiveDashboard,
    DashboardMetrics,
    get_global_dashboard
)

# Error handling
from .errors.error_formatter import (
    ErrorFormatter,
    ErrorCategory,
    ErrorSeverity,
    FormattedError,
    ErrorSuggestion,
    format_error,
    print_error
)

from .errors.error_suggestions import (
    ErrorSuggestionEngine,
    suggest_fixes,
    get_similar_issues
)

from .errors.error_recovery import (
    ErrorRecovery,
    RecoveryStrategy,
    get_global_recovery,
    retry,
    fallback
)

# Recommendations
from .recommendations.command_recommender import (
    CommandRecommender,
    CommandRecommendation,
    ProjectContext,
    get_global_recommender
)

# Core UX
from .core.ux_manager import (
    UXManager,
    UXConfig,
    ThemeMode,
    VerbosityLevel,
    OutputFormat,
    get_ux_manager
)

__all__ = [
    # Progress
    "ProgressTracker",
    "ProgressStatus",
    "ProgressItem",
    "get_global_tracker",
    "StepTracker",
    "StepStatus",
    "Step",
    "LiveDashboard",
    "DashboardMetrics",
    "get_global_dashboard",

    # Errors
    "ErrorFormatter",
    "ErrorCategory",
    "ErrorSeverity",
    "FormattedError",
    "ErrorSuggestion",
    "format_error",
    "print_error",
    "ErrorSuggestionEngine",
    "suggest_fixes",
    "get_similar_issues",
    "ErrorRecovery",
    "RecoveryStrategy",
    "get_global_recovery",
    "retry",
    "fallback",

    # Recommendations
    "CommandRecommender",
    "CommandRecommendation",
    "ProjectContext",
    "get_global_recommender",

    # Core
    "UXManager",
    "UXConfig",
    "ThemeMode",
    "VerbosityLevel",
    "OutputFormat",
    "get_ux_manager",
]