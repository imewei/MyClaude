#!/usr/bin/env python3
"""ExecutionError hook for science-suite.

Captures JAX compilation errors, OOM, and numerical divergence context.
"""

import json
import os
import sys


def classify_error(message: str) -> str:
    """Classify scientific computing errors."""
    message_lower = message.lower()

    if "oom" in message_lower or "out of memory" in message_lower:
        return "OOM: Consider reducing batch size, using gradient checkpointing, or switching to float32→float16"
    if "xla" in message_lower or "jit" in message_lower or "compilation" in message_lower:
        return "JAX/XLA compilation error: Check for dynamic shapes, Python control flow in JIT, or tracer leaks"
    if "nan" in message_lower or "diverge" in message_lower:
        return "Numerical divergence: Check learning rate, initialization, or numerical precision"
    if "singular" in message_lower or "not positive definite" in message_lower:
        return "Linear algebra error: Matrix may be ill-conditioned. Add regularization or check input data"

    return "Unclassified compute error"


def main() -> None:
    """Capture and classify execution error."""
    try:
        error_message = os.environ.get("ERROR_MESSAGE", "unknown error")
        tool_name = os.environ.get("TOOL_NAME", "unknown")
        classification = classify_error(error_message)

        result = {
            "status": "success",
            "additionalContext": (
                f"Science compute error in {tool_name}: {classification}. "
                f"Original: {error_message[:200]}"
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"ExecutionError hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
