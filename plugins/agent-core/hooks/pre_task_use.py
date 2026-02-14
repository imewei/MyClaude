#!/usr/bin/env python3
"""PreToolUse hook for Task tool calls.

Injects additional context about available agent types and their capabilities
when the Task tool is invoked, helping the orchestrator make better routing decisions.
"""

import json
import os
import sys


AGENT_CAPABILITIES = {
    "orchestrator": "Workflow coordination, team assembly, dependency management",
    "context-specialist": "Context engineering, memory systems, token budget management",
    "reasoning-engine": "Advanced reasoning, CoT/ToT, logical analysis",
    "software-architect": "System design, API architecture, technical strategy",
    "app-developer": "Web/mobile apps, React, Next.js, Flutter",
    "systems-engineer": "Low-level systems, C/C++/Rust/Go, CLI tools",
    "devops-architect": "Cloud (AWS/Azure/GCP), Kubernetes, IaC",
    "sre-expert": "Reliability, observability, SLO/SLI, incident response",
    "automation-engineer": "CI/CD, GitHub Actions, GitLab CI, Git workflows",
    "quality-specialist": "Code review, security audit, test automation",
    "debugger-pro": "Root cause analysis, log correlation, memory profiling",
    "documentation-expert": "Technical docs, manuals, tutorials",
    "ai-engineer": "LLM apps, RAG systems, agent orchestration",
    "ml-expert": "Classical ML, MLOps, scikit-learn, XGBoost",
    "neural-network-master": "Deep learning, Transformers, CNNs, training diagnostics",
    "python-pro": "Python systems, type-driven design, uv/ruff, PyO3",
    "jax-pro": "JAX transformations, NumPyro, NLSQ, GPU computing",
    "julia-pro": "Julia, SciML, DifferentialEquations.jl",
    "research-expert": "Research methodology, evidence synthesis, visualization",
    "simulation-expert": "MD simulations, statistical mechanics, HPC",
    "statistical-physicist": "Statistical physics, correlation functions, phase transitions",
    "prompt-engineer": "Prompt design, LLM optimization, evaluation",
}


def main() -> None:
    """Provide agent routing context for Task tool calls."""
    try:
        tool_input = os.environ.get("TOOL_INPUT", "{}")

        try:
            input_data = json.loads(tool_input)
        except json.JSONDecodeError:
            input_data = {}

        subagent_type = input_data.get("subagent_type", "")

        result = {"status": "success"}

        if subagent_type and subagent_type in AGENT_CAPABILITIES:
            result["additionalContext"] = (
                f"Agent '{subagent_type}' specializes in: "
                f"{AGENT_CAPABILITIES[subagent_type]}. "
                f"Leverage Opus 4.6 adaptive thinking for complex sub-tasks."
            )

        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"PreToolUse hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
