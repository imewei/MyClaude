#!/bin/bash
#
# agent-suggest-hook.sh
# User prompt submit hook for intelligent agent suggestions
#
# Triggered before every user message is processed
# Analyzes input and suggests relevant agents
# Non-blocking, fast execution (<50ms target)
#
# Exit codes:
#   0 - Always (never block user input)

INPUT="$1"

# Quick pattern-based suggestions (no Node.js overhead)
# Returns 0 if suggestion made, 1 if no match
suggest_agent() {
  local pattern="$1"
  local agent="$2"
  local description="$3"

  if echo "$INPUT" | grep -qiE "$pattern"; then
    echo "💡 Agent suggestion: $agent - $description" >&2
    echo "   Invoke with: /agent --use $agent <your-task>" >&2
    echo "   Or auto-match: /agent $INPUT" >&2
    return 0
  fi
  return 1
}

# Only suggest if input is substantial (>15 chars) and not already a command
if [ ${#INPUT} -lt 15 ]; then
  exit 0
fi

if echo "$INPUT" | grep -qE "^/"; then
  # Already a slash command, don't suggest
  exit 0
fi

# Fast pattern matching for common domains
# Each pattern maps to a specific agent

# JAX / ML Frameworks
suggest_agent "jax|flax|optax|nnx|numpyro" "jax-pro" "JAX programming & optimization" && exit 0

# Quantum Computing
suggest_agent "quantum|qiskit|cirq|pennylane|qubit" "jax-scientific-domains" "Quantum computing & scientific domains" && exit 0

# HPC / Performance
suggest_agent "hpc|high.performance|parallel|distributed|optimization|numerical" "hpc-numerical-coordinator" "HPC & numerical optimization" && exit 0

# Neural Networks / Deep Learning
suggest_agent "neural.network|transformer|cnn|rnn|lstm|attention|deep.learning" "neural-architecture-engineer" "Neural architecture design" && exit 0

# Machine Learning Pipelines
suggest_agent "ml.pipeline|mlops|training.pipeline|model.deployment" "ml-pipeline-coordinator" "ML pipeline & MLOps" && exit 0

# Database / Data Engineering
suggest_agent "database|sql|postgresql|mongodb|mysql|redis|data.engineering" "database-workflow-engineer" "Database & data engineering" && exit 0

# Data Processing
suggest_agent "data.processing|etl|data.pipeline|data.warehouse" "data-engineering-coordinator" "Data processing & pipelines" && exit 0

# DevOps / Infrastructure
suggest_agent "devops|docker|kubernetes|k8s|ci.cd|terraform|ansible" "devops-security-engineer" "DevOps & infrastructure" && exit 0

# Security / Auditing
suggest_agent "security|vulnerability|audit|penetration|exploit" "devops-security-engineer" "Security & auditing" && exit 0

# Documentation
suggest_agent "documentation|docs|readme|api.doc|sphinx|mkdocs" "documentation-architect" "Documentation & technical writing" && exit 0

# Testing / Quality
suggest_agent "test|testing|pytest|jest|unit.test|integration.test|tdd" "code-quality-master" "Testing & code quality" && exit 0

# Code Quality / Refactoring
suggest_agent "refactor|code.quality|clean.code|code.smell|lint" "code-quality-master" "Code quality & refactoring" && exit 0

# Simulation / Physics
suggest_agent "simulation|physics|cfd|computational.fluid|molecular.dynamics" "simulation-expert" "Scientific simulation" && exit 0

# Correlation Functions / Scattering
suggest_agent "correlation|scattering|sans|saxs|pair.distribution" "correlation-function-expert" "Correlation functions & scattering" && exit 0

# Frontend / Web Development
suggest_agent "frontend|react|vue|angular|svelte|web.app|ui|ux" "fullstack-developer" "Full-stack web development" && exit 0

# Backend / API Development
suggest_agent "backend|api|rest|graphql|express|fastapi|node\.js" "fullstack-developer" "Backend & API development" && exit 0

# Visualization / Plotting
suggest_agent "visualiz|plot|chart|graph|matplotlib|plotly|dashboard" "visualization-interface-master" "Visualization & interfaces" && exit 0

# Research / Analysis
suggest_agent "research|analysis|literature.review|paper|academic" "research-intelligence-master" "Research & intelligence" && exit 0

# System Architecture
suggest_agent "architect|system.design|scalability|microservice|design.pattern" "systems-architect" "System architecture & design" && exit 0

# Multi-Agent Orchestration
suggest_agent "multi.agent|orchestrat|coordinat|complex.workflow" "multi-agent-orchestrator" "Multi-agent orchestration" && exit 0

# AI/ML General
suggest_agent "machine.learning|artificial.intelligence|model.training|inference" "ml-pipeline-coordinator" "ML/AI workflows" && exit 0

# Scientific Computing General
suggest_agent "scientific.computing|numerical.method|scipy|numpy" "hpc-numerical-coordinator" "Scientific computing" && exit 0

# Code Adoption / Migration
suggest_agent "adopt|migrate|port|legacy|moderniz" "scientific-code-adoptor" "Code adoption & migration" && exit 0

# Command Creation
suggest_agent "slash.command|create.command|custom.command" "command-systems-engineer" "Command system engineering" && exit 0

# No match found - silent exit (no suggestion)
# Optionally, could do semantic matching here with Node.js (adds ~50-100ms):
#
# MATCHES=$(node ~/.claude/scripts/agent-cli.mjs match "$INPUT" 2>/dev/null | jq -r 'if length > 0 and .[0].score > 0.75 then "💡 High-confidence match: " + .[0].agent.name + " (" + (.[0].score * 100 | floor | tostring) + "% confidence)\n   Use: /agent --use " + .[0].agent.name + " or /agent " + input.stdin else empty end' 2>/dev/null)
#
# if [ -n "$MATCHES" ]; then
#   echo "$MATCHES" >&2
# fi

exit 0  # Always succeed, never block user input
