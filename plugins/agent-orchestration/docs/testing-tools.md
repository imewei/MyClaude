# Agent Testing Tools

## Overview

Reliable agents require rigorous testing. This document outlines the tools and frameworks used for validating agent behavior, performance, and safety.

## Testing Frameworks

### 1. Evals (Evaluation Frameworks)
- **Promptfoo**: Tool for testing prompts against test cases. Good for deterministic assertions.
- **DeepEval**: Open-source evaluation framework for LLMs (faithfulness, answer relevancy, hallucination).
- **PyTest**: Standard Python testing framework, adapted for agent unit tests (mocking LLM calls).

### 2. Simulation Environments
- **AgentBench**: Comprehensive framework for evaluating agents across different environments (OS, Database, Knowledge Graph).
- **WebArena**: Environment for testing web-browsing agents.

## Types of Tests

### 1. Unit Tests
**Scope**: Individual tool functions or prompt logic.
**Tool**: `pytest` + `unittest.mock`.
**Example**: Verifying that the `calculator` tool correctly adds numbers, or that the prompt template renders correctly.

### 2. Integration Tests
**Scope**: End-to-end agent workflow with mocked LLM or cached responses.
**Tool**: Custom scripts, `vcrpy` (to record/replay API interactions).
**Example**: Agent receives a task, plans, calls tools, and produces a final answer.

### 3. Regression Tests (Golden Datasets)
**Scope**: Ensuring new changes don't break previously working scenarios.
**Method**: Maintain a `golden_dataset.json` of inputs and expected outputs. Run the agent against this set on every PR.

### 4. Adversarial / Red Teaming
**Scope**: Testing for safety and robustness against jailbreaks or confusing inputs.
**Tool**: `garak` (LLM vulnerability scanner).

## Metrics & Reporting

- **Weights & Biases (W&B)**: For tracking experiment runs and prompt versions.
- **Arize Phoenix**: Observability for LLM applications.

## Continuous Integration (CI)

Automate tests using GitHub Actions or similar:
1. **Linting**: Check code style.
2. **Static Analysis**: Check for potential bugs.
3. **Unit Tests**: Run fast tests.
4. **Eval Run**: Run a subset of the golden dataset (due to cost/time).

## Recommended Tool Stack for this Plugin

For `agent-orchestration`, we recommend:
1. **Runner**: `pytest` for functional tests.
2. **Evals**: Custom script using `phase-3-testing.md` methodology.
3. **Mocking**: `unittest.mock` for external API calls during dev.
