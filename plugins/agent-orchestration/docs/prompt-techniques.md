# Prompt Engineering Techniques for Agents

## Overview

Advanced prompt engineering is the lever for maximizing the cognitive capabilities of Large Language Models (LLMs) acting as agents. This guide covers techniques specific to agentic workflows.

## Core Techniques

### 1. Chain-of-Thought (CoT)
**Description**: Instructing the model to "think step-by-step" before acting.
**Benefit**: Reduces logic errors and improves complex reasoning.
**Pattern**:
```
User: [Task]
Agent: [Thought process: Analyze -> Plan -> Check Constraints] -> [Action]
```

### 2. Few-Shot Learning
**Description**: Providing example input-output pairs in the context.
**Benefit**: Drastically improves adherence to specific formats and styles.
**Pattern**:
```
Example 1:
Input: "Search for weather in NY"
Output: Call Tool: weather_api(location="New York, NY")

Example 2:
Input: "Who is the CEO of Anthropic?"
Output: Call Tool: search_web(query="Anthropic CEO")

Input: [User Input]
Output: ...
```

### 3. ReAct (Reasoning + Acting)
**Description**: Interleaving reasoning traces with action execution.
**Benefit**: Allows the agent to adjust its plan based on tool outputs.
**Pattern**:
`Thought` -> `Action` -> `Observation` -> `Thought` -> ...

### 4. Constitutional AI / System Directives
**Description**: Embedding high-level principles and constraints in the system prompt.
**Benefit**: Ensures safety, alignment, and consistent personality.
**Example**:
"You are a helpful assistant. You must never reveal PII. You must always verify facts before stating them."

## Advanced Strategies

### 1. Dynamic Prompt Construction
**Description**: Assembling prompts programmatically based on task context.
- Injecting relevant documentation snippets.
- Retrieving similar past successful examples (RAG for prompts).

### 2. Self-Reflection / Self-Correction
**Description**: Asking the model to critique its own plan or output before finalizing it.
**Pattern**:
"Review your plan. Are there any edge cases you missed? If so, revise the plan."

### 3. Role Prompting
**Description**: Assigning a specific expert persona.
**Benefit**: Biases the model towards domain-specific knowledge and terminology.
**Example**: "You are a Senior Python Backend Engineer specialized in Django..."

## Optimizing for Agents

- **Tool Definition Clarity**: Describe tools not just by what they do, but *when* to use them.
- **Output Structuring**: Force structured outputs (JSON, XML) for reliable parsing by the orchestration layer.
- **Error Handling Instructions**: Explicitly tell the agent how to handle tool failures (e.g., "If search fails, try a broader query").
