# Agent Success Metrics

## Overview

Defining and tracking the right metrics is crucial for objectively measuring agent performance and the impact of optimization efforts. This guide details key performance indicators (KPIs) for AI agents.

## Core Metrics

### 1. Task Success Rate (TSR)
**Definition**: The percentage of tasks completed successfully without human intervention or failure.
**Formula**: `(Successful Tasks / Total Tasks) * 100`
**Target**: >85% for autonomous agents, >95% for assisted workflows.

### 2. User Correction Rate (UCR)
**Definition**: The average number of times a user has to correct the agent or refine instructions per task.
**Formula**: `Total Corrections / Total Tasks`
**Target**: <0.5 per task (ideally <0.1).

### 3. Response Latency
**Definition**: Time taken from user input to receiving the first token (TTFT) and total completion time.
**Metrics**:
- **P50 (Median)**: Typical performance.
- **P95**: Worst-case performance for most users.
- **P99**: Outlier performance.
**Target**: <2s TTFT, Total time varies by task complexity.

### 4. Cost Efficiency
**Definition**: The cost incurred to perform a task, typically measured in tokens or dollars.
**Metric**: `Cost per 1000 Successful Tasks`.
**Goal**: Minimize while maintaining quality.

## Quality Metrics

### 1. Hallucination Rate
**Definition**: Frequency of factually incorrect or invented information.
**Measurement**: Manual audit of sample tasks or automated fact-checking against trusted sources.

### 2. Context Adherence
**Definition**: How well the agent respects provided context and constraints.
**Measurement**: Pass/Fail on constraint check tests.

### 3. Tool Usage Accuracy
**Definition**: Percentage of times the agent selects the correct tool and parameters for a given sub-task.
**Target**: >98%.

## User Experience Metrics

### 1. User Satisfaction (CSAT)
**Definition**: Direct user rating (1-5 stars) after task completion.
**Target**: >4.5/5.

### 2. Retention Rate
**Definition**: Percentage of users who return to use the agent within a specific period (e.g., weekly).

## Operational Metrics

### 1. Error Rate
**Definition**: Percentage of system errors (timeouts, API failures, crashes).
**Target**: <0.1%.

### 2. Throughput
**Definition**: Number of tasks processed per minute/hour.

## Dashboarding

Recommended visualization for metrics:
- **Real-time**: Error Rate, Latency, Throughput.
- **Daily**: Success Rate, Cost, User Satisfaction.
- **Weekly**: Context Adherence, Hallucination Rate trends.
