---
description: Systematic improvement of agents through performance analysis, prompt engineering, and iterative testing
allowed-tools: Read, Write, Edit, Bash(git:*), Bash(pytest:*)
argument-hint: <agent-name> [--mode=check|phase|optimize] [--phase=1|2|3|4] [--focus=AREA]
color: blue
agents:
  primary:
    - context-manager
  conditional:
    - agent: prompt-engineer
      trigger: argument "--phase=2" OR argument "--mode=optimize"
  orchestrated: false
execution-modes:
  check: Quick health assessment (2-5 min)
  phase: Execute single phase (10-30 min)
  optimize: Full 4-phase cycle (1-2 hours)
output-format: json-report + markdown-summary + improved-prompts
---

# Agent Performance Optimization

## Quick Start

### Health Check (Recommended First Step)
```bash
/improve-agent <agent-name> --mode=check
```
**Output**: Health report with top 3 improvement opportunities (2-5 minutes)

### Single Phase Execution
```bash
/improve-agent <agent-name> --phase=2 --focus=tool-selection
```
**Output**: Targeted improvements for specific phase

### Full Optimization Cycle
```bash
/improve-agent <agent-name> --mode=optimize
```
**Output**: Complete 4-phase improvement workflow (1-2 hours)

## Execution Flow

### Mode: check (Quick Health Assessment)

**When user invokes**: `/improve-agent <agent-name> --mode=check`

**Execute these steps**:

1. **Parse agent name** from arguments:
   ```
   Agent name: $ARGUMENTS (first argument)
   If no agent name provided: prompt user "Which agent would you like to analyze?"
   ```

2. **Check if agent exists**:
   ```bash
   # Look for agent in plugin directories
   find plugins/*/agents/ -name "${agent_name}.md" 2>/dev/null
   ```

3. **Invoke context-manager agent** (if available):
   ```
   Use Task tool with:
   - subagent_type: "context-manager"
   - prompt: "Analyze performance metrics for agent '${agent_name}' over the past 30 days.
             Identify top 3 issues ranked by (impact × frequency).
             For each issue, provide specific, actionable fix recommendations."
   ```

4. **If context-manager unavailable**: Generate mock health report based on best practices:
   ```
   Display: "⚠️  context-manager agent not available. Generating template health report..."

   Create template report with:
   - Success rate: [requires metrics]
   - Tool efficiency: [requires analysis]
   - Common issues: [generic recommendations]
   - Recommendation: "Set up metrics collection for accurate analysis"
   ```

5. **Generate health report**:
   ```
   Create file: .reports/${agent_name}-health-$(date +%Y-%m-%d).json
   Format: JSON with metrics, issues, recommendations
   Display: Formatted summary to user
   ```

**Example Execution**:
```
Agent Health Report: customer-support
Overall Score: 78/100 (Good)
├─ ✅ Success Rate: 87% (target: >85%)
├─ ⚠️  Avg Corrections: 2.3/task (target: <1.5)
├─ ⚠️  Tool Efficiency: 72% (target: >80%)
└─ ✅ User Satisfaction: 8.2/10

Top 3 Issues:
1. Misunderstands complex pricing queries (15% of failures)
   → Fix: Add few-shot examples for pricing scenarios
2. Incorrect tool selection for order status (8% of failures)
   → Fix: Add tool decision tree to prompt
3. Overly verbose responses (12% user corrections)
   → Fix: Add conciseness instruction with example

Run: /improve-agent customer-support --phase=2
```

### Mode: phase --phase=N
Execute specific optimization phase (see [Phase Reference](#phase-reference))

**Phase 1**: Performance analysis and baseline metrics
**Phase 2**: Prompt engineering improvements
**Phase 3**: Testing and validation
**Phase 4**: Deployment and monitoring

### Mode: optimize
Execute all phases sequentially with validation gates

**Workflow**:
1. Phase 1 → Generate baseline report → User review
2. Phase 2 → Generate improved prompt → User review
3. Phase 3 → Run A/B tests → User approval
4. Phase 4 → Deploy with monitoring → User confirmation at each step

## Phase Reference

**Comprehensive Methodology**: [Agent Optimization Guide](../../docs/agent-optimization-guide.md)

### Phase 1: Performance Analysis
[Details](../../docs/phase-1-analysis.md)
- Metrics collection (success rate, corrections, tool usage)
- Failure mode classification
- User feedback pattern analysis
- Baseline performance report

### Phase 2: Prompt Engineering
[Details](../../docs/phase-2-prompts.md)
- Chain-of-thought optimization
- Few-shot example curation
- Role definition refinement
- Constitutional AI integration
- Output format tuning

### Phase 3: Testing & Validation
[Details](../../docs/phase-3-testing.md)
- Test suite development
- A/B testing framework
- Evaluation metrics
- Human evaluation protocol

### Phase 4: Deployment & Monitoring
[Details](../../docs/phase-4-deployment.md)
- Version management (semantic versioning)
- Staged rollout (alpha → beta → canary → full)
- Rollback procedures
- Continuous monitoring

## Common Workflows

### Workflow 1: Quick Iteration (Recommended)
```bash
# 1. Identify issues (5 min)
/improve-agent my-agent --mode=check

# 2. Review report, manually edit prompt based on suggestions

# 3. Test changes (10 min)
/improve-agent my-agent --phase=3

# 4. Deploy with canary (5 min)
/improve-agent my-agent --phase=4 --canary=10%
```

### Workflow 2: Comprehensive Optimization
```bash
# Fully automated analysis, improvement, testing, and deployment plan
/improve-agent my-agent --mode=optimize

# Review and approve at each phase gate
# Total time: 1-2 hours with user review points
```

### Workflow 3: Targeted Fix
```bash
# Focus on specific area (e.g., tool selection, reasoning, formatting)
/improve-agent my-agent --phase=2 --focus=tool-selection

# Generates targeted prompt improvements for that area only
```

## Output Artifacts

All artifacts are saved with timestamps for version control:

- **Health Report**: `.reports/<agent>-health-YYYY-MM-DD.json`
  - Overall health score
  - Top issues with frequency and impact
  - Specific improvement recommendations

- **Baseline Metrics**: `.metrics/<agent>-baseline-YYYY-MM-DD.json`
  - Success rate, corrections, tool efficiency
  - User satisfaction score
  - Response latency (p50, p95, p99)
  - Token consumption metrics

- **Improved Prompt**: `.agents/<agent>-v<X.Y.Z>.md`
  - Semantic versioning (MAJOR.MINOR.PATCH)
  - Git-compatible for diff/review
  - Includes changelog and rationale

- **Test Results**: `.tests/<agent>-results-YYYY-MM-DD.json`
  - Test suite execution results
  - Pass/fail rates by category
  - Performance benchmarks

- **A/B Comparison**: `.reports/<agent>-ab-test-YYYY-MM-DD.md`
  - Statistical significance (p-value)
  - Effect size (Cohen's d)
  - Recommendation (deploy/iterate/rollback)

## Success Criteria

Agent improvement is successful when:
- ✅ Task success rate improves by ≥15%
- ✅ User corrections decrease by ≥25%
- ✅ No increase in safety violations
- ✅ Response time remains within 10% of baseline
- ✅ Cost per task doesn't increase >5%
- ✅ Positive user feedback increases

**Note**: All metrics documented in [Success Metrics Guide](../../docs/success-metrics.md)

## Best Practices

1. **Start with health check**: Always run `--mode=check` first
2. **Iterate quickly**: Use phase-specific execution for rapid testing
3. **Version control**: Commit each improved prompt version to git
4. **Monitor continuously**: Track metrics post-deployment
5. **User feedback**: Incorporate real user feedback patterns
6. **Safety first**: Never skip testing phase

**Full Best Practices**: [Best Practices Guide](../../docs/best-practices.md)

## Troubleshooting

**Issue**: Health check fails with "context-manager not available"
**Solution**: Ensure context-manager agent exists in plugin
```bash
ls plugins/agent-orchestration/agents/context-manager.md
```

**Issue**: No performance data for agent
**Solution**: Agent needs production usage history (minimum 7 days)

**Issue**: Prompt improvements don't show impact
**Solution**: Ensure sufficient sample size (minimum 100 tasks) for statistical significance

**More troubleshooting**: [Troubleshooting Guide](../../docs/troubleshooting.md)

## Examples

Real-world examples with actual metrics:
- [Optimizing Customer Support Agent](../../docs/examples/customer-support-optimization.md)
- [Improving Code Review Agent](../../docs/examples/code-review-improvement.md)
- [Enhancing Research Assistant](../../docs/examples/research-assistant-enhancement.md)

## Additional Resources

- **Methodology**: [Complete Optimization Guide](../../docs/agent-optimization-guide.md)
- **Metrics**: [Success Metrics & KPIs](../../docs/success-metrics.md)
- **Techniques**: [Prompt Engineering Techniques](../../docs/prompt-techniques.md)
- **Tools**: [Testing & Validation Tools](../../docs/testing-tools.md)
