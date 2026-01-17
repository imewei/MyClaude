---
description: Systematic agent improvement through performance analysis and prompt
  engineering
triggers:
- /improve-agent
- systematic agent improvement through
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<agent-name> [--mode=check|phase|optimize] [--phase=1|2|3|4] [--focus=AREA]`
The agent should parse these arguments from the user's request.

# Agent Performance Optimization

## Quick Start

| Command | Duration | Output |
|---------|----------|--------|
| `/improve-agent <name> --mode=check` | 2-5 min | Health report with top 3 opportunities |
| `/improve-agent <name> --phase=2` | 10-30 min | Targeted improvements for specific phase |
| `/improve-agent <name> --mode=optimize` | 1-2 hours | Complete 4-phase improvement |

---

## Mode: check (Health Assessment)

### Execution Steps

1. Parse agent name from arguments
2. Find agent in `plugins/*/agents/`
3. Use the `context-manager` skill for metrics analysis
4. Generate health report with actionable fixes

### Health Report Format

```
Agent Health Report: <name>
Overall Score: X/100
├─ Success Rate: X% (target: >85%)
├─ Avg Corrections: X/task (target: <1.5)
├─ Tool Efficiency: X% (target: >80%)
└─ User Satisfaction: X/10

Top 3 Issues:
1. [Issue] → Fix: [Specific recommendation]
2. [Issue] → Fix: [Specific recommendation]
3. [Issue] → Fix: [Specific recommendation]
```

---

## Mode: phase --phase=N

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| 1 | Performance analysis | Baseline metrics, failure modes |
| 2 | Prompt engineering | Chain-of-thought, few-shot, constitutional AI |
| 3 | Testing & validation | Test suite, A/B testing, evaluation |
| 4 | Deployment & monitoring | Versioning, staged rollout, monitoring |

**Detailed methodology:** [Agent Optimization Guide](../../plugins/agent-orchestration/docs/agent-optimization-guide.md)

---

## Mode: optimize (Full Cycle)

Execute all phases sequentially with validation gates:

1. Phase 1 → Baseline report → User review
2. Phase 2 → Improved prompt → User review
3. Phase 3 → A/B tests → User approval
4. Phase 4 → Deploy with monitoring → User confirmation

---

## Common Workflows

### Quick Iteration (Recommended)
```bash
/improve-agent my-agent --mode=check     # 5 min
# Manually edit prompt based on suggestions
/improve-agent my-agent --phase=3        # 10 min
/improve-agent my-agent --phase=4 --canary=10%
```

### Targeted Fix
```bash
/improve-agent my-agent --phase=2 --focus=tool-selection
```

---

## Output Artifacts

| Artifact | Location | Content |
|----------|----------|---------|
| Health Report | `.reports/<agent>-health-YYYY-MM-DD.json` | Score, issues, recommendations |
| Baseline Metrics | `.metrics/<agent>-baseline-YYYY-MM-DD.json` | Success rate, latency, tokens |
| Improved Prompt | `.agents/<agent>-v<X.Y.Z>.md` | Semantic versioned, git-compatible |
| Test Results | `.tests/<agent>-results-YYYY-MM-DD.json` | Pass/fail, benchmarks |
| A/B Comparison | `.reports/<agent>-ab-test-YYYY-MM-DD.md` | Statistical significance |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Task success rate | +15% |
| User corrections | -25% |
| Safety violations | No increase |
| Response time | Within 10% of baseline |
| Cost per task | No increase >5% |

---

## Best Practices

1. **Start with health check** - Always run `--mode=check` first
2. **Iterate quickly** - Use phase-specific execution
3. **Version control** - Commit each improved prompt
4. **Monitor continuously** - Track post-deployment
5. **User feedback** - Incorporate real patterns
6. **Safety first** - Never skip testing phase

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| context-manager not available | Verify `plugins/agent-orchestration/agents/context-manager.md` exists |
| No performance data | Agent needs 7+ days production usage |
| Improvements don't show impact | Need 100+ tasks for statistical significance |

**More:** [Troubleshooting Guide](../../plugins/agent-orchestration/docs/agent-optimization-guide.md#troubleshooting)

---

## External Documentation

| Document | Purpose |
|----------|---------|
| [Agent Optimization Guide](../../plugins/agent-orchestration/docs/agent-optimization-guide.md) | Complete methodology |
| [Success Metrics](../../plugins/backend-development/docs/backend-development/success-metrics.md) | KPIs and thresholds |
| [Prompt Techniques](../../plugins/llm-application-dev/docs/prompt-patterns.md) | Improvement patterns |
| [Testing Tools](../../plugins/llm-application-dev/docs/ai-testing-deployment.md) | Validation frameworks |
