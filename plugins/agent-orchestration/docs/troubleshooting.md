# Troubleshooting Agent Orchestration

## Overview

This guide provides solutions for common issues encountered when using the `agent-orchestration` plugin and its associated agents.

## Common Issues

### 1. Agent Availability

**Symptom**: "Agent X unavailable" or "Missing primary agent".
**Cause**:
- The agent markdown file is missing in `plugins/*/agents/`.
- Plugin dependencies are not installed.
- Typos in the agent name reference.
**Solution**:
- Verify the agent file exists.
- Run `/plugin list` to check installed plugins.
- Check `plugin.json` for correct configuration.

### 2. Context Window Exceeded

**Symptom**: "Context length exceeded" or agent fails to respond to long threads.
**Cause**:
- Conversation history is too long.
- Too many large files loaded into context.
**Solution**:
- Use `context-manager` to summarize history.
- Use `read_file` with line limits.
- Reset the session if appropriate.

### 3. Tool Execution Failures

**Symptom**: "Tool execution failed" or "Invalid arguments".
**Cause**:
- The agent is hallucinating tool parameters.
- The underlying tool environment (e.g., Python venv) is broken.
- Missing permissions for file operations.
**Solution**:
- Check the tool definition and examples in the prompt.
- Verify the environment (e.g., `uv pip install ...`).
- Check file permissions.

### 4. Orchestration Loops

**Symptom**: Agents passing tasks back and forth indefinitely.
**Cause**:
- Unclear handover criteria.
- Overlapping responsibilities.
**Solution**:
- Refine agent role definitions in their `.md` files.
- Use the `multi-agent-orchestrator` to enforce a strict plan.
- Implement a maximum hop count or timeout.

### 5. Performance Issues

**Symptom**: Slow response times or high latency.
**Cause**:
- Serial execution of independent tasks.
- Heavy processing in tool implementation.
- Network latency for external APIs.
**Solution**:
- Use `--parallel` mode where possible.
- Optimize tool code (see `agent-performance-optimization` skill).
- Cache API results.

## Debugging

### Enable Verbose Logging

To see detailed agent thought processes and tool calls:
```bash
# Set environment variable
export CLAUDE_LOG_LEVEL=DEBUG
```

### Inspecting Artifacts

Check the generated artifacts for clues:
- `.optimization/logs/` for execution logs.
- `.optimization/reports/` for intermediate outputs.

## Getting Help

If you cannot resolve the issue:
1. Check the [Claude Code Documentation](https://docs.anthropic.com/claude-code).
2. Open an issue on the plugin repository.
3. Consult the `systems-architect` agent for architectural debugging.
