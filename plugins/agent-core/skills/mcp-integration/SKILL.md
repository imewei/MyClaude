---
name: mcp-integration
description: Guide for using integrated MCP servers in Claude Code. Use when connecting to MCP servers, configuring tool integrations, understanding MCP tool naming conventions, or coordinating cross-tool workflows.
---

# MCP Integration Guide

## Expert Agent

For MCP server integration and cross-tool coordination, delegate to:

- **`context-specialist`**: Manages context retrieval via Context7, information synthesis, and memory systems.
  - *Location*: `plugins/agent-core/agents/context-specialist.md`
- **`orchestrator`**: Coordinates MCP tool usage across multiple agents and workflows.
  - *Location*: `plugins/agent-core/agents/orchestrator.md`

This skill documents the integrated MCP servers available across MyClaude plugins.

## Available MCP Servers

These servers are provided by official Claude Code plugins and are available for use in MyClaude commands.

### Agent Core Suite Commands

| Server | Purpose | Command |
|--------|---------|---------|
| **sequential-thinking** | Chain-of-thought reasoning with revision/branching | `/ultra-think` (native implementation) |
| **context7** | Library documentation retrieval | `/docs-lookup` |

### Infrastructure & Ops Suite Commands

| Server | Purpose | Command |
|--------|---------|---------|
| **serena** | Semantic code analysis & symbol navigation | `/code-analyze` |
| **github** | GitHub API operations (issues, PRs, repos) | `/github-assist` |

> **Note**: MCP servers are provided by official Claude plugins. Commands here use `allowed-tools` to pre-authorize access to these existing servers.

## Quick Reference

### Sequential Thinking
Best for complex problems requiring structured analysis:
```
/ultra-think How should I architect a microservices system for high availability?
```
Note: `ultra-think` provides a native implementation with modes, frameworks, and session management.

### Context7 Documentation
Best for looking up current API references:
```
/docs-lookup How do I use React Query's useInfiniteQuery hook?
```

### Serena Code Analysis
Best for understanding codebases semantically:
```
/code-analyze Find all usages of the AuthService class
```

### GitHub Operations
Best for repository management:
```
/github-assist List open PRs that need review
```

## Combining MCP Servers

For complex tasks, combine multiple MCP capabilities:

1. **Research + Implement**: Use Context7 to understand library APIs, then Serena to find where to implement
2. **Analyze + Plan**: Use Serena to understand existing code, then Sequential Thinking to plan changes
3. **Review + Track**: Use GitHub to find PR changes, then Serena to analyze the code diff

## Tool Naming Convention

MCP tools follow this pattern:
```
mcp__plugin_<plugin-name>_<server-name>__<tool-name>
```

Examples:
- `mcp__plugin_serena_serena__find_symbol`
- `mcp__plugin_github_github__search_issues`
- `mcp__plugin_context7_context7__query-docs`

## Related Skills

- `multi-agent-coordination` -- Workflow patterns for coordinating MCP tool usage across agents
- `llm-application-patterns` -- RAG and prompt engineering patterns that leverage MCP retrieval

## Checklist

- [ ] Verify MCP server plugin is installed and listed in `claude mcp list`
- [ ] Confirm tool naming follows `mcp__plugin_<plugin-name>_<server-name>__<tool-name>` convention
- [ ] Check `allowed-tools` in command frontmatter matches the MCP tools actually needed
- [ ] Test each MCP server connection independently before combining in workflows
- [ ] Validate that sequential-thinking responses include revision and branching metadata
- [ ] Ensure Context7 queries specify the correct library name and version
- [ ] Verify Serena semantic analysis targets the correct project and symbol scope
- [ ] Confirm GitHub MCP operations use appropriate authentication and repository context
- [ ] Test cross-server workflows end-to-end (e.g., Context7 lookup then Serena analysis)
- [ ] Review error handling for MCP server timeouts or unavailability
