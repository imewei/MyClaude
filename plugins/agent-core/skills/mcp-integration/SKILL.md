---
name: mcp-integration
version: "2.2.0"
description: Guide for using integrated MCP servers (serena, github, sequential-thinking, context7)
---

# MCP Integration Guide

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
