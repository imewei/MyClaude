---
version: "2.1.0"
name: code-analyze
description: Semantic code analysis using Serena MCP for symbol navigation and understanding
argument-hint: "[symbol-name] [--path=relative/path] [--depth=N] [--substring]"
color: blue
allowed-tools:
  - mcp__plugin_serena_serena__find_symbol
  - mcp__plugin_serena_serena__get_symbols_overview
  - mcp__plugin_serena_serena__find_referencing_symbols
  - mcp__plugin_serena_serena__read_file
  - mcp__plugin_serena_serena__search_for_pattern
---

# Semantic Code Analysis with Serena

Use Serena MCP for intelligent code navigation and analysis.

## Capabilities

- **Symbol Navigation**: Find classes, methods, functions by name path
- **Reference Tracking**: Discover where symbols are used
- **Overview Generation**: Get structured view of file contents
- **Pattern Search**: Find code patterns across the codebase

## Analysis Strategies

### Find a Symbol
```json
{
  "name_path_pattern": "ClassName/methodName",
  "include_body": true,
  "depth": 1
}
```

### Get File Overview
```json
{
  "relative_path": "src/module.py"
}
```

### Find All References
```json
{
  "name_path_pattern": "TargetSymbol",
  "relative_path": "src/"
}
```

## Best Practices

1. Use `relative_path` to restrict searches for faster results
2. Start with `include_body: false`, then read bodies only when needed
3. Use `depth: 1` to see immediate children (e.g., class methods)
4. Use `substring_matching: true` for partial name matches

## User Request

$ARGUMENTS
