---
name: tool-use-patterns
description: Design effective tool use for AI agents including function calling, tool selection strategies, tool chaining, error recovery, and parallel tool execution. Use when building agents that interact with external tools, APIs, or when optimizing tool call efficiency.
---

# Tool Use Patterns

## Expert Agent

For tool orchestration and agent workflow design, delegate to:

- **`orchestrator`**: Coordinates multi-step tool workflows, manages dependencies, and handles failures.
  - *Location*: `plugins/agent-core/agents/orchestrator.md`

Comprehensive guide for designing, implementing, and optimizing tool use in AI agent systems.

---

## 1. Tool Definition Best Practices

### Schema Design

Each tool definition needs: `name`, `description`, `parameters` (with types), `required` fields, and at least one `example`. Keep descriptions under 200 characters. Specify parameter types, defaults, and constraints explicitly.

### Naming Conventions

| Pattern | Example | When to Use |
|---------|---------|-------------|
| `verb_noun` | `search_files` | Standard CRUD and query operations |
| `noun_verb` | `database_query` | Namespace grouping by resource |
| `get_/set_/list_/create_/delete_` | `list_users` | RESTful resource operations |

### Description Guidelines

- Start with an action verb: "Search", "Create", "Calculate".
- State what the tool returns, not just what it does.
- Include constraints: max input size, rate limits, required permissions.
- Mention when NOT to use the tool (reduces misuse).

---

## 2. Tool Selection Strategies

### Decision Matrix

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| **Name Matching** | LLM picks tool by name/description | < 10 tools |
| **Category Routing** | First classify intent, then select within category | 10-50 tools |
| **Embedding Search** | Embed query + tool descriptions, nearest match | 50+ tools |
| **Planner-Executor** | Separate planning step selects tools in advance | Complex workflows |

### Category Routing

Group tools by domain (file ops, git ops, web ops, code analysis). First classify the user intent into a category, then select the specific tool within that category. This two-step approach scales to 50+ tools.

---

## 3. Tool Chaining Patterns

### Sequential Chain

Execute tools in order, passing output of one as input to the next.

```
search_files("TODO") -> read_file(match.path) -> analyze_code(content) -> create_issue(analysis)
```

### Conditional Chain

Branch based on tool output.

```
run_tests()
  |-- if passed -> git_commit()
  |-- if failed -> read_error(log) -> suggest_fix(error) -> apply_fix(suggestion) -> run_tests()
```

### Fan-Out / Fan-In

Execute multiple tools in parallel, then aggregate results.

```
[search_docs(query), search_code(query), search_issues(query)]
  -> merge_results(doc_hits, code_hits, issue_hits)
  -> rank_by_relevance(merged)
```

### Chain Composition Table

| Pattern | Parallelism | Dependencies | Use Case |
|---------|------------|--------------|----------|
| Sequential | None | Each step depends on previous | Data pipelines |
| Conditional | None | Branch on result | Error handling |
| Fan-Out/In | Full | Independent then merge | Multi-source search |
| Map-Reduce | Partial | Map parallel, reduce serial | Batch processing |

---

## 4. Error Handling and Recovery

### Error Categories

| Category | Example | Recovery Strategy |
|----------|---------|-------------------|
| **Transient** | Network timeout, rate limit | Retry with exponential backoff |
| **Permanent** | Invalid parameters, not found | Fix parameters or use fallback tool |
| **Partial** | Truncated results | Request remaining data |
| **Permission** | Access denied | Escalate or use alternative |

### Retry and Fallback

- **Retry**: Use exponential backoff (base_delay * 2^attempt) for transient errors. Max 3 retries.
- **Fallback chains**: Define ordered alternatives (e.g., vector_search -> keyword_search -> full_text).

---

## 5. Parallel Tool Execution

### When to Parallelize

| Condition | Parallelize? | Reason |
|-----------|-------------|--------|
| Tools share no inputs/outputs | Yes | No data dependency |
| Tool B needs Tool A output | No | Sequential dependency |
| Both tools modify same resource | No | Race condition risk |
| Gathering data from multiple sources | Yes | Reduces total latency |

### Execution Planner

Build a dependency graph of tool calls. Group calls with no unmet dependencies into parallel batches. Execute batches sequentially, marking completed tools after each batch. Detect circular dependencies before execution starts.

---

## 6. Tool Use Checklist

- [ ] Each tool has a clear, action-oriented name and description
- [ ] Required vs optional parameters are correctly marked
- [ ] At least one usage example per tool definition
- [ ] Selection strategy chosen based on tool count
- [ ] Chaining pattern identified for multi-step workflows
- [ ] Error handling covers transient, permanent, and partial failures
- [ ] Retry strategy with backoff configured for external APIs
- [ ] Parallel execution plan respects data dependencies
- [ ] Fallback tools defined for critical operations
- [ ] Tool call logging captures inputs, outputs, and latency

---

## Related Skills

- `multi-agent-coordination` -- Agent workflows that coordinate complex tool chains across agents
- `mcp-integration` -- MCP server integration for tool discovery and invocation
