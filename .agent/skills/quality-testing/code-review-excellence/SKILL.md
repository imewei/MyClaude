---
name: code-review-excellence
version: "1.0.7"
maturity: "5-Expert"
specialization: Systematic Code Review & Feedback
description: Systematic code review with constructive feedback, security/performance analysis, knowledge sharing. Use for PRs, feedback on changes, security/perf validation, test validation, review standards.
---

# Code Review Excellence

## 6-Step Process

1. **Context**: Scope, CI/CD status, risk, time available
2. **Architecture**: Design fit, patterns, scalability, tech debt
3. **Line-by-line**: Logic, edge cases, errors, clarity, DRY
4. **Security & Perf**: Input validation, auth, XSS/SQLi, N+1, complexity, cache
5. **Tests**: Happy/edge/error paths, clarity, isolation
6. **Feedback**: Prioritize (blocking/important/nit), acknowledge good work, explain why, clear decision

## Severity

| Level | Mark | Description | Action |
|-------|------|-------------|--------|
| Blocking | 🔴 | Security, logic errors | Must fix |
| Important | 🟡 | Best practices, tests | Should fix |
| Nit | 🟢 | Style, suggestions | Optional |

## Comment Template

```markdown
## Summary
[Overview]

## 🎉 Strengths
- [Well done]

## 🔴 Required (Blocking)
### 1. [Issue]
**Location**: `file.ts:25`
**Issue**: [What]
**Impact**: [Why]
**Fix**: ```code fix```

## 🟡 Important
1. [Improvement]

## 🟢 Nice-to-Have
- [Minor]

## ❓ Questions
- [Clarify]

## Verdict
[✅ Approve | 💬 Comment | 🔄 Request Changes]
```

## Checklists

**Security**: Input validation, auth/authz, parameterized queries, no hardcoded secrets, safe error messages

**Performance**: No N+1, indexed queries, caching, appropriate complexity

**Tests**: Happy path, edge cases, errors, ≥80% coverage

## Language Anti-Patterns

**Python**:
```python
# ❌ Mutable default
def add(item, items=[]): items.append(item)
# ✅ None
def add(item, items=None): items = items or []; items.append(item)
```

**TypeScript**:
```typescript
// ❌ any
function process(data: any) {...}
// ✅ Types
interface Data {value: string}
function process(data: Data) {...}
```

## Communication

| Principle | Target | Behavior |
|-----------|--------|----------|
| Constructive | 95% | Code not person, explain |
| Thorough | 90% | Follow framework, security/perf |
| Actionable | 93% | Severity, specific fixes |
| Educational | 88% | Explain why, resources |
| Efficient | 85% | 24h response, high-impact |

## Pitfalls

| Anti-Pattern | Fix |
|--------------|-----|
| Perfectionism | Use linters, approve when bar met |
| Scope creep | Keep PRs focused |
| Delays | 24h response |
| Harsh tone | Collaborative language |
| Rubber stamp | Actually review |

## Best Practices

- 24h response
- 60min session limit
- Split if >500 lines
- Balance criticism/praise
- Offer pairing
- Use templates
