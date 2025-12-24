---
name: code-review-excellence
version: "1.0.6"
maturity: "5-Expert"
specialization: Systematic Code Review & Constructive Feedback
description: Transform code reviews into knowledge sharing through systematic analysis and constructive feedback. Use when reviewing PRs, providing feedback on code changes, evaluating security/performance, validating tests, or establishing review standards.
---

# Code Review Excellence

Systematic code review with constructive feedback, security/performance analysis, and team knowledge sharing.

---

## Delegation Strategy

| Scenario | Delegate To |
|----------|-------------|
| System design before implementation | architect-review |
| Comprehensive security audit | security-auditor |
| Performance profiling | performance-engineer |
| Write missing tests | test-automator |
| Code implementation | domain-specific developer |

---

## 6-Step Review Framework

### Step 1: Context Gathering
- PR scope and business context
- CI/CD status and coverage
- Risk level and dependencies
- Review time available

### Step 2: Architecture Review
- Design fit for problem
- Pattern consistency
- Scalability considerations
- Technical debt introduced

### Step 3: Line-by-Line Analysis
- Logic correctness, edge cases
- Error handling
- Code clarity, naming
- DRY violations

### Step 4: Security & Performance
- Input validation, auth checks
- SQL injection, XSS prevention
- N+1 queries, algorithmic complexity
- Memory leaks, caching

### Step 5: Test Validation
- Coverage for happy/edge/error paths
- Test clarity and independence
- Behavior vs implementation testing

### Step 6: Feedback Synthesis
- Prioritize blocking vs important vs nit
- Acknowledge good work
- Explain the "why"
- Clear decision: Approve/Comment/Request Changes

---

## Severity Classification

| Level | Emoji | Description | Action |
|-------|-------|-------------|--------|
| Blocking | 🔴 | Security, logic errors | Must fix |
| Important | 🟡 | Best practices, tests | Should fix |
| Nit | 🟢 | Style, suggestions | Optional |

---

## Review Comment Template

```markdown
## Summary
[Brief overview]

## 🎉 Strengths
- [What was done well]

## 🔴 Required Changes (Blocking)
### 1. [Issue Title]
**Location**: `file.ts:25`
**Issue**: [Description]
**Impact**: [Why it matters]
**Fix**:
```code
suggested fix
```

## 🟡 Important Suggestions
1. [Improvement with approach]

## 🟢 Nice-to-Have
- [Minor suggestion]

## ❓ Questions
- [Clarification needed]

## Verdict
[✅ Approve | 💬 Comment | 🔄 Request Changes]
```

---

## Security Checklist

- [ ] Input validation and sanitization
- [ ] Authentication/authorization checks
- [ ] Parameterized SQL queries
- [ ] Secrets not hardcoded
- [ ] Error messages don't leak info

## Performance Checklist

- [ ] No N+1 queries
- [ ] Queries use indexes
- [ ] Expensive operations cached
- [ ] Appropriate algorithmic complexity

## Test Checklist

- [ ] Happy path tested
- [ ] Edge cases covered
- [ ] Error cases tested
- [ ] Coverage ≥80%

---

## Language-Specific Patterns

### Python
```python
# ❌ Mutable default
def add_item(item, items=[]):
    items.append(item)

# ✅ Use None
def add_item(item, items=None):
    items = items or []
    items.append(item)
```

### TypeScript
```typescript
// ❌ Using any
function process(data: any) { ... }

// ✅ Proper types
interface DataPayload { value: string }
function process(data: DataPayload) { ... }
```

---

## Communication Principles

| Principle | Target | Key Behavior |
|-----------|--------|--------------|
| Constructive | 95% | Focus on code not person, explain reasoning |
| Thorough | 90% | Follow framework, check security/performance |
| Actionable | 93% | Severity levels, specific fixes, implementable |
| Educational | 88% | Explain why, share resources, mentor |
| Efficient | 85% | Review within 24h, focus on high-impact |

---

## Common Pitfalls

| Anti-Pattern | Fix |
|--------------|-----|
| Perfectionism | Use linters, approve when quality bar met |
| Scope creep | Keep PRs focused |
| Delayed reviews | Respond within 24h |
| Harsh tone | Use collaborative language |
| Rubber stamping | Actually review the code |

---

## Best Practices

- Respond within 24 hours
- Limit sessions to 60 minutes
- Request PR split if >500 lines
- Balance criticism with praise
- Offer to pair on complex issues
- Use templates for consistency

---

## Review Checklist

- [ ] Context understood
- [ ] Architecture reviewed
- [ ] Security issues checked
- [ ] Performance assessed
- [ ] Tests validated
- [ ] Feedback prioritized
- [ ] Tone is constructive
- [ ] Decision is clear

---

**Version**: 1.0.5
