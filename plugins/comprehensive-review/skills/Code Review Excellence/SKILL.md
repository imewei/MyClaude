---
name: Code Review Excellence
version: "2.1.0"
maturity: "5-Expert"
specialization: Code Review
description: Conduct systematic code reviews with structured analysis and constructive feedback. Use when reviewing PRs, analyzing code for bugs/security/performance, or establishing review standards.
---

# Code Review Excellence

Systematic code review with security assessment and constructive feedback.

---

## Six-Step Review Framework

1. **Understand Context** - PR description, issues, design docs
2. **High-Level Review** - Architecture and overall approach
3. **Detailed Analysis** - Line-by-line for bugs and edge cases
4. **Security Check** - Vulnerabilities and security concerns
5. **Performance Review** - Algorithmic efficiency and resources
6. **Constructive Feedback** - Actionable, empathetic suggestions

---

## Review Categories

| Category | Examples |
|----------|----------|
| Must Fix | Security vulnerabilities, bugs, data loss |
| Should Fix | Performance issues, maintainability |
| Consider | Style preferences, alternatives |
| Praise | Excellent code, clever solutions |

---

## Security Review Pattern

```python
# BAD - SQL injection vulnerable
query = f"SELECT * FROM users WHERE id = {user_id}"

# GOOD - parameterized query
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

---

## Performance Review Pattern

```python
# BAD - N+1 queries
for user in users:
    orders = Order.objects.filter(user=user)

# GOOD - eager loading
users = User.objects.prefetch_related('orders').all()
```

---

## Feedback Principles

| Principle | Description |
|-----------|-------------|
| Empathy | Consider author's perspective |
| Specificity | Point to exact lines |
| Educational | Explain the "why" |
| Balanced | Acknowledge good code |
| Actionable | Clear next steps |

---

## PR Checklist

- [ ] Logic correct, edge cases handled
- [ ] Input validation present
- [ ] No SQL injection/XSS vulnerabilities
- [ ] No N+1 queries or memory leaks
- [ ] Tests cover critical paths
- [ ] Documentation updated

---

## Best Practices

| Practice | Guideline |
|----------|-----------|
| Review size | 200-400 lines per session |
| Timeliness | Review within 24 hours |
| Automation | Let linters catch style |
| Follow-up | Ensure comments addressed |

---

**Version**: 1.0.5
