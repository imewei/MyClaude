---
name: code-review
version: "2.1.0"
description: Systematic process for code review focused on security, performance, maintainability, and knowledge sharing.
---

# Code Review Excellence

Expert guide for conducting thorough and constructive code reviews.

## 1. The 6-Step Review Process

1.  **Context**: Understand the scope and risks of the change.
2.  **Architecture**: Verify design patterns and system fit.
3.  **Logic**: Check for edge cases, error handling, and DRY principles.
4.  **Security & Perf**: Look for vulnerabilities and performance bottlenecks (e.g., N+1 queries).
5.  **Tests**: Ensure adequate coverage and meaningful assertions.
6.  **Feedback**: Provide actionable, prioritized, and constructive comments.

## 2. Review Checklists

### Security
- [ ] No hardcoded secrets or credentials.
- [ ] Input validation and output encoding present.
- [ ] Protection against common attacks (SQLi, XSS, CSRF).

### Performance
- [ ] Efficient database queries (indexed, no N+1).
- [ ] Minimal memory allocations in hot paths.
- [ ] Appropriate use of caching.

## 3. Communication Standards

- **Prioritize**: Distinguish between blocking issues, important improvements, and minor nits.
- **Tone**: Use collaborative language ("We should consider...") rather than accusatory.
- **Efficiency**: Aim for a 24-hour response time on reviews.

## 4. Code Review Checklist

- [ ] Logic is correct and handles error states.
- [ ] Code follows project style and idiomatic patterns.
- [ ] Automated tests pass and cover new logic.
- [ ] Documentation is updated if public APIs changed.
