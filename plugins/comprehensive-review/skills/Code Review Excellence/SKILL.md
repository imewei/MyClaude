---
name: Code Review Excellence
description: Conduct systematic, high-quality code reviews with structured analysis, security assessment, and constructive feedback. Use when reviewing pull requests, GitHub PRs, GitLab merge requests, or any code changes submitted for review. Use when analyzing code for bugs, security vulnerabilities, performance issues, or maintainability concerns. Use when writing code review comments, PR feedback, or review summaries. Use when evaluating code quality, test coverage, error handling, or architectural decisions. Use when assessing third-party libraries, dependencies, or external code contributions. Use when training team members on code review practices or establishing review standards. Use when conducting security-focused reviews for authentication, authorization, input validation, SQL injection, XSS, or CSRF vulnerabilities. Use when performing performance reviews for N+1 queries, memory leaks, algorithmic complexity, or resource management. Use when checking code against style guides, linting rules, or team conventions.
---

# Code Review Excellence

Master effective code review practices with systematic analysis, constructive feedback, and team collaboration.

## When to use this skill

- Reviewing pull requests (PRs) on GitHub, GitLab, Bitbucket, or Azure DevOps
- Writing code review comments or feedback on merge requests
- Analyzing code changes for bugs, edge cases, or logical errors
- Conducting security-focused code reviews for vulnerabilities (SQL injection, XSS, CSRF, authentication flaws)
- Performing performance reviews (N+1 queries, memory leaks, inefficient algorithms)
- Evaluating test coverage and test quality in submitted code
- Assessing architectural changes and design pattern usage
- Reviewing code for compliance with style guides, linting rules, or team conventions
- Providing constructive feedback that educates and improves team skills
- Establishing code review checklists, standards, and best practices
- Training junior developers on effective code review techniques
- Evaluating third-party code, dependencies, or open-source contributions
- Checking error handling, logging, and observability in code changes
- Reviewing database migrations, API changes, or schema modifications
- Assessing code readability, documentation, and maintainability

## Core Concepts

### 1. Six-Step Review Framework

1. **Understand Context**: Read the PR description, related issues, and design docs
2. **High-Level Review**: Assess architecture, design patterns, and overall approach
3. **Detailed Analysis**: Line-by-line review for bugs, edge cases, and issues
4. **Security Check**: Identify potential vulnerabilities and security concerns
5. **Performance Review**: Evaluate algorithmic efficiency and resource usage
6. **Constructive Feedback**: Provide actionable, empathetic suggestions

### 2. Constitutional AI Principles for Reviews

- **Empathy First**: Consider the author's perspective and effort
- **Specificity**: Point to exact lines and provide concrete suggestions
- **Educational**: Explain the "why" behind suggestions
- **Balanced**: Acknowledge good code alongside improvement areas
- **Actionable**: Every comment should have a clear next step

### 3. Review Categories

- **Must Fix**: Security vulnerabilities, bugs, data loss risks
- **Should Fix**: Performance issues, maintainability concerns
- **Consider**: Style preferences, alternative approaches
- **Praise**: Highlight excellent code and clever solutions

## Quick Start

```markdown
## PR Review Checklist

### Correctness
- [ ] Logic is correct and handles edge cases
- [ ] Error handling is appropriate
- [ ] No obvious bugs or regressions

### Security
- [ ] Input validation present
- [ ] No SQL injection or XSS vulnerabilities
- [ ] Secrets are not exposed

### Performance
- [ ] No N+1 queries or inefficient algorithms
- [ ] Resources are properly cleaned up
- [ ] Caching considered where appropriate

### Maintainability
- [ ] Code is readable and self-documenting
- [ ] Tests cover critical paths
- [ ] Documentation updated if needed
```

## Review Patterns

### Pattern 1: Security-Focused Review

```python
# Example: Identifying SQL injection vulnerability

# BAD - vulnerable to SQL injection
query = f"SELECT * FROM users WHERE id = {user_id}"

# GOOD - parameterized query
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

### Pattern 2: Performance Review

```python
# Example: Identifying N+1 query problem

# BAD - N+1 queries
for user in users:
    orders = Order.objects.filter(user=user)  # Query per user!

# GOOD - eager loading
users = User.objects.prefetch_related('orders').all()
```

### Pattern 3: Constructive Feedback Examples

```markdown
# Instead of:
"This code is wrong."

# Say:
"This approach might cause issues when the input is empty.
Consider adding a check like: `if not items: return []`
This ensures we handle the edge case gracefully."
```

## Best Practices

1. **Review in Small Chunks**: Limit review sessions to 200-400 lines
2. **Use Checklists**: Consistent review criteria across the team
3. **Automate What You Can**: Let linters catch style issues
4. **Be Timely**: Review within 24 hours when possible
5. **Follow Up**: Ensure comments are addressed before approval
