# Code Review Best Practices

Comprehensive guide to effective code review practices, systematic analysis approaches, and team collaboration strategies for production-grade quality assurance.

## Core Review Principles

### 1. Constructive & Empathetic Communication

**Goal**: Provide feedback that improves code while maintaining positive team dynamics.

**Practices**:
- Frame feedback as suggestions, not demands
- Explain the "why" behind every comment
- Acknowledge good code and clever solutions
- Use "we" language to show collaboration
- Distinguish between blocking issues and suggestions

**Examples**:
```
❌ Bad: "This code is wrong and inefficient."
✅ Good: "Consider using a map here instead of nested loops - it would reduce complexity from O(n²) to O(n) and make the intent clearer."

❌ Bad: "You forgot error handling."
✅ Good: "I notice we're not handling the case where the API returns null. Should we add a check here to prevent potential runtime errors?"
```

### 2. Thoroughness & Systematic Analysis

**Goal**: Ensure comprehensive coverage without getting lost in minutiae.

**Review Layers** (priority order):
1. **Correctness**: Does the code do what it's supposed to?
2. **Security**: Are there vulnerabilities or data exposure risks?
3. **Performance**: Are there obvious bottlenecks or inefficiencies?
4. **Maintainability**: Can future developers understand and modify this?
5. **Testing**: Is there adequate test coverage?
6. **Style**: Does it follow team conventions?

**Systematic Approach**:
```
Phase 1: High-Level Review (5-10% of time)
- Understand overall purpose and design
- Check if approach makes sense
- Identify architectural concerns
- Review PR description and context

Phase 2: Line-by-Line Analysis (60-70% of time)
- Read every changed line carefully
- Check logic, edge cases, error handling
- Verify security and performance
- Note code smells and violations

Phase 3: Integration Review (20-30% of time)
- How does this fit with existing code?
- Are there side effects or breaking changes?
- Is documentation updated?
- Are tests comprehensive?
```

### 3. Actionable & Prioritized Feedback

**Goal**: Make it clear what must be fixed vs. what's optional.

**Priority Levels**:
- **🚨 Blocker (P0)**: Must fix before merge
  - Security vulnerabilities
  - Data loss risks
  - Breaking changes without migration
  - Critical bugs

- **⚠️  High (P1)**: Should fix before merge
  - Performance issues
  - Missing error handling
  - Significant technical debt
  - Important test gaps

- **📝 Medium (P2)**: Fix in follow-up PR
  - Code refactoring opportunities
  - Minor performance improvements
  - Documentation gaps
  - Test enhancements

- **💡 Low (P3)**: Nice-to-have
  - Style improvements
  - Variable naming
  - Comment clarifications
  - Code organization

**Feedback Format**:
```
[Priority] Issue description
Reason: Why this matters
Suggestion: Specific recommendation
Example: Code sample if helpful
```

### 4. Knowledge Sharing & Team Growth

**Goal**: Use reviews as teaching moments without being condescending.

**Practices**:
- Share relevant resources (documentation, blog posts, patterns)
- Explain unfamiliar patterns or techniques
- Ask questions to understand author's reasoning
- Admit when you're uncertain and invite discussion
- Highlight learning opportunities

**Examples**:
```
"I haven't seen this pattern before. Could you explain why you chose this approach over [alternative]? I'd like to understand the trade-offs."

"Great use of the Factory pattern here! For others reading this: this pattern is useful because [explanation]. Here's a good resource: [link]"

"I see you're using Array.reduce() here. For readers less familiar with functional programming, this is equivalent to [imperative version]. The reduce version is more idiomatic in modern JS."
```

### 5. Efficiency & Process Optimization

**Goal**: Provide value without creating bottlenecks.

**Time Management**:
- Small PRs (<400 lines): 15-30 minutes
- Medium PRs (400-800 lines): 30-60 minutes
- Large PRs (>800 lines): Request split or allocate 1-2 hours

**Efficiency Tips**:
- Review PRs promptly (within 1 business day)
- Use review tools and automation (linters, security scanners)
- Create review templates for common patterns
- Batch similar feedback items
- Know when to take discussion offline

**When to Approve Quickly**:
- Urgent hotfixes (review for correctness only)
- Dependency updates with passing tests
- Documentation-only changes
- Minor configuration tweaks

**When to Take Time**:
- New features with complex logic
- Security-sensitive code
- Performance-critical paths
- API/interface changes

---

## Review Checklist Templates

### General Code Review Checklist

**Correctness**:
- [ ] Code does what PR description claims
- [ ] Logic handles edge cases correctly
- [ ] Error conditions are properly handled
- [ ] No obvious bugs or logical errors

**Security**:
- [ ] Input validation for all user inputs
- [ ] No SQL injection vulnerabilities
- [ ] No XSS vulnerabilities
- [ ] Sensitive data properly encrypted/protected
- [ ] Authentication/authorization correctly implemented
- [ ] No secrets in code

**Performance**:
- [ ] No obvious performance bottlenecks
- [ ] Database queries are optimized (no N+1)
- [ ] Appropriate use of caching
- [ ] Resource cleanup (connections, files, etc.)
- [ ] No memory leaks

**Maintainability**:
- [ ] Code is readable and well-organized
- [ ] Functions/methods are appropriately sized
- [ ] Variable names are descriptive
- [ ] Complex logic has explanatory comments
- [ ] No code duplication
- [ ] Follows team/project conventions

**Testing**:
- [ ] New code is covered by tests
- [ ] Tests are meaningful (not just for coverage)
- [ ] Edge cases are tested
- [ ] Tests follow AAA pattern
- [ ] No flaky tests

**Documentation**:
- [ ] Public APIs are documented
- [ ] README updated if needed
- [ ] Breaking changes noted
- [ ] Migration guide provided if needed

### Framework-Specific Checklists

**React/Frontend**:
- [ ] Components follow single responsibility
- [ ] State management is appropriate
- [ ] No prop drilling (use context if needed)
- [ ] Accessibility (ARIA labels, keyboard navigation)
- [ ] Performance (memo, lazy loading where appropriate)
- [ ] Mobile responsiveness considered

**Backend/API**:
- [ ] API contracts are consistent
- [ ] Proper HTTP status codes used
- [ ] Rate limiting considered
- [ ] Pagination for list endpoints
- [ ] API versioning if breaking changes
- [ ] Transaction handling correct

**Database**:
- [ ] Migrations are reversible
- [ ] Indexes on frequently queried columns
- [ ] Foreign key constraints appropriate
- [ ] No data loss in migration
- [ ] Performance impact considered

---

## Common Code Smells & How to Address Them

### 1. Long Methods/Functions

**Smell**: Functions over 50 lines or doing multiple things

**Fix**: Extract smaller focused functions
```python
# Before: 100-line method
def process_order(order):
    # validation (20 lines)
    # calculation (30 lines)
    # persistence (25 lines)
    # notification (25 lines)

# After: Extracted methods
def process_order(order):
    validate_order(order)
    total = calculate_total(order)
    save_order(order, total)
    send_notifications(order)
```

### 2. Complex Conditionals

**Smell**: Nested if/else or long boolean expressions

**Fix**: Extract to well-named methods or use early returns
```javascript
// Before
if (user.isActive && user.hasPermission('admin') && !user.isBanned) {
  if (action === 'delete' || action === 'modify') {
    // ... complex logic
  }
}

// After
if (canPerformAdminAction(user, action)) {
  // ... logic
}

function canPerformAdminAction(user, action) {
  return user.isActive
    && user.hasPermission('admin')
    && !user.isBanned
    && ['delete', 'modify'].includes(action);
}
```

### 3. Magic Numbers

**Smell**: Hardcoded numbers without explanation

**Fix**: Extract to named constants
```python
# Before
if price > 100 and quantity < 5:

# After
BULK_DISCOUNT_THRESHOLD = 100
SMALL_ORDER_LIMIT = 5

if price > BULK_DISCOUNT_THRESHOLD and quantity < SMALL_ORDER_LIMIT:
```

### 4. Code Duplication

**Smell**: Same logic repeated in multiple places

**Fix**: Extract to shared function or class
```typescript
// Before: Duplicated validation
function createUser(data) {
  if (!data.email || !data.email.includes('@')) throw new Error('Invalid email');
  // ...
}

function updateUser(id, data) {
  if (!data.email || !data.email.includes('@')) throw new Error('Invalid email');
  // ...
}

// After: Shared validation
function validateEmail(email) {
  if (!email || !email.includes('@')) {
    throw new Error('Invalid email');
  }
}

function createUser(data) {
  validateEmail(data.email);
  // ...
}
```

### 5. God Objects/Classes

**Smell**: Classes doing too many things (>500 lines, >10 methods)

**Fix**: Apply Single Responsibility Principle - split into focused classes

### 6. Inappropriate Intimacy

**Smell**: Classes accessing each other's internals excessively

**Fix**: Use proper encapsulation, interfaces, dependency injection

---

## Review Communication Guidelines

### Asking Questions

**Use questions to:**
- Understand author's reasoning
- Point out potential issues gently
- Encourage discussion

**Examples**:
```
"Could this function return null? If so, should we handle that case?"
"I'm not familiar with this library - what advantages does it provide over [alternative]?"
"Have you considered how this will behave when [edge case]?"
"What's the expected performance impact of this change at scale?"
```

### Providing Suggestions

**Format**: State observation → Explain concern → Suggest solution

**Examples**:
```
"I notice this query runs inside a loop. This could cause N+1 performance issues. Consider fetching all items in a single query using a WHERE IN clause."

"This error message exposes database structure details. This could be a security risk. Let's use a generic error message for users and log the details internally."
```

### Approving with Caveats

**When to use**: Minor issues that shouldn't block merge but should be addressed

**Format**:
```
"Approving with minor suggestions:
- Consider extracting the validation logic into a separate function for reusability
- The variable name `temp` could be more descriptive

These aren't blockers but would improve code quality. Feel free to address in this PR or a follow-up."
```

### Requesting Changes

**When to use**: Critical issues that must be fixed

**Format**: Be specific and constructive
```
"Requesting changes for the following critical issues:
1. [Security] User input isn't sanitized before database query (line 47) - this creates SQL injection vulnerability
2. [Bug] Division by zero possible when count is 0 (line 82)
3. [Breaking] This changes the API response format without version bump

Please address these issues before merging."
```

---

## Review Response Best Practices

### As PR Author

**Responding to Feedback**:
```
✅ "Great catch! I've updated the code to handle that edge case."
✅ "That's a good point. I chose this approach because [reason], but I see your concern. How about we [compromise]?"
✅ "I hadn't considered that security implication. I've added input validation and will research best practices for this pattern."
⚠️  "Fixed" (too brief, doesn't confirm understanding)
❌ "This is fine as is" (dismissive without justification)
```

**When Disagreeing**:
```
✅ "I understand your concern about performance, but I ran benchmarks and the difference is negligible (<1ms). Here's the data: [link]. However, if you still think we should optimize, I'm happy to do so."
❌ "I disagree" (no reasoning or openness to discussion)
```

### Escalation Path

**When to escalate**:
- Fundamental disagreement on approach
- Unclear requirements
- Security/compliance concerns
- Significant time pressure vs. quality tradeoff

**How to escalate**:
1. Try to resolve through discussion first
2. Bring in tech lead or architect for technical decisions
3. Bring in product for requirement clarifications
4. Document the discussion and decision for future reference

---

## Time-Saving Review Tools

### Automated Checks
- **Linters**: ESLint, Pylint, RuboCop, Checkstyle
- **Security Scanners**: Snyk, Dependabot, SonarQube, CodeQL
- **Test Coverage**: Codecov, Coveralls, Jest coverage
- **Performance**: Lighthouse, WebPageTest, profilers
- **Type Checking**: TypeScript, mypy, Flow

### Code Review Platforms
- **GitHub**: Pull request reviews, suggested changes
- **GitLab**: Merge request reviews
- **Gerrit**: Change-based reviews
- **Phabricator**: Differential reviews
- **Review Board**: Pre-commit reviews

### Browser Extensions
- **Octotree**: GitHub code tree navigation
- **Refined GitHub**: Enhanced GitHub UI
- **CodeStream**: IDE-integrated reviews

---

## Metrics & Continuous Improvement

### Review Quality Metrics
- Average review time
- Number of issues found per review
- Defect escape rate (issues found in production)
- PR revert rate
- Time to first review
- Review cycle time (open to merge)

### Team Health Metrics
- Review participation (reviews per developer)
- Feedback tone (constructive vs. critical)
- Knowledge sharing (learning opportunities identified)
- Collaboration (discussion threads, questions asked)

### Process Improvement
- Regular retrospectives on review process
- Share examples of excellent reviews
- Create team-specific guidelines
- Invest in automation to reduce manual checks
- Balance thoroughness with velocity

---

This guide provides a foundation for effective code reviews. Adapt these practices to your team's culture, technology stack, and workflow. The goal is always better code through constructive collaboration.
