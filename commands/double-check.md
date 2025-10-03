---
description: Multi-angle validation and completeness check with ultrathink reasoning
argument-hint: [work-to-validate]
color: orange
agents:
  primary:
    - multi-agent-orchestrator
    - code-quality-master
  conditional:
    - agent: research-intelligence-master
      trigger: pattern "research|paper|publication|methodology"
    - agent: systems-architect
      trigger: pattern "architecture|design.*pattern|system.*design|scalability"
  orchestrated: true
---

# Double-Check & Validation

## Your Task: $ARGUMENTS

**Execute systematic validation**:

### 1. Define "Complete"
For this task, complete means:
- All requirements addressed
- No missing edge cases
- Quality meets standards
- Documentation sufficient
- Tests cover functionality

### 2. Multi-Angle Analysis
Evaluate from these perspectives:
- **Functional**: Does it work as intended?
- **Quality**: Is code clean, maintainable?
- **Performance**: Any bottlenecks or inefficiencies?
- **Security**: Any vulnerabilities introduced?
- **User Experience**: Is it intuitive, accessible?
- **Maintainability**: Can others understand and modify?

### 3. Completeness Checklist
- [ ] Primary goal achieved
- [ ] Edge cases handled
- [ ] Error handling robust
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance acceptable
- [ ] Security considerations addressed

### 4. Gap Analysis
Identify what's missing or could be improved:
- **Critical gaps**: Must fix before shipping
- **Important gaps**: Should address soon
- **Nice-to-have**: Future improvements

### 5. Alternative Approaches
Consider if alternative approaches might be better:
- Different algorithm?
- Simpler implementation?
- More robust error handling?
- Better abstraction?

**Provide detailed validation report with specific issues found and recommendations**