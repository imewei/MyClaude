---
model: claude-sonnet-4-0
---

Comprehensive multi-agent code review system with specialized reviewers and automated quality gates:

[Extended thinking: This advanced review system orchestrates multiple specialized agents across different phases to provide holistic code quality assessment. It includes scope analysis, parallel specialized reviews, cross-cutting concern analysis, automated issue prioritization, fix suggestions, validation gates, and CI/CD integration. The system ensures comprehensive coverage while maintaining efficiency through intelligent coordination.]

## Phase 1: Review Scope Analysis & Planning

### Code Structure Analysis
Use Task tool with subagent_type="search-specialist" to understand review scope:

Prompt: "Analyze codebase structure for comprehensive review of: $ARGUMENTS. Provide:
1. File and component inventory with change analysis
2. Technology stack identification and review requirements
3. Code complexity assessment and review priority areas
4. Integration points and external dependencies analysis
5. Test coverage gaps requiring focused attention
6. Create structured review plan with agent assignments"

### Change Impact Analysis
Use Task tool with subagent_type="git-workflow-manager" to assess changes:

Prompt: "Analyze code changes and git history for: $ARGUMENTS. Determine:
1. Change scope and affected components
2. Risk assessment based on change patterns
3. Historical issue patterns in modified areas
4. Reviewer assignment recommendations
5. Regression testing requirements
6. Create change impact summary for targeted review"

## Phase 2: Core Quality Reviews (Parallel Execution)

Run the following agents IN PARALLEL using multiple Task tool calls in a single message:

### Code Quality & Best Practices
Use Task tool with subagent_type="code-reviewer":

Prompt: "Perform comprehensive code quality review of: $ARGUMENTS. Analyze:
1. Code style consistency and readability standards
2. SOLID principles adherence and design pattern usage
3. Code complexity, duplication, and maintainability metrics
4. Error handling patterns and defensive programming
5. Performance implications of code structure
6. Provide specific line-by-line feedback with improvement suggestions"

### Security & Compliance Review
Use Task tool with subagent_type="security-auditor":

Prompt: "Conduct comprehensive security review of: $ARGUMENTS. Examine:
1. Authentication, authorization, and session management
2. Input validation, sanitization, and injection vulnerabilities
3. Sensitive data handling and encryption practices
4. Security configuration and hardening measures
5. OWASP Top 10 compliance and vulnerability assessment
6. Provide severity ratings, exploitation scenarios, and remediation steps"

### Architecture & Design Review
Use Task tool with subagent_type="architect-reviewer":

Prompt: "Review architecture and design quality of: $ARGUMENTS. Evaluate:
1. Service boundaries, coupling, and cohesion analysis
2. Scalability patterns and performance architecture
3. Design pattern appropriateness and consistency
4. API design quality and versioning strategies
5. Data flow optimization and dependency management
6. Provide architectural improvement recommendations with trade-off analysis"

### Performance & Optimization Review
Use Task tool with subagent_type="performance-engineer":

Prompt: "Analyze performance aspects of: $ARGUMENTS. Review:
1. Algorithm efficiency and computational complexity
2. Memory usage patterns and resource optimization
3. Database query optimization and caching strategies
4. Network request optimization and lazy loading
5. Concurrency and async operation efficiency
6. Provide performance benchmarks and optimization recommendations"

### Test Quality & Coverage Review
Use Task tool with subagent_type="test-automator":

Prompt: "Review testing strategy and implementation for: $ARGUMENTS. Assess:
1. Test coverage analysis and gap identification
2. Test quality, maintainability, and effectiveness
3. Testing strategy appropriateness (unit, integration, e2e)
4. Mock usage and test isolation quality
5. Test performance and CI/CD integration
6. Provide testing improvement recommendations and missing test scenarios"

### Documentation & Knowledge Review
Use Task tool with subagent_type="documentation-expert":

Prompt: "Review documentation quality and completeness for: $ARGUMENTS. Analyze:
1. Code documentation and inline comment quality
2. API documentation completeness and accuracy
3. README, setup, and onboarding documentation
4. Architecture decision records and design docs
5. Knowledge transfer and maintainability documentation
6. Provide documentation improvement recommendations and templates"

## Phase 3: Specialized Domain Reviews (Parallel Execution)

Run the following specialized agents IN PARALLEL:

### Accessibility & Usability Review
Use Task tool with subagent_type="accessibility-tester":

Prompt: "Review accessibility and usability of: $ARGUMENTS. Evaluate:
1. WCAG 2.1 compliance and accessibility standards
2. Screen reader compatibility and keyboard navigation
3. Color contrast and visual accessibility
4. Mobile responsiveness and cross-platform compatibility
5. User experience patterns and interaction design
6. Provide accessibility audit report with priority fixes"

### Dependency & Supply Chain Review
Use Task tool with subagent_type="dependency-manager":

Prompt: "Analyze dependencies and supply chain security for: $ARGUMENTS. Review:
1. Dependency vulnerability analysis and risk assessment
2. License compatibility and legal compliance
3. Package maintenance and update strategy
4. Dependency graph optimization and bloat reduction
5. Supply chain security and trusted source verification
6. Provide dependency recommendations and security updates"

### DevOps & Deployment Review
Use Task tool with subagent_type="devops-engineer":

Prompt: "Review deployment and operations aspects of: $ARGUMENTS. Assess:
1. CI/CD pipeline quality and deployment strategies
2. Infrastructure-as-code and configuration management
3. Monitoring, logging, and observability implementation
4. Containerization and orchestration best practices
5. Environment consistency and configuration management
6. Provide DevOps improvement recommendations and automation opportunities"

### Data Handling & Privacy Review
Use Task tool with subagent_type="data-engineer":

Prompt: "Review data handling and privacy practices for: $ARGUMENTS. Analyze:
1. Data validation, sanitization, and integrity measures
2. Privacy compliance (GDPR, CCPA) and data protection
3. Data storage optimization and query performance
4. Data pipeline reliability and error handling
5. Backup, recovery, and data lifecycle management
6. Provide data governance recommendations and compliance improvements"

### UI/UX Design Review
Use Task tool with subagent_type="ui-designer":

Prompt: "Review user interface and experience design for: $ARGUMENTS. Evaluate:
1. Design system consistency and component reusability
2. User interaction patterns and workflow optimization
3. Visual hierarchy and information architecture
4. Responsive design and cross-device experience
5. Loading states, error handling, and user feedback
6. Provide UI/UX improvement recommendations with design principles"

## Phase 4: Cross-Cutting Analysis & Integration Review

### Error Handling & Resilience Review
Use Task tool with subagent_type="error-detective":

Prompt: "Analyze error handling and system resilience for: $ARGUMENTS. Review:
1. Error propagation patterns and exception handling
2. Failure recovery mechanisms and circuit breakers
3. Logging and monitoring for error detection
4. Input validation and boundary condition handling
5. System degradation and fallback strategies
6. Provide resilience improvement recommendations and error handling patterns"

### Compliance & Regulatory Review
Use Task tool with subagent_type="compliance-auditor":

Prompt: "Review regulatory compliance and standards adherence for: $ARGUMENTS. Assess:
1. Industry-specific compliance requirements (HIPAA, PCI-DSS, SOX)
2. Data protection and privacy regulation compliance
3. Security framework adherence (ISO 27001, NIST)
4. Audit trail and record-keeping requirements
5. Compliance automation and continuous monitoring
6. Provide compliance gap analysis and remediation roadmap"

## Phase 5: Issue Consolidation & Prioritization

### Findings Analysis & Prioritization
Use Task tool with subagent_type="data-analyst" to consolidate all review findings:

Prompt: "Consolidate and prioritize all review findings for: $ARGUMENTS. Create:
1. Comprehensive issue inventory with severity classification
2. Cross-agent finding correlation and duplicate resolution
3. Risk-impact matrix for prioritization
4. Fix effort estimation and resource requirements
5. Dependency analysis between issues
6. Create actionable review report with clear next steps"

### Automated Fix Recommendations
Use Task tool with subagent_type="refactoring-specialist":

Prompt: "Generate automated fix recommendations for: $ARGUMENTS. Provide:
1. Automated refactoring opportunities and safe transformations
2. Code generation templates for common improvements
3. Tool-assisted fix procedures and scripts
4. Manual fix guidance with step-by-step instructions
5. Verification procedures for each fix category
6. Create fix implementation plan with validation checkpoints"

## Phase 6: Quality Gates & Validation

### Pre-Merge Validation
Use Task tool with subagent_type="qa-expert" for final validation:

Prompt: "Perform pre-merge quality validation for: $ARGUMENTS. Ensure:
1. All critical and high-priority issues are addressed
2. Code meets minimum quality standards and gates
3. Security vulnerabilities are properly mitigated
4. Test coverage meets project requirements
5. Documentation is complete and accurate
6. Create merge readiness report with go/no-go recommendation"

### CI/CD Integration Check
Use Task tool with subagent_type="deployment-engineer":

Prompt: "Validate CI/CD integration and deployment readiness for: $ARGUMENTS. Verify:
1. Pipeline compatibility and deployment safety
2. Environment configuration and rollback procedures
3. Monitoring and alerting setup for new changes
4. Feature flag and gradual rollout capabilities
5. Post-deployment validation and health checks
6. Create deployment plan with risk mitigation strategies"

## Comprehensive Review Report Structure

### Executive Summary
- **Overall Quality Score**: Calculated from all review dimensions
- **Merge Recommendation**: Go/No-Go with conditions
- **Risk Assessment**: High/Medium/Low with risk factors
- **Effort Estimation**: Time and resources required for fixes

### Critical Issues (🚨 Block Merge)
- **Security Vulnerabilities**: Exploitable security flaws
- **Functional Defects**: Breaking changes or critical bugs
- **Architecture Violations**: Major design principle violations
- **Compliance Failures**: Regulatory or policy violations

### High Priority Issues (⚠️ Fix Before Next Release)
- **Performance Degradation**: Significant performance impacts
- **Quality Debt**: Technical debt with high interest
- **Missing Critical Tests**: Untested critical functionality
- **Documentation Gaps**: Missing essential documentation

### Medium Priority Issues (📋 Address in Sprint)
- **Code Quality**: Style, complexity, and maintainability issues
- **Test Coverage**: Missing non-critical test coverage
- **Performance Optimization**: Non-critical performance improvements
- **Accessibility Issues**: Non-blocking accessibility improvements

### Low Priority Issues (💡 Future Improvements)
- **Style Inconsistencies**: Minor formatting and style issues
- **Refactoring Opportunities**: Code improvement suggestions
- **Documentation Enhancements**: Nice-to-have documentation
- **Tool Automation**: Process improvement opportunities

### Positive Findings (✅ Excellent Practices)
- **Security Best Practices**: Well-implemented security measures
- **Quality Architecture**: Excellent design patterns and structure
- **Comprehensive Testing**: Outstanding test coverage and quality
- **Clear Documentation**: Excellent documentation practices
- **Performance Optimization**: Efficient and optimized code

### Metrics & Analytics
- **Code Quality Score**: Aggregated quality metrics
- **Security Score**: Security posture assessment
- **Test Coverage**: Percentage and quality metrics
- **Performance Impact**: Measured performance changes
- **Complexity Metrics**: Cyclomatic complexity and maintainability
- **Documentation Coverage**: Documentation completeness percentage

### Action Plan & Next Steps
1. **Immediate Actions**: Critical fixes required before merge
2. **Short-term Plan**: High-priority improvements for next sprint
3. **Long-term Roadmap**: Medium and low priority improvements
4. **Process Improvements**: Review process and tooling enhancements
5. **Knowledge Sharing**: Learning opportunities and best practices

### Automated Fix Scripts
- **Linting Fixes**: Automated code style corrections
- **Security Patches**: Automated dependency updates
- **Refactoring Scripts**: Safe automated refactoring
- **Test Generation**: Automated test scaffolding
- **Documentation Templates**: Auto-generated documentation stubs

## Command Usage

### Basic Usage
```bash
/multi-agent-review [file-path-or-PR-number]
```

### Advanced Usage
```bash
/multi-agent-review [target] --scope=[full|security|performance|quality|docs] --priority=[critical|all] --format=[detailed|summary|json] --auto-fix=[true|false] --gate=[strict|standard|permissive]
```

### Arguments & Options
- `target`: File path, directory, or PR number to review
- `--scope`: Limit review to specific domains
- `--priority`: Filter issues by priority level
- `--format`: Output format for review results
- `--auto-fix`: Enable automated fix suggestions
- `--gate`: Quality gate strictness level

### Integration Examples
```bash
# Full review of current changes
/multi-agent-review . --scope=full --gate=strict

# Security-focused review
/multi-agent-review src/ --scope=security --priority=critical

# PR review with auto-fix suggestions
/multi-agent-review PR-123 --auto-fix=true --format=detailed

# Quick quality check
/multi-agent-review app.py --scope=quality --format=summary
```

Target for review: $ARGUMENTS