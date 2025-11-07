---
description: Advanced reflection engine for AI reasoning, session analysis, and research optimization with multi-agent orchestration and meta-cognitive insights
version: "1.0.3"
allowed-tools: Bash(find:*), Bash(grep:*), Bash(git:*), Read, Grep, Task
argument-hint: [session|code|research|workflow] [--mode=quick-check|standard] [--depth=shallow|deep|ultradeep] [--agents=all|specific]
color: purple

execution-modes:
  quick-check:
    description: "Fast health assessment of current work"
    time: "2-5 minutes"
    output: "Health scores + top 3 observations + recommendations"

  standard:
    description: "Comprehensive reflection with multi-agent analysis"
    time: "15-45 minutes"
    output: "Detailed reflection report with strategic recommendations"

agents:
  primary:
    - research-intelligence
  conditional:
    - agent: systems-architect
      trigger: pattern "architecture|design|system" OR argument "code"
    - agent: code-quality
      trigger: pattern "quality|test|lint" OR argument "workflow"
  orchestrated: true

required-plugins: []
graceful-fallbacks: []
---

# Advanced Reflection & Meta-Analysis Engine

**Version**: 1.0.4
**Purpose**: Deep reflection on AI reasoning, session effectiveness, research quality, and development practices.

## Quick Start

**Fast health check** (2-5 min):
```bash
/reflection --mode=quick-check
```

**Comprehensive analysis** (15-45 min):
```bash
/reflection session --depth=deep
/reflection research --agents=all
/reflection code --depth=shallow
```

---

## Phase 0: Reflection Context Discovery

### Session Context
- Current working directory: !`pwd`
- Git repository: !`git rev-parse --show-toplevel 2>/dev/null || echo "Not a git repo"`
- Recent activity: !`git log --oneline --since="24 hours ago" 2>/dev/null | wc -l` commits in 24h
- Active branch: !`git branch --show-current 2>/dev/null`

### Project Analysis
- Total files: !`find . -type f 2>/dev/null | grep -v ".git" | wc -l`
- Code files: !`find . -name "*.py" -o -name "*.jl" -o -name "*.js" -o -name "*.rs" 2>/dev/null | wc -l`
- Documentation: !`find . -name "*.md" -o -name "*.rst" 2>/dev/null | wc -l` files
- Tests: !`find . -name "*test*" -o -name "*spec*" 2>/dev/null | wc -l` files

### Research Indicators
- Papers: !`find . -name "*.tex" -o -name "*.bib" 2>/dev/null | wc -l` files
- Data: !`find . -name "*.csv" -o -name "*.h5" -o -name "*.npz" 2>/dev/null | wc -l` files
- Notebooks: !`find . -name "*.ipynb" 2>/dev/null | wc -l` files
- Figures: !`find . -name "*.png" -o -name "*.pdf" -o -name "*.svg" 2>/dev/null | grep -i "fig\|plot" | wc -l` visualizations

---

## Execution Modes

### Mode 1: Quick Check (--mode=quick-check)

**Purpose**: Fast health assessment of current work (2-5 minutes)

**Workflow**:
```bash
# Step 1: Detect context
! git rev-parse --git-dir > /dev/null 2>&1 && echo "Context: Git repository" || echo "Context: Non-git"

# Step 2: Analyze recent activity
! git log --oneline --since="1 day ago" | wc -l  # Commits today
! find . -name "*.py" -mtime -1 | wc -l  # Modified files

# Step 3: Quick metrics
! git diff --stat  # Changes overview
! grep -r "TODO\|FIXME\|XXX" --include="*.py" --include="*.jl" --include="*.js" | wc -l  # Tech debt

# Step 4: Generate health report
```

**Expected Output**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reflection Quick Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Session Activity:
- Commits: 12 (today)
- Files modified: 23
- Lines changed: +487, -123
- Active time: ~4.5 hours

Code Quality:
- Technical debt markers: 8 TODOs, 2 FIXMEs
- Test coverage: 78% (estimated)
- Documentation: Moderate

Top 3 Observations:
1. ⚠️  High commit frequency - consider consolidating
2. ✅ Good test coverage maintained
3. 📋 8 TODOs added - plan cleanup session

Recommendations:
- Review and consolidate recent commits
- Address 2 FIXME items before proceeding
- Schedule TODO cleanup (est. 1 hour)

Next: /reflection session --depth=deep
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Mode 2: Standard (default)

**Purpose**: Comprehensive reflection with multi-agent analysis (15-45 minutes)

**Invocation**: `/reflection [session|code|research|workflow]`

**Process**:
1. Context discovery (Phase 0)
2. Multi-agent reflection coordination
3. Dimension-specific analysis
4. Strategic insight synthesis
5. Actionable recommendation generation

---

## Phase 1: Reflection Architecture

### Reflection Dimensions

```yaml
reflection_framework:
  meta_cognitive:
    - AI reasoning pattern analysis
    - Decision-making process evaluation
    - Cognitive bias detection
    - Confidence calibration
    - Learning from feedback loops

  technical:
    - Code quality assessment
    - Architecture review
    - Performance bottleneck identification
    - Technical debt quantification
    - Testing coverage and quality

  research:
    - Methodology soundness
    - Experimental design validation
    - Statistical rigor assessment
    - Reproducibility evaluation
    - Publication readiness scoring

  collaborative:
    - Communication effectiveness
    - Stakeholder alignment
    - Documentation quality
    - Knowledge transfer efficiency
    - Team workflow optimization

  strategic:
    - Goal alignment verification
    - Priority assessment
    - Resource allocation review
    - Risk identification
    - Long-term impact projection
```

### Orchestration Structure

```yaml
orchestration:
  coordinator: MetaReflectionOrchestrator

  process:
    1_discovery:
      - Gather session context
      - Identify reflection dimensions needed
      - Select appropriate specialist agents

    2_parallel_reflection:
      - Launch dimension-specific reflections
      - Each agent analyzes from their perspective
      - Generate independent assessments

    3_synthesis:
      - Identify cross-dimension patterns
      - Resolve contradictions
      - Prioritize insights by impact

    4_meta_analysis:
      - Evaluate reflection quality
      - Assess confidence levels
      - Identify blind spots

    5_recommendations:
      - Generate actionable items
      - Prioritize by impact × feasibility
      - Define success metrics
```

---

## Phase 2: Multi-Agent Reflection System

The reflection engine coordinates multiple specialist agents for comprehensive analysis.

### Agent Coordination

**MetaReflectionOrchestrator** coordinates reflection across agents:

```python
class MetaReflectionOrchestrator:
    """
    Coordinates reflection across multiple agents and dimensions

    Capabilities:
    - Parallel agent reflection execution
    - Cross-agent pattern synthesis
    - Meta-cognitive analysis
    - Strategic insight generation
    - Actionable recommendation synthesis
    """

    def orchestrate_reflection(self, context, depth='deep'):
        """
        Multi-layered reflection process

        Layers:
        1. Individual agent reflections (parallel)
        2. Cross-agent pattern synthesis
        3. Meta-cognitive analysis
        4. Strategic insight generation
        5. Actionable recommendation synthesis
        """
        # Parallel reflection
        reflections = self.parallel_agent_reflection(context)

        # Pattern identification
        patterns = self.identify_cross_agent_patterns(reflections)

        # Meta-analysis
        meta_insights = self.analyze_reasoning_patterns(
            reflections, patterns
        )

        # Strategic synthesis
        strategy = self.synthesize_strategic_insights(
            reflections, patterns, meta_insights
        )

        # Actionable recommendations
        recommendations = self.generate_actionable_plan(strategy)

        return ReflectionReport(
            dimensions=reflections,
            patterns=patterns,
            meta_insights=meta_insights,
            strategy=strategy,
            recommendations=recommendations
        )
```

> 📚 **Detailed Implementation**: See [docs/reflection/multi-agent-reflection-system.md](../docs/reflection/multi-agent-reflection-system.md)

---

## Phase 3: Session Analysis & Reasoning Reflection

### ConversationReflectionEngine

Analyzes AI reasoning patterns, conversation effectiveness, and problem-solving approaches.

**Key Capabilities**:
- Reasoning pattern identification (inductive, deductive, abductive)
- Problem-solving strategy assessment
- Communication effectiveness scoring
- Stakeholder need alignment
- Knowledge transfer quality

**Analysis Dimensions**:
1. **Reasoning Patterns** (40%)
   - Logic chain coherence
   - Assumption tracking
   - Evidence utilization
   - Conclusion validity

2. **Problem-Solving** (30%)
   - Approach selection appropriateness
   - Solution creativity and novelty
   - Alternative exploration depth
   - Trade-off analysis quality

3. **Communication** (20%)
   - Clarity and precision
   - Technical depth appropriateness
   - Stakeholder adaptation
   - Documentation quality

4. **Effectiveness** (10%)
   - Goal achievement
   - Efficiency and speed
   - User satisfaction
   - Learning and improvement

**Scoring Framework**:
- **9-10**: Exceptional - Model example
- **7-8**: Strong - Minor improvements possible
- **5-6**: Adequate - Notable gaps exist
- **3-4**: Weak - Significant improvements needed
- **1-2**: Poor - Major restructuring required

> 📚 **Detailed Implementation**: See [docs/reflection/session-analysis-engine.md](../docs/reflection/session-analysis-engine.md)

---

## Phase 4: Scientific Research Reflection

### ResearchReflectionEngine

Validates research methodology, experimental design, data quality, and publication readiness.

**Core Assessment Areas**:

1. **Methodology Soundness** (Score: /10)
   - Hypothesis clarity and testability
   - Appropriate method selection
   - Control condition adequacy
   - Parameter space coverage

2. **Reproducibility** (Score: /10)
   - Code availability and quality
   - Environment specification completeness
   - Data availability or synthetic alternatives
   - Documentation comprehensiveness

3. **Experimental Design** (Score: /10)
   - Statistical power analysis
   - Sample size adequacy
   - Ablation study completeness
   - Parameter sensitivity testing

4. **Data Quality** (Score: /10)
   - Measurement accuracy
   - Collection protocol consistency
   - Bias identification and mitigation
   - Missing data handling

5. **Analysis Rigor** (Score: /10)
   - Statistical test appropriateness
   - Effect size reporting
   - Confidence interval inclusion
   - Multiple testing corrections

6. **Publication Readiness** (Score: /10)
   - Manuscript completeness
   - Figure quality (publication-ready)
   - Reproducibility package
   - Limitations discussion

**Critical Thresholds**:
- **< 5.0**: Major revisions required before submission
- **5.0-6.5**: Significant improvements needed
- **6.5-8.0**: Minor revisions for publication
- **8.0-9.0**: Strong publication candidate
- **> 9.0**: High-impact potential

> 📚 **Detailed Implementation & Scoring Rubrics**: See [docs/reflection/research-reflection-engine.md](../docs/reflection/research-reflection-engine.md)

**Example Assessment**:
```
Research Project Reflection
Overall Score: 7.5/10 - Strong with Critical Gaps

Strengths:
✅ Rigorous experimental design (8.5/10)
✅ High statistical power (8.0/10)
✅ Novel methodological contribution (8.5/10)

Critical Issues:
🔴 Sample size insufficient (n=30, need n=100)
⚠️  Reproducibility gaps (6.0/10)
⚠️  Missing ablation studies

Priority Actions:
1. Collect 70 additional samples (2 weeks)
2. Complete ablation matrix (1 week)
3. Create reproducibility package (3 days)

Expected Timeline: 3-4 weeks to publication ready
```

---

## Phase 5: Code Quality & Workflow Reflection

### DevelopmentReflectionEngine

Assesses code quality, testing practices, technical debt, and development workflow effectiveness.

**Assessment Dimensions**:

1. **Code Quality** (Score: /10)
   - Architecture clarity and modularity
   - Naming consistency and clarity
   - Documentation completeness
   - Error handling robustness
   - Performance considerations

2. **Testing Practices** (Score: /10)
   - Test coverage percentage
   - Test quality and assertions
   - Edge case handling
   - Integration test coverage
   - CI/CD integration

3. **Technical Debt** (Score: /10, inverse)
   - TODO/FIXME density
   - Code duplication percentage
   - Complexity metrics (cyclomatic)
   - Deprecated API usage
   - Security vulnerabilities

4. **Development Workflow** (Score: /10)
   - Commit message quality
   - Branch strategy adherence
   - Code review thoroughness
   - Documentation updates
   - Version control hygiene

**Scoring Interpretation**:
- **8-10**: Excellent practices, sustainable development
- **6-7**: Good practices, minor improvements needed
- **4-5**: Adequate but concerning patterns emerging
- **2-3**: Poor practices, technical debt accumulating
- **0-1**: Critical issues, immediate intervention required

**Example Metrics**:
```
Code Reflection: Project XYZ
Overall Health: 7.2/10 - Good with Areas of Concern

Metrics:
- Test Coverage: 78% (target: 80%)
- Code Complexity: 6.2 avg (target: <10)
- Technical Debt: 23 items (8 TODO, 12 FIXME, 3 XXX)
- Documentation: 65% coverage

Trends:
📈 Test coverage improving (+5% this month)
📉 Technical debt increasing (+8 items)
➡️  Complexity stable

Priority Actions:
1. Address 3 XXX markers (critical)
2. Plan TODO cleanup session (4 hours)
3. Document 6 undocumented modules
```

> 📚 **Detailed Metrics & Tools**: See [docs/reflection/development-reflection-engine.md](../docs/reflection/development-reflection-engine.md)

---

## Phase 6: Comprehensive Report Generation

The reflection engine generates structured reports tailored to each reflection type.

### Report Templates

**Session Reflection Template**:
```markdown
# Session Reflection Report

## Executive Summary
- Overall Effectiveness: {score}/10
- Key Strengths: [3-5 bullet points]
- Primary Improvements: [3-5 bullet points]

## Reasoning Quality
- Pattern Analysis: {score}/10
- Logic Coherence: {score}/10
- Evidence Usage: {score}/10

## Problem-Solving
- Approach Selection: {score}/10
- Solution Quality: {score}/10
- Alternative Exploration: {score}/10

## Communication
- Clarity: {score}/10
- Technical Depth: {score}/10
- Stakeholder Alignment: {score}/10

## Actionable Recommendations
1. [High priority action]
2. [High priority action]
3. [Medium priority action]
```

**Research Reflection Template**:
```markdown
# Research Project Reflection

## Executive Summary
- Overall Score: {score}/10
- Publication Readiness: {tier} (Tier 1/2/3)
- Timeline to Submission: {weeks}

## Critical Findings
- ✅ Strengths: [top 3]
- 🔴 Critical Issues: [must-fix items]
- ⚠️  Improvements: [should-fix items]

## Detailed Assessment
- Methodology: {score}/10
- Reproducibility: {score}/10
- Experimental Design: {score}/10
- Data Quality: {score}/10
- Analysis Rigor: {score}/10
- Publication Readiness: {score}/10

## Priority Actions
[Ordered by urgency and impact]

## Publication Strategy
- Target Venue: [journal/conference]
- Expected Timeline: [weeks to submission]
- Success Probability: [percentage]
```

**Code Reflection Template**:
```markdown
# Development Workflow Reflection

## Health Score: {score}/10

## Quality Metrics
- Test Coverage: {percentage}%
- Code Complexity: {average}
- Technical Debt: {count} items
- Documentation: {percentage}%

## Trends
- [Metric 1]: [trend with direction]
- [Metric 2]: [trend with direction]

## Priority Actions
1. 🔴 [Critical item]
2. ⚠️  [Important item]
3. 📋 [Nice-to-have]

## Long-term Recommendations
[Strategic improvements for sustained health]
```

> 📚 **Complete Templates with Examples**: See [docs/reflection/reflection-report-templates.md](../docs/reflection/reflection-report-templates.md)

---

## Best Practices

### When to Reflect

**Session Reflection** (after significant work):
- After completing major features
- Before important commits/PRs
- After debugging complex issues
- Weekly team retrospectives

**Research Reflection** (periodic checkpoints):
- Before manuscript submission
- After completing experiments
- Monthly progress reviews
- Before presenting findings

**Code Reflection** (continuous):
- Before merging to main
- After refactoring sessions
- Quarterly technical debt reviews
- Sprint retrospectives

### Maximizing Reflection Value

1. **Be Specific**: Vague reflections produce vague insights
2. **Track Over Time**: Compare reflections to identify trends
3. **Act on Insights**: Reflection without action is wasted effort
4. **Share Learnings**: Team reflections compound knowledge
5. **Iterate**: Reflection quality improves with practice

### Common Pitfalls

❌ **Confirmation Bias**: Only looking for expected patterns
✅ **Solution**: Explicitly seek contradictory evidence

❌ **Recency Bias**: Overweighting recent events
✅ **Solution**: Review entire session/project timeline

❌ **Perfection Paralysis**: Setting unrealistic standards
✅ **Solution**: Focus on meaningful improvements

❌ **Reflection Fatigue**: Over-analyzing without action
✅ **Solution**: Set clear action items, limit analysis depth

---

## Integration with Other Commands

**With ultra-think**:
```bash
# Use ultra-think for deep problem analysis
/ultra-think "Optimize database queries" --depth=deep

# Then reflect on the reasoning process
/reflection session --depth=shallow
```

**With code optimization**:
```bash
# After optimization work
/multi-agent-optimize src/ --mode=scan

# Reflect on code quality improvements
/reflection code --depth=deep
```

**With agent improvement**:
```bash
# After improving an agent
/improve-agent customer-support --mode=optimize

# Reflect on the improvement process
/reflection workflow
```

---

## Documentation & Resources

**Core Documentation**:
- [Multi-Agent Reflection System](../docs/reflection/multi-agent-reflection-system.md) - Orchestration patterns
- [Research Reflection Engine](../docs/reflection/research-reflection-engine.md) - Scientific validation
- [Session Analysis Engine](../docs/reflection/session-analysis-engine.md) - Reasoning assessment
- [Development Reflection Engine](../docs/reflection/development-reflection-engine.md) - Code quality
- [Reflection Report Templates](../docs/reflection/reflection-report-templates.md) - Complete examples

**Examples**:
- [Research Reflection Example](../docs/examples/research-reflection-example.md) - Real project analysis
- [Session Reflection Example](../docs/examples/session-reflection-example.md) - Reasoning patterns
- [Code Reflection Example](../docs/examples/code-reflection-example.md) - Technical debt assessment

**Guides**:
- [Best Practices Guide](../docs/guides/best-practices.md) - Maximizing reflection value
- [Advanced Features](../docs/guides/advanced-features.md) - Multi-agent coordination

---

## Version History

**v1.0.4** (2025-11-06):
- Reduced token usage by 30.4% (1704→1186 lines)
- Added --mode=quick-check for fast health assessments
- Enhanced YAML frontmatter with execution modes
- Created comprehensive external documentation
- Improved report template organization

**v1.0.3** (2025-11-06):
- Version consolidation release

**v1.0.2** (2025-01-29):
- Added Constitutional AI framework
- Enhanced with chain-of-thought reasoning

**v1.0.0**:
- Initial release

---

*For questions or issues, see [plugin documentation](https://myclaude.readthedocs.io/en/latest/plugins/ai-reasoning.html)*
