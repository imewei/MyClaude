# Reasoning Patterns Taxonomy

## Overview

This taxonomy catalogs different types of reasoning patterns used in AI responses, providing examples, effectiveness metrics, and usage guidelines for meta-cognitive reflection.

## Pattern 1: Deductive Reasoning

### Definition
Drawing specific conclusions from general principles using logical inference.

### Structure
```
Premise 1: General principle or rule
Premise 2: Specific case or condition
Conclusion: Specific outcome logically following from premises
```

### Examples

**Example 1: Code Quality**
```
Premise 1: All Python functions should include type hints for maintainability
Premise 2: This is a Python function without type hints
Conclusion: This function should be updated to include type hints
```

**Example 2: Testing**
```
Premise 1: Functions with side effects require integration tests
Premise 2: This function modifies database state (side effect)
Conclusion: This function requires integration tests
```

### Effectiveness Metrics
- **Logical Validity**: 98% (very high when premises are correct)
- **Appropriate Use Cases**: Code review, policy application, rule-based decisions
- **Strengths**: Certainty when premises are valid, clear reasoning chain
- **Weaknesses**: Only as good as premises, can be rigid

### Usage Guidelines
- ✅ Use when: Clear rules or principles apply, logical certainty needed
- ⚠️ Caution when: Premises may be incomplete or contextual
- ❌ Avoid when: Creative or exploratory thinking required

---

## Pattern 2: Inductive Reasoning

### Definition
Drawing general conclusions from specific observations or examples.

### Structure
```
Observation 1: Specific instance with pattern
Observation 2: Another specific instance with same pattern
Observation 3: Yet another instance with same pattern
Conclusion: General principle or pattern likely applies broadly
```

### Examples

**Example 1: Performance Pattern**
```
Observation 1: Function A with nested loops has O(n²) performance issues
Observation 2: Function B with similar nested loop pattern has same issues
Observation 3: Function C also exhibits this pattern and performance problem
Conclusion: This nested loop pattern generally causes quadratic performance issues
```

**Example 2: Bug Pattern**
```
Observation 1: Memory leak in component using callback without cleanup
Observation 2: Same leak pattern in another component with similar callback
Observation 3: Third component shows same issue
Conclusion: Callbacks without cleanup generally cause memory leaks
```

### Effectiveness Metrics
- **Accuracy**: 75% (depends on sample size and representativeness)
- **Appropriate Use Cases**: Pattern detection, hypothesis generation, trend analysis
- **Strengths**: Good for discovering patterns, adapts to data
- **Weaknesses**: Requires sufficient examples, prone to overgeneralization

### Usage Guidelines
- ✅ Use when: Sufficient examples available (3+ instances), pattern recognition needed
- ⚠️ Caution when: Limited samples, high variability in data
- ❌ Avoid when: Single example, premature generalization risk
- **Improvement**: Increase sample size before drawing strong conclusions

---

## Pattern 3: Abductive Reasoning

### Definition
Inferring the best or most likely explanation for observations.

### Structure
```
Observation: Something unexpected or requiring explanation
Possible Explanations: [Cause 1, Cause 2, Cause 3, ...]
Evaluation: Assess likelihood of each explanation
Best Explanation: Most plausible cause given evidence
```

### Examples

**Example 1: Test Failures**
```
Observation: Tests passing locally but failing in CI
Possible Causes:
  - Environment differences (OS, Python version)
  - Dependency version mismatch
  - Time-dependent test behavior
  - Network connectivity issues
Evaluation: Dependency versions not pinned, CI uses different Python version
Best Explanation: Dependency version mismatch most likely (supports evidence)
```

**Example 2: Performance Degradation**
```
Observation: API response time increased from 100ms to 500ms
Possible Causes:
  - Database query inefficiency
  - Increased data volume
  - Network latency
  - Memory leak causing GC pressure
Evaluation: Coincides with database migration, query plan changed
Best Explanation: Database query inefficiency (query plan regression)
```

### Effectiveness Metrics
- **Best Explanation Accuracy**: 85% (high when hypothesis space is complete)
- **Appropriate Use Cases**: Debugging, root cause analysis, hypothesis generation
- **Strengths**: Excellent for diagnostics, handles uncertainty
- **Weaknesses**: May miss non-obvious causes, requires good hypothesis generation

### Usage Guidelines
- ✅ Use when: Multiple possible causes exist, debugging needed
- ⚠️ Caution when: Hypothesis space may be incomplete
- ❌ Avoid when: Certainty required, no evidence for comparison
- **Best Practice**: Generate comprehensive hypothesis space before selecting best explanation

---

## Pattern 4: Analogical Reasoning

### Definition
Drawing parallels from similar domains or situations to understand or solve problems.

### Structure
```
Problem: Current situation or challenge
Analogy: Similar situation from different domain
Mapping: Correspondence between problem and analogy
Solution: Insights or approaches transferred from analogy
```

### Examples

**Example 1: Data Pipeline Optimization**
```
Problem: Slow data pipeline processing large datasets
Analogy: Manufacturing assembly line optimization
Mapping:
  - Pipeline stages ↔ Assembly stations
  - Data batches ↔ Product batches
  - Bottlenecks ↔ Slow stations
Solution: Profile each stage (station), optimize slowest bottleneck, parallelize where possible
```

**Example 2: Microservices Architecture**
```
Problem: Designing service boundaries
Analogy: Restaurant organization (kitchen, dining, bar)
Mapping:
  - Services ↔ Restaurant departments
  - APIs ↔ Communication between departments
  - Data ownership ↔ Department responsibilities
Solution: Group by business capability, clear interfaces, independent operation
```

### Effectiveness Metrics
- **Helpfulness**: 92% (very high for explanation and understanding)
- **Appropriate Use Cases**: Teaching, design patterns, problem-solving
- **Strengths**: Excellent pedagogical value, aids understanding complex concepts
- **Weaknesses**: Can oversimplify, analogies may break down at edges

### Usage Guidelines
- ✅ Use when: Explaining complex concepts, exploring design options
- ⚠️ Caution when: Analogy may be imperfect or misleading
- ❌ Avoid when: Precision required, analogical reasoning insufficient
- **Best Practice**: Explicitly state where analogy applies and where it breaks down

---

## Pattern 5: Causal Reasoning

### Definition
Identifying cause-effect relationships and causal chains.

### Structure
```
Cause: Initial condition or action
Mechanism: Causal pathway
Effect: Resulting outcome
Chain: Cause → Intermediate Effect → Final Effect
```

### Examples

**Example 1: Memory Performance Issue**
```
Cause: Large array allocation inside loop
Mechanism:
  → Repeated memory allocation
  → Memory pressure increases
  → Garbage collector runs frequently
  → GC pauses execution
Effect: Slow performance (5x slower than expected)
```

**Example 2: Test Failure Cascade**
```
Cause: Database connection not properly mocked
Mechanism:
  → Test connects to real database
  → Creates test data in production
  → Violates production data constraints
  → Throws constraint violation exception
Effect: Test fails with confusing error message
```

### Effectiveness Metrics
- **Causal Chain Accuracy**: 88% (high when mechanisms understood)
- **Average Chain Depth**: 3.2 steps
- **Appropriate Use Cases**: Root cause analysis, debugging, impact assessment
- **Strengths**: Reveals underlying mechanisms, predicts consequences
- **Weaknesses**: May miss hidden causes, requires domain knowledge

### Usage Guidelines
- ✅ Use when: Understanding mechanisms, predicting impacts, debugging
- ⚠️ Caution when: Multiple causal pathways possible, interactions complex
- ❌ Avoid when: Correlation without causation, insufficient evidence
- **Best Practice**: Trace complete causal chain, verify each link

---

## Reasoning Quality Metrics

### Logical Validity
**Definition**: Conclusions follow logically from premises
**Target**: >95% for formal reasoning
**Assessment**: Check argument structure, identify logical fallacies

### Evidence Strength
**Definition**: Claims supported by sufficient evidence
**Target**: >85% of claims have supporting evidence
**Assessment**: Verify evidence quality, quantity, relevance

### Uncertainty Handling
**Definition**: Explicit acknowledgment of uncertainty
**Target**: >90% of uncertain statements flagged
**Assessment**: Check for hedging language, confidence levels

### Bias Presence
**Definition**: Cognitive biases affect reasoning
**Target**: <5 significant bias instances per session
**Assessment**: Use cognitive bias checklist

### Alternative Consideration
**Definition**: Multiple approaches considered
**Target**: Average 3+ alternatives per decision
**Assessment**: Count explicit alternatives mentioned

---

## Reasoning Effectiveness Assessment

### High Effectiveness (9-10/10)
- Multiple reasoning types used appropriately
- Logical validity >98%
- Evidence-based with minimal bias
- Uncertainty well-quantified
- Multiple alternatives considered
- Clear, traceable reasoning chains

### Good Effectiveness (7-8/10)
- Appropriate reasoning types selected
- Logical validity >90%
- Most claims supported by evidence
- Some uncertainty acknowledged
- At least 2 alternatives considered
- Generally clear reasoning

### Moderate Effectiveness (5-6/10)
- Reasoning types sometimes appropriate
- Logical validity >75%
- Evidence present but incomplete
- Uncertainty rarely acknowledged
- Limited alternatives considered
- Reasoning occasionally unclear

### Low Effectiveness (1-4/10)
- Inappropriate reasoning types
- Logical validity <75%
- Insufficient evidence
- Uncertainty ignored
- No alternatives considered
- Unclear or circular reasoning

---

## Common Reasoning Antipatterns

### Antipattern 1: Hasty Generalization
**Description**: Drawing broad conclusions from insufficient examples
**Example**: One function with recursion is slow → All recursion is slow
**Fix**: Require minimum 3-5 examples before generalizing

### Antipattern 2: False Dichotomy
**Description**: Presenting only two options when more exist
**Example**: "Use SQL or NoSQL" (ignoring NewSQL, graph DBs, etc.)
**Fix**: Explicitly enumerate all reasonable alternatives

### Antipattern 3: Circular Reasoning
**Description**: Conclusion restates premise without adding insight
**Example**: "Code is maintainable because it's easy to maintain"
**Fix**: Provide independent evidence for conclusion

### Antipattern 4: Post Hoc Fallacy
**Description**: Assuming correlation implies causation
**Example**: "Deployed on Friday, system crashed → Friday deployments cause crashes"
**Fix**: Verify causal mechanism, not just temporal correlation

### Antipattern 5: Appeal to Authority
**Description**: Accepting claim solely based on source
**Example**: "This is correct because expert X said so"
**Fix**: Evaluate claim on its merits, even from authorities

---

## Usage in Meta-Cognitive Reflection

### Analysis Process

1. **Pattern Identification**
   - Read conversation or reasoning chain
   - Tag each reasoning instance with type
   - Count occurrences of each pattern

2. **Effectiveness Evaluation**
   - For each pattern instance, assess appropriateness
   - Check logical validity (deductive)
   - Verify sample size (inductive)
   - Evaluate hypothesis completeness (abductive)
   - Check analogy applicability (analogical)
   - Trace causal chain (causal)

3. **Quality Metrics**
   - Calculate logical validity percentage
   - Assess evidence strength
   - Check uncertainty handling
   - Identify biases
   - Count alternatives considered

4. **Insights Generation**
   - Identify reasoning strengths
   - Note recurring antipatterns
   - Assess overall effectiveness
   - Generate improvement recommendations

5. **Report Documentation**
   - Document pattern usage statistics
   - Include effectiveness metrics
   - Provide concrete examples
   - List actionable improvements

### Example Analysis Output

```markdown
## Reasoning Pattern Analysis

### Pattern Distribution
- Deductive: 45 instances (34%)
- Inductive: 32 instances (24%)
- Abductive: 28 instances (21%)
- Analogical: 19 instances (14%)
- Causal: 52 instances (39%)
Total: 132 reasoning instances identified

### Effectiveness Assessment

**Deductive Reasoning**: 9.5/10 ✅
- 98% logical validity
- Appropriate premise selection
- Clear reasoning chains
- Strong performance

**Inductive Reasoning**: 6.5/10 ⚠️
- 75% accuracy (below target 85%)
- Issue: Small sample sizes (avg 2.3 examples)
- Recommendation: Require 3+ examples before generalizing

**Abductive Reasoning**: 8.5/10 ✅
- 85% best explanation selected
- Good hypothesis generation
- Thorough evaluation process

**Analogical Reasoning**: 9.2/10 ⭐
- Excellent pedagogical value
- Clear, helpful analogies
- Appropriate applicability statements

**Causal Reasoning**: 8.8/10 ✅
- Strong causal chain tracing
- Average depth 3.2 steps
- Good mechanism understanding

### Key Insights

**Strengths**:
1. Excellent use of deductive reasoning for rule application
2. Highly effective analogical reasoning for explanations
3. Strong causal analysis for debugging

**Areas for Improvement**:
1. ⚠️ Inductive reasoning needs larger sample sizes
2. ⚠️ Uncertainty quantification could be more explicit
3. ⚠️ Alternative generation averaging 2.1 (target: 3+)

**Recommendations**:
1. Require minimum 3 examples before making inductive generalizations
2. Explicitly quantify uncertainty using probability or confidence levels
3. Systematically generate at least 3 alternatives before selecting approach
```
