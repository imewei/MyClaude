---
version: "1.0.3"
category: "llm-application-dev"
command: "/prompt-optimize"
description: Optimize prompts for better LLM performance through advanced techniques including CoT, few-shot learning, and constitutional AI
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: "<prompt_to_optimize>"
color: green
execution_modes:
  quick: "5-10 minutes - Basic prompt analysis and quick optimization recommendations"
  standard: "15-25 minutes - Comprehensive optimization with techniques and examples"
  comprehensive: "30-45 minutes - Full optimization with meta-prompt generation and A/B testing strategy"
agents:
  primary:
    - prompt-engineer
  conditional:
    - agent: ai-engineer
      trigger: pattern "code.*generation|implementation.*example"
  orchestrated: false
---

# Prompt Optimization

Transform basic instructions into production-ready prompts through advanced techniques. Effective prompt engineering can improve accuracy by 40%, reduce hallucinations by 30%, and cut costs by 50-80%.

## Quick Reference

| Topic | External Documentation | Lines |
|-------|------------------------|-------|
| **Optimization Techniques** | [prompt-patterns.md](../docs/prompt-patterns.md) | ~500 |
| **Reference Examples** | [prompt-examples.md](../docs/prompt-examples.md) | ~400 |
| **Evaluation Methods** | [prompt-evaluation.md](../docs/prompt-evaluation.md) | ~300 |

**Total External Documentation**: ~1,200 lines of patterns, examples, and testing strategies

## Requirements

$ARGUMENTS

## Core Workflow

### Phase 1: Analyze Current Prompt

**Evaluate prompt across key dimensions**:

**Assessment Framework**:
- **Clarity** (1-10): Ambiguity points, explicit vs implicit expectations
- **Structure**: Logical flow, section boundaries
- **Model Alignment**: Capability utilization, token efficiency
- **Performance**: Success rate, failure modes, edge cases

**Decomposition**:
- Core objective and constraints
- Output format requirements
- Context dependencies
- Variable elements

**Example Analysis**:
```
Original: "Analyze this customer feedback and determine sentiment"

Issues:
- Ambiguous: What aspects to analyze?
- Missing: Output format specification
- Incomplete: No handling of mixed sentiment
- Score: 4/10 clarity
```

**Complete Framework**: [prompt-patterns.md#analysis-framework](../docs/prompt-patterns.md)

### Phase 2: Apply Chain-of-Thought

**Enhance reasoning with step-by-step thinking**:

**Standard CoT Pattern**:
```python
# Before
prompt = "Analyze this customer feedback and determine sentiment"

# After: CoT enhanced
prompt = """Analyze this customer feedback step by step:

1. Identify key phrases indicating emotion
2. Categorize each phrase (positive/negative/neutral)
3. Consider context and intensity
4. Weigh overall balance
5. Determine dominant sentiment and confidence

Customer feedback: {feedback}

Step 1 - Key emotional phrases:"""
```

**CoT Variants**:
- **Zero-Shot CoT**: Add "Let's think step-by-step"
- **Few-Shot CoT**: Provide examples with reasoning
- **Tree-of-Thoughts**: Explore multiple solution paths

**Complete Patterns**: [prompt-patterns.md#chain-of-thought-patterns](../docs/prompt-patterns.md#chain-of-thought-patterns)

### Phase 3: Add Few-Shot Learning

**Provide strategic examples**:

**Example Selection Strategy**:
1. **Simple case**: Demonstrates basic pattern
2. **Edge case**: Shows handling of complexity
3. **Error case**: What NOT to do (counter-example)

```python
few_shot = """
Example 1 (Simple case):
Input: "Great product, fast shipping!"
Output: {"sentiment": "positive", "confidence": 0.95}

Example 2 (Edge case - mixed):
Input: "Good quality but expensive"
Output: {"sentiment": "mixed", "positive": 0.6, "negative": 0.4}

Example 3 (What NOT to do):
Wrong: {"sentiment": "yes"}  # Not specific
Correct: {"sentiment": "positive", "confidence": 0.87}

Now apply to: {actual_input}
"""
```

**Complete Guide**: [prompt-patterns.md#few-shot-learning](../docs/prompt-patterns.md#few-shot-learning)

### Phase 4: Apply Constitutional AI

**Add self-correction for safety and quality**:

**Self-Critique Pattern**:
```python
constitutional = """{initial_instruction}

Review your response against these principles:

1. ACCURACY: Verify claims, flag uncertainties
2. SAFETY: Check for harm, bias, ethical issues
3. QUALITY: Clarity, consistency, completeness

Initial Response: [Generate]
Self-Review: [Evaluate against principles]
Final Response: [Refined based on review]
"""
```

**Benefits**:
- Reduces harmful outputs by 40%
- Improves factual accuracy by 25%
- Better handling of edge cases

**Complete Framework**: [prompt-patterns.md#constitutional-ai](../docs/prompt-patterns.md#constitutional-ai)

### Phase 5: Model-Specific Optimization

**Optimize for target LLM**:

**GPT-4 Optimization**:
```python
gpt4_optimized = """##CONTEXT##
{structured_context}

##OBJECTIVE##
{specific_goal}

##INSTRUCTIONS##
1. {numbered_steps}

##OUTPUT FORMAT##
```json
{"structured": "response"}
```
"""
```

**Claude Optimization**:
```python
claude_optimized = """<context>
{background_information}
</context>

<task>{clear_objective}</task>

<thinking>
1. Understanding requirements...
2. Planning approach...
</thinking>

<output_format>{xml_structure}</output_format>
"""
```

**Complete Patterns**: [prompt-patterns.md#model-specific-optimization](../docs/prompt-patterns.md#model-specific-optimization)

### Phase 6: Evaluate and Test

**Test optimized prompt**:

**Testing Protocol**:
1. **Test Cases**: 20 total (10 typical, 5 edge, 3 adversarial, 2 out-of-scope)
2. **Metrics**: Success rate, quality score, efficiency, safety
3. **LLM-as-Judge**: Automated quality evaluation

**LLM-as-Judge Pattern**:
```python
judge_prompt = """Evaluate AI response quality.

## Original Task
{prompt}

## Response
{output}

## Rate 1-10 with justification:
1. TASK COMPLETION: Fully addressed?
2. ACCURACY: Factually correct?
3. REASONING: Logical and structured?
4. FORMAT: Matches requirements?
5. SAFETY: Unbiased and safe?

Overall: []/50
Recommendation: Accept/Revise/Reject
"""
```

**Complete Evaluation**: [prompt-evaluation.md](../docs/prompt-evaluation.md)

## Mode-Specific Execution

### Quick Mode (5-10 minutes)

**Process**:
1. Analyze current prompt (Phase 1)
2. Apply ONE technique (CoT or Few-Shot)
3. Quick validation with 3 test cases

**Output**:
- Optimized prompt
- Top 3 improvements made
- Expected impact estimate

### Standard Mode (15-25 minutes) - DEFAULT

**Process**:
1. Complete analysis (Phase 1)
2. Apply CoT + Few-Shot (Phases 2-3)
3. Add Constitutional AI (Phase 4)
4. Model-specific tuning (Phase 5)
5. Evaluation with 10 test cases (Phase 6)

**Output**:
- Fully optimized prompt
- Detailed optimization report
- Performance comparison
- Usage guidelines

### Comprehensive Mode (30-45 minutes)

**Process**:
1. All phases + Meta-prompt generation
2. Create prompt variants for A/B testing
3. Generate comprehensive test suite (20+ cases)
4. Production deployment strategy
5. Monitoring recommendations

**Output**:
- Multiple prompt variants
- A/B testing plan
- Deployment checklist
- Performance tracking setup

## Success Criteria

✅ Current prompt analyzed with clarity score
✅ CoT reasoning applied where appropriate
✅ Few-shot examples provided for complex tasks
✅ Constitutional AI principles integrated
✅ Model-specific optimization applied
✅ Testing protocol defined
✅ Performance metrics projected
✅ External documentation referenced

## Agent Integration

- **prompt-engineer**: Primary agent for prompt optimization and technique selection
- **ai-engineer**: Triggered for code generation examples and implementation

## Best Practices

1. **Start Simple**: Don't over-engineer prompts, add complexity as needed
2. **Test Early**: Validate with real examples before full deployment
3. **Measure Impact**: Track success rate, quality, cost improvements
4. **Iterate**: Use feedback to refine prompts continuously
5. **Version Control**: Track prompt changes and performance over time
6. **A/B Test**: Compare variants before committing to one
7. **Document**: Explain prompt design decisions for team

## Common Optimization Patterns

### For Reasoning Tasks
✅ **Use CoT**: Step-by-step thinking improves accuracy
✅ **Provide Examples**: Show desired reasoning process
❌ **Skip Validation**: Always ask model to verify reasoning

### For Classification Tasks
✅ **Few-Shot**: Examples of each class
✅ **Structured Output**: JSON/XML for consistency
❌ **Ambiguous Labels**: Define clear class boundaries

### For Generation Tasks
✅ **Clear Constraints**: Length, style, format requirements
✅ **Quality Criteria**: What makes output "good"
❌ **Open-Ended**: Unbounded generation often poor quality

### For RAG Applications
✅ **Citation Requirements**: Force source attribution
✅ **Gap Handling**: Explicit instructions for missing info
❌ **Hallucination Risk**: Don't ask beyond context

## Reference Examples

### Customer Support Optimization
**See**: [prompt-examples.md#customer-support](../docs/prompt-examples.md#customer-support-example)
- Before/After comparison
- Complete optimized prompt
- Expected improvements: +35% resolution rate

### Data Analysis Optimization
**See**: [prompt-examples.md#data-analysis](../docs/prompt-examples.md#data-analysis-example)
- Framework-based approach
- Statistical rigor
- Expected improvements: +40% insight quality

### Code Generation Optimization
**See**: [prompt-examples.md#code-generation](../docs/prompt-examples.md#code-generation-example)
- Security-first approach
- Self-review integration
- Expected improvements: +50% code quality

### Meta-Prompt Generator
**See**: [prompt-examples.md#meta-prompt-generator](../docs/prompt-examples.md#meta-prompt-generator)
- Generates optimized prompts automatically
- Uses decision tree for technique selection
- Includes evaluation criteria

## Evaluation & Testing

### Testing Strategies
- **Unit Tests**: Individual prompt components
- **Integration Tests**: Full prompt in context
- **A/B Tests**: Compare variants
- **Regression Tests**: Ensure no degradation

**Complete Guide**: [prompt-evaluation.md#testing-protocols](../docs/prompt-evaluation.md#testing-protocols)

### Production Monitoring
- **Success Rate**: Track task completion
- **User Satisfaction**: Collect feedback
- **Cost Efficiency**: Monitor token usage
- **Safety Metrics**: Harmful output rate

**Monitoring Setup**: [prompt-evaluation.md#production-monitoring](../docs/prompt-evaluation.md#production-monitoring)

## Output Format

**Optimization Report**:
```yaml
analysis:
  original_assessment:
    clarity: 4/10
    strengths: [simple, concise]
    weaknesses: [ambiguous, no structure]
    token_count: 15
    estimated_success: 60%

improvements_applied:
  - technique: "Chain-of-Thought"
    impact: "+25% reasoning accuracy"
  - technique: "Few-Shot Learning"
    impact: "+30% task adherence"
  - technique: "Constitutional AI"
    impact: "-40% harmful outputs"

performance_projection:
  success_rate: 60% → 88%
  quality_score: 6.5/10 → 8.7/10
  token_efficiency: +15%
  safety_score: 7/10 → 9.5/10

deployment_strategy:
  model: "Claude Sonnet 4.5"
  temperature: 0.7
  max_tokens: 2000
  testing: "A/B test 48h, 5% traffic"
  monitoring: ["success_rate", "latency", "feedback"]

next_steps:
  immediate: ["Test with samples", "Validate safety"]
  short_term: ["A/B test in production", "Collect feedback"]
  long_term: ["Fine-tune based on data", "Develop variants"]
```

## See Also

- **External Docs**:
  - [Prompt Patterns](../docs/prompt-patterns.md) - Complete technique library
  - [Prompt Examples](../docs/prompt-examples.md) - Production-ready examples
  - [Prompt Evaluation](../docs/prompt-evaluation.md) - Testing and monitoring

- **Related Commands**:
  - `/ai-assistant` - Build AI assistants with optimized prompts
  - `/langchain-agent` - Create agents with prompt optimization

---

Remember: **The best prompt consistently produces desired outputs with minimal post-processing while maintaining safety and efficiency.**
