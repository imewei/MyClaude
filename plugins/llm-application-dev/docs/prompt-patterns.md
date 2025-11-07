# Prompt Engineering Patterns

Advanced prompting techniques including Chain-of-Thought, Few-Shot Learning, Constitutional AI, and model-specific optimizations.

## Chain-of-Thought Patterns

### Standard CoT

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

### Zero-Shot CoT

```python
enhanced = original + "\n\nLet's approach this step-by-step, breaking down the problem into smaller components."
```

### Tree-of-Thoughts

```python
tot_prompt = """Explore multiple solution paths:

Problem: {problem}

Approach A: [Path 1]
Approach B: [Path 2]  
Approach C: [Path 3]

Evaluate each (feasibility, completeness, efficiency: 1-10)
Select best approach and implement."""
```

## Few-Shot Learning

### Strategic Example Selection

```python
few_shot = """
Example 1 (Simple case):
Input: {simple_input}
Output: {simple_output}

Example 2 (Edge case):
Input: {complex_input}
Output: {complex_output}

Example 3 (Error case - what NOT to do):
Wrong: {wrong_approach}
Correct: {correct_output}

Now apply to: {actual_input}"""
```

## Constitutional AI

### Self-Critique Loop

```python
constitutional = """{initial_instruction}

Review your response against these principles:

1. ACCURACY: Verify claims, flag uncertainties
2. SAFETY: Check for harm, bias, ethical issues
3. QUALITY: Clarity, consistency, completeness

Initial Response: [Generate]
Self-Review: [Evaluate]
Final Response: [Refined]"""
```

## Model-Specific Optimization

### GPT-4 Optimization

```python
gpt4_optimized = """##CONTEXT##
{structured_context}

##OBJECTIVE##
{specific_goal}

##INSTRUCTIONS##
1. {numbered_steps}
2. {clear_actions}

##OUTPUT FORMAT##
```json
{"structured": "response"}
```

##EXAMPLES##
{few_shot_examples}"""
```

### Claude Optimization

```python
claude_optimized = """<context>
{background_information}
</context>

<task>
{clear_objective}
</task>

<thinking>
1. Understanding requirements...
2. Identifying components...
3. Planning approach...
</thinking>

<output_format>
{xml_structured_response}
</output_format>"""
```

### Gemini Optimization

```python
gemini_optimized = """**System Context:** {background}
**Primary Objective:** {goal}

**Process:**
1. {action} {target}
2. {measurement} {criteria}

**Output Structure:**
- Format: {type}
- Length: {tokens}
- Style: {tone}

**Quality Constraints:**
- Factual accuracy with citations
- No speculation without disclaimers"""
```

## RAG Integration

### RAG-Optimized Prompt

```python
rag_prompt = """## Context Documents
{retrieved_documents}

## Query
{user_question}

## Integration Instructions

1. RELEVANCE: Identify relevant docs, note confidence
2. SYNTHESIS: Combine info, cite sources [Source N]
3. COVERAGE: Address all aspects, state gaps
4. RESPONSE: Comprehensive answer with citations

Example: "Based on [Source 1], {answer}. [Source 3] corroborates: {detail}."""
```

## Evaluation Framework

### Testing Protocol

```python
evaluation = """## Test Cases (20 total)
- Typical cases: 10
- Edge cases: 5
- Adversarial: 3
- Out-of-scope: 2

## Metrics
1. Success Rate: {X/20}
2. Quality (0-100): Accuracy, Completeness, Coherence
3. Efficiency: Tokens, time, cost
4. Safety: Harmful outputs, hallucinations, bias"""
```

### LLM-as-Judge

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
Recommendation: Accept/Revise/Reject"""
```

---

**See Also**:
- [Prompt Examples](./prompt-examples.md) - Reference implementations
- [Prompt Evaluation](./prompt-evaluation.md) - Testing strategies
- Command: `/prompt-optimize`
