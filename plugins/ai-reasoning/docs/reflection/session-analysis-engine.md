# Session Analysis Engine

**Version**: 1.0.3
**Purpose**: AI reasoning pattern analysis and conversation effectiveness assessment

---

## ConversationReflectionEngine

Analyzes AI reasoning patterns, conversation effectiveness, and problem-solving approaches.

### Analysis Dimensions

**1. Reasoning Patterns (40%)**
- Logic chain coherence
- Assumption tracking and validation
- Evidence utilization and quality
- Conclusion validity and support

**2. Problem-Solving (30%)**
- Approach selection appropriateness
- Solution creativity and novelty
- Alternative exploration depth
- Trade-off analysis quality

**3. Communication (20%)**
- Clarity and precision
- Technical depth appropriateness
- Stakeholder adaptation
- Documentation quality

**4. Effectiveness (10%)**
- Goal achievement
- Efficiency and speed
- User satisfaction
- Learning and improvement

### Scoring Framework

- **9-10**: Exceptional - Model example
- **7-8**: Strong - Minor improvements possible
- **5-6**: Adequate - Notable gaps exist
- **3-4**: Weak - Significant improvements needed
- **1-2**: Poor - Major restructuring required

### Example Assessment

```yaml
session_reflection:
  overall_score: 8.2/10

  reasoning_patterns:
    score: 8.5
    strengths:
      - Strong logical coherence
      - Explicit assumption tracking
      - Evidence-based conclusions
    weaknesses:
      - Could explore more alternatives

  problem_solving:
    score: 8.0
    approach: "Systems Thinking"
    creativity: "High"
    completeness: "Good"

  communication:
    score: 8.0
    clarity: 9
    technical_depth: 8
    adaptation: 7

  effectiveness:
    score: 8.5
    goal_achieved: true
    efficiency: "High"
    user_satisfaction: 9/10

recommendations:
  - Explore 1-2 additional alternative approaches
  - Add more concrete examples in explanations
  - Consider stakeholder technical level earlier
```

---

*Part of the ai-reasoning plugin documentation*
