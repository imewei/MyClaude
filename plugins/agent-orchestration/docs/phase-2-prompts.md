# Phase 2: Prompt Engineering

## Overview

Phase 2 applies advanced prompt engineering techniques to address issues identified in Phase 1. This is where you transform insights into improved agent behavior.

**Duration**: 2-4 hours
**Prerequisites**: Completed Phase 1 baseline analysis
**Outputs**: Improved agent prompt, version increment, changelog

---

## Quick Start

```bash
# Execute Phase 2 for specific focus area
/improve-agent <agent-name> --phase=2 --focus=tool-selection

# Execute Phase 2 as part of full optimization
/improve-agent <agent-name> --mode=optimize
```

---

## Technique 1: Chain-of-Thought (CoT) Enhancement

### When to Use
- Complex multi-step reasoning tasks
- High error rate on reasoning-heavy tasks
- Need for debugging/auditability

### Implementation

**Before** (implicit reasoning):
```markdown
You are a customer support agent. Help users with their inquiries efficiently.
```

**After** (explicit CoT):
```markdown
You are a customer support agent. For each user inquiry, think step-by-step:

1. **Understand**: What is the user asking? Extract key entities (order ID, product name, etc.)
2. **Determine action**: Which tool or information source is needed?
3. **Verify**: Do I have all required information? If not, ask.
4. **Execute**: Call appropriate tool or provide information.
5. **Validate**: Does this answer fully address the user's concern?
6. **Respond**: Format response with empathy and clarity.

Example:
User: "I haven't received my order yet"

Your thinking:
1. Understand: User wants order status update
2. Determine: Need get_order_status() tool
3. Verify: Missing order ID → must ask
4. Execute: (will call tool after getting order ID)
5. Validate: (check result before responding)
6. Respond: Ask for order ID empathetically

Your response: "I'd be happy to check on your order! Could you provide your
order number? You can find it in your confirmation email starting with 'ORD-'."
```

**Expected improvement**: +10-15% success rate on complex tasks

---

## Technique 2: Few-Shot Example Optimization

### Selection Strategy

**Coverage analysis**:
```python
def analyze_task_distribution(agent_logs):
    """Identify which tasks need examples most"""
    task_types = {}

    for task in agent_logs:
        task_type = task['type']
        success = task['status'] == 'success'

        if task_type not in task_types:
            task_types[task_type] = {'total': 0, 'success': 0}

        task_types[task_type]['total'] += 1
        if success:
            task_types[task_type]['success'] += 1

    # Calculate success rate
    for task_type in task_types:
        stats = task_types[task_type]
        stats['success_rate'] = stats['success'] / stats['total']

    # Prioritize: high frequency + low success rate
    prioritized = sorted(
        task_types.items(),
        key=lambda x: x[1]['total'] * (1 - x[1]['success_rate']),
        reverse=True
    )

    return prioritized
```

### Example Template

**High-quality example structure**:
```markdown
## Example 1: Simple Pricing Query (Common - 87% success)

**User**: "How much does the Pro plan cost?"

**Agent Reasoning**:
- Task type: Pricing inquiry
- Information needed: Pro plan pricing
- Tool: get_pricing_info(plan="Pro")
- User intent: Likely considering upgrade

**Agent Response**:
"The Pro plan costs $29/month (billed monthly) or $290/year (17% savings).

It includes:
- Unlimited projects
- Priority support (response within 2 hours)
- Advanced analytics dashboard
- API access

Would you like help upgrading your account?"

**Why this works**:
✅ Directly answers the question
✅ Provides both monthly and annual options
✅ Lists key features (value proposition)
✅ Proactive offer to help next step
✅ Friendly, not pushy tone
```

### Negative Examples

**Also show what NOT to do**:
```markdown
## ❌ Bad Example: Vague Pricing Response

**User**: "How much does the Pro plan cost?"

**Agent Response**: "Our Pro plan has great features! Let me know if you have questions."

**Why this fails**:
- Doesn't answer the question (no price given)
- Forces user to repeat/clarify
- Wastes user's time
- Damages trust

**Correct approach**: Directly provide pricing first, then elaborate if asked.
```

---

## Technique 3: Role Definition Refinement

### Structure

```markdown
# Role: Customer Support Agent v1.1.0

## Core Identity
You are an empathetic, knowledgeable customer support specialist focused on
resolving user issues efficiently while building trust and satisfaction.

## Expertise Domains
1. **Product knowledge** (complete feature set, all pricing tiers)
2. **Order processing** (tracking, shipping, returns, refunds)
3. **Account management** (login, billing, subscriptions)
4. **Technical troubleshooting** (common issues, workarounds)

## Communication Style
- **Tone**: Friendly but professional, empathetic
- **Conciseness**: Answer directly, then elaborate if needed
- **Proactivity**: Anticipate follow-up questions, offer related help
- **Clarity**: Use simple language, avoid jargon unless user does first

## Decision-Making Framework

### When to use each tool:
- `get_order_status(order_id)`: User asks about "my order", "shipping", "delivery"
- `get_pricing_info(plan)`: User asks about "cost", "price", "how much"
- `search_docs(query)`: User asks technical "how to" questions
- `create_support_ticket(details)`: Complex issues beyond your capabilities

### When to escalate to human:
- Refund requests >$100 (policy: manager approval required)
- Account security concerns (potential fraud)
- Legal or compliance questions
- Customer is clearly frustrated after 2+ exchanges

## Constraints (MUST follow)
1. **Never promise**: Features not yet released, delivery dates not in system
2. **Never share**: Other users' information, internal pricing/costs
3. **Always verify**: Identity before discussing account details
4. **Always document**: Issues in support ticket system

## Success Criteria
- User's question fully answered (target: >80% first response)
- No safety/constraint violations (target: 100%)
- User satisfaction (target: >8/10)
- Response time (target: <30 seconds p95)
```

---

## Technique 4: Constitutional AI Integration

### Self-Critique Loop

**Add to prompt**:
```markdown
## Before Responding: Self-Critique

Run through this checklist:

1. **Accuracy**: Have I verified all facts? Any uncertain information?
2. **Completeness**: Did I fully address the user's request?
3. **Constraints**: Did I violate any rules (promising unavailable features, etc.)?
4. **Tone**: Is my response empathetic and professional?
5. **Format**: Does the structure match requirements?

If any check fails, revise before responding.
```

**Implementation**:
```markdown
**Example with self-critique**:

User: "Can I get a refund for my Pro subscription?"

Initial thought:
"Yes, we offer full refunds within 30 days. I'll process that for you now."

Self-critique:
- Constraint check: ❌ I'm not authorized to process refunds >$100 (Pro is $290/year)
- Accuracy: ⚠️ Need to verify refund eligibility (within 30 days? subscription vs one-time?)

Revised response:
"I'd be happy to help with your refund request. To ensure I provide accurate
information, could you tell me:
1. When did you subscribe to the Pro plan?
2. Is this a monthly ($29) or annual ($290) subscription?

Our standard refund policy is 30 days for annual plans. Depending on your
subscription date, I may need to connect you with our billing team who can
process this for you."
```

---

## Technique 5: Output Format Tuning

### Template Strategy

**For structured tasks, provide explicit templates**:
```markdown
## Order Status Response Template

Use this structure for order status inquiries:

**Order #{order_id}**
**Status**: {status_label}
**Ordered**: {order_date}
**Expected Delivery**: {delivery_date}

**Current Location**: {tracking_location}

**Next Steps**:
- {action_1}
- {action_2}

[Track Your Package]({tracking_url})

{optional_note_based_on_status}
```

**Example filled template**:
```markdown
**Order #ORD-2847593**
**Status**: In Transit ✓
**Ordered**: May 28, 2025
**Expected Delivery**: June 2, 2025

**Current Location**: Chicago, IL Distribution Center

**Next Steps**:
- Package will arrive at local facility tomorrow
- Out for delivery on June 2nd
- You'll receive an SMS when driver is 10 stops away

[Track Your Package](https://track.example.com/ORD-2847593)

Note: We're experiencing slight delays in the Chicago area due to weather.
Your package may arrive 1 day later than originally estimated.
```

---

## Implementation Workflow

### Step-by-Step Process

1. **Load current prompt**: Read `.agents/<agent>-v<current>.md`

2. **Apply improvements**:
   ```bash
   # Based on Phase 1 findings:
   # Issue 1: Tool selection (72% efficiency)
   # Issue 2: Complex pricing queries (15% failure)
   # Issue 3: Response verbosity (47 complaints)

   # Add: Tool decision tree (Technique 3)
   # Add: 3 pricing few-shot examples (Technique 2)
   # Add: Conciseness guideline (Technique 5)
   ```

3. **Version increment**:
   ```bash
   # Current: v1.0.0
   # Changes: Minor improvements (backward compatible)
   # New version: v1.1.0
   ```

4. **Create changelog**:
   ```markdown
   # Changelog v1.0.0 → v1.1.0

   ## Added
   - Tool decision tree for better tool selection
   - 3 few-shot examples for complex pricing queries
   - Conciseness guideline with template

   ## Expected Impact
   - Tool efficiency: 72% → 87% (+15pp)
   - Pricing query success: 85% → 95% (+10pp)
   - User satisfaction: 8.2 → 8.5 (+0.3)

   ## Testing Plan
   - A/B test with 50% traffic split
   - Minimum 100 tasks per variant
   - 7-day evaluation period
   ```

5. **Save new version**: `.agents/<agent>-v1.1.0.md`

6. **Git commit**:
   ```bash
   git add .agents/customer-support-v1.1.0.md
   git commit -m "feat(customer-support): improve tool selection and pricing queries

   - Add tool decision tree
   - Add 3 pricing few-shot examples
   - Add conciseness guideline

   Expected impact:
   - Tool efficiency: +15pp
   - Pricing success: +10pp

   Version: 1.0.0 → 1.1.0"
   ```

---

## Validation

Before Phase 3, verify:

- [ ] All Phase 1 issues addressed (at least top 3)
- [ ] New version saved with semantic versioning
- [ ] Changelog documents all changes and expected impact
- [ ] Prompt tested manually with 5-10 examples
- [ ] No constraints violated in new prompt
- [ ] Stakeholders reviewed changes
- [ ] Git committed with descriptive message

---

## Common Mistakes

1. **Over-engineering**: Adding complexity that doesn't address real issues
2. **Insufficient examples**: 1-2 examples rarely enough (aim for 3-5)
3. **Vague instructions**: "Be helpful" vs "Provide 2-3 actionable next steps"
4. **Ignoring constraints**: Improvements that violate safety rules
5. **No measurable goals**: "Improve responses" vs "Reduce corrections by 25%"

---

## Prompt Engineering Checklist

**Clarity**:
- [ ] Role clearly defined (one sentence purpose)
- [ ] Examples show, not just tell
- [ ] Instructions unambiguous
- [ ] No contradictory directives

**Completeness**:
- [ ] Covers all common scenarios (≥80% of tasks)
- [ ] Includes edge case handling
- [ ] Provides error recovery patterns
- [ ] Specifies output format

**Effectiveness**:
- [ ] Addresses Phase 1 findings
- [ ] Includes self-critique mechanism
- [ ] Provides decision-making framework
- [ ] Shows both good and bad examples

---

## Tools & Resources

- **Prompt testing**: Create test harness with 20-30 examples
- **Version control**: Git for prompts, semantic versioning
- **Collaboration**: Prompt review with team before deployment
- **Templates**: Reusable prompt components library

---

## Next Steps

Once prompt improvements complete:
1. Proceed to [Phase 3: Testing & Validation](phase-3-testing.md)
2. Run A/B test to measure impact
3. Iterate based on test results

---

**See also**:
- [Prompt Engineering Techniques](prompt-techniques.md) - Deep dive
- [Phase 1: Performance Analysis](phase-1-analysis.md) - Previous phase
- [Phase 3: Testing & Validation](phase-3-testing.md) - Next phase
