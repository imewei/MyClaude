# Prompt Structure, Best Practices, and Optimization Techniques

Use this reference when drafting (Craft Mode Phase 3) or rewriting (Optimize Mode Step 2). A professional prompt does not need every section — adapt the structure to fit the task. Rigid templates produce rigid outputs.

---

## Standard Prompt Sections

```
[Role]
Who the AI should be, what expertise it brings. One or two sentences.

[Context / Background]
Information the AI needs to understand the task. Wrap in descriptive XML tags
like <context>, <background>, or domain-specific tags.
Include the "why" — motivation helps the AI generalize beyond literal instructions.

[Task]
Clear, specific description of what to do.
Use numbered steps for sequential work.
Use bullet points for parallel requirements.
Be explicit: "Create X" not "Can you help with X?"

[Input Specification]
If the prompt will receive variable input, define placeholders like {{VARIABLE_NAME}}
and describe what goes in each. Use XML tags to wrap input sections.

[Output Format]
What the output should look like. Be specific about structure, length, tone, audience.
Tell the AI what to do ("write in flowing prose") rather than what not to do
("don't use bullet points").

[Examples]
1–3 examples wrapped in <example> tags showing input → output pairs.
Make them diverse enough to cover common cases and edge cases.
Examples are one of the most powerful tools for steering AI output.

[Constraints / Guardrails]
Boundaries and edge case handling.
Include failure modes the user identified — these become explicit constraints.
Frame positively: "use accessible language" not "avoid jargon."
```

---

## When to Skip Sections

| Task type | Sections needed |
|---|---|
| Simple one-off ("rewrite this paragraph formally") | Task + Output Format only |
| Repeated workflow | Add Role + Input Specification with `{{placeholders}}` + 1–2 Examples |
| High-stakes deliverable | All sections + reasoning strategy (see `reasoning-strategies.md`) |

Do not force sections the task does not need.

---

## Mapping Seven Dimensions to Prompt Sections

Every clarified dimension should appear somewhere in the prompt:

| Seven-Dimension answer | Prompt section it feeds |
|---|---|
| Outcome | Task (opening sentence) |
| Stakes | Context / Background (motivation) |
| Success Criteria | Output Format |
| Failure Modes | Constraints / Guardrails |
| Hidden Context | Context / Background |
| Components | Task (numbered steps or sub-tasks) |
| The Hard Part | Constraints + Examples (where to be tightest) |

---

## Optimization Techniques

Apply these when rewriting an existing prompt. Use judgment — not every prompt needs every technique.

### 1. Clarity & Specificity
The single highest-impact improvement. Replace vague verbs with action verbs; add counts, purpose, scope, and constraints.

**Before:** "Help me with a presentation."
**After:** "Create a 10-slide outline for our quarterly sales meeting covering Q2 performance, top-selling products, and Q3 targets. Provide 3–4 key points per slide."

**Checklist:**
- Replace "help me with" → "write / analyze / outline / compare"
- Specify quantities (number of slides, word count, number of ideas)
- State the purpose and end use
- Add constraints (audience, format, length)

---

### 2. Output Structure Definition
Tell the AI exactly how to format the response.

**Before:** "Analyze our sales data."
**After:** "Present the analysis as: (1) Executive Summary (2–3 sentences), (2) Key Metrics table with quarterly totals and top performer, (3) three notable trends with brief explanations, (4) three data-driven recommendations with rationale."

**Patterns:** numbered sections, tables for comparisons, explicit word/section limits.

---

### 3. Audience Definition
Different audiences require different vocabulary, depth, and tone.

**Before:** "Write about cybersecurity."
**After:** "Write a 1000-word blog post about cybersecurity for small business owners who are not tech-savvy. Use plain language, practical actionable tips, and a slightly humorous tone."

**Ask:** Technical or non-technical? Senior or junior? Internal or external? Consumer or professional?

---

### 4. Tone & Style Guidance
Specify the voice. Reference a style guide if one exists.

**Before:** "Write a product description."
**After:** "Write a ~200-word product description in a professional but engaging tone. Brand voice: friendly, innovative, health-conscious. Highlight ergonomic features, health benefits, sustainable materials, and close with a call to action."

**Tone options:** formal / casual / friendly / technical / humorous / empathetic

---

### 5. Role / Persona Assignment
Assigning an expert role unlocks more nuanced, domain-aware responses.

**Before:** "Help me prepare for a negotiation."
**After:** "You are a fabric supplier for my backpack manufacturing company. As the supplier, provide three objections to a 10% price reduction request, a counterargument for each, and two alternative proposals. Then switch roles and advise me as the buyer on how to approach this negotiation."

**Strong patterns:** "Act as a senior [role] with 15 years of experience in [domain]."

---

### 6. Step-by-Step Thinking (Chain of Thought)
For complex or analytical tasks, ask the AI to reason through the problem.

**Before:** "How can I improve team productivity?"
**After:** "Think through this step-by-step: (1) identify current productivity blockers, (2) suggest solutions for each, (3) identify implementation challenges, (4) suggest measurement methods. Explain your reasoning at each step and summarize at the end."

**Trigger phrases:** "think step-by-step," "walk me through your reasoning," "analyze each factor in order."

---

### 7. Examples (Few-Shot)
Show the desired style, tone, or format. Works especially well for emails, creative writing, and structured content.

**Before:** "Write a professional email."
**After:** "Here is a similar email I sent before: [example]. Draft a new email in the same tone for a situation where we are delayed by one month due to supply chain issues."

---

### 8. Document Q&A — Specificity + Citations
When asking the AI to analyze a document, specify the focus and request citations.

**Before:** "Summarize this report."
**After:** "From the attached 'Tech Industry Trends 2023' report, provide a 2-paragraph summary focused on AI and ML trends. Then answer these three questions: [questions]. Cite the specific section or page number for each answer."

---

### 9. Uncertainty Acknowledgment
For fact-heavy tasks, explicitly give the AI permission to say it does not know. Reduces confident hallucinations.

**Add to any research or factual prompt:** "If you are unsure about specific figures or claims, say so rather than guessing."

---

### 10. Iterative Structure (Constrained Draft then Expand)
When the user is not sure what they want and iteration cost is high, ask for a short version first, confirm direction, then expand.

**Pattern:** "First write a one-paragraph version. Once confirmed, expand to full length."

---

## Best Practices to Apply

Apply when drafting or rewriting:

- **Be clear and direct.** Specificity beats vagueness every time. If the user wants thorough, detailed output, say so explicitly rather than hoping the AI infers it.
- **Explain the why.** When adding a constraint or instruction, briefly explain the reasoning. "Write for a general audience because this will be published on our public blog" is better than just "Write for a general audience." Explaining the why helps the AI generalize correctly.
- **Use XML tags for structure.** Tags like `<instructions>`, `<context>`, `<example>`, `<input>` help the AI parse complex prompts without ambiguity. Use consistent, descriptive tag names.
- **Show, do not just tell.** A few well-crafted examples (3–5) are more reliable than paragraphs of explanation. Make examples relevant, diverse, and structured.
- **Put long data at the top.** If the prompt involves large inputs, place them above the instructions. Queries and instructions at the end improve response quality.
- **Frame positively.** Instead of "don't use jargon," write "use language accessible to a general audience." Tell the AI what to do, not what to avoid.
- **Match prompt style to desired output.** For prose output, write the prompt in prose. For structured data output, structure the prompt accordingly.
- **Give a role when it helps.** A single sentence setting the AI's persona focuses its behavior and tone.
- **Use step-by-step instructions** for sequential tasks. Numbered lists help when order and completeness matter.

---

## Final-Draft Checklist

Before presenting the prompt (both modes), confirm:

1. The outcome is stated explicitly in the Task section.
2. At least one failure mode from clarification appears in Constraints.
3. All hidden context is captured in Context / Background.
4. Output format specifies structure, length, tone, and audience.
5. If a reasoning strategy was accepted, it is woven in (see `reasoning-strategies.md`).
6. The prompt is self-contained — works when pasted into any AI tool without extra context.
7. The prompt matches the language the user was writing in.
8. Complexity of prompt matches complexity of task — no over-engineering.
