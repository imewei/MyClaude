# Prompt Structure and Best Practices

Use this reference in Phase 3 of the `thinkfirst` workflow, once the Seven Dimensions are clear enough to draft. A professional prompt does not need every section below — adapt the structure to fit the task. Rigid templates produce rigid outputs.

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
Boundaries, things to avoid (framed as positive alternatives), edge case handling.
Include failure modes the user identified — these become explicit constraints.
```

---

## Best Practices to Apply

Follow these prompt engineering principles when writing the draft:

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

## Mapping Seven Dimensions to Prompt Sections

The clarification phase produces content that maps cleanly into the prompt sections:

| Seven-Dimension answer      | Prompt section it feeds              |
|-----------------------------|--------------------------------------|
| Outcome                     | Task (opening sentence)              |
| Stakes                      | Context / Background (motivation)    |
| Success Criteria            | Output Format                        |
| Failure Modes               | Constraints / Guardrails             |
| Hidden Context              | Context / Background                 |
| Components                  | Task (numbered steps or sub-tasks)   |
| The Hard Part               | Constraints + Examples (where to be tightest) |

Use the table as a checklist while drafting — every clarified dimension should appear somewhere in the prompt.

---

## When to Skip Sections

Match the complexity of the prompt to the complexity of the task:

- **Simple one-off task** (e.g., "rewrite this paragraph more formally") — `Task` + brief `Output Format`. Skip the rest.
- **Repeated workflow** — add `Role`, `Input Specification` with `{{placeholders}}`, and 1–2 `Examples`.
- **High-stakes deliverable** — include all sections, especially `Constraints` and `Examples`, and consider embedding a reasoning strategy (see `reasoning-strategies.md`).

Do not force sections that the task does not need. A prompt for a simple task should be simple.

---

## Final-Draft Checklist

Before presenting the prompt to the user in Phase 4, confirm:

1. The outcome is stated explicitly in the Task section.
2. At least one failure mode from Phase 2 appears in Constraints.
3. Every piece of hidden context is captured in Context / Background.
4. The output format specifies structure, length, tone, and audience.
5. If the user accepted a reasoning strategy in Phase 2, it is woven into the draft (see `reasoning-strategies.md`).
6. The prompt is self-contained — it would work when pasted into any AI tool without additional context.
7. The prompt matches the language the user was writing in.
