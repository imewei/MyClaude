# The Seven Dimensions of Clarification

Use these seven dimensions as the checklist for Phase 2 of the `thinkfirst` workflow. Do not ask them in a fixed order, and skip any dimension the user has already answered clearly. Ask one question at a time, wait for the response, then decide what to ask next based on what was said.

A simple task might need only 3–4 dimensions fully elaborated. A high-stakes or complex task may need all seven. Stop asking when there is enough clarity to write a solid prompt — do not drag the conversation out for its own sake.

---

## 1. Outcome

Identify what the user is actually trying to accomplish. Not the task ("write a blog post"), but the result ("convince mid-level managers their AI strategy has a blind spot"). If the user cannot say it in one sentence, help them get there.

Look for the verb-plus-goal phrase. "Write X" is a task; "cause the reader to do Y" is an outcome. The outcome shapes every other decision in the prompt.

## 2. Stakes

Surface why the task matters. What happens if it goes well? What happens if it does not? This dimension calibrates how much precision the prompt needs. A throwaway internal doc needs a different level of rigor than a client-facing deliverable.

Stakes also tie directly to Dimension 9 thinking — when stakes are high, consider suggesting Confidence Signaling in the reasoning-strategy phase.

## 3. Success Criteria

Pin down what "done" looks like. If someone handed the user the finished output, what would make them say "yes, that's exactly it"? Push for specifics: format, length, tone, audience, level of detail.

If the user answers in generalities ("it should be good"), offer a concrete menu: "Should it be closer to a one-page memo, a three-slide deck outline, or a full essay?"

## 4. Failure Modes

Surface what would make the user say "no, that's not what I meant" — even if the output looks polished. This is the most overlooked dimension and often the most valuable.

Failure modes become the constraints section of the final prompt. Ask directly: "What would be a disappointing version of this, even if it was well-written?" Common answers: "too generic," "sounds like ChatGPT wrote it," "misses the actual audience," "repeats what everyone already knows."

## 5. Hidden Context

Extract the domain knowledge, institutional norms, and unwritten rules that an outsider would not know. This is the stuff that lives in the user's head and would be invisible to any AI.

Hidden context needs to become explicit in the prompt. Examples: "our CEO hates em-dashes," "this market is heavily regulated and we cannot make efficacy claims," "the audience already read the previous report so do not re-explain the basics."

Ask: "What would someone need to know about your situation that they could not figure out from the brief alone?"

## 6. Components

Break the task into its pieces. What depends on what? What could be done independently? This shapes the prompt's structure and any needed step-by-step breakdown.

A task with clean independent components may fit a simple list. A task with dependencies may need Validation Gates in the reasoning-strategy phase.

## 7. The Hard Part

Every task has one piece that is genuinely difficult and many pieces that are just effort. Identify the hard part. Where are the judgment calls? Where could this go sideways?

This dimension matters most for prompt quality — the hard part is where the final prompt needs the tightest constraints and the most detail. People tend to gloss over it because it is uncomfortable to sit with uncertainty.

If the user cannot name the hard part, help find it. Ask: "What part of this are you least sure about?" or "If this output comes back wrong, where do you predict the failure will be?"

**Special rule for lack of technical knowledge:** If the user's uncertainty comes from not knowing the technical options, do not push them to decide. Step in with 2–3 concrete approaches, sketch the trade-offs, and recommend one. The division of expertise is clear: the user owns the domain, and the prompt-crafter owns the technique.

---

## How to Ask

- **Be conversational, not interrogative.** Avoid "What is your desired outcome?" Prefer "So it sounds like the goal is to [X] — is it more about [A] or [B]?"
- **Build on the user's own words.** Quote back phrases they used. Show that the previous answer was heard.
- **Offer choices when the user is uncertain.** Especially for Dimension 7: when the user does not know the technical details, suggest 2–3 approaches with brief pros/cons and recommend one. "There are a couple of ways to do this: [A] is simpler, [B] gives more control. For what you are describing, [A] fits best — does that sound right?"
- **Know when to stop.** If enough dimensions are clear to write a solid prompt, move on to Phase 3. Do not ask clarifying questions for their own sake.
- **Match the user's language and energy.** If they write in Chinese, respond in Chinese. If they are casual, be casual. If they are precise, be precise.

---

## Signal that Phase 2 is Complete

Phase 2 is complete when all of the following are true:

1. The outcome can be stated in one sentence.
2. The success criteria are specific enough that a draft could be evaluated against them.
3. At least one failure mode has been identified.
4. The hard part has been named (or explicitly declared absent).
5. Any domain-specific hidden context has been captured.

At that point, summarize what was learned, announce readiness to draft, and proceed to Phase 3.
