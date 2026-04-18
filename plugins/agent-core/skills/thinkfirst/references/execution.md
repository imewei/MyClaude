# Execution — Running the Approved Prompt

After the final prompt is delivered (in either Craft or Optimize mode), offer to run it here. Seeing what the prompt actually produces — without opening a fresh session — is the core value-add over "hand it off and hope."

## Offer the Three Paths

Close the delivery message with a short offer in prose. Match the user's language and adapt labels to context. Three clear paths:

> *Ready when you are — want me to run it?*
>
> - **Run it now** — I will execute the prompt as a fresh turn and show the output.
> - **Tweak it first** — we keep iterating on the prompt before running it.
> - **I'll take it from here** — you will use the prompt elsewhere; we are done.

Wait for an explicit choice. Silence is not consent. Do not execute until the user affirmatively says "run it" or equivalent ("yes", "go ahead", "run it now", Chinese-language equivalents, etc.).

## Three Pre-Execution Checks

Before executing, confirm the prompt is actually runnable.

### 1. Placeholders

If the prompt contains `[TOPIC]`, `{{INPUT}}`, `<document>`, or explicitly expects variable input ("Given the following text: ..."), it is not yet runnable. Ask for the missing values one at a time — do not invent them. Common patterns that signal a placeholder:

- Bracketed variable names in all caps or title case inside the prompt body.
- A "Given the following..." or "Analyze this..." opener with no content attached.
- An Input section that describes what input is expected but omits the actual input.

### 2. Meta / System Prompts

If the prompt defines a persona or system role ("You are a code reviewer...", "Act as a senior editor...") without a concrete query, running it alone produces nothing useful — the model has a character but no task. Ask what the user wants to test it with.

Typical shape: Role + Constraints + Output Format, but no actual question or request to act on. Offer to pair it with a sample input.

### 3. Heavy Prompts

If execution will take a while, require web search, or invoke tools, flag it briefly first:

> *This will search the web and take a couple of minutes — still proceed?*

Flags for "heavy":
- Requires external lookups (web search, API calls, file reads the skill cannot yet verify).
- Asks for long output (e.g., "write a 5000-word draft").
- Involves multi-step reasoning that will visibly consume context.

Do not add friction for normal cases. A two-paragraph creative rewrite does not need a confirmation gate.

## Executing

Treat the approved prompt as a fresh user turn. Respond to it on its own terms — do not carry over voice, tone, or context from the crafting conversation. The output should look exactly like what a clean session would produce.

Visually separate the execution so the prompt above remains a durable artifact:

```
---

**Running the prompt:**

[output]
```

**Do not restate the prompt inside the execution block.** It already exists above as the canonical artifact; repeating it adds noise and fragments the record.

**Do not break character.** If the prompt sets a persona, play the persona in the output. Meta-commentary about the prompt ("As the code reviewer you asked me to be...") leaks the crafting context and undermines the test.

## After Execution

Offer three paths in conversational prose. No structured menu is needed — by this point the right direction depends on what the user just saw.

- **Tweak the prompt** — if the output reveals a weakness in the prompt itself → return to Craft Phase 4 or Optimize Step 2.
- **Iterate on this output** — if the output is close but needs polish → stay in execution context and refine the output directly, without re-running the prompt.
- **We're done** — satisfied → close out.

When the user asks for "changes" without specifying what, **disambiguate before acting**:

> *Want me to change the prompt itself, or just the output we just generated? Prompt changes mean we re-run; output changes stay with this one.*

This rule prevents the common failure where prompt-level edits silently turn into output-level edits (or vice versa), leaving the user with a prompt that no longer matches the output.

## Language Matching

Everything in this phase — offers, check questions, flags, disambiguation — follows the user's language. If the conversation has been in Chinese, so are the three path labels, the pre-execution questions, and the after-execution offer. Do not switch to English just because this reference is in English.

## What Not to Do

- Do not execute silently after delivering the prompt.
- Do not assume "thanks" or "great" means "run it."
- Do not rewrite the prompt mid-execution because the output looks off — stop, surface the issue, and ask whether to tweak the prompt or iterate on output.
- Do not bury the prompt in a chat bubble above a long execution — keep the prompt visually reachable.
- Do not invent placeholder values. If the prompt needs an input and the user has not provided one, ask.
