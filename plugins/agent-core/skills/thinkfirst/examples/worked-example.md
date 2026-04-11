# Worked Example: Brain Dump → Interview → Final Prompt

This is a short end-to-end transcript showing the `thinkfirst` workflow in action. Use it as a model for pacing, question style, and how the Seven Dimensions surface in real conversation. The quoted dialogue uses second person because it is user-facing speech — that is the only place second person belongs in the skill.

---

## The User's Brain Dump

> "Okay so I want to write something to convince our execs that the AI strategy is too narrow. They keep talking about chatbots and nothing else. I have like 15 years in the industry, I know this is a mistake, but when I've tried to bring it up in meetings it hasn't landed. I don't want to sound like I'm whining. Can you help me write a prompt for this?"

---

## Phase 1: Summarize and Open

**Assistant reply:**

> "Got it — so the goal is a written piece that helps your execs see past the chatbot-only framing of AI strategy, and it has to land without sounding like a complaint. Before I draft anything: when you say 'helps them see past it,' are you after something that shifts their position in one read, or something that opens the door to a follow-up conversation? Those two goals need pretty different tones."

**What this does:** Summarizes back (Dimension 1: Outcome) and asks a forced choice that surfaces the real goal. Does not produce a prompt yet.

**User:**

> "Honestly, opens the door. I don't expect one memo to flip them. I want them to come back and ask me to elaborate."

---

## Phase 2 (continued): Cycle Through Dimensions

**Assistant:**

> "That helps. The 'come back and ask' goal means the piece needs to plant a specific question in their head — not make a full case. What is the one question you most want them walking away asking? For example: 'Are we betting everything on the wrong channel?' or 'What are our competitors doing with AI beyond chat?'"

**User:**

> "The second one. Competitors. They respond to competitive framing more than anything else."

**Assistant:**

> "Good — so the implicit thesis is 'our AI strategy is narrower than our competitors', and the test for success is whether the reader walks away wanting a competitive benchmark. One more: what would make this piece land wrong even if it was well-written? What should it *not* sound like?"

> (This is Dimension 4: Failure Modes — deliberately framed to surface the user's real fears.)

**User:**

> "It should not sound like I'm trashing the current strategy. I need the people who championed chatbots to not feel attacked. Also it should not read like a generic AI think-piece — they get those in their inbox every day."

**Assistant:**

> "Noted — no attacking the existing effort, and no generic think-piece tone. Last question, then I think I have enough: what is the hard part of this piece, the part where it could go sideways? I am guessing it is the balance between 'this is narrow' and 'I am not calling you wrong' — but tell me if it is something else."

**User:**

> "Yeah that is exactly it. I need to land the narrowness without it feeling like an attack."

---

## Reasoning-Strategy Suggestion

**Assistant:**

> "One thing I want to offer before drafting: because the hard part is tone — specifically avoiding an attacking feel — I would suggest an adversarial self-review strategy. The AI would draft the memo, then re-read it as if it were one of the chatbot champions and revise any line that reads as criticism, then give you only the revised version. Want me to build that in? It adds about three lines to the prompt."

**User:**

> "Yes, do that."

---

## Phase 3: The Final Prompt

```
[Role]
Act as a senior industry strategist with 15 years of experience writing
persuasive internal memos for executive audiences.

[Context]
<context>
The reader of this memo is an executive team that is currently invested in
a chatbot-centric AI strategy. The author believes this strategy is too
narrow compared to what competitors are doing. The goal is not to change
their minds in one read — it is to plant a competitive-benchmarking question
in their head so they come back to ask for a follow-up analysis.

The author is trusted but has struggled to surface this concern in meetings
without it being heard as criticism of the existing effort.
</context>

[Task]
Write a one-page internal memo addressed to the executive team. The memo
should:

1. Open with a single, concrete competitive observation (not a thesis statement).
2. Build toward an implicit question: "Is our AI strategy narrower than
   our competitors'?" — without stating that question directly.
3. Close with an offer to provide a short competitive benchmark if
   the executives would find it useful.

[Output Format]
- One page, approximately 400–500 words.
- Prose, not bullet points. Tone is thoughtful and collegial.
- Audience: senior executives who championed the current strategy.
- Voice: informed peer, not critic.

[Constraints]
- Do not criticize the current chatbot strategy directly or by implication.
- Do not use generic AI-think-piece phrasing ("AI is transforming X,"
  "the next frontier," etc.).
- Do not present a full counter-thesis. The memo's job is to invite
  a conversation, not to win one.
- Frame competitor activity as observation, not judgment.

[Self-Review Step]
After producing the draft, re-read it from the perspective of an executive
who personally championed the chatbot strategy. Identify any line that could
be read as criticism of that person's decision. Revise each such line to
remove the implied criticism while preserving the substance. Present only
the revised version of the memo.
```

---

## What the Example Demonstrates

- **One question per turn.** No multi-part interrogation.
- **Building on the user's own words.** "Come back and ask," "competitive framing," and "land the narrowness" all get quoted back.
- **Dimensions surface naturally.** Outcome → Success Criteria → Failure Modes → Hard Part. Dimensions 2, 5, 6 were skipped because the user's context already covered them.
- **Reasoning strategy proposed only after the hard part was named.** The adversarial self-review was specifically justified by the tone risk, not tacked on as a template.
- **The final prompt maps cleanly to the clarifications.** Every failure mode appears in Constraints. The hard part drives the Self-Review Step.
