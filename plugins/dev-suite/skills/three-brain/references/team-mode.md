# Team mode — roles, loop, and shutdown

Loaded when running the persistent team: who each pane is, the semi-automatic review
loop, and the team-stop flow. Execution steps live in `team-mode-execution.md`.

### Team Roles

| Role           | Dev Team                                    | Content Team                                    |
|----------------|---------------------------------------------|-------------------------------------------------|
| Creator        | developer — implements features/fixes       | author — writes articles, scripts, newsletters  |
| Codex reviewer | bugs, security, concurrency, edge cases     | logic, accuracy, structure, fact-checking       |
| Agy reviewer   | architecture, design patterns, alternatives | readability, engagement, style, audience fit    |

### Workflow Loop (Semi-Automatic)

1. **User assigns task** → Team Lead prompts the creator pane
2. **Creator completes** → Team Lead reads the pane and shows the result to the user
3. **User approves** → Team Lead submits to both reviewer panes **without** `--wait`, then `herdr agent wait` on each (a `prompt --wait` on the first blocks the second from ever starting — see Parallel Consensus)
4. **Reviewers report** → Team Lead reads both panes and consolidates, naming the effort/degradation level each landed on so a `low`-effort retry never reads like an `xhigh` first pass:
   ```
   ## Codex Review [effort: {level} — {N} retries]
   {findings}
   ## Agy Review [degradation: {level}]
   {findings}
   ```
5. **User decides** → "Revise" (loop to step 1) or "Pass" (next task or end)

User controls every transition. No autonomous loops.

### Execution Steps

The five steps — project detection, preflight, adopting what is live,
seeding each pane with its role, and confirming to the user — are in
`references/team-mode-execution.md`.

### team-stop Flow

1. Read `.three-brain/owned-panes.jsonl`. **If it is missing or empty, close nothing** and say so — with no ownership record you cannot prove any pane is yours, and guessing risks killing the user's own agent.
2. For each recorded pane, confirm it still hosts the agent you started (`herdr agent get <pane_id>` — pane IDs are never reused, but the occupant may have been replaced). Tell it to wrap up, then `herdr pane close <pane_id>`.
3. Delete `.three-brain/owned-panes.jsonl` after the closures succeed, so a re-run does not try again.
4. Leave every adopted agent running and say so — it was the user's before this team existed. In the common all-adopted case, team-stop closes nothing.
5. Report:

```text
Team shut down. Closed: {panes you started}. Left running: {agents you adopted}.
```
