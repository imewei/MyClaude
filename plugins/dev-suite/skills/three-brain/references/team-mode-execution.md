# Team mode — execution steps

Loaded by `three-brain` when starting a persistent team: project detection, preflight,
adopting live panes, seeding each pane with its role, and confirming to the user.

### Execution Steps

#### 1. Project detection

1. Explicitly specified → use as-is
2. CWD is inside a project → derive the name, lowercase it, and cut a `<slug>` that keeps every agent name inside 32 characters
3. Ambiguous → ask the user

#### 2. Preflight

```bash
test "${HERDR_ENV:-}" = 1 || echo "NOT_IN_HERDR"
herdr agent list
```

Not in Herdr → say so and stop. Otherwise resolve one handle per kind (`claude`, `codex`, `agy`) scoped to this working directory, exactly as in Transport step 1. Adopted agents are **not yours to close at team-stop** — only ones you start are.

#### 3. Adopt what is live; start only what is missing

For each of the three kinds, take the handle discovery returned. Start a pane **only** for a kind with no live agent in this working directory.

**Ownership must outlive this conversation.** team-stop may run in a later session that has no memory of what was started, and the difference between "a pane I created" and "the user's own agent" is the difference between cleanup and destroying someone's work. Record it on disk as you create each pane:

```bash
mkdir -p .three-brain
# append one line per pane YOU created, never for an adopted agent
echo '{"pane_id":"<id>","name":"<name>","kind":"<codex|agy|claude>","slug":"<slug>"}' >> .three-brain/owned-panes.jsonl
```

The `.jsonl` extension is already covered by this repo's `.gitignore`; in a project where it is not, add `.three-brain/` before writing — it is session state, not project content.

```bash
herdr pane split --current --direction right --cwd "$PWD" --no-focus
herdr agent start dev-<slug> --kind claude --pane <id-from-.result.pane.pane_id>
```

Repeat per missing kind (`--kind codex`, `--kind agy`), alternating `right`/`down` per `herdr pane layout` rather than splitting the same direction three times, which leaves unusably narrow columns.

In the common case all three are already running and this step does nothing.

#### 4. Seed each pane with its role

**Submit prompt text safely.** Role text contains double quotes and `$`; pasting it inside `"..."` truncates the prompt, splits it into extra argv entries, or lets the shell expand it. Build it as one literal argument via a quoted heredoc:

```bash
ROLE=$(cat <<'ROLE_EOF'
<paste the role block verbatim — 'ROLE_EOF' quoted means no expansion>
ROLE_EOF
)
herdr agent prompt "$TARGET" "$ROLE" --wait --timeout 300000
```

`"$TARGET"` is quoted too: a `pane_id` is safe, but quoting keeps a surprising handle from splitting.

Read `references/agent-prompts.md` for the role-seeding text. Seed a pane you just started. For an **adopted** agent, `herdr agent read <target> --source recent-unwrapped --lines 40` first — it may be mid-conversation on unrelated work, in which case say so and ask the user before repurposing it. Send the seed with `herdr agent prompt <name> "<role text>" --wait --timeout 300000`. There are no reviewer subagents to spawn — the panes *are* the reviewers, and Herdr's own `idle`/`working`/`blocked`/`done` states replace the dispatcher bookkeeping a wrapper agent used to do.

#### 5. Confirm to user

```
Team ready (Herdr).
Team: {slug}  Type: {Dev / Content}
claude → {handle}  codex → {handle}  agy → {handle}
Adopted: {list}   Started this session: {list}
Awaiting your first task.
```
