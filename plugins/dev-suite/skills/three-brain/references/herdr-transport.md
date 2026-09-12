# Herdr transport — operational detail

Loaded by `three-brain` when it actually needs to drive the transport: discovering panes,
starting a missing agent, prompting one, reading the reply, or degrading when a kind is
unavailable. Kept out of SKILL.md because routing decisions do not need it.

### 1. Preflight and discovery

```bash
test "${HERDR_ENV:-}" = 1
```

If this fails, say you are not running inside Herdr and stop. Do not fall back to shelling out to `codex` or `agy` directly — this skill has no non-Herdr path.

**Normally Claude, Codex, and Agy are already running in the session.** Discovery is the main path; starting an agent is the exception. Find what is live and hand it work:

```bash
herdr agent list
```

Each entry is an `AgentInfo`. Four fields decide everything:

| Field | Use |
|---|---|
| `agent` | The **kind** — `codex`, `agy`, `claude`. Match on this, never on the name |
| `name` | The assigned name, **or `null`** for an agent the user started themselves |
| `pane_id` | Always present. The fallback target, and the only target for an unnamed agent |
| `agent_status` | `idle` / `working` / `blocked` / `done` / `unknown` |

Resolve one handle per kind with the bundled script — it encodes the three
selection rules that are easy to get wrong from memory, and it is checked by
`scripts/test-find-agents.sh` (runs without Herdr, off fixture JSON):

```bash
scripts/find-agents.sh              # -> one TSV line per kind: kind, count, handle
```

| Rule it enforces | Why it matters |
|---|---|
| Match on `agent` (the kind), take `.name // .pane_id` | A user-started agent often has `name: null`; agent commands accept either |
| Compare `cwd` by **equality**, not prefix | `startswith` adopts `/repo-old` when `$PWD` is `/repo` |
| Exclude `$HERDR_PANE_ID` | The list includes the pane running this skill; prompting yourself with `--wait` deadlocks |

Act on the count, never on the first line:

| Count | Do |
|---|---|
| 1 | Use that handle |
| 0 | No agent of that kind here — start one (step 2) |
| 2+ | **Ask the user which.** The extras are other people's or other tasks' agents |

Match `agent_status` before sending:

| Status | Do |
|---|---|
| `idle` / `done` | Ready. Prompt it |
| `working` | Busy on someone else's turn. Wait (`herdr agent wait <target> --timeout …`) or pick another; do not queue a prompt on top |
| `blocked` | An approval dialog is open. Read it, show the user, ask. Never auto-answer |
| `unknown` | Herdr can't classify it. This is **not** proof it is free — `agent read` before assuming |

### 2. Start an agent only when its kind is missing

If no agent of the needed kind is live in this working directory, create one — and record that you created it, because team-stop closes only those.

**One split and one `agent start` per kind.** A pane hosts exactly one agent; reusing a pane ID for a second `agent start` replaces or fails against the first.

```bash
# repeat this pair per missing kind — never reuse a pane id across two starts
herdr pane split --current --direction right --cwd "$PWD" --no-focus
#   -> read PANE from .result.pane.pane_id
herdr agent start cdx-<slug> --kind codex --pane "$PANE"

herdr pane split --current --direction down --cwd "$PWD" --no-focus
#   -> read a NEW PANE id from .result.pane.pane_id
herdr agent start agy-<slug> --kind agy --pane "$PANE"
```

Split a wide pane `right` and a narrow or tall pane `down`; check with `herdr pane layout --pane "$HERDR_PANE_ID"`. Always `--no-focus` — the user's focus stays where it was.

**Name scoping applies only to agents you start.** Names are unique among live agents on the server, so a bare `codex-reviewer` collides the moment a second team starts. Suffix a short project slug and stay inside `[a-z][a-z0-9_-]{0,31}`: `cdx-payments`, `agy-payments`, `dev-payments`.

`agent start` requires a pane already sitting at an interactive shell prompt, and never creates layout itself. It returns `agent_not_ready` if the agent is blocked during startup — the name still works for `agent read` and `agent send-keys`, so inspect before re-issuing.

### 3. Prompt and wait

```bash
herdr agent prompt $CODEX "<prompt text>" --wait --timeout 600000
```

`--wait` blocks until the first settled `idle`, `done`, or `blocked`. Do not add `--until` for ordinary work; it is only for waiting on a state-specific condition such as an already-running agent asking for input.

Prompt text goes straight through — no temp file, no stdin pipe, no shell quoting of a diff into an argv. To review a diff, tell the agent where to look (`Review the uncommitted changes in this repo`); these are agentic CLIs sitting in the working directory and they read files themselves.

Distinguish the failure returns, because they mean different things:

| Return | Meaning | Do |
|---|---|---|
| `agent_blocked` | An approval/question dialog was already open; **nothing was sent** | `agent read` the dialog, show it to the user, ask. Never auto-answer |
| `agent_prompt_stalled` | Submitted, but no `working`/`blocked` activity within 5 s | Inspect with `agent get` before deciding; do not blind-resubmit |
| `timeout` | Your timeout expired (submission time counts toward it) | Same — the prompt may well have landed |

A timeout or stall is not proof the prompt was never delivered. Re-sending a review prompt twice gets you two turns of work and a confused pane.

### 4. Read the result — expect two rounds

```bash
herdr agent read $CODEX --source recent-unwrapped --lines 200
```

`recent-unwrapped` joins soft wraps and is the right source for transcripts.

**Codex and Agy render on the terminal's alternate screen, so this often returns a truncated response, and raising `--lines` cannot recover it** — rows that leave the alternate screen never enter Herdr's host scrollback. That is a property of the TUI, not a misconfiguration, so budget for the second round rather than treating it as an error:

```bash
# only after a first read came back truncated
herdr agent prompt $CODEX "Write your complete response as Markdown to a file under /tmp and reply with only the file path." --wait --timeout 300000
herdr agent read $CODEX --source recent-unwrapped --lines 20   # capture just the path
```

Then Read that file directly. Do **not** ask for file output in the initial prompt — first try the direct read, since a short answer comes back whole and one round is cheaper than two.

### 5. Degradation

**There is no effort ladder.** Reasoning effort is fixed when the agent process starts — for an adopted pane it was set by whoever launched it, and you cannot change it by re-prompting. Do not append "use xhigh reasoning" hints and do not label a retry with an effort level: the label would be fiction, and a reader comparing an `[effort: low]` retry against an `[effort: xhigh]` first pass would be comparing two identical configurations.

Retry by **narrowing the request**, in the same pane, keeping its context: simplify the prompt → reduce the number of dimensions asked for → ask about one file instead of the whole diff.

After two narrowing attempts fail, stop and label the result `[Claude Fallback — <agent> retries failed]`. Restarting an agent to change its effort is not an option here: it destroys the pane's context, and for an adopted pane it destroys the user's session.

### 6. Hard rules

- Never skip the agent prompt and review it yourself. A same-model review is the one thing this skill exists to avoid.
- Never auto-answer a `blocked` dialog. Surface it and ask.
- Never close a pane, tab, or workspace you did not create. If you reused an existing agent, it is not yours to shut down.
- Never run bare `herdr` (it launches the TUI) and never `herdr server stop`.
- Parse IDs out of the JSON responses. Do not guess `w1:p2` from position.
- Use `--format ansi` only when colour is itself the evidence.

---
