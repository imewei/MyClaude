#!/usr/bin/env bash
# Resolve one Herdr agent handle per kind, scoped to a working directory.
#
# Why this is a script and not inline instructions: every three-brain invocation
# needs the same selection, and it has three edge cases that are easy to get
# wrong when rewritten from memory — prefix-vs-equality on cwd, a null `name`
# on a user-started agent, and the caller's own pane appearing in the list.
# Getting any of them wrong sends a review to the wrong agent or deadlocks the
# session. Written once, checked once.
#
# Usage:   find-agents.sh [--cwd DIR] [--self PANE_ID] [kind ...]
#          (kinds default: claude codex agy)
# Input:   `herdr agent list` JSON on stdin, or live if omitted.
# Output:  one TSV line per kind:  <kind>\t<count>\t<handle|->
#          count 0 -> start one;  1 -> use handle;  2+ -> ask the user which.
# Env:     PWD / HERDR_PANE_ID are the defaults for --cwd / --self.
#          Both are overridable by flag because bash resets PWD at startup,
#          so an exported PWD cannot reach this script — which also makes the
#          flags the only way to test the selection logic off a fixture.
set -euo pipefail

scope="$PWD"; self="${HERDR_PANE_ID:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --cwd)  scope="$2"; shift 2 ;;
    --self) self="$2";  shift 2 ;;
    *) break ;;
  esac
done
kinds=("$@"); [ ${#kinds[@]} -eq 0 ] && kinds=(claude codex agy)

payload=$( [ -t 0 ] && herdr agent list || cat )

for kind in "${kinds[@]}"; do
  # Match on `agent` (the kind). Compare cwd by equality after trimming a
  # trailing slash — startswith() would adopt /repo-old when PWD is /repo.
  # Prefer `name`, fall back to pane_id: agent commands accept either, and a
  # user-started agent often has no name.
  handles=$(printf '%s' "$payload" | jq -r \
    --arg kind "$kind" --arg cwd "$scope" --arg self "$self" '
      [ .result.agents[]?
        | select(.agent == $kind)
        | select(.pane_id != $self)
        | select((((.cwd // .foreground_cwd // "") | rtrimstr("/")) == ($cwd | rtrimstr("/"))))
        | (.name // .pane_id) ]
      | .[]' 2>/dev/null || true)

  if [ -z "$handles" ]; then
    printf '%s\t0\t-\n' "$kind"
  else
    n=$(printf '%s\n' "$handles" | grep -c .)
    if [ "$n" -eq 1 ]; then printf '%s\t1\t%s\n' "$kind" "$handles"
    else printf '%s\t%s\t%s\n' "$kind" "$n" "$(printf '%s' "$handles" | paste -sd, -)"; fi
  fi
done
