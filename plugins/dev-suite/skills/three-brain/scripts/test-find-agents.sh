#!/usr/bin/env bash
# Self-check for find-agents.sh. Runs without Herdr — feeds fixture JSON.
set -uo pipefail
cd "$(dirname "$0")"
pass=0; fail=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then pass=$((pass+1)); printf 'ok   %s\n' "$1"
  else fail=$((fail+1)); printf 'FAIL %s\n       want: %s\n       got:  %s\n' "$1" "$2" "$3"; fi
}
F='{"result":{"agents":[
 {"agent":"claude","name":null,"pane_id":"w1:p1","cwd":"/repo"},
 {"agent":"codex","name":"cdx-main","pane_id":"w1:p2","cwd":"/repo"},
 {"agent":"codex","name":null,"pane_id":"w1:p3","cwd":"/repo-old"},
 {"agent":"agy","name":null,"pane_id":"w1:p4","foreground_cwd":"/repo/"},
 {"agent":"claude","name":"other","pane_id":"w1:p5","cwd":"/repo"}]}}'

run() { printf '%s' "$F" | ./find-agents.sh --cwd "$1" --self "$2" "$3"; }

check "excludes caller's own pane"      "claude	1	other"    "$(run /repo w1:p1 claude)"
check "cwd is equality, not prefix"     "codex	1	cdx-main" "$(run /repo w1:p1 codex)"
check "null name falls back to pane_id" "agy	1	w1:p4"    "$(run /repo w1:p1 agy)"
check "trailing slash normalized"       "agy	1	w1:p4"    "$(run /repo/ w1:p1 agy)"
check "no match in other dir -> 0"      "codex	0	-"        "$(run /elsewhere w1:p1 codex)"
check "kind absent entirely -> 0"       "agy	0	-"        "$(run /repo-old w1:p1 agy)"
# without self-exclusion both claude panes match -> must report 2 and refuse to pick
check "multi-match reports all"         "claude	2	w1:p1,other" "$(run /repo '' claude)"
check "empty agent list -> 0"           "codex	0	-" "$(printf '{"result":{"agents":[]}}' | ./find-agents.sh --cwd /repo --self x codex)"

printf '\n%d passed, %d failed\n' "$pass" "$fail"; [ "$fail" -eq 0 ]
