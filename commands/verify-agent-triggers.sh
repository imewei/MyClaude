#!/usr/bin/env bash

# Agent Triggering Criteria Verification Script
# This script verifies that all agents have proper triggering criteria sections

set -e

AGENTS_DIR="$HOME/.claude/agents"
TOTAL_AGENTS=0
COMPLETE_AGENTS=0
MISSING_AGENTS=0

echo "=========================================="
echo "Agent Triggering Criteria Verification"
echo "=========================================="
echo ""

# Count total agents (excluding template)
cd "$AGENTS_DIR"
for file in *.md; do
    if [[ "$file" != "AGENT_TEMPLATE.md" ]]; then
        ((TOTAL_AGENTS++))
    fi
done

echo "Total agents to verify: $TOTAL_AGENTS"
echo ""
echo "Checking agents..."
echo ""

# Check each agent for triggering criteria
for file in *.md; do
    if [[ "$file" == "AGENT_TEMPLATE.md" ]]; then
        continue
    fi

    if grep -q "## Triggering Criteria" "$file"; then
        echo "✅ COMPLETE: $file"
        ((COMPLETE_AGENTS++))
    else
        echo "❌ MISSING: $file"
        ((MISSING_AGENTS++))
    fi
done

echo ""
echo "=========================================="
echo "Verification Results"
echo "=========================================="
echo "Total agents: $TOTAL_AGENTS"
echo "Complete: $COMPLETE_AGENTS"
echo "Missing: $MISSING_AGENTS"
echo "Completion: $((100 * COMPLETE_AGENTS / TOTAL_AGENTS))%"
echo ""

if [[ $MISSING_AGENTS -eq 0 ]]; then
    echo "✅ All agents have triggering criteria!"
    exit 0
else
    echo "❌ $MISSING_AGENTS agents are missing triggering criteria"
    exit 1
fi
