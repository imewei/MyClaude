#!/bin/bash
#
# validate-agent-consistency.sh
# Validates that all agent references in commands exist in personal agents
#
# Exit codes:
#   0 - All agent references are valid
#   1 - Invalid agent references found
#   2 - Script error (missing directories, etc.)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
AGENTS_DIR="$HOME/.claude/agents"
COMMANDS_DIR="$HOME/.claude/commands"

# Check if directories exist
if [[ ! -d "$AGENTS_DIR" ]]; then
  echo -e "${RED}❌ Error: Agents directory not found: $AGENTS_DIR${NC}"
  exit 2
fi

if [[ ! -d "$COMMANDS_DIR" ]]; then
  echo -e "${RED}❌ Error: Commands directory not found: $COMMANDS_DIR${NC}"
  exit 2
fi

echo "🔍 Validating agent consistency..."
echo ""

# Extract personal agent names (excluding template)
echo "📋 Extracting personal agent names..."
PERSONAL_AGENTS=$(grep -h "^name: " "$AGENTS_DIR"/*.md 2>/dev/null | \
  grep -v "agent-name-here" | \
  sed 's/^name: //' | \
  sort -u)

AGENT_COUNT=$(echo "$PERSONAL_AGENTS" | wc -l | tr -d ' ')
echo -e "${GREEN}   Found $AGENT_COUNT personal agents${NC}"

# Extract agent references from commands
echo ""
echo "🔍 Extracting agent references from commands..."

# Find all agent references in command frontmatter
# This includes both primary and conditional agents

# Primary agents (simple list format: "    - agent-name")
COMMAND_REFS=$(grep -rh "^\s*-\s*[a-z][a-z-]*$" "$COMMANDS_DIR"/*.md 2>/dev/null | \
  sed 's/^[[:space:]]*-[[:space:]]*//' | \
  sort -u)

# Conditional agents ("    - agent: agent-name" format)
# Use awk to extract just the agent name (last field)
CONDITIONAL_REFS=$(grep -rh "agent:" "$COMMANDS_DIR"/*.md 2>/dev/null | \
  grep -E "^\s+- agent:" | \
  awk '{print $NF}' | \
  sort -u)

# Combine and deduplicate
ALL_REFS=$(echo -e "$COMMAND_REFS\n$CONDITIONAL_REFS" | sort -u | grep -v "^$")

REF_COUNT=$(echo "$ALL_REFS" | wc -l | tr -d ' ')
echo -e "${GREEN}   Found $REF_COUNT unique agent references${NC}"

# Validate each reference
echo ""
echo "✅ Validating agent references..."
echo ""

# Debug: Show first few personal agents
# echo "DEBUG: First 3 personal agents:"
# echo "$PERSONAL_AGENTS" | head -3
# echo ""

INVALID_REFS=()
VALID_COUNT=0

while IFS= read -r ref; do
  # Trim only leading and trailing whitespace from ref
  ref=$(echo "$ref" | sed -e 's/^[[:space:]]*//;s/[[:space:]]*$//')

  if echo "$PERSONAL_AGENTS" | grep -qx "$ref"; then
    VALID_COUNT=$((VALID_COUNT + 1))
    # Uncomment for verbose output:
    # echo -e "${GREEN}   ✓${NC} $ref"
  else
    INVALID_REFS+=("$ref")
    echo -e "${RED}   ✗${NC} $ref ${YELLOW}(not found in personal agents)${NC}"
  fi
done <<< "$ALL_REFS"

# Report results
echo ""
echo "═══════════════════════════════════════════════════════"

if [[ ${#INVALID_REFS[@]} -eq 0 ]]; then
  echo -e "${GREEN}✅ SUCCESS: All $VALID_COUNT agent references are valid!${NC}"
  echo ""
  echo "Personal agents: $AGENT_COUNT"
  echo "Command references: $REF_COUNT"
  echo "Valid references: $VALID_COUNT"
  echo ""
  exit 0
else
  echo -e "${RED}❌ FAILURE: Found ${#INVALID_REFS[@]} invalid agent reference(s)${NC}"
  echo ""
  echo "Invalid references:"
  for invalid in "${INVALID_REFS[@]}"; do
    echo -e "  ${RED}✗${NC} $invalid"
  done
  echo ""
  echo -e "${YELLOW}Hint: Check that the 'name:' field in personal agent files matches command references${NC}"
  echo ""
  exit 1
fi
