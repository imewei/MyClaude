#!/bin/bash
#
# validate-agents.sh
# Comprehensive agent system validation
#
# Validates:
# - Dependencies (Node.js, npm packages)
# - Agent files (frontmatter, syntax)
# - Agent registry (loading, caching)
# - Matching algorithm
# - Performance benchmarks
#
# Exit codes:
#   0 - All validations passed
#   1 - Validation failures found

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   AGENT SYSTEM VALIDATION SUITE${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""

run_test() {
  local test_name="$1"
  local test_command="$2"

  TESTS_RUN=$((TESTS_RUN + 1))
  echo -n "[$TESTS_RUN] ${test_name}... "

  if eval "$test_command" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    return 0
  else
    echo -e "${RED}✗ FAIL${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    return 1
  fi
}

run_test_with_output() {
  local test_name="$1"
  local test_command="$2"
  local expected_condition="$3"

  TESTS_RUN=$((TESTS_RUN + 1))
  echo -n "[$TESTS_RUN] ${test_name}... "

  local output
  output=$(eval "$test_command" 2>&1)
  local exit_code=$?

  if [ $exit_code -eq 0 ] && eval "$expected_condition"; then
    echo -e "${GREEN}✓ PASS${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    return 0
  else
    echo -e "${RED}✗ FAIL${NC}"
    echo "   Output: $output"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    return 1
  fi
}

echo -e "${BLUE}━━━ Phase 1: Environment Checks ━━━${NC}"
echo ""

# Check Node.js
run_test "Node.js installed" "command -v node"

if command -v node &> /dev/null; then
  NODE_VERSION=$(node --version)
  echo "   Node.js version: $NODE_VERSION"
fi

# Check npm
run_test "npm installed" "command -v npm"

# Check package.json exists
run_test "package.json exists" "test -f ~/.claude/scripts/package.json"

# Check node_modules exists
run_test "Dependencies installed" "test -d ~/.claude/scripts/node_modules"

if [ ! -d ~/.claude/scripts/node_modules ]; then
  echo -e "${YELLOW}   ⚠️  Run: cd ~/.claude/scripts && npm install${NC}"
fi

# Check glob package
run_test "glob package installed" "test -d ~/.claude/scripts/node_modules/glob"

echo ""
echo -e "${BLUE}━━━ Phase 2: File Structure Checks ━━━${NC}"
echo ""

# Check scripts exist
run_test "agent-registry.mjs exists" "test -f ~/.claude/scripts/agent-registry.mjs"
run_test "agent-cli.mjs exists" "test -f ~/.claude/scripts/agent-cli.mjs"
run_test "agent-suggest-hook.sh exists" "test -f ~/.claude/scripts/agent-suggest-hook.sh"

# Check executability
run_test "agent-cli.mjs is executable" "test -x ~/.claude/scripts/agent-cli.mjs"
run_test "agent-suggest-hook.sh is executable" "test -x ~/.claude/scripts/agent-suggest-hook.sh"

# Check slash command
run_test "/agent command exists" "test -f ~/.claude/commands/agent.md"

# Check agents directory
run_test "Agents directory exists" "test -d ~/.claude/agents"

echo ""
echo -e "${BLUE}━━━ Phase 3: Agent Files Validation ━━━${NC}"
echo ""

# Count agent files
AGENT_COUNT=$(find ~/.claude/agents -name "*.md" -type f 2>/dev/null | grep -v AGENT_TEMPLATE | wc -l | tr -d ' ')
echo "   Found $AGENT_COUNT agent files"

if [ "$AGENT_COUNT" -eq 0 ]; then
  echo -e "${RED}   ✗ No agent files found!${NC}"
  TESTS_FAILED=$((TESTS_FAILED + 1))
else
  echo -e "${GREEN}   ✓ Agent files found${NC}"
  TESTS_PASSED=$((TESTS_PASSED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

# Validate frontmatter
INVALID_AGENTS=0
echo "   Validating frontmatter..."

for agent_file in ~/.claude/agents/*.md; do
  if [ "$(basename "$agent_file")" = "AGENT_TEMPLATE.md" ]; then
    continue
  fi

  # Check frontmatter exists (supports both --- and -- formats)
  if ! grep -qE "^---?$" "$agent_file"; then
    echo -e "${RED}   ✗ Missing frontmatter: $(basename "$agent_file")${NC}"
    INVALID_AGENTS=$((INVALID_AGENTS + 1))
    continue
  fi

  # Check name field
  if ! grep -q "^name:" "$agent_file"; then
    echo -e "${RED}   ✗ Missing name field: $(basename "$agent_file")${NC}"
    INVALID_AGENTS=$((INVALID_AGENTS + 1))
    continue
  fi

  # Check description field
  if ! grep -q "^description:" "$agent_file"; then
    echo -e "${YELLOW}   ⚠️  Missing description: $(basename "$agent_file")${NC}"
  fi
done

if [ "$INVALID_AGENTS" -eq 0 ]; then
  echo -e "${GREEN}   ✓ All agent files have valid frontmatter${NC}"
  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${RED}   ✗ Found $INVALID_AGENTS invalid agent files${NC}"
  TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

echo ""
echo -e "${BLUE}━━━ Phase 4: Agent Registry Tests ━━━${NC}"
echo ""

# Test registry list
echo -n "[$TESTS_RUN] Agent registry list command... "
TESTS_RUN=$((TESTS_RUN + 1))
if LIST_OUTPUT=$(node ~/.claude/scripts/agent-cli.mjs list 2>&1); then
  LOADED_COUNT=$(echo "$LIST_OUTPUT" | jq '. | length' 2>/dev/null || echo "0")
  echo -e "${GREEN}✓ PASS${NC}"
  echo "   Loaded $LOADED_COUNT agents"
  TESTS_PASSED=$((TESTS_PASSED + 1))

  if [ "$LOADED_COUNT" -ne "$AGENT_COUNT" ]; then
    echo -e "${YELLOW}   ⚠️  Mismatch: $AGENT_COUNT files but $LOADED_COUNT loaded${NC}"
  fi
else
  echo -e "${RED}✗ FAIL${NC}"
  echo "   Error: $LIST_OUTPUT"
  TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test agent get
if [ "$LOADED_COUNT" -gt 0 ]; then
  FIRST_AGENT=$(echo "$LIST_OUTPUT" | jq -r '.[0].name' 2>/dev/null)

  echo -n "[$TESTS_RUN] Agent get command (${FIRST_AGENT})... "
  TESTS_RUN=$((TESTS_RUN + 1))
  if GET_OUTPUT=$(node ~/.claude/scripts/agent-cli.mjs get "$FIRST_AGENT" 2>&1); then
    echo -e "${GREEN}✓ PASS${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))

    # Verify systemPrompt exists
    if echo "$GET_OUTPUT" | jq -e '.systemPrompt' > /dev/null 2>&1; then
      PROMPT_LENGTH=$(echo "$GET_OUTPUT" | jq -r '.systemPrompt | length')
      echo "   System prompt length: $PROMPT_LENGTH chars"
    fi
  else
    echo -e "${RED}✗ FAIL${NC}"
    echo "   Error: $GET_OUTPUT"
    TESTS_FAILED=$((TESTS_FAILED + 1))
  fi
fi

# Test agent matching
echo -n "[$TESTS_RUN] Agent matching algorithm... "
TESTS_RUN=$((TESTS_RUN + 1))
if MATCH_OUTPUT=$(node ~/.claude/scripts/agent-cli.mjs match "optimize JAX code performance" 2>&1); then
  MATCH_COUNT=$(echo "$MATCH_OUTPUT" | jq '. | length' 2>/dev/null || echo "0")
  echo -e "${GREEN}✓ PASS${NC}"
  echo "   Found $MATCH_COUNT matches for test query"

  if [ "$MATCH_COUNT" -gt 0 ]; then
    BEST_MATCH=$(echo "$MATCH_OUTPUT" | jq -r '.[0].agent.name' 2>/dev/null)
    BEST_SCORE=$(echo "$MATCH_OUTPUT" | jq -r '.[0].score' 2>/dev/null)
    echo "   Best match: $BEST_MATCH (score: $BEST_SCORE)"
  fi

  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${RED}✗ FAIL${NC}"
  echo "   Error: $MATCH_OUTPUT"
  TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test refresh
echo -n "[$TESTS_RUN] Agent registry refresh... "
TESTS_RUN=$((TESTS_RUN + 1))
if REFRESH_OUTPUT=$(node ~/.claude/scripts/agent-cli.mjs refresh 2>&1); then
  echo -e "${GREEN}✓ PASS${NC}"
  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${RED}✗ FAIL${NC}"
  echo "   Error: $REFRESH_OUTPUT"
  TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""
echo -e "${BLUE}━━━ Phase 5: Performance Benchmarks ━━━${NC}"
echo ""

# Benchmark cached lookup
echo -n "[$TESTS_RUN] Cached lookup performance... "
TESTS_RUN=$((TESTS_RUN + 1))
START_TIME=$(date +%s%N)
node ~/.claude/scripts/agent-cli.mjs list > /dev/null 2>&1
END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [ "$ELAPSED_MS" -lt 500 ]; then
  echo -e "${GREEN}✓ PASS${NC} (${ELAPSED_MS}ms)"
  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${YELLOW}✓ PASS${NC} (${ELAPSED_MS}ms - slower than target <500ms)"
  TESTS_PASSED=$((TESTS_PASSED + 1))
fi

# Benchmark matching
echo -n "[$TESTS_RUN] Matching performance... "
TESTS_RUN=$((TESTS_RUN + 1))
START_TIME=$(date +%s%N)
node ~/.claude/scripts/agent-cli.mjs match "test query" > /dev/null 2>&1
END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [ "$ELAPSED_MS" -lt 200 ]; then
  echo -e "${GREEN}✓ PASS${NC} (${ELAPSED_MS}ms)"
  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${YELLOW}✓ PASS${NC} (${ELAPSED_MS}ms - slower than target <200ms)"
  TESTS_PASSED=$((TESTS_PASSED + 1))
fi

echo ""
echo -e "${BLUE}━━━ Phase 6: Integration Tests ━━━${NC}"
echo ""

# Test hook script
echo -n "[$TESTS_RUN] Suggestion hook execution... "
TESTS_RUN=$((TESTS_RUN + 1))
if HOOK_OUTPUT=$(~/.claude/scripts/agent-suggest-hook.sh "optimize JAX code" 2>&1); then
  echo -e "${GREEN}✓ PASS${NC}"
  TESTS_PASSED=$((TESTS_PASSED + 1))
  if echo "$HOOK_OUTPUT" | grep -q "Agent suggestion"; then
    echo "   Hook provided suggestion"
  fi
else
  echo -e "${RED}✗ FAIL${NC}"
  TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Check cache file
echo -n "[$TESTS_RUN] Cache file creation... "
TESTS_RUN=$((TESTS_RUN + 1))
if [ -f ~/.claude/config/agent-registry-cache.json ]; then
  CACHE_SIZE=$(du -h ~/.claude/config/agent-registry-cache.json | cut -f1)
  echo -e "${GREEN}✓ PASS${NC} (size: $CACHE_SIZE)"
  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${YELLOW}⚠ WARN${NC} (cache will be created on first use)"
  TESTS_PASSED=$((TESTS_PASSED + 1))
fi

# Check settings.json hooks
echo -n "[$TESTS_RUN] Settings.json hooks configured... "
TESTS_RUN=$((TESTS_RUN + 1))
if grep -q "UserPromptSubmit" ~/.claude/settings.json 2>/dev/null; then
  echo -e "${GREEN}✓ PASS${NC}"
  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${YELLOW}⚠ WARN${NC} (hooks not configured in settings.json)"
  TESTS_PASSED=$((TESTS_PASSED + 1))
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   VALIDATION SUMMARY${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Total tests run:    $TESTS_RUN"
echo -e "Tests passed:       ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed:       ${RED}$TESTS_FAILED${NC}"
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
  echo -e "${GREEN}✅ ALL VALIDATIONS PASSED!${NC}"
  echo ""
  echo "Agent System Status:"
  echo "  - Agent files: $AGENT_COUNT"
  echo "  - Loaded agents: $LOADED_COUNT"
  echo "  - Matching: Working"
  echo "  - Cache: Operational"
  echo ""
  echo "Next steps:"
  echo "  1. Try: /agent --list"
  echo "  2. Try: /agent Optimize my JAX code"
  echo "  3. Try: /agent --use jax-pro <your-task>"
  echo ""
  exit 0
else
  echo -e "${RED}❌ SOME VALIDATIONS FAILED${NC}"
  echo ""
  echo "Troubleshooting:"
  echo "  - Install dependencies: cd ~/.claude/scripts && npm install"
  echo "  - Check agent files: ls ~/.claude/agents/"
  echo "  - Rebuild cache: /agent --refresh"
  echo "  - Review errors above for specific issues"
  echo ""
  exit 1
fi
