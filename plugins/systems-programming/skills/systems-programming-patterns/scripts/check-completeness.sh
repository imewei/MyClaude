#!/usr/bin/env bash
# Check completeness of the systems-programming-patterns skill

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=== Completeness Check for systems-programming-patterns (Parallel) ==="
echo

# Setup temporary directory for parallel results
RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

# Function to write results
record_result() {
    local name="$1"
    local passed="$2"
    local failed="$3"
    echo "$passed $failed" > "$RESULTS_DIR/$name.stats"
}

# Check function (modified for parallel execution)
check() {
    local description="$1"
    shift
    local condition="$*"

    if eval "$condition"; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        return 1
    fi
}

# 1. Core Files Check
check_core_files() {
    local passed=0
    local failed=0

    echo -e "${BLUE}=== Core Files ===${NC}"
    if check "SKILL.md exists" "[[ -f '$SKILL_DIR/SKILL.md' ]]"; then ((passed++)); else ((failed++)); fi
    if check "SKILL.md has content (>500 lines)" "[[ $(wc -l < '$SKILL_DIR/SKILL.md') -gt 500 ]]"; then ((passed++)); else ((failed++)); fi
    if check "SKILL.md has frontmatter" "grep -q '^name:' '$SKILL_DIR/SKILL.md'"; then ((passed++)); else ((failed++)); fi
    if check "SKILL.md has description" "grep -q '^description:' '$SKILL_DIR/SKILL.md'"; then ((passed++)); else ((failed++)); fi

    record_result "core" $passed $failed
}

# 2. Reference Files Check
check_references() {
    local passed=0
    local failed=0

    echo -e "${BLUE}=== Reference Files ===${NC}"
    if check "references/ directory exists" "[[ -d '$SKILL_DIR/references' ]]"; then ((passed++)); else ((failed++)); fi
    if check "profiling-guide.md exists" "[[ -f '$SKILL_DIR/references/profiling-guide.md' ]]"; then ((passed++)); else ((failed++)); fi
    if check "memory-pools.md exists" "[[ -f '$SKILL_DIR/references/memory-pools.md' ]]"; then ((passed++)); else ((failed++)); fi

    # Size checks
    if [[ -f "$SKILL_DIR/references/memory-pools.md" ]]; then
        if check "memory-pools.md has content (>1000 lines)" "[[ $(wc -l < '$SKILL_DIR/references/memory-pools.md') -gt 1000 ]]"; then ((passed++)); else ((failed++)); fi
    fi

    record_result "references" $passed $failed
}

# 3. Assets & Scripts Check
check_assets_scripts() {
    local passed=0
    local failed=0

    echo -e "${BLUE}=== Assets & Scripts ===${NC}"
    if check "assets/ directory exists" "[[ -d '$SKILL_DIR/assets' ]]"; then ((passed++)); else ((failed++)); fi
    if check "scripts/ directory exists" "[[ -d '$SKILL_DIR/scripts' ]]"; then ((passed++)); else ((failed++)); fi

    if [[ -d "$SKILL_DIR/scripts" ]]; then
        if check "validate-links.sh executable" "[[ -x '$SKILL_DIR/scripts/validate-links.sh' ]]"; then ((passed++)); else ((failed++)); fi
    fi

    record_result "assets" $passed $failed
}

# Run checks in parallel
check_core_files &
PID1=$!
check_references &
PID2=$!
check_assets_scripts &
PID3=$!

# Wait for all checks
wait $PID1
wait $PID2
wait $PID3

echo

# Aggregate results
total_checks=0
passed_checks=0
failed_checks=0

for stat_file in "$RESULTS_DIR"/*.stats; do
    if [[ -f "$stat_file" ]]; then
        read p f < "$stat_file"
        ((passed_checks += p))
        ((failed_checks += f))
    fi
done

total_checks=$((passed_checks + failed_checks))

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo "Total checks: $total_checks"
echo -e "${GREEN}Passed: $passed_checks${NC}"
echo -e "${RED}Failed: $failed_checks${NC}"

percentage=$((passed_checks * 100 / total_checks))
echo "Completeness: $percentage%"

echo

if [[ $failed_checks -eq 0 ]]; then
    echo -e "${GREEN}✓ All completeness checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ $failed_checks check(s) failed${NC}"
    exit 1
fi
