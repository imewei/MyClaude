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

echo "=== Completeness Check for systems-programming-patterns ==="
echo

total_checks=0
passed_checks=0
failed_checks=0

# Check function
check() {
    local description="$1"
    shift
    local condition="$*"

    ((total_checks++))

    if eval "$condition"; then
        echo -e "${GREEN}✓${NC} $description"
        ((passed_checks++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        ((failed_checks++))
        return 1
    fi
}

# Core Files
echo -e "${BLUE}=== Core Files ===${NC}"
check "SKILL.md exists" "[[ -f '$SKILL_DIR/SKILL.md' ]]"
check "SKILL.md has content (>500 lines)" "[[ $(wc -l < '$SKILL_DIR/SKILL.md') -gt 500 ]]"
check "SKILL.md has frontmatter" "grep -q '^name:' '$SKILL_DIR/SKILL.md'"
check "SKILL.md has description" "grep -q '^description:' '$SKILL_DIR/SKILL.md'"
echo

# Reference Files
echo -e "${BLUE}=== Reference Files ===${NC}"
check "references/ directory exists" "[[ -d '$SKILL_DIR/references' ]]"
check "profiling-guide.md exists" "[[ -f '$SKILL_DIR/references/profiling-guide.md' ]]"
check "memory-pools.md exists" "[[ -f '$SKILL_DIR/references/memory-pools.md' ]]"
check "lock-free-patterns.md exists" "[[ -f '$SKILL_DIR/references/lock-free-patterns.md' ]]"
check "raii-patterns.md exists" "[[ -f '$SKILL_DIR/references/raii-patterns.md' ]]"
check "rcu-patterns.md exists" "[[ -f '$SKILL_DIR/references/rcu-patterns.md' ]]"
check "common-bugs.md exists" "[[ -f '$SKILL_DIR/references/common-bugs.md' ]]"

# Check reference file sizes (should have content)
if [[ -f "$SKILL_DIR/references/memory-pools.md" ]]; then
    check "memory-pools.md has content (>1000 lines)" "[[ $(wc -l < '$SKILL_DIR/references/memory-pools.md') -gt 1000 ]]"
fi

if [[ -f "$SKILL_DIR/references/lock-free-patterns.md" ]]; then
    check "lock-free-patterns.md has content (>500 lines)" "[[ $(wc -l < '$SKILL_DIR/references/lock-free-patterns.md') -gt 500 ]]"
fi

echo

# Assets
echo -e "${BLUE}=== Assets ===${NC}"
check "assets/ directory exists" "[[ -d '$SKILL_DIR/assets' ]]"

if [[ -d "$SKILL_DIR/assets" ]]; then
    asset_count=$(find "$SKILL_DIR/assets" -type f -name "*.md" | wc -l)
    check "assets/ has diagram files (>2)" "[[ $asset_count -gt 2 ]]"

    check "assets/README.md exists" "[[ -f '$SKILL_DIR/assets/README.md' ]]"
fi

echo

# Scripts
echo -e "${BLUE}=== Scripts ===${NC}"
check "scripts/ directory exists" "[[ -d '$SKILL_DIR/scripts' ]]"

if [[ -d "$SKILL_DIR/scripts" ]]; then
    check "validate-links.sh exists" "[[ -f '$SKILL_DIR/scripts/validate-links.sh' ]]"
    check "check-completeness.sh exists" "[[ -f '$SKILL_DIR/scripts/check-completeness.sh' ]]"

    # Check if scripts are executable
    if [[ -f "$SKILL_DIR/scripts/validate-links.sh" ]]; then
        check "validate-links.sh is executable" "[[ -x '$SKILL_DIR/scripts/validate-links.sh' ]]"
    fi

    if [[ -f "$SKILL_DIR/scripts/check-completeness.sh" ]]; then
        check "check-completeness.sh is executable" "[[ -x '$SKILL_DIR/scripts/check-completeness.sh' ]]"
    fi
fi

echo

# Content Quality Checks
echo -e "${BLUE}=== Content Quality ===${NC}"

if [[ -f "$SKILL_DIR/SKILL.md" ]]; then
    check "SKILL.md mentions all 4 languages (C, C++, Rust, Go)" \
        "grep -qi 'rust' '$SKILL_DIR/SKILL.md' && grep -qi 'c++' '$SKILL_DIR/SKILL.md' && grep -qi 'golang\|\\bgo\\b' '$SKILL_DIR/SKILL.md'"

    check "SKILL.md has code blocks" "grep -q '\`\`\`' '$SKILL_DIR/SKILL.md'"

    check "SKILL.md has memory management section" "grep -qi 'memory.*management' '$SKILL_DIR/SKILL.md'"

    check "SKILL.md has concurrency section" "grep -qi 'concurrency\|concurrent' '$SKILL_DIR/SKILL.md'"

    check "SKILL.md has performance section" "grep -qi 'performance' '$SKILL_DIR/SKILL.md'"
fi

echo

# Reference Completeness
echo -e "${BLUE}=== Reference Completeness ===${NC}"

# Extract all reference links from SKILL.md
if [[ -f "$SKILL_DIR/SKILL.md" ]]; then
    echo "Checking all referenced files exist..."

    while IFS= read -r ref_path; do
        full_path="$SKILL_DIR/$ref_path"
        if [[ -f "$full_path" ]]; then
            echo -e "  ${GREEN}✓${NC} $ref_path"
            ((passed_checks++))
        else
            echo -e "  ${RED}✗${NC} $ref_path (referenced but missing)"
            ((failed_checks++))
        fi
        ((total_checks++))
    done < <(grep -oP 'references/[a-z-]+\.md' "$SKILL_DIR/SKILL.md" | sort -u)
fi

echo

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
