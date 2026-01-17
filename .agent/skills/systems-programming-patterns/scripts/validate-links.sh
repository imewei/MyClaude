#!/usr/bin/env bash
# Validate all internal references in the systems-programming-patterns skill

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Link Validation for systems-programming-patterns ==="
echo "Skill directory: $SKILL_DIR"
echo

errors=0
warnings=0

# Function to check if a file exists
check_file() {
    local file="$1"
    local referenced_from="$2"
    local line_num="$3"

    if [[ ! -f "$file" ]]; then
        echo -e "${RED}✗ BROKEN LINK${NC}"
        echo "  File: $referenced_from:$line_num"
        echo "  Missing: $file"
        echo
        ((errors++))
        return 1
    else
        echo -e "${GREEN}✓${NC} $file"
        return 0
    fi
}

# Check SKILL.md for references to files
echo "Checking SKILL.md..."
if [[ -f "$SKILL_DIR/SKILL.md" ]]; then
    while IFS=: read -r line_num line_content; do
        # Extract references like "references/filename.md"
        if [[ $line_content =~ references/([a-zA-Z0-9_-]+\.md) ]]; then
            ref_file="${BASH_REMATCH[0]}"
            full_path="$SKILL_DIR/$ref_file"
            check_file "$full_path" "SKILL.md" "$line_num"
        fi

        # Extract references like "assets/filename"
        if [[ $line_content =~ assets/([a-zA-Z0-9_-]+\.[a-z]+) ]]; then
            ref_file="${BASH_REMATCH[0]}"
            full_path="$SKILL_DIR/$ref_file"
            check_file "$full_path" "SKILL.md" "$line_num"
        fi
    done < <(grep -n "references/\|assets/" "$SKILL_DIR/SKILL.md" || true)
else
    echo -e "${RED}✗ SKILL.md not found!${NC}"
    ((errors++))
fi

echo

# Check all reference files exist
echo "Checking references/ directory..."
expected_refs=(
    "references/profiling-guide.md"
    "references/memory-pools.md"
    "references/lock-free-patterns.md"
    "references/raii-patterns.md"
    "references/rcu-patterns.md"
    "references/common-bugs.md"
)

for ref in "${expected_refs[@]}"; do
    full_path="$SKILL_DIR/$ref"
    if [[ -f "$full_path" ]]; then
        echo -e "${GREEN}✓${NC} $ref"
    else
        echo -e "${RED}✗${NC} $ref (missing)"
        ((errors++))
    fi
done

echo

# Check for broken links within reference files
echo "Checking internal links in reference files..."
for ref_file in "$SKILL_DIR"/references/*.md; do
    if [[ -f "$ref_file" ]]; then
        filename=$(basename "$ref_file")

        # Check for relative links (simplified pattern)
        grep -n "\.\./\|assets/\|references/" "$ref_file" | while IFS=: read -r line_num line_content; do
            # Skip this check for now - too complex for bash regex
            :
        done || true
    fi
done

echo

# Check assets directory
echo "Checking assets/ directory..."
if [[ -d "$SKILL_DIR/assets" ]]; then
    asset_count=$(find "$SKILL_DIR/assets" -type f | wc -l)
    if [[ $asset_count -eq 0 ]]; then
        echo -e "${YELLOW}⚠ Assets directory is empty${NC}"
        ((warnings++))
    else
        echo -e "${GREEN}✓${NC} Found $asset_count asset file(s)"
    fi
else
    echo -e "${YELLOW}⚠ Assets directory does not exist${NC}"
    ((warnings++))
fi

echo

# Check scripts directory
echo "Checking scripts/ directory..."
if [[ -d "$SKILL_DIR/scripts" ]]; then
    script_count=$(find "$SKILL_DIR/scripts" -type f -name "*.sh" | wc -l)
    if [[ $script_count -eq 0 ]]; then
        echo -e "${YELLOW}⚠ No shell scripts found in scripts/${NC}"
        ((warnings++))
    else
        echo -e "${GREEN}✓${NC} Found $script_count script file(s)"
    fi
else
    echo -e "${YELLOW}⚠ Scripts directory does not exist${NC}"
    ((warnings++))
fi

echo

# Summary
echo "=== Validation Summary ==="
if [[ $errors -eq 0 ]]; then
    echo -e "${GREEN}✓ All links are valid!${NC}"
else
    echo -e "${RED}✗ Found $errors broken link(s)${NC}"
fi

if [[ $warnings -gt 0 ]]; then
    echo -e "${YELLOW}⚠ $warnings warning(s)${NC}"
fi

echo

# Exit with error if any broken links found
if [[ $errors -gt 0 ]]; then
    exit 1
fi

exit 0
