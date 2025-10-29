#!/usr/bin/env bash
# Simple completeness check

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Simple Completeness Check ==="
echo "Skill directory: $SKILL_DIR"
echo

errors=0

# Check core files
echo "Core files:"
[[ -f "$SKILL_DIR/SKILL.md" ]] && echo "✓ SKILL.md" || { echo "✗ SKILL.md"; ((errors++)); }

# Check references
echo
echo "Reference files:"
for ref in memory-pools lock-free-patterns raii-patterns rcu-patterns common-bugs profiling-guide; do
    if [[ -f "$SKILL_DIR/references/$ref.md" ]]; then
        echo "✓ $ref.md"
    else
        echo "✗ $ref.md"
        ((errors++))
    fi
done

# Check assets
echo
echo "Assets:"
asset_count=$(find "$SKILL_DIR/assets" -type f 2>/dev/null | wc -l)
echo "Found $asset_count file(s) in assets/"

# Check scripts
echo
echo "Scripts:"
script_count=$(find "$SKILL_DIR/scripts" -type f -name "*.sh" 2>/dev/null | wc -l)
echo "Found $script_count script(s)"

echo
if [[ $errors -eq 0 ]]; then
    echo "✓ All checks passed!"
    exit 0
else
    echo "✗ $errors error(s) found"
    exit 1
fi
