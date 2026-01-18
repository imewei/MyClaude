#!/usr/bin/env bash
# Validate python-development plugin structure and content

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=== Python Development Plugin Validation (Parallel) ==="
echo "Plugin directory: $PLUGIN_DIR"
echo

# Temporary directory for results
RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

# Function to record results
record_result() {
    local name="$1"
    local errors="$2"
    local warnings="$3"
    echo "$errors $warnings" > "$RESULTS_DIR/$name.stats"
}

# 1. Core Plugin Files Check
check_core() {
    local errors=0
    local warnings=0

    echo -e "${BLUE}=== Core Plugin Files ===${NC}"
    if [[ -f "$PLUGIN_DIR/plugin.json" ]]; then
        echo -e "${GREEN}✓${NC} plugin.json exists"

        if python3 -m json.tool "$PLUGIN_DIR/plugin.json" >/dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} plugin.json is valid JSON"
        else
            echo -e "${RED}✗${NC} plugin.json has invalid JSON syntax"
            ((errors++))
        fi

        for field in name version description agents skills; do
            if grep -q "\"$field\"" "$PLUGIN_DIR/plugin.json"; then
                echo -e "${GREEN}✓${NC} plugin.json has '$field' field"
            else
                echo -e "${RED}✗${NC} plugin.json missing '$field' field"
                ((errors++))
            fi
        done
    else
        echo -e "${RED}✗${NC} plugin.json not found"
        ((errors++))
    fi

    record_result "core" $errors $warnings
}

# 2. Agents Check
check_agents() {
    local errors=0
    local warnings=0

    echo -e "${BLUE}=== Agents ===${NC}"
    if [[ -d "$PLUGIN_DIR/agents" ]]; then
        agent_count=$(find "$PLUGIN_DIR/agents" -name "*.md" | wc -l)
        echo "Found $agent_count agent file(s)"

        for agent in "$PLUGIN_DIR/agents"/*.md; do
            if [[ -f "$agent" ]]; then
                agent_name=$(basename "$agent" .md)

                if head -5 "$agent" | grep -q "^name:"; then
                    echo -e "${GREEN}✓${NC} $agent_name has frontmatter"
                else
                    echo -e "${RED}✗${NC} $agent_name missing frontmatter"
                    ((errors++))
                fi

                if grep -q "^model:" "$agent"; then
                    echo -e "${GREEN}✓${NC} $agent_name specifies model"
                else
                    echo -e "${YELLOW}⚠${NC} $agent_name missing model specification"
                    ((warnings++))
                fi
            fi
        done
    else
        echo -e "${RED}✗${NC} agents/ directory not found"
        ((errors++))
    fi

    record_result "agents" $errors $warnings
}

# 3. Skills Check
check_skills() {
    local errors=0
    local warnings=0

    echo -e "${BLUE}=== Skills ===${NC}"
    if [[ -d "$PLUGIN_DIR/skills" ]]; then
        skill_count=$(find "$PLUGIN_DIR/skills" -name "SKILL.md" | wc -l)
        echo "Found $skill_count skill(s)"

        for skill in "$PLUGIN_DIR/skills"/*/SKILL.md; do
            if [[ -f "$skill" ]]; then
                skill_name=$(basename $(dirname "$skill"))

                if head -5 "$skill" | grep -q "^name:"; then
                    echo -e "${GREEN}✓${NC} $skill_name has frontmatter"
                else
                    echo -e "${RED}✗${NC} $skill_name missing frontmatter"
                    ((errors++))
                fi

                line_count=$(wc -l < "$skill")
                if [[ $line_count -gt 100 ]]; then
                    echo -e "${GREEN}✓${NC} $skill_name has substantial content ($line_count lines)"
                else
                    echo -e "${YELLOW}⚠${NC} $skill_name may be too short ($line_count lines)"
                    ((warnings++))
                fi
            fi
        done
    else
        echo -e "${RED}✗${NC} skills/ directory not found"
        ((errors++))
    fi

    record_result "skills" $errors $warnings
}

# 4. Commands Check
check_commands() {
    local errors=0
    local warnings=0

    echo -e "${BLUE}=== Commands ===${NC}"
    if [[ -d "$PLUGIN_DIR/commands" ]]; then
        cmd_count=$(find "$PLUGIN_DIR/commands" -name "*.md" | wc -l)
        echo "Found $cmd_count command(s)"
    else
        echo -e "${YELLOW}⚠${NC} commands/ directory not found (optional)"
    fi

    record_result "commands" $errors $warnings
}

# Run checks in parallel
check_core &
PID1=$!
check_agents &
PID2=$!
check_skills &
PID3=$!
check_commands &
PID4=$!

# Wait for all checks
wait $PID1
wait $PID2
wait $PID3
wait $PID4

echo

# Aggregate results
errors=0
warnings=0

for stat_file in "$RESULTS_DIR"/*.stats; do
    if [[ -f "$stat_file" ]]; then
        read e w < "$stat_file"
        ((errors += e))
        ((warnings += w))
    fi
done

echo

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo "Errors: $errors"
echo "Warnings: $warnings"
echo

if [[ $errors -eq 0 ]]; then
    echo -e "${GREEN}✓ Validation passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Validation failed with $errors error(s)${NC}"
    exit 1
fi
