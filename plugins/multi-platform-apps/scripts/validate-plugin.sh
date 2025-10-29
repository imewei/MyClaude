#!/bin/bash

# Validation script for multi-platform-apps plugin
# Checks plugin structure, agent files, and command files

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get plugin directory
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Multi-Platform Apps Plugin Validation ==="
echo "Plugin directory: $PLUGIN_DIR"
echo

# Counters
ERRORS=0
WARNINGS=0

# Core plugin files
echo -e "${BLUE}=== Core Plugin Files ===${NC}"

if [ -f "$PLUGIN_DIR/plugin.json" ]; then
    echo -e "${GREEN}✓${NC} plugin.json exists"

    # Validate JSON syntax
    if python3 -m json.tool "$PLUGIN_DIR/plugin.json" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} plugin.json is valid JSON"
    else
        echo -e "${RED}✗${NC} plugin.json has invalid JSON syntax"
        ((ERRORS++))
    fi

    # Check required fields
    for field in name version description agents commands; do
        if grep -q "\"$field\"" "$PLUGIN_DIR/plugin.json"; then
            echo -e "${GREEN}✓${NC} plugin.json has '$field' field"
        else
            echo -e "${RED}✗${NC} plugin.json missing '$field' field"
            ((ERRORS++))
        fi
    done
else
    echo -e "${RED}✗${NC} plugin.json not found"
    ((ERRORS++))
fi

echo

# Agents validation
echo -e "${BLUE}=== Agents ===${NC}"

AGENT_COUNT=$(find "$PLUGIN_DIR/agents" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
echo "Found $AGENT_COUNT agent file(s)"

if [ "$AGENT_COUNT" -gt 0 ]; then
    for agent_file in "$PLUGIN_DIR/agents"/*.md; do
        agent_name=$(basename "$agent_file" .md)

        # Check frontmatter exists
        if head -5 "$agent_file" | grep -q "^name:"; then
            echo -e "${GREEN}✓${NC} $agent_name has frontmatter"
        else
            echo -e "${RED}✗${NC} $agent_name missing frontmatter"
            ((ERRORS++))
        fi

        # Check model specified
        if head -10 "$agent_file" | grep -q "^model:"; then
            echo -e "${GREEN}✓${NC} $agent_name specifies model"
        else
            echo -e "${YELLOW}⚠${NC} $agent_name missing model specification"
            ((WARNINGS++))
        fi
    done
else
    echo -e "${RED}✗${NC} No agent files found"
    ((ERRORS++))
fi

echo

# Commands validation
echo -e "${BLUE}=== Commands ===${NC}"

COMMAND_COUNT=$(find "$PLUGIN_DIR/commands" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
echo "Found $COMMAND_COUNT command(s)"

if [ "$COMMAND_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠${NC} No commands found (optional)"
fi

echo

# Skills validation (optional for now)
echo -e "${BLUE}=== Skills ===${NC}"

if [ -d "$PLUGIN_DIR/skills" ]; then
    SKILL_COUNT=$(find "$PLUGIN_DIR/skills" -name "SKILL.md" 2>/dev/null | wc -l | tr -d ' ')
    echo "Found $SKILL_COUNT skill(s)"

    if [ "$SKILL_COUNT" -eq 0 ]; then
        echo -e "${YELLOW}⚠${NC} No skills found (recommended to add for educational depth)"
        ((WARNINGS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} Skills directory not found (recommended to create)"
    ((WARNINGS++))
fi

echo

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ Validation passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Validation failed with $ERRORS error(s)${NC}"
    exit 1
fi
