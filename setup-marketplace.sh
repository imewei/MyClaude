#!/bin/bash
# Scientific Computing Marketplace Setup Script
# Creates a custom Claude Code marketplace with selected plugins

set -euo pipefail

# Configuration
MARKETPLACE_ROOT="/Users/b80985/Projects/MyClaude"
SOURCE_MARKETPLACE="$HOME/.claude/plugins/marketplaces/claude-code-workflows"
MARKETPLACE_NAME="scientific-computing-workflows"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "========================================="
echo " Scientific Computing Marketplace Setup"
echo "========================================="
echo ""

# Verify source marketplace exists
if [ ! -d "$SOURCE_MARKETPLACE" ]; then
    log_error "Source marketplace not found: $SOURCE_MARKETPLACE"
    exit 1
fi

# Phase 1: Create directory structure
log_info "Phase 1: Creating directory structure..."
mkdir -p "$MARKETPLACE_ROOT"/{.claude-plugin,docs,plugins,scripts}
log_success "Directory structure created"
echo ""

# Phase 2: Copy selected plugins
log_info "Phase 2: Copying selected plugins..."
SELECTED_PLUGINS=(
    "comprehensive-review"
    "code-documentation"
    "agent-orchestration"
    "unit-testing"
    "codebase-cleanup"
    "incident-response"
    "machine-learning-ops"
    "data-engineering"
    "framework-migration"
    "observability-monitoring"
    "backend-development"
    "frontend-mobile-development"
    "multi-platform-apps"
    "cicd-automation"
    "debugging-toolkit"
    "git-pr-workflows"
)

COPIED_COUNT=0
for plugin in "${SELECTED_PLUGINS[@]}"; do
    if [ -d "$SOURCE_MARKETPLACE/plugins/$plugin" ]; then
        log_info "  Copying: $plugin"
        cp -R "$SOURCE_MARKETPLACE/plugins/$plugin" "$MARKETPLACE_ROOT/plugins/"
        ((COPIED_COUNT++))
    else
        log_warning "  Plugin not found (skipping): $plugin"
    fi
done
log_success "$COPIED_COUNT plugins copied"
echo ""

# Phase 3: Create custom-scientific-computing plugin
log_info "Phase 3: Creating custom-scientific-computing plugin..."
CUSTOM_PLUGIN="$MARKETPLACE_ROOT/plugins/custom-scientific-computing"
mkdir -p "$CUSTOM_PLUGIN"/{agents,commands}

# Copy custom agents (if they exist)
if [ -d "$HOME/.claude/agents" ]; then
    SCIENTIFIC_AGENTS=(
        "hpc-numerical-coordinator.md"
        "jax-pro.md"
        "jax-scientific-domains.md"
        "neural-architecture-engineer.md"
        "correlation-function-expert.md"
        "simulation-expert.md"
        "scientific-code-adoptor.md"
        "visualization-interface-master.md"
        "research-intelligence-master.md"
        "command-systems-engineer.md"
        "database-workflow-engineer.md"
        "multi-agent-orchestrator.md"
    )

    AGENT_COUNT=0
    for agent in "${SCIENTIFIC_AGENTS[@]}"; do
        if [ -f "$HOME/.claude/agents/$agent" ]; then
            cp "$HOME/.claude/agents/$agent" "$CUSTOM_PLUGIN/agents/"
            ((AGENT_COUNT++))
        fi
    done
    log_success "  Copied $AGENT_COUNT custom agents"
else
    log_warning "  No custom agents directory found"
fi

# Copy and rename custom commands (if they exist)
if [ -d "$HOME/.claude/commands" ]; then
    # Use parallel arrays to avoid associative array issues
    declare -a SOURCE_COMMANDS=(
        "multi-agent-optimize.md"
        "explain-code.md"
        "generate-tests.md"
        "ci-setup.md"
        "adopt-code.md"
        "double-check.md"
        "fix-commit-errors.md"
        "fix-imports.md"
        "lint-plugins.md"
        "run-all-tests.md"
        "ultra-think.md"
        "reflection.md"
        "update-claudemd.md"
        "update-docs.md"
    )

    declare -a TARGET_COMMANDS=(
        "sci-multi-optimize.md"
        "explain-scientific.md"
        "sci-test-gen.md"
        "sci-ci-setup.md"
        "adopt-code.md"
        "double-check.md"
        "fix-commit-errors.md"
        "fix-imports.md"
        "lint-plugins.md"
        "run-all-tests.md"
        "ultra-think.md"
        "reflection.md"
        "update-claudemd.md"
        "update-docs.md"
    )

    COMMAND_COUNT=0
    for i in "${!SOURCE_COMMANDS[@]}"; do
        source="${SOURCE_COMMANDS[$i]}"
        target="${TARGET_COMMANDS[$i]}"
        if [ -f "$HOME/.claude/commands/$source" ]; then
            cp "$HOME/.claude/commands/$source" "$CUSTOM_PLUGIN/commands/$target"
            ((COMMAND_COUNT++))
        fi
    done
    log_success "  Copied $COMMAND_COUNT custom commands (with renames)"
else
    log_warning "  No custom commands directory found"
fi

log_success "Custom scientific-computing plugin created"
echo ""

# Phase 4: Copy LICENSE from source
log_info "Phase 4: Copying LICENSE..."
if [ -f "$SOURCE_MARKETPLACE/LICENSE" ]; then
    cp "$SOURCE_MARKETPLACE/LICENSE" "$MARKETPLACE_ROOT/"
    log_success "LICENSE copied"
else
    log_warning "Source LICENSE not found (you may need to create one)"
fi
echo ""

# Phase 5: Generate .gitignore
log_info "Phase 5: Creating .gitignore..."
cat > "$MARKETPLACE_ROOT/.gitignore" <<'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary files
*.tmp
*.temp
.cache/

# Personal notes
NOTES.md
TODO.md
EOF
log_success ".gitignore created"
echo ""

# Phase 6: Initialize git repository
log_info "Phase 6: Initializing git repository..."
cd "$MARKETPLACE_ROOT"

if [ -d ".git" ]; then
    log_warning "Git repository already exists (skipping init)"
else
    git init
    git add .
    git commit -m "Initial commit: Scientific computing marketplace

- 16 selected plugins from claude-code-workflows
- Custom scientific-computing plugin with HPC/JAX/ML agents
- Renamed conflicting commands (multi-agent-optimize → sci-multi-optimize, etc.)
- Preserved unique scientific computing agents

🤖 Generated with Claude Code"
    log_success "Git repository initialized"
fi
echo ""

# Phase 7: Register marketplace with Claude Code
log_info "Phase 7: Registering marketplace with Claude Code..."
CLAUDE_MARKETPLACES="$HOME/.claude/plugins/marketplaces"
TARGET_LINK="$CLAUDE_MARKETPLACES/$MARKETPLACE_NAME"

mkdir -p "$CLAUDE_MARKETPLACES"

if [ -L "$TARGET_LINK" ]; then
    log_warning "Removing existing symlink..."
    rm "$TARGET_LINK"
fi

ln -s "$MARKETPLACE_ROOT" "$TARGET_LINK"
log_success "Marketplace registered at: $TARGET_LINK"
echo ""

echo "========================================="
echo " Setup Complete!"
echo "========================================="
echo ""
echo "✓ Marketplace created: $MARKETPLACE_ROOT"
echo "✓ $COPIED_COUNT plugins copied"
echo "✓ Custom scientific-computing plugin created"
echo "✓ Git repository initialized"
echo "✓ Registered as: $MARKETPLACE_NAME"
echo ""
echo "Next steps:"
echo "1. Review plugins in: $MARKETPLACE_ROOT/plugins/"
echo "2. Generate marketplace.json: ./generate-metadata.sh"
echo "3. Restart Claude Code"
echo "4. List plugins: /plugin list"
echo "5. Install your custom plugin: /plugin install custom-scientific-computing"
echo ""
echo "Note: You still need to generate marketplace.json metadata."
echo "Run: cd $MARKETPLACE_ROOT && ./scripts/generate-metadata.sh"
