#!/bin/bash
# Generate marketplace.json from existing plugins

set -euo pipefail

MARKETPLACE_ROOT="/Users/b80985/Projects/MyClaude"
METADATA_FILE="$MARKETPLACE_ROOT/.claude-plugin/marketplace.json"
SOURCE_MARKETPLACE="$HOME/.claude/plugins/marketplaces/claude-code-workflows/.claude-plugin/marketplace.json"

echo "Generating marketplace metadata..."

# Read source marketplace metadata for reference
if [ ! -f "$SOURCE_MARKETPLACE" ]; then
    echo "Warning: Source marketplace.json not found"
    SOURCE_VERSION="1.2.2"
else
    SOURCE_VERSION=$(jq -r '.metadata.version' "$SOURCE_MARKETPLACE" 2>/dev/null || echo "1.2.2")
fi

# Start building marketplace.json
cat > "$METADATA_FILE" <<EOF
{
  "name": "scientific-computing-workflows",
  "owner": {
    "name": "$(git config user.name 2>/dev/null || echo 'Your Name')",
    "email": "$(git config user.email 2>/dev/null || echo 'your.email@example.com')",
    "url": "https://github.com/$(git config user.name 2>/dev/null || echo 'yourusername')"
  },
  "metadata": {
    "description": "Custom marketplace for scientific computing workflows with specialized HPC/JAX/ML agents and selected claude-code-workflows plugins",
    "version": "0.1.0",
    "based_on": "claude-code-workflows v${SOURCE_VERSION}"
  },
  "plugins": [
EOF

# Function to extract plugin metadata from source
extract_plugin_metadata() {
    local plugin_name="$1"
    local source_json="$SOURCE_MARKETPLACE"

    # Extract plugin metadata from source marketplace.json
    jq -r --arg name "$plugin_name" '.plugins[] | select(.name == $name)' "$source_json" 2>/dev/null || echo "{}"
}

# Process each plugin directory
FIRST=true
for plugin_dir in "$MARKETPLACE_ROOT/plugins"/*/; do
    plugin_name=$(basename "$plugin_dir")

    echo "  Processing: $plugin_name"

    # Check if this is a source plugin or custom plugin
    if [ "$plugin_name" != "custom-scientific-computing" ]; then
        # Get metadata from source marketplace.json
        if [ -f "$SOURCE_MARKETPLACE" ]; then
            plugin_json=$(jq -r --arg name "$plugin_name" '.plugins[] | select(.name == $name)' "$SOURCE_MARKETPLACE" 2>/dev/null)

            if [ -n "$plugin_json" ] && [ "$plugin_json" != "null" ]; then
                # Add version suffix to indicate custom version
                plugin_json=$(echo "$plugin_json" | jq '.version += "-custom"')

                # Add comma separator if not first
                if [ "$FIRST" = false ]; then
                    echo "," >> "$METADATA_FILE"
                fi
                FIRST=false

                echo "$plugin_json" | jq '.' >> "$METADATA_FILE"
            fi
        fi
    else
        # Custom plugin - build metadata from scratch
        if [ "$FIRST" = false ]; then
            echo "," >> "$METADATA_FILE"
        fi
        FIRST=false

        # Count agents and commands
        AGENT_FILES=$(find "$plugin_dir/agents" -name "*.md" 2>/dev/null | sort)
        COMMAND_FILES=$(find "$plugin_dir/commands" -name "*.md" 2>/dev/null | sort)

        cat >> "$METADATA_FILE" <<CUSTOM_EOF
    {
      "name": "custom-scientific-computing",
      "source": "./plugins/custom-scientific-computing",
      "description": "Specialized agents for HPC, JAX, ML, and scientific computing workflows with custom commands for scientific development",
      "version": "0.1.0",
      "author": {
        "name": "$(git config user.name 2>/dev/null || echo 'Your Name')",
        "url": "https://github.com/$(git config user.name 2>/dev/null || echo 'yourusername')"
      },
      "homepage": "https://github.com/$(git config user.name 2>/dev/null || echo 'yourusername')/scientific-computing-workflows",
      "repository": "https://github.com/$(git config user.name 2>/dev/null || echo 'yourusername')/scientific-computing-workflows",
      "license": "MIT",
      "keywords": [
        "scientific-computing",
        "hpc",
        "jax",
        "machine-learning",
        "numerical-methods",
        "simulation",
        "custom"
      ],
      "category": "ai-ml",
      "strict": false,
      "commands": [
CUSTOM_EOF

        # Add command files
        FIRST_CMD=true
        for cmd_file in $COMMAND_FILES; do
            if [ "$FIRST_CMD" = false ]; then
                echo "," >> "$METADATA_FILE"
            fi
            FIRST_CMD=false

            cmd_path="./commands/$(basename "$cmd_file")"
            echo -n "        \"$cmd_path\"" >> "$METADATA_FILE"
        done

        cat >> "$METADATA_FILE" <<CUSTOM_EOF2

      ],
      "agents": [
CUSTOM_EOF2

        # Add agent files
        FIRST_AGENT=true
        for agent_file in $AGENT_FILES; do
            if [ "$FIRST_AGENT" = false ]; then
                echo "," >> "$METADATA_FILE"
            fi
            FIRST_AGENT=false

            agent_path="./agents/$(basename "$agent_file")"
            echo -n "        \"$agent_path\"" >> "$METADATA_FILE"
        done

        cat >> "$METADATA_FILE" <<CUSTOM_EOF3

      ]
    }
CUSTOM_EOF3
    fi
done

# Close plugins array and JSON
cat >> "$METADATA_FILE" <<'EOF'
  ]
}
EOF

echo ""
echo "✓ Marketplace metadata generated: $METADATA_FILE"
echo ""
echo "Metadata summary:"
jq -r '.plugins[] | "  - \(.name) (v\(.version))"' "$METADATA_FILE"
echo ""
echo "Total plugins: $(jq '.plugins | length' "$METADATA_FILE")"
