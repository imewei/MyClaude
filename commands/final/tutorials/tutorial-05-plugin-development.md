# Tutorial 05: Plugin Development

**Duration**: 60 minutes | **Level**: Intermediate | **Prerequisites**: Tutorials 01-04

---

## Learning Objectives

- Understand plugin architecture
- Install and use existing plugins
- Create custom command plugins
- Build custom agent plugins
- Publish plugins to the community

---

## Part 1: Using Existing Plugins (15 minutes)

### Discover Plugins
```bash
# List available plugins
/plugin list

# Search for specific functionality
/plugin search linter
/plugin search custom-analyzer
```

### Install Plugin
```bash
# Install from registry
/plugin install code-complexity-analyzer

# Verify installation
/plugin list --installed

# Use plugin command
/analyze-complexity src/
```

---

## Part 2: Creating a Custom Command Plugin (20 minutes)

### Step 1: Create Plugin Structure
```bash
# Generate plugin template
/plugin create my-custom-linter --type=command

# Creates structure:
# my-custom-linter/
# ├── plugin.py
# ├── config.yaml
# ├── README.md
# └── tests/
```

### Step 2: Implement Plugin
**plugin.py**:
```python
from plugins.core import CommandPlugin

class MyCustomLinter(CommandPlugin):
    name = "custom-linter"
    version = "1.0.0"

    def execute(self, context):
        # Custom linting logic
        files = context.get_files("*.py")
        issues = []

        for file in files:
            issues.extend(self.lint_file(file))

        return {
            "issues": issues,
            "score": self.calculate_score(issues)
        }

    def lint_file(self, file):
        # Your custom linting rules
        issues = []
        # Check for TODO comments
        # Check for long functions
        # Check custom patterns
        return issues
```

### Step 3: Test Plugin
```bash
# Test locally
/plugin test ./my-custom-linter

# Install locally for testing
/plugin install ./my-custom-linter --dev

# Use the plugin
/custom-linter src/
```

---

## Part 3: Creating an Agent Plugin (15 minutes)

### Create Custom Agent
```python
from plugins.core import AgentPlugin

class DomainExpertAgent(AgentPlugin):
    name = "domain-expert"
    category = "domain-specific"

    def analyze(self, code, context):
        # Domain-specific analysis
        insights = {
            "domain_patterns": self.find_patterns(code),
            "best_practices": self.check_practices(code),
            "recommendations": self.generate_recommendations(code)
        }
        return insights

    def find_patterns(self, code):
        # Detect domain-specific patterns
        pass
```

### Register Agent
```yaml
# config.yaml
plugin:
  type: agent
  name: domain-expert-agent

agent:
  category: domain-specific
  priority: high
  capabilities:
    - pattern-recognition
    - domain-analysis
    - best-practices
```

---

## Part 4: Publishing Plugins (10 minutes)

### Prepare for Publishing
```bash
# Add metadata
/plugin metadata my-custom-linter \
  --description="Custom linting rules" \
  --author="Your Name" \
  --license="MIT"

# Run quality checks
/check-code-quality my-custom-linter/
/generate-tests my-custom-linter/ --coverage=90
```

### Publish
```bash
# Package plugin
/plugin package my-custom-linter

# Publish to registry
/plugin publish my-custom-linter \
  --registry=community \
  --tags="linter,code-quality"
```

---

## Practice Projects

**Project 1**: Create a custom formatter plugin
**Project 2**: Build a project-specific analyzer agent
**Project 3**: Integrate with external tool (ESLint, Prettier)

---

## Summary

✅ Plugin architecture mastered
✅ Custom command plugin created
✅ Agent plugin implemented
✅ Publishing process learned

**Next**: [Tutorial 06: Agent System →](tutorial-06-agent-orchestration.md)