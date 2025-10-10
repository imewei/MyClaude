---
description: Smart git commit with conventional commit format and atomic commit guidance
allowed-tools: Bash(git:*)
argument-hint: [commit-message] [--no-verify]
color: green
agents:
  primary:
    - code-quality
  conditional:
    - agent: devops-security-engineer
      trigger: files "ci/|.github/|Dockerfile"
  orchestrated: false
---

# Smart Git Commit

## Context
- Staged files: !`git diff --cached --name-only 2>/dev/null | wc -l` files
- Unstaged changes: !`git diff --name-only 2>/dev/null | wc -l` files
- Git status: !`git status --short 2>/dev/null | head -10`
- Recent commits: !`git log --oneline -3 2>/dev/null`

## Your Task: $ARGUMENTS

**Create conventional commit following these guidelines**:

### 1. Pre-Commit Checks
```bash
# Run automatically (skip with --no-verify)
npm run lint  # or appropriate linter
npm test      # if tests exist
npm run build # verify build succeeds
```

### 2. Auto-Stage Analysis
If no files staged:
- Analyze all changed files
- Suggest logical commit groupings
- Recommend splitting if >5 unrelated files changed

### 3. Conventional Commit Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types** (with emojis for reference):
- `feat`: ✨ New feature
- `fix`: 🐛 Bug fix
- `docs`: 📝 Documentation
- `style`: 💄 Formatting, no code change
- `refactor`: ♻️ Code restructure, no behavior change
- `perf`: ⚡ Performance improvement
- `test`: ✅ Add/update tests
- `chore`: 🔧 Maintenance, tooling
- `ci`: 👷 CI/CD changes
- `build`: 📦 Build system changes

**Examples**:
```
feat(auth): add OAuth2 login flow
fix(api): resolve race condition in user creation
docs: update installation guide for v2.0
refactor: extract validation logic to utils
```

### 4. Commit Message Guidelines
**Subject** (50 chars max):
- Use imperative mood ("add" not "added")
- No period at end
- Lowercase after colon
- Clear and concise

**Body** (72 chars per line):
- Explain WHY, not WHAT (code shows what)
- Include context and reasoning
- Reference issues: "Fixes #123"

**DON'T**:
- Mention "Claude" or AI assistance
- Use flowery or marketing language
- Add extraneous details
- Be vague ("fix stuff", "update code")

**DO**:
- Be specific and factual
- Focus on the change itself
- Use straightforward language

### 5. Atomic Commit Check
**One commit should**:
- Address one logical change
- Pass all tests independently
- Be independently revertible

**Split if**:
- Multiple unrelated bug fixes
- Feature + refactor
- Multiple modules changed
- >300 lines changed

### 6. Execution
```bash
# If no files staged, auto-stage related files
git add <files>

# Create commit
git commit -m "type(scope): subject" -m "body"

# Verify
git show HEAD --stat
```

**Create clean, atomic commit with clear conventional format**
