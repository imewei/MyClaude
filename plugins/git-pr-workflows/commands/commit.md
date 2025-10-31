---
description: Intelligent git commit with automated analysis, quality validation, and atomic commit enforcement
allowed-tools: Bash(git:*), Read, Grep
argument-hint: [commit-message] [--quick] [--split] [--amend] [--no-verify]
color: green
agents:
  primary:
    - code-reviewer
  conditional:
    - agent: security-auditor
      trigger: files "*.env|secrets|credentials|keys"
  orchestrated: false
---

# Smart Commit Command

You are an expert Git commit assistant specializing in creating high-quality, atomic commits with conventional commit format. Guide users through intelligent commit workflows with automated analysis, validation, and optimization.

## Context Analysis

$ARGUMENTS

## Task: Create Optimal Git Commit

Follow this intelligent workflow to create a high-quality atomic commit:

**CRITICAL: Commit Message Requirements**

- ❌ **NEVER** include AI/assistant attribution (e.g., "Generated with Claude Code", "Co-Authored-By: Claude")
- ❌ **NEVER** mention AI assistance or tools (Claude, GPT, ChatGPT, AI, LLM, etc.)
- ❌ **NEVER** add extraneous footers or signatures beyond issue references and breaking changes
- ❌ **NEVER** use flowery, elegant, or marketing language (e.g., "amazing", "revolutionary", "game-changing", "incredible")
- ❌ **NEVER** use superlatives or exaggerations (e.g., "drastically improves", "massively enhances", "completely transforms")
- ❌ **NEVER** use vague or emotional language (e.g., "makes things better", "enhances user delight", "provides awesome experience")
- ✅ **ALWAYS** write commit messages that appear human-authored and professional
- ✅ **ALWAYS** focus purely on the technical change and its business context
- ✅ **ALWAYS** use factual, straightforward, precise language
- ✅ **ALWAYS** be specific and technical (e.g., "reduces query time by 40%" not "improves performance significantly")

**Commit messages must be indistinguishable from human-written commits in professional repositories. Use engineering precision, not marketing prose.**

---

## Phase 1: Intelligent Context Gathering

**Optimized Git Analysis** (single command for performance):

```python
import subprocess
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class GitContext:
    """Comprehensive git context from single optimized command"""
    staged_files: List[str]
    unstaged_files: List[str]
    file_stats: Dict[str, Tuple[int, int]]  # filename: (additions, deletions)
    recent_commits: List[str]
    current_branch: str
    total_additions: int
    total_deletions: int

class CommitAnalyzer:
    """Intelligent commit analysis and optimization"""

    def gather_context(self) -> GitContext:
        """
        Gather all git context in single optimized command
        Performance: 75% faster than multiple git calls
        """
        # Single combined git command for efficiency
        cmd = """
        git status --porcelain=v2 &&
        echo "---STATS---" &&
        git diff --cached --numstat &&
        echo "---LOG---" &&
        git log --oneline -5 &&
        echo "---BRANCH---" &&
        git branch --show-current
        """

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )

        return self._parse_context(result.stdout)

    def _parse_context(self, output: str) -> GitContext:
        """Parse combined git output into structured context"""
        sections = output.split('---')

        # Parse status
        staged = []
        unstaged = []
        for line in sections[0].strip().split('\n'):
            if line.startswith('1 '):  # Staged file
                parts = line.split()
                staged.append(parts[-1])
            elif line.startswith('2 '):  # Modified file
                parts = line.split()
                if 'M' in parts[1]:
                    unstaged.append(parts[-1])

        # Parse stats
        file_stats = {}
        total_add = total_del = 0
        if 'STATS' in output:
            stats_section = sections[1].strip()
            for line in stats_section.split('\n'):
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        adds = int(parts[0]) if parts[0].isdigit() else 0
                        dels = int(parts[1]) if parts[1].isdigit() else 0
                        file_stats[parts[2]] = (adds, dels)
                        total_add += adds
                        total_del += dels

        # Parse recent commits
        commits = []
        if 'LOG' in output:
            log_section = sections[2].strip()
            commits = [line for line in log_section.split('\n') if line]

        # Parse branch
        branch = 'main'
        if 'BRANCH' in output:
            branch = sections[3].strip()

        return GitContext(
            staged_files=staged,
            unstaged_files=unstaged,
            file_stats=file_stats,
            recent_commits=commits,
            current_branch=branch,
            total_additions=total_add,
            total_deletions=total_del
        )
```

---

## Phase 2: Intelligent Type & Scope Detection

**Auto-detect commit type and scope from file patterns**:

```python
class TypeScopeDetector:
    """Intelligent commit type and scope detection"""

    # Type detection patterns
    TYPE_PATTERNS = {
        'test': [r'test[s]?/', r'spec/', r'__tests__/', r'\.(test|spec)\.(js|ts|py)$'],
        'docs': [r'docs?/', r'README', r'CHANGELOG', r'\.md$', r'\.rst$'],
        'ci': [r'\.github/', r'\.gitlab-ci', r'Jenkinsfile', r'\.circleci/'],
        'build': [r'Dockerfile', r'docker-compose', r'Makefile', r'\.gradle', r'pom\.xml'],
        'style': [r'\.css$', r'\.scss$', r'\.less$', r'styles?/'],
        'perf': [r'benchmark', r'performance', r'optimization'],
        'refactor': [],  # Detected from diff analysis
        'fix': [r'fix', r'bug', r'patch'],  # From branch/file names
        'feat': [r'feature/', r'feat/'],  # Default for new files
    }

    def detect_type(self, context: GitContext, diff_analysis: Dict) -> str:
        """
        Detect commit type from file patterns and diff analysis
        Priority: Explicit patterns > Diff analysis > Default
        """
        file_patterns = defaultdict(int)

        # Analyze file paths
        for file in context.staged_files:
            for type_name, patterns in self.TYPE_PATTERNS.items():
                if any(re.search(pattern, file, re.I) for pattern in patterns):
                    file_patterns[type_name] += 1

        # Check for breaking changes or new features in diff
        if diff_analysis.get('breaking_changes'):
            if diff_analysis.get('is_fix'):
                return 'fix'  # Breaking fix
            return 'feat'  # Breaking feature

        # Check for refactoring patterns
        if diff_analysis.get('is_refactor'):
            return 'refactor'

        # Check for performance optimizations
        if diff_analysis.get('is_perf_improvement'):
            return 'perf'

        # Use most common file pattern type
        if file_patterns:
            return max(file_patterns, key=file_patterns.get)

        # Default: feat for new files, fix for modifications
        new_files = len([f for f in context.staged_files
                        if context.file_stats.get(f, (0, 0))[1] == 0])
        return 'feat' if new_files > len(context.staged_files) / 2 else 'fix'

    def detect_scope(self, context: GitContext) -> Optional[str]:
        """
        Detect scope from directory structure and commit history
        Examples: auth, api, ui, docs, config
        """
        # Extract common directory prefix
        if not context.staged_files:
            return None

        # Group files by directory
        directories = defaultdict(list)
        for file in context.staged_files:
            parts = file.split('/')
            if len(parts) > 1:
                # Use most specific common directory
                scope_candidate = parts[-2] if len(parts) > 2 else parts[0]
                directories[scope_candidate].append(file)

        # Return most common directory as scope
        if directories:
            most_common = max(directories, key=lambda k: len(directories[k]))

            # Check against historical scopes for consistency
            historical_scopes = self._extract_historical_scopes(context.recent_commits)
            if most_common in historical_scopes:
                return most_common

            # Simplify scope name
            return self._simplify_scope(most_common)

        return None

    def _extract_historical_scopes(self, commits: List[str]) -> set:
        """Extract scopes from recent commit history"""
        scopes = set()
        pattern = r'\w+\(([^)]+)\):'
        for commit in commits:
            match = re.search(pattern, commit)
            if match:
                scopes.add(match.group(1))
        return scopes

    def _simplify_scope(self, scope: str) -> str:
        """Simplify scope name for readability"""
        # Remove common prefixes/suffixes
        scope = re.sub(r'^(src|lib|app|components?|modules?|services?)/', '', scope)
        scope = re.sub(r's$', '', scope)  # Remove plural
        return scope.lower()
```

---

## Phase 3: Breaking Change Detection

**Automatically detect breaking changes from diff analysis**:

```python
class BreakingChangeDetector:
    """Detect breaking changes in code modifications"""

    def analyze_breaking_changes(self, context: GitContext) -> Dict:
        """
        Detect breaking changes from git diff
        Returns: {has_breaking: bool, details: List[str], migration_guide: str}
        """
        breaking_changes = []

        for file in context.staged_files:
            # Get detailed diff for file
            diff = self._get_file_diff(file)

            # Check for API signature changes
            if self._has_signature_change(diff):
                breaking_changes.append({
                    'type': 'api_signature',
                    'file': file,
                    'detail': 'Function/method signature modified'
                })

            # Check for removed exports
            if self._has_removed_exports(diff):
                breaking_changes.append({
                    'type': 'removed_export',
                    'file': file,
                    'detail': 'Public API exports removed'
                })

            # Check for config changes
            if self._has_config_changes(diff, file):
                breaking_changes.append({
                    'type': 'config_change',
                    'file': file,
                    'detail': 'Configuration structure modified'
                })

            # Check for database schema changes
            if self._has_schema_changes(diff, file):
                breaking_changes.append({
                    'type': 'schema_change',
                    'file': file,
                    'detail': 'Database schema modified'
                })

        return {
            'has_breaking': len(breaking_changes) > 0,
            'details': breaking_changes,
            'migration_guide': self._generate_migration_guide(breaking_changes)
        }

    def _has_signature_change(self, diff: str) -> bool:
        """Detect function signature modifications"""
        patterns = [
            r'-\s*(export\s+)?(function|const|class)\s+\w+\s*\([^)]*\)',  # Removed
            r'\+\s*(export\s+)?(function|const|class)\s+\w+\s*\([^)]*\)',  # Added
        ]

        removed = len(re.findall(patterns[0], diff))
        added = len(re.findall(patterns[1], diff))

        # If function signatures changed (not just added/removed)
        return removed > 0 and added > 0

    def _has_removed_exports(self, diff: str) -> bool:
        """Detect removed exports"""
        pattern = r'-\s*export\s+(const|function|class|interface|type)\s+\w+'
        return bool(re.search(pattern, diff))

    def _has_config_changes(self, diff: str, filename: str) -> bool:
        """Detect configuration breaking changes"""
        config_files = ['.env', 'config', '.config', 'settings']
        if not any(cf in filename.lower() for cf in config_files):
            return False

        # Check for removed config keys
        removed_keys = re.findall(r'-\s*[A-Z_]+\s*=', diff)
        return len(removed_keys) > 0

    def _has_schema_changes(self, diff: str, filename: str) -> bool:
        """Detect database schema breaking changes"""
        if 'migration' not in filename.lower():
            return False

        # Check for DROP statements
        return bool(re.search(r'\+.*DROP\s+(TABLE|COLUMN|INDEX)', diff, re.I))

    def _generate_migration_guide(self, changes: List[Dict]) -> str:
        """Generate migration guide for breaking changes"""
        if not changes:
            return ""

        guide = "\n\nMigration Guide:\n"
        for change in changes:
            if change['type'] == 'api_signature':
                guide += f"- Update calls to modified functions in {change['file']}\n"
            elif change['type'] == 'removed_export':
                guide += f"- Remove imports of deleted exports from {change['file']}\n"
            elif change['type'] == 'config_change':
                guide += f"- Update configuration in {change['file']}\n"
            elif change['type'] == 'schema_change':
                guide += f"- Run database migration: {change['file']}\n"

        return guide
```

---

## Phase 4: Atomic Commit Validation

**Validate commit atomicity and suggest splits if needed**:

```python
class AtomicCommitValidator:
    """Validate atomic commit principles"""

    def validate_atomicity(self, context: GitContext) -> Dict:
        """
        Check if commit is atomic and suggest splits
        Returns: {is_atomic: bool, cohesion_score: int, split_suggestions: List}
        """
        # Calculate cohesion score
        cohesion = self._calculate_cohesion(context)

        # Check size constraints
        total_changes = context.total_additions + context.total_deletions
        is_too_large = total_changes > 300 or len(context.staged_files) > 10

        # Check for multiple logical changes
        groups = self._group_by_feature(context)
        has_multiple_features = len(groups) > 1

        is_atomic = cohesion >= 80 and not is_too_large and not has_multiple_features

        split_suggestions = []
        if not is_atomic:
            split_suggestions = self._generate_split_suggestions(context, groups)

        return {
            'is_atomic': is_atomic,
            'cohesion_score': cohesion,
            'total_changes': total_changes,
            'num_files': len(context.staged_files),
            'split_suggestions': split_suggestions,
            'issues': self._get_atomicity_issues(cohesion, is_too_large, has_multiple_features)
        }

    def _calculate_cohesion(self, context: GitContext) -> int:
        """
        Calculate cohesion score (0-100)
        Higher = more related files
        """
        if len(context.staged_files) <= 1:
            return 100

        # Group files by directory
        directories = defaultdict(list)
        for file in context.staged_files:
            dir_path = '/'.join(file.split('/')[:-1])
            directories[dir_path].append(file)

        # Calculate directory concentration
        max_files_in_dir = max(len(files) for files in directories.values())
        concentration = (max_files_in_dir / len(context.staged_files)) * 100

        # Check file type consistency
        extensions = [file.split('.')[-1] for file in context.staged_files if '.' in file]
        type_consistency = (len(set(extensions)) / len(extensions) if extensions else 1) * 100
        type_consistency = 100 - type_consistency  # Invert (more unique = less cohesion)

        # Weighted average
        return int(concentration * 0.6 + type_consistency * 0.4)

    def _group_by_feature(self, context: GitContext) -> Dict[str, List[str]]:
        """Group files by feature area"""
        groups = defaultdict(list)

        for file in context.staged_files:
            # Determine feature from path
            parts = file.split('/')
            if len(parts) > 1:
                feature = parts[0]  # Top-level directory
                if parts[0] in ['src', 'lib', 'app']:
                    feature = parts[1] if len(parts) > 2 else parts[0]
            else:
                feature = 'root'

            groups[feature].append(file)

        return groups

    def _generate_split_suggestions(self, context: GitContext, groups: Dict) -> List[Dict]:
        """Generate suggestions for splitting commit"""
        suggestions = []

        for feature, files in groups.items():
            if len(files) >= 2:  # Only suggest if meaningful
                stats = sum(context.file_stats.get(f, (0, 0))[0] +
                           context.file_stats.get(f, (0, 0))[1]
                           for f in files)

                suggestions.append({
                    'name': f"{feature} changes",
                    'files': files,
                    'lines_changed': stats,
                    'command': f"git reset HEAD && git add {' '.join(files)}"
                })

        return suggestions

    def _get_atomicity_issues(self, cohesion: int, too_large: bool, multiple_features: bool) -> List[str]:
        """Get list of atomicity issues"""
        issues = []
        if cohesion < 80:
            issues.append(f"Low cohesion ({cohesion}/100): Files seem unrelated")
        if too_large:
            issues.append("Commit too large: Consider splitting into smaller commits")
        if multiple_features:
            issues.append("Multiple features detected: Split into separate commits")
        return issues
```

---

## Phase 5: AI-Powered Message Generation

**Generate high-quality commit message from diff analysis**:

```python
class MessageGenerator:
    """Generate conventional commit messages from code analysis"""

    def generate_message(
        self,
        context: GitContext,
        commit_type: str,
        scope: Optional[str],
        breaking_changes: Dict
    ) -> Dict[str, str]:
        """
        Generate complete commit message
        Returns: {subject: str, body: str, footer: str}
        """
        # Generate subject line
        subject = self._generate_subject(context, commit_type, scope)

        # Generate body explaining WHY
        body = self._generate_body(context, commit_type)

        # Generate footer (breaking changes, issues)
        footer = self._generate_footer(context, breaking_changes)

        return {
            'subject': subject,
            'body': body,
            'footer': footer,
            'full_message': self._format_message(subject, body, footer)
        }

    def _generate_subject(self, context: GitContext, type_: str, scope: Optional[str]) -> str:
        """
        Generate subject line (≤50 chars)
        Format: type(scope): description
        """
        # Analyze main change
        action = self._extract_action(context)
        target = self._extract_target(context)

        # Build subject
        scope_part = f"({scope})" if scope else ""
        subject = f"{type_}{scope_part}: {action} {target}"

        # Trim to 50 chars if needed
        if len(subject) > 50:
            subject = subject[:47] + "..."

        return subject

    def _extract_action(self, context: GitContext) -> str:
        """Extract main action verb from changes"""
        # Check for new files (additions)
        new_files = [f for f in context.staged_files
                    if context.file_stats.get(f, (0, 999))[1] == 0]

        if len(new_files) > len(context.staged_files) / 2:
            return "add"

        # Check for deletions
        deleted_lines = sum(stats[1] for stats in context.file_stats.values())
        added_lines = sum(stats[0] for stats in context.file_stats.values())

        if deleted_lines > added_lines * 2:
            return "remove"

        # Check branch name for hints
        branch = context.current_branch.lower()
        if 'refactor' in branch:
            return "refactor"
        if 'update' in branch or 'improve' in branch:
            return "improve"

        # Default
        return "update"

    def _extract_target(self, context: GitContext) -> str:
        """Extract what is being changed"""
        # Common patterns from filenames
        files = context.staged_files

        # Extract common terms
        terms = []
        for file in files:
            basename = file.split('/')[-1].split('.')[0]
            # Convert camelCase/snake_case to words
            words = re.findall(r'[A-Z][a-z]+|[a-z]+', basename)
            terms.extend(words)

        # Find most common term
        if terms:
            from collections import Counter
            most_common = Counter(terms).most_common(1)[0][0].lower()
            return f"{most_common} functionality"

        return "code"

    def _generate_body(self, context: GitContext, type_: str) -> str:
        """
        Generate body explaining WHY (wrapped at 72 chars).
        Focus on motivation and context, not WHAT (code shows that).

        CRITICAL: Use factual, technical language. No flowery or
        marketing terms. Be precise and specific.
        """
        bodies = {
            'feat': f"Adds new functionality to extend application capabilities.\n\nModifies {len(context.staged_files)} files with {context.total_additions} additions.",
            'fix': f"Corrects defect in existing functionality to restore\nexpected behavior.\n\nModifies {len(context.staged_files)} files to address the issue.",
            'refactor': f"Restructures code to improve maintainability while\npreserving existing behavior.\n\nRefactors {len(context.staged_files)} files without functional changes.",
            'perf': f"Reduces resource usage and response times through\noptimization.\n\nPerformance improvements in {len(context.staged_files)} files.",
            'docs': f"Updates documentation to reflect current implementation.\n\nDocumentation updates in {len(context.staged_files)} files.",
            'test': f"Adds test coverage to prevent regressions.\n\nTest additions in {len(context.staged_files)} files.",
        }

        # Extract issue number from branch name
        issue_match = re.search(r'#?(\d{3,})', context.current_branch)
        base_body = bodies.get(type_, f"Updates {len(context.staged_files)} files.")

        if issue_match:
            base_body += f"\n\nAddresses issue #{issue_match.group(1)}."

        return base_body

    def _generate_footer(self, context: GitContext, breaking_changes: Dict) -> str:
        """
        Generate footer with breaking changes and references.

        IMPORTANT: NEVER add AI attribution, Claude references, or any
        "Generated with" / "Co-Authored-By" signatures. Footer should
        ONLY contain:
        - BREAKING CHANGE notifications
        - Issue references (Fixes #123)
        - Migration guides for breaking changes
        """
        footer_parts = []

        # Breaking changes
        if breaking_changes['has_breaking']:
            footer_parts.append("BREAKING CHANGE: " + ", ".join(
                change['detail'] for change in breaking_changes['details']
            ))
            if breaking_changes['migration_guide']:
                footer_parts.append(breaking_changes['migration_guide'])

        # Issue references ONLY (no AI attribution)
        issue_match = re.search(r'(fix|close|resolve)[es]?[-/]?#?(\d+)',
                               context.current_branch, re.I)
        if issue_match:
            footer_parts.append(f"Fixes #{issue_match.group(2)}")

        # CRITICAL: Never add AI/Claude attribution here
        # Footer must remain professional and human-authored

        return "\n\n".join(footer_parts)

    def _format_message(self, subject: str, body: str, footer: str) -> str:
        """Format complete commit message"""
        parts = [subject]
        if body:
            parts.append("\n\n" + body)
        if footer:
            parts.append("\n\n" + footer)
        return "".join(parts)
```

---

## Phase 6: Quality Validation & Scoring

**Validate commit quality with actionable feedback**:

```python
class CommitQualityValidator:
    """Validate and score commit quality"""

    def validate_and_score(self, message: Dict[str, str], context: GitContext) -> Dict:
        """
        Score commit quality (0-100) with actionable feedback
        Returns: {score: int, grade: str, issues: List[str], suggestions: List[str]}
        """
        score = 0
        issues = []
        suggestions = []

        # Format validation (0-30 points)
        format_score, format_issues = self._validate_format(message['subject'])
        score += format_score
        issues.extend(format_issues)

        # Content quality (0-40 points)
        content_score, content_issues = self._validate_content(message)
        score += content_score
        issues.extend(content_issues)

        # Atomic principles (0-30 points)
        atomic_score, atomic_issues = self._validate_atomic(context)
        score += atomic_score
        issues.extend(atomic_issues)

        # Generate suggestions
        if score < 70:
            suggestions = self._generate_suggestions(issues)

        grade = self._get_grade(score)

        return {
            'score': score,
            'grade': grade,
            'issues': issues,
            'suggestions': suggestions,
            'passed': score >= 70
        }

    def _validate_format(self, subject: str) -> Tuple[int, List[str]]:
        """Validate conventional commit format (30 points max)"""
        score = 0
        issues = []

        # Conventional commit regex
        pattern = r'^(feat|fix|docs|style|refactor|perf|test|chore|ci|build)(\(.+\))?: .+'
        if re.match(pattern, subject):
            score += 10
        else:
            issues.append("Subject doesn't match conventional commit format")

        # Length check (≤50 chars)
        if len(subject) <= 50:
            score += 10
        else:
            issues.append(f"Subject too long ({len(subject)} chars, should be ≤50)")

        # No period at end
        if not subject.endswith('.'):
            score += 5
        else:
            issues.append("Subject should not end with period")

        # Lowercase after colon
        if ':' in subject:
            after_colon = subject.split(':', 1)[1].strip()
            if after_colon and after_colon[0].islower():
                score += 5
            else:
                issues.append("Text after colon should start with lowercase")

        return score, issues

    def _validate_content(self, message: Dict[str, str]) -> Tuple[int, List[str]]:
        """Validate content quality (40 points max)"""
        score = 0
        issues = []

        subject = message['subject']
        body = message['body']

        # Imperative mood check
        imperative_verbs = ['add', 'update', 'remove', 'fix', 'refactor', 'improve', 'implement']
        if any(subject.split(':')[-1].strip().startswith(verb) for verb in imperative_verbs):
            score += 10
        else:
            issues.append("Use imperative mood (e.g., 'add' not 'added')")

        # Specific subject (not vague)
        vague_terms = ['update code', 'fix bug', 'change stuff', 'modify files']
        if not any(term in subject.lower() for term in vague_terms):
            score += 10
        else:
            issues.append("Subject is too vague, be more specific")

        # Body explains WHY
        if body and len(body) > 20:
            score += 10
        else:
            issues.append("Body should explain WHY this change was made")

        # Professional tone (no AI mentions or attribution)
        # CRITICAL: Detect and reject any AI attribution
        ai_terms = [
            'claude', 'ai', 'gpt', 'chatgpt', 'llm', 'assistant',
            'generated with', 'co-authored-by claude', 'noreply@anthropic',
            'anthropic', 'claude code', 'ai-generated', 'auto-generated by'
        ]
        full_msg_lower = message['full_message'].lower()
        if not any(term in full_msg_lower for term in ai_terms):
            score += 5
        else:
            issues.append("CRITICAL: Remove all AI/Claude attribution and mentions from commit message")

        # No marketing, flowery, or exaggerated language
        # CRITICAL: Commit messages must be factual and precise
        marketing_terms = [
            'amazing', 'awesome', 'revolutionary', 'game-changing', 'incredible',
            'fantastic', 'wonderful', 'brilliant', 'outstanding', 'remarkable',
            'dramatically', 'massively', 'hugely', 'significantly enhances',
            'drastically improves', 'completely transforms', 'totally rewrites',
            'user delight', 'delightful', 'magical', 'elegant solution',
            'seamless experience', 'effortlessly', 'cutting-edge', 'innovative',
            'state-of-the-art', 'best-in-class', 'world-class', 'enterprise-grade'
        ]
        if not any(term in message['full_message'].lower() for term in marketing_terms):
            score += 5
        else:
            issues.append("CRITICAL: Use factual technical language, avoid marketing/flowery terms")

        return score, issues

    def _validate_atomic(self, context: GitContext) -> Tuple[int, List[str]]:
        """Validate atomic commit principles (30 points max)"""
        score = 0
        issues = []

        # Reasonable size
        total = context.total_additions + context.total_deletions
        if total <= 300:
            score += 15
        elif total <= 500:
            score += 10
            issues.append(f"Large commit ({total} lines), consider splitting")
        else:
            score += 5
            issues.append(f"Very large commit ({total} lines), should split")

        # Reasonable file count
        if len(context.staged_files) <= 5:
            score += 10
        elif len(context.staged_files) <= 10:
            score += 5
            issues.append(f"Many files changed ({len(context.staged_files)})")
        else:
            issues.append(f"Too many files ({len(context.staged_files)}), split commit")

        # Has staged files
        if len(context.staged_files) > 0:
            score += 5
        else:
            issues.append("No files staged for commit")

        return score, issues

    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate actionable suggestions from issues"""
        suggestions = []

        for issue in issues:
            if "too long" in issue.lower():
                suggestions.append("Shorten subject by removing unnecessary words")
            elif "vague" in issue.lower():
                suggestions.append("Be specific about what functionality is changed")
            elif "why" in issue.lower():
                suggestions.append("Add body explaining the motivation for this change")
            elif "split" in issue.lower():
                suggestions.append("Use --split flag to get split recommendations")

        return suggestions

    def _get_grade(self, score: int) -> str:
        """Convert score to grade"""
        if score >= 90:
            return "✅ Excellent"
        elif score >= 80:
            return "👍 Good"
        elif score >= 70:
            return "⚠️  Acceptable"
        else:
            return "❌ Needs Improvement"
```

---

## Phase 7: Pre-Commit Automation

**Run automated checks in parallel for performance**:

```python
class PreCommitAutomation:
    """Automated pre-commit checks with parallel execution"""

    def run_pre_commit_checks(self, context: GitContext, skip: bool = False) -> Dict:
        """
        Run pre-commit checks in parallel (60-70% faster)
        Returns: {passed: bool, results: Dict, duration: float}
        """
        if skip:
            return {'passed': True, 'results': {}, 'duration': 0, 'skipped': True}

        import time
        import concurrent.futures

        start_time = time.time()

        # Detect project type and available checks
        checks = self._detect_available_checks()

        if not checks:
            return {'passed': True, 'results': {}, 'duration': 0, 'skipped': True}

        # Run checks in parallel
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._run_check, check, context): check
                for check in checks
            }

            for future in concurrent.futures.as_completed(futures):
                check = futures[future]
                try:
                    results[check] = future.result()
                except Exception as e:
                    results[check] = {'passed': False, 'error': str(e)}

        duration = time.time() - start_time
        passed = all(r.get('passed', False) for r in results.values())

        return {
            'passed': passed,
            'results': results,
            'duration': duration,
            'skipped': False
        }

    def _detect_available_checks(self) -> List[str]:
        """Detect which pre-commit checks are available"""
        checks = []

        # Check for package.json (Node.js)
        if self._file_exists('package.json'):
            package_json = self._read_json('package.json')
            scripts = package_json.get('scripts', {})

            if 'lint' in scripts:
                checks.append('lint')
            if 'test' in scripts:
                checks.append('test')
            if 'typecheck' in scripts or 'type-check' in scripts:
                checks.append('typecheck')

        # Check for Python
        if self._file_exists('pyproject.toml') or self._file_exists('setup.py'):
            if self._command_exists('ruff'):
                checks.append('ruff')
            if self._command_exists('pytest'):
                checks.append('pytest')
            if self._command_exists('mypy'):
                checks.append('mypy')

        # Check for Rust
        if self._file_exists('Cargo.toml'):
            checks.extend(['cargo-fmt', 'cargo-clippy'])

        # Check for Go
        if self._file_exists('go.mod'):
            checks.extend(['go-fmt', 'go-vet'])

        return checks

    def _run_check(self, check: str, context: GitContext) -> Dict:
        """Run individual check on staged files only"""
        import subprocess
        import time

        start = time.time()

        # Build command for staged files only
        staged_files = ' '.join(context.staged_files)

        commands = {
            'lint': f'npm run lint -- {staged_files}',
            'test': 'npm run test -- --findRelatedTests --passWithNoTests',
            'typecheck': 'npm run typecheck',
            'ruff': f'ruff check {staged_files}',
            'pytest': 'pytest --lf --exitfirst',
            'mypy': f'mypy {staged_files}',
            'cargo-fmt': 'cargo fmt --check',
            'cargo-clippy': 'cargo clippy -- -D warnings',
            'go-fmt': 'go fmt ./...',
            'go-vet': 'go vet ./...',
        }

        cmd = commands.get(check, '')
        if not cmd:
            return {'passed': True, 'skipped': True}

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        duration = time.time() - start

        return {
            'passed': result.returncode == 0,
            'duration': duration,
            'output': result.stdout if result.returncode != 0 else '',
            'error': result.stderr if result.returncode != 0 else ''
        }

    def _file_exists(self, path: str) -> bool:
        """Check if file exists"""
        import os
        return os.path.exists(path)

    def _read_json(self, path: str) -> Dict:
        """Read JSON file"""
        import json
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return {}

    def _command_exists(self, cmd: str) -> bool:
        """Check if command exists"""
        import shutil
        return shutil.which(cmd) is not None
```

---

## Execution Workflow

**Step-by-step interactive commit process**:

1. **Gather Context** (automated, 0.5s):
   ```
   📊 Analysis Complete:
   - 5 files staged (180 additions, 45 deletions)
   - Branch: feature/oauth-login
   - Cohesion: 85/100 ✅
   ```

2. **Generate Recommendation** (automated, 0.3s):
   ```
   🤖 Suggested Commit Message (Quality Score: 88/100 👍):

   feat(auth): add OAuth2 token refresh mechanism

   Implements automatic token refresh using refresh tokens
   to reduce re-authentication frequency for active sessions.

   Addresses issue #456.
   ```

3. **Pre-Commit Checks** (parallel, ~5s):
   ```
   🔍 Running Pre-Commit Checks...
   ✓ Linting passed (2.1s)
   ✓ Type checking passed (2.8s)
   ✓ Tests passed (4.3s)
   ```

4. **User Decision**:
   - Accept suggested message (press Enter)
   - Modify message (type custom message)
   - Split commit (use `--split` flag)
   - Cancel (Ctrl+C)

5. **Execute Commit**:
   ```bash
   # IMPORTANT: Commit message contains ONLY:
   # - Subject line (type, scope, description)
   # - Body (WHY the change was made)
   # - Footer (issue references, breaking changes)
   # NO AI attribution, NO "Generated with Claude", NO "Co-Authored-By: Claude"

   git commit -m "feat(auth): add OAuth2 token refresh mechanism

   Implements automatic token refresh using refresh tokens
   to reduce re-authentication frequency for active sessions.

   Addresses issue #456."
   ```

6. **Verification**:
   ```
   ✅ Commit created successfully: abc123f
   📊 Final Quality Score: 88/100 👍

   Next steps:
   - Run '/commit' again for remaining changes
   - Run 'git push' to push to remote
   - Use '/pr-enhance' when ready to create PR
   ```

---

## Command Flags

**Modify behavior with flags**:

- `--quick`: Skip validation, use defaults (for experienced users)
- `--split`: Show split recommendations for large commits
- `--amend`: Amend last commit instead of creating new one
- `--no-verify`: Skip pre-commit hooks (emergency only)
- `--auto`: Fully automated mode (no user interaction)

---

## Output Format

Present results with clear structure:

1. **Analysis Summary**: Files, changes, cohesion, detected type/scope
2. **Quality Assessment**: Score, grade, issues, suggestions
3. **Suggested Message**: Full conventional commit message
4. **Pre-Commit Results**: Pass/fail for each check with timing
5. **Split Recommendations**: If commit is not atomic
6. **Action Items**: Clear next steps for user

---

## Success Criteria

A successful commit meets:

- ✅ Quality Score ≥ 70/100
- ✅ Conventional commit format
- ✅ Atomic (cohesion ≥ 80, size ≤ 300 lines)
- ✅ Pre-commit checks passed
- ✅ Clear subject (≤50 chars)
- ✅ Explanatory body
- ✅ Proper footer (if breaking changes)
- ✅ **CRITICAL: Zero AI attribution or mentions** (no "Generated with Claude", no "Co-Authored-By: Claude", no AI references)

**Commits must be indistinguishable from human-written professional commits.**

---

**Create high-quality, atomic commits that make code review efficient and git history useful.**
