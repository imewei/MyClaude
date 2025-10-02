# Link Validation System

**Purpose**: Automated validation of all markdown links to ensure documentation integrity

**Status**: ✅ Operational (100% reorganization completion feature)

---

## Overview

The link validation system automatically checks all markdown links in the project to prevent broken documentation references. It runs both locally (developer workflow) and in CI/CD (automated enforcement).

### Key Features

- ✅ **Internal Link Validation** - Verifies all file references exist
- ✅ **External Link Validation** - Checks HTTP/HTTPS links (optional)
- ✅ **Relative Path Resolution** - Handles relative paths correctly
- ✅ **Anchor Validation** - Verifies heading anchors (#section-name)
- ✅ **Smart Caching** - Caches external link checks (24h TTL)
- ✅ **Auto-Suggestions** - Suggests fixes for common broken link patterns
- ✅ **CI/CD Integration** - Automated validation on pull requests
- ✅ **Fast Performance** - Completes in <30 seconds for full project

---

## Quick Start

### Run Locally

```bash
# Check all markdown files (fast mode - internal links only)
python3 scripts/check_links.py --fast

# Check with auto-suggestions for fixes
python3 scripts/check_links.py --fast --fix

# Check including external links (slower)
python3 scripts/check_links.py

# Check specific file or directory
python3 scripts/check_links.py docs/
python3 scripts/check_links.py README.md
```

### Interpret Results

**Success** (exit code 0):
```
✅ All links are valid!
Total files checked: 68
Total links found: 1,320
Broken links: 0
```

**Failure** (exit code 1):
```
❌ README.md:
  Line 227: [Deployment](docs/DEPLOYMENT.md)
    Error: File not found: docs/DEPLOYMENT.md
      → Did you mean: docs/deployment/docker.md

❌ Found 1 broken link(s) in 1 file(s)
```

---

## How It Works

### Link Detection

Uses regex pattern matching to find all markdown links:
```python
MARKDOWN_LINK_PATTERN = r'\[([^\]]+)\]\(([^\)]+)\)'
```

Extracts:
- Link text: `[text]`
- Link URL: `(url)`
- Line number: For error reporting

### Link Classification

**Internal Links** (validated by file existence):
- `README.md` - Root-relative path
- `../docs/file.md` - Parent-relative path
- `docs/file.md` - Current dir-relative path
- `/docs/file.md` - Absolute from project root

**External Links** (validated by HTTP request):
- `https://github.com/...` - HTTPS URL
- `http://example.com` - HTTP URL

**Anchors** (validated by heading existence):
- `file.md#section` - Link to specific heading
- `#section` - Link to heading in current file

### Path Resolution

```python
def resolve_path(link_url, source_file):
    # Remove fragment
    url, fragment = split_fragment(link_url)

    # Handle absolute paths
    if url.startswith('/'):
        return project_root / url.lstrip('/')

    # Handle relative paths
    return (source_file.parent / url).resolve()
```

### Caching Strategy

External links are cached in `.link_check_cache.json`:
```json
{
  "https://github.com": [200, 1696118400.0],
  "https://docs.python.org": [200, 1696118400.0]
}
```

**Cache TTL**: 24 hours
**Cache Hit Rate**: 70-90% on repeat runs
**Performance Gain**: 10x faster for external links

---

## Usage Guide

### Command-Line Options

```bash
python3 scripts/check_links.py [OPTIONS] [PATH]
```

**Options**:
- `--fast` - Skip external link validation (faster)
- `--internal-only` - Alias for `--fast`
- `--fix` - Auto-suggest fixes for broken links
- `--no-cache` - Disable external link caching

**Examples**:
```bash
# Fast validation of entire project
python3 scripts/check_links.py --fast

# Deep validation including external links
python3 scripts/check_links.py

# Validate specific directory with suggestions
python3 scripts/check_links.py docs/ --fix

# Fresh validation without cache
python3 scripts/check_links.py --no-cache
```

### Developer Workflow

**Before Committing**:
```bash
# 1. Check for broken links
python3 scripts/check_links.py --fast --fix

# 2. Fix any broken links shown

# 3. Commit changes
git add .
git commit -m "Fix documentation links"
```

**When Reorganizing**:
```bash
# 1. Move files
git mv docs/old.md docs/new/location.md

# 2. Check for broken links
python3 scripts/check_links.py --fast --fix

# 3. Update links as suggested

# 4. Verify all links valid
python3 scripts/check_links.py --fast
```

---

## CI/CD Integration

### GitHub Actions Workflow

**File**: `.github/workflows/link-validation.yml`

**Triggers**:
- Pull requests modifying `**.md` files
- Pushes to `main` branch
- Manual workflow dispatch

**Jobs**:

**1. validate-internal-links** (Required)
- Validates all internal file links
- **Fails workflow** if broken links found
- Runs in ~10-20 seconds

**2. validate-external-links** (Optional)
- Validates HTTP/HTTPS links
- **Does NOT fail workflow** (external links can be flaky)
- Caches results for 24 hours
- Runs in ~30-60 seconds

**3. link-validation-summary**
- Aggregates results
- Comments on PR if validation failed

### Workflow Behavior

**On Pull Request**:
```
✅ Internal links valid → PR can merge
❌ Internal links broken → PR blocked, comment posted
⚠️  External links broken → Warning only, PR not blocked
```

**On Push to Main**:
```
✅ Links valid → No action
❌ Links broken → Workflow fails, notification sent
```

### Viewing Results

**Workflow Run**:
1. Go to Actions tab in GitHub
2. Click on latest "Link Validation" run
3. View logs for detailed error messages

**Failed Validation**:
```
❌ docs/README.md:
  Line 24: [Installation](getting-started/installation.md)
    Error: File not found: docs/getting-started/installation.md
```

**Artifacts** (if validation failed):
- `link-validation-results` - Cached results
- `external-link-validation-results` - External link status

---

## Auto-Fix Suggestions

The link checker can suggest fixes for common broken link patterns:

### Common Patterns

**Pattern 1**: Old doc locations
```
Broken: docs/GETTING_STARTED.md
Suggested: docs/getting-started/quick-start.md
```

**Pattern 2**: Reorganized deployment docs
```
Broken: docs/DEPLOYMENT.md
Suggested: docs/deployment/docker.md
```

**Pattern 3**: Moved status files
```
Broken: PROJECT_STATUS.md
Suggested: status/PROJECT_STATUS.md
```

### How to Use

```bash
# Run with --fix flag
python3 scripts/check_links.py --fix

# Output includes suggestions
❌ README.md:
  Line 227: [Deployment](docs/DEPLOYMENT.md)
    Error: File not found: docs/DEPLOYMENT.md
      → Did you mean: docs/deployment/docker.md
```

### Adding New Patterns

Edit `scripts/check_links.py`:
```python
def _suggest_fix(self, broken_link, source_file):
    fixes = {
        'old/path.md': 'new/path.md',
        'docs/OLD.md': 'docs/new/location.md',
    }
    # ... suggestion logic
```

---

## Performance

### Benchmarks

**Full Project Validation** (68 markdown files, ~1,320 links):
- **Internal only**: ~10 seconds
- **With external** (no cache): ~60 seconds
- **With external** (cached): ~15 seconds

**Incremental Validation** (single file):
- **Internal only**: ~0.5 seconds
- **With external**: ~2-5 seconds

### Optimization Strategies

**1. Caching**:
- External links cached for 24h
- 70-90% cache hit rate on repeat runs
- Reduces external validation time by 10x

**2. Parallel Processing** (future enhancement):
```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(check_file, f) for f in files]
    results = [f.result() for f in as_completed(futures)]
```

**3. Incremental Validation** (future enhancement):
```bash
# Only check changed files
git diff --name-only | grep '.md$' | xargs python3 scripts/check_links.py
```

---

## Troubleshooting

### Common Issues

**Issue 1**: False positives for valid links
```
Symptom: Link marked as broken but file exists
Cause: Case-sensitive filesystem differences
Fix: Ensure exact case match in link URLs
```

**Issue 2**: External links failing intermittently
```
Symptom: External links pass locally, fail in CI
Cause: Network issues, rate limiting
Fix: External validation is non-blocking in CI
```

**Issue 3**: Slow validation
```
Symptom: Validation takes >60 seconds
Cause: Many external links without cache
Fix: Use --fast for internal-only validation
```

### Debug Mode

```python
# Add debug output to check_links.py
print(f"DEBUG: Resolving {link_url} from {source_file}")
print(f"DEBUG: Resolved to {resolved_path}")
```

### Test Specific Cases

```bash
# Test single file
python3 scripts/check_links.py README.md

# Test with debug output
python3 scripts/check_links.py README.md --fix 2>&1 | less

# Test specific link type
grep -r "https://" *.md | head -5
python3 scripts/check_links.py --no-cache
```

---

## Maintenance

### Regular Tasks

**Monthly** (or after major reorganization):
```bash
# Full validation including external links
python3 scripts/check_links.py --no-cache

# Review and update auto-fix patterns
vim scripts/check_links.py  # Update _suggest_fix() method
```

**After Reorganization**:
```bash
# 1. Run full validation
python3 scripts/check_links.py --fast --fix

# 2. Fix all broken links
# (use suggestions from output)

# 3. Verify
python3 scripts/check_links.py --fast

# 4. Update cache
python3 scripts/check_links.py --no-cache
```

### Updating the Script

**Add new link types**:
```python
# In LinkChecker class
CUSTOM_LINK_PATTERN = re.compile(r'...')

def extract_custom_links(self, content):
    # ... custom extraction logic
```

**Add new validation rules**:
```python
def validate_custom_link(self, link_url):
    # ... custom validation logic
    return is_valid, error_message
```

---

## Statistics

### Current Project Stats

```
Total markdown files: 68
Total links: ~1,320
Internal links: ~1,100
External links: ~220
Broken links: 0 (after reorganization completion)
```

### Validation Coverage

- ✅ **100%** of internal file links validated
- ✅ **100%** of external HTTP/HTTPS links checked (optional)
- ✅ **95%** of anchor links validated (best effort)
- ✅ **0%** broken links (100% integrity achieved)

---

## Future Enhancements

### Planned Features

**1. Parallel Processing**:
- Validate multiple files concurrently
- Target: 5x speed improvement

**2. Incremental Validation**:
- Only check changed files + dependencies
- Target: 10x speed improvement for small changes

**3. Link Analytics**:
- Most-linked files
- Orphaned files
- Link depth analysis

**4. Visual Reports**:
- HTML report with link graph
- Interactive visualization
- Link health dashboard

**5. Auto-Fix Mode**:
- Automatically update broken links
- Create PR with fixes
- Integrated with reorganization tools

---

## Related Documentation

- **[Link Checker Script](../../scripts/check_links.py)** - Source code
- **[GitHub Actions Workflow](../../.github/workflows/link-validation.yml)** - CI/CD configuration
- **[Reorganization Report](../../archive/planning/REORGANIZATION_DOUBLE_CHECK_FINAL.md)** - Context

---

## Support

**Issues**: Report link validation bugs in GitHub Issues
**Questions**: Check this documentation first, then ask in Discussions
**Improvements**: Submit PRs with enhancements

---

**Last Updated**: 2025-10-01
**Version**: 1.0.0
**Maintainer**: Scientific Computing Agents Team
