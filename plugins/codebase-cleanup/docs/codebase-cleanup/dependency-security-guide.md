# Dependency Security Guide

**Version**: 1.0.3
**Category**: codebase-cleanup
**Purpose**: Comprehensive guide for dependency security scanning, vulnerability detection, and remediation

## CVE Database Integration

### NPM Vulnerability Scanning

```javascript
const https = require('https');

async function checkNpmVulnerabilities(packageName, version) {
    const options = {
        hostname: 'registry.npmjs.org',
        path: `/-/npm/v1/security/advisories/bulk`,
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    };

    const payload = JSON.stringify({
        [packageName]: [version]
    });

    return new Promise((resolve, reject) => {
        const req = https.request(options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const advisories = JSON.parse(data);
                    resolve(advisories[packageName] || []);
                } catch (error) {
                    reject(error);
                }
            });
        });

        req.on('error', reject);
        req.write(payload);
        req.end();
    });
}
```

### Python Safety Check Integration

```python
import requests
from typing import List, Dict

def check_python_vulnerabilities(package_name: str, version: str) -> List[Dict]:
    """
    Check Python package vulnerabilities using PyPI API and Safety DB
    """
    vulnerabilities = []

    # Check PyPI for package metadata
    pypi_response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
    if pypi_response.status_code == 200:
        package_data = pypi_response.json()

        # Check Safety DB for known vulnerabilities
        safety_url = 'https://pyup.io/api/v1/safety/'
        safety_response = requests.post(safety_url, json={
            'dependencies': [{
                'name': package_name,
                'version': version
            }]
        })

        if safety_response.status_code == 200:
            vulnerabilities = safety_response.json().get('vulnerabilities', [])

    return vulnerabilities
```

## Severity Scoring System

### CVSS Score Calculation

```python
class VulnerabilitySeverity:
    """Calculate and categorize vulnerability severity"""

    SEVERITY_THRESHOLDS = {
        'critical': 9.0,
        'high': 7.0,
        'moderate': 4.0,
        'low': 0.1
    }

    def __init__(self, cvss_score: float):
        self.cvss_score = cvss_score
        self.category = self._categorize()

    def _categorize(self) -> str:
        """Categorize vulnerability by CVSS score"""
        if self.cvss_score >= self.SEVERITY_THRESHOLDS['critical']:
            return 'critical'
        elif self.cvss_score >= self.SEVERITY_THRESHOLDS['high']:
            return 'high'
        elif self.cvss_score >= self.SEVERITY_THRESHOLDS['moderate']:
            return 'moderate'
        else:
            return 'low'

    def get_priority_score(self, is_direct: bool, has_exploit: bool) -> float:
        """
        Calculate priority score based on multiple factors

        Factors:
        - Base CVSS score
        - Direct vs transitive dependency
        - Exploit availability
        - Public disclosure status
        """
        score = self.cvss_score

        # Direct dependencies get higher priority
        if is_direct:
            score *= 1.5

        # Known exploits significantly increase priority
        if has_exploit:
            score *= 2.0

        return min(score, 10.0)  # Cap at 10
```

### Risk Assessment Matrix

| Severity | CVSS Range | Action Required | Timeline |
|----------|-----------|-----------------|----------|
| Critical | 9.0-10.0 | Immediate patch | < 24 hours |
| High | 7.0-8.9 | Urgent update | < 1 week |
| Moderate | 4.0-6.9 | Scheduled fix | < 1 month |
| Low | 0.1-3.9 | Monitor | Next release |

## Supply Chain Security

### Typosquatting Detection

```python
def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate edit distance between two package names"""
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def check_typosquatting(package_name: str, popular_packages: List[str]) -> Dict:
    """
    Check if package name might be typosquatting a popular package
    """
    suspicious_matches = []

    for popular in popular_packages:
        distance = calculate_levenshtein_distance(
            package_name.lower(),
            popular.lower()
        )

        # Distance of 1-2 is suspicious (e.g., 'reqests' vs 'requests')
        if 0 < distance <= 2:
            suspicious_matches.append({
                'package': popular,
                'distance': distance,
                'risk': 'high' if distance == 1 else 'medium'
            })

    return {
        'is_suspicious': len(suspicious_matches) > 0,
        'matches': suspicious_matches
    }
```

### Maintainer Change Detection

```python
def check_maintainer_changes(package_name: str, registry: str = 'npm') -> Dict:
    """
    Check for recent maintainer changes which could indicate compromise
    """
    if registry == 'npm':
        response = requests.get(f'https://registry.npmjs.org/{package_name}')
        if response.status_code == 200:
            data = response.json()
            versions = data.get('versions', {})

            # Get maintainers for last 5 versions
            recent_versions = sorted(versions.keys())[-5:]
            maintainer_history = []

            for version in recent_versions:
                version_data = versions[version]
                maintainers = version_data.get('maintainers', [])
                maintainer_emails = [m['email'] for m in maintainers]
                maintainer_history.append({
                    'version': version,
                    'maintainers': maintainer_emails
                })

            # Detect changes
            changes = []
            for i in range(1, len(maintainer_history)):
                prev_maintainers = set(maintainer_history[i-1]['maintainers'])
                curr_maintainers = set(maintainer_history[i]['maintainers'])

                added = curr_maintainers - prev_maintainers
                removed = prev_maintainers - curr_maintainers

                if added or removed:
                    changes.append({
                        'version': maintainer_history[i]['version'],
                        'added': list(added),
                        'removed': list(removed)
                    })

            return {
                'has_changes': len(changes) > 0,
                'changes': changes,
                'risk_level': 'high' if len(changes) > 2 else 'medium' if changes else 'low'
            }

    return {'has_changes': False, 'changes': []}
```

## Automated Remediation

### Update Script Generation

```bash
#!/bin/bash
# Auto-generated dependency update script

set -e  # Exit on error

echo "🔒 Security Update Automation"
echo "=============================="

# Backup current state
echo "📦 Creating backup..."
cp package.json package.json.backup
cp package-lock.json package-lock.json.backup 2>/dev/null || true

# Update vulnerable packages
echo "🔧 Updating vulnerable packages..."

# Critical vulnerabilities (update immediately)
npm update package1@^2.1.5 --save
npm update package2@~3.0.1 --save-dev

# Run security audit
echo "🔍 Running security audit..."
npm audit

# Run tests
echo "🧪 Running test suite..."
npm test

if [ $? -eq 0 ]; then
    echo "✅ All tests passed - updates successful"
    rm package.json.backup package-lock.json.backup 2>/dev/null || true
else
    echo "❌ Tests failed - reverting changes..."
    mv package.json.backup package.json
    mv package-lock.json.backup package-lock.json 2>/dev/null || true
    exit 1
fi

echo "✨ Security updates completed successfully"
```

### Pull Request Template

```markdown
## 🔒 Security Dependency Update

### Summary
This PR updates {count} dependencies to address security vulnerabilities.

### Critical Vulnerabilities Fixed ({critical_count})

| Package | Current | Updated | CVE | Severity |
|---------|---------|---------|-----|----------|
| {package1} | {current_version} | {new_version} | CVE-2024-XXXXX | Critical |

### High Severity Updates ({high_count})

| Package | Current | Updated | Issue |
|---------|---------|---------|-------|
| {package2} | {current_version} | {new_version} | XSS vulnerability |

### Moderate/Low Updates ({other_count})

- {package3}: {current} → {new} (dependency update)
- {package4}: {current} → {new} (maintenance release)

### Testing
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Manual smoke testing completed
- [ ] No breaking changes identified

### Security Checklist
- [ ] All critical vulnerabilities addressed
- [ ] No new high-severity issues introduced
- [ ] License compliance verified
- [ ] No unexpected dependency additions

### Deployment Notes
- Can be deployed immediately after approval
- No migration steps required
- No configuration changes needed

cc: @security-team @devops-team
```

## Best Practices

### Regular Audit Schedule

```yaml
security_audit_schedule:
  daily:
    - Run automated vulnerability scan
    - Check for new advisories
    - Update severity dashboard

  weekly:
    - Review moderate/low vulnerabilities
    - Plan update schedule
    - Update dependency inventory

  monthly:
    - Comprehensive security review
    - License compliance audit
    - Supply chain assessment
    - Update security policies
```

### Dependency Approval Workflow

```python
class DependencyApprovalWorkflow:
    """Workflow for approving new dependencies"""

    APPROVAL_CRITERIA = {
        'min_weekly_downloads': 1000,
        'min_github_stars': 100,
        'max_critical_vulnerabilities': 0,
        'max_high_vulnerabilities': 0,
        'required_license_types': ['MIT', 'Apache-2.0', 'BSD-3-Clause'],
        'max_dependency_age_years': 3,
        'min_maintainer_count': 2
    }

    def evaluate_dependency(self, package_info: Dict) -> Dict:
        """
        Evaluate if dependency meets approval criteria
        """
        checks = {
            'popularity': package_info['weekly_downloads'] >= self.APPROVAL_CRITERIA['min_weekly_downloads'],
            'community': package_info.get('github_stars', 0) >= self.APPROVAL_CRITERIA['min_github_stars'],
            'security': package_info['critical_vulns'] <= self.APPROVAL_CRITERIA['max_critical_vulnerabilities'],
            'license': package_info['license'] in self.APPROVAL_CRITERIA['required_license_types'],
            'maintenance': len(package_info['maintainers']) >= self.APPROVAL_CRITERIA['min_maintainer_count']
        }

        return {
            'approved': all(checks.values()),
            'checks': checks,
            'recommendations': self._generate_recommendations(checks)
        }

    def _generate_recommendations(self, checks: Dict) -> List[str]:
        """Generate recommendations for failed checks"""
        recommendations = []

        if not checks['popularity']:
            recommendations.append("Low download count - verify package necessity")
        if not checks['community']:
            recommendations.append("Limited community support - consider alternatives")
        if not checks['security']:
            recommendations.append("Security vulnerabilities found - do not use")
        if not checks['license']:
            recommendations.append("Incompatible license - legal review required")
        if not checks['maintenance']:
            recommendations.append("Insufficient maintainers - risk of abandonment")

        return recommendations
```
