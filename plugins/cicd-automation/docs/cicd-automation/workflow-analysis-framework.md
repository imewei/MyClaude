# Workflow Analysis Framework

**Version**: 1.0.3
**Command**: `/workflow-automate`
**Category**: CI/CD Automation

## Overview

The Workflow Analysis Framework provides automated project analysis to identify existing CI/CD workflows, manual processes, and automation opportunities. This framework powers the discovery phase of the `/workflow-automate` command.

---

## WorkflowAnalyzer Class

Complete Python implementation for comprehensive project analysis:

```python
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class TechStack:
    """Detected technology stack"""
    languages: List[str]
    frameworks: List[str]
    build_tools: List[str]
    testing: List[str]
    package_managers: List[str]

@dataclass
class AutomationOpportunity:
    """Automation recommendation"""
    priority: str  # high, medium, low
    category: str  # ci_cd, testing, deployment, security, documentation
    recommendation: str
    tools: List[str]
    effort: str  # low, medium, high
    estimated_time: str
    impact: str  # high, medium, low

@dataclass
class WorkflowFile:
    """Detected workflow file"""
    path: str
    type: str  # github_actions, gitlab_ci, jenkins, circleci, etc.
    stages: List[str]
    jobs: List[str]

class WorkflowAnalyzer:
    """
    Analyzes projects to identify existing workflows and automation opportunities
    """

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.workflows: List[WorkflowFile] = []
        self.manual_scripts: List[str] = []
        self.tech_stack: Optional[TechStack] = None
        self.complexity_score: int = 0

    def analyze_project(self) -> Dict[str, Any]:
        """
        Complete project analysis

        Returns:
            Dictionary containing:
            - current_workflows: List of detected CI/CD workflows
            - manual_processes: List of manual scripts and processes
            - automation_opportunities: Prioritized recommendations
            - tech_stack: Detected technologies
            - complexity_score: Project complexity (0-100)
        """
        self.workflows = self._find_existing_workflows()
        self.manual_scripts = self._identify_manual_processes()
        self.tech_stack = self._detect_tech_stack()
        self.complexity_score = self._calculate_complexity()

        build_info = self._analyze_build_process()
        test_info = self._analyze_test_process()
        deploy_info = self._analyze_deployment_process()

        opportunities = self._generate_recommendations({
            'workflows': self.workflows,
            'manual_scripts': self.manual_scripts,
            'tech_stack': self.tech_stack,
            'build': build_info,
            'test': test_info,
            'deploy': deploy_info,
            'complexity': self.complexity_score
        })

        return {
            'current_workflows': [asdict(w) for w in self.workflows],
            'manual_processes': self.manual_scripts,
            'automation_opportunities': [asdict(opp) for opp in opportunities],
            'tech_stack': asdict(self.tech_stack) if self.tech_stack else {},
            'build_process': build_info,
            'test_process': test_info,
            'deployment_process': deploy_info,
            'complexity_score': self.complexity_score,
            'complexity_level': self._get_complexity_level(self.complexity_score)
        }

    def _find_existing_workflows(self) -> List[WorkflowFile]:
        """Detect existing CI/CD workflow files"""
        workflows = []

        # GitHub Actions
        gh_actions_dir = self.project_path / '.github' / 'workflows'
        if gh_actions_dir.exists():
            for yml_file in gh_actions_dir.glob('*.yml') + gh_actions_dir.glob('*.yaml'):
                workflow = self._parse_github_actions(yml_file)
                if workflow:
                    workflows.append(workflow)

        # GitLab CI
        gitlab_ci = self.project_path / '.gitlab-ci.yml'
        if gitlab_ci.exists():
            workflow = self._parse_gitlab_ci(gitlab_ci)
            if workflow:
                workflows.append(workflow)

        # Jenkins
        jenkinsfile = self.project_path / 'Jenkinsfile'
        if jenkinsfile.exists():
            workflows.append(WorkflowFile(
                path=str(jenkinsfile),
                type='jenkins',
                stages=self._extract_jenkins_stages(jenkinsfile),
                jobs=[]
            ))

        # CircleCI
        circleci_config = self.project_path / '.circleci' / 'config.yml'
        if circleci_config.exists():
            workflow = self._parse_circleci(circleci_config)
            if workflow:
                workflows.append(workflow)

        return workflows

    def _parse_github_actions(self, file_path: Path) -> Optional[WorkflowFile]:
        """Parse GitHub Actions workflow file"""
        try:
            import yaml
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)

            jobs = list(data.get('jobs', {}).keys())
            return WorkflowFile(
                path=str(file_path),
                type='github_actions',
                stages=[],  # GitHub Actions doesn't have explicit stages
                jobs=jobs
            )
        except Exception:
            return None

    def _parse_gitlab_ci(self, file_path: Path) -> Optional[WorkflowFile]:
        """Parse GitLab CI configuration"""
        try:
            import yaml
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)

            stages = data.get('stages', [])
            jobs = [k for k in data.keys() if not k.startswith('.') and k not in ['stages', 'variables', 'include']]

            return WorkflowFile(
                path=str(file_path),
                type='gitlab_ci',
                stages=stages,
                jobs=jobs
            )
        except Exception:
            return None

    def _parse_circleci(self, file_path: Path) -> Optional[WorkflowFile]:
        """Parse CircleCI configuration"""
        try:
            import yaml
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)

            workflows = data.get('workflows', {})
            jobs = []
            for workflow_name, workflow_data in workflows.items():
                jobs.extend(workflow_data.get('jobs', []))

            return WorkflowFile(
                path=str(file_path),
                type='circleci',
                stages=[],
                jobs=jobs
            )
        except Exception:
            return None

    def _extract_jenkins_stages(self, file_path: Path) -> List[str]:
        """Extract stages from Jenkinsfile"""
        try:
            content = file_path.read_text()
            # Simple regex to find stage declarations
            stages = re.findall(r"stage\s*\(\s*['\"]([^'\"]+)['\"]", content)
            return stages
        except Exception:
            return []

    def _identify_manual_processes(self) -> List[str]:
        """Identify manual scripts that could be automated"""
        manual_scripts = []

        # Common script patterns
        script_patterns = [
            'build.sh', 'build.bash',
            'deploy.sh', 'deploy.bash',
            'release.sh', 'release.bash',
            'test.sh', 'test.bash',
            'setup.sh', 'setup.bash',
            'install.sh', 'install.bash'
        ]

        for pattern in script_patterns:
            for script in self.project_path.rglob(pattern):
                # Exclude hidden directories
                if not any(part.startswith('.') for part in script.parts):
                    manual_scripts.append(str(script.relative_to(self.project_path)))

        # Check README for manual process documentation
        readme_files = list(self.project_path.glob('README.md')) + list(self.project_path.glob('README.rst'))
        for readme in readme_files:
            manual_processes = self._extract_manual_processes_from_readme(readme)
            manual_scripts.extend(manual_processes)

        return list(set(manual_scripts))  # Remove duplicates

    def _extract_manual_processes_from_readme(self, readme_path: Path) -> List[str]:
        """Extract manual process descriptions from README"""
        try:
            content = readme_path.read_text()
            processes = []

            # Look for common manual process indicators
            patterns = [
                r'## (?:Building|Build|Deployment|Deploy|Testing|Test).*?\n(.*?)(?=\n##|\Z)',
                r'### (?:How to build|How to deploy|How to test).*?\n(.*?)(?=\n###|\Z)'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    # Check if it contains command-line instructions
                    if re.search(r'```|`\w+`|\$', match):
                        processes.append(f"Manual process documented in {readme_path.name}")

            return processes
        except Exception:
            return []

    def _detect_tech_stack(self) -> TechStack:
        """Detect technology stack from project files"""
        languages = []
        frameworks = []
        build_tools = []
        testing = []
        package_managers = []

        # JavaScript/TypeScript
        if (self.project_path / 'package.json').exists():
            languages.append('JavaScript')
            package_managers.append('npm')

            try:
                with open(self.project_path / 'package.json') as f:
                    pkg = json.load(f)
                    deps = {**pkg.get('dependencies', {}), **pkg.get('devDependencies', {})}

                    # Detect frameworks
                    if 'react' in deps:
                        frameworks.append('React')
                    if 'vue' in deps:
                        frameworks.append('Vue')
                    if '@angular/core' in deps:
                        frameworks.append('Angular')
                    if 'next' in deps:
                        frameworks.append('Next.js')
                    if 'express' in deps:
                        frameworks.append('Express')
                    if 'fastify' in deps:
                        frameworks.append('Fastify')

                    # Detect build tools
                    if 'webpack' in deps:
                        build_tools.append('Webpack')
                    if 'vite' in deps:
                        build_tools.append('Vite')
                    if 'rollup' in deps:
                        build_tools.append('Rollup')

                    # Detect testing
                    if 'jest' in deps:
                        testing.append('Jest')
                    if 'vitest' in deps:
                        testing.append('Vitest')
                    if 'mocha' in deps:
                        testing.append('Mocha')
                    if '@playwright/test' in deps:
                        testing.append('Playwright')
                    if 'cypress' in deps:
                        testing.append('Cypress')
            except Exception:
                pass

        if (self.project_path / 'tsconfig.json').exists():
            languages.append('TypeScript')

        # Python
        if (self.project_path / 'requirements.txt').exists() or (self.project_path / 'pyproject.toml').exists():
            languages.append('Python')
            package_managers.append('pip')

            # Detect Python frameworks
            req_files = list(self.project_path.glob('requirements*.txt'))
            for req_file in req_files:
                try:
                    content = req_file.read_text().lower()
                    if 'django' in content:
                        frameworks.append('Django')
                    if 'flask' in content:
                        frameworks.append('Flask')
                    if 'fastapi' in content:
                        frameworks.append('FastAPI')
                    if 'pytest' in content:
                        testing.append('Pytest')
                except Exception:
                    pass

        # Go
        if (self.project_path / 'go.mod').exists():
            languages.append('Go')
            package_managers.append('go modules')
            build_tools.append('go build')
            testing.append('go test')

        # Rust
        if (self.project_path / 'Cargo.toml').exists():
            languages.append('Rust')
            package_managers.append('Cargo')
            build_tools.append('Cargo')
            testing.append('Cargo test')

        # Java
        if (self.project_path / 'pom.xml').exists():
            languages.append('Java')
            build_tools.append('Maven')
        if (self.project_path / 'build.gradle').exists() or (self.project_path / 'build.gradle.kts').exists():
            languages.append('Java')
            build_tools.append('Gradle')

        return TechStack(
            languages=languages,
            frameworks=frameworks,
            build_tools=build_tools,
            testing=testing,
            package_managers=package_managers
        )

    def _analyze_build_process(self) -> Dict[str, Any]:
        """Analyze build process configuration"""
        if not self.tech_stack:
            return {}

        build_info = {
            'automated': len(self.workflows) > 0,
            'build_commands': [],
            'dockerfile_present': (self.project_path / 'Dockerfile').exists()
        }

        # Extract build commands from package.json
        if 'npm' in self.tech_stack.package_managers:
            try:
                with open(self.project_path / 'package.json') as f:
                    pkg = json.load(f)
                    scripts = pkg.get('scripts', {})
                    if 'build' in scripts:
                        build_info['build_commands'].append(f"npm run build")
            except Exception:
                pass

        return build_info

    def _analyze_test_process(self) -> Dict[str, Any]:
        """Analyze testing setup"""
        test_info = {
            'frameworks': self.tech_stack.testing if self.tech_stack else [],
            'automated': False,
            'test_commands': []
        }

        # Check if tests are automated in workflows
        for workflow in self.workflows:
            if any('test' in job.lower() for job in workflow.jobs):
                test_info['automated'] = True

        return test_info

    def _analyze_deployment_process(self) -> Dict[str, Any]:
        """Analyze deployment configuration"""
        deploy_info = {
            'automated': False,
            'platforms': []
        }

        # Check for deployment in workflows
        for workflow in self.workflows:
            if any('deploy' in job.lower() for job in workflow.jobs):
                deploy_info['automated'] = True

        # Detect deployment platforms
        if (self.project_path / 'vercel.json').exists():
            deploy_info['platforms'].append('Vercel')
        if (self.project_path / 'netlify.toml').exists():
            deploy_info['platforms'].append('Netlify')
        if (self.project_path / 'Dockerfile').exists():
            deploy_info['platforms'].append('Docker')

        return deploy_info

    def _calculate_complexity(self) -> int:
        """
        Calculate project complexity score (0-100)

        Factors:
        - Number of languages: +10 per language
        - Number of frameworks: +5 per framework
        - Number of services/packages: +2 per package
        - Manual processes: +10 per manual script
        - Existing workflows: -5 per workflow (reduces complexity)
        """
        score = 0

        if self.tech_stack:
            score += len(self.tech_stack.languages) * 10
            score += len(self.tech_stack.frameworks) * 5

        score += len(self.manual_scripts) * 10
        score -= len(self.workflows) * 5

        # Cap at 100
        return min(max(score, 0), 100)

    def _get_complexity_level(self, score: int) -> str:
        """Convert complexity score to level"""
        if score < 20:
            return 'simple'
        elif score < 50:
            return 'medium'
        elif score < 75:
            return 'complex'
        else:
            return 'epic'

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[AutomationOpportunity]:
        """Generate prioritized automation recommendations"""
        opportunities = []

        # No CI/CD pipeline
        if len(self.workflows) == 0:
            opportunities.append(AutomationOpportunity(
                priority='high',
                category='ci_cd',
                recommendation='Implement CI/CD pipeline with automated testing and deployment',
                tools=['GitHub Actions', 'GitLab CI'],
                effort='medium',
                estimated_time='2-4 hours',
                impact='high'
            ))

        # Manual build scripts
        if any('build' in script for script in self.manual_scripts):
            opportunities.append(AutomationOpportunity(
                priority='high',
                category='ci_cd',
                recommendation='Automate build process in CI/CD pipeline',
                tools=['GitHub Actions', 'GitLab CI'],
                effort='low',
                estimated_time='30-60 minutes',
                impact='medium'
            ))

        # No automated testing
        if not analysis.get('test_info', {}).get('automated'):
            opportunities.append(AutomationOpportunity(
                priority='high',
                category='testing',
                recommendation='Add automated test execution to CI/CD pipeline',
                tools=[f for f in self.tech_stack.testing] if self.tech_stack else ['Jest', 'Pytest'],
                effort='low',
                estimated_time='1-2 hours',
                impact='high'
            ))

        # No automated deployment
        if not analysis.get('deploy_info', {}).get('automated'):
            opportunities.append(AutomationOpportunity(
                priority='medium',
                category='deployment',
                recommendation='Implement automated deployment pipeline',
                tools=['GitHub Actions', 'GitLab CI', 'ArgoCD'],
                effort='high',
                estimated_time='4-8 hours',
                impact='high'
            ))

        # Dockerfile but no container registry push
        if analysis.get('build_info', {}).get('dockerfile_present'):
            if not any('docker' in job.lower() for wf in self.workflows for job in wf.jobs):
                opportunities.append(AutomationOpportunity(
                    priority='medium',
                    category='ci_cd',
                    recommendation='Add Docker image build and push to CI/CD',
                    tools=['Docker', 'GitHub Container Registry', 'Docker Hub'],
                    effort='medium',
                    estimated_time='1-2 hours',
                    impact='medium'
                ))

        # Sort by priority (high > medium > low)
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        opportunities.sort(key=lambda x: priority_order[x.priority], reverse=True)

        return opportunities
```

---

## Usage Example

```python
# Analyze current project
analyzer = WorkflowAnalyzer('/path/to/project')
analysis = analyzer.analyze_project()

# Print results
print(json.dumps(analysis, indent=2))
```

### Example Output

```json
{
  "automation_opportunities": [
    {
      "priority": "high",
      "category": "ci_cd",
      "recommendation": "Implement CI/CD pipeline with automated testing and deployment",
      "tools": ["GitHub Actions", "GitLab CI"],
      "effort": "medium",
      "estimated_time": "2-4 hours",
      "impact": "high"
    },
    {
      "priority": "high",
      "category": "testing",
      "recommendation": "Add automated test execution to CI/CD pipeline",
      "tools": ["Jest"],
      "effort": "low",
      "estimated_time": "1-2 hours",
      "impact": "high"
    },
    {
      "priority": "medium",
      "category": "deployment",
      "recommendation": "Implement automated deployment pipeline",
      "tools": ["GitHub Actions", "GitLab CI", "ArgoCD"],
      "effort": "high",
      "estimated_time": "4-8 hours",
      "impact": "high"
    }
  ],
  "complexity_score": 65,
  "complexity_level": "medium",
  "tech_stack": {
    "languages": ["TypeScript", "Python"],
    "frameworks": ["React", "FastAPI"],
    "build_tools": ["Webpack", "npm"],
    "testing": ["Jest", "Pytest"],
    "package_managers": ["npm", "pip"]
  },
  "current_workflows": [],
  "manual_processes": [
    "scripts/build.sh",
    "scripts/deploy.sh",
    "Manual process documented in README.md"
  ],
  "build_process": {
    "automated": false,
    "build_commands": ["npm run build"],
    "dockerfile_present": true
  },
  "test_process": {
    "frameworks": ["Jest", "Pytest"],
    "automated": false,
    "test_commands": []
  },
  "deployment_process": {
    "automated": false,
    "platforms": ["Docker"]
  }
}
```

---

## Complexity Scoring

### Score Ranges

| Score | Level | Description | Automation Effort |
|-------|-------|-------------|-------------------|
| 0-19 | Simple | Single language, few dependencies | 1-2 hours |
| 20-49 | Medium | Multiple languages or frameworks | 2-4 hours |
| 50-74 | Complex | Multiple languages, frameworks, services | 4-8 hours |
| 75-100 | Epic | Polyglot, microservices, complex deployment | 8+ hours |

### Scoring Formula

```
Score = (Languages × 10) + (Frameworks × 5) + (Manual Scripts × 10) - (Existing Workflows × 5)
Capped at: 0-100
```

---

## Integration with /workflow-automate

The WorkflowAnalyzer is automatically invoked during Phase 1 (Discovery & Analysis) of the `/workflow-automate` command:

```yaml
Phase 1: Discovery & Analysis
  ├─ Run WorkflowAnalyzer on project directory
  ├─ Display complexity score and detected tech stack
  ├─ Show automation opportunities ranked by priority
  └─ Ask user to confirm recommended workflows
```

---

For complete workflow implementation patterns and orchestration, see:
- [github-actions-reference.md](github-actions-reference.md)
- [gitlab-ci-reference.md](gitlab-ci-reference.md)
- [workflow-orchestration-patterns.md](workflow-orchestration-patterns.md)
