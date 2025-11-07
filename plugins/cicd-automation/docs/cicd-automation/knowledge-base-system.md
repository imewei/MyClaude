# Knowledge Base System

**Version**: 1.0.3
**Command**: `/fix-commit-errors`
**Category**: CI/CD Automation

## Overview

The Knowledge Base System provides adaptive learning for CI/CD error resolution by tracking successful fixes, updating confidence scores using Bayesian statistics, and continuously improving solution recommendations.

---

## Knowledge Base Schema

### Data Structure

```json
{
  "version": "1.0",
  "last_updated": "2025-11-06T14:30:00Z",

  "error_patterns": [
    {
      "id": "npm-eresolve-001",
      "pattern": "ERESOLVE.*peer dependency.*react@",
      "category": "dependency_conflict",
      "subcategory": "peer_dependency_mismatch",
      "root_cause": "React version mismatch in peer dependencies",
      "first_seen": "2025-09-01T10:00:00Z",
      "last_seen": "2025-11-06T14:15:30Z",
      "occurrences": 25,
      "solutions": [
        {
          "solution_id": "npm-legacy-peer-deps",
          "action_type": "workflow_modification",
          "description": "Install with --legacy-peer-deps flag",
          "implementation": {
            "files": [".github/workflows/ci.yml"],
            "changes": "sed -i 's/npm ci/npm ci --legacy-peer-deps/' .github/workflows/ci.yml"
          },
          "risk_level": "low",
          "reversibility": "high",
          "success_rate": 0.88,
          "applications": 22,
          "successes": 19,
          "failures": 3,
          "average_resolution_time_seconds": 180,
          "confidence": 0.88,
          "last_updated": "2025-11-06T14:30:00Z"
        },
        {
          "solution_id": "update-react-version",
          "action_type": "package_update",
          "description": "Update React to v18 to resolve peer dependency",
          "implementation": {
            "packages": ["react@^18.2.0", "react-dom@^18.2.0"],
            "command": "npm install react@^18.2.0 react-dom@^18.2.0"
          },
          "risk_level": "medium",
          "reversibility": "medium",
          "success_rate": 0.95,
          "applications": 10,
          "successes": 9,
          "failures": 1,
          "average_resolution_time_seconds": 600,
          "confidence": 0.92,
          "last_updated": "2025-11-05T10:20:00Z"
        }
      ],
      "related_patterns": ["npm-eresolve-002", "npm-peer-dep-001"],
      "tags": ["npm", "peer-dependency", "react"]
    }
  ],

  "successful_fixes": [
    {
      "fix_id": "fix-20251106-001",
      "timestamp": "2025-11-06T14:30:00Z",
      "run_id": "12345",
      "workflow_name": "CI/CD Pipeline",
      "error_pattern_id": "npm-eresolve-001",
      "solution_id": "npm-legacy-peer-deps",
      "commit_sha": "abc123def456",
      "branch": "feature/new-api",
      "rerun_successful": true,
      "resolution_time_seconds": 175,
      "iterations": 1,
      "context": {
        "node_version": "18",
        "npm_version": "9.5.0",
        "repository": "org/repo"
      }
    }
  ],

  "failed_fixes": [
    {
      "fix_id": "fix-20251105-003",
      "timestamp": "2025-11-05T16:45:00Z",
      "run_id": "12340",
      "error_pattern_id": "npm-eresolve-001",
      "solution_id": "npm-legacy-peer-deps",
      "failure_reason": "Additional type errors after fix",
      "next_action": "tried_alternative_solution",
      "context": {}
    }
  ],

  "statistics": {
    "total_errors_analyzed": 185,
    "unique_error_patterns": 52,
    "total_fixes_attempted": 120,
    "auto_fixed": 95,
    "manual_intervention_required": 25,
    "overall_success_rate": 0.79,
    "average_resolution_time_seconds": 320,
    "average_iterations_to_success": 1.6
  },

  "metadata": {
    "schema_version": "1.0",
    "created_at": "2025-09-01T00:00:00Z",
    "repository": "org/repo",
    "environment": "production"
  }
}
```

---

## Learning Algorithms

### Pattern Extraction

```python
class PatternExtractor:
    def extract_new_pattern(self, error_entries: List[ErrorEntry]) -> Optional[ErrorPattern]:
        """
        Extract new error pattern from multiple similar error instances
        """
        # 1. Group similar errors
        similar_errors = self.cluster_similar_errors(error_entries)

        if len(similar_errors) < 3:
            return None  # Need at least 3 instances to create pattern

        # 2. Extract common regex pattern
        common_pattern = self.find_common_pattern(similar_errors)

        # 3. Identify root cause
        root_cause = self.analyze_root_cause(similar_errors)

        # 4. Create pattern entry
        new_pattern = ErrorPattern(
            id=self.generate_pattern_id(common_pattern),
            pattern=common_pattern,
            category=self.classify_category(similar_errors),
            subcategory=self.classify_subcategory(similar_errors),
            root_cause=root_cause,
            first_seen=min(e.timestamp for e in similar_errors),
            last_seen=max(e.timestamp for e in similar_errors),
            occurrences=len(similar_errors),
            solutions=[],  # Will be populated after first successful fix
            related_patterns=[],
            tags=self.extract_tags(similar_errors)
        )

        return new_pattern

    def cluster_similar_errors(self, errors: List[ErrorEntry]) -> List[ErrorEntry]:
        """Use TF-IDF + cosine similarity to find similar errors"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        messages = [e.message for e in errors]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(messages)

        similarities = cosine_similarity(tfidf_matrix)

        # Find errors with >0.8 similarity
        similar_indices = []
        for i in range(len(similarities)):
            for j in range(i+1, len(similarities)):
                if similarities[i][j] > 0.8:
                    similar_indices.extend([i, j])

        return [errors[i] for i in set(similar_indices)]
```

### Solution Success Tracking

```python
class SolutionTracker:
    def update_solution_stats(
        self,
        pattern_id: str,
        solution_id: str,
        success: bool,
        resolution_time: int
    ):
        """Update solution statistics after fix attempt"""

        pattern = self.kb.get_pattern(pattern_id)
        solution = pattern.get_solution(solution_id)

        # Update counts
        solution.applications += 1
        if success:
            solution.successes += 1
        else:
            solution.failures += 1

        # Update success rate (simple average)
        solution.success_rate = solution.successes / solution.applications

        # Update average resolution time (moving average)
        if success:
            alpha = 0.3  # Weight for new observation
            solution.average_resolution_time_seconds = (
                alpha * resolution_time +
                (1 - alpha) * solution.average_resolution_time_seconds
            )

        # Update confidence using Bayesian approach
        solution.confidence = self.calculate_bayesian_confidence(solution)

        solution.last_updated = datetime.now()

        self.kb.save()
```

### Bayesian Confidence Scoring

```python
class BayesianConfidenceCalculator:
    def calculate_confidence(
        self,
        prior: float,
        successes: int,
        failures: int,
        prior_strength: float = 5.0
    ) -> float:
        """
        Calculate posterior confidence using Beta distribution

        prior: Initial belief in success rate (0-1)
        successes: Number of successful applications
        failures: Number of failed applications
        prior_strength: How many pseudo-observations the prior represents
        """
        # Beta distribution parameters
        alpha_prior = prior * prior_strength
        beta_prior = (1 - prior) * prior_strength

        # Update with observed data
        alpha_posterior = alpha_prior + successes
        beta_posterior = beta_prior + failures

        # Calculate posterior mean (expected success rate)
        posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)

        return posterior_mean

    def calculate_confidence_interval(
        self,
        successes: int,
        failures: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for success rate"""
        from scipy.stats import beta

        alpha = successes + 1
        beta_param = failures + 1

        lower = beta.ppf((1 - confidence_level) / 2, alpha, beta_param)
        upper = beta.ppf((1 + confidence_level) / 2, alpha, beta_param)

        return (lower, upper)
```

---

## Continuous Improvement

### Confidence Calibration

```python
def calibrate_confidence_scores(kb: KnowledgeBase):
    """
    Recalibrate all confidence scores based on recent performance
    """
    for pattern in kb.error_patterns:
        for solution in pattern.solutions:
            # Get recent fixes (last 30 days)
            recent_fixes = kb.get_recent_fixes(
                pattern_id=pattern.id,
                solution_id=solution.id,
                days=30
            )

            if len(recent_fixes) >= 5:  # Minimum sample size
                # Calculate recent success rate
                recent_successes = sum(1 for f in recent_fixes if f.rerun_successful)
                recent_success_rate = recent_successes / len(recent_fixes)

                # Update confidence with recent data weighted more heavily
                # 70% recent, 30% historical
                solution.confidence = (
                    0.7 * recent_success_rate +
                    0.3 * solution.success_rate
                )

                kb.save()
```

### Pattern Refinement

```python
def refine_pattern_regex(pattern: ErrorPattern, new_errors: List[ErrorEntry]):
    """
    Refine regex pattern to capture more variations
    """
    # Collect all error messages that should match this pattern
    all_messages = [e.message for e in new_errors]

    # Use regex generalization algorithm
    generalized_pattern = generalize_regex(
        pattern.pattern,
        all_messages
    )

    # Validate new pattern doesn't create false positives
    false_positive_rate = test_pattern_on_other_errors(
        generalized_pattern,
        kb.get_all_errors()
    )

    if false_positive_rate < 0.05:  # Less than 5% false positive
        pattern.pattern = generalized_pattern
        pattern.last_updated = datetime.now()
        kb.save()
```

---

## Cross-Repository Learning

### Anonymized Pattern Sharing

```python
class CrossRepoLearning:
    """
    Optional: Share anonymized error patterns across repositories
    """

    def anonymize_pattern(self, pattern: ErrorPattern) -> dict:
        """Remove sensitive information before sharing"""
        return {
            "pattern": pattern.pattern,
            "category": pattern.category,
            "subcategory": pattern.subcategory,
            "solutions": [
                {
                    "action_type": s.action_type,
                    "description": s.description,
                    "risk_level": s.risk_level,
                    "success_rate": s.success_rate,
                    "applications": s.applications
                }
                for s in pattern.solutions
            ],
            "tags": pattern.tags
        }

    def import_community_patterns(self, source_url: str):
        """Import patterns from community knowledge base"""
        community_patterns = fetch_json(source_url)

        for cp in community_patterns:
            # Check if pattern already exists
            existing = self.kb.find_pattern_by_regex(cp['pattern'])

            if existing:
                # Merge solution statistics
                self.merge_solution_stats(existing, cp)
            else:
                # Add new pattern with community data
                new_pattern = self.create_pattern_from_community_data(cp)
                self.kb.add_pattern(new_pattern)

        self.kb.save()
```

---

## Knowledge Base Queries

### Query Interface

```python
class KnowledgeBaseQuery:
    def find_best_solutions(
        self,
        error_pattern_id: str,
        context: dict,
        min_confidence: float = 0.5
    ) -> List[Solution]:
        """Find best solutions for error pattern with context filtering"""

        pattern = self.kb.get_pattern(error_pattern_id)
        if not pattern:
            return []

        # Filter solutions by confidence
        candidates = [
            s for s in pattern.solutions
            if s.confidence >= min_confidence
        ]

        # Context-aware filtering
        if 'node_version' in context:
            candidates = self.filter_by_node_version(candidates, context['node_version'])

        if 'environment' in context:
            candidates = self.filter_by_environment(candidates, context['environment'])

        # Sort by confidence * success_rate
        candidates.sort(
            key=lambda s: s.confidence * s.success_rate,
            reverse=True
        )

        return candidates

    def get_similar_patterns(self, error_message: str, limit: int = 5) -> List[ErrorPattern]:
        """Find similar error patterns using semantic similarity"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        all_patterns = self.kb.get_all_patterns()
        pattern_texts = [p.pattern for p in all_patterns]

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(pattern_texts + [error_message])

        # Last vector is the query error message
        similarities = cosine_similarity(vectors[-1:], vectors[:-1])[0]

        # Get top N most similar
        top_indices = similarities.argsort()[-limit:][::-1]

        return [all_patterns[i] for i in top_indices]
```

---

## Performance Metrics

### Knowledge Base Health

```python
def calculate_kb_health_score(kb: KnowledgeBase) -> float:
    """
    Calculate overall knowledge base health score (0-100)
    """
    metrics = {
        'pattern_coverage': calculate_pattern_coverage(kb),  # 0-1
        'solution_quality': calculate_avg_success_rate(kb),  # 0-1
        'freshness': calculate_freshness(kb),  # 0-1
        'confidence_accuracy': calculate_confidence_accuracy(kb)  # 0-1
    }

    weights = {
        'pattern_coverage': 0.25,
        'solution_quality': 0.35,
        'freshness': 0.15,
        'confidence_accuracy': 0.25
    }

    health_score = sum(
        metrics[k] * weights[k]
        for k in metrics
    ) * 100

    return health_score

def calculate_confidence_accuracy(kb: KnowledgeBase) -> float:
    """
    Measure how well confidence scores predict actual success
    """
    recent_fixes = kb.get_recent_fixes(days=90)

    # Group by confidence buckets
    buckets = {
        'high': [],  # 0.8-1.0
        'medium': [],  # 0.5-0.8
        'low': []  # 0.0-0.5
    }

    for fix in recent_fixes:
        solution = kb.get_solution(fix.pattern_id, fix.solution_id)
        confidence = solution.confidence

        if confidence >= 0.8:
            bucket = 'high'
        elif confidence >= 0.5:
            bucket = 'medium'
        else:
            bucket = 'low'

        buckets[bucket].append(fix.rerun_successful)

    # Expected vs actual success rates
    expected = {'high': 0.9, 'medium': 0.65, 'low': 0.35}
    actual = {
        k: (sum(v) / len(v) if v else 0)
        for k, v in buckets.items()
    }

    # Calculate accuracy (1 - mean absolute error)
    mae = sum(abs(expected[k] - actual[k]) for k in expected) / len(expected)
    accuracy = 1 - mae

    return max(0, accuracy)
```

---

## Storage and Persistence

### File-Based Storage

```python
class FileBasedKnowledgeBase:
    def __init__(self, kb_path: str = '.github/fix-commit-errors/knowledge.json'):
        self.kb_path = kb_path
        self.data = self.load()

    def load(self) -> dict:
        """Load knowledge base from file"""
        if not os.path.exists(self.kb_path):
            return self.create_empty_kb()

        with open(self.kb_path, 'r') as f:
            return json.load(f)

    def save(self):
        """Save knowledge base to file"""
        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)

        with open(self.kb_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def create_empty_kb(self) -> dict:
        """Create empty knowledge base structure"""
        return {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "error_patterns": [],
            "successful_fixes": [],
            "failed_fixes": [],
            "statistics": {
                "total_errors_analyzed": 0,
                "unique_error_patterns": 0,
                "total_fixes_attempted": 0,
                "auto_fixed": 0,
                "manual_intervention_required": 0,
                "overall_success_rate": 0.0,
                "average_resolution_time_seconds": 0,
                "average_iterations_to_success": 0.0
            },
            "metadata": {
                "schema_version": "1.0",
                "created_at": datetime.now().isoformat()
            }
        }
```

---

## Knowledge Base Maintenance

### Cleanup Old Data

```python
def cleanup_old_fixes(kb: KnowledgeBase, retention_days: int = 180):
    """Remove fix records older than retention period"""
    cutoff_date = datetime.now() - timedelta(days=retention_days)

    kb.successful_fixes = [
        f for f in kb.successful_fixes
        if datetime.fromisoformat(f['timestamp']) > cutoff_date
    ]

    kb.failed_fixes = [
        f for f in kb.failed_fixes
        if datetime.fromisoformat(f['timestamp']) > cutoff_date
    ]

    kb.save()
```

### Merge Duplicate Patterns

```python
def merge_duplicate_patterns(kb: KnowledgeBase):
    """Identify and merge duplicate error patterns"""
    patterns = kb.error_patterns

    i = 0
    while i < len(patterns):
        j = i + 1
        while j < len(patterns):
            similarity = calculate_pattern_similarity(patterns[i], patterns[j])

            if similarity > 0.9:  # Very similar patterns
                # Merge j into i
                patterns[i] = merge_patterns(patterns[i], patterns[j])
                patterns.pop(j)
            else:
                j += 1
        i += 1

    kb.save()
```

---

## Integration with Multi-Agent System

Agent 4 (Knowledge Base Consultant) uses this system to:

1. **Query** for matching error patterns
2. **Retrieve** ranked solution recommendations
3. **Track** fix outcomes
4. **Update** confidence scores
5. **Learn** new patterns over time

See [multi-agent-error-analysis.md](multi-agent-error-analysis.md) for complete integration details.
