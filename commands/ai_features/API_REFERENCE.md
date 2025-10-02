## API Reference

Comprehensive API documentation for the AI Features module.

## Core Classes

### AISystem

Main coordinator for all AI features.

```python
from ai_features.ai_system import AISystem, AISystemConfig

# Initialize
config = AISystemConfig(
    enable_claude_api=True,
    enable_local_models=False,
    cache_enabled=True
)
ai_system = AISystem(config)
```

#### Methods

##### `analyze_code_semantics(code_path: Path) -> AIAnalysisResult`
Perform semantic analysis on code.

**Parameters:**
- `code_path`: Path to code file or directory

**Returns:** `AIAnalysisResult` with semantic information

**Example:**
```python
result = ai_system.analyze_code_semantics(Path("mymodule.py"))
print(f"Patterns found: {len(result.data['patterns'])}")
```

##### `predict_performance(code: str, context: Optional[Dict]) -> AIAnalysisResult`
Predict code performance characteristics.

**Parameters:**
- `code`: Source code string
- `context`: Optional execution context (input_size, etc.)

**Returns:** Performance prediction with complexity and bottlenecks

**Example:**
```python
result = ai_system.predict_performance(code, {"input_size": 10000})
print(f"Time: {result.data['time_complexity']}")
```

##### `generate_code(generation_type: str, params: Dict) -> AIAnalysisResult`
Generate code using AI.

**Parameters:**
- `generation_type`: "boilerplate", "tests", "docstrings", "pattern"
- `params`: Generation parameters

**Returns:** Generated code

**Example:**
```python
result = ai_system.generate_code("class", {
    "name": "MyClass",
    "attributes": ["data"],
    "methods": ["process"]
})
```

##### `search_code(query: str, search_type: str, top_k: int) -> AIAnalysisResult`
Search code using neural search.

**Parameters:**
- `query`: Search query (natural language or code)
- `search_type`: "functionality", "example", "signature"
- `top_k`: Number of results

**Returns:** Search results with scores

##### `explain_code(code: str, detail_level: str) -> AIAnalysisResult`
Get AI explanation of code (requires Claude API).

**Parameters:**
- `code`: Code to explain
- `detail_level`: "brief", "detailed", "expert"

**Returns:** Natural language explanation

##### `review_code(code: str, focus: Optional[List[str]]) -> AIAnalysisResult`
Get AI-powered code review (requires Claude API).

**Parameters:**
- `code`: Code to review
- `focus`: Areas to focus on ["security", "performance", etc.]

**Returns:** Code review with issues and suggestions

---

### SemanticAnalyzer

Deep semantic code analysis.

```python
from ai_features.understanding.semantic_analyzer import SemanticAnalyzer

analyzer = SemanticAnalyzer()
```

#### Methods

##### `analyze_codebase(root_path: Path) -> SemanticGraph`
Analyze entire codebase.

**Returns:** `SemanticGraph` with entities, relationships, patterns, and smells

**Example:**
```python
graph = analyzer.analyze_codebase(Path("/project"))
print(f"Entities: {len(graph.entities)}")
print(f"Patterns: {len(graph.patterns)}")
print(f"Code smells: {len(graph.smells)}")
```

##### `analyze_file(file_path: Path) -> Dict[str, Any]`
Analyze single file.

**Returns:** Analysis results with entities, complexity, and intent

##### `get_semantic_summary() -> Dict[str, Any]`
Get summary of semantic analysis.

**Returns:** Summary statistics

#### Data Classes

##### `SemanticGraph`
```python
@dataclass
class SemanticGraph:
    entities: Dict[str, SemanticEntity]
    relationships: Dict[str, List[Tuple[str, str, str]]]
    patterns: List[Dict[str, Any]]
    smells: List[Dict[str, Any]]
```

##### `SemanticEntity`
```python
@dataclass
class SemanticEntity:
    name: str
    node_type: SemanticNode
    file_path: Path
    line_number: int
    semantics: Dict[str, Any]
    relationships: List[str]
    metadata: Dict[str, Any]
```

---

### CodeEmbedder

Generate and manage code embeddings.

```python
from ai_features.understanding.code_embedder import CodeEmbedder

embedder = CodeEmbedder()
```

#### Methods

##### `embed_code(code: str, code_id: str, ...) -> CodeEmbedding`
Generate embedding for code snippet.

**Parameters:**
- `code`: Code text
- `code_id`: Unique identifier
- `file_path`: Source file path
- `code_type`: "function", "class", "module"
- `name`: Name of entity
- `metadata`: Additional metadata

**Returns:** `CodeEmbedding` object

##### `embed_file(file_path: Path) -> List[CodeEmbedding]`
Generate embeddings for all entities in file.

##### `find_similar(query_embedding: np.ndarray, top_k: int, threshold: float) -> List[Tuple[CodeEmbedding, float]]`
Find similar code by embedding vector.

##### `find_similar_code(query_code: str, top_k: int, threshold: float) -> List[Tuple[CodeEmbedding, float]]`
Find similar code by code text.

##### `detect_duplicates(threshold: float) -> List[Tuple[CodeEmbedding, CodeEmbedding, float]]`
Detect duplicate code using semantic similarity.

**Example:**
```python
embeddings = embedder.embed_file(Path("module.py"))
duplicates = embedder.detect_duplicates(threshold=0.9)
for emb1, emb2, similarity in duplicates:
    print(f"Duplicate: {emb1.name} ≈ {emb2.name} ({similarity:.2%})")
```

##### `cluster_code(n_clusters: int) -> Dict[int, List[CodeEmbedding]]`
Cluster code by semantic similarity.

##### `save_embeddings(output_path: Path)`
Save embeddings to disk.

##### `load_embeddings(input_path: Path)`
Load embeddings from disk.

---

### PerformancePredictor

ML-based performance prediction.

```python
from ai_features.prediction.performance_predictor import PerformancePredictor

predictor = PerformancePredictor()
```

#### Methods

##### `predict(code: str, context: Optional[Dict]) -> PerformancePrediction`
Predict performance for code.

**Parameters:**
- `code`: Source code
- `context`: Execution context (input_size, etc.)

**Returns:** `PerformancePrediction` object

**Example:**
```python
prediction = predictor.predict(code, {"input_size": 10000})
print(f"Time: {prediction.time_complexity}")
print(f"Space: {prediction.space_complexity}")
print(f"Runtime: {prediction.estimated_runtime:.2f}s")
print(f"Memory: {prediction.estimated_memory:.2f}MB")

for bottleneck in prediction.bottlenecks:
    print(f"Bottleneck: {bottleneck['type']}")
    print(f"  {bottleneck['description']}")
    print(f"  Suggestion: {bottleneck['suggestion']}")
```

#### Data Classes

##### `PerformancePrediction`
```python
@dataclass
class PerformancePrediction:
    code_id: str
    bottlenecks: List[Dict[str, Any]]
    time_complexity: str
    space_complexity: str
    estimated_runtime: float  # seconds
    estimated_memory: float  # MB
    optimization_opportunities: List[Dict[str, Any]]
    confidence: float
```

---

### CodeGenerator

AI-powered code generation.

```python
from ai_features.generation.code_generator import CodeGenerator

generator = CodeGenerator()
```

#### Methods

##### `generate_boilerplate(template_type: str, params: Dict) -> GeneratedCode`
Generate boilerplate code.

**Parameters:**
- `template_type`: "class", "function", "module", "cli"
- `params`: Template parameters

**Example:**
```python
result = generator.generate_boilerplate("class", {
    "name": "DataProcessor",
    "attributes": ["data", "config"],
    "methods": ["process", "validate"]
})
print(result.code)
```

##### `generate_tests(source_code: str, framework: str) -> GeneratedCode`
Generate test code.

**Parameters:**
- `source_code`: Code to test
- `framework`: "pytest", "unittest"

##### `generate_docstrings(source_code: str, style: str) -> GeneratedCode`
Generate docstrings for code.

**Parameters:**
- `source_code`: Code to document
- `style`: "google", "numpy", "sphinx"

##### `generate_pattern(pattern_name: str, params: Dict) -> GeneratedCode`
Generate design pattern implementation.

**Parameters:**
- `pattern_name`: "singleton", "factory", "builder", "observer", "strategy"
- `params`: Pattern parameters

**Example:**
```python
result = generator.generate_pattern("singleton", {
    "name": "ConfigManager"
})
```

##### `generate_api_client(api_spec: Dict) -> GeneratedCode`
Generate API client from specification.

---

### NeuralSearch

AI-powered code search.

```python
from ai_features.search.neural_search import NeuralSearch

search = NeuralSearch()
```

#### Methods

##### `index_codebase(root_path: Path)`
Index codebase for search.

**Example:**
```python
search.index_codebase(Path("/project"))
```

##### `search_by_functionality(query: str, top_k: int) -> List[SearchResult]`
Search code by describing functionality.

**Example:**
```python
results = search.search_by_functionality(
    "function that implements binary search",
    top_k=10
)
for result in results:
    print(f"{result.entity_name}: {result.score:.2f}")
```

##### `search_by_example(example_code: str, top_k: int) -> List[SearchResult]`
Find similar code by providing example.

##### `search_by_signature(function_signature: str, top_k: int) -> List[SearchResult]`
Search for functions with similar signatures.

#### Data Classes

##### `SearchResult`
```python
@dataclass
class SearchResult:
    file_path: Path
    code_snippet: str
    entity_name: str
    entity_type: str
    score: float
    line_number: int
    metadata: Dict[str, Any]
```

---

### ClaudeIntegration

Integration with Claude API for advanced AI features.

```python
from ai_features.integration.claude_integration import ClaudeIntegration

claude = ClaudeIntegration(api_key="your-api-key")
# Or use environment variable ANTHROPIC_API_KEY
```

#### Methods

##### `explain_code(code: str, context: Optional[Dict], detail_level: str) -> ClaudeResponse`
Get natural language explanation.

**Parameters:**
- `code`: Code to explain
- `context`: Additional context
- `detail_level`: "brief", "detailed", "expert"

**Example:**
```python
response = claude.explain_code(code, detail_level="detailed")
print(response.content)
print(f"Tokens used: {response.usage}")
```

##### `review_code(code: str, focus: Optional[List[str]]) -> ClaudeResponse`
Get AI-powered code review.

**Parameters:**
- `code`: Code to review
- `focus`: ["security", "performance", "style", etc.]

##### `suggest_improvements(code: str, goal: str) -> ClaudeResponse`
Get improvement suggestions.

**Parameters:**
- `goal`: "performance", "readability", "maintainability"

##### `generate_code(description: str, context: Optional[Dict], language: str) -> ClaudeResponse`
Generate code from description.

##### `refactor_code(code: str, refactoring_type: str) -> ClaudeResponse`
Get refactoring suggestions.

**Parameters:**
- `refactoring_type`: "extract_method", "simplify", "general"

##### `analyze_architecture(codebase_summary: Dict, question: str) -> ClaudeResponse`
Analyze architecture and answer questions.

#### Data Classes

##### `ClaudeResponse`
```python
@dataclass
class ClaudeResponse:
    task_type: TaskType
    content: str
    metadata: Dict[str, Any]
    usage: Dict[str, int]
```

---

### AIReviewer

Automated code review.

```python
from ai_features.review.ai_reviewer import AIReviewer

reviewer = AIReviewer()
```

#### Methods

##### `review_file(file_path: Path, focus: Optional[List[str]]) -> ReviewResult`
Review a file.

**Parameters:**
- `file_path`: Path to file
- `focus`: ["bug", "security", "performance", "style", "best_practice"]

**Example:**
```python
result = reviewer.review_file(
    Path("module.py"),
    focus=["security", "performance"]
)
print(f"Quality: {result.overall_quality:.2%}")
print(result.summary)

for issue in result.issues:
    print(f"\n[{issue.severity.value}] {issue.title}")
    print(f"  Line {issue.line_number}: {issue.description}")
    if issue.auto_fixable:
        print(f"  Can be auto-fixed")
```

#### Data Classes

##### `ReviewResult`
```python
@dataclass
class ReviewResult:
    file_path: Path
    issues: List[CodeIssue]
    metrics: Dict[str, Any]
    summary: str
    overall_quality: float  # 0-1
```

##### `CodeIssue`
```python
@dataclass
class CodeIssue:
    category: IssueCategory
    severity: IssueSeverity
    title: str
    description: str
    file_path: Path
    line_number: int
    suggestion: str
    auto_fixable: bool
```

---

### AdaptiveSelector

Learn optimal agent selection from historical data.

```python
from ai_features.agents.adaptive_selector import AdaptiveSelector

selector = AdaptiveSelector()
```

#### Methods

##### `select_agents(task_type: str, context: Dict, max_agents: int) -> List[str]`
Select optimal agents based on performance history.

**Example:**
```python
agents = selector.select_agents(
    "optimization",
    {"language": "python"},
    max_agents=5
)
```

##### `record_execution(agent_name: str, task_type: str, success: bool, duration: float, quality_score: float)`
Record agent execution for learning.

**Example:**
```python
selector.record_execution(
    "code-quality-master",
    "testing",
    success=True,
    duration=5.2,
    quality_score=0.9
)
```

##### `get_agent_stats(agent_name: str) -> Dict[str, Any]`
Get performance statistics for agent.

---

## Enumerations

### AIFeature
Available AI features.
```python
class AIFeature(Enum):
    SEMANTIC_ANALYSIS = "semantic_analysis"
    CODE_EMBEDDING = "code_embedding"
    PERFORMANCE_PREDICTION = "performance_prediction"
    CODE_GENERATION = "code_generation"
    NEURAL_SEARCH = "neural_search"
    CODE_REVIEW = "code_review"
    CODE_EXPLANATION = "code_explanation"
```

### SemanticNode
Types of semantic nodes.
```python
class SemanticNode(Enum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DECORATOR = "decorator"
    PATTERN = "pattern"
    ANTIPATTERN = "antipattern"
```

### DesignPattern
Common design patterns.
```python
class DesignPattern(Enum):
    SINGLETON = "singleton"
    FACTORY = "factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    DECORATOR = "decorator"
    # ... more patterns
```

### CodeSmell
Common code smells.
```python
class CodeSmell(Enum):
    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    GOD_OBJECT = "god_object"
    DUPLICATE_CODE = "duplicate_code"
    # ... more smells
```

### IssueSeverity
Severity levels for code issues.
```python
class IssueSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
```

---

## Common Patterns

### Pattern 1: Comprehensive Code Analysis
```python
from ai_features.ai_system import AISystem

ai = AISystem()

# Analyze semantics
semantic = ai.analyze_code_semantics(Path("code.py"))
print(f"Patterns: {len(semantic.data['patterns'])}")

# Predict performance
perf = ai.predict_performance(code)
print(f"Complexity: {perf.data['time_complexity']}")

# Review code
review = ai.review_code(code)
print(f"Quality: {review.data['overall_quality']}")
```

### Pattern 2: Code Search and Discovery
```python
from ai_features.ai_system import AISystem

ai = AISystem()

# Index codebase
ai.index_codebase_for_search(Path("/project"))

# Search by functionality
results = ai.search_code(
    "function that validates email addresses",
    search_type="functionality"
)

# Search by example
results = ai.search_code(
    sample_code,
    search_type="example"
)
```

### Pattern 3: AI-Assisted Development
```python
from ai_features.ai_system import AISystem

ai = AISystem()

# Get explanation
explanation = ai.explain_code(code, detail_level="detailed")

# Generate improvements
generated = ai.generate_code("refactored", {
    "source_code": code,
    "goal": "performance"
})

# Review changes
review = ai.review_code(generated.data['code'])
```

---

## Error Handling

All API methods can raise exceptions. Always use try-except:

```python
try:
    result = ai_system.analyze_code_semantics(path)
    if not result.success:
        print(f"Analysis failed: {result.data.get('error')}")
except Exception as e:
    print(f"Error: {e}")
```

---

## Performance Tips

1. **Enable Caching**: Set `cache_enabled=True` in config
2. **Batch Processing**: Process multiple files in batches
3. **Incremental Analysis**: Only analyze changed files
4. **Model Selection**: Use appropriate model size for task
5. **Parallel Execution**: Use multiprocessing for large codebases

---

## Version Compatibility

API Version: 1.0.0
Python: >= 3.8
Dependencies: See requirements.txt