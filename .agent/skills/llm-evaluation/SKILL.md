---
name: llm-evaluation
version: "1.0.7"
description: Implement LLM evaluation with automated metrics (BLEU, ROUGE, BERTScore), LLM-as-judge patterns (pointwise, pairwise), human evaluation frameworks, A/B testing with statistical significance, and regression detection. Use when measuring LLM performance, comparing prompts/models, or setting up CI/CD quality gates.
---

# LLM Evaluation

Comprehensive evaluation strategies from automated metrics to human assessment.

## Evaluation Types

| Type | Speed | Cost | Best For |
|------|-------|------|----------|
| Automated Metrics | Fast | Low | Regression testing, CI/CD |
| LLM-as-Judge | Medium | Medium | Quality scoring at scale |
| Human Evaluation | Slow | High | Final validation, edge cases |

## Automated Metrics

### Text Generation

| Metric | Use Case | Library |
|--------|----------|---------|
| BLEU | Translation | nltk |
| ROUGE | Summarization | rouge-score |
| BERTScore | Semantic similarity | bert-score |
| Perplexity | Language model confidence | transformers |

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(reference, hypothesis)
    return {k: v.fmeasure for k, v in scores.items()}
```

### Retrieval (RAG)

| Metric | Measures |
|--------|----------|
| MRR | Mean Reciprocal Rank |
| NDCG | Graded relevance |
| Precision@K | Relevant in top K |
| Recall@K | Coverage in top K |

### Custom Metrics

```python
def calculate_groundedness(response, context):
    """Check if response is grounded in context."""
    nli = pipeline("text-classification", model="microsoft/deberta-large-mnli")
    result = nli(f"{context} [SEP] {response}")[0]
    return result['score'] if result['label'] == 'ENTAILMENT' else 0.0
```

## LLM-as-Judge

### Pointwise Scoring

```python
def llm_judge_quality(response, question):
    prompt = f"""Rate the response on 1-10 for:
1. Accuracy (factually correct)
2. Helpfulness (answers the question)
3. Clarity (well-written)

Question: {question}
Response: {response}

Return JSON: {{"accuracy": <1-10>, "helpfulness": <1-10>, "clarity": <1-10>}}
"""
    return openai.chat(model="gpt-4", messages=[{"role": "user", "content": prompt}])
```

### Pairwise Comparison

```python
def compare_responses(question, response_a, response_b):
    prompt = f"""Compare these responses:

Question: {question}
Response A: {response_a}
Response B: {response_b}

Which is better? Return: {{"winner": "A"|"B"|"tie", "reasoning": "..."}}
"""
    return openai.chat(model="gpt-4", messages=[{"role": "user", "content": prompt}])
```

## Human Evaluation

### Rating Dimensions

| Dimension | Description |
|-----------|-------------|
| Accuracy | Factually correct |
| Relevance | Answers the question |
| Coherence | Logical flow |
| Fluency | Natural language |
| Safety | No harmful content |

### Inter-Rater Agreement

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(rater1_scores, rater2_scores)
# κ < 0.2: Poor, 0.4-0.6: Moderate, 0.6-0.8: Substantial, >0.8: Near-perfect
```

## A/B Testing

```python
from scipy import stats
import numpy as np

class ABTest:
    def __init__(self):
        self.a_scores, self.b_scores = [], []

    def add_result(self, variant, score):
        (self.a_scores if variant == "A" else self.b_scores).append(score)

    def analyze(self, alpha=0.05):
        t_stat, p_value = stats.ttest_ind(self.a_scores, self.b_scores)
        pooled_std = np.sqrt((np.std(self.a_scores)**2 + np.std(self.b_scores)**2) / 2)
        cohens_d = (np.mean(self.b_scores) - np.mean(self.a_scores)) / pooled_std

        return {
            "p_value": p_value,
            "significant": p_value < alpha,
            "effect_size": "small" if abs(cohens_d) < 0.5 else "large",
            "winner": "B" if np.mean(self.b_scores) > np.mean(self.a_scores) else "A"
        }
```

## Regression Detection

```python
class RegressionDetector:
    def __init__(self, baseline_results, threshold=0.05):
        self.baseline = baseline_results
        self.threshold = threshold

    def check(self, new_results):
        regressions = []
        for metric, baseline in self.baseline.items():
            new = new_results.get(metric)
            if new and (new - baseline) / baseline < -self.threshold:
                regressions.append({
                    "metric": metric,
                    "baseline": baseline,
                    "current": new,
                    "change": (new - baseline) / baseline
                })
        return {"has_regression": len(regressions) > 0, "regressions": regressions}
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Multiple metrics | Diverse metrics for comprehensive view |
| Representative data | Test on real-world, diverse examples |
| Baselines | Always compare against baseline |
| Statistical rigor | Proper significance tests |
| Continuous eval | Integrate into CI/CD |
| Error analysis | Investigate failures |

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| Single metric | Goodhart's Law - gaming |
| Small sample size | Unreliable conclusions |
| Data contamination | Testing on training data |
| Ignoring variance | No statistical uncertainty |
| Metric mismatch | Not aligned with business goals |

## Checklist

- [ ] Automated metrics for CI/CD
- [ ] LLM-as-judge for quality scoring
- [ ] Human evaluation for edge cases
- [ ] Statistical significance tests
- [ ] Regression detection before deploy
- [ ] Error analysis and debugging
