# Prompt Evaluation

Testing protocols, metrics, and production monitoring for prompt engineering.

## Testing Protocols

### LLM-as-Judge Framework

```python
judge_prompt = """Evaluate AI response quality.

## Original Task
{prompt}

## Response
{output}

## Rate 1-10 with justification:
1. TASK COMPLETION: Fully addressed?
2. ACCURACY: Factually correct?
3. REASONING: Logical and structured?
4. FORMAT: Matches requirements?
5. SAFETY: Unbiased and safe?

Overall: []/50
Recommendation: Accept/Revise/Reject"""
```

### Automated Testing Suite

```python
class PromptEvaluator:
    def evaluate_prompt(self, prompt: str, test_cases: List[Dict]):
        """Run automated evaluation"""
        results = {
            'total_tests': len(test_cases),
            'passed': 0,
            'failed': 0,
            'metrics': {
                'accuracy': [],
                'relevance': [],
                'coherence': []
            }
        }
        
        for test_case in test_cases:
            response = self.llm.generate(prompt.format(**test_case['input']))
            score = self.score_response(response, test_case['expected'])
            
            if score['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1
            
            for metric, value in score.items():
                results['metrics'][metric].append(value)
        
        return results
```

## A/B Testing

### A/B Test Framework

```python
class ABTestingFramework:
    def create_ab_test(self, 
                      test_name: str,
                      variants: List[Dict[str, Any]],
                      metrics: List[str]):
        """Create A/B test for prompt improvements"""
        test = {
            'id': generate_test_id(),
            'name': test_name,
            'variants': variants,
            'metrics': metrics,
            'allocation': self.calculate_traffic_allocation(variants),
            'duration': self.estimate_test_duration(metrics)
        }
        
        self.deploy_test(test)
        return test
    
    async def analyze_test_results(self, test_id: str):
        """Analyze A/B test results"""
        data = await self.collect_test_data(test_id)
        
        results = {}
        for metric in data['metrics']:
            analysis = self.statistical_analysis(
                data['control'][metric],
                data['variant'][metric]
            )
            
            results[metric] = {
                'control_mean': analysis['control_mean'],
                'variant_mean': analysis['variant_mean'],
                'lift': analysis['lift'],
                'p_value': analysis['p_value'],
                'significant': analysis['p_value'] < 0.05
            }
        
        return results
```

## Production Monitoring

### Prompt Performance Metrics

```python
class PromptMetrics:
    def track_metrics(self):
        """Monitor prompt performance in production"""
        return {
            'quality_metrics': {
                'success_rate': 'percentage of successful completions',
                'user_satisfaction': 'feedback ratings',
                'task_completion': 'goal achievement rate'
            },
            'efficiency_metrics': {
                'avg_tokens': 'token usage per request',
                'response_time': 'latency p50/p95/p99',
                'cost_per_request': 'API cost tracking'
            },
            'safety_metrics': {
                'harmful_outputs': 'safety filter triggers',
                'hallucination_rate': 'factual accuracy checks',
                'bias_detection': 'fairness metrics'
            }
        }
```

### Continuous Monitoring

```python
def monitor_prompt_quality():
    """Continuous quality monitoring"""
    metrics = {
        'hourly_success_rate': track_success_rate(window='1h'),
        'token_efficiency': track_token_usage(),
        'user_feedback': collect_user_ratings(),
        'error_patterns': identify_failure_modes()
    }
    
    if metrics['hourly_success_rate'] < 0.85:
        alert('Prompt quality degradation detected')
        trigger_review_process()
    
    return metrics
```

## Prompt Versioning

### Version Control

```python
class PromptVersion:
    def __init__(self, base_prompt):
        self.version = "1.0.0"
        self.base_prompt = base_prompt
        self.variants = {}
        self.performance_history = []
    
    def rollout_strategy(self):
        return {
            "canary": 5,  # 5% traffic
            "staged": [10, 25, 50, 100],  # staged rollout
            "rollback_threshold": 0.8,  # rollback if success < 80%
            "monitoring_period": "24h"
        }
```

---

**See Also**:
- [Prompt Patterns](./prompt-patterns.md) - Optimization techniques
- [Prompt Examples](./prompt-examples.md) - Reference implementations
- Command: `/prompt-optimize`
