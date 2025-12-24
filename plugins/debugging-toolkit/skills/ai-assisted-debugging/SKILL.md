---
name: ai-assisted-debugging
version: "1.0.5"
maturity: "5-Expert"
specialization: AI-Powered Debugging & RCA
description: Leverage AI/LLMs for automated stack trace analysis, intelligent root cause detection, and ML-driven log correlation in distributed systems. Use when analyzing Python/JavaScript/Go runtime exceptions, debugging Kubernetes pod failures, implementing automated anomaly detection on logs/metrics, or correlating git commits with production incidents.
---

# AI-Assisted Debugging

AI-powered debugging framework for automated RCA and intelligent log correlation.

---

## Technique Selection

| Technique | Use Case | Tools |
|-----------|----------|-------|
| LLM Stack Trace Analysis | Error explanation + fix suggestions | GPT-5, Claude Sonnet 4.5 |
| ML Log Anomaly Detection | Large log volume analysis | Isolation Forest, sklearn |
| Distributed Trace Analysis | Microservice bottlenecks | OpenTelemetry, Jaeger |
| Change Correlation | Recent deploy → incident | Git, deployment logs |
| Predictive Detection | Failure forecasting | ARIMA, Prophet |

---

## LLM Stack Trace Analysis

```python
import openai
import json

class AIDebugAssistant:
    def __init__(self, api_key: str, model: str = "gpt-5"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def analyze_stack_trace(self, error_trace: str, context_code: str = "") -> dict:
        prompt = f"""Analyze this error:

ERROR TRACE:
{error_trace}

RELEVANT CODE:
{context_code}

Provide JSON with: root_cause, location, fixes (list), prevention (list)"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Expert debugging assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def generate_debug_statements(self, function_code: str, issue: str) -> str:
        prompt = f"""Add strategic debug logging for: {issue}

CODE:
{function_code}

Add: entry/exit logs, variable states, conditional branches, exception handling."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
```

---

## ML Log Anomaly Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class LogAnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        window = 5
        df['response_time_mean'] = df['response_time'].rolling(window).mean()
        df['error_rate'] = (df['error_count'].rolling(window).sum() /
                           df['request_count'].rolling(window).sum())

        features = ['response_time', 'response_time_mean', 'error_rate',
                   'cpu_usage', 'memory_usage']
        return df[features].fillna(0).values

    def train(self, normal_logs: pd.DataFrame):
        features = self.extract_features(normal_logs)
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled)

    def detect(self, logs: pd.DataFrame) -> pd.DataFrame:
        features = self.extract_features(logs)
        scaled = self.scaler.transform(features)
        logs['anomaly'] = self.model.predict(scaled) == -1
        logs['anomaly_score'] = self.model.score_samples(scaled)
        return logs

# Usage
detector = LogAnomalyDetector()
detector.train(normal_logs)
anomalies = detector.detect(production_logs)
```

---

## Kubernetes Pod Debugging

```bash
#!/bin/bash
POD=$1; NS=${2:-default}

echo "=== Pod Status ===" && kubectl get pod $POD -n $NS -o wide
echo "=== Events ===" && kubectl get events -n $NS --field-selector involvedObject.name=$POD | tail -10
echo "=== Logs ===" && kubectl logs $POD -n $NS --tail=50
echo "=== Resources ===" && kubectl top pod $POD -n $NS

# AI Analysis
python3 <<PYTHON
import anthropic, os
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4.5-20250514",
    max_tokens=4096,
    system="Kubernetes debugging expert.",
    messages=[{"role": "user", "content": f"Analyze pod $POD issues and suggest fixes."}]
)
print(response.content[0].text)
PYTHON
```

---

## Docker Container Debugging

```python
import docker

def debug_container(container_id: str):
    client = docker.from_env()
    container = client.containers.get(container_id)

    issues = []

    # OOMKilled check
    if container.attrs['State'].get('OOMKilled'):
        issues.append({'type': 'OOM', 'fix': 'Increase memory limit'})

    # Exit code analysis
    exit_code = container.attrs['State'].get('ExitCode', 0)
    if exit_code != 0:
        issues.append({'type': 'EXIT_CODE', 'code': exit_code})

    # Memory usage check
    stats = container.stats(stream=False)
    mem_usage = stats['memory_stats']['usage'] / stats['memory_stats']['limit']
    if mem_usage > 0.9:
        issues.append({'type': 'HIGH_MEMORY', 'usage': f'{mem_usage:.1%}'})

    # Health check status
    health = container.attrs.get('State', {}).get('Health', {})
    if health.get('Status') == 'unhealthy':
        issues.append({'type': 'UNHEALTHY', 'logs': health.get('Log', [])[-3:]})

    return issues
```

---

## OpenTelemetry Trace Analysis

```python
class TraceAnalyzer:
    def __init__(self):
        self.slow_spans = []

    def analyze(self, spans: list) -> str:
        for span in spans:
            duration_ms = (span['end_time'] - span['start_time']) * 1000
            if duration_ms > 1000:  # > 1 second
                self.slow_spans.append({
                    'name': span['name'],
                    'duration_ms': duration_ms,
                    'attributes': span.get('attributes', {})
                })

        report = "Performance Bottlenecks:\n"
        for span in sorted(self.slow_spans, key=lambda x: -x['duration_ms'])[:5]:
            report += f"\n{span['name']}: {span['duration_ms']:.0f}ms"
            if 'db.system' in span['attributes']:
                report += " → Optimize query or add caching"
            elif 'http.method' in span['attributes']:
                report += " → Consider async or CDN"
        return report
```

---

## Automated RCA Pipeline

```python
class RCAPipeline:
    def __init__(self, ai_assistant, log_detector):
        self.ai = ai_assistant
        self.detector = log_detector

    def analyze_incident(self, incident: dict) -> dict:
        results = {'timestamp': incident['timestamp'], 'analysis': {}}

        # 1. Correlate recent changes
        recent_changes = self.filter_changes(incident['changes'], incident['timestamp'])
        if recent_changes:
            results['analysis']['suspected_cause'] = 'Recent deployment'

        # 2. Detect log anomalies
        anomalies = self.detector.detect(pd.DataFrame(incident['logs']))
        results['analysis']['anomalies'] = anomalies[anomalies['anomaly']].to_dict('records')

        # 3. AI root cause analysis
        results['analysis']['ai_rca'] = self.ai.analyze_stack_trace(
            incident.get('error_trace', ''))

        # 4. Generate recommendations
        results['recommendations'] = self.generate_fixes(results['analysis'])
        return results

    def filter_changes(self, changes, incident_time):
        import dateutil.parser
        incident_dt = dateutil.parser.parse(incident_time)
        return [c for c in changes
                if (incident_dt - dateutil.parser.parse(c['timestamp'])).total_seconds() < 3600]
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| **Prompt engineering** | Include language, framework, environment context |
| **Temperature** | Low (0.2) for deterministic debugging |
| **Context** | Provide error + code + logs together |
| **Correlation** | Check recent deploys/configs before deep analysis |
| **Anomaly thresholds** | Set based on normal baseline (10% contamination) |
| **Trace analysis** | Focus on spans > P95 latency |

---

## Debugging Prompt Template

```python
def debug_prompt(context: dict) -> str:
    return f"""Error in {context['language']} / {context['framework']}

ERROR:
{context['error']}

CODE:
{context.get('code', 'N/A')}

LOGS:
{context.get('logs', 'N/A')}

Provide:
1. Root cause (WHY, not just WHAT)
2. 2-3 specific fixes with code
3. Tests to prevent regression
4. Monitoring/alerts to catch earlier"""
```

---

## Checklist

- [ ] LLM configured for stack trace analysis
- [ ] Anomaly detector trained on normal logs
- [ ] K8s/Docker debugging scripts ready
- [ ] OpenTelemetry trace collection enabled
- [ ] Change correlation integrated (git, deploys)
- [ ] RCA pipeline connects all data sources
- [ ] Prompts include sufficient context

---

**Version**: 1.0.5
