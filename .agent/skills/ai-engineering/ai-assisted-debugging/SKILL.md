---
name: ai-assisted-debugging
version: "1.0.7"
maturity: "5-Expert"
specialization: AI-Powered Debugging & RCA
description: AI/LLM for automated stack trace analysis, intelligent RCA, ML log correlation in distributed systems. Use for Python/JS/Go exceptions, K8s pod failures, automated anomaly detection on logs/metrics, correlating git commits with production incidents.
---

# AI-Assisted Debugging

## Technique Selection

| Technique | Use Case | Tools |
|-----------|----------|-------|
| LLM Stack Analysis | Error explanation + fixes | GPT-5, Claude Sonnet 4.5 |
| ML Anomaly Detection | Large log volumes | Isolation Forest, sklearn |
| Trace Analysis | Microservice bottlenecks | OpenTelemetry, Jaeger |
| Change Correlation | Recent deploy → incident | Git, deploy logs |
| Predictive | Failure forecasting | ARIMA, Prophet |

## LLM Stack Analysis

```python
import openai

class AIDebugAssistant:
    def __init__(self, api_key, model="gpt-5"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def analyze(self, error, code=""):
        prompt = f"Analyze:\nERROR:\n{error}\nCODE:\n{code}\nProvide JSON: root_cause, location, fixes, prevention"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":"Expert debugger"},{"role":"user","content":prompt}],
            temperature=0.2, response_format={"type":"json_object"}
        )
        return json.loads(resp.choices[0].message.content)
```

## ML Log Anomaly Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class LogAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def extract_features(self, df):
        df['rt_mean'] = df['response_time'].rolling(5).mean()
        df['err_rate'] = df['error_count'].rolling(5).sum() / df['request_count'].rolling(5).sum()
        return df[['response_time','rt_mean','err_rate','cpu','memory']].fillna(0).values

    def train(self, normal_logs):
        self.model.fit(self.scaler.fit_transform(self.extract_features(normal_logs)))

    def detect(self, logs):
        scaled = self.scaler.transform(self.extract_features(logs))
        logs['anomaly'] = self.model.predict(scaled) == -1
        logs['score'] = self.model.score_samples(scaled)
        return logs
```

## K8s Pod Debugging

```bash
#!/bin/bash
POD=$1; NS=${2:-default}
echo "=== Status ===" && kubectl get pod $POD -n $NS -o wide
echo "=== Events ===" && kubectl get events -n $NS --field-selector involvedObject.name=$POD | tail -10
echo "=== Logs ===" && kubectl logs $POD -n $NS --tail=50
echo "=== Resources ===" && kubectl top pod $POD -n $NS
```

## Docker Container Debug

```python
import docker

def debug_container(cid):
    c = docker.from_env().containers.get(cid)
    issues = []
    if c.attrs['State'].get('OOMKilled'): issues.append({'type':'OOM','fix':'Increase memory'})
    exit_code = c.attrs['State'].get('ExitCode', 0)
    if exit_code != 0: issues.append({'type':'EXIT','code':exit_code})
    stats = c.stats(stream=False)
    mem_usage = stats['memory_stats']['usage'] / stats['memory_stats']['limit']
    if mem_usage > 0.9: issues.append({'type':'HIGH_MEM','usage':f'{mem_usage:.1%}'})
    health = c.attrs.get('State',{}).get('Health',{})
    if health.get('Status') == 'unhealthy': issues.append({'type':'UNHEALTHY','logs':health.get('Log',[])[-3:]})
    return issues
```

## OpenTelemetry Trace Analysis

```python
class TraceAnalyzer:
    def analyze(self, spans):
        slow = []
        for s in spans:
            dur_ms = (s['end_time'] - s['start_time']) * 1000
            if dur_ms > 1000: slow.append({'name':s['name'],'dur_ms':dur_ms,'attrs':s.get('attributes',{})})
        report = "Bottlenecks:\n"
        for s in sorted(slow, key=lambda x: -x['dur_ms'])[:5]:
            report += f"\n{s['name']}: {s['dur_ms']:.0f}ms"
            if 'db.system' in s['attrs']: report += " → Optimize query/cache"
            elif 'http.method' in s['attrs']: report += " → Async/CDN"
        return report
```

## Automated RCA Pipeline

```python
class RCAPipeline:
    def __init__(self, ai, detector):
        self.ai, self.detector = ai, detector

    def analyze_incident(self, incident):
        result = {'timestamp': incident['timestamp'], 'analysis': {}}
        # Recent changes
        recent = [c for c in incident['changes']
                  if (incident_dt - parse(c['timestamp'])).seconds < 3600]
        if recent: result['analysis']['suspected'] = 'Recent deployment'
        # Anomalies
        anomalies = self.detector.detect(pd.DataFrame(incident['logs']))
        result['analysis']['anomalies'] = anomalies[anomalies['anomaly']].to_dict('records')
        # AI RCA
        result['analysis']['ai_rca'] = self.ai.analyze(incident.get('error_trace',''))
        result['recommendations'] = self.generate_fixes(result['analysis'])
        return result
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Prompt engineering | Include lang, framework, env context |
| Temperature | Low (0.2) for deterministic |
| Context | Error + code + logs together |
| Correlation | Check deploys/configs first |
| Anomaly thresholds | 10% contamination baseline |
| Trace analysis | Focus on >P95 latency spans |

## Debug Prompt Template

```python
def debug_prompt(ctx):
    return f"""Error in {ctx['language']}/{ctx['framework']}
ERROR: {ctx['error']}
CODE: {ctx.get('code','N/A')}
LOGS: {ctx.get('logs','N/A')}
Provide: 1) Root cause (WHY), 2) Fixes with code, 3) Tests, 4) Monitoring"""
```

## Checklist

- [ ] LLM for stack trace analysis
- [ ] Anomaly detector trained on normal logs
- [ ] K8s/Docker debug scripts ready
- [ ] OpenTelemetry traces enabled
- [ ] Change correlation (git, deploys)
- [ ] RCA pipeline connects all sources
- [ ] Prompts have sufficient context
