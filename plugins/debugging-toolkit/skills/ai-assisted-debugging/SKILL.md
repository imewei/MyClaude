---
name: ai-assisted-debugging
description: AI-powered debugging with GitHub Copilot, LLM-driven root cause analysis, automated log correlation, anomaly detection, and modern debugging tools (GDB, VS Code Debugger, Chrome DevTools) for distributed systems
tools: Read, Write, Bash, Grep, python, gdb, lldb, kubectl, docker
integration: Use for accelerating debugging with AI assistance and automated RCA
---

# AI-Assisted Debugging Mastery

Complete framework for leveraging AI to accelerate debugging, automate root cause analysis, and integrate observability data for real-time insights in distributed systems.

## When to Use This Skill

- **AI-powered code debugging**: GitHub Copilot, Cursor, CodeWhisperer for hypothesis generation
- **Automated root cause analysis**: ML-driven log correlation and anomaly detection
- **Stack trace interpretation**: LLM-based error analysis and fix suggestions
- **Distributed system debugging**: Kubernetes, Docker, microservices troubleshooting
- **Observability integration**: OpenTelemetry, Prometheus, Datadog for telemetry analysis
- **Predictive failure detection**: Time-series forecasting and anomaly detection

## Core AI Debugging Techniques

### 1. LLM-Driven Stack Trace Analysis

```python
import openai
import traceback
import sys

class AIDebugAssistant:
    """AI-powered debugging assistant using latest LLMs (2025)."""

    def __init__(self, api_key: str, model: str = "gpt-5", provider: str = "openai"):
        """
        Initialize AI debugging assistant.

        Parameters
        ----------
        api_key : str
            API key for the LLM provider
        model : str
            Model to use: "gpt-5" (OpenAI), "claude-sonnet-4.5" (Anthropic)
        provider : str
            LLM provider: "openai" or "anthropic"
        """
        self.provider = provider
        self.model = model

        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def analyze_stack_trace(self, error_trace: str, context_code: str = "") -> dict:
        """
        Analyze stack trace with AI to identify root cause and suggest fixes.

        Parameters
        ----------
        error_trace : str
            Full error traceback
        context_code : str
            Relevant code context around the error

        Returns
        -------
        dict
            Analysis with root cause, fix suggestions, and prevention tips
        """
        prompt = f"""
You are an expert debugging assistant. Analyze this error:

ERROR TRACE:
{error_trace}

RELEVANT CODE:
{context_code}

Provide:
1. Root cause explanation
2. Specific line/function causing the issue
3. 2-3 concrete fix suggestions with code
4. Prevention recommendations

Format as JSON with keys: root_cause, location, fixes (list), prevention (list)
"""

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert debugging assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower for more deterministic debugging
                response_format={"type": "json_object"}
            )
            import json
            return json.loads(response.choices[0].message.content)

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.2,
                system="You are an expert debugging assistant. Always respond with valid JSON.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            import json
            return json.loads(response.content[0].text)

    def generate_debug_statements(self, function_code: str, suspected_issue: str) -> str:
        """
        Generate strategic debug logging statements.

        Parameters
        ----------
        function_code : str
            Function to instrument with debug statements
        suspected_issue : str
            Description of the suspected issue

        Returns
        -------
        str
            Instrumented code with debug statements
        """
        prompt = f"""
Add strategic debug logging to diagnose this issue: {suspected_issue}

ORIGINAL CODE:
{function_code}

Add debug statements that:
1. Log function entry/exit
2. Log variable states at critical points
3. Log conditional branches taken
4. Catch and log exceptions

Return complete instrumented code with logging.
"""

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

# Usage example
def buggy_function(data):
    """Example function with a bug."""
    try:
        result = []
        for item in data:
            # Bug: division by zero if item['value'] == 0
            processed = item['total'] / item['value']
            result.append(processed)
        return result
    except Exception as e:
        trace = traceback.format_exc()
        print(f"Error occurred: {trace}")

        # AI analysis with GPT-5 (OpenAI)
        assistant = AIDebugAssistant(
            api_key="YOUR_OPENAI_API_KEY",
            model="gpt-5",
            provider="openai"
        )

        # Or use Claude Sonnet 4.5 (Anthropic)
        # assistant = AIDebugAssistant(
        #     api_key="YOUR_ANTHROPIC_API_KEY",
        #     model="claude-sonnet-4.5-20250514",
        #     provider="anthropic"
        # )

        analysis = assistant.analyze_stack_trace(
            error_trace=trace,
            context_code=inspect.getsource(buggy_function)
        )

        print("AI Analysis:")
        print(f"Root Cause: {analysis['root_cause']}")
        print(f"Location: {analysis['location']}")
        print("Suggested Fixes:")
        for i, fix in enumerate(analysis['fixes'], 1):
            print(f"  {i}. {fix}")

        raise
```

### 2. Automated Log Correlation with ML

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class LogAnomalyDetector:
    """ML-based log anomaly detection for RCA."""

    def __init__(self, contamination=0.1):
        """
        Initialize anomaly detector.

        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies (default: 10%)
        """
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.feature_names = None

    def extract_log_features(self, log_df: pd.DataFrame) -> np.ndarray:
        """
        Extract numerical features from log data.

        Parameters
        ----------
        log_df : pd.DataFrame
            DataFrame with columns: timestamp, level, response_time,
            error_count, request_count, cpu_usage, memory_usage

        Returns
        -------
        np.ndarray
            Feature matrix for ML model
        """
        features = []

        # Time-based features
        log_df['hour'] = pd.to_datetime(log_df['timestamp']).dt.hour
        log_df['minute'] = pd.to_datetime(log_df['timestamp']).dt.minute

        # Rolling statistics (5-minute windows)
        window = 5
        log_df['response_time_mean'] = log_df['response_time'].rolling(window).mean()
        log_df['response_time_std'] = log_df['response_time'].rolling(window).std()
        log_df['error_rate'] = log_df['error_count'].rolling(window).sum() / \
                                log_df['request_count'].rolling(window).sum()

        # Select features
        feature_columns = [
            'hour', 'response_time', 'response_time_mean', 'response_time_std',
            'error_count', 'request_count', 'error_rate',
            'cpu_usage', 'memory_usage'
        ]

        self.feature_names = feature_columns
        return log_df[feature_columns].fillna(0).values

    def train(self, normal_log_df: pd.DataFrame):
        """Train on normal (non-anomalous) logs."""
        features = self.extract_log_features(normal_log_df)
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)

    def detect_anomalies(self, log_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in new log data.

        Returns
        -------
        pd.DataFrame
            Original data with 'anomaly' and 'anomaly_score' columns
        """
        features = self.extract_log_features(log_df)
        features_scaled = self.scaler.transform(features)

        # Predict (-1 for anomalies, 1 for normal)
        predictions = self.model.predict(features_scaled)
        scores = self.model.score_samples(features_scaled)

        log_df['anomaly'] = predictions == -1
        log_df['anomaly_score'] = scores

        return log_df

    def get_anomaly_explanation(self, anomaly_row: pd.Series) -> str:
        """Generate human-readable explanation of anomaly."""
        explanations = []

        # Check each feature for unusual values
        if anomaly_row['response_time'] > anomaly_row['response_time_mean'] * 2:
            explanations.append(
                f"Response time ({anomaly_row['response_time']:.2f}ms) is "
                f"2x higher than recent average"
            )

        if anomaly_row['error_rate'] > 0.05:
            explanations.append(
                f"Error rate ({anomaly_row['error_rate']:.1%}) exceeds threshold"
            )

        if anomaly_row['cpu_usage'] > 90:
            explanations.append(
                f"CPU usage ({anomaly_row['cpu_usage']:.1f}%) critically high"
            )

        if anomaly_row['memory_usage'] > 90:
            explanations.append(
                f"Memory usage ({anomaly_row['memory_usage']:.1f}%) critically high"
            )

        return " | ".join(explanations) if explanations else "Multiple features anomalous"

# Usage
detector = LogAnomalyDetector(contamination=0.1)

# Train on normal logs
normal_logs = pd.read_csv('normal_logs.csv')
detector.train(normal_logs)

# Detect anomalies in new logs
new_logs = pd.read_csv('production_logs.csv')
results = detector.detect_anomalies(new_logs)

# Report anomalies
anomalies = results[results['anomaly']]
print(f"Found {len(anomalies)} anomalies:")
for idx, row in anomalies.iterrows():
    explanation = detector.get_anomaly_explanation(row)
    print(f"  {row['timestamp']}: {explanation}")
```

### 3. Modern Debugging Tool Integration

#### GDB/LLDB with Python Scripting

```python
# GDB Python script for automated debugging
import gdb

class SmartBreakpoint(gdb.Breakpoint):
    """Intelligent breakpoint with conditional logging."""

    def __init__(self, location, condition=None, log_vars=None):
        super().__init__(location)
        self.condition_str = condition
        self.log_vars = log_vars or []
        self.hit_count = 0

    def stop(self):
        """Called when breakpoint is hit."""
        self.hit_count += 1

        # Log variable states
        frame = gdb.selected_frame()
        print(f"\n--- Breakpoint Hit #{self.hit_count} ---")
        print(f"Location: {frame.name()} at {frame.find_sal().symtab.filename}:"
              f"{frame.find_sal().line}")

        for var in self.log_vars:
            try:
                value = gdb.parse_and_eval(var)
                print(f"  {var} = {value}")
            except gdb.error as e:
                print(f"  {var} = <unavailable: {e}>")

        # AI-driven decision: continue or stop?
        # For demo, stop every 10th hit
        return self.hit_count % 10 == 0

# Set intelligent breakpoints
SmartBreakpoint(
    "my_function",
    condition="x > 100",
    log_vars=["x", "y", "result"]
)

gdb.execute("run")
```

#### VS Code Debugger Configuration

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: AI-Assisted Debug",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "logToFile": true,
      "postDebugTask": "analyze-debug-logs"
    }
  ],
  "tasks": [
    {
      "label": "analyze-debug-logs",
      "type": "shell",
      "command": "python",
      "args": [
        "analyze_debug_logs.py",
        "${workspaceFolder}/.vscode/debug.log"
      ],
      "problemMatcher": []
    }
  ]
}
```

### 4. Distributed System Debugging

#### Kubernetes Pod Debugging

```bash
#!/bin/bash
# AI-assisted Kubernetes debugging script

POD_NAME=$1
NAMESPACE=${2:-default}

echo "🔍 Debugging pod: $POD_NAME in namespace: $NAMESPACE"

# 1. Get pod status
echo "\n=== Pod Status ==="
kubectl get pod $POD_NAME -n $NAMESPACE -o wide

# 2. Get recent events
echo "\n=== Recent Events ==="
kubectl get events -n $NAMESPACE --field-selector involvedObject.name=$POD_NAME \
  --sort-by='.lastTimestamp' | tail -10

# 3. Get logs
echo "\n=== Recent Logs ==="
kubectl logs $POD_NAME -n $NAMESPACE --tail=50

# 4. Check resource usage
echo "\n=== Resource Usage ==="
kubectl top pod $POD_NAME -n $NAMESPACE

# 5. Describe pod for detailed info
echo "\n=== Pod Details ==="
kubectl describe pod $POD_NAME -n $NAMESPACE

# 6. AI Analysis
echo "\n=== AI Root Cause Analysis ==="
# Collect all debug info
DEBUG_DATA=$(cat <<EOF
Pod: $POD_NAME
Status: $(kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.status.phase}')
Events: $(kubectl get events -n $NAMESPACE --field-selector involvedObject.name=$POD_NAME -o json)
Logs: $(kubectl logs $POD_NAME -n $NAMESPACE --tail=100 2>&1)
Resources: $(kubectl top pod $POD_NAME -n $NAMESPACE 2>&1)
EOF
)

# Call AI assistant (requires API key)
python3 <<PYTHON
import os
import json

# Choose your LLM provider (OpenAI GPT-5 or Anthropic Claude Sonnet 4.5)
PROVIDER = os.environ.get('LLM_PROVIDER', 'anthropic')  # or 'openai'

debug_data = """$DEBUG_DATA"""

if PROVIDER == 'openai':
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a Kubernetes debugging expert. Analyze pod issues and suggest fixes."},
            {"role": "user", "content": f"Analyze this Kubernetes pod debug data and suggest fixes:\n\n{debug_data}"}
        ],
        temperature=0.2
    )
    print(response.choices[0].message.content)

elif PROVIDER == 'anthropic':
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    response = client.messages.create(
        model="claude-sonnet-4.5-20250514",
        max_tokens=4096,
        temperature=0.2,
        system="You are a Kubernetes debugging expert. Analyze pod issues and suggest fixes.",
        messages=[
            {"role": "user", "content": f"Analyze this Kubernetes pod debug data and suggest fixes:\n\n{debug_data}"}
        ]
    )
    print(response.content[0].text)
PYTHON
```

#### Docker Container Debugging

```python
import docker
import json

def debug_container_with_ai(container_id: str):
    """AI-assisted Docker container debugging."""
    client = docker.from_env()

    try:
        container = client.containers.get(container_id)

        # Collect debug information
        debug_info = {
            'status': container.status,
            'logs': container.logs(tail=100).decode('utf-8'),
            'stats': container.stats(stream=False),
            'top': container.top(),
            'attrs': container.attrs
        }

        # Check for common issues
        issues = []

        # 1. OOMKilled?
        if container.attrs['State'].get('OOMKilled'):
            issues.append({
                'type': 'OOM',
                'message': 'Container was killed due to out-of-memory',
                'fix': 'Increase memory limit in docker-compose or deployment'
            })

        # 2. Exit code analysis
        exit_code = container.attrs['State'].get('ExitCode', 0)
        if exit_code != 0:
            issues.append({
                'type': 'EXIT_CODE',
                'code': exit_code,
                'message': f'Container exited with code {exit_code}',
                'fix': 'Check application logs for errors'
            })

        # 3. Resource constraints
        stats = debug_info['stats']
        memory_usage = stats['memory_stats']['usage'] / stats['memory_stats']['limit']
        if memory_usage > 0.9:
            issues.append({
                'type': 'HIGH_MEMORY',
                'usage': f'{memory_usage:.1%}',
                'message': 'Memory usage above 90%',
                'fix': 'Increase memory limit or optimize application'
            })

        # 4. Health check failures
        health = container.attrs.get('State', {}).get('Health', {})
        if health.get('Status') == 'unhealthy':
            issues.append({
                'type': 'UNHEALTHY',
                'message': 'Health check failing',
                'logs': health.get('Log', [])[-3:],  # Last 3 health check logs
                'fix': 'Review health check configuration'
            })

        # AI-powered RCA
        if issues:
            print("🔍 Detected Issues:")
            for issue in issues:
                print(f"\n  [{issue['type']}] {issue['message']}")
                print(f"  Fix: {issue['fix']}")

            # Call AI for deeper analysis (GPT-5 or Claude Sonnet 4.5)
            provider = os.environ.get('LLM_PROVIDER', 'anthropic')

            prompt = f"""
Analyze this Docker container debugging data:

CONTAINER: {container_id}
STATUS: {container.status}
ISSUES DETECTED: {json.dumps(issues, indent=2)}
LOGS:
{debug_info['logs']}

Provide:
1. Root cause analysis
2. Step-by-step fix instructions
3. Prevention recommendations
"""

            if provider == 'openai':
                from openai import OpenAI
                client_ai = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

                response = client_ai.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": "You are a Docker debugging expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                analysis = response.choices[0].message.content

            elif provider == 'anthropic':
                import anthropic
                client_ai = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

                response = client_ai.messages.create(
                    model="claude-sonnet-4.5-20250514",
                    max_tokens=4096,
                    temperature=0.2,
                    system="You are a Docker debugging expert.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                analysis = response.content[0].text

            print("\n🤖 AI Analysis:")
            print(analysis)

        else:
            print("✅ No obvious issues detected")

    except docker.errors.NotFound:
        print(f"❌ Container {container_id} not found")
    except Exception as e:
        print(f"❌ Error debugging container: {e}")

# Usage
debug_container_with_ai("my-app-container")
```

## Observability Integration

### OpenTelemetry Trace Analysis

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
import time

class AITraceAnalyzer:
    """AI-powered trace analysis for performance debugging."""

    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.slow_spans = []

    def analyze_trace(self, trace_id: str):
        """
        Analyze distributed trace for performance issues.

        Parameters
        ----------
        trace_id : str
            Trace ID to analyze
        """
        # Fetch trace data (example structure)
        spans = self.get_trace_spans(trace_id)

        # Identify slow spans
        for span in spans:
            duration_ms = (span['end_time'] - span['start_time']) * 1000

            if duration_ms > 1000:  # Slower than 1 second
                self.slow_spans.append({
                    'name': span['name'],
                    'duration_ms': duration_ms,
                    'attributes': span.get('attributes', {})
                })

        # AI-powered bottleneck analysis
        if self.slow_spans:
            return self.generate_performance_report()
        else:
            return "No performance issues detected"

    def generate_performance_report(self) -> str:
        """Generate human-readable performance report."""
        report = "Performance Bottlenecks:\n"

        # Sort by duration
        sorted_spans = sorted(self.slow_spans, key=lambda x: x['duration_ms'], reverse=True)

        for i, span in enumerate(sorted_spans[:5], 1):  # Top 5
            report += f"\n{i}. {span['name']}: {span['duration_ms']:.2f}ms"

            # Suggest optimizations
            if 'db.system' in span['attributes']:
                report += "\n   → Optimize database query or add caching"
            elif 'http.method' in span['attributes']:
                report += "\n   → Consider async processing or CDN"
            elif 'rpc.service' in span['attributes']:
                report += "\n   → Review RPC call necessity or batch requests"

        return report

    def get_trace_spans(self, trace_id: str):
        """Mock method to fetch trace spans."""
        # In production, fetch from OpenTelemetry backend
        return [
            {
                'name': 'database_query',
                'start_time': time.time(),
                'end_time': time.time() + 2.5,
                'attributes': {'db.system': 'postgresql', 'db.statement': 'SELECT * FROM users'}
            },
            {
                'name': 'external_api_call',
                'start_time': time.time(),
                'end_time': time.time() + 1.2,
                'attributes': {'http.method': 'POST', 'http.url': 'https://api.example.com'}
            }
        ]

# Usage
analyzer = AITraceAnalyzer()
report = analyzer.analyze_trace("abc123")
print(report)
```

## Best Practices

### 1. Prompt Engineering for Debugging

```python
def create_debugging_prompt(error_type: str, context: dict) -> str:
    """
    Create optimized prompt for LLM debugging assistance.

    Parameters
    ----------
    error_type : str
        Type of error (syntax, runtime, logic, performance)
    context : dict
        Debugging context (code, logs, traces, metrics)

    Returns
    -------
    str
        Optimized prompt for LLM
    """
    base_prompt = f"""
You are an expert debugging assistant specializing in {error_type} errors.

CONTEXT:
- Language: {context.get('language', 'Unknown')}
- Framework: {context.get('framework', 'N/A')}
- Environment: {context.get('environment', 'production')}

ERROR:
{context['error']}

CODE:
{context.get('code', 'N/A')}

LOGS (last 20 lines):
{context.get('logs', 'N/A')}

Your task:
1. Identify the root cause
2. Explain WHY it's failing (not just WHAT is failing)
3. Provide 2-3 specific fixes with code examples
4. Suggest tests to prevent regression
5. Recommend monitoring/alerts to catch this earlier

Be concise but thorough. Prioritize actionable fixes.
"""

    # Add specialized sections based on error type
    if error_type == 'performance':
        base_prompt += """
PERFORMANCE CONTEXT:
- Current latency: {context.get('latency', 'N/A')}
- Expected latency: {context.get('expected_latency', 'N/A')}
- Resource usage: {context.get('resources', 'N/A')}

Focus on:
- Algorithmic complexity
- Database query optimization
- Caching opportunities
- Async/parallel processing
"""

    return base_prompt

# Usage
prompt = create_debugging_prompt('performance', {
    'language': 'Python',
    'framework': 'FastAPI',
    'environment': 'production',
    'error': 'Response time > 5s',
    'code': 'def slow_endpoint(): ...',
    'latency': '5.2s',
    'expected_latency': '500ms'
})
```

### 2. Automated RCA Pipeline

```python
class AutomatedRCAPipeline:
    """End-to-end automated root cause analysis."""

    def __init__(self):
        self.log_analyzer = LogAnomalyDetector()
        self.ai_assistant = AIDebugAssistant(api_key="YOUR_KEY")

    def analyze_incident(self, incident_data: dict) -> dict:
        """
        Full RCA pipeline from detection to fix suggestion.

        Parameters
        ----------
        incident_data : dict
            {
                'alert_name': str,
                'timestamp': str,
                'logs': list,
                'metrics': dict,
                'traces': list,
                'recent_changes': list  # Git commits, deploys, config changes
            }

        Returns
        -------
        dict
            RCA report with root cause, impact, fix suggestions
        """
        results = {
            'incident_id': incident_data.get('alert_name'),
            'timestamp': incident_data['timestamp'],
            'analysis': {}
        }

        # 1. Correlate recent changes with incident
        changes_before_incident = self.filter_recent_changes(
            incident_data['recent_changes'],
            incident_data['timestamp']
        )

        if changes_before_incident:
            results['analysis']['suspected_cause'] = 'Recent deployment or config change'
            results['analysis']['changes'] = changes_before_incident

        # 2. Analyze logs for anomalies
        log_df = pd.DataFrame(incident_data['logs'])
        anomalies = self.log_analyzer.detect_anomalies(log_df)

        if len(anomalies[anomalies['anomaly']]) > 0:
            results['analysis']['log_anomalies'] = anomalies[anomalies['anomaly']].to_dict('records')

        # 3. AI-powered correlation
        ai_analysis = self.ai_assistant.analyze_stack_trace(
            error_trace=incident_data.get('error_trace', 'N/A'),
            context_code=self.get_relevant_code(changes_before_incident)
        )

        results['analysis']['ai_rca'] = ai_analysis

        # 4. Generate actionable report
        results['recommendations'] = self.generate_recommendations(results['analysis'])

        return results

    def filter_recent_changes(self, changes: list, incident_time: str) -> list:
        """Filter changes within 1 hour before incident."""
        import dateutil.parser
        incident_dt = dateutil.parser.parse(incident_time)

        recent = []
        for change in changes:
            change_dt = dateutil.parser.parse(change['timestamp'])
            if (incident_dt - change_dt).total_seconds() < 3600:  # 1 hour
                recent.append(change)

        return recent

    def get_relevant_code(self, changes: list) -> str:
        """Extract code from recent changes."""
        code = []
        for change in changes:
            if change['type'] == 'commit':
                code.append(f"Commit {change['hash']}: {change['diff']}")
        return "\n".join(code)

    def generate_recommendations(self, analysis: dict) -> list:
        """Generate actionable fix recommendations."""
        recommendations = []

        if 'suspected_cause' in analysis:
            if 'deployment' in analysis['suspected_cause'].lower():
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'Rollback deployment',
                    'command': 'kubectl rollout undo deployment/app'
                })

        if 'ai_rca' in analysis and 'fixes' in analysis['ai_rca']:
            for fix in analysis['ai_rca']['fixes']:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'Apply code fix',
                    'details': fix
                })

        return recommendations
```

This skill provides comprehensive AI-assisted debugging capabilities for modern distributed systems with focus on automation, observability, and intelligent root cause analysis!
