# Multi-Agent Reflection System

**Version**: 1.0.3
**Purpose**: Orchestration patterns for coordinating multiple specialist agents in reflection analysis

---

## Overview

The Multi-Agent Reflection System coordinates multiple specialist agents to provide comprehensive analysis across different dimensions. This document describes the orchestration patterns, agent roles, and coordination mechanisms.

---

## MetaReflectionOrchestrator

The central coordinator for multi-agent reflection processes.

### Core Implementation

```python
class MetaReflectionOrchestrator:
    """
    Coordinates reflection across multiple agents and dimensions

    Capabilities:
    - Parallel agent reflection execution
    - Cross-agent pattern synthesis
    - Meta-cognitive analysis
    - Strategic insight generation
    - Actionable recommendation synthesis
    """

    def __init__(self):
        self.agents = {
            'session': ConversationReflectionEngine(),
            'research': ResearchReflectionEngine(),
            'code': DevelopmentReflectionEngine(),
            'strategic': StrategicReflectionEngine()
        }
        self.context_manager = ReflectionContextManager()
        self.pattern_detector = CrossAgentPatternDetector()

    def orchestrate_reflection(self, context, depth='deep'):
        """
        Multi-layered reflection process

        Layers:
        1. Individual agent reflections (parallel)
        2. Cross-agent pattern synthesis
        3. Meta-cognitive analysis
        4. Strategic insight generation
        5. Actionable recommendation synthesis

        Args:
            context: Reflection context (session, code, research, etc.)
            depth: Analysis depth (shallow/deep/ultradeep)

        Returns:
            ReflectionReport with comprehensive findings
        """
        # Layer 1: Parallel reflection
        reflections = self.parallel_agent_reflection(context, depth)

        # Layer 2: Pattern identification
        patterns = self.identify_cross_agent_patterns(reflections)

        # Layer 3: Meta-analysis
        meta_insights = self.analyze_reasoning_patterns(
            reflections, patterns
        )

        # Layer 4: Strategic synthesis
        strategy = self.synthesize_strategic_insights(
            reflections, patterns, meta_insights
        )

        # Layer 5: Actionable recommendations
        recommendations = self.generate_actionable_plan(strategy)

        return ReflectionReport(
            dimensions=reflections,
            patterns=patterns,
            meta_insights=meta_insights,
            strategy=strategy,
            recommendations=recommendations,
            confidence=self._calculate_confidence(reflections)
        )

    def parallel_agent_reflection(self, context, depth):
        """
        Execute reflections in parallel across agents

        Returns dict of {agent_name: reflection_result}
        """
        from concurrent.futures import ThreadPoolExecutor

        active_agents = self._select_agents(context)

        with ThreadPoolExecutor(max_workers=len(active_agents)) as executor:
            futures = {
                name: executor.submit(agent.reflect, context, depth)
                for name, agent in active_agents.items()
            }

            reflections = {
                name: future.result()
                for name, future in futures.items()
            }

        return reflections

    def identify_cross_agent_patterns(self, reflections):
        """
        Find patterns across different agent reflections

        Pattern types:
        - Recurring themes (mentioned by 3+ agents)
        - Contradictions (agents disagree)
        - Blind spots (no agent addresses)
        - Complementary insights (agents reinforce each other)
        """
        patterns = {
            'recurring_themes': [],
            'contradictions': [],
            'blind_spots': [],
            'complementary_insights': []
        }

        # Extract all insights from reflections
        all_insights = []
        for agent, reflection in reflections.items():
            all_insights.extend([
                {
                    'agent': agent,
                    'insight': insight,
                    'confidence': reflection.confidence
                }
                for insight in reflection.insights
            ])

        # Detect recurring themes
        theme_counts = {}
        for insight in all_insights:
            theme = self._extract_theme(insight['insight'])
            if theme not in theme_counts:
                theme_counts[theme] = []
            theme_counts[theme].append(insight)

        patterns['recurring_themes'] = [
            {
                'theme': theme,
                'agents': list(set(i['agent'] for i in insights)),
                'count': len(insights),
                'avg_confidence': sum(i['confidence'] for i in insights) / len(insights)
            }
            for theme, insights in theme_counts.items()
            if len(insights) >= 3
        ]

        # Detect contradictions
        patterns['contradictions'] = self.pattern_detector.find_contradictions(
            all_insights
        )

        # Identify blind spots
        patterns['blind_spots'] = self.pattern_detector.find_blind_spots(
            reflections, self.context_manager.get_expected_dimensions(context)
        )

        # Find complementary insights
        patterns['complementary_insights'] = self.pattern_detector.find_complementary(
            all_insights
        )

        return patterns

    def analyze_reasoning_patterns(self, reflections, patterns):
        """
        Meta-cognitive analysis of reasoning quality

        Analyzes:
        - Reasoning coherence across agents
        - Confidence calibration
        - Evidence quality
        - Logic consistency
        - Assumption validity
        """
        meta_insights = {
            'reasoning_coherence': self._assess_coherence(reflections),
            'confidence_calibration': self._assess_calibration(reflections),
            'evidence_quality': self._assess_evidence(reflections),
            'logic_consistency': self._assess_logic(reflections, patterns),
            'assumption_validity': self._assess_assumptions(reflections)
        }

        return meta_insights

    def synthesize_strategic_insights(self, reflections, patterns, meta_insights):
        """
        Generate strategic insights from all layers

        Prioritizes by:
        - Impact: How much improvement potential?
        - Urgency: How critical is this?
        - Feasibility: How easy to address?
        - Confidence: How certain are we?
        """
        insights = []

        # From recurring themes
        for theme in patterns['recurring_themes']:
            if theme['avg_confidence'] > 0.75 and theme['count'] >= 3:
                insights.append({
                    'type': 'recurring_theme',
                    'insight': theme['theme'],
                    'priority': self._calculate_priority(theme),
                    'agents_agree': theme['agents']
                })

        # From contradictions (need resolution)
        for contradiction in patterns['contradictions']:
            insights.append({
                'type': 'contradiction',
                'insight': f"Agents disagree on: {contradiction['topic']}",
                'priority': 'high',
                'resolution_needed': True
            })

        # From blind spots (gaps to address)
        for blind_spot in patterns['blind_spots']:
            insights.append({
                'type': 'blind_spot',
                'insight': f"No analysis of: {blind_spot}",
                'priority': 'medium',
                'investigation_needed': True
            })

        # From meta-insights
        if meta_insights['reasoning_coherence'] < 0.7:
            insights.append({
                'type': 'meta',
                'insight': 'Reasoning coherence needs improvement',
                'priority': 'high'
            })

        return sorted(insights, key=lambda x: self._priority_score(x), reverse=True)

    def generate_actionable_plan(self, strategy):
        """
        Convert strategic insights into actionable recommendations

        Categorizes by:
        - Immediate (this session)
        - Short-term (this week)
        - Medium-term (this month)
        - Long-term (this quarter)
        """
        plan = {
            'immediate': [],
            'short_term': [],
            'medium_term': [],
            'long_term': []
        }

        for insight in strategy:
            action = self._insight_to_action(insight)
            timeframe = self._determine_timeframe(insight)
            plan[timeframe].append(action)

        return plan
```

---

## Agent Coordination Patterns

### Pattern 1: Sequential Reflection

Execute agents one after another, each building on previous insights.

```python
def sequential_reflection(self, context):
    """
    Sequential reflection pattern

    Use when:
    - Agents have strong dependencies
    - Each agent needs previous results
    - Order matters for analysis
    """
    results = {}

    # Session reflection first (understand what happened)
    results['session'] = self.agents['session'].reflect(context)

    # Research reflection (if applicable)
    if context.type == 'research':
        context.add_insights(results['session'])
        results['research'] = self.agents['research'].reflect(context)

    # Code reflection (with session context)
    if context.has_code:
        context.add_insights(results['session'])
        results['code'] = self.agents['code'].reflect(context)

    # Strategic synthesis (with all insights)
    context.add_all_insights(results)
    results['strategic'] = self.agents['strategic'].reflect(context)

    return results
```

### Pattern 2: Parallel Reflection

Execute all agents simultaneously for independent analysis.

```python
def parallel_reflection(self, context):
    """
    Parallel reflection pattern

    Use when:
    - Agents are independent
    - Speed is priority
    - No inter-agent dependencies
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            name: executor.submit(agent.reflect, context)
            for name, agent in self.agents.items()
        }

        results = {
            name: future.result()
            for name, future in futures.items()
        }

    return results
```

### Pattern 3: Hierarchical Reflection

Coordinate agents in layers with synthesis at each level.

```python
def hierarchical_reflection(self, context):
    """
    Hierarchical reflection pattern

    Layers:
    1. Domain-specific agents (parallel)
    2. Cross-domain synthesis
    3. Meta-cognitive analysis
    4. Strategic recommendations
    """
    # Layer 1: Domain agents (parallel)
    domain_results = self.parallel_reflection(context)

    # Layer 2: Cross-domain synthesis
    synthesis = self._synthesize_domains(domain_results)

    # Layer 3: Meta-cognitive
    meta_analysis = self._meta_analyze(domain_results, synthesis)

    # Layer 4: Strategic
    strategy = self._generate_strategy(
        domain_results, synthesis, meta_analysis
    )

    return {
        'domain': domain_results,
        'synthesis': synthesis,
        'meta': meta_analysis,
        'strategy': strategy
    }
```

---

## Agent Communication Protocols

### Message Passing

```python
class AgentMessage:
    """
    Standard message format for inter-agent communication
    """
    def __init__(self, sender, recipient, message_type, content, priority='normal'):
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type  # 'insight', 'question', 'correction', 'request'
        self.content = content
        self.priority = priority
        self.timestamp = datetime.now()

class MessageBroker:
    """
    Facilitates communication between agents
    """
    def __init__(self):
        self.messages = []
        self.subscriptions = {}

    def publish(self, message):
        """Agent publishes a message"""
        self.messages.append(message)
        self._notify_subscribers(message)

    def subscribe(self, agent_name, message_types):
        """Agent subscribes to message types"""
        if agent_name not in self.subscriptions:
            self.subscriptions[agent_name] = []
        self.subscriptions[agent_name].extend(message_types)
```

---

## Example: Full Orchestration

```python
# Initialize orchestrator
orchestrator = MetaReflectionOrchestrator()

# Prepare context
context = ReflectionContext(
    type='session',
    session_data=get_session_history(),
    code_changes=get_git_diff(),
    depth='deep'
)

# Execute orchestrated reflection
report = orchestrator.orchestrate_reflection(context, depth='deep')

# Access results
print(f"Overall Confidence: {report.confidence:.2f}")
print(f"Patterns Found: {len(report.patterns['recurring_themes'])}")
print(f"Recommendations: {len(report.recommendations['immediate'])}")

# Detailed insights
for dimension, reflection in report.dimensions.items():
    print(f"\n{dimension} Score: {reflection.score}/10")
    for insight in reflection.top_insights:
        print(f"  - {insight}")
```

---

## Best Practices

### 1. Agent Selection

Choose agents based on context:
```python
def select_agents(context):
    """
    Intelligent agent selection based on context
    """
    agents = []

    # Always include session agent for reasoning analysis
    agents.append('session')

    # Conditional agents
    if context.has_code:
        agents.append('code')
    if context.type == 'research':
        agents.append('research')
    if context.strategic_decision:
        agents.append('strategic')

    return agents
```

### 2. Confidence Aggregation

```python
def aggregate_confidence(reflections):
    """
    Aggregate confidence across agents

    Methods:
    - Average: Simple mean
    - Weighted: By agent expertise
    - Conservative: Minimum confidence
    - Optimistic: Maximum confidence
    """
    # Weighted by agent reliability
    weights = {
        'session': 1.0,
        'research': 0.9,  # Slightly less weight
        'code': 0.95,
        'strategic': 0.85
    }

    weighted_sum = sum(
        reflections[agent].confidence * weights.get(agent, 1.0)
        for agent in reflections
    )
    total_weight = sum(weights.get(agent, 1.0) for agent in reflections)

    return weighted_sum / total_weight
```

### 3. Contradiction Resolution

```python
def resolve_contradiction(agent1_view, agent2_view):
    """
    Strategies for resolving agent contradictions
    """
    # Strategy 1: Evidence-based (prefer agent with more evidence)
    if agent1_view.evidence_count > agent2_view.evidence_count * 1.5:
        return agent1_view

    # Strategy 2: Confidence-based (prefer higher confidence if significant)
    if agent1_view.confidence - agent2_view.confidence > 0.2:
        return agent1_view

    # Strategy 3: Domain expertise (prefer specialist)
    if agent1_view.agent_expertise > agent2_view.agent_expertise:
        return agent1_view

    # Strategy 4: Flag for human review
    return {
        'resolution': 'manual_review_needed',
        'views': [agent1_view, agent2_view],
        'reason': 'Unable to auto-resolve, similar evidence and confidence'
    }
```

---

## Performance Optimization

### Caching Strategies

```python
class ReflectionCache:
    """
    Cache agent reflections for performance
    """
    def __init__(self, ttl=3600):
        self.cache = {}
        self.ttl = ttl

    def get(self, context_hash, agent):
        """Get cached reflection if available and fresh"""
        key = f"{context_hash}:{agent}"
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['reflection']
        return None

    def set(self, context_hash, agent, reflection):
        """Cache reflection result"""
        key = f"{context_hash}:{agent}"
        self.cache[key] = {
            'reflection': reflection,
            'timestamp': time.time()
        }
```

### Async Execution

```python
import asyncio

async def async_parallel_reflection(self, context):
    """
    Async parallel reflection for better performance
    """
    tasks = [
        agent.reflect_async(context)
        for agent in self.agents.values()
    ]

    results = await asyncio.gather(*tasks)

    return {
        agent_name: result
        for agent_name, result in zip(self.agents.keys(), results)
    }
```

---

## Related Documentation

- [Session Analysis Engine](session-analysis-engine.md) - Conversation and reasoning reflection
- [Research Reflection Engine](research-reflection-engine.md) - Scientific methodology assessment
- [Development Reflection Engine](development-reflection-engine.md) - Code quality reflection

---

*Part of the ai-reasoning plugin documentation*
