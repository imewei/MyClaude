---
version: "1.0.3"
category: "llm-application-dev"
command: "/ai-assistant"
description: Build production-ready AI assistants with natural language understanding, conversation management, and intelligent response generation
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: "<assistant_description>"
color: blue
execution_modes:
  quick: "5-10 minutes - Generate basic chatbot architecture with core components"
  standard: "15-25 minutes - Build complete AI assistant with NLP, dialog management, and deployment"
  comprehensive: "30-45 minutes - Production-ready system with testing, monitoring, and continuous improvement"
agents:
  primary:
    - ai-engineer
  conditional:
    - agent: prompt-engineer
      trigger: pattern "prompt|llm.*prompt|response.*generation"
  orchestrated: false
---

# AI Assistant Development

Build production-ready AI assistants with natural language understanding, conversation management, and intelligent response generation.

## Quick Reference

| Topic | External Documentation | Lines |
|-------|------------------------|-------|
| **Architecture Patterns** | [ai-assistant-architecture.md](../docs/ai-assistant-architecture.md) | ~600 |
| **LLM Integration** | [llm-integration-patterns.md](../docs/llm-integration-patterns.md) | ~400 |
| **Testing & Deployment** | [ai-testing-deployment.md](../docs/ai-testing-deployment.md) | ~500 |

**Total External Documentation**: ~1,500 lines of comprehensive patterns and implementations

## Requirements

$ARGUMENTS

## Core Workflow

### Phase 1: Architecture Design

**Define assistant architecture** with core components:

**Key Components**:
1. **Natural Language Understanding (NLU)**
   - Intent classification
   - Entity extraction
   - Sentiment analysis

2. **Dialog Management**
   - Conversation state tracking
   - Flow control
   - Action selection

3. **Response Generation**
   - Template-based responses
   - LLM-powered generation
   - Personalization

4. **Context Management**
   - Short-term memory (conversation)
   - Long-term memory (user profile)
   - Entity state tracking

**Architecture Reference**: [ai-assistant-architecture.md#architecture-overview](../docs/ai-assistant-architecture.md#architecture-overview)

**Quick Start**:
```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ConversationContext:
    """Maintains conversation state and context"""
    user_id: str
    session_id: str
    messages: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    conversation_state: Dict[str, Any]
    metadata: Dict[str, Any]

class AIAssistant:
    def __init__(self, config: Dict[str, Any]):
        self.nlu = NLUComponent()
        self.dialog_manager = DialogManager()
        self.response_generator = ResponseGenerator()
        self.context_manager = ContextManager()

    async def process_message(self, message: str, context: ConversationContext):
        """Process user message through AI assistant pipeline"""
        # NLU: Understand user intent
        nlu_result = await self.nlu.process(message, context)

        # Dialog: Determine action
        dialog_result = await self.dialog_manager.process_turn(context, nlu_result)

        # Response: Generate appropriate response
        response = await self.response_generator.generate(
            dialog_result['action'],
            context,
            dialog_result['response_data']
        )

        # Context: Update conversation state
        updated_context = await self.context_manager.manage_context(
            {'message': message, 'response': response},
            context
        )

        return {'response': response, 'context': updated_context}
```

**Complete Implementations**: [ai-assistant-architecture.md](../docs/ai-assistant-architecture.md)

### Phase 2: NLP Pipeline Implementation

**Build natural language processing** capabilities:

**Core NLP Tasks**:
1. **Intent Detection**
   - Multi-intent classification
   - Confidence scoring
   - Fallback handling

2. **Entity Extraction**
   - Named entity recognition
   - Custom domain entities
   - Entity resolution

3. **Sentiment Analysis**
   - Emotion detection
   - Urgency classification
   - User satisfaction tracking

**Implementation Pattern**:
```python
class NLPPipeline:
    async def process_message(self, message: str, context: ConversationContext):
        """Process through NLP pipeline"""
        # Parallel processing
        tasks = [
            self.detect_intent(message),
            self.extract_entities(message),
            self.analyze_sentiment(message)
        ]

        results = await asyncio.gather(*tasks)

        return {
            'intent': results[0],
            'entities': results[1],
            'sentiment': results[2]
        }
```

**Detailed Patterns**: [ai-assistant-architecture.md#nlp-pipeline-implementation](../docs/ai-assistant-architecture.md#nlp-pipeline-implementation)

### Phase 3: Conversation Flow Design

**Design multi-turn conversation flows**:

**Flow Components**:
- **Greeting Flow**: Welcome and understand user needs
- **Task Completion Flow**: Slot filling and confirmation
- **Error Handling Flow**: Clarification and fallback
- **Farewell Flow**: Conversation closure

**Flow Engine Pattern**:
```python
class ConversationFlowEngine:
    def design_conversation_flow(self):
        """Design multi-turn flows"""
        return {
            'task_completion_flow': {
                'nodes': [
                    {'id': 'understand_task', 'type': 'nlu_processing'},
                    {'id': 'check_requirements', 'type': 'validation'},
                    {'id': 'request_missing_info', 'type': 'slot_filling'},
                    {'id': 'confirm_task', 'type': 'confirmation'},
                    {'id': 'execute_task', 'type': 'action'}
                ]
            }
        }
```

**Flow Patterns**: [ai-assistant-architecture.md#conversation-flow-design](../docs/ai-assistant-architecture.md#conversation-flow-design)

### Phase 4: LLM Integration

**Integrate LLM providers** for intelligent responses:

**Provider Setup**:
- **OpenAI**: GPT-4, GPT-4o
- **Anthropic**: Claude Sonnet 4.5
- **Local**: Ollama, vLLM

**Integration Pattern**:
```python
class LLMIntegration:
    def __init__(self):
        self.providers = {
            'anthropic': AnthropicProvider(model="claude-sonnet-4-5"),
            'openai': OpenAIProvider(model="gpt-4"),
            'local': LocalLLMProvider()
        }
        self.fallback_chain = ['anthropic', 'openai', 'local']

    async def generate_response(self, prompt: str, **kwargs):
        """Generate with fallback"""
        for provider_name in self.fallback_chain:
            try:
                return await self.providers[provider_name].complete(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"{provider_name} failed: {e}")
                continue

        return self.get_fallback_response()
```

**Complete Integration**: [llm-integration-patterns.md](../docs/llm-integration-patterns.md)

### Phase 5: Context Management

**Implement sophisticated context tracking**:

**Memory Hierarchy**:
1. **Working Memory**: Last N messages
2. **Short-Term Memory**: Session context
3. **Long-Term Memory**: User profile

**Context Manager**:
```python
class ContextManager:
    async def manage_context(self, new_input, current_context):
        """Manage conversation context"""
        # Resolve references (pronouns, temporal)
        resolved_input = await self.resolve_references(new_input, current_context)

        # Detect topic shifts
        if self.detect_topic_shift(resolved_input, current_context):
            current_context = self.handle_topic_shift(current_context)

        # Update entity states
        current_context = self.update_entity_state(resolved_input, current_context)

        return current_context
```

**Advanced Patterns**: [ai-assistant-architecture.md#context-management-systems](../docs/ai-assistant-architecture.md#context-management-systems)

### Phase 6: Testing & Deployment

**Test and deploy** AI assistant:

**Testing Strategy**:
1. **Unit Tests**: Individual components
2. **Integration Tests**: Component interactions
3. **Conversation Tests**: Multi-turn flows
4. **Performance Tests**: Latency and throughput

**Test Pattern**:
```python
class ConversationTest:
    async def test_multi_turn_conversation(self):
        """Test complete flow"""
        conversation = [
            {'user': "I need help", 'expected_intent': 'request_help'},
            {'user': "My order 12345", 'expected_entities': [{'type': 'order_id'}]}
        ]

        for turn in conversation:
            response = await assistant.process_message(turn['user'], context)
            assert response['intent'] == turn['expected_intent']
```

**Deployment Options**:
- **Docker**: Containerized deployment
- **Kubernetes**: Scalable orchestration
- **Serverless**: AWS Lambda, Google Cloud Functions

**Testing Guide**: [ai-testing-deployment.md#testing-frameworks](../docs/ai-testing-deployment.md#testing-frameworks)

**Deployment Guide**: [ai-testing-deployment.md#deployment-architecture](../docs/ai-testing-deployment.md#deployment-architecture)

### Phase 7: Monitoring & Improvement

**Monitor and continuously improve**:

**Monitoring Metrics**:
- **Real-time**: Active sessions, response time, success rate
- **Conversation**: Average length, completion rate, satisfaction
- **System**: Model inference time, cache hit rate, errors

**Analytics Setup**:
```python
class AssistantAnalytics:
    def create_monitoring(self):
        return {
            'metrics': {
                'active_sessions': 'gauge',
                'response_time_p95': 'histogram',
                'intent_accuracy': 'gauge',
                'user_satisfaction': 'gauge'
            },
            'alerts': [
                {'name': 'high_fallback_rate', 'condition': 'fallback_rate > 0.2'},
                {'name': 'slow_response', 'condition': 'response_time_p95 > 2000'}
            ]
        }
```

**Monitoring Guide**: [ai-testing-deployment.md#monitoring-systems](../docs/ai-testing-deployment.md#monitoring-systems)

## Mode-Specific Execution

### Quick Mode (5-10 minutes)

**Generated**:
- Basic assistant architecture
- Simple intent classification
- Template-based responses
- In-memory context

**Skip**: Advanced NLP, LLM integration, production deployment

**Output**: Functional prototype for testing concepts

### Standard Mode (15-25 minutes) - DEFAULT

**Generated**:
- Complete AI assistant architecture
- NLP pipeline (intent, entities, sentiment)
- LLM-powered response generation
- Conversation flow design
- Context management
- Basic deployment (Docker)

**Include**: Testing framework, monitoring setup

**Output**: Production-ready AI assistant

### Comprehensive Mode (30-45 minutes)

**Generated**:
- All from standard mode
- Advanced conversation flows
- Multi-provider LLM integration with fallbacks
- Hierarchical memory system
- Kubernetes deployment
- Comprehensive testing suite
- Full monitoring and analytics
- Continuous improvement pipeline

**Include**: Performance optimization, scaling guide, best practices

**Output**: Enterprise-grade AI assistant with full observability

## Success Criteria

✅ Architecture designed with all core components
✅ NLP pipeline implemented (intent, entities, sentiment)
✅ Conversation flows designed for key use cases
✅ LLM integration with fallback providers
✅ Context management implemented
✅ Testing framework created
✅ Deployment configured (Docker/K8s)
✅ Monitoring and analytics set up
✅ External documentation referenced

## Agent Integration

- **ai-engineer**: Primary agent for AI assistant architecture and implementation
- **prompt-engineer**: Triggered for response generation and prompt optimization

## Best Practices

1. **Start Simple**: Begin with template responses, add LLM gradually
2. **Test Early**: Write conversation tests before full implementation
3. **Monitor Everything**: Track success rate, latency, user satisfaction
4. **Iterate**: Use feedback to improve intent detection and responses
5. **Secure by Default**: Validate inputs, sanitize outputs, protect user data
6. **Cost Optimize**: Cache responses, use smaller models for simple tasks
7. **Fail Gracefully**: Always have fallback responses and error handling

## Common Patterns

### Intent Classification
✅ **Multi-intent**: Handle compound requests ("Cancel order AND update address")
✅ **Hierarchical**: Use intent taxonomy (customer_service → order_management → cancel_order)
❌ **Over-fitting**: Don't create too many specific intents, use parameters instead

### Response Generation
✅ **Hybrid**: Templates for common cases, LLM for complex
✅ **Personalization**: Adapt tone and length based on user preferences
❌ **Over-reliance on LLM**: Use templates when responses are predictable

### Context Management
✅ **Pruning**: Remove old context to stay within token limits
✅ **Entity Tracking**: Maintain state of discussed entities
❌ **Unbounded Memory**: Always have maximum context length

## See Also

- **External Docs**:
  - [AI Assistant Architecture](../docs/ai-assistant-architecture.md) - Complete architecture patterns
  - [LLM Integration Patterns](../docs/llm-integration-patterns.md) - Provider integration
  - [AI Testing & Deployment](../docs/ai-testing-deployment.md) - Testing and deployment

- **Related Commands**:
  - `/langchain-agent` - Build LangChain-based agents
  - `/prompt-optimize` - Optimize prompts for better responses

---

Focus on **user value first**, **iterate quickly**, and **measure everything** to build AI assistants that truly help users.
