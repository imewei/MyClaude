# LLM Integration Patterns

Production-ready patterns for integrating LLMs from multiple providers (OpenAI, Anthropic, local models) with robust error handling, response generation strategies, and function calling.

## Table of Contents

- [Provider Integration](#provider-integration)
- [Response Generation Strategies](#response-generation-strategies)
- [Function Calling Interfaces](#function-calling-interfaces)
- [Error Handling Patterns](#error-handling-patterns)

## Provider Integration

### Multi-Provider LLM Integration Layer

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def stream(self, prompt: str, **kwargs):
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT integration"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    async def stream(self, prompt: str, **kwargs):
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class AnthropicProvider(LLMProvider):
    """Anthropic Claude integration"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', 4096),
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
        )
        return response.content[0].text

class LocalLLMProvider(LLMProvider):
    """Local LLM integration (Ollama, vLLM, etc.)"""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    async def complete(self, prompt: str, **kwargs) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, **kwargs}
            ) as response:
                result = await response.json()
                return result['response']
```

### LLM Integration Manager

```python
class LLMIntegrationLayer:
    """Manages multiple LLM providers with fallback"""

    def __init__(self):
        self.providers = {
            'openai': OpenAIProvider(os.getenv('OPENAI_API_KEY')),
            'anthropic': AnthropicProvider(os.getenv('ANTHROPIC_API_KEY')),
            'local': LocalLLMProvider('http://localhost:11434', 'llama2')
        }
        self.current_provider = 'anthropic'
        self.fallback_chain = ['anthropic', 'openai', 'local']

    async def setup_llm_integration(self, provider: str, config: Dict[str, Any]):
        """Setup LLM integration"""
        self.current_provider = provider
        return {
            'provider': provider,
            'capabilities': self.providers[provider].get_capabilities(),
            'rate_limits': self.get_rate_limits(provider)
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_completion(self,
                                 prompt: str,
                                 system_prompt: str = None,
                                 **kwargs):
        """Generate completion with fallback handling"""
        for provider_name in self.fallback_chain:
            try:
                provider = self.providers[provider_name]
                response = await provider.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs
                )

                if self.is_valid_response(response):
                    return {
                        'provider': provider_name,
                        'response': response,
                        'tokens_used': self.estimate_tokens(response)
                    }

            except RateLimitError:
                logger.warning(f"{provider_name} rate limited, trying fallback")
                continue
            except Exception as e:
                logger.error(f"{provider_name} error: {e}")
                continue

        return self.get_fallback_response()
```

## Response Generation Strategies

### Intelligent Response Generator

```python
class ResponseGenerator:
    """Multi-strategy response generation"""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.templates = self._load_response_templates()
        self.personality = self._load_personality_config()

    async def generate_response(self,
                               intent: str,
                               context: ConversationContext,
                               data: Dict[str, Any]) -> str:
        """Generate contextual responses"""

        # Select response strategy
        if self.should_use_template(intent):
            response = self.generate_from_template(intent, data)
        elif self.should_use_llm(intent, context):
            response = await self.generate_with_llm(intent, context, data)
        else:
            response = self.generate_hybrid_response(intent, context, data)

        # Apply personality and tone
        response = self.apply_personality(response, context)

        # Ensure response appropriateness
        response = self.validate_response(response, context)

        return response

    async def generate_with_llm(self, intent, context, data):
        """Generate response using LLM"""
        # Construct prompt
        prompt = self.build_llm_prompt(intent, context, data)

        # Set generation parameters
        params = {
            'temperature': self.get_temperature(intent),
            'max_tokens': 150,
            'stop_sequences': ['\n\n', 'User:', 'Human:']
        }

        # Generate response
        response = await self.llm.generate(prompt, **params)

        # Post-process response
        return self.post_process_llm_response(response)

    def build_llm_prompt(self, intent, context, data):
        """Build context-aware prompt for LLM"""
        return f"""You are a helpful AI assistant with the following characteristics:
{self.personality.description}

Conversation history:
{self.format_conversation_history(context.messages[-5:])}

User intent: {intent}
Relevant data: {json.dumps(data, indent=2)}

Generate a helpful, concise response that:
1. Addresses the user's intent
2. Uses the provided data appropriately
3. Maintains conversation continuity
4. Follows the personality guidelines

Response:"""

    def generate_from_template(self, intent, data):
        """Generate response from templates"""
        template = self.templates.get(intent)
        if not template:
            return self.get_fallback_response()

        # Select template variant
        variant = self.select_template_variant(template, data)

        # Fill template slots
        response = variant
        for key, value in data.items():
            response = response.replace(f"{{{key}}}", str(value))

        return response

    def generate_hybrid_response(self, intent, context, data):
        """Hybrid: template with LLM enhancement"""
        base_response = self.generate_from_template(intent, data)

        # Enhance with LLM for personalization
        enhancement_prompt = f"""
Enhance this response with personality and context:

Base response: {base_response}
User context: {context.user_profile}

Enhanced response:"""

        enhanced = await self.llm.generate(enhancement_prompt, max_tokens=100)
        return enhanced
```

### Response Personalization

```python
def apply_personality(self, response, context):
    """Apply personality traits to response"""
    # Add personality markers
    if self.personality.get('friendly'):
        response = self.add_friendly_markers(response)

    if self.personality.get('professional'):
        response = self.ensure_professional_tone(response)

    # Adjust based on user preferences
    if context.user_profile.get('prefers_brief'):
        response = self.make_concise(response)

    return response
```

## Function Calling Interfaces

### LLM Function Calling

```python
class FunctionCallingInterface:
    """Structured function calling for LLMs"""

    def __init__(self):
        self.functions = {}

    def register_function(self,
                         name: str,
                         func: callable,
                         description: str,
                         parameters: Dict[str, Any]):
        """Register a function for LLM to call"""
        self.functions[name] = {
            'function': func,
            'description': description,
            'parameters': parameters,
            'schema': self._generate_schema(parameters)
        }

    def _generate_schema(self, parameters):
        """Generate OpenAI function schema"""
        return {
            "type": "object",
            "properties": parameters,
            "required": [k for k, v in parameters.items() if v.get('required', False)]
        }

    async def process_function_call(self, llm_response):
        """Process function calls from LLM"""
        if 'function_call' not in llm_response:
            return llm_response

        function_name = llm_response['function_call']['name']
        arguments = llm_response['function_call']['arguments']

        if function_name not in self.functions:
            return {'error': f'Unknown function: {function_name}'}

        # Validate arguments
        validated_args = self.validate_arguments(function_name, arguments)

        # Execute function
        result = await self.functions[function_name]['function'](**validated_args)

        # Return result for LLM to process
        return {
            'function_result': result,
            'function_name': function_name
        }

    def get_function_definitions(self):
        """Get function definitions for LLM"""
        return [
            {
                'name': name,
                'description': func['description'],
                'parameters': func['schema']
            }
            for name, func in self.functions.items()
        ]
```

### Tool Use Pattern

```python
async def llm_with_tools(prompt: str, tools: List[Dict]):
    """LLM generation with tool use"""
    response = await llm.generate(
        prompt,
        functions=tools,
        function_call="auto"
    )

    if response.get('function_call'):
        # Execute function
        function_result = await execute_function(
            response['function_call']['name'],
            response['function_call']['arguments']
        )

        # Continue conversation with function result
        followup = await llm.generate(
            prompt,
            functions=tools,
            messages=[
                {"role": "function", "content": str(function_result)}
            ]
        )
        return followup

    return response
```

## Error Handling Patterns

### Comprehensive Error Handling

```python
class RobustLLMClient:
    """LLM client with robust error handling"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    async def generate(self, prompt: str, **kwargs):
        """Generate with retries and fallbacks"""
        try:
            response = await self.llm.complete(prompt, **kwargs)

            # Validate response
            if not self.is_valid_response(response):
                raise InvalidResponseError("Response validation failed")

            return response

        except RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            await asyncio.sleep(self.backoff_time)
            raise  # Retry

        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            # Don't retry, switch provider
            return await self.use_fallback_provider(prompt, **kwargs)

        except InvalidResponseError as e:
            logger.warning(f"Invalid response: {e}")
            # Retry with modified prompt
            modified_prompt = self.modify_prompt_for_retry(prompt)
            return await self.generate(modified_prompt, **kwargs)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return self.get_safe_fallback_response()

    async def use_fallback_provider(self, prompt, **kwargs):
        """Switch to fallback LLM provider"""
        if self.fallback_provider:
            return await self.fallback_provider.complete(prompt, **kwargs)
        return self.get_safe_fallback_response()

    def get_safe_fallback_response(self):
        """Safe fallback when all providers fail"""
        return {
            'response': "I'm experiencing technical difficulties. Please try again shortly.",
            'fallback': True,
            'should_escalate': True
        }
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Prevent cascading failures"""

    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        """Reset on successful call"""
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        """Increment failure counter"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

---

**See Also**:
- [AI Assistant Architecture](./ai-assistant-architecture.md) - Overall system design
- [AI Testing & Deployment](./ai-testing-deployment.md) - Testing LLM integrations
- Command: `/ai-assistant` - Interactive development with LLM integration
