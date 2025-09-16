# 🔌 AI Providers - Multi-Provider AI Integration

This directory contains the AI provider implementations that power the AI Functionality system. Each provider offers different models, pricing, and capabilities to suit various use cases.

## 🏗️ Architecture

All providers implement the `BaseAIProvider` interface, ensuring consistent behavior and easy interchangeability:

```python
from AI_Functionality.core.base_provider import BaseAIProvider, AIRequest, AIResponse

class CustomProvider(BaseAIProvider):
    async def generate_response(self, request: AIRequest) -> AIResponse:
        # Provider-specific implementation
        pass
```

## 🔌 Available Providers

### 1. OpenAI Provider (`openai_provider.py`)

**Best for**: Production applications requiring high-quality responses

#### Configuration
```python
from AI_Functionality import DataAnalyst

analyst = DataAnalyst(
    openai_api_key="sk-your-openai-api-key",
    openai_model="gpt-4o-mini",  # Default model
    primary_provider="openai"
)
```

#### Available Models
| Model | Description | Use Case | Cost |
|-------|-------------|----------|------|
| `gpt-4o` | Latest GPT-4 optimized | Complex analysis, highest quality | $$$ |
| `gpt-4o-mini` | Cost-effective GPT-4 | Balanced quality/cost | $$ |
| `gpt-4-turbo` | High-speed GPT-4 | Fast complex tasks | $$$ |
| `gpt-3.5-turbo` | Fast and economical | Simple analysis, high volume | $ |

#### Features
- ✅ Streaming responses
- ✅ Function calling (for structured data)
- ✅ JSON mode output
- ✅ High rate limits (tier-dependent)
- ✅ Excellent reasoning capabilities

#### Getting Started
1. **Get API Key**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Set up billing**: Add payment method for usage-based pricing
3. **Configure limits**: Set spending limits to control costs
4. **Test connection**: Use the built-in connection test

```python
# Test OpenAI connection
response = await analyst.test_provider_connection("openai")
print(f"Connection status: {response.status}")
```

#### Pricing (as of 2024)
- **GPT-4o**: ~$15/1M input tokens, ~$60/1M output tokens
- **GPT-4o-mini**: ~$0.15/1M input tokens, ~$0.60/1M output tokens
- **GPT-3.5-turbo**: ~$0.50/1M input tokens, ~$1.50/1M output tokens

### 2. NVIDIA AI Provider (`nvidia_provider.py`)

**Best for**: Development and complex reasoning tasks with cost-effectiveness

#### Configuration
```python
analyst = DataAnalyst(
    nvidia_api_key="nvapi-your-nvidia-api-key",
    nvidia_model="qwen/qwen2.5-72b-instruct",  # Default model
    primary_provider="nvidia"
)
```

#### Available Models
| Model | Description | Use Case | Cost |
|-------|-------------|----------|------|
| `qwen/qwen2.5-72b-instruct` | Advanced reasoning model | Complex analysis, math | FREE |
| `meta/llama-3.1-405b-instruct` | Largest Llama model | Highest quality reasoning | FREE |
| `meta/llama-3.1-70b-instruct` | Balanced Llama model | General purpose | FREE |
| `meta/llama-3.1-8b-instruct` | Fast Llama model | Quick analysis | FREE |
| `mistralai/mixtral-8x22b-instruct-v0.1` | Mixture of experts | Specialized tasks | FREE |
| `google/gemma-2-27b-it` | Google's instruction-tuned | Code and reasoning | FREE |

#### Features
- ✅ **Free tier available** with generous limits
- ✅ Powerful reasoning capabilities
- ✅ Latest open-source models
- ✅ High-quality mathematical reasoning
- ✅ Good code understanding

#### Getting Started
1. **Get API Key**: Visit [NVIDIA AI](https://build.nvidia.com)
2. **Create account**: Free registration required
3. **Generate key**: Create API key in dashboard
4. **Start using**: No payment method required for free tier

```python
# Test NVIDIA connection
response = await analyst.test_provider_connection("nvidia")
print(f"Free tier status: {response.free_tier_available}")
```

#### Rate Limits (Free Tier)
- **Requests**: 1,000 requests per day
- **Tokens**: 10M input tokens per day
- **Concurrent**: 10 concurrent requests
- **Models**: Access to all foundation models

### 3. OpenRouter Provider (`openrouter_provider.py`)

**Best for**: Experimentation and accessing diverse AI models

#### Configuration
```python
analyst = DataAnalyst(
    openrouter_api_key="sk-or-v1-your-openrouter-key",
    openrouter_model="anthropic/claude-3-sonnet",  # Default model
    primary_provider="openrouter"
)
```

#### Available Models
| Model | Provider | Use Case | Cost |
|-------|----------|----------|------|
| `anthropic/claude-3-opus` | Anthropic | Highest quality analysis | $$$$ |
| `anthropic/claude-3-sonnet` | Anthropic | Balanced quality/speed | $$$ |
| `anthropic/claude-3-haiku` | Anthropic | Fast responses | $$ |
| `google/gemini-pro` | Google | Multimodal capabilities | $$ |
| `meta-llama/llama-3-70b-instruct` | Meta | Open-source reasoning | $ |
| `openai/gpt-4` | OpenAI | Via OpenRouter | $$$ |
| `cohere/command-r-plus` | Cohere | Enterprise features | $$$ |

#### Features
- ✅ Access to 150+ models from 30+ providers
- ✅ Unified API across all models
- ✅ Competitive pricing
- ✅ Model comparison tools
- ✅ Usage analytics

#### Getting Started
1. **Get API Key**: Visit [OpenRouter](https://openrouter.ai/keys)
2. **Add credits**: Purchase credits for usage
3. **Browse models**: Explore available models and pricing
4. **Set preferences**: Configure default models and limits

```python
# Test OpenRouter connection and list models
response = await analyst.test_provider_connection("openrouter")
available_models = response.available_models
print(f"Available models: {len(available_models)}")
```

#### Pricing Benefits
- **No markup on free models**: Free tier models are truly free
- **Bulk discounts**: Volume pricing available
- **Per-token billing**: Pay only for what you use
- **Real-time pricing**: Always see current costs

## 🔄 Multi-Provider Configuration

### Recommended Setup

For production applications, use multiple providers for reliability:

```python
# Multi-provider setup with smart fallback
analyst = DataAnalyst(
    # Primary: Free tier for development
    nvidia_api_key="nvapi-your-nvidia-key",
    primary_provider="nvidia",

    # Fallback: High-quality when needed
    openai_api_key="sk-your-openai-key",

    # Emergency fallback: Access to many models
    openrouter_api_key="sk-or-your-openrouter-key",

    fallback_providers=["openai", "openrouter"],

    # Provider-specific configurations
    provider_configs={
        "nvidia": {
            "model": "qwen/qwen2.5-72b-instruct",
            "temperature": 0.3,
            "max_tokens": 2000
        },
        "openai": {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 2000
        },
        "openrouter": {
            "model": "anthropic/claude-3-sonnet",
            "temperature": 0.3,
            "max_tokens": 2000
        }
    }
)
```

### Provider Selection Logic

The system automatically selects providers based on:

1. **Primary provider availability**: Try primary first
2. **Error handling**: Switch to fallback on errors
3. **Rate limiting**: Switch when rate limited
4. **Cost optimization**: Use free tiers when possible
5. **Model capabilities**: Select best model for task

```python
# Custom provider selection
async def custom_provider_logic(request: AIRequest):
    if request.complexity == "high":
        return "openai"  # Use GPT-4 for complex tasks
    elif request.cost_sensitive:
        return "nvidia"  # Use free tier for cost-sensitive
    else:
        return "openrouter"  # Use diverse models for variety

analyst.set_provider_selector(custom_provider_logic)
```

## 🛠️ Provider Implementation

### Creating Custom Providers

You can create custom providers for other AI services:

```python
from AI_Functionality.core.base_provider import BaseAIProvider, AIRequest, AIResponse
import httpx
import asyncio

class CustomAIProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = "default-model"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.custom-ai.com/v1"

    async def generate_response(self, request: AIRequest) -> AIResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt}
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=request.timeout or 60
            )

            if response.status_code == 200:
                data = response.json()
                return AIResponse(
                    content=data["choices"][0]["message"]["content"],
                    provider="custom",
                    model=self.model,
                    usage={
                        "prompt_tokens": data["usage"]["prompt_tokens"],
                        "completion_tokens": data["usage"]["completion_tokens"]
                    }
                )
            else:
                raise Exception(f"API Error: {response.status_code}")

    async def test_connection(self) -> Dict[str, Any]:
        try:
            test_request = AIRequest(
                prompt="Hello, this is a test.",
                max_tokens=10
            )
            await self.generate_response(test_request)
            return {"status": "success", "provider": "custom"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Register custom provider
analyst.register_provider("custom", CustomAIProvider(api_key="your-key"))
```

## 📊 Provider Comparison

### Quality Comparison

| Provider | Reasoning | Code | Math | Creative | Speed |
|----------|-----------|------|------|----------|-------|
| OpenAI GPT-4o | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| NVIDIA Qwen2.5-72B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Claude-3 Sonnet | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### Cost Comparison (1M tokens)

| Provider/Model | Input Cost | Output Cost | Free Tier |
|----------------|------------|-------------|-----------|
| **OpenAI GPT-4o-mini** | $0.15 | $0.60 | No |
| **NVIDIA Qwen2.5-72B** | Free | Free | Yes (10M/day) |
| **OpenRouter Claude Sonnet** | $3.00 | $15.00 | No |

### Feature Comparison

| Feature | OpenAI | NVIDIA | OpenRouter |
|---------|--------|--------|------------|
| **Free Tier** | ❌ | ✅ | ⚪ (Limited) |
| **Streaming** | ✅ | ✅ | ✅ |
| **Function Calling** | ✅ | ⚪ | ✅ |
| **JSON Mode** | ✅ | ❌ | ✅ |
| **Image Input** | ✅ | ⚪ | ✅ |
| **Rate Limits** | High | Medium | Variable |

## 🔧 Configuration Best Practices

### 1. Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-your-openai-key
NVIDIA_API_KEY=nvapi-your-nvidia-key
OPENROUTER_API_KEY=sk-or-your-openrouter-key

# Provider preferences
AI_PRIMARY_PROVIDER=nvidia
AI_FALLBACK_PROVIDERS=openai,openrouter
```

### 2. Cost Control

```python
# Set spending limits and monitoring
analyst = DataAnalyst(
    # API keys...
    cost_limits={
        "daily_limit": 10.00,      # $10 per day maximum
        "monthly_limit": 200.00,   # $200 per month maximum
        "per_request_limit": 0.50  # $0.50 per request maximum
    },
    cost_tracking=True
)

# Monitor costs
costs = analyst.get_cost_summary()
print(f"Today's costs: ${costs['daily_cost']:.2f}")
print(f"Month's costs: ${costs['monthly_cost']:.2f}")
```

### 3. Performance Optimization

```python
# Optimize for speed vs quality
fast_config = {
    "primary_provider": "nvidia",     # Free and fast
    "model": "meta/llama-3.1-8b-instruct",  # Smaller model
    "temperature": 0.1,               # More deterministic
    "max_tokens": 1000               # Shorter responses
}

quality_config = {
    "primary_provider": "openai",     # Highest quality
    "model": "gpt-4o",               # Best model
    "temperature": 0.3,               # Balanced creativity
    "max_tokens": 2000               # Detailed responses
}

# Switch configs based on use case
if analysis_type == "quick_summary":
    analyst.update_config(fast_config)
else:
    analyst.update_config(quality_config)
```

## 🚨 Error Handling

### Common Provider Issues

```python
# Handle provider-specific errors
try:
    response = await analyst.analyze_dataset(dataset_info)
except ProviderError as e:
    if e.error_type == "rate_limit":
        print(f"Rate limited by {e.provider}. Retry after: {e.retry_after}s")
    elif e.error_type == "invalid_key":
        print(f"Invalid API key for {e.provider}")
    elif e.error_type == "insufficient_credits":
        print(f"Insufficient credits for {e.provider}")
    else:
        print(f"Provider error: {e}")

except TimeoutError:
    print("Request timed out. Try reducing max_tokens or using a faster provider.")

except Exception as e:
    print(f"Unexpected error: {e}")
```

### Automatic Failover

```python
# Configure automatic failover
analyst = DataAnalyst(
    # Provider configs...
    failover_config={
        "max_retries": 3,
        "retry_delay": 1.0,
        "backoff_factor": 2.0,
        "failover_on_errors": ["rate_limit", "timeout", "server_error"]
    }
)
```

## 📈 Monitoring & Analytics

### Provider Performance

```python
# Get provider statistics
stats = analyst.get_provider_stats()

for provider, provider_stats in stats.items():
    print(f"\n{provider.upper()} Statistics:")
    print(f"  Requests: {provider_stats['total_requests']}")
    print(f"  Success Rate: {provider_stats['success_rate']:.1%}")
    print(f"  Avg Response Time: {provider_stats['avg_response_time']:.2f}s")
    print(f"  Total Cost: ${provider_stats['total_cost']:.2f}")
```

### Usage Analytics

```python
# Track usage patterns
usage = analyst.get_usage_analytics(days=30)

print(f"Most used provider: {usage['top_provider']}")
print(f"Peak usage time: {usage['peak_hour']}")
print(f"Cost per analysis: ${usage['cost_per_analysis']:.3f}")
print(f"Cache hit rate: {usage['cache_hit_rate']:.1%}")
```

---

## 🔗 Additional Resources

- **OpenAI Documentation**: [https://platform.openai.com/docs](https://platform.openai.com/docs)
- **NVIDIA AI Documentation**: [https://docs.nvidia.com/ai](https://docs.nvidia.com/ai)
- **OpenRouter API Reference**: [https://openrouter.ai/docs](https://openrouter.ai/docs)

For technical support or provider-specific issues, consult the respective provider's documentation and support channels.