# Base Provider Interface Documentation

## 📋 Overview

The `base_provider.py` module defines the fundamental interfaces and data structures that all AI providers must implement. It provides standardized request/response formats and abstract base classes for consistent AI provider integration.

## 🎯 Purpose

- **Primary Role**: Define common interface for all AI providers
- **Key Responsibility**: Standardize AI request/response formats across providers
- **Core Function**: Abstract provider-specific implementations behind unified interface
- **Integration Point**: Foundation for multi-provider AI system architecture

## 🏗️ Architecture

```python
BaseAIProvider Interface
├── Request/Response Models
│   ├── AIRequest (standardized input format)
│   ├── AIResponse (standardized output format)
│   └── Common metadata structures
├── Abstract Methods
│   ├── generate_response() (core AI generation)
│   ├── estimate_tokens() (token counting)
│   ├── get_available_models() (model enumeration)
│   └── health_check() (provider availability)
├── Common Functionality
│   ├── Request validation
│   ├── Client initialization
│   └── Error handling patterns
└── Provider Registration
    ├── Provider identification
    ├── Configuration management
    └── Capability declaration
```

## 📊 Data Models

### AIRequest

**Standardized format for all AI provider requests**

```python
@dataclass
class AIRequest:
    """
    Unified request format for all AI providers
    
    This class standardizes how requests are made to different AI providers,
    allowing the system to switch between providers seamlessly.
    """
    
    # Core Content
    prompt: str                           # Main user prompt/question
    system_prompt: Optional[str] = None   # System context/instructions
    
    # Generation Parameters
    temperature: float = 0.7              # Randomness control (0.0-1.0)
    max_tokens: Optional[int] = None      # Maximum response length
    model: Optional[str] = None           # Specific model to use
    
    # Context & Metadata
    context_data: Dict[str, Any] = None   # Additional context information
    
    def __post_init__(self):
        """Initialize context_data if None"""
        if self.context_data is None:
            self.context_data = {}

# Usage Examples:

# Basic request
basic_request = AIRequest(
    prompt="Analyze this dataset quality",
    temperature=0.1
)

# Advanced request with system context
advanced_request = AIRequest(
    prompt="What are the main quality issues in this data?",
    system_prompt="You are a data quality expert. Provide detailed analysis.",
    temperature=0.2,
    max_tokens=1500,
    model="gpt-4o",
    context_data={
        "dataset_id": "customer-data-2023",
        "analysis_type": "quality_assessment",
        "user_role": "data_analyst"
    }
)

# Interactive Q&A request
qa_request = AIRequest(
    prompt="How can I improve data completeness?",
    system_prompt="Continue our conversation about data quality improvements.",
    context_data={
        "conversation_history": [
            {"role": "user", "content": "What quality issues exist?"},
            {"role": "assistant", "content": "I found several completeness issues..."}
        ],
        "dataset_context": {"completeness_score": 0.78}
    }
)
```

### AIResponse

**Standardized format for all AI provider responses**

```python
@dataclass
class AIResponse:
    """
    Unified response format from all AI providers
    
    Provides consistent response structure regardless of which AI provider
    was used to generate the response.
    """
    
    # Core Content
    content: str                          # Generated response text
    
    # Provider Information
    provider: str                         # Provider used ("openai", "anthropic", etc.)
    model: str                           # Specific model used
    
    # Usage & Performance Metrics
    usage: Dict[str, Any]                # Token usage and billing information
    cached: bool = False                 # Whether response came from cache
    
    # Additional Information
    metadata: Dict[str, Any] = None      # Provider-specific metadata
    
    def __post_init__(self):
        """Initialize metadata if None"""
        if self.metadata is None:
            self.metadata = {}

# Response Examples:

# Standard analysis response
analysis_response = AIResponse(
    content="This dataset shows high quality with 92% completeness...",
    provider="openai",
    model="gpt-4o",
    usage={
        "prompt_tokens": 150,
        "completion_tokens": 400,
        "total_tokens": 550,
        "cost_usd": 0.0165
    },
    cached=False,
    metadata={
        "response_time": 2.3,
        "timestamp": "2023-12-15T10:30:00Z",
        "request_id": "req_abc123"
    }
)

# Cached response example
cached_response = AIResponse(
    content="Based on previous analysis, the data quality score is 87%...",
    provider="openai",
    model="gpt-4o",
    usage={
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0
    },
    cached=True,
    metadata={
        "cache_hit": True,
        "original_timestamp": "2023-12-15T09:45:00Z",
        "cache_source": "semantic_similarity"
    }
)
```

## 🔧 Base Provider Class

### BaseAIProvider

**Abstract base class that all AI providers must implement**

```python
class BaseAIProvider(ABC):
    """
    Abstract base class for all AI providers
    
    Defines the contract that all AI providers must follow to ensure
    consistent behavior across different AI services.
    """
    
    def __init__(self, api_key: str, default_model: str, **kwargs):
        """
        Initialize provider with configuration
        
        Args:
            api_key: API key for the provider service
            default_model: Default model to use for requests
            **kwargs: Provider-specific configuration options
            
        Common Configuration Options:
            - timeout: Request timeout in seconds (default: 30)
            - max_retries: Maximum retry attempts (default: 3)
            - base_url: Custom API endpoint URL
            - organization: Organization ID (for some providers)
            - rate_limit: Requests per minute limit
        """
        self.api_key = api_key
        self.default_model = default_model
        self.config = kwargs
        self._client = None
        
        # Extract common configuration
        self.timeout = kwargs.get('timeout', 30)
        self.max_retries = kwargs.get('max_retries', 3)
        self.rate_limit = kwargs.get('rate_limit', None)
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return unique provider identifier
        
        Returns:
            str: Provider name (e.g., "openai", "anthropic", "nvidia")
            
        Usage:
            This property is used for:
            - Logging and monitoring
            - Cache key generation
            - Provider selection logic
            - Usage analytics
        """
        pass
    
    @abstractmethod
    def _initialize_client(self):
        """
        Initialize provider-specific client
        
        This method should:
        1. Create the provider's API client
        2. Configure authentication
        3. Set up any provider-specific options
        4. Validate API key if possible
        
        Should be called lazily (on first use) for performance.
        """
        pass
    
    @abstractmethod
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """
        Generate AI response from standardized request
        
        Args:
            request: Standardized AIRequest object
            
        Returns:
            AIResponse: Standardized response with content and metadata
            
        Raises:
            ProviderError: When provider-specific errors occur
            ValidationError: When request validation fails
            TimeoutError: When request exceeds timeout
            
        Implementation Requirements:
        1. Validate the request using validate_request()
        2. Initialize client if not already done
        3. Transform AIRequest to provider-specific format
        4. Make API call with proper error handling
        5. Transform response to AIResponse format
        6. Include accurate usage and cost information
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            int: Estimated token count
            
        This is used for:
        - Cost estimation before making requests
        - Request optimization (splitting large requests)
        - Usage planning and budgeting
        - Performance optimization
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider
        
        Returns:
            List[str]: Model identifiers available for use
            
        Should return:
        - All models accessible with current API key
        - Models should be returned in preference order
        - Include both current and deprecated models
        - Format as provider-specific model names
        """
        pass
    
    def validate_request(self, request: AIRequest) -> bool:
        """
        Validate request parameters
        
        Args:
            request: AIRequest to validate
            
        Returns:
            bool: True if request is valid
            
        Validation checks:
        1. Prompt is not empty
        2. Temperature is in valid range (0.0-2.0)
        3. Max tokens is reasonable (if specified)
        4. Model is available (if specified)
        5. Provider-specific validations
        """
        # Basic validation
        if not request.prompt or not request.prompt.strip():
            return False
            
        if not (0.0 <= request.temperature <= 2.0):
            return False
            
        if request.max_tokens and request.max_tokens <= 0:
            return False
            
        return True
    
    async def health_check(self) -> bool:
        """
        Check if provider is accessible and responding
        
        Returns:
            bool: True if provider is healthy
            
        Performs:
        1. Simple API call with minimal cost
        2. Validates API key and connectivity
        3. Checks for any service disruptions
        4. Returns quickly (under 10 seconds)
        """
        try:
            test_request = AIRequest(
                prompt="Hello",
                max_tokens=10,
                temperature=0
            )
            response = await self.generate_response(test_request)
            return response is not None and len(response.content) > 0
        except Exception:
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get provider capabilities and features
        
        Returns:
            Dict describing provider capabilities
        """
        return {
            "provider": self.provider_name,
            "models": self.get_available_models(),
            "supports_streaming": False,  # Override in subclasses
            "supports_function_calling": False,
            "supports_image_input": False,
            "max_tokens": 4096,  # Default, override in subclasses
            "rate_limits": self.rate_limit
        }
    
    def __str__(self):
        """String representation of provider"""
        return f"{self.provider_name}Provider(model={self.default_model})"
    
    def __repr__(self):
        """Detailed string representation"""
        return f"{self.__class__.__name__}(provider={self.provider_name}, model={self.default_model})"
```

## 🔧 Implementation Patterns

### Provider Implementation Template

**Template for implementing new AI providers**

```python
from .base_provider import BaseAIProvider, AIRequest, AIResponse
import httpx
import json
import asyncio

class CustomAIProvider(BaseAIProvider):
    """
    Implementation template for new AI providers
    """
    
    @property
    def provider_name(self) -> str:
        return "custom_provider"
    
    def _initialize_client(self):
        """Initialize provider-specific client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.get('base_url', 'https://api.customprovider.com'),
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                timeout=self.timeout
            )
    
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate response using custom provider API"""
        
        # 1. Validate request
        if not self.validate_request(request):
            raise ValueError("Invalid request parameters")
        
        # 2. Initialize client
        self._initialize_client()
        
        # 3. Transform to provider format
        provider_request = {
            "model": request.model or self.default_model,
            "messages": [
                {"role": "system", "content": request.system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": request.prompt}
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 2000
        }
        
        # 4. Make API call with retries
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    "/v1/chat/completions",
                    json=provider_request
                )
                response.raise_for_status()
                break
                
            except httpx.HTTPStatusError as e:
                if attempt == self.max_retries - 1:
                    raise ProviderError(f"API request failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # 5. Parse response
        response_data = response.json()
        
        # 6. Transform to standard format
        return AIResponse(
            content=response_data['choices'][0]['message']['content'],
            provider=self.provider_name,
            model=response_data['model'],
            usage={
                "prompt_tokens": response_data['usage']['prompt_tokens'],
                "completion_tokens": response_data['usage']['completion_tokens'],
                "total_tokens": response_data['usage']['total_tokens'],
                "cost_usd": self._calculate_cost(response_data['usage'])
            },
            metadata={
                "request_id": response_data.get('id'),
                "response_time": response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using simple heuristic"""
        # Simple estimation: ~4 characters per token
        return max(1, len(text) // 4)
    
    def get_available_models(self) -> List[str]:
        """Return available models for this provider"""
        return [
            "custom-model-large",
            "custom-model-medium", 
            "custom-model-small"
        ]
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate request cost based on usage"""
        # Example pricing (customize per provider)
        prompt_cost = usage['prompt_tokens'] * 0.00001  # $0.01 per 1K tokens
        completion_cost = usage['completion_tokens'] * 0.00002  # $0.02 per 1K tokens
        return prompt_cost + completion_cost
```

### Error Handling

**Standardized error handling patterns**

```python
class ProviderError(Exception):
    """Base exception for provider-specific errors"""
    
    def __init__(self, message: str, provider: str, error_code: str = None, retry_after: int = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.retry_after = retry_after
        self.timestamp = datetime.now()

class ValidationError(ProviderError):
    """Request validation errors"""
    pass

class RateLimitError(ProviderError):
    """Rate limiting errors"""
    pass

class AuthenticationError(ProviderError):
    """Authentication/authorization errors"""
    pass

class ModelNotFoundError(ProviderError):
    """Model not available errors"""
    pass

# Error handling in provider implementations
async def generate_response(self, request: AIRequest) -> AIResponse:
    try:
        # ... implementation
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise AuthenticationError(
                "Invalid API key", 
                self.provider_name, 
                "invalid_api_key"
            )
        elif e.response.status_code == 429:
            retry_after = int(e.response.headers.get('Retry-After', 60))
            raise RateLimitError(
                "Rate limit exceeded", 
                self.provider_name, 
                "rate_limit_exceeded",
                retry_after
            )
        elif e.response.status_code == 404:
            raise ModelNotFoundError(
                f"Model not found: {request.model}", 
                self.provider_name, 
                "model_not_found"
            )
        else:
            raise ProviderError(
                f"HTTP {e.response.status_code}: {e.response.text}", 
                self.provider_name,
                f"http_{e.response.status_code}"
            )
    
    except httpx.TimeoutException:
        raise ProviderError(
            f"Request timeout after {self.timeout}s", 
            self.provider_name, 
            "timeout"
        )
    
    except Exception as e:
        raise ProviderError(
            f"Unexpected error: {str(e)}", 
            self.provider_name, 
            "unexpected_error"
        )
```

## 🔧 Provider Registration

### Provider Registry Pattern

**System for managing multiple providers**

```python
class ProviderRegistry:
    """
    Registry for managing multiple AI providers
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseAIProvider] = {}
        self.default_provider: Optional[str] = None
    
    def register_provider(self, provider: BaseAIProvider, is_default: bool = False):
        """Register a new AI provider"""
        self.providers[provider.provider_name] = provider
        
        if is_default or not self.default_provider:
            self.default_provider = provider.provider_name
    
    def get_provider(self, provider_name: Optional[str] = None) -> BaseAIProvider:
        """Get provider by name or default"""
        name = provider_name or self.default_provider
        
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not registered")
        
        return self.providers[name]
    
    def get_available_providers(self) -> List[str]:
        """Get list of registered provider names"""
        return list(self.providers.keys())
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered providers"""
        results = {}
        
        for name, provider in self.providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception:
                results[name] = False
        
        return results

# Usage example
registry = ProviderRegistry()

# Register providers
registry.register_provider(OpenAIProvider(api_key="sk-..."), is_default=True)
registry.register_provider(AnthropicProvider(api_key="sk-ant-..."))
registry.register_provider(NVIDIAProvider(api_key="nvapi-..."))

# Use providers
primary_provider = registry.get_provider()  # Gets default
specific_provider = registry.get_provider("anthropic")

# Health check
health_status = await registry.health_check_all()
print(f"Provider health: {health_status}")
```

## 🚀 Usage Examples

### Basic Provider Usage

```python
# Direct provider usage
from AI_Functionality.providers.openai_provider import OpenAIProvider

# Initialize provider
provider = OpenAIProvider(
    api_key="sk-your-openai-key",
    default_model="gpt-4o",
    timeout=30,
    max_retries=3
)

# Create request
request = AIRequest(
    prompt="Analyze the quality of this dataset",
    system_prompt="You are a data quality expert",
    temperature=0.1,
    max_tokens=1500
)

# Generate response
response = await provider.generate_response(request)

print(f"Response: {response.content}")
print(f"Model: {response.model}")
print(f"Tokens used: {response.usage['total_tokens']}")
print(f"Cost: ${response.usage['cost_usd']:.4f}")
```

### Multi-Provider Fallback

```python
async def generate_with_fallback(request: AIRequest, providers: List[BaseAIProvider]) -> AIResponse:
    """
    Try multiple providers with automatic fallback
    """
    for provider in providers:
        try:
            # Check if provider is healthy
            if not await provider.health_check():
                continue
            
            # Try to generate response
            response = await provider.generate_response(request)
            return response
            
        except (RateLimitError, AuthenticationError, ModelNotFoundError) as e:
            logger.warning(f"Provider {provider.provider_name} failed: {e}")
            continue
        
        except Exception as e:
            logger.error(f"Unexpected error with {provider.provider_name}: {e}")
            continue
    
    raise ProviderError("All providers failed", "fallback_system", "all_providers_failed")

# Usage
providers = [
    OpenAIProvider(api_key="sk-..."),
    AnthropicProvider(api_key="sk-ant-..."),
    NVIDIAProvider(api_key="nvapi-...")
]

request = AIRequest(prompt="Analyze this data")
response = await generate_with_fallback(request, providers)
```

### Provider Capabilities

```python
# Check provider capabilities
provider = OpenAIProvider(api_key="sk-...")
capabilities = provider.get_capabilities()

print(f"Provider: {capabilities['provider']}")
print(f"Available models: {capabilities['models']}")
print(f"Max tokens: {capabilities['max_tokens']}")
print(f"Supports streaming: {capabilities['supports_streaming']}")

# Model selection based on capabilities
def select_best_model(provider: BaseAIProvider, task_complexity: str) -> str:
    """Select best model based on task complexity"""
    models = provider.get_available_models()
    
    if task_complexity == "simple" and "gpt-3.5-turbo" in models:
        return "gpt-3.5-turbo"
    elif task_complexity == "complex" and "gpt-4o" in models:
        return "gpt-4o"
    else:
        return models[0]  # Default to first available

# Usage
best_model = select_best_model(provider, "complex")
request = AIRequest(prompt="Complex analysis task", model=best_model)
```

## ⚡ Performance Considerations

### Optimization Strategies

1. **Lazy Client Initialization**: Initialize API clients only when needed
2. **Connection Pooling**: Reuse HTTP connections for multiple requests
3. **Request Batching**: Combine multiple requests where possible
4. **Token Estimation**: Pre-estimate costs before making requests
5. **Health Monitoring**: Regular health checks to avoid failed requests

### Memory Management

```python
class ProviderPool:
    """Pool of providers with resource management"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
    
    async def generate_response(self, provider: BaseAIProvider, request: AIRequest) -> AIResponse:
        """Generate response with concurrency control"""
        async with self.semaphore:
            self.active_requests += 1
            try:
                return await provider.generate_response(request)
            finally:
                self.active_requests -= 1
```

This comprehensive documentation covers all aspects of the Base Provider interface, from basic concepts to advanced implementation patterns.