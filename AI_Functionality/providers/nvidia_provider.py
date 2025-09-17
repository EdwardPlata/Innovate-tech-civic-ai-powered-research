"""
NVIDIA Provider Implementation

Provides access to NVIDIA AI models through NVIDIA API.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import json

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from ..core.base_provider import BaseAIProvider, AIRequest, AIResponse
except ImportError:
    from core.base_provider import BaseAIProvider, AIRequest, AIResponse


logger = logging.getLogger(__name__)


class NvidiaProvider(BaseAIProvider):
    """NVIDIA API provider"""

    # NVIDIA models available through their API
    MODELS = [
        # Reasoning Models
        "qwen/qwen2.5-72b-instruct",
        "qwen/qwen-2-72b-instruct",
        "qwen/qwen2-72b-instruct",
        "deepseek/deepseek-coder-6.7b-instruct",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.1-405b-instruct",
        "microsoft/phi-3-medium-128k-instruct",
        "microsoft/phi-3-mini-128k-instruct",
        "mistralai/mistral-7b-instruct-v0.3",
        "mistralai/mixtral-8x7b-instruct-v0.1",
        "mistralai/mixtral-8x22b-instruct-v0.1",
        "google/gemma-2b-it",
        "google/gemma-7b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        # Legacy models
        "nvidia/llama-3-70b-instruct",
        "nvidia/mixtral-8x7b-instruct",
        "nvidia/nemotron-4-340b-instruct",
        "nvidia/codellama-70b-instruct"
    ]

    # Reasoning models for enhanced analysis
    REASONING_MODELS = [
        "qwen/qwen2.5-72b-instruct",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.1-70b-instruct",
        "mistralai/mixtral-8x22b-instruct-v0.1"
    ]

    DEFAULT_MODEL = "qwen/qwen2.5-72b-instruct"
    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, api_key: str, model: str = None, **kwargs):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx library not available. Install with: pip install httpx")

        super().__init__(
            api_key=api_key,
            default_model=model or self.DEFAULT_MODEL,
            **kwargs
        )

        # Initialize tracking variables
        self._total_requests = 0
        self._total_tokens = 0
        self._last_request_time = None
        self._successful_requests = 0
        self._failed_requests = 0

        self._initialize_client()

    @property
    def provider_name(self) -> str:
        return "NVIDIA"

    def _initialize_client(self):
        """Initialize HTTP client for NVIDIA API"""
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Scout-Data-Discovery/1.0"
            },
            timeout=120.0  # NVIDIA models can be slower
        )
        logger.info(f"Initialized NVIDIA client with model: {self.default_model}")

    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate response using NVIDIA API"""
        if not self.validate_request(request):
            raise ValueError("Invalid request parameters")

        model = request.model or self.default_model

        # Prepare messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": 1.0,
            "stream": False
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
            # Track request
            self._total_requests += 1
            import time
            self._last_request_time = time.time()

            # Make API call
            response = await self._client.post(
                "/chat/completions",
                json=payload
            )
            response.raise_for_status()

            data = response.json()

            # Extract response data
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            # Track tokens
            if usage:
                self._total_tokens += usage.get("total_tokens", 0)

            # Track success
            self._successful_requests += 1

            return AIResponse(
                content=content,
                provider=self.provider_name,
                model=model,
                usage=usage,
                metadata={
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "response_id": data.get("id"),
                    "provider_model": model,
                    "request_timestamp": self._last_request_time
                }
            )

        except httpx.HTTPStatusError as e:
            self._failed_requests += 1
            error_details = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"NVIDIA HTTP error: {error_details}")

            # Add more specific error handling
            if e.response.status_code == 401:
                raise ValueError("Invalid NVIDIA API key. Please check your API key.")
            elif e.response.status_code == 429:
                raise ValueError("NVIDIA API rate limit exceeded. Please try again later.")
            elif e.response.status_code == 400:
                raise ValueError(f"Invalid request to NVIDIA API: {e.response.text}")
            else:
                raise ValueError(f"NVIDIA API error: {error_details}")

        except httpx.TimeoutException:
            self._failed_requests += 1
            logger.error("NVIDIA API timeout")
            raise TimeoutError("NVIDIA API request timed out")

        except Exception as e:
            self._failed_requests += 1
            logger.error(f"NVIDIA API error: {e}")
            raise ValueError(f"NVIDIA API error: {str(e)}")

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation for NVIDIA models"""
        # Conservative estimation: ~4 chars per token
        return len(text) // 4

    def get_available_models(self) -> List[str]:
        """Get available NVIDIA models"""
        return self.MODELS.copy()

    def get_model_info(self, model: str = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        model = model or self.default_model

        model_info = {
            "name": model,
            "provider": self.provider_name,
            "context_length": 32768,  # Most NVIDIA models support 32k context
            "max_tokens": 4096,
            "supports_streaming": True,
            "supports_system_prompt": True,
            "cost_per_1k_tokens": {
                "input": 0.0,  # Free tier
                "output": 0.0  # Free tier
            }
        }

        # Model-specific adjustments
        if "405b" in model:
            model_info.update({
                "context_length": 128000,  # Llama 3.1 405B has longer context
                "description": "Largest Llama model with exceptional reasoning capabilities"
            })
        elif "qwen2.5-72b" in model:
            model_info.update({
                "description": "Advanced reasoning model optimized for complex analysis"
            })
        elif "mixtral-8x22b" in model:
            model_info.update({
                "description": "Mixture of experts model for specialized tasks"
            })
        elif "gemma-2-27b" in model:
            model_info.update({
                "description": "Google's instruction-tuned model for general tasks"
            })

        return model_info

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to NVIDIA API"""
        try:
            test_request = AIRequest(
                prompt="Hello, this is a connection test. Please respond with 'Connection successful!'",
                max_tokens=10,
                temperature=0.1
            )

            start_time = asyncio.get_event_loop().time()
            response = await self.generate_response(test_request)
            end_time = asyncio.get_event_loop().time()

            return {
                "status": "success",
                "provider": self.provider_name,
                "model": self.default_model,
                "response_time": round(end_time - start_time, 2),
                "test_response": response.content[:100],
                "free_tier_available": True,
                "rate_limits": {
                    "requests_per_day": 1000,
                    "tokens_per_day": 10000000,
                    "concurrent_requests": 10
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "provider": self.provider_name,
                "error": str(e),
                "error_type": self._classify_error(e)
            }

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for better handling"""
        error_str = str(error).lower()

        if "401" in error_str or "unauthorized" in error_str:
            return "invalid_api_key"
        elif "429" in error_str or "rate limit" in error_str:
            return "rate_limit_exceeded"
        elif "400" in error_str or "bad request" in error_str:
            return "invalid_request"
        elif "timeout" in error_str or "connection" in error_str:
            return "connection_error"
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            return "server_error"
        else:
            return "unknown_error"

    def validate_request(self, request: AIRequest) -> bool:
        """Validate request parameters for NVIDIA API"""
        if not request.prompt:
            return False

        # Check model availability
        model = request.model or self.default_model
        if model not in self.MODELS:
            logger.warning(f"Model {model} not in known NVIDIA models list")

        # Validate token limits
        if request.max_tokens and request.max_tokens > 4096:
            logger.warning(f"max_tokens {request.max_tokens} may exceed model limits")

        # Validate temperature
        if request.temperature < 0 or request.temperature > 2:
            logger.warning(f"Temperature {request.temperature} outside recommended range [0, 2]")

        return True

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        import time
        total_requests = self._total_requests
        success_rate = (self._successful_requests / total_requests * 100) if total_requests > 0 else 0

        return {
            "provider": self.provider_name,
            "model": self.default_model,
            "total_requests": total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": round(success_rate, 2),
            "total_tokens": self._total_tokens,
            "last_request_time": self._last_request_time,
            "last_request_ago": round(time.time() - self._last_request_time) if self._last_request_time else None,
            "free_tier": True,
            "cost_estimate": 0.0,  # NVIDIA free tier
            "rate_limits": {
                "requests_per_day": 1000,
                "tokens_per_day": 10000000,
                "concurrent_requests": 10
            }
        }

    def get_reasoning_models(self) -> List[str]:
        """Get models optimized for reasoning tasks"""
        return self.REASONING_MODELS.copy()

    def is_reasoning_model(self, model: str = None) -> bool:
        """Check if model is optimized for reasoning"""
        model = model or self.default_model
        return model in self.REASONING_MODELS

    async def stream_response(self, request: AIRequest):
        """Stream response from NVIDIA API (for future implementation)"""
        # NVIDIA API supports streaming, but implementing it would require
        # additional complexity. For now, we use non-streaming responses.
        logger.info("Streaming not yet implemented for NVIDIA provider")
        return await self.generate_response(request)

    def get_provider_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive information about provider capabilities"""
        return {
            "provider_name": self.provider_name,
            "base_url": self.BASE_URL,
            "default_model": self.default_model,
            "available_models": len(self.MODELS),
            "reasoning_models": len(self.REASONING_MODELS),
            "capabilities": {
                "streaming": False,  # Not implemented yet
                "function_calling": False,
                "image_input": False,
                "json_mode": False,
                "system_prompts": True,
                "temperature_control": True,
                "max_tokens_control": True
            },
            "limits": {
                "max_context_length": 128000,  # For largest models
                "max_output_tokens": 4096,
                "temperature_range": [0.0, 2.0],
                "requests_per_day": 1000,
                "tokens_per_day": 10000000
            },
            "pricing": {
                "free_tier": True,
                "input_cost_per_1k_tokens": 0.0,
                "output_cost_per_1k_tokens": 0.0,
                "note": "NVIDIA provides free access to foundation models"
            },
            "features": {
                "mathematical_reasoning": True,
                "code_understanding": True,
                "multilingual": True,
                "instruction_following": True,
                "complex_reasoning": True
            }
        }

    def reset_stats(self):
        """Reset usage statistics"""
        self._total_requests = 0
        self._total_tokens = 0
        self._last_request_time = None
        self._successful_requests = 0
        self._failed_requests = 0
        logger.info("NVIDIA provider statistics reset")

    def get_model_recommendations(self, task_type: str = "general") -> List[str]:
        """Get model recommendations based on task type"""
        recommendations = {
            "reasoning": [
                "qwen/qwen2.5-72b-instruct",
                "meta/llama-3.1-405b-instruct",
                "meta/llama-3.1-70b-instruct"
            ],
            "code": [
                "deepseek/deepseek-coder-6.7b-instruct",
                "qwen/qwen2.5-72b-instruct",
                "meta/llama-3.1-70b-instruct"
            ],
            "math": [
                "qwen/qwen2.5-72b-instruct",
                "meta/llama-3.1-405b-instruct",
                "mistralai/mixtral-8x22b-instruct-v0.1"
            ],
            "general": [
                "qwen/qwen2.5-72b-instruct",
                "meta/llama-3.1-70b-instruct",
                "google/gemma-2-27b-it"
            ],
            "fast": [
                "meta/llama-3.1-8b-instruct",
                "google/gemma-2b-it",
                "microsoft/phi-3-mini-128k-instruct"
            ]
        }

        return recommendations.get(task_type.lower(), recommendations["general"])

    def __repr__(self) -> str:
        return f"NvidiaProvider(model='{self.default_model}', api_key='***{self.api_key[-4:]}')"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, '_client') and self._client:
            await self._client.aclose()

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, '_client') and self._client:
            try:
                asyncio.create_task(self._client.aclose())
            except RuntimeError:
                # Event loop might be closed
                pass