"""
OpenRouter Provider Implementation

Provides access to multiple AI models through OpenRouter API.
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


class OpenRouterProvider(BaseAIProvider):
    """OpenRouter API provider with access to multiple models"""

    # Popular models available on OpenRouter
    MODELS = [
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-pro",
        "meta-llama/llama-3-70b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "cohere/command-r-plus"
    ]

    DEFAULT_MODEL = "openai/gpt-4o-mini"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str = None, app_name: str = "Scout-Data-Discovery", **kwargs):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx library not available. Install with: pip install httpx")

        super().__init__(
            api_key=api_key,
            default_model=model or self.DEFAULT_MODEL,
            **kwargs
        )
        self.app_name = app_name
        self._initialize_client()

    @property
    def provider_name(self) -> str:
        return "OpenRouter"

    def _initialize_client(self):
        """Initialize HTTP client for OpenRouter"""
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://scout-data-discovery.com",
                "X-Title": self.app_name,
                "Content-Type": "application/json"
            },
            timeout=60.0
        )
        logger.info(f"Initialized OpenRouter client with model: {self.default_model}")

    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate response using OpenRouter API"""
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
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
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

            return AIResponse(
                content=content,
                provider=self.provider_name,
                model=model,
                usage=usage,
                metadata={
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "response_id": data.get("id"),
                    "provider_model": model
                }
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (model-agnostic)"""
        # Conservative estimation: ~4 chars per token
        return len(text) // 4

    def get_available_models(self) -> List[str]:
        """Get available OpenRouter models"""
        return self.MODELS.copy()

    async def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information from OpenRouter"""
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching model info: {e}")
            return {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()