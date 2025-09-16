"""
Base AI Provider Interface

Defines the common interface that all AI providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import asyncio


@dataclass
class AIResponse:
    """Standardized AI response format"""
    content: str
    provider: str
    model: str
    usage: Dict[str, Any]
    cached: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AIRequest:
    """Standardized AI request format"""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    model: Optional[str] = None
    context_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.context_data is None:
            self.context_data = {}


class BaseAIProvider(ABC):
    """Base class for all AI providers"""

    def __init__(self, api_key: str, default_model: str, **kwargs):
        self.api_key = api_key
        self.default_model = default_model
        self.config = kwargs
        self._client = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name"""
        pass

    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client"""
        pass

    @abstractmethod
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate AI response from request"""
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass

    def validate_request(self, request: AIRequest) -> bool:
        """Validate request parameters"""
        if not request.prompt or not request.prompt.strip():
            return False
        return True

    async def health_check(self) -> bool:
        """Check if the provider is accessible"""
        try:
            test_request = AIRequest(
                prompt="Hello",
                max_tokens=10,
                temperature=0
            )
            response = await self.generate_response(test_request)
            return response is not None
        except Exception:
            return False

    def __str__(self):
        return f"{self.provider_name}Provider(model={self.default_model})"