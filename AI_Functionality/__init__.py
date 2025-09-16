"""
AI Functionality Module for Scout Data Discovery

Provides AI-powered data analysis capabilities with support for multiple providers:
- OpenAI GPT models
- OpenRouter (access to various models)
- NVIDIA AI models

Features:
- Smart caching (prompt and semantic)
- Data analysis and insights
- Interactive Q&A capabilities
- Multi-provider fallback
"""

from .core.ai_analyst import DataAnalyst
from .core.cache_manager import CacheManager
from .providers.openai_provider import OpenAIProvider
from .providers.openrouter_provider import OpenRouterProvider
from .providers.nvidia_provider import NvidiaProvider

__version__ = "1.0.0"

__all__ = [
    "DataAnalyst",
    "CacheManager",
    "OpenAIProvider",
    "OpenRouterProvider",
    "NvidiaProvider"
]