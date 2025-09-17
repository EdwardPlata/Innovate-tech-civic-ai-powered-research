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

try:
    from .core.ai_analyst import DataAnalyst
    from .core.cache_manager import CacheManager
    from .providers.openai_provider import OpenAIProvider
    from .providers.openrouter_provider import OpenRouterProvider
    from .providers.nvidia_provider import NvidiaProvider
except ImportError:
    # Fallback for when module is run directly
    from core.ai_analyst import DataAnalyst
    from core.cache_manager import CacheManager
    from providers.openai_provider import OpenAIProvider
    from providers.openrouter_provider import OpenRouterProvider
    from providers.nvidia_provider import NvidiaProvider

# Import AnalysisType for easy access
try:
    from .core.ai_analyst import AnalysisType
except ImportError:
    from core.ai_analyst import AnalysisType

__version__ = "1.0.0"

__all__ = [
    "DataAnalyst",
    "AnalysisType",
    "CacheManager",
    "OpenAIProvider",
    "OpenRouterProvider",
    "NvidiaProvider"
]