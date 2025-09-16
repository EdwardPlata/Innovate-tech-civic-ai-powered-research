"""
Unified AI Manager

Provides a single interface for managing multiple AI providers with automatic
failover, load balancing, and enhanced error handling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import os
from pathlib import Path
import time
from collections import defaultdict

from .base_provider import BaseAIProvider, AIRequest, AIResponse
from .cache_manager import CacheManager
from ..providers.openai_provider import OpenAIProvider
from ..providers.openrouter_provider import OpenRouterProvider
from ..providers.nvidia_provider import NvidiaProvider


logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider status states"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class UnifiedAIManager:
    """
    Unified AI Manager with intelligent provider selection and failover
    """

    def __init__(
        self,
        cache_dir: str = "./ai_cache",
        enable_semantic_cache: bool = True,
        **provider_configs
    ):
        """
        Initialize Unified AI Manager

        Args:
            cache_dir: Directory for caching
            enable_semantic_cache: Enable semantic similarity caching
            **provider_configs: API keys and configs for providers
        """
        self.provider_configs = provider_configs
        self.provider_stats = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'avg_response_time': 0.0,
            'last_used': None,
            'status': ProviderStatus.UNAVAILABLE
        })

        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            enable_semantic=enable_semantic_cache
        )

        # Initialize providers
        self.providers: Dict[str, BaseAIProvider] = {}
        self.provider_priorities = []  # Dynamic priority based on performance
        self._initialize_all_providers()

        logger.info(f"UnifiedAIManager initialized with {len(self.providers)} providers")

    def _initialize_all_providers(self):
        """Initialize all available AI providers"""

        # Initialize OpenAI
        openai_key = self.provider_configs.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                self.providers['openai'] = OpenAIProvider(
                    api_key=openai_key,
                    model=self.provider_configs.get('openai_model')
                )
                self.provider_stats['openai']['status'] = ProviderStatus.AVAILABLE
                self.provider_priorities.append('openai')
                logger.info("✅ OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")

        # Initialize OpenRouter
        openrouter_key = self.provider_configs.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
        if openrouter_key:
            try:
                self.providers['openrouter'] = OpenRouterProvider(
                    api_key=openrouter_key,
                    model=self.provider_configs.get('openrouter_model')
                )
                self.provider_stats['openrouter']['status'] = ProviderStatus.AVAILABLE
                self.provider_priorities.append('openrouter')
                logger.info("✅ OpenRouter provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter: {e}")

        # Initialize NVIDIA
        nvidia_key = self.provider_configs.get('nvidia_api_key') or os.getenv('NVIDIA_API_KEY')
        if nvidia_key:
            try:
                self.providers['nvidia'] = NvidiaProvider(
                    api_key=nvidia_key,
                    model=self.provider_configs.get('nvidia_model')
                )
                self.provider_stats['nvidia']['status'] = ProviderStatus.AVAILABLE
                self.provider_priorities.append('nvidia')
                logger.info("✅ NVIDIA provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA: {e}")

        if not self.providers:
            raise ValueError("No AI providers could be initialized. Check your API keys.")

        # Sort by preference (OpenAI -> OpenRouter -> NVIDIA)
        preferred_order = ['openai', 'openrouter', 'nvidia']
        self.provider_priorities = sorted(
            self.provider_priorities,
            key=lambda x: preferred_order.index(x) if x in preferred_order else 999
        )

    async def generate_response(
        self,
        request: AIRequest,
        use_cache: bool = True,
        max_retries: int = 2
    ) -> AIResponse:
        """
        Generate AI response with intelligent provider selection and failover

        Args:
            request: AI request
            use_cache: Whether to use caching
            max_retries: Maximum retry attempts per provider

        Returns:
            AI response
        """

        # Check cache first
        if use_cache:
            cached_response = await self.cache_manager.get_cached_response(request)
            if cached_response:
                return cached_response

        # Try providers in priority order
        last_exception = None
        for provider_name in self._get_provider_order():
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]

            # Skip if provider is in error state and hasn't cooled down
            if self._should_skip_provider(provider_name):
                continue

            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()

                    # Update stats
                    self.provider_stats[provider_name]['requests'] += 1

                    # Generate response
                    response = await provider.generate_response(request)

                    # Update success stats
                    response_time = time.time() - start_time
                    self._update_provider_success(provider_name, response_time)

                    # Cache the response
                    if use_cache:
                        await self.cache_manager.cache_response(request, response)

                    return response

                except Exception as e:
                    last_exception = e
                    self._update_provider_failure(provider_name, str(e))

                    logger.warning(f"Provider {provider_name} attempt {attempt + 1} failed: {e}")

                    if attempt < max_retries:
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

        # If all providers failed
        if last_exception:
            raise Exception(f"All AI providers failed. Last error: {last_exception}")
        else:
            raise Exception("No available AI providers")

    async def answer_question(
        self,
        question: str,
        dataset_info: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question with proper context formatting

        Args:
            question: User's question
            dataset_info: Dataset information
            context: Additional context
            use_cache: Whether to use caching

        Returns:
            Formatted answer response
        """

        # Build context
        full_context = ""
        if dataset_info:
            full_context += f"Dataset: {dataset_info.get('name', 'Unknown')}\n"
            full_context += f"Description: {dataset_info.get('description', 'No description')}\n"
            full_context += f"Category: {dataset_info.get('category', 'Unknown')}\n"

        if context:
            full_context += f"\nAdditional Context:\n{context}\n"

        prompt = f"""Based on the following information, please answer this question: "{question}"

{full_context}

Please provide a comprehensive and helpful answer based on the available information. If the answer cannot be determined from the provided information, please explain what additional data would be needed."""

        request = AIRequest(
            prompt=prompt,
            system_prompt="You are an expert data analyst. Provide clear, accurate, and helpful answers based on the available data and context.",
            temperature=0.4,
            max_tokens=1500
        )

        try:
            response = await self.generate_response(request, use_cache=use_cache)

            return {
                'answer': response.content,
                'provider': response.provider,
                'model': response.model,
                'cached': response.cached,
                'metadata': {
                    'question': question,
                    'context_provided': bool(full_context),
                    'dataset_info': bool(dataset_info),
                    'cached': response.cached,
                    'response_time': time.time()
                }
            }

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'provider': 'error',
                'model': 'none',
                'cached': False,
                'metadata': {
                    'error': str(e),
                    'question': question
                }
            }

    def _get_provider_order(self) -> List[str]:
        """Get providers ordered by performance and availability"""
        available_providers = [
            name for name in self.provider_priorities
            if name in self.providers and
            self.provider_stats[name]['status'] in [ProviderStatus.AVAILABLE, ProviderStatus.RATE_LIMITED]
        ]

        # Sort by success rate and response time
        def score_provider(name):
            stats = self.provider_stats[name]
            if stats['requests'] == 0:
                return 0.5  # Neutral score for untested providers

            success_rate = stats['successes'] / stats['requests']
            # Lower response time is better (invert for scoring)
            response_score = 1 / (1 + stats['avg_response_time'])

            return success_rate * 0.7 + response_score * 0.3

        return sorted(available_providers, key=score_provider, reverse=True)

    def _should_skip_provider(self, provider_name: str) -> bool:
        """Check if provider should be skipped based on recent failures"""
        stats = self.provider_stats[provider_name]

        if stats['status'] == ProviderStatus.ERROR:
            # Skip if failed recently (simple cooldown)
            if stats['last_used'] and time.time() - stats['last_used'] < 60:
                return True

        return False

    def _update_provider_success(self, provider_name: str, response_time: float):
        """Update provider stats after successful request"""
        stats = self.provider_stats[provider_name]
        stats['successes'] += 1
        stats['last_used'] = time.time()
        stats['status'] = ProviderStatus.AVAILABLE

        # Update average response time
        if stats['successes'] == 1:
            stats['avg_response_time'] = response_time
        else:
            stats['avg_response_time'] = (
                stats['avg_response_time'] * 0.8 + response_time * 0.2
            )

    def _update_provider_failure(self, provider_name: str, error: str):
        """Update provider stats after failed request"""
        stats = self.provider_stats[provider_name]
        stats['failures'] += 1
        stats['last_used'] = time.time()

        # Check if it's a rate limit error
        if 'rate limit' in error.lower() or 'quota' in error.lower():
            stats['status'] = ProviderStatus.RATE_LIMITED
        else:
            stats['status'] = ProviderStatus.ERROR

    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all providers"""
        results = {}

        for name, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                results[name] = is_healthy

                if is_healthy:
                    self.provider_stats[name]['status'] = ProviderStatus.AVAILABLE
                else:
                    self.provider_stats[name]['status'] = ProviderStatus.ERROR

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False
                self.provider_stats[name]['status'] = ProviderStatus.ERROR

        return results

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive status of all providers"""
        status = {}

        for name in self.providers:
            provider = self.providers[name]
            stats = self.provider_stats[name]

            status[name] = {
                'available': stats['status'] == ProviderStatus.AVAILABLE,
                'model': provider.default_model,
                'provider_class': provider.__class__.__name__,
                'status': stats['status'].value,
                'requests': stats['requests'],
                'successes': stats['successes'],
                'failures': stats['failures'],
                'success_rate': stats['successes'] / max(stats['requests'], 1),
                'avg_response_time': stats['avg_response_time'],
                'last_used': stats['last_used']
            }

        return status

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics"""
        return self.cache_manager.get_cache_stats()

    def clear_cache(self, cache_type: str = "all"):
        """Clear specified cache"""
        self.cache_manager.clear_cache(cache_type)

    def add_provider(self, name: str, provider: BaseAIProvider):
        """Dynamically add a new provider"""
        self.providers[name] = provider
        self.provider_priorities.append(name)
        self.provider_stats[name]['status'] = ProviderStatus.AVAILABLE
        logger.info(f"Added provider: {name}")

    def remove_provider(self, name: str):
        """Remove a provider"""
        if name in self.providers:
            del self.providers[name]
            if name in self.provider_priorities:
                self.provider_priorities.remove(name)
            logger.info(f"Removed provider: {name}")

    def get_best_provider(self) -> Optional[str]:
        """Get the currently best performing provider"""
        ordered = self._get_provider_order()
        return ordered[0] if ordered else None