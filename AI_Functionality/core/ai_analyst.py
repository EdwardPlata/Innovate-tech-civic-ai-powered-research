"""
AI Data Analyst Core Module

Main orchestrator for AI-powered data analysis with multi-provider support,
caching, and specialized analysis capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import os
from pathlib import Path

from .base_provider import BaseAIProvider, AIRequest, AIResponse
from .cache_manager import CacheManager
from ..providers.openai_provider import OpenAIProvider
from ..providers.openrouter_provider import OpenRouterProvider
from ..providers.nvidia_provider import NvidiaProvider


logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of analysis the AI analyst can perform"""
    OVERVIEW = "overview"
    QUALITY = "quality"
    INSIGHTS = "insights"
    RELATIONSHIPS = "relationships"
    TRENDS = "trends"
    RECOMMENDATIONS = "recommendations"
    CUSTOM = "custom"


class DataAnalyst:
    """
    AI-powered data analyst with multi-provider support and advanced caching
    """

    def __init__(
        self,
        primary_provider: str = "openai",
        fallback_providers: List[str] = None,
        cache_dir: str = "./ai_cache",
        enable_semantic_cache: bool = True,
        **provider_configs
    ):
        """
        Initialize AI Data Analyst

        Args:
            primary_provider: Primary AI provider ("openai", "openrouter", "nvidia")
            fallback_providers: List of fallback providers
            cache_dir: Directory for caching
            enable_semantic_cache: Enable semantic similarity caching
            **provider_configs: API keys and configs for providers
        """
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []
        self.provider_configs = provider_configs

        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            enable_semantic=enable_semantic_cache
        )

        # Initialize providers
        self.providers: Dict[str, BaseAIProvider] = {}
        self._initialize_providers()

        logger.info(f"DataAnalyst initialized with primary: {primary_provider}")

    def _initialize_providers(self):
        """Initialize AI providers based on available API keys"""

        # OpenAI
        openai_key = self.provider_configs.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                self.providers['openai'] = OpenAIProvider(
                    api_key=openai_key,
                    model=self.provider_configs.get('openai_model')
                )
                logger.info("✅ OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")

        # OpenRouter
        openrouter_key = self.provider_configs.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
        if openrouter_key:
            try:
                self.providers['openrouter'] = OpenRouterProvider(
                    api_key=openrouter_key,
                    model=self.provider_configs.get('openrouter_model')
                )
                logger.info("✅ OpenRouter provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter: {e}")

        # NVIDIA
        nvidia_key = self.provider_configs.get('nvidia_api_key') or os.getenv('NVIDIA_API_KEY')
        if nvidia_key:
            try:
                self.providers['nvidia'] = NvidiaProvider(
                    api_key=nvidia_key,
                    model=self.provider_configs.get('nvidia_model')
                )
                logger.info("✅ NVIDIA provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA: {e}")

        if not self.providers:
            raise ValueError("No AI providers could be initialized. Check your API keys.")

    async def analyze_dataset(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]] = None,
        analysis_type: AnalysisType = AnalysisType.OVERVIEW,
        custom_prompt: Optional[str] = None,
        use_cache: bool = True,
        timeout: int = 60,
        progress_callback: Optional[callable] = None
    ) -> AIResponse:
        """
        Analyze a dataset with AI

        Args:
            dataset_info: Dataset metadata
            sample_data: Sample records from the dataset
            analysis_type: Type of analysis to perform
            custom_prompt: Custom analysis prompt
            use_cache: Whether to use caching
            timeout: Request timeout in seconds
            progress_callback: Optional callback for progress updates

        Returns:
            AI analysis response
        """

        # Report progress
        if progress_callback:
            progress_callback("Starting analysis...", 0.1)

        # Check cache first
        if use_cache:
            if progress_callback:
                progress_callback("Checking cache...", 0.2)

            cache_key = f"{dataset_info.get('id')}_{analysis_type.value}"
            cached_analysis = self.cache_manager.get_cached_context(cache_key)
            if cached_analysis:
                if progress_callback:
                    progress_callback("Found cached result!", 1.0)
                return AIResponse(
                    content=cached_analysis['content'],
                    provider=cached_analysis['provider'],
                    model=cached_analysis['model'],
                    usage=cached_analysis.get('usage', {}),
                    cached=True,
                    metadata=cached_analysis.get('metadata', {})
                )

        # Prepare analysis request with chunking for large datasets
        if progress_callback:
            progress_callback("Preparing analysis request...", 0.3)

        # Handle large sample data by chunking
        processed_sample_data = sample_data
        if sample_data and len(sample_data) > 50:
            if progress_callback:
                progress_callback("Large dataset detected, using optimized sampling...", 0.4)

            # Use intelligent sampling for large datasets
            processed_sample_data = self._optimize_sample_data(sample_data)

        if custom_prompt:
            request = AIRequest(
                prompt=custom_prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.3,
                max_tokens=2000,
                timeout=timeout
            )
        else:
            request = self._create_analysis_request(dataset_info, processed_sample_data, analysis_type)
            request.timeout = timeout

        # Try to get response from providers with timeout
        if progress_callback:
            progress_callback("Sending request to AI provider...", 0.5)

        response = await self._get_ai_response_with_timeout(request, use_cache=use_cache, timeout=timeout, progress_callback=progress_callback)

        # Cache the response
        if use_cache and response:
            if progress_callback:
                progress_callback("Caching results...", 0.9)

            cache_data = {
                'content': response.content,
                'provider': response.provider,
                'model': response.model,
                'usage': response.usage,
                'metadata': response.metadata
            }
            cache_key = f"{dataset_info.get('id')}_{analysis_type.value}"
            self.cache_manager.cache_context_analysis(cache_key, cache_data)

        if progress_callback:
            progress_callback("Analysis complete!", 1.0)

        return response

    async def answer_question(
        self,
        question: str,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]] = None,
        use_cache: bool = True
    ) -> AIResponse:
        """
        Answer a specific question about the dataset

        Args:
            question: User's question
            dataset_info: Dataset metadata
            sample_data: Sample records
            use_cache: Whether to use caching

        Returns:
            AI response to the question
        """

        # Build context
        context = self._build_dataset_context(dataset_info, sample_data)

        prompt = f"""Based on the following dataset information, please answer this question: "{question}"

{context}

Please provide a comprehensive answer based on the available data and metadata. If the answer cannot be determined from the provided information, please explain what additional data would be needed."""

        request = AIRequest(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.4,
            max_tokens=1500
        )

        return await self._get_ai_response(request, use_cache=use_cache)

    def _create_analysis_request(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]],
        analysis_type: AnalysisType
    ) -> AIRequest:
        """Create analysis request based on type"""

        context = self._build_dataset_context(dataset_info, sample_data)

        prompts = {
            AnalysisType.OVERVIEW: f"""Provide a comprehensive overview of this NYC Open Data dataset.

{context}

Please analyze and provide:

1. **Data Summary**: What this dataset contains and its primary purpose
2. **Key Characteristics**: Important features and structure
3. **Data Quality Indicators**: Completeness and reliability assessment
4. **Potential Applications**: How this data could be used
5. **Notable Insights**: Interesting patterns or findings
6. **Analysis Recommendations**: Suggested next steps

Format your response with clear sections and actionable insights.""",

            AnalysisType.QUALITY: f"""Assess the quality and reliability of this dataset.

{context}

Evaluate:

1. **Completeness**: Missing data patterns
2. **Consistency**: Format and value consistency
3. **Accuracy**: Potential accuracy issues
4. **Timeliness**: Data freshness and update frequency
5. **Usability**: Analysis readiness
6. **Quality Score**: Rate 1-10 with rationale
7. **Improvement Recommendations**: Specific suggestions

Focus on actionable quality insights.""",

            AnalysisType.INSIGHTS: f"""Extract key insights and patterns from this dataset.

{context}

Analyze:

1. **Distribution Patterns**: Key statistical distributions
2. **Trends**: Temporal or categorical trends
3. **Correlations**: Relationships between variables
4. **Outliers**: Notable exceptions or anomalies
5. **Geographic Patterns**: Spatial insights (if applicable)
6. **Predictive Opportunities**: Forecasting potential

Highlight the most valuable and actionable insights.""",

            AnalysisType.RELATIONSHIPS: f"""Analyze relationships within this dataset and with other potential datasets.

{context}

Focus on:

1. **Internal Relationships**: Correlations within the data
2. **Categorical Relationships**: Patterns across categories
3. **Temporal Relationships**: Time-based correlations
4. **External Connection Opportunities**: How this dataset could link to others
5. **Network Analysis Potential**: Relationship mapping opportunities

Identify the strongest and most valuable relationships."""
        }

        prompt = prompts.get(analysis_type, prompts[AnalysisType.OVERVIEW])

        return AIRequest(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.3,
            max_tokens=2000
        )

    def _build_dataset_context(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]] = None
    ) -> str:
        """Build comprehensive dataset context for AI analysis"""

        context = f"""Dataset Information:
- ID: {dataset_info.get('id', 'Unknown')}
- Name: {dataset_info.get('name', 'Unnamed Dataset')}
- Description: {dataset_info.get('description', 'No description available')}
- Category: {dataset_info.get('category', 'Uncategorized')}
- Columns: {dataset_info.get('columns_count', 0)}
- Download Count: {dataset_info.get('download_count', 'Unknown')}
- Last Updated: {dataset_info.get('updated_at', 'Unknown')}
- Tags: {', '.join(dataset_info.get('tags', [])) if dataset_info.get('tags') else 'None'}"""

        if dataset_info.get('quality_score'):
            context += f"\n- Quality Score: {dataset_info['quality_score']}/100"

        if sample_data and len(sample_data) > 0:
            context += f"\n\nSample Data ({len(sample_data)} records):"

            # Show column names from first record
            if sample_data[0]:
                columns = list(sample_data[0].keys())
                context += f"\nColumns: {', '.join(columns)}"

            # Show first few records
            for i, record in enumerate(sample_data[:3]):
                context += f"\nRecord {i+1}: {record}"

            if len(sample_data) > 3:
                context += f"\n... and {len(sample_data) - 3} more records"

        return context

    def _get_system_prompt(self) -> str:
        """Get system prompt for AI analysis"""
        return """You are an expert data analyst specializing in NYC Open Data and urban analytics.

Your role is to:
- Provide clear, actionable insights based on data evidence
- Identify patterns, trends, and anomalies
- Assess data quality and usability
- Suggest practical applications and next steps
- Format responses with clear structure and bullet points
- Focus on value and actionability for data users

Always ground your analysis in the provided data and be transparent about limitations."""

    async def _get_ai_response(self, request: AIRequest, use_cache: bool = True) -> AIResponse:
        """Get AI response with provider fallback"""

        # Try cache first
        if use_cache:
            cached_response = await self.cache_manager.get_cached_response(request)
            if cached_response:
                return cached_response

        # Try primary provider first
        if self.primary_provider in self.providers:
            try:
                response = await self.providers[self.primary_provider].generate_response(request)
                if use_cache:
                    await self.cache_manager.cache_response(request, response)
                return response
            except Exception as e:
                logger.error(f"Primary provider {self.primary_provider} failed: {e}")

        # Try fallback providers
        for provider_name in self.fallback_providers:
            if provider_name in self.providers:
                try:
                    logger.info(f"Trying fallback provider: {provider_name}")
                    response = await self.providers[provider_name].generate_response(request)
                    if use_cache:
                        await self.cache_manager.cache_response(request, response)
                    return response
                except Exception as e:
                    logger.error(f"Fallback provider {provider_name} failed: {e}")

        # If all providers fail
        raise Exception("All AI providers failed to generate response")

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                'available': True,
                'model': provider.default_model,
                'provider_class': provider.__class__.__name__
            }
        return status

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics"""
        return self.cache_manager.get_cache_stats()

    def clear_cache(self, cache_type: str = "all"):
        """Clear specified cache"""
        self.cache_manager.clear_cache(cache_type)

    def _optimize_sample_data(self, sample_data: List[Dict]) -> List[Dict]:
        """Optimize sample data for AI analysis by intelligent sampling"""
        if len(sample_data) <= 50:
            return sample_data

        # Take first 20, last 20, and 10 random from middle
        optimized_sample = []

        # First 20 records
        optimized_sample.extend(sample_data[:20])

        # 10 random from middle
        if len(sample_data) > 40:
            import random
            middle_start = 20
            middle_end = len(sample_data) - 20
            if middle_end > middle_start:
                middle_indices = random.sample(range(middle_start, middle_end), min(10, middle_end - middle_start))
                for idx in sorted(middle_indices):
                    optimized_sample.append(sample_data[idx])

        # Last 20 records
        if len(sample_data) > 20:
            optimized_sample.extend(sample_data[-20:])

        # Remove duplicates while preserving order
        seen = set()
        deduplicated_sample = []
        for item in optimized_sample:
            item_key = str(sorted(item.items()))
            if item_key not in seen:
                seen.add(item_key)
                deduplicated_sample.append(item)

        return deduplicated_sample[:50]  # Ensure we don't exceed 50 items

    async def _get_ai_response_with_timeout(
        self,
        request: AIRequest,
        use_cache: bool = True,
        timeout: int = 60,
        progress_callback: Optional[callable] = None
    ) -> AIResponse:
        """Get AI response with timeout and progress tracking"""
        import asyncio

        # Try cache first
        if use_cache:
            cached_response = await self.cache_manager.get_cached_response(request)
            if cached_response:
                return cached_response

        # Wrapper function for the actual AI call
        async def ai_call_with_progress():
            # Try primary provider first
            if self.primary_provider in self.providers:
                try:
                    if progress_callback:
                        progress_callback(f"Calling {self.primary_provider.upper()}...", 0.6)

                    response = await self.providers[self.primary_provider].generate_response(request)
                    if use_cache:
                        await self.cache_manager.cache_response(request, response)

                    if progress_callback:
                        progress_callback("Response received!", 0.8)

                    return response
                except Exception as e:
                    logger.error(f"Primary provider {self.primary_provider} failed: {e}")
                    if progress_callback:
                        progress_callback(f"Primary provider failed, trying fallback...", 0.7)

            # Try fallback providers
            for provider_name in self.fallback_providers:
                if provider_name in self.providers:
                    try:
                        if progress_callback:
                            progress_callback(f"Trying {provider_name.upper()}...", 0.7)

                        logger.info(f"Trying fallback provider: {provider_name}")
                        response = await self.providers[provider_name].generate_response(request)
                        if use_cache:
                            await self.cache_manager.cache_response(request, response)

                        if progress_callback:
                            progress_callback("Response received from fallback!", 0.8)

                        return response
                    except Exception as e:
                        logger.error(f"Fallback provider {provider_name} failed: {e}")
                        continue

            # If all providers fail
            raise Exception("All AI providers failed to generate response")

        # Execute with timeout
        try:
            response = await asyncio.wait_for(ai_call_with_progress(), timeout=timeout)
            return response
        except asyncio.TimeoutError:
            error_msg = f"AI request timed out after {timeout} seconds"
            logger.error(error_msg)
            raise Exception(error_msg)