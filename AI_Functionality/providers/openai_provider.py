"""
OpenAI Provider Implementation

Supports GPT-4, GPT-3.5, and other OpenAI models with advanced features.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import tiktoken

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from ..core.base_provider import BaseAIProvider, AIRequest, AIResponse
except ImportError:
    from core.base_provider import BaseAIProvider, AIRequest, AIResponse


logger = logging.getLogger(__name__)


class OpenAIProvider(BaseAIProvider):
    """OpenAI API provider with GPT models"""

    MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ]

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str, model: str = None, **kwargs):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")

        super().__init__(
            api_key=api_key,
            default_model=model or self.DEFAULT_MODEL,
            **kwargs
        )
        self._initialize_client()

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def _initialize_client(self):
        """Initialize OpenAI async client"""
        try:
            self._client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client with model: {self.default_model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate response using OpenAI API"""
        if not self.validate_request(request):
            raise ValueError("Invalid request parameters")

        model = request.model or self.default_model

        # Prepare messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        try:
            # Make API call
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )

            # Extract response data
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            return AIResponse(
                content=content,
                provider=self.provider_name,
                model=model,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken"""
        try:
            # Get encoding for the model
            if "gpt-4" in self.default_model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.default_model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding

            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Token estimation error: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

    def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        return self.MODELS.copy()

    async def generate_data_analysis(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]] = None,
        analysis_type: str = "overview"
    ) -> AIResponse:
        """Generate specialized data analysis"""

        # Prepare context
        context = f"""
Dataset Information:
- Name: {dataset_info.get('name', 'Unknown')}
- Description: {dataset_info.get('description', 'No description')}
- Columns: {dataset_info.get('columns_count', 0)}
- Records: {dataset_info.get('download_count', 'Unknown')}
- Category: {dataset_info.get('category', 'Uncategorized')}
- Last Updated: {dataset_info.get('updated_at', 'Unknown')}
"""

        if sample_data and len(sample_data) > 0:
            context += f"\nSample Data Preview:\n"
            # Include first few records
            for i, record in enumerate(sample_data[:3]):
                context += f"Record {i+1}: {record}\n"

            if len(sample_data) > 3:
                context += f"... and {len(sample_data) - 3} more records\n"

        # Choose analysis prompt based on type
        if analysis_type == "overview":
            prompt = f"""As a data analyst, provide a comprehensive overview of this NYC Open Data dataset.

{context}

Please analyze and provide:

1. **Data Summary**: What this dataset contains and its primary purpose
2. **Key Insights**: Notable patterns, trends, or interesting findings
3. **Data Quality Assessment**: Completeness, consistency, and reliability indicators
4. **Potential Use Cases**: How this data could be valuable for analysis or decision-making
5. **Analysis Recommendations**: Suggested analytical approaches or questions to explore

Format your response in clear sections with actionable insights."""

        elif analysis_type == "quality":
            prompt = f"""As a data quality expert, assess this dataset's quality and reliability.

{context}

Evaluate:

1. **Completeness**: Missing data patterns and gaps
2. **Consistency**: Data format and value consistency
3. **Accuracy**: Potential accuracy issues or anomalies
4. **Timeliness**: Update frequency and data freshness
5. **Usability**: How user-friendly and analysis-ready this data is
6. **Recommendations**: Specific steps to improve data quality

Provide a quality score (1-10) and detailed rationale."""

        elif analysis_type == "insights":
            prompt = f"""As a data scientist, extract key insights and patterns from this dataset.

{context}

Focus on:

1. **Statistical Patterns**: Key distributions, outliers, correlations
2. **Temporal Trends**: Changes over time (if applicable)
3. **Categorical Analysis**: Distribution across categories/groups
4. **Geographic Patterns**: Spatial distribution insights (if applicable)
5. **Anomalies**: Unusual patterns or outliers worth investigating
6. **Predictive Potential**: What could be predicted or forecasted

Highlight the most actionable insights."""

        else:
            prompt = f"""Analyze this NYC Open Data dataset and provide insights based on the user's focus: {analysis_type}

{context}

Provide a thorough analysis tailored to the specified focus area."""

        request = AIRequest(
            prompt=prompt,
            system_prompt="You are an expert data analyst specializing in public datasets and urban data analysis. Provide clear, actionable insights backed by evidence from the data.",
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=2000
        )

        return await self.generate_response(request)