"""
AI-Powered Data Explorer

Integrates NVIDIA AI statistician prompts with dataset analysis,
enabling natural language exploration with pandas and plotly code generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import json
from pathlib import Path

from .ai_analyst import DataAnalyst
from .base_provider import AIRequest, AIResponse
from ..prompts.nvidia_statistician import NvidiaStatisticianPrompt, DatasetContext

logger = logging.getLogger(__name__)


class DataExplorer:
    """
    AI-powered data exploration system using NVIDIA's professional statistician persona

    This class enables:
    - Natural language dataset exploration
    - Professional statistical analysis with pandas
    - Interactive plotly visualization generation
    - Dataset joining and integration
    - Statistical hypothesis testing
    - Comprehensive EDA automation
    """

    def __init__(
        self,
        ai_analyst: DataAnalyst,
        prefer_nvidia: bool = True,
        fallback_providers: Optional[List[str]] = None
    ):
        """
        Initialize Data Explorer

        Args:
            ai_analyst: DataAnalyst instance with configured providers
            prefer_nvidia: Whether to prefer NVIDIA for statistical analysis
            fallback_providers: Fallback providers if NVIDIA is unavailable
        """
        self.ai_analyst = ai_analyst
        self.prefer_nvidia = prefer_nvidia
        self.fallback_providers = fallback_providers or ["openai", "openrouter"]

        # Initialize prompt generator
        self.statistician_prompt = NvidiaStatisticianPrompt()

        # Cache for dataset contexts
        self.dataset_cache: Dict[str, DatasetContext] = {}

        logger.info("DataExplorer initialized with professional statistician capabilities")

    async def explore_dataset(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        user_question: str,
        dataset_description: str = "Dataset for analysis",
        available_datasets: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Explore dataset using natural language with AI statistician

        Args:
            dataframe: Primary dataset to analyze
            dataset_name: Name of the dataset
            user_question: Natural language question about the data
            dataset_description: Description of the dataset
            available_datasets: Other datasets available for joining

        Returns:
            Dictionary containing analysis results, code, and insights
        """

        try:
            logger.info(f"Starting dataset exploration: '{user_question}' on {dataset_name}")

            # Create dataset context
            dataset_context = self.statistician_prompt.create_dataset_context_from_dataframe(
                dataframe, dataset_name, dataset_description
            )

            # Create contexts for available datasets
            available_contexts = []
            if available_datasets:
                for name, df in available_datasets.items():
                    ctx = self.statistician_prompt.create_dataset_context_from_dataframe(
                        df, name, f"Available dataset: {name}"
                    )
                    available_contexts.append(ctx)

            # Generate exploration prompt
            prompt = self.statistician_prompt.generate_exploration_prompt(
                dataset_context=dataset_context,
                user_question=user_question,
                available_datasets=available_contexts
            )

            # Get AI analysis
            response = await self._get_statistician_response(prompt, "dataset_exploration")

            # Parse response and extract code
            analysis_result = self._parse_statistician_response(response)

            # Add metadata
            analysis_result.update({
                "dataset_name": dataset_name,
                "user_question": user_question,
                "dataset_shape": dataframe.shape,
                "exploration_type": "natural_language_query",
                "available_datasets": list(available_datasets.keys()) if available_datasets else [],
                "timestamp": pd.Timestamp.now().isoformat()
            })

            logger.info(f"Dataset exploration completed successfully")
            return analysis_result

        except Exception as e:
            logger.error(f"Dataset exploration failed: {e}")
            return {
                "error": str(e),
                "dataset_name": dataset_name,
                "user_question": user_question,
                "timestamp": pd.Timestamp.now().isoformat()
            }

    async def create_visualization(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        chart_request: str,
        dataset_description: str = "Dataset for visualization",
        analysis_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create statistical visualizations using AI

        Args:
            dataframe: Dataset to visualize
            dataset_name: Name of the dataset
            chart_request: Natural language description of desired chart
            dataset_description: Description of the dataset
            analysis_context: Additional context about the analysis

        Returns:
            Dictionary containing visualization code and insights
        """

        try:
            logger.info(f"Creating visualization: '{chart_request}' for {dataset_name}")

            # Create dataset context
            dataset_context = self.statistician_prompt.create_dataset_context_from_dataframe(
                dataframe, dataset_name, dataset_description
            )

            # Generate visualization prompt
            prompt = self.statistician_prompt.generate_visualization_prompt(
                dataset_context=dataset_context,
                chart_request=chart_request,
                analysis_context=analysis_context
            )

            # Get AI analysis
            response = await self._get_statistician_response(prompt, "visualization_creation")

            # Parse response
            analysis_result = self._parse_statistician_response(response)

            # Add metadata
            analysis_result.update({
                "dataset_name": dataset_name,
                "chart_request": chart_request,
                "dataset_shape": dataframe.shape,
                "visualization_type": "ai_generated",
                "timestamp": pd.Timestamp.now().isoformat()
            })

            logger.info("Visualization creation completed successfully")
            return analysis_result

        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return {
                "error": str(e),
                "dataset_name": dataset_name,
                "chart_request": chart_request,
                "timestamp": pd.Timestamp.now().isoformat()
            }

    async def join_datasets(
        self,
        primary_dataframe: pd.DataFrame,
        datasets_to_join: Dict[str, pd.DataFrame],
        join_objective: str,
        analysis_goal: str,
        primary_dataset_name: str = "Primary Dataset"
    ) -> Dict[str, Any]:
        """
        Join datasets intelligently using AI guidance

        Args:
            primary_dataframe: Main dataset to join to
            datasets_to_join: Dictionary of datasets to consider joining
            join_objective: What user wants to achieve with joining
            analysis_goal: Overall analysis objective
            primary_dataset_name: Name of the primary dataset

        Returns:
            Dictionary containing join strategy, code, and results
        """

        try:
            logger.info(f"Planning dataset joins for: '{analysis_goal}'")

            # Create dataset contexts
            primary_context = self.statistician_prompt.create_dataset_context_from_dataframe(
                primary_dataframe, primary_dataset_name, "Primary dataset for joining"
            )

            join_contexts = []
            for name, df in datasets_to_join.items():
                ctx = self.statistician_prompt.create_dataset_context_from_dataframe(
                    df, name, f"Dataset available for joining: {name}"
                )
                join_contexts.append(ctx)

            # Generate joining prompt
            prompt = self.statistician_prompt.generate_dataset_joining_prompt(
                primary_dataset=primary_context,
                datasets_to_join=join_contexts,
                join_objective=join_objective,
                user_analysis_goal=analysis_goal
            )

            # Get AI analysis
            response = await self._get_statistician_response(prompt, "dataset_joining")

            # Parse response
            analysis_result = self._parse_statistician_response(response)

            # Add metadata
            analysis_result.update({
                "primary_dataset": primary_dataset_name,
                "datasets_to_join": list(datasets_to_join.keys()),
                "join_objective": join_objective,
                "analysis_goal": analysis_goal,
                "joining_type": "ai_guided",
                "timestamp": pd.Timestamp.now().isoformat()
            })

            logger.info("Dataset joining analysis completed successfully")
            return analysis_result

        except Exception as e:
            logger.error(f"Dataset joining failed: {e}")
            return {
                "error": str(e),
                "primary_dataset": primary_dataset_name,
                "join_objective": join_objective,
                "timestamp": pd.Timestamp.now().isoformat()
            }

    async def perform_statistical_test(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        hypothesis: str,
        dataset_description: str = "Dataset for statistical testing",
        test_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform statistical hypothesis testing with AI guidance

        Args:
            dataframe: Dataset for testing
            dataset_name: Name of the dataset
            hypothesis: Hypothesis to test in natural language
            dataset_description: Description of the dataset
            test_type: Specific statistical test to use (optional)

        Returns:
            Dictionary containing test results, code, and interpretation
        """

        try:
            logger.info(f"Performing statistical test: '{hypothesis}' on {dataset_name}")

            # Create dataset context
            dataset_context = self.statistician_prompt.create_dataset_context_from_dataframe(
                dataframe, dataset_name, dataset_description
            )

            # Generate testing prompt
            prompt = self.statistician_prompt.generate_statistical_testing_prompt(
                dataset_context=dataset_context,
                hypothesis=hypothesis,
                test_type=test_type
            )

            # Get AI analysis
            response = await self._get_statistician_response(prompt, "statistical_testing")

            # Parse response
            analysis_result = self._parse_statistician_response(response)

            # Add metadata
            analysis_result.update({
                "dataset_name": dataset_name,
                "hypothesis": hypothesis,
                "test_type": test_type,
                "dataset_shape": dataframe.shape,
                "testing_type": "statistical_hypothesis",
                "timestamp": pd.Timestamp.now().isoformat()
            })

            logger.info("Statistical testing completed successfully")
            return analysis_result

        except Exception as e:
            logger.error(f"Statistical testing failed: {e}")
            return {
                "error": str(e),
                "dataset_name": dataset_name,
                "hypothesis": hypothesis,
                "timestamp": pd.Timestamp.now().isoformat()
            }

    async def comprehensive_eda(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        dataset_description: str = "Dataset for comprehensive analysis",
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive Exploratory Data Analysis (EDA)

        Args:
            dataframe: Dataset to analyze
            dataset_name: Name of the dataset
            dataset_description: Description of the dataset
            focus_areas: Specific areas to focus on

        Returns:
            Dictionary containing comprehensive EDA results
        """

        try:
            logger.info(f"Starting comprehensive EDA for {dataset_name}")

            # Create dataset context
            dataset_context = self.statistician_prompt.create_dataset_context_from_dataframe(
                dataframe, dataset_name, dataset_description
            )

            # Generate EDA prompt
            prompt = self.statistician_prompt.generate_comprehensive_eda_prompt(
                dataset_context=dataset_context,
                focus_areas=focus_areas
            )

            # Get AI analysis
            response = await self._get_statistician_response(prompt, "comprehensive_eda")

            # Parse response
            analysis_result = self._parse_statistician_response(response)

            # Add metadata
            analysis_result.update({
                "dataset_name": dataset_name,
                "dataset_shape": dataframe.shape,
                "focus_areas": focus_areas or ["comprehensive"],
                "eda_type": "comprehensive_professional",
                "timestamp": pd.Timestamp.now().isoformat()
            })

            logger.info("Comprehensive EDA completed successfully")
            return analysis_result

        except Exception as e:
            logger.error(f"Comprehensive EDA failed: {e}")
            return {
                "error": str(e),
                "dataset_name": dataset_name,
                "timestamp": pd.Timestamp.now().isoformat()
            }

    async def _get_statistician_response(self, prompt: str, analysis_type: str) -> AIResponse:
        """
        Get response from AI analyst using NVIDIA statistician configuration

        Args:
            prompt: Generated prompt for the statistician
            analysis_type: Type of analysis being performed

        Returns:
            AI response from the statistician
        """

        # Create AI request with statistician system prompt
        request = AIRequest(
            prompt=prompt,
            system_prompt=self.statistician_prompt.system_prompt,
            temperature=0.2,  # Low temperature for consistent statistical analysis
            max_tokens=4000,   # Longer responses for detailed analysis
            model=None  # Use default model for the provider
        )

        # Prefer NVIDIA for statistical analysis if available
        if self.prefer_nvidia and "nvidia" in self.ai_analyst.providers:
            try:
                response = await self.ai_analyst.providers["nvidia"].generate_response(request)
                return response
            except Exception as e:
                logger.warning(f"NVIDIA provider failed, trying fallback: {e}")

        # Try fallback providers
        for provider_name in self.fallback_providers:
            if provider_name in self.ai_analyst.providers:
                try:
                    logger.info(f"Using fallback provider: {provider_name}")
                    response = await self.ai_analyst.providers[provider_name].generate_response(request)
                    return response
                except Exception as e:
                    logger.warning(f"Fallback provider {provider_name} failed: {e}")
                    continue

        raise Exception("All AI providers failed for statistical analysis")

    def _parse_statistician_response(self, response: AIResponse) -> Dict[str, Any]:
        """
        Parse statistician response and extract structured information

        Args:
            response: AI response from statistician

        Returns:
            Structured analysis results
        """

        content = response.content

        # Try to extract different sections
        sections = {
            "executive_summary": self._extract_section(content, ["Executive Summary", "Summary", "Key Findings"]),
            "methodology": self._extract_section(content, ["Statistical Analysis", "Methodology", "Approach"]),
            "python_code": self._extract_code_blocks(content),
            "insights": self._extract_section(content, ["Insights", "Recommendations", "Findings"]),
            "follow_ups": self._extract_section(content, ["Follow-ups", "Next Steps", "Additional Analysis"])
        }

        # Clean up sections
        for key, value in sections.items():
            if value:
                sections[key] = value.strip()

        return {
            "raw_response": content,
            "sections": sections,
            "provider": response.provider,
            "model": response.model,
            "usage": response.usage,
            "metadata": response.metadata,
            "analysis_complete": True
        }

    def _extract_section(self, content: str, section_headers: List[str]) -> Optional[str]:
        """Extract a section from the response content"""

        content_lower = content.lower()

        for header in section_headers:
            header_lower = header.lower()

            # Look for the header
            start_idx = content_lower.find(header_lower)
            if start_idx != -1:
                # Find the end of this section (next header or end of content)
                remaining_content = content[start_idx:]

                # Look for next section markers
                next_section_markers = [
                    "**", "##", "###", "----", "====",
                    "executive summary", "methodology", "python code",
                    "insights", "recommendations", "follow-ups"
                ]

                end_idx = len(remaining_content)
                for marker in next_section_markers:
                    marker_idx = remaining_content.lower().find(marker, len(header_lower) + 10)  # Skip the current header
                    if marker_idx != -1 and marker_idx < end_idx:
                        end_idx = marker_idx

                section_content = remaining_content[:end_idx]

                # Clean up the section
                # Remove the header line
                lines = section_content.split('\n')
                if len(lines) > 1:
                    section_content = '\n'.join(lines[1:])

                return section_content.strip()

        return None

    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract Python code blocks from the response"""

        code_blocks = []

        # Look for ```python or ``` code blocks
        import re

        # Pattern for ```python ... ``` blocks
        python_pattern = r'```python\s*\n(.*?)\n```'
        python_matches = re.findall(python_pattern, content, re.DOTALL | re.IGNORECASE)
        code_blocks.extend(python_matches)

        # Pattern for generic ``` blocks that might contain Python
        generic_pattern = r'```\s*\n(.*?)\n```'
        generic_matches = re.findall(generic_pattern, content, re.DOTALL)

        for match in generic_matches:
            # Check if it looks like Python code
            if any(keyword in match.lower() for keyword in ['import', 'pandas', 'plotly', 'df.', 'plt.']):
                if match not in code_blocks:  # Avoid duplicates
                    code_blocks.append(match)

        return [code.strip() for code in code_blocks if code.strip()]

    def get_exploration_capabilities(self) -> Dict[str, Any]:
        """Get information about data exploration capabilities"""

        return {
            "exploration_types": [
                "natural_language_queries",
                "statistical_visualization",
                "dataset_joining",
                "hypothesis_testing",
                "comprehensive_eda"
            ],
            "supported_analysis": [
                "descriptive_statistics",
                "correlation_analysis",
                "distribution_analysis",
                "outlier_detection",
                "time_series_analysis",
                "categorical_analysis",
                "multivariate_analysis"
            ],
            "visualization_types": [
                "interactive_plotly_charts",
                "statistical_plots",
                "distribution_plots",
                "correlation_heatmaps",
                "time_series_plots",
                "dashboard_components"
            ],
            "joining_capabilities": [
                "intelligent_key_detection",
                "join_type_recommendation",
                "data_quality_validation",
                "referential_integrity_checks"
            ],
            "statistical_tests": [
                "hypothesis_testing",
                "significance_testing",
                "confidence_intervals",
                "effect_size_calculation",
                "power_analysis"
            ],
            "provider_info": {
                "preferred_provider": "nvidia",
                "fallback_providers": self.fallback_providers,
                "statistical_expertise": "professional_phd_level"
            }
        }