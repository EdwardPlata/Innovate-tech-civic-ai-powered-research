"""
Insights Engine - Automated AI-powered insights generation

Provides periodic analysis, trend detection, and automated insight generation
for datasets and user patterns.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from pathlib import Path

from .base_provider import BaseAIProvider, AIRequest, AIResponse
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of insights that can be generated"""
    TREND_ANALYSIS = "trend_analysis"
    USAGE_PATTERNS = "usage_patterns"
    DATA_QUALITY_SHIFTS = "data_quality_shifts"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE = "predictive"
    COMPARATIVE = "comparative"


class InsightPriority(Enum):
    """Priority levels for insights"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Insight:
    """Represents an AI-generated insight"""
    id: str
    type: InsightType
    priority: InsightPriority
    title: str
    description: str
    content: str
    evidence: List[str]
    recommendations: List[str]
    confidence_score: float
    timestamp: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []


class InsightsEngine:
    """AI-powered insights generation and management system"""

    def __init__(
        self,
        ai_analyst,
        cache_dir: str = "./insights_cache",
        insights_storage_dir: str = "./insights_storage"
    ):
        """
        Initialize Insights Engine

        Args:
            ai_analyst: DataAnalyst instance for AI operations
            cache_dir: Directory for caching insights
            insights_storage_dir: Directory for storing generated insights
        """
        self.ai_analyst = ai_analyst
        self.cache_dir = Path(cache_dir)
        self.storage_dir = Path(insights_storage_dir)

        # Create directories
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.storage_dir.mkdir(exist_ok=True, parents=True)

        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir=str(self.cache_dir),
            enable_semantic=True
        )

        # Insights storage
        self.insights: List[Insight] = []
        self.insight_history: Dict[str, List[Insight]] = {}

        # Load existing insights
        self._load_insights()

        logger.info("InsightsEngine initialized")

    async def generate_dataset_insights(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]] = None,
        analysis_history: Optional[List[Dict]] = None
    ) -> List[Insight]:
        """
        Generate comprehensive insights for a dataset

        Args:
            dataset_info: Dataset metadata
            sample_data: Sample records from dataset
            analysis_history: Previous analysis results for trend detection

        Returns:
            List of generated insights
        """
        logger.info(f"Generating insights for dataset: {dataset_info.get('id')}")

        insights = []

        try:
            # Generate different types of insights
            insight_generators = [
                self._generate_quality_insights,
                self._generate_usage_insights,
                self._generate_trend_insights,
                self._generate_predictive_insights,
                self._generate_recommendations
            ]

            for generator in insight_generators:
                try:
                    new_insights = await generator(dataset_info, sample_data, analysis_history)
                    insights.extend(new_insights)
                except Exception as e:
                    logger.warning(f"Insight generator {generator.__name__} failed: {e}")

            # Store insights
            for insight in insights:
                self._store_insight(insight)

            logger.info(f"Generated {len(insights)} insights for dataset {dataset_info.get('id')}")
            return insights

        except Exception as e:
            logger.error(f"Failed to generate dataset insights: {e}")
            return []

    async def generate_platform_insights(
        self,
        usage_statistics: Dict[str, Any],
        user_patterns: Dict[str, Any],
        system_health: Dict[str, Any]
    ) -> List[Insight]:
        """
        Generate platform-wide insights about usage, performance, and trends

        Args:
            usage_statistics: Platform usage data
            user_patterns: User behavior patterns
            system_health: System performance metrics

        Returns:
            List of platform insights
        """
        logger.info("Generating platform-wide insights")

        insights = []

        try:
            # Platform usage insights
            usage_insights = await self._analyze_platform_usage(usage_statistics, user_patterns)
            insights.extend(usage_insights)

            # Performance insights
            performance_insights = await self._analyze_system_performance(system_health)
            insights.extend(performance_insights)

            # User behavior insights
            behavior_insights = await self._analyze_user_behavior(user_patterns)
            insights.extend(behavior_insights)

            # Store insights
            for insight in insights:
                self._store_insight(insight)

            logger.info(f"Generated {len(insights)} platform insights")
            return insights

        except Exception as e:
            logger.error(f"Failed to generate platform insights: {e}")
            return []

    async def _generate_quality_insights(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]],
        analysis_history: Optional[List[Dict]]
    ) -> List[Insight]:
        """Generate data quality insights"""

        insights = []

        # Build context for quality analysis
        context = self._build_dataset_context(dataset_info, sample_data)

        # Historical quality context
        quality_history = ""
        if analysis_history:
            quality_scores = [h.get('quality_score') for h in analysis_history if h.get('quality_score')]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                quality_history = f"Historical average quality score: {avg_quality:.1f}/100"

        prompt = f"""
        Analyze the data quality of this dataset and provide specific insights:

        {context}
        {quality_history}

        Focus on:
        1. **Quality Assessment**: Current quality score and key factors
        2. **Quality Trends**: How quality has changed over time (if historical data available)
        3. **Critical Issues**: Most important quality problems to address
        4. **Quick Wins**: Easy improvements that would have high impact
        5. **Quality Risks**: Potential quality degradation risks

        Provide specific, actionable insights about data quality.
        """

        try:
            response = await self._get_ai_response(prompt, "quality_insights")

            # Parse response and create insight
            insight = Insight(
                id=self._generate_insight_id("quality", dataset_info.get('id')),
                type=InsightType.DATA_QUALITY_SHIFTS,
                priority=InsightPriority.HIGH,
                title=f"Data Quality Analysis: {dataset_info.get('name', 'Dataset')}",
                description="AI-powered analysis of data quality trends and recommendations",
                content=response.content,
                evidence=[],
                recommendations=self._extract_recommendations(response.content),
                confidence_score=0.85,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=7),
                metadata={
                    "dataset_id": dataset_info.get('id'),
                    "analysis_type": "quality",
                    "provider": response.provider if hasattr(response, 'provider') else "unknown"
                },
                tags=["quality", "data-assessment", "recommendations"]
            )

            insights.append(insight)

        except Exception as e:
            logger.error(f"Failed to generate quality insights: {e}")

        return insights

    async def _generate_usage_insights(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]],
        analysis_history: Optional[List[Dict]]
    ) -> List[Insight]:
        """Generate usage pattern insights"""

        insights = []

        download_count = dataset_info.get('download_count', 0)
        updated_at = dataset_info.get('updated_at')

        # Determine usage priority
        if download_count > 10000:
            priority = InsightPriority.HIGH
            usage_level = "high"
        elif download_count > 1000:
            priority = InsightPriority.MEDIUM
            usage_level = "moderate"
        else:
            priority = InsightPriority.LOW
            usage_level = "low"

        prompt = f"""
        Analyze the usage patterns and popularity of this dataset:

        Dataset: {dataset_info.get('name')}
        Downloads: {download_count:,}
        Category: {dataset_info.get('category', 'Unknown')}
        Last Updated: {updated_at}
        Usage Level: {usage_level}

        Provide insights on:
        1. **Usage Significance**: What the download count indicates
        2. **User Interest**: Why this dataset might be popular/unpopular
        3. **Usage Trends**: Expected usage patterns
        4. **Value Proposition**: Why users choose this dataset
        5. **Growth Potential**: Opportunities to increase usage

        Focus on actionable insights about dataset adoption and value.
        """

        try:
            response = await self._get_ai_response(prompt, "usage_insights")

            insight = Insight(
                id=self._generate_insight_id("usage", dataset_info.get('id')),
                type=InsightType.USAGE_PATTERNS,
                priority=priority,
                title=f"Usage Analysis: {usage_level.title()} Adoption Dataset",
                description=f"Analysis of download patterns and user adoption ({download_count:,} downloads)",
                content=response.content,
                evidence=[f"Download count: {download_count:,}", f"Category: {dataset_info.get('category')}"],
                recommendations=self._extract_recommendations(response.content),
                confidence_score=0.90,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=14),
                metadata={
                    "dataset_id": dataset_info.get('id'),
                    "download_count": download_count,
                    "usage_level": usage_level
                },
                tags=["usage", "adoption", "popularity"]
            )

            insights.append(insight)

        except Exception as e:
            logger.error(f"Failed to generate usage insights: {e}")

        return insights

    async def _generate_trend_insights(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]],
        analysis_history: Optional[List[Dict]]
    ) -> List[Insight]:
        """Generate trend analysis insights"""

        insights = []

        if not analysis_history or len(analysis_history) < 2:
            # Not enough history for trend analysis
            return insights

        # Analyze trends in the data
        trend_context = self._build_trend_context(analysis_history)

        prompt = f"""
        Analyze trends and changes in this dataset over time:

        Dataset: {dataset_info.get('name')}
        {trend_context}

        Identify:
        1. **Data Trends**: Changes in data patterns over time
        2. **Quality Trends**: Quality improvements or degradation
        3. **Usage Trends**: Changes in popularity and adoption
        4. **Content Evolution**: How the dataset content has evolved
        5. **Future Projections**: Expected future trends

        Provide insights about how this dataset is evolving and what to expect.
        """

        try:
            response = await self._get_ai_response(prompt, "trend_insights")

            insight = Insight(
                id=self._generate_insight_id("trends", dataset_info.get('id')),
                type=InsightType.TREND_ANALYSIS,
                priority=InsightPriority.MEDIUM,
                title=f"Trend Analysis: {dataset_info.get('name', 'Dataset')} Evolution",
                description="Analysis of historical changes and future projections",
                content=response.content,
                evidence=[f"Historical analyses: {len(analysis_history)}"],
                recommendations=self._extract_recommendations(response.content),
                confidence_score=0.75,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=30),
                metadata={
                    "dataset_id": dataset_info.get('id'),
                    "history_length": len(analysis_history),
                    "analysis_type": "trends"
                },
                tags=["trends", "evolution", "forecasting"]
            )

            insights.append(insight)

        except Exception as e:
            logger.error(f"Failed to generate trend insights: {e}")

        return insights

    async def _generate_predictive_insights(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]],
        analysis_history: Optional[List[Dict]]
    ) -> List[Insight]:
        """Generate predictive insights"""

        insights = []

        # Only generate predictive insights for datasets with temporal data
        if not sample_data:
            return insights

        # Check if dataset has time-based columns
        time_columns = self._identify_time_columns(sample_data)

        if not time_columns:
            return insights

        context = self._build_dataset_context(dataset_info, sample_data)

        prompt = f"""
        Analyze this dataset for predictive opportunities and forecasting potential:

        {context}

        Time-based columns identified: {', '.join(time_columns)}

        Analyze:
        1. **Predictive Potential**: What can be forecasted from this data
        2. **Time Series Patterns**: Seasonal, cyclical, or trend patterns
        3. **Leading Indicators**: Variables that predict future outcomes
        4. **Forecasting Opportunities**: Specific predictions that would be valuable
        5. **Model Recommendations**: Suggested approaches for prediction

        Focus on practical predictive applications and business value.
        """

        try:
            response = await self._get_ai_response(prompt, "predictive_insights")

            insight = Insight(
                id=self._generate_insight_id("predictive", dataset_info.get('id')),
                type=InsightType.PREDICTIVE,
                priority=InsightPriority.MEDIUM,
                title=f"Predictive Opportunities: {dataset_info.get('name', 'Dataset')}",
                description="AI analysis of forecasting potential and time series patterns",
                content=response.content,
                evidence=[f"Time columns: {', '.join(time_columns)}"],
                recommendations=self._extract_recommendations(response.content),
                confidence_score=0.70,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=21),
                metadata={
                    "dataset_id": dataset_info.get('id'),
                    "time_columns": time_columns,
                    "predictive_potential": True
                },
                tags=["predictive", "forecasting", "time-series"]
            )

            insights.append(insight)

        except Exception as e:
            logger.error(f"Failed to generate predictive insights: {e}")

        return insights

    async def _generate_recommendations(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]],
        analysis_history: Optional[List[Dict]]
    ) -> List[Insight]:
        """Generate actionable recommendations"""

        insights = []

        context = self._build_dataset_context(dataset_info, sample_data)

        prompt = f"""
        Provide actionable recommendations for this dataset:

        {context}

        Generate specific recommendations for:
        1. **Data Users**: How to best utilize this dataset
        2. **Data Stewards**: How to improve dataset quality and value
        3. **Platform Managers**: How to better promote and organize this data
        4. **Analysts**: Best practices for analysis and interpretation
        5. **Integration**: How this dataset connects with others

        Focus on specific, actionable advice that creates real value.
        """

        try:
            response = await self._get_ai_response(prompt, "recommendations")

            insight = Insight(
                id=self._generate_insight_id("recommendations", dataset_info.get('id')),
                type=InsightType.RECOMMENDATION,
                priority=InsightPriority.MEDIUM,
                title=f"Action Plan: {dataset_info.get('name', 'Dataset')} Optimization",
                description="AI-generated recommendations for maximizing dataset value",
                content=response.content,
                evidence=[],
                recommendations=self._extract_recommendations(response.content),
                confidence_score=0.80,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=60),
                metadata={
                    "dataset_id": dataset_info.get('id'),
                    "recommendation_scope": "comprehensive"
                },
                tags=["recommendations", "optimization", "best-practices"]
            )

            insights.append(insight)

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")

        return insights

    async def _analyze_platform_usage(
        self,
        usage_statistics: Dict[str, Any],
        user_patterns: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze platform-wide usage patterns"""

        insights = []

        prompt = f"""
        Analyze platform usage patterns and provide insights:

        Usage Statistics:
        {json.dumps(usage_statistics, indent=2)}

        User Patterns:
        {json.dumps(user_patterns, indent=2)}

        Provide insights on:
        1. **Usage Trends**: How platform usage is changing
        2. **Popular Features**: Most and least used features
        3. **User Behavior**: Patterns in how users interact with the platform
        4. **Growth Opportunities**: Areas for platform improvement
        5. **Resource Optimization**: Efficiency improvements

        Focus on actionable insights for platform optimization.
        """

        try:
            response = await self._get_ai_response(prompt, "platform_usage")

            insight = Insight(
                id=self._generate_insight_id("platform", "usage"),
                type=InsightType.USAGE_PATTERNS,
                priority=InsightPriority.HIGH,
                title="Platform Usage Analysis & Optimization",
                description="AI analysis of platform usage patterns and optimization opportunities",
                content=response.content,
                evidence=[],
                recommendations=self._extract_recommendations(response.content),
                confidence_score=0.85,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=7),
                metadata={
                    "analysis_scope": "platform_wide",
                    "data_points": len(usage_statistics) + len(user_patterns)
                },
                tags=["platform", "usage", "optimization"]
            )

            insights.append(insight)

        except Exception as e:
            logger.error(f"Failed to analyze platform usage: {e}")

        return insights

    async def _analyze_system_performance(self, system_health: Dict[str, Any]) -> List[Insight]:
        """Analyze system performance and health"""

        insights = []

        # Determine priority based on system health
        critical_issues = system_health.get('critical_issues', 0)
        if critical_issues > 0:
            priority = InsightPriority.CRITICAL
        elif system_health.get('warnings', 0) > 3:
            priority = InsightPriority.HIGH
        else:
            priority = InsightPriority.MEDIUM

        prompt = f"""
        Analyze system performance and health metrics:

        System Health Data:
        {json.dumps(system_health, indent=2)}

        Provide insights on:
        1. **Performance Status**: Overall system health assessment
        2. **Critical Issues**: Urgent problems requiring attention
        3. **Performance Trends**: How performance is changing over time
        4. **Bottlenecks**: System constraints and limitations
        5. **Optimization Opportunities**: Performance improvements

        Focus on actionable performance and reliability insights.
        """

        try:
            response = await self._get_ai_response(prompt, "system_performance")

            insight = Insight(
                id=self._generate_insight_id("system", "performance"),
                type=InsightType.ANOMALY_DETECTION,
                priority=priority,
                title="System Performance & Health Analysis",
                description="AI monitoring of system performance metrics and health indicators",
                content=response.content,
                evidence=[f"Critical issues: {critical_issues}"],
                recommendations=self._extract_recommendations(response.content),
                confidence_score=0.90,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=6),
                metadata={
                    "critical_issues": critical_issues,
                    "system_status": system_health.get('status', 'unknown')
                },
                tags=["performance", "monitoring", "system-health"]
            )

            insights.append(insight)

        except Exception as e:
            logger.error(f"Failed to analyze system performance: {e}")

        return insights

    async def _analyze_user_behavior(self, user_patterns: Dict[str, Any]) -> List[Insight]:
        """Analyze user behavior patterns"""

        insights = []

        prompt = f"""
        Analyze user behavior patterns and provide behavioral insights:

        User Behavior Data:
        {json.dumps(user_patterns, indent=2)}

        Analyze:
        1. **User Segmentation**: Different types of users and their behaviors
        2. **Engagement Patterns**: How users interact with the platform
        3. **Feature Adoption**: Which features are popular and why
        4. **User Journey**: Common paths users take through the platform
        5. **Improvement Opportunities**: Ways to enhance user experience

        Focus on insights that improve user satisfaction and engagement.
        """

        try:
            response = await self._get_ai_response(prompt, "user_behavior")

            insight = Insight(
                id=self._generate_insight_id("user", "behavior"),
                type=InsightType.USAGE_PATTERNS,
                priority=InsightPriority.MEDIUM,
                title="User Behavior Analysis & Experience Optimization",
                description="AI analysis of user interaction patterns and experience improvement opportunities",
                content=response.content,
                evidence=[],
                recommendations=self._extract_recommendations(response.content),
                confidence_score=0.80,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=14),
                metadata={
                    "user_segments": user_patterns.get('segments', 0),
                    "analysis_scope": "user_experience"
                },
                tags=["user-behavior", "engagement", "experience"]
            )

            insights.append(insight)

        except Exception as e:
            logger.error(f"Failed to analyze user behavior: {e}")

        return insights

    def get_insights(
        self,
        insight_type: Optional[InsightType] = None,
        priority: Optional[InsightPriority] = None,
        dataset_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Insight]:
        """
        Retrieve insights with filtering

        Args:
            insight_type: Filter by insight type
            priority: Filter by priority level
            dataset_id: Filter by dataset ID
            limit: Maximum number of insights to return

        Returns:
            Filtered list of insights
        """
        filtered_insights = self.insights

        # Apply filters
        if insight_type:
            filtered_insights = [i for i in filtered_insights if i.type == insight_type]

        if priority:
            filtered_insights = [i for i in filtered_insights if i.priority == priority]

        if dataset_id:
            filtered_insights = [i for i in filtered_insights if i.metadata.get('dataset_id') == dataset_id]

        # Remove expired insights
        current_time = datetime.now()
        filtered_insights = [
            i for i in filtered_insights
            if i.expires_at is None or i.expires_at > current_time
        ]

        # Sort by priority and timestamp
        priority_order = {
            InsightPriority.CRITICAL: 4,
            InsightPriority.HIGH: 3,
            InsightPriority.MEDIUM: 2,
            InsightPriority.LOW: 1
        }

        filtered_insights.sort(
            key=lambda x: (priority_order[x.priority], x.timestamp),
            reverse=True
        )

        return filtered_insights[:limit]

    def get_insight_summary(self) -> Dict[str, Any]:
        """Get summary of all insights"""
        current_time = datetime.now()
        active_insights = [
            i for i in self.insights
            if i.expires_at is None or i.expires_at > current_time
        ]

        summary = {
            "total_insights": len(active_insights),
            "by_type": {},
            "by_priority": {},
            "critical_count": 0,
            "recent_count": 0
        }

        # Count by type and priority
        for insight in active_insights:
            insight_type = insight.type.value
            priority = insight.priority.value

            summary["by_type"][insight_type] = summary["by_type"].get(insight_type, 0) + 1
            summary["by_priority"][priority] = summary["by_priority"].get(priority, 0) + 1

            if insight.priority == InsightPriority.CRITICAL:
                summary["critical_count"] += 1

            # Recent insights (last 24 hours)
            if insight.timestamp > (current_time - timedelta(days=1)):
                summary["recent_count"] += 1

        return summary

    def _build_dataset_context(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]]
    ) -> str:
        """Build context string for dataset analysis"""

        context = f"""
        Dataset Information:
        - Name: {dataset_info.get('name', 'Unknown')}
        - ID: {dataset_info.get('id', 'Unknown')}
        - Description: {dataset_info.get('description', 'No description')}
        - Category: {dataset_info.get('category', 'Unknown')}
        - Download Count: {dataset_info.get('download_count', 0):,}
        - Last Updated: {dataset_info.get('updated_at', 'Unknown')}
        - Columns: {dataset_info.get('columns_count', 0)}
        """

        if sample_data:
            context += f"\n\nSample Data ({len(sample_data)} records):"
            for i, record in enumerate(sample_data[:3]):
                context += f"\nRecord {i+1}: {record}"

        return context

    def _build_trend_context(self, analysis_history: List[Dict]) -> str:
        """Build context for trend analysis"""

        context = f"Analysis History ({len(analysis_history)} entries):\n"

        for i, analysis in enumerate(analysis_history[-5:]):  # Last 5 analyses
            timestamp = analysis.get('timestamp', 'Unknown')
            quality_score = analysis.get('quality_score', 'N/A')
            analysis_type = analysis.get('analysis_type', 'Unknown')

            context += f"- {timestamp}: {analysis_type} (Quality: {quality_score})\n"

        return context

    def _identify_time_columns(self, sample_data: List[Dict]) -> List[str]:
        """Identify columns that contain time/date data"""

        time_columns = []

        if not sample_data:
            return time_columns

        # Check first record for time-like column names and values
        first_record = sample_data[0]

        for column, value in first_record.items():
            column_lower = column.lower()

            # Check column name patterns
            time_patterns = ['date', 'time', 'timestamp', 'created', 'updated', 'year', 'month']
            if any(pattern in column_lower for pattern in time_patterns):
                time_columns.append(column)
                continue

            # Check value patterns (simple heuristic)
            if isinstance(value, str):
                # Look for date-like patterns
                if len(value) >= 8 and any(char in value for char in ['-', '/', ':']):
                    try:
                        # Try to parse as date
                        from datetime import datetime
                        datetime.fromisoformat(value.replace('Z', '+00:00').replace('T', ' '))
                        time_columns.append(column)
                    except:
                        pass

        return time_columns

    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract actionable recommendations from AI response"""

        recommendations = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()

            # Look for recommendation patterns
            if any(pattern in line.lower() for pattern in ['recommend', 'should', 'consider', 'suggest']):
                # Clean up the line
                clean_line = line.lstrip('- •*0123456789. ')
                if len(clean_line) > 10:  # Filter out very short lines
                    recommendations.append(clean_line)

        return recommendations[:5]  # Return top 5 recommendations

    async def _get_ai_response(self, prompt: str, analysis_type: str) -> AIResponse:
        """Get AI response for insight generation"""

        # Create a mock dataset info for AI analysis
        dataset_info = {
            "id": f"insights_{analysis_type}",
            "name": f"Insights Analysis - {analysis_type.title()}",
            "description": f"AI-powered insights generation for {analysis_type}",
            "category": "Insights"
        }

        response = await self.ai_analyst.answer_question(
            question=prompt,
            dataset_info=dataset_info,
            use_cache=True
        )

        return response

    def _generate_insight_id(self, insight_type: str, context: str) -> str:
        """Generate unique insight ID"""
        content = f"{insight_type}_{context}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _store_insight(self, insight: Insight):
        """Store insight in memory and persistent storage"""

        # Add to memory
        self.insights.append(insight)

        # Store in dataset history
        dataset_id = insight.metadata.get('dataset_id', 'platform')
        if dataset_id not in self.insight_history:
            self.insight_history[dataset_id] = []

        self.insight_history[dataset_id].append(insight)

        # Persist to disk
        self._save_insight_to_disk(insight)

        # Clean up old insights (keep last 100 per dataset)
        self._cleanup_old_insights()

    def _save_insight_to_disk(self, insight: Insight):
        """Save insight to persistent storage"""

        try:
            insight_file = self.storage_dir / f"insight_{insight.id}.json"

            insight_data = {
                "id": insight.id,
                "type": insight.type.value,
                "priority": insight.priority.value,
                "title": insight.title,
                "description": insight.description,
                "content": insight.content,
                "evidence": insight.evidence,
                "recommendations": insight.recommendations,
                "confidence_score": insight.confidence_score,
                "timestamp": insight.timestamp.isoformat(),
                "expires_at": insight.expires_at.isoformat() if insight.expires_at else None,
                "metadata": insight.metadata,
                "tags": insight.tags
            }

            with open(insight_file, 'w') as f:
                json.dump(insight_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save insight to disk: {e}")

    def _load_insights(self):
        """Load existing insights from persistent storage"""

        try:
            insight_files = list(self.storage_dir.glob("insight_*.json"))

            for insight_file in insight_files:
                try:
                    with open(insight_file, 'r') as f:
                        data = json.load(f)

                    # Reconstruct insight object
                    insight = Insight(
                        id=data["id"],
                        type=InsightType(data["type"]),
                        priority=InsightPriority(data["priority"]),
                        title=data["title"],
                        description=data["description"],
                        content=data["content"],
                        evidence=data["evidence"],
                        recommendations=data["recommendations"],
                        confidence_score=data["confidence_score"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
                        metadata=data["metadata"],
                        tags=data["tags"]
                    )

                    # Only load non-expired insights
                    if insight.expires_at is None or insight.expires_at > datetime.now():
                        self.insights.append(insight)

                except Exception as e:
                    logger.warning(f"Failed to load insight from {insight_file}: {e}")

            logger.info(f"Loaded {len(self.insights)} insights from storage")

        except Exception as e:
            logger.warning(f"Failed to load insights: {e}")

    def _cleanup_old_insights(self):
        """Clean up old and expired insights"""

        current_time = datetime.now()

        # Remove expired insights
        self.insights = [
            i for i in self.insights
            if i.expires_at is None or i.expires_at > current_time
        ]

        # Limit insights per dataset
        for dataset_id, insights in self.insight_history.items():
            if len(insights) > 100:
                # Keep only the most recent 100
                self.insight_history[dataset_id] = sorted(
                    insights,
                    key=lambda x: x.timestamp,
                    reverse=True
                )[:100]

        # Clean up disk storage for expired insights
        try:
            insight_files = list(self.storage_dir.glob("insight_*.json"))
            for insight_file in insight_files:
                try:
                    with open(insight_file, 'r') as f:
                        data = json.load(f)

                    if data.get("expires_at"):
                        expires_at = datetime.fromisoformat(data["expires_at"])
                        if expires_at <= current_time:
                            insight_file.unlink()  # Delete expired insight file

                except Exception as e:
                    logger.warning(f"Failed to clean up insight file {insight_file}: {e}")

        except Exception as e:
            logger.warning(f"Failed to clean up insight files: {e}")