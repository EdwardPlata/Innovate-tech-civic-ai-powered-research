"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


# Dataset Models
class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str
    download_count: int
    updated_at: Optional[str]
    category: Optional[str]
    tags: List[str] = []
    columns_count: int
    quality_score: Optional[float] = None


class QualityAssessment(BaseModel):
    dataset_id: str
    overall_score: float
    grade: str
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    timeliness_score: float
    usability_score: float
    missing_percentage: float
    insights: List[str] = []


class SearchRequest(BaseModel):
    search_terms: List[str]
    limit: Optional[int] = 50
    include_quality: bool = False


class RelationshipRequest(BaseModel):
    dataset_id: str
    similarity_threshold: float = 0.3
    max_related: int = 10


class RelationshipResponse(BaseModel):
    dataset_id: str
    related_datasets: List[Dict[str, Any]]
    network_stats: Dict[str, Any]


# AI Analysis Models
class AIAnalysisRequest(BaseModel):
    dataset_id: str
    analysis_type: str = "overview"  # overview, quality, insights, relationships
    custom_prompt: Optional[str] = None
    include_sample: bool = False
    sample_size: int = 100


class AIQuestionRequest(BaseModel):
    dataset_id: str
    question: str
    include_sample: bool = False
    sample_size: int = 100


class MultiDatasetAnalysisRequest(BaseModel):
    dataset_ids: List[str]
    analysis_type: str = "comparison"  # comparison, correlation, integration, insights
    join_strategy: Optional[str] = "auto"  # auto, inner, left, outer, none
    custom_prompt: Optional[str] = None
    include_samples: bool = True
    sample_size: int = 1000
    generate_visualizations: bool = True


class DatasetSelection(BaseModel):
    dataset_id: str
    name: str
    category: str
    selected_columns: Optional[List[str]] = None
    join_column: Optional[str] = None


class MultiDatasetProject(BaseModel):
    project_name: str
    description: str
    datasets: List[DatasetSelection]
    analysis_goals: List[str]
    created_at: Optional[datetime] = None


class VisualizationRequest(BaseModel):
    datasets: List[str]
    chart_type: str = "auto"  # auto, bar, line, scatter, heatmap, network
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_by: Optional[str] = None
    title: Optional[str] = None
    custom_query: Optional[str] = None


# AI Configuration Models
class AIConfigRequest(BaseModel):
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    primary_provider: str = "openai"
    fallback_providers: List[str] = ["openrouter"]
    enable_semantic_cache: bool = True


class AIKeyUpdate(BaseModel):
    provider: str  # openai, openrouter, nvidia
    api_key: str
    model: Optional[str] = None


# Codebase Analysis Models
class CodebaseAnalysisRequest(BaseModel):
    codebase_path: str
    analysis_focus: str = "overview"  # overview, architecture, quality, security
    file_extensions: Optional[List[str]] = None
    exclude_dirs: Optional[List[str]] = None


class CodebaseQuestionRequest(BaseModel):
    codebase_path: str
    question: str
    context_files: Optional[List[str]] = None
    force_rechunk: bool = False


class CodeSuggestionsRequest(BaseModel):
    codebase_path: str
    file_path: str
    improvement_type: str = "all"  # all, performance, readability, security, maintainability


# Insights Engine Models
class InsightsGenerationRequest(BaseModel):
    dataset_id: Optional[str] = None
    dataset_info: Optional[Dict[str, Any]] = None
    sample_data: Optional[List[Dict]] = None
    analysis_history: Optional[List[Dict]] = None
    insight_types: Optional[List[str]] = None  # Filter which types of insights to generate


class PlatformInsightsRequest(BaseModel):
    usage_statistics: Dict[str, Any]
    user_patterns: Dict[str, Any]
    system_health: Dict[str, Any]


class InsightsFilterRequest(BaseModel):
    insight_type: Optional[str] = None
    priority: Optional[str] = None
    dataset_id: Optional[str] = None
    limit: int = 10
    include_expired: bool = False


# Data Explorer Models
class DataExplorationRequest(BaseModel):
    dataset_data: List[Dict[str, Any]]  # Actual dataset records
    dataset_name: str
    dataset_description: str = "Dataset for analysis"
    user_question: str
    available_datasets: Optional[Dict[str, List[Dict[str, Any]]]] = None


class DataVisualizationRequest(BaseModel):
    dataset_data: List[Dict[str, Any]]
    dataset_name: str
    dataset_description: str = "Dataset for visualization"
    chart_request: str
    analysis_context: Optional[str] = None
    visualization_options: Optional[Dict[str, Any]] = None


class DatasetJoinRequest(BaseModel):
    primary_dataset: List[Dict[str, Any]]
    primary_dataset_name: str
    datasets_to_join: Dict[str, List[Dict[str, Any]]]
    join_objective: str
    analysis_goal: str


class StatisticalTestRequest(BaseModel):
    dataset_data: List[Dict[str, Any]]
    dataset_name: str
    dataset_description: str = "Dataset for statistical testing"
    hypothesis: str
    test_type: Optional[str] = None


class ComprehensiveEDARequest(BaseModel):
    dataset_data: List[Dict[str, Any]]
    dataset_name: str
    dataset_description: str = "Dataset for comprehensive analysis"
    focus_areas: Optional[List[str]] = None
