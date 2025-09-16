"""
Scout Data Discovery Package

A comprehensive Python package for automated data discovery, quality assessment,
and curation based on Scout (https://scout.tsdataclinic.com/) methodology.
"""

__version__ = "1.0.0"
__author__ = "Data Analytics Team"
__description__ = "Scout-inspired data discovery and quality assessment toolkit"

from .src.scout_discovery import ScoutDataDiscovery
from .src.data_quality import DataQualityAssessor
from .src.enhanced_api_client import EnhancedNYCDataClient, ScoutIntegratedClient, SoQLQueryBuilder
from .src.multi_dataset_workflow import MultiDatasetWorkflow, RelationshipGraph, UnifiedQuery
from .src.unified_query_executor import UnifiedQueryExecutor, ExecutionResults
from .src.workflow_orchestrator import MultiDatasetOrchestrator, WorkflowConfig
from .src.column_relationship_mapper import ColumnAnalyzer, RelationshipMapper, ColumnRelationship
from .src.exceptions import ScoutDiscoveryError, DataQualityError, APIError

__all__ = [
    'ScoutDataDiscovery',
    'DataQualityAssessor',
    'EnhancedNYCDataClient',
    'ScoutIntegratedClient',
    'SoQLQueryBuilder',
    'MultiDatasetWorkflow',
    'RelationshipGraph',
    'UnifiedQuery',
    'UnifiedQueryExecutor',
    'ExecutionResults',
    'MultiDatasetOrchestrator',
    'WorkflowConfig',
    'ColumnAnalyzer',
    'RelationshipMapper',
    'ColumnRelationship',
    'ScoutDiscoveryError',
    'DataQualityError',
    'APIError'
]