"""
Multi-Dataset Workflow Orchestrator

Complete workflow orchestrator that ties together dataset discovery, relationship mapping,
query generation, and execution for seamless multi-dataset operations.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .scout_discovery import ScoutDataDiscovery
from .multi_dataset_workflow import MultiDatasetWorkflow, RelationshipGraph, UnifiedQuery
from .unified_query_executor import UnifiedQueryExecutor, ExecutionResults
from .column_relationship_mapper import ColumnRelationship, RelationshipType
from .exceptions import ScoutDiscoveryError, ValidationError, ConfigurationError


@dataclass
class WorkflowConfig:
    """Configuration for workflow orchestration"""
    min_relationship_confidence: float = 0.4
    max_related_datasets: int = 10
    quality_threshold: float = 70
    enable_query_optimization: bool = True
    max_sample_size: int = 10000
    default_join_strategy: str = "best_match"
    auto_export_results: bool = True
    cache_intermediate_results: bool = True


@dataclass
class WorkflowStep:
    """Individual workflow step tracking"""
    step_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: Optional[float] = None

    def complete(self, result: Any = None):
        """Mark step as completed"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = "completed"
        self.result = result

    def fail(self, error: str):
        """Mark step as failed"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = "failed"
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_name': self.step_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'duration': self.duration,
            'error': self.error
        }


@dataclass
class WorkflowResults:
    """Complete workflow execution results"""
    workflow_id: str
    source_dataset_id: str
    total_execution_time: float
    steps_executed: List[WorkflowStep]
    relationship_graph: Optional[RelationshipGraph] = None
    unified_query: Optional[UnifiedQuery] = None
    execution_results: Optional[ExecutionResults] = None
    final_dataset: Optional[pd.DataFrame] = None
    workflow_config: Optional[WorkflowConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'source_dataset_id': self.source_dataset_id,
            'total_execution_time': self.total_execution_time,
            'steps_executed': [step.to_dict() for step in self.steps_executed],
            'relationship_graph': self.relationship_graph.to_dict() if self.relationship_graph else None,
            'unified_query': self.unified_query.to_dict() if self.unified_query else None,
            'execution_results': self.execution_results.to_dict() if self.execution_results else None,
            'final_dataset_shape': {
                'rows': len(self.final_dataset),
                'columns': len(self.final_dataset.columns)
            } if self.final_dataset is not None else None,
            'workflow_config': asdict(self.workflow_config) if self.workflow_config else None
        }

    def get_successful_steps(self) -> List[WorkflowStep]:
        """Get all successful workflow steps"""
        return [step for step in self.steps_executed if step.status == "completed"]

    def get_failed_steps(self) -> List[WorkflowStep]:
        """Get all failed workflow steps"""
        return [step for step in self.steps_executed if step.status == "failed"]

    def is_successful(self) -> bool:
        """Check if workflow completed successfully"""
        return all(step.status == "completed" for step in self.steps_executed)


class MultiDatasetOrchestrator:
    """
    Complete orchestrator for multi-dataset discovery and analysis workflows
    """

    def __init__(self,
                 scout: ScoutDataDiscovery,
                 config: Optional[WorkflowConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize workflow orchestrator

        Args:
            scout: ScoutDataDiscovery instance
            config: Workflow configuration
            logger: Optional logger instance
        """
        self.scout = scout
        self.config = config or WorkflowConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Initialize workflow components
        self.workflow_engine = MultiDatasetWorkflow(
            scout=scout,
            min_relationship_confidence=self.config.min_relationship_confidence,
            max_related_datasets=self.config.max_related_datasets,
            logger=logger
        )

        self.query_executor = UnifiedQueryExecutor(
            scout_instance=scout,
            enable_optimization=self.config.enable_query_optimization,
            max_sample_size=self.config.max_sample_size,
            logger=logger
        )

        # Workflow state
        self.active_workflows = {}
        self.workflow_history = []

        # Callbacks for workflow events
        self.step_callbacks: Dict[str, List[Callable]] = {
            'step_start': [],
            'step_complete': [],
            'step_fail': [],
            'workflow_complete': [],
            'workflow_fail': []
        }

    def run_complete_workflow(self,
                            source_dataset_id: str,
                            search_terms: Optional[List[str]] = None,
                            custom_config: Optional[WorkflowConfig] = None,
                            selected_datasets: Optional[List[str]] = None,
                            column_selection: Optional[Dict[str, List[str]]] = None,
                            filters: Optional[Dict[str, Dict[str, Any]]] = None,
                            date_range: Optional[Tuple[datetime, datetime]] = None) -> WorkflowResults:
        """
        Run complete multi-dataset workflow from discovery to final results

        Args:
            source_dataset_id: Primary dataset to start from
            search_terms: Optional search terms for discovery
            custom_config: Optional custom configuration
            selected_datasets: Specific datasets to include
            column_selection: Custom column selection
            filters: Custom filters per dataset
            date_range: Date range for temporal filtering

        Returns:
            WorkflowResults with complete execution information
        """
        workflow_id = str(uuid.uuid4())
        start_time = datetime.now()
        config = custom_config or self.config

        self.logger.info(f"Starting complete workflow {workflow_id} for dataset {source_dataset_id}")

        steps = []
        relationship_graph = None
        unified_query = None
        execution_results = None
        final_dataset = None

        try:
            # Step 1: Discover Related Datasets
            step1 = WorkflowStep("discover_related_datasets", datetime.now())
            steps.append(step1)
            step1.status = "running"
            self._notify_step_start(step1)

            try:
                relationship_graph = self.workflow_engine.discover_related_datasets(
                    source_dataset_id=source_dataset_id,
                    search_terms=search_terms,
                    quality_threshold=config.quality_threshold
                )
                step1.complete(relationship_graph)
                self._notify_step_complete(step1)

            except Exception as e:
                step1.fail(str(e))
                self._notify_step_fail(step1)
                raise

            # Step 2: Create Unified Query
            step2 = WorkflowStep("create_unified_query", datetime.now())
            steps.append(step2)
            step2.status = "running"
            self._notify_step_start(step2)

            try:
                unified_query = self.workflow_engine.create_unified_query(
                    relationship_graph=relationship_graph,
                    selected_datasets=selected_datasets,
                    column_selection=column_selection,
                    join_strategy=config.default_join_strategy
                )

                # Apply custom filters and date range
                if filters:
                    unified_query.filters.update(filters)
                if date_range:
                    unified_query.date_range = date_range

                step2.complete(unified_query)
                self._notify_step_complete(step2)

            except Exception as e:
                step2.fail(str(e))
                self._notify_step_fail(step2)
                raise

            # Step 3: Execute Unified Query
            step3 = WorkflowStep("execute_unified_query", datetime.now())
            steps.append(step3)
            step3.status = "running"
            self._notify_step_start(step3)

            try:
                execution_results = self.query_executor.execute(
                    unified_query=unified_query,
                    sample_mode=True,
                    return_individual=True
                )

                final_dataset = execution_results.merged_result

                step3.complete(execution_results)
                self._notify_step_complete(step3)

            except Exception as e:
                step3.fail(str(e))
                self._notify_step_fail(step3)
                raise

            # Step 4: Export Results (if enabled)
            if config.auto_export_results:
                step4 = WorkflowStep("export_results", datetime.now())
                steps.append(step4)
                step4.status = "running"
                self._notify_step_start(step4)

                try:
                    export_files = self._export_workflow_results(
                        workflow_id, relationship_graph, unified_query, execution_results
                    )
                    step4.complete(export_files)
                    self._notify_step_complete(step4)

                except Exception as e:
                    step4.fail(str(e))
                    self._notify_step_fail(step4)
                    # Don't raise - export failure shouldn't fail the whole workflow

            # Create final results
            total_time = (datetime.now() - start_time).total_seconds()

            workflow_results = WorkflowResults(
                workflow_id=workflow_id,
                source_dataset_id=source_dataset_id,
                total_execution_time=total_time,
                steps_executed=steps,
                relationship_graph=relationship_graph,
                unified_query=unified_query,
                execution_results=execution_results,
                final_dataset=final_dataset,
                workflow_config=config
            )

            # Store in history and notify
            self.workflow_history.append(workflow_results)
            self._notify_workflow_complete(workflow_results)

            self.logger.info(f"Workflow {workflow_id} completed successfully in {total_time:.2f} seconds")
            return workflow_results

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Workflow {workflow_id} failed: {str(e)}"
            self.logger.error(error_msg)

            # Create error results
            workflow_results = WorkflowResults(
                workflow_id=workflow_id,
                source_dataset_id=source_dataset_id,
                total_execution_time=total_time,
                steps_executed=steps,
                relationship_graph=relationship_graph,
                unified_query=unified_query,
                execution_results=execution_results,
                final_dataset=final_dataset,
                workflow_config=config
            )

            self._notify_workflow_fail(workflow_results)
            return workflow_results

    def run_discovery_only(self,
                          source_dataset_id: str,
                          search_terms: Optional[List[str]] = None,
                          quality_threshold: float = None) -> RelationshipGraph:
        """
        Run only the dataset discovery phase

        Args:
            source_dataset_id: Primary dataset
            search_terms: Optional search terms
            quality_threshold: Quality threshold for filtering

        Returns:
            RelationshipGraph with discovered relationships
        """
        self.logger.info(f"Running discovery-only workflow for {source_dataset_id}")

        threshold = quality_threshold or self.config.quality_threshold

        relationship_graph = self.workflow_engine.discover_related_datasets(
            source_dataset_id=source_dataset_id,
            search_terms=search_terms,
            quality_threshold=threshold
        )

        self.logger.info(f"Discovery completed: found {len(relationship_graph.related_datasets)} related datasets")
        return relationship_graph

    def create_query_from_graph(self,
                              relationship_graph: RelationshipGraph,
                              **query_options) -> UnifiedQuery:
        """
        Create unified query from existing relationship graph

        Args:
            relationship_graph: Existing relationship graph
            **query_options: Additional query options

        Returns:
            UnifiedQuery ready for execution
        """
        self.logger.info("Creating unified query from relationship graph")

        unified_query = self.workflow_engine.create_unified_query(
            relationship_graph=relationship_graph,
            **query_options
        )

        self.logger.info(f"Query created with {len(unified_query.joins)} joins")
        return unified_query

    def execute_query_only(self, unified_query: UnifiedQuery) -> ExecutionResults:
        """
        Execute only the query phase

        Args:
            unified_query: UnifiedQuery to execute

        Returns:
            ExecutionResults with query results
        """
        self.logger.info(f"Executing query {unified_query.query_id}")

        execution_results = self.query_executor.execute(unified_query)

        self.logger.info(f"Query execution completed: {execution_results.total_rows_final} final rows")
        return execution_results

    def analyze_relationships(self, relationship_graph: RelationshipGraph) -> Dict[str, Any]:
        """
        Analyze relationship graph and provide insights

        Args:
            relationship_graph: RelationshipGraph to analyze

        Returns:
            Dictionary with relationship analysis
        """
        analysis = {
            'source_dataset': relationship_graph.source_dataset,
            'total_related_datasets': len(relationship_graph.related_datasets),
            'relationship_types': {},
            'high_confidence_relationships': [],
            'join_candidates': [],
            'data_integration_potential': 'low'
        }

        # Analyze relationship types
        all_relationships = []
        for dataset_id, relationships in relationship_graph.related_datasets.items():
            all_relationships.extend(relationships)

        if all_relationships:
            # Count relationship types
            type_counts = {}
            high_confidence = []
            join_candidates = []

            for rel in all_relationships:
                rel_type = rel.relationship_type.value
                type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

                if rel.confidence_score > 0.7:
                    high_confidence.append({
                        'source_dataset': rel.source_column.dataset_id,
                        'target_dataset': rel.target_column.dataset_id,
                        'source_column': rel.source_column.name,
                        'target_column': rel.target_column.name,
                        'confidence': rel.confidence_score,
                        'type': rel_type
                    })

                if rel.join_potential > 0.6:
                    join_candidates.append({
                        'left_dataset': rel.source_column.dataset_id,
                        'right_dataset': rel.target_column.dataset_id,
                        'left_column': rel.source_column.name,
                        'right_column': rel.target_column.name,
                        'join_potential': rel.join_potential,
                        'confidence': rel.confidence_score
                    })

            analysis['relationship_types'] = type_counts
            analysis['high_confidence_relationships'] = high_confidence
            analysis['join_candidates'] = join_candidates

            # Assess integration potential
            if len(high_confidence) > 2 and len(join_candidates) > 1:
                analysis['data_integration_potential'] = 'high'
            elif len(high_confidence) > 0 or len(join_candidates) > 0:
                analysis['data_integration_potential'] = 'medium'

        return analysis

    def suggest_query_optimizations(self, unified_query: UnifiedQuery) -> List[Dict[str, str]]:
        """
        Suggest optimizations for unified query

        Args:
            unified_query: UnifiedQuery to analyze

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Check for too many datasets
        if len(unified_query.datasets) > 5:
            suggestions.append({
                'type': 'performance',
                'suggestion': 'Consider reducing the number of datasets for better performance',
                'impact': 'high'
            })

        # Check for low-confidence joins
        low_confidence_joins = [j for j in unified_query.joins if j.confidence < 0.6]
        if low_confidence_joins:
            suggestions.append({
                'type': 'data_quality',
                'suggestion': f'{len(low_confidence_joins)} joins have low confidence scores - verify join keys',
                'impact': 'medium'
            })

        # Check for missing filters
        if not unified_query.filters and not unified_query.date_range:
            suggestions.append({
                'type': 'performance',
                'suggestion': 'Add filters or date range to reduce data volume',
                'impact': 'medium'
            })

        # Check column selection
        total_columns = sum(len(cols) for cols in unified_query.selected_columns.values())
        if total_columns > 20:
            suggestions.append({
                'type': 'usability',
                'suggestion': 'Consider selecting fewer columns for easier analysis',
                'impact': 'low'
            })

        return suggestions

    def _export_workflow_results(self, workflow_id: str,
                               relationship_graph: RelationshipGraph,
                               unified_query: UnifiedQuery,
                               execution_results: ExecutionResults) -> Dict[str, str]:
        """Export complete workflow results"""
        files = {}

        # Export relationship graph
        if relationship_graph:
            graph_files = self.workflow_engine.save_workflow_results(
                relationship_graph, unified_query, f"workflow_{workflow_id}"
            )
            files.update(graph_files)

        # Export execution results
        if execution_results:
            result_files = self.query_executor.export_results(
                execution_results, f"workflow_{workflow_id}", export_individual=True
            )
            files.update(result_files)

        return files

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for workflow events"""
        if event_type in self.step_callbacks:
            self.step_callbacks[event_type].append(callback)

    def _notify_step_start(self, step: WorkflowStep):
        """Notify step start callbacks"""
        for callback in self.step_callbacks['step_start']:
            try:
                callback(step)
            except Exception as e:
                self.logger.warning(f"Step start callback failed: {str(e)}")

    def _notify_step_complete(self, step: WorkflowStep):
        """Notify step complete callbacks"""
        for callback in self.step_callbacks['step_complete']:
            try:
                callback(step)
            except Exception as e:
                self.logger.warning(f"Step complete callback failed: {str(e)}")

    def _notify_step_fail(self, step: WorkflowStep):
        """Notify step fail callbacks"""
        for callback in self.step_callbacks['step_fail']:
            try:
                callback(step)
            except Exception as e:
                self.logger.warning(f"Step fail callback failed: {str(e)}")

    def _notify_workflow_complete(self, results: WorkflowResults):
        """Notify workflow complete callbacks"""
        for callback in self.step_callbacks['workflow_complete']:
            try:
                callback(results)
            except Exception as e:
                self.logger.warning(f"Workflow complete callback failed: {str(e)}")

    def _notify_workflow_fail(self, results: WorkflowResults):
        """Notify workflow fail callbacks"""
        for callback in self.step_callbacks['workflow_fail']:
            try:
                callback(results)
            except Exception as e:
                self.logger.warning(f"Workflow fail callback failed: {str(e)}")

    def get_workflow_history(self) -> List[WorkflowResults]:
        """Get history of executed workflows"""
        return self.workflow_history.copy()

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        stats = {
            'total_workflows': len(self.workflow_history),
            'successful_workflows': len([w for w in self.workflow_history if w.is_successful()]),
            'failed_workflows': len([w for w in self.workflow_history if not w.is_successful()]),
            'avg_execution_time': 0,
            'query_executor_stats': self.query_executor.get_execution_stats()
        }

        if self.workflow_history:
            stats['avg_execution_time'] = sum(w.total_execution_time for w in self.workflow_history) / len(self.workflow_history)

            # Step-wise statistics
            step_stats = {}
            for workflow in self.workflow_history:
                for step in workflow.steps_executed:
                    if step.step_name not in step_stats:
                        step_stats[step.step_name] = {
                            'total_executions': 0,
                            'successful_executions': 0,
                            'avg_duration': 0,
                            'total_duration': 0
                        }

                    step_stats[step.step_name]['total_executions'] += 1
                    if step.status == 'completed':
                        step_stats[step.step_name]['successful_executions'] += 1

                    if step.duration:
                        step_stats[step.step_name]['total_duration'] += step.duration

            # Calculate averages
            for step_name, step_data in step_stats.items():
                if step_data['total_executions'] > 0:
                    step_data['avg_duration'] = step_data['total_duration'] / step_data['total_executions']
                    step_data['success_rate'] = step_data['successful_executions'] / step_data['total_executions']

            stats['step_statistics'] = step_stats

        return stats