"""
Unified Query Executor

Advanced execution engine for multi-dataset queries with intelligent data merging,
join optimization, and result aggregation capabilities.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from pathlib import Path
import uuid
from functools import reduce

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from .multi_dataset_workflow import UnifiedQuery, QueryJoin
from .enhanced_api_client import SoQLQueryBuilder
from .exceptions import ScoutDiscoveryError, ValidationError, ConfigurationError


@dataclass
class ExecutionPlan:
    """Optimized execution plan for unified query"""
    query_id: str
    execution_order: List[str]  # Dataset IDs in execution order
    join_sequence: List[QueryJoin]  # Joins in optimal order
    estimated_cost: float
    optimization_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'execution_order': self.execution_order,
            'join_sequence': [
                {
                    'left_dataset': j.left_dataset,
                    'right_dataset': j.right_dataset,
                    'left_column': j.left_column,
                    'right_column': j.right_column,
                    'join_type': j.join_type,
                    'confidence': j.confidence
                } for j in self.join_sequence
            ],
            'estimated_cost': self.estimated_cost,
            'optimization_notes': self.optimization_notes
        }


@dataclass
class ExecutionResults:
    """Results of unified query execution"""
    query_id: str
    execution_time: float
    datasets_processed: int
    total_rows_fetched: int
    total_rows_final: int
    join_success_rate: float
    individual_results: Dict[str, pd.DataFrame]
    merged_result: Optional[pd.DataFrame] = None
    execution_plan: Optional[ExecutionPlan] = None
    warnings: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'execution_time': self.execution_time,
            'datasets_processed': self.datasets_processed,
            'total_rows_fetched': self.total_rows_fetched,
            'total_rows_final': self.total_rows_final,
            'join_success_rate': self.join_success_rate,
            'individual_dataset_shapes': {
                dataset_id: {'rows': len(df), 'columns': len(df.columns)}
                for dataset_id, df in self.individual_results.items()
            },
            'merged_result_shape': {
                'rows': len(self.merged_result),
                'columns': len(self.merged_result.columns)
            } if self.merged_result is not None else None,
            'execution_plan': self.execution_plan.to_dict() if self.execution_plan else None,
            'warnings': self.warnings,
            'errors': self.errors
        }


class QueryOptimizer:
    """Optimizer for multi-dataset query execution"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def create_execution_plan(self, unified_query: UnifiedQuery,
                            dataset_sizes: Optional[Dict[str, int]] = None) -> ExecutionPlan:
        """
        Create optimized execution plan for unified query

        Args:
            unified_query: UnifiedQuery to optimize
            dataset_sizes: Optional dataset size estimates for cost calculation

        Returns:
            ExecutionPlan with optimized execution strategy
        """
        try:
            self.logger.info(f"Creating execution plan for query {unified_query.query_id}")

            # Analyze join dependencies
            join_graph = self._build_join_graph(unified_query.joins)

            # Determine optimal execution order
            execution_order = self._optimize_execution_order(
                unified_query.primary_dataset,
                join_graph,
                dataset_sizes or {}
            )

            # Optimize join sequence
            optimized_joins = self._optimize_join_sequence(
                unified_query.joins,
                execution_order
            )

            # Calculate estimated cost
            estimated_cost = self._estimate_execution_cost(
                execution_order,
                optimized_joins,
                dataset_sizes or {}
            )

            # Generate optimization notes
            notes = self._generate_optimization_notes(
                execution_order,
                optimized_joins,
                dataset_sizes
            )

            plan = ExecutionPlan(
                query_id=unified_query.query_id,
                execution_order=execution_order,
                join_sequence=optimized_joins,
                estimated_cost=estimated_cost,
                optimization_notes=notes
            )

            self.logger.info(f"Execution plan created with cost estimate: {estimated_cost:.2f}")
            return plan

        except Exception as e:
            error_msg = f"Execution plan creation failed: {str(e)}"
            self.logger.error(error_msg)
            raise ScoutDiscoveryError(error_msg) from e

    def _build_join_graph(self, joins: List[QueryJoin]) -> Dict[str, List[str]]:
        """Build dependency graph from joins"""
        graph = {}
        for join in joins:
            if join.left_dataset not in graph:
                graph[join.left_dataset] = []
            if join.right_dataset not in graph:
                graph[join.right_dataset] = []

            graph[join.left_dataset].append(join.right_dataset)

        return graph

    def _optimize_execution_order(self, primary_dataset: str,
                                 join_graph: Dict[str, List[str]],
                                 dataset_sizes: Dict[str, int]) -> List[str]:
        """Optimize execution order based on dataset sizes and join dependencies"""

        # Start with primary dataset
        execution_order = [primary_dataset]
        remaining = set(join_graph.keys()) - {primary_dataset}

        # Greedily add datasets, preferring smaller datasets first
        while remaining:
            # Find datasets that can be joined next
            candidates = []
            for dataset in remaining:
                # Check if it can join with any dataset already in execution order
                can_join = any(
                    target in execution_order
                    for target in join_graph.get(dataset, [])
                )
                if can_join:
                    size = dataset_sizes.get(dataset, 1000)  # Default size
                    candidates.append((dataset, size))

            if not candidates:
                # Add remaining datasets in size order
                candidates = [(d, dataset_sizes.get(d, 1000)) for d in remaining]

            # Choose smallest dataset
            next_dataset = min(candidates, key=lambda x: x[1])[0]
            execution_order.append(next_dataset)
            remaining.remove(next_dataset)

        return execution_order

    def _optimize_join_sequence(self, joins: List[QueryJoin],
                               execution_order: List[str]) -> List[QueryJoin]:
        """Optimize join sequence based on execution order"""
        optimized = []

        for i, dataset in enumerate(execution_order[1:], 1):
            # Find join that connects this dataset to previous ones
            for join in joins:
                if ((join.left_dataset == dataset and join.right_dataset in execution_order[:i]) or
                    (join.right_dataset == dataset and join.left_dataset in execution_order[:i])):
                    optimized.append(join)
                    break

        return optimized

    def _estimate_execution_cost(self, execution_order: List[str],
                               joins: List[QueryJoin],
                               dataset_sizes: Dict[str, int]) -> float:
        """Estimate execution cost based on data sizes and join complexity"""
        cost = 0.0

        # Base cost for fetching each dataset
        for dataset in execution_order:
            size = dataset_sizes.get(dataset, 1000)
            cost += size * 0.001  # Base fetch cost

        # Join costs
        for join in joins:
            left_size = dataset_sizes.get(join.left_dataset, 1000)
            right_size = dataset_sizes.get(join.right_dataset, 1000)

            # Join cost is roughly O(n*m) for nested loop, O(n+m) for hash join
            # Assume hash join for simplicity
            join_cost = (left_size + right_size) * 0.01

            # Penalty for low confidence joins
            if join.confidence < 0.7:
                join_cost *= 1.5

            cost += join_cost

        return cost

    def _generate_optimization_notes(self, execution_order: List[str],
                                   joins: List[QueryJoin],
                                   dataset_sizes: Optional[Dict[str, int]]) -> List[str]:
        """Generate human-readable optimization notes"""
        notes = []

        notes.append(f"Execution order optimized for {len(execution_order)} datasets")

        if dataset_sizes:
            largest_dataset = max(dataset_sizes, key=dataset_sizes.get)
            notes.append(f"Largest dataset: {largest_dataset} ({dataset_sizes[largest_dataset]:,} rows)")

        low_confidence_joins = [j for j in joins if j.confidence < 0.6]
        if low_confidence_joins:
            notes.append(f"Warning: {len(low_confidence_joins)} joins have low confidence scores")

        if len(joins) > 5:
            notes.append("Complex query with multiple joins - consider reducing scope")

        return notes


class UnifiedQueryExecutor:
    """
    Advanced executor for unified multi-dataset queries
    """

    def __init__(self,
                 scout_instance,
                 enable_optimization: bool = True,
                 max_sample_size: int = 10000,
                 join_memory_limit_mb: int = 500,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize unified query executor

        Args:
            scout_instance: ScoutDataDiscovery instance for data access
            enable_optimization: Whether to optimize query execution
            max_sample_size: Maximum sample size per dataset
            join_memory_limit_mb: Memory limit for join operations
            logger: Optional logger instance
        """
        self.scout = scout_instance
        self.enable_optimization = enable_optimization
        self.max_sample_size = max_sample_size
        self.join_memory_limit_mb = join_memory_limit_mb
        self.logger = logger or logging.getLogger(__name__)

        # Initialize optimizer
        self.optimizer = QueryOptimizer(logger) if enable_optimization else None

        # Execution statistics
        self.execution_stats = {
            'queries_executed': 0,
            'total_execution_time': 0,
            'datasets_processed': 0,
            'successful_joins': 0,
            'failed_joins': 0
        }

    def execute(self, unified_query: UnifiedQuery,
                sample_mode: bool = True,
                return_individual: bool = True) -> ExecutionResults:
        """
        Execute unified query with optimization and error handling

        Args:
            unified_query: UnifiedQuery to execute
            sample_mode: Whether to use sampling for large datasets
            return_individual: Whether to return individual dataset results

        Returns:
            ExecutionResults with complete execution information
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"Executing unified query {unified_query.query_id}")

            # Create execution plan
            execution_plan = None
            if self.optimizer:
                dataset_sizes = self._estimate_dataset_sizes(unified_query.datasets)
                execution_plan = self.optimizer.create_execution_plan(unified_query, dataset_sizes)

            # Execute individual dataset queries
            individual_results = self._execute_individual_queries(
                unified_query,
                execution_plan,
                sample_mode
            )

            # Perform joins if requested
            merged_result = None
            join_success_rate = 0.0

            if unified_query.joins and len(individual_results) > 1:
                merged_result, join_success_rate = self._execute_joins(
                    individual_results,
                    unified_query.joins,
                    execution_plan
                )

            # Calculate results
            execution_time = (datetime.now() - start_time).total_seconds()
            total_rows_fetched = sum(len(df) for df in individual_results.values())
            total_rows_final = len(merged_result) if merged_result is not None else total_rows_fetched

            # Create results object
            results = ExecutionResults(
                query_id=unified_query.query_id,
                execution_time=execution_time,
                datasets_processed=len(individual_results),
                total_rows_fetched=total_rows_fetched,
                total_rows_final=total_rows_final,
                join_success_rate=join_success_rate,
                individual_results=individual_results if return_individual else {},
                merged_result=merged_result,
                execution_plan=execution_plan
            )

            # Update statistics
            self._update_execution_stats(results)

            self.logger.info(f"Query execution completed in {execution_time:.2f} seconds")
            return results

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Query execution failed: {str(e)}"
            self.logger.error(error_msg)

            # Return error results
            return ExecutionResults(
                query_id=unified_query.query_id,
                execution_time=execution_time,
                datasets_processed=0,
                total_rows_fetched=0,
                total_rows_final=0,
                join_success_rate=0.0,
                individual_results={},
                errors=[error_msg]
            )

    def _execute_individual_queries(self, unified_query: UnifiedQuery,
                                   execution_plan: Optional[ExecutionPlan],
                                   sample_mode: bool) -> Dict[str, pd.DataFrame]:
        """Execute individual dataset queries"""
        results = {}

        # Use execution plan order if available
        dataset_order = execution_plan.execution_order if execution_plan else unified_query.datasets

        for dataset_id in dataset_order:
            try:
                self.logger.info(f"Fetching data from {dataset_id}")

                # Build query
                query_builder = self._build_dataset_query(unified_query, dataset_id, sample_mode)

                # Execute query
                df = self.scout.download_dataset_sample(
                    dataset_id,
                    query_builder=query_builder
                )

                # Validate result
                if df.empty:
                    self.logger.warning(f"Dataset {dataset_id} returned empty result")
                else:
                    results[dataset_id] = df
                    self.logger.info(f"Retrieved {len(df)} rows from {dataset_id}")

            except Exception as e:
                self.logger.error(f"Failed to fetch data from {dataset_id}: {str(e)}")
                # Continue with other datasets

        return results

    def _build_dataset_query(self, unified_query: UnifiedQuery,
                           dataset_id: str, sample_mode: bool) -> SoQLQueryBuilder:
        """Build SoQL query for individual dataset"""
        query_builder = self.scout.create_query_builder()

        # Select columns
        selected_columns = unified_query.selected_columns.get(dataset_id, [])
        if selected_columns:
            query_builder = query_builder.select(*selected_columns)

        # Apply filters
        if dataset_id in unified_query.filters:
            filters = unified_query.filters[dataset_id]

            # Apply conditions
            for condition in filters.get('conditions', []):
                query_builder = query_builder.where(condition)

            # Apply date range
            if 'date_column' in filters and unified_query.date_range:
                date_col = filters['date_column']
                start_date, end_date = unified_query.date_range
                query_builder = query_builder.where_date_range(date_col, start_date, end_date)

        # Apply aggregations
        if dataset_id in unified_query.aggregations:
            aggs = unified_query.aggregations[dataset_id]
            # Note: Complex aggregations would need custom SoQL handling
            for column, operation in aggs.items():
                if operation.lower() in ['count', 'sum', 'avg', 'min', 'max']:
                    query_builder = query_builder.select(f"{operation.lower()}({column}) as {column}_{operation.lower()}")

        # Apply limit
        limit = unified_query.limit or self.max_sample_size
        if sample_mode:
            limit = min(limit, self.max_sample_size)

        query_builder = query_builder.limit(limit)

        return query_builder

    def _execute_joins(self, individual_results: Dict[str, pd.DataFrame],
                      joins: List[QueryJoin],
                      execution_plan: Optional[ExecutionPlan]) -> Tuple[pd.DataFrame, float]:
        """Execute join operations on individual results"""

        if not joins:
            # If no joins specified, return concatenated results
            combined_df = pd.concat(list(individual_results.values()), ignore_index=True)
            return combined_df, 1.0

        # Use optimized join sequence if available
        join_sequence = execution_plan.join_sequence if execution_plan else joins

        successful_joins = 0
        total_joins = len(join_sequence)

        # Start with first dataset
        result_df = None
        processed_datasets = set()

        for join in join_sequence:
            try:
                left_df = individual_results.get(join.left_dataset)
                right_df = individual_results.get(join.right_dataset)

                if left_df is None or right_df is None:
                    self.logger.warning(f"Missing data for join: {join.left_dataset} -> {join.right_dataset}")
                    continue

                # Prepare DataFrames for join
                if result_df is None:
                    # First join
                    result_df = left_df.copy()
                    processed_datasets.add(join.left_dataset)

                # Determine which DataFrame to join
                if join.right_dataset not in processed_datasets:
                    join_df = right_df.copy()
                    join_on_left = join.left_column
                    join_on_right = join.right_column
                elif join.left_dataset not in processed_datasets:
                    join_df = left_df.copy()
                    join_on_left = join.right_column
                    join_on_right = join.left_column
                else:
                    continue  # Both datasets already processed

                # Add dataset prefix to avoid column conflicts
                dataset_suffix = f"_{join.right_dataset}"
                join_df = join_df.add_suffix(dataset_suffix)
                join_on_right = join_on_right + dataset_suffix

                # Perform join
                if join.join_type.lower() == 'inner':
                    result_df = result_df.merge(join_df, left_on=join_on_left, right_on=join_on_right, how='inner')
                elif join.join_type.lower() == 'left':
                    result_df = result_df.merge(join_df, left_on=join_on_left, right_on=join_on_right, how='left')
                elif join.join_type.lower() == 'right':
                    result_df = result_df.merge(join_df, left_on=join_on_left, right_on=join_on_right, how='right')
                elif join.join_type.lower() == 'outer':
                    result_df = result_df.merge(join_df, left_on=join_on_left, right_on=join_on_right, how='outer')

                processed_datasets.add(join.right_dataset if join.right_dataset not in processed_datasets else join.left_dataset)
                successful_joins += 1

                self.logger.info(f"Successfully joined {join.left_dataset} with {join.right_dataset}: {len(result_df)} rows")

            except Exception as e:
                self.logger.error(f"Join failed ({join.left_dataset} -> {join.right_dataset}): {str(e)}")
                continue

        # If no successful joins, return concatenated data
        if result_df is None or successful_joins == 0:
            self.logger.warning("No successful joins, returning concatenated data")
            result_df = pd.concat(list(individual_results.values()), ignore_index=True)

        join_success_rate = successful_joins / max(total_joins, 1)
        return result_df, join_success_rate

    def _estimate_dataset_sizes(self, datasets: List[str]) -> Dict[str, int]:
        """Estimate dataset sizes for optimization"""
        sizes = {}
        for dataset_id in datasets:
            # Try to get cached information
            if dataset_id in self.scout.results.get('metadata_cache', {}):
                metadata = self.scout.results['metadata_cache'][dataset_id]
                # Rough estimate based on download count and page views
                estimated_size = max(
                    metadata.get('download_count', 1000) * 10,
                    metadata.get('page_views', 1000)
                )
                sizes[dataset_id] = min(estimated_size, 1000000)  # Cap at 1M
            else:
                sizes[dataset_id] = 5000  # Default estimate

        return sizes

    def _update_execution_stats(self, results: ExecutionResults):
        """Update execution statistics"""
        self.execution_stats['queries_executed'] += 1
        self.execution_stats['total_execution_time'] += results.execution_time
        self.execution_stats['datasets_processed'] += results.datasets_processed

        if results.join_success_rate > 0.5:
            self.execution_stats['successful_joins'] += 1
        else:
            self.execution_stats['failed_joins'] += 1

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = self.execution_stats.copy()

        if stats['queries_executed'] > 0:
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['queries_executed']
            stats['avg_datasets_per_query'] = stats['datasets_processed'] / stats['queries_executed']
            stats['join_success_rate'] = stats['successful_joins'] / (stats['successful_joins'] + stats['failed_joins']) if (stats['successful_joins'] + stats['failed_joins']) > 0 else 0

        return stats

    def export_results(self, results: ExecutionResults,
                      output_dir: Optional[str] = None,
                      export_individual: bool = True) -> Dict[str, str]:
        """Export execution results to files"""
        try:
            output_path = Path(output_dir) if output_dir else Path("query_results")
            output_path.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            files_created = {}

            # Export merged results
            if results.merged_result is not None:
                merged_file = output_path / f"merged_results_{results.query_id}_{timestamp}.csv"
                results.merged_result.to_csv(merged_file, index=False)
                files_created['merged_results'] = str(merged_file)

            # Export individual results
            if export_individual and results.individual_results:
                individual_dir = output_path / "individual_datasets"
                individual_dir.mkdir(exist_ok=True)

                for dataset_id, df in results.individual_results.items():
                    individual_file = individual_dir / f"{dataset_id}_{timestamp}.csv"
                    df.to_csv(individual_file, index=False)
                    files_created[f'individual_{dataset_id}'] = str(individual_file)

            # Export execution metadata
            metadata_file = output_path / f"execution_metadata_{results.query_id}_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            files_created['execution_metadata'] = str(metadata_file)

            self.logger.info(f"Results exported to {len(files_created)} files")
            return files_created

        except Exception as e:
            self.logger.error(f"Failed to export results: {str(e)}")
            return {}