"""
Multi-Dataset Discovery and Query Workflow

Comprehensive workflow for discovering related datasets using Scout, mapping column
relationships, and creating unified query objects for multi-table data operations.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import uuid

from .scout_discovery import ScoutDataDiscovery
from .enhanced_api_client import SoQLQueryBuilder
from .column_relationship_mapper import (
    ColumnAnalyzer, RelationshipMapper, ColumnMetadata,
    ColumnRelationship, RelationshipType, ColumnType
)
from .exceptions import ScoutDiscoveryError, ValidationError, ConfigurationError


@dataclass
class DatasetSchema:
    """Complete schema information for a dataset"""
    dataset_id: str
    dataset_name: str
    columns: List[ColumnMetadata]
    quality_score: float
    update_frequency: Optional[str] = None
    total_records: Optional[int] = None
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_id': self.dataset_id,
            'dataset_name': self.dataset_name,
            'columns': [asdict(col) for col in self.columns],
            'quality_score': self.quality_score,
            'update_frequency': self.update_frequency,
            'total_records': self.total_records,
            'last_updated': self.last_updated
        }


@dataclass
class RelationshipGraph:
    """Graph of relationships between datasets"""
    source_dataset: str
    related_datasets: Dict[str, List[ColumnRelationship]]
    relationship_summary: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_dataset': self.source_dataset,
            'related_datasets': {
                dataset_id: [rel.to_dict() for rel in relationships]
                for dataset_id, relationships in self.related_datasets.items()
            },
            'relationship_summary': self.relationship_summary
        }


@dataclass
class QueryJoin:
    """Definition of a join operation between datasets"""
    left_dataset: str
    right_dataset: str
    left_column: str
    right_column: str
    join_type: str  # 'inner', 'left', 'right', 'outer'
    relationship_type: str
    confidence: float


@dataclass
class UnifiedQuery:
    """Unified query object for multi-dataset operations"""
    query_id: str
    primary_dataset: str
    datasets: List[str]
    joins: List[QueryJoin]
    selected_columns: Dict[str, List[str]]  # dataset_id -> column_names
    filters: Dict[str, Dict[str, Any]]  # dataset_id -> filter_conditions
    aggregations: Dict[str, Dict[str, str]]  # dataset_id -> {column: operation}
    date_range: Optional[Tuple[datetime, datetime]] = None
    limit: Optional[int] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'primary_dataset': self.primary_dataset,
            'datasets': self.datasets,
            'joins': [asdict(join) for join in self.joins],
            'selected_columns': self.selected_columns,
            'filters': self.filters,
            'aggregations': self.aggregations,
            'date_range': [dt.isoformat() if dt else None for dt in (self.date_range or (None, None))],
            'limit': self.limit,
            'created_at': self.created_at.isoformat()
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class MultiDatasetWorkflow:
    """
    Complete workflow for multi-dataset discovery, relationship mapping, and querying
    """

    def __init__(self,
                 scout: ScoutDataDiscovery,
                 min_relationship_confidence: float = 0.4,
                 max_related_datasets: int = 10,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize multi-dataset workflow

        Args:
            scout: ScoutDataDiscovery instance
            min_relationship_confidence: Minimum confidence for column relationships
            max_related_datasets: Maximum number of related datasets to analyze
            logger: Optional logger instance
        """
        self.scout = scout
        self.min_relationship_confidence = min_relationship_confidence
        self.max_related_datasets = max_related_datasets
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.column_analyzer = ColumnAnalyzer(logger)
        self.relationship_mapper = RelationshipMapper(logger)

        # Cache for analyzed schemas
        self.schema_cache = {}
        self.relationship_cache = {}

    def discover_related_datasets(self,
                                 source_dataset_id: str,
                                 search_terms: Optional[List[str]] = None,
                                 quality_threshold: float = 70) -> RelationshipGraph:
        """
        Discover datasets related to the source dataset using Scout

        Args:
            source_dataset_id: ID of the source dataset
            search_terms: Optional search terms for discovery
            quality_threshold: Minimum quality score for related datasets

        Returns:
            RelationshipGraph with related datasets and relationships
        """
        try:
            self.logger.info(f"Discovering related datasets for {source_dataset_id}")

            # Step 1: Analyze source dataset schema
            source_schema = self._analyze_dataset_schema(source_dataset_id)

            # Step 2: Get dataset metadata to extract search terms
            if search_terms is None:
                search_terms = self._extract_search_terms_from_schema(source_schema)

            self.logger.info(f"Using search terms: {search_terms}")

            # Step 3: Use Scout to find related datasets
            related_datasets_df = self.scout.search_datasets_advanced(
                query=' '.join(search_terms),
                min_downloads=50,  # Filter for datasets with some usage
                limit=self.max_related_datasets * 2  # Get more candidates
            )

            if related_datasets_df.empty:
                self.logger.warning("No related datasets found")
                return RelationshipGraph(
                    source_dataset=source_dataset_id,
                    related_datasets={},
                    relationship_summary={}
                )

            # Step 4: Filter by quality and analyze schemas
            quality_datasets = []
            for _, dataset in related_datasets_df.iterrows():
                if dataset['id'] == source_dataset_id:
                    continue  # Skip self

                try:
                    # Quick quality check if available
                    if dataset['id'] in self.scout.results.get('quality_assessments', {}):
                        quality_score = self.scout.results['quality_assessments'][dataset['id']]['overall_scores']['total_score']
                    else:
                        quality_score = 75  # Default assumption

                    if quality_score >= quality_threshold:
                        quality_datasets.append(dataset)

                except Exception as e:
                    self.logger.warning(f"Failed to assess dataset {dataset['id']}: {str(e)}")
                    continue

            # Limit to max related datasets
            quality_datasets = quality_datasets[:self.max_related_datasets]

            # Step 5: Analyze relationships with each related dataset
            related_schemas = {}
            relationship_graph = {}
            relationship_summary = {}

            for dataset in quality_datasets:
                dataset_id = dataset['id']
                self.logger.info(f"Analyzing relationships with {dataset_id}")

                try:
                    # Analyze target dataset schema
                    target_schema = self._analyze_dataset_schema(dataset_id, dataset['name'])
                    related_schemas[dataset_id] = target_schema

                    # Find relationships between source and target
                    relationships = self._find_cross_dataset_relationships(source_schema, target_schema)

                    if relationships:
                        relationship_graph[dataset_id] = relationships

                        # Summarize relationship types
                        type_counts = {}
                        for rel in relationships:
                            rel_type = rel.relationship_type.value
                            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

                        relationship_summary[dataset_id] = type_counts

                        self.logger.info(f"Found {len(relationships)} relationships with {dataset_id}")

                except Exception as e:
                    self.logger.error(f"Failed to analyze relationships with {dataset_id}: {str(e)}")
                    continue

            return RelationshipGraph(
                source_dataset=source_dataset_id,
                related_datasets=relationship_graph,
                relationship_summary=relationship_summary
            )

        except Exception as e:
            error_msg = f"Related dataset discovery failed: {str(e)}"
            self.logger.error(error_msg)
            raise ScoutDiscoveryError(error_msg) from e

    def create_unified_query(self,
                           relationship_graph: RelationshipGraph,
                           selected_datasets: Optional[List[str]] = None,
                           column_selection: Optional[Dict[str, List[str]]] = None,
                           join_strategy: str = "best_match") -> UnifiedQuery:
        """
        Create a unified query object for multi-dataset operations

        Args:
            relationship_graph: Graph of dataset relationships
            selected_datasets: Specific datasets to include (None = all)
            column_selection: Specific columns per dataset (None = smart selection)
            join_strategy: Strategy for selecting joins ('best_match', 'all', 'high_confidence')

        Returns:
            UnifiedQuery object ready for execution
        """
        try:
            self.logger.info("Creating unified query from relationship graph")

            source_dataset = relationship_graph.source_dataset

            # Select datasets to include
            if selected_datasets is None:
                # Include all datasets with relationships
                target_datasets = list(relationship_graph.related_datasets.keys())
            else:
                target_datasets = [d for d in selected_datasets if d in relationship_graph.related_datasets]

            all_datasets = [source_dataset] + target_datasets

            # Generate joins based on relationships
            joins = self._generate_joins(relationship_graph, target_datasets, join_strategy)

            # Select columns if not specified
            if column_selection is None:
                column_selection = self._smart_column_selection(
                    relationship_graph, target_datasets
                )

            # Create unified query
            query = UnifiedQuery(
                query_id=str(uuid.uuid4()),
                primary_dataset=source_dataset,
                datasets=all_datasets,
                joins=joins,
                selected_columns=column_selection,
                filters={},  # To be filled by user
                aggregations={},  # To be filled by user
                limit=10000  # Default limit
            )

            self.logger.info(f"Created unified query with {len(joins)} joins across {len(all_datasets)} datasets")

            return query

        except Exception as e:
            error_msg = f"Unified query creation failed: {str(e)}"
            self.logger.error(error_msg)
            raise ScoutDiscoveryError(error_msg) from e

    def execute_unified_query(self,
                            unified_query: UnifiedQuery,
                            sample_size: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Execute unified query and return results

        Args:
            unified_query: UnifiedQuery object to execute
            sample_size: Sample size per dataset for testing

        Returns:
            Dictionary mapping dataset IDs to DataFrames
        """
        try:
            self.logger.info(f"Executing unified query {unified_query.query_id}")

            results = {}

            # Execute individual dataset queries
            for dataset_id in unified_query.datasets:
                self.logger.info(f"Fetching data from {dataset_id}")

                # Get column selection for this dataset
                selected_columns = unified_query.selected_columns.get(dataset_id, [])

                # Build SoQL query
                query_builder = self.scout.create_query_builder()

                if selected_columns:
                    query_builder = query_builder.select(*selected_columns)

                # Apply filters if specified
                if dataset_id in unified_query.filters:
                    filters = unified_query.filters[dataset_id]
                    for filter_condition in filters.get('conditions', []):
                        query_builder = query_builder.where(filter_condition)

                # Apply date range if specified
                if unified_query.date_range and 'date_column' in unified_query.filters.get(dataset_id, {}):
                    date_col = unified_query.filters[dataset_id]['date_column']
                    start_date, end_date = unified_query.date_range
                    query_builder = query_builder.where_date_range(date_col, start_date, end_date)

                # Apply limit
                query_builder = query_builder.limit(min(sample_size, unified_query.limit or sample_size))

                # Download data
                try:
                    df = self.scout.download_dataset_sample(
                        dataset_id,
                        query_builder=query_builder
                    )
                    results[dataset_id] = df
                    self.logger.info(f"Retrieved {len(df)} rows from {dataset_id}")

                except Exception as e:
                    self.logger.error(f"Failed to fetch data from {dataset_id}: {str(e)}")
                    results[dataset_id] = pd.DataFrame()  # Empty DataFrame as fallback

            # Log execution summary
            total_rows = sum(len(df) for df in results.values())
            self.logger.info(f"Query execution completed: {total_rows} total rows across {len(results)} datasets")

            return results

        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            self.logger.error(error_msg)
            raise ScoutDiscoveryError(error_msg) from e

    def _analyze_dataset_schema(self, dataset_id: str, dataset_name: str = None) -> DatasetSchema:
        """Analyze dataset schema and cache results"""
        if dataset_id in self.schema_cache:
            return self.schema_cache[dataset_id]

        try:
            # Download sample data for analysis
            sample_df = self.scout.download_dataset_sample(dataset_id, sample_size=500)

            if sample_df.empty:
                raise ValidationError(f"Dataset {dataset_id} is empty")

            # Analyze each column
            columns = []
            for col_name in sample_df.columns:
                column_metadata = self.column_analyzer.analyze_column(
                    sample_df[col_name], dataset_id, dataset_name or dataset_id
                )
                columns.append(column_metadata)

            # Get quality score if available
            quality_score = 75  # Default
            if dataset_id in self.scout.results.get('quality_assessments', {}):
                quality_score = self.scout.results['quality_assessments'][dataset_id]['overall_scores']['total_score']

            schema = DatasetSchema(
                dataset_id=dataset_id,
                dataset_name=dataset_name or dataset_id,
                columns=columns,
                quality_score=quality_score,
                total_records=len(sample_df)
            )

            self.schema_cache[dataset_id] = schema
            return schema

        except Exception as e:
            self.logger.error(f"Schema analysis failed for {dataset_id}: {str(e)}")
            raise ValidationError(f"Schema analysis failed: {str(e)}") from e

    def _extract_search_terms_from_schema(self, schema: DatasetSchema) -> List[str]:
        """Extract search terms from dataset schema"""
        terms = set()

        # Add semantic tags from columns
        for column in schema.columns:
            terms.update(column.semantic_tags)

        # Add column names (cleaned)
        for column in schema.columns:
            # Clean column name and extract meaningful terms
            cleaned_name = column.name.lower().replace('_', ' ').replace('-', ' ')
            words = [word for word in cleaned_name.split() if len(word) > 2]
            terms.update(words)

        # Remove common/generic terms
        exclude_terms = {'text', 'data', 'value', 'field', 'column', 'number', 'count', 'total', 'the', 'and', 'or'}
        terms = terms - exclude_terms

        return list(terms)[:10]  # Limit to top 10 terms

    def _find_cross_dataset_relationships(self, source_schema: DatasetSchema,
                                        target_schema: DatasetSchema) -> List[ColumnRelationship]:
        """Find relationships between columns in two datasets"""
        relationships = []

        for source_column in source_schema.columns:
            column_relationships = self.relationship_mapper.find_relationships(
                source_column,
                target_schema.columns,
                min_confidence=self.min_relationship_confidence
            )
            relationships.extend(column_relationships)

        return relationships

    def _generate_joins(self, relationship_graph: RelationshipGraph,
                       target_datasets: List[str], strategy: str) -> List[QueryJoin]:
        """Generate join operations based on relationships"""
        joins = []

        for dataset_id in target_datasets:
            if dataset_id not in relationship_graph.related_datasets:
                continue

            relationships = relationship_graph.related_datasets[dataset_id]

            # Select best relationship for joining based on strategy
            if strategy == "best_match":
                # Use highest confidence relationship with join potential
                best_rel = max(relationships,
                             key=lambda r: r.confidence_score * r.join_potential,
                             default=None)
                if best_rel and best_rel.join_potential > 0.5:
                    joins.append(self._create_join_from_relationship(best_rel))

            elif strategy == "high_confidence":
                # Use relationships with high confidence
                for rel in relationships:
                    if rel.confidence_score > 0.7 and rel.join_potential > 0.5:
                        joins.append(self._create_join_from_relationship(rel))

            elif strategy == "all":
                # Use all viable relationships
                for rel in relationships:
                    if rel.join_potential > 0.4:
                        joins.append(self._create_join_from_relationship(rel))

        return joins

    def _create_join_from_relationship(self, relationship: ColumnRelationship) -> QueryJoin:
        """Create QueryJoin from ColumnRelationship"""
        # Determine join type based on relationship
        if relationship.relationship_type == RelationshipType.EXACT_MATCH:
            join_type = "inner"
        elif relationship.relationship_type in [RelationshipType.REFERENCE, RelationshipType.HIERARCHICAL]:
            join_type = "left"
        else:
            join_type = "inner"  # Default

        return QueryJoin(
            left_dataset=relationship.source_column.dataset_id,
            right_dataset=relationship.target_column.dataset_id,
            left_column=relationship.source_column.name,
            right_column=relationship.target_column.name,
            join_type=join_type,
            relationship_type=relationship.relationship_type.value,
            confidence=relationship.confidence_score
        )

    def _smart_column_selection(self, relationship_graph: RelationshipGraph,
                               target_datasets: List[str]) -> Dict[str, List[str]]:
        """Smart selection of columns to include in unified query"""
        selection = {}

        # For source dataset, include key columns and high-value columns
        source_schema = self.schema_cache.get(relationship_graph.source_dataset)
        if source_schema:
            source_columns = []
            for column in source_schema.columns:
                # Always include identifiers, dates, and categorical data
                if ('identifier' in column.semantic_tags or
                    'temporal' in column.semantic_tags or
                    'categorical' in column.semantic_tags or
                    column.data_type in [ColumnType.IDENTIFIER, ColumnType.DATE, ColumnType.DATETIME]):
                    source_columns.append(column.name)
                # Include columns with low null percentage
                elif column.null_percentage < 20:
                    source_columns.append(column.name)

            selection[relationship_graph.source_dataset] = source_columns[:10]  # Limit columns

        # For target datasets, include columns involved in relationships + key columns
        for dataset_id in target_datasets:
            if dataset_id not in relationship_graph.related_datasets:
                continue

            target_columns = set()

            # Add columns involved in relationships
            for relationship in relationship_graph.related_datasets[dataset_id]:
                target_columns.add(relationship.target_column.name)

            # Add key columns from schema
            target_schema = self.schema_cache.get(dataset_id)
            if target_schema:
                for column in target_schema.columns:
                    if ('identifier' in column.semantic_tags or
                        'temporal' in column.semantic_tags or
                        column.data_type in [ColumnType.IDENTIFIER, ColumnType.DATE]):
                        target_columns.add(column.name)

            selection[dataset_id] = list(target_columns)[:8]  # Limit columns

        return selection

    def save_workflow_results(self,
                            relationship_graph: RelationshipGraph,
                            unified_query: UnifiedQuery,
                            output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save workflow results to files"""
        try:
            output_path = Path(output_dir) if output_dir else Path("workflow_results")
            output_path.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            files_created = {}

            # Save relationship graph
            graph_file = output_path / f"relationship_graph_{timestamp}.json"
            with open(graph_file, 'w') as f:
                json.dump(relationship_graph.to_dict(), f, indent=2, default=str)
            files_created['relationship_graph'] = str(graph_file)

            # Save unified query
            query_file = output_path / f"unified_query_{timestamp}.json"
            with open(query_file, 'w') as f:
                f.write(unified_query.to_json())
            files_created['unified_query'] = str(query_file)

            # Save human-readable summary
            summary_file = output_path / f"workflow_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write(self._generate_workflow_summary(relationship_graph, unified_query))
            files_created['summary'] = str(summary_file)

            self.logger.info(f"Workflow results saved to {output_path}")
            return files_created

        except Exception as e:
            self.logger.error(f"Failed to save workflow results: {str(e)}")
            return {}

    def _generate_workflow_summary(self, relationship_graph: RelationshipGraph,
                                 unified_query: UnifiedQuery) -> str:
        """Generate human-readable workflow summary"""
        summary = []
        summary.append("MULTI-DATASET WORKFLOW SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Generated: {datetime.now().isoformat()}")
        summary.append("")

        # Source dataset info
        summary.append(f"Source Dataset: {relationship_graph.source_dataset}")
        summary.append(f"Related Datasets Found: {len(relationship_graph.related_datasets)}")
        summary.append("")

        # Relationship summary
        summary.append("RELATIONSHIP SUMMARY:")
        summary.append("-" * 30)
        for dataset_id, rel_summary in relationship_graph.relationship_summary.items():
            summary.append(f"{dataset_id}:")
            for rel_type, count in rel_summary.items():
                summary.append(f"  {rel_type}: {count} relationships")
        summary.append("")

        # Query info
        summary.append("UNIFIED QUERY:")
        summary.append("-" * 20)
        summary.append(f"Query ID: {unified_query.query_id}")
        summary.append(f"Primary Dataset: {unified_query.primary_dataset}")
        summary.append(f"Total Datasets: {len(unified_query.datasets)}")
        summary.append(f"Joins Configured: {len(unified_query.joins)}")
        summary.append("")

        # Join details
        if unified_query.joins:
            summary.append("JOIN OPERATIONS:")
            summary.append("-" * 20)
            for i, join in enumerate(unified_query.joins, 1):
                summary.append(f"{i}. {join.left_dataset}.{join.left_column} -> {join.right_dataset}.{join.right_column}")
                summary.append(f"   Type: {join.join_type}, Confidence: {join.confidence:.3f}")

        return "\n".join(summary)