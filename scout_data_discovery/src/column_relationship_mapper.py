"""
Column Relationship Mapper

Advanced system for analyzing and mapping column relationships across multiple datasets
using Scout methodology for intelligent data discovery and schema alignment.
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, Counter
import difflib

from .exceptions import ValidationError, ConfigurationError


class ColumnType(Enum):
    """Enhanced column type classification"""
    TEXT = "text"
    NUMERIC = "numeric"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    LOCATION = "location"
    COORDINATE = "coordinate"
    ADDRESS = "address"
    PHONE = "phone"
    EMAIL = "email"
    URL = "url"
    JSON = "json"
    UNKNOWN = "unknown"


class RelationshipType(Enum):
    """Types of relationships between columns"""
    EXACT_MATCH = "exact_match"          # Same column name and type
    SEMANTIC_MATCH = "semantic_match"     # Similar meaning, different names
    TYPE_COMPATIBLE = "type_compatible"   # Same data type
    HIERARCHICAL = "hierarchical"         # Parent-child relationship
    TEMPORAL = "temporal"                 # Time-based relationship
    GEOGRAPHIC = "geographic"             # Location-based relationship
    REFERENCE = "reference"               # Foreign key relationship
    DERIVED = "derived"                   # One can be calculated from other
    COMPLEMENTARY = "complementary"       # Together provide complete picture


@dataclass
class ColumnMetadata:
    """Comprehensive column metadata"""
    name: str
    dataset_id: str
    dataset_name: str
    data_type: ColumnType
    sample_values: List[Any]
    null_percentage: float
    unique_count: int
    total_count: int
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_length: Optional[float] = None
    common_patterns: List[str] = None
    semantic_tags: List[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        if self.common_patterns is None:
            self.common_patterns = []
        if self.semantic_tags is None:
            self.semantic_tags = []


@dataclass
class ColumnRelationship:
    """Relationship between two columns"""
    source_column: ColumnMetadata
    target_column: ColumnMetadata
    relationship_type: RelationshipType
    confidence_score: float
    compatibility_score: float
    join_potential: float
    semantic_similarity: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'source_dataset': self.source_column.dataset_id,
            'source_column': self.source_column.name,
            'target_dataset': self.target_column.dataset_id,
            'target_column': self.target_column.name,
            'relationship_type': self.relationship_type.value,
            'confidence_score': self.confidence_score,
            'compatibility_score': self.compatibility_score,
            'join_potential': self.join_potential,
            'semantic_similarity': self.semantic_similarity,
            'notes': self.notes
        }


class ColumnAnalyzer:
    """Advanced column analysis and type detection"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Pattern definitions for column type detection
        self.patterns = {
            ColumnType.DATE: [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}-\d{2}-\d{4}',
            ],
            ColumnType.DATETIME: [
                r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            ],
            ColumnType.PHONE: [
                r'\(\d{3}\) \d{3}-\d{4}',
                r'\d{3}-\d{3}-\d{4}',
                r'\d{10}',
            ],
            ColumnType.EMAIL: [
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            ],
            ColumnType.URL: [
                r'^https?://',
                r'www\.',
            ],
            ColumnType.COORDINATE: [
                r'^-?\d+\.\d+$',  # Decimal degrees
                r'^\d+°\d+\'\d+"[NSEW]$',  # DMS format
            ],
            ColumnType.ADDRESS: [
                r'\d+\s+\w+\s+(st|street|ave|avenue|rd|road|blvd|boulevard)',
            ]
        }

        # Semantic keyword mapping
        self.semantic_keywords = {
            'identifier': ['id', 'key', 'number', 'code', 'reference'],
            'location': ['address', 'location', 'place', 'street', 'ave', 'road', 'borough'],
            'geographic': ['lat', 'latitude', 'lng', 'longitude', 'coord', 'zip', 'postal'],
            'temporal': ['date', 'time', 'created', 'updated', 'modified', 'year', 'month'],
            'categorical': ['type', 'category', 'class', 'status', 'level', 'grade'],
            'numeric': ['count', 'amount', 'total', 'sum', 'avg', 'number', 'quantity'],
            'descriptive': ['name', 'title', 'description', 'comment', 'note', 'detail']
        }

    def analyze_column(self, column: pd.Series, dataset_id: str, dataset_name: str) -> ColumnMetadata:
        """
        Comprehensive column analysis

        Args:
            column: Pandas Series representing the column
            dataset_id: Dataset identifier
            dataset_name: Human-readable dataset name

        Returns:
            ColumnMetadata with comprehensive analysis
        """
        try:
            # Basic statistics
            total_count = len(column)
            null_count = column.isnull().sum()
            null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
            unique_count = column.nunique()

            # Sample values (non-null)
            non_null_values = column.dropna()
            sample_size = min(20, len(non_null_values))
            sample_values = non_null_values.head(sample_size).tolist()

            # Detect column type
            data_type = self._detect_column_type(column)

            # Additional statistics based on type
            min_value = None
            max_value = None
            avg_length = None

            if data_type in [ColumnType.NUMERIC, ColumnType.INTEGER, ColumnType.FLOAT]:
                if not non_null_values.empty:
                    min_value = float(non_null_values.min())
                    max_value = float(non_null_values.max())
            elif data_type in [ColumnType.TEXT, ColumnType.CATEGORICAL]:
                if not non_null_values.empty:
                    lengths = non_null_values.astype(str).str.len()
                    avg_length = float(lengths.mean())

            # Detect patterns
            common_patterns = self._detect_patterns(column)

            # Generate semantic tags
            semantic_tags = self._generate_semantic_tags(column.name, sample_values, data_type)

            return ColumnMetadata(
                name=column.name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                data_type=data_type,
                sample_values=sample_values,
                null_percentage=null_percentage,
                unique_count=unique_count,
                total_count=total_count,
                min_value=min_value,
                max_value=max_value,
                avg_length=avg_length,
                common_patterns=common_patterns,
                semantic_tags=semantic_tags
            )

        except Exception as e:
            self.logger.error(f"Column analysis failed for {column.name}: {str(e)}")
            raise ValidationError(f"Column analysis failed: {str(e)}") from e

    def _detect_column_type(self, column: pd.Series) -> ColumnType:
        """Detect column type using multiple heuristics"""

        # Get non-null values as strings
        non_null_values = column.dropna()
        if non_null_values.empty:
            return ColumnType.UNKNOWN

        str_values = non_null_values.astype(str)
        sample_values = str_values.head(100)  # Use larger sample for type detection

        # Check pandas dtype first
        if pd.api.types.is_integer_dtype(column):
            return ColumnType.INTEGER
        elif pd.api.types.is_float_dtype(column):
            return ColumnType.FLOAT
        elif pd.api.types.is_bool_dtype(column):
            return ColumnType.BOOLEAN
        elif pd.api.types.is_datetime64_any_dtype(column):
            return ColumnType.DATETIME

        # Pattern-based detection
        for col_type, patterns in self.patterns.items():
            match_count = 0
            for pattern in patterns:
                matches = sample_values.str.match(pattern, case=False, na=False).sum()
                match_count += matches

            # If majority of values match pattern
            if match_count / len(sample_values) > 0.7:
                return col_type

        # Numeric detection for string columns
        numeric_count = pd.to_numeric(sample_values, errors='coerce').notna().sum()
        if numeric_count / len(sample_values) > 0.8:
            # Check if integers
            try:
                numeric_values = pd.to_numeric(sample_values, errors='coerce').dropna()
                if (numeric_values == numeric_values.astype(int)).all():
                    return ColumnType.INTEGER
                else:
                    return ColumnType.FLOAT
            except:
                return ColumnType.NUMERIC

        # Categorical detection
        unique_ratio = column.nunique() / len(column)
        if unique_ratio < 0.05 and column.nunique() < 50:  # Low cardinality
            return ColumnType.CATEGORICAL

        # Default to text
        return ColumnType.TEXT

    def _detect_patterns(self, column: pd.Series, max_patterns: int = 5) -> List[str]:
        """Detect common patterns in column values"""
        try:
            non_null_values = column.dropna().astype(str)
            if non_null_values.empty:
                return []

            # Pattern frequency analysis
            pattern_counts = Counter()

            for value in non_null_values.head(100):  # Sample for performance
                # Generate pattern by replacing digits and letters
                pattern = re.sub(r'\d', 'N', str(value))
                pattern = re.sub(r'[a-zA-Z]', 'A', pattern)
                pattern_counts[pattern] += 1

            # Return most common patterns
            return [pattern for pattern, count in pattern_counts.most_common(max_patterns)]

        except Exception as e:
            self.logger.warning(f"Pattern detection failed for column {column.name}: {str(e)}")
            return []

    def _generate_semantic_tags(self, column_name: str, sample_values: List[Any],
                               data_type: ColumnType) -> List[str]:
        """Generate semantic tags based on column name and content"""
        tags = set()

        # Name-based tagging
        column_lower = column_name.lower()
        for category, keywords in self.semantic_keywords.items():
            if any(keyword in column_lower for keyword in keywords):
                tags.add(category)

        # Type-based tagging
        tags.add(data_type.value)

        # Content-based tagging
        if sample_values:
            # Check for ID-like values
            if all(isinstance(v, (int, str)) and str(v).isdigit() for v in sample_values[:5]):
                tags.add('identifier')

            # Check for location indicators
            location_indicators = ['ny', 'nyc', 'manhattan', 'brooklyn', 'queens', 'bronx', 'staten']
            if any(any(indicator in str(v).lower() for indicator in location_indicators)
                   for v in sample_values[:10]):
                tags.add('nyc_specific')

        return list(tags)


class RelationshipMapper:
    """Maps relationships between columns across datasets"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.column_analyzer = ColumnAnalyzer(logger)

        # Relationship scoring weights
        self.weights = {
            'name_similarity': 0.3,
            'type_compatibility': 0.25,
            'semantic_similarity': 0.25,
            'value_compatibility': 0.2
        }

    def find_relationships(self, source_metadata: ColumnMetadata,
                          target_columns: List[ColumnMetadata],
                          min_confidence: float = 0.3) -> List[ColumnRelationship]:
        """
        Find relationships between a source column and target columns

        Args:
            source_metadata: Source column metadata
            target_columns: List of potential target columns
            min_confidence: Minimum confidence threshold

        Returns:
            List of ColumnRelationship objects above threshold
        """
        relationships = []

        for target_metadata in target_columns:
            # Skip self-comparison
            if (source_metadata.dataset_id == target_metadata.dataset_id and
                source_metadata.name == target_metadata.name):
                continue

            relationship = self._analyze_relationship(source_metadata, target_metadata)

            if relationship.confidence_score >= min_confidence:
                relationships.append(relationship)

        # Sort by confidence score
        relationships.sort(key=lambda r: r.confidence_score, reverse=True)

        return relationships

    def _analyze_relationship(self, source: ColumnMetadata, target: ColumnMetadata) -> ColumnRelationship:
        """Analyze relationship between two columns"""

        # Calculate different similarity scores
        name_similarity = self._calculate_name_similarity(source.name, target.name)
        type_compatibility = self._calculate_type_compatibility(source.data_type, target.data_type)
        semantic_similarity = self._calculate_semantic_similarity(source.semantic_tags, target.semantic_tags)
        value_compatibility = self._calculate_value_compatibility(source, target)

        # Overall confidence score
        confidence_score = (
            name_similarity * self.weights['name_similarity'] +
            type_compatibility * self.weights['type_compatibility'] +
            semantic_similarity * self.weights['semantic_similarity'] +
            value_compatibility * self.weights['value_compatibility']
        )

        # Determine relationship type
        relationship_type = self._determine_relationship_type(
            source, target, name_similarity, type_compatibility, semantic_similarity
        )

        # Calculate join potential
        join_potential = self._calculate_join_potential(source, target)

        # Compatibility score (for filtering/aggregation potential)
        compatibility_score = max(type_compatibility, semantic_similarity)

        # Generate notes
        notes = self._generate_relationship_notes(source, target, relationship_type)

        return ColumnRelationship(
            source_column=source,
            target_column=target,
            relationship_type=relationship_type,
            confidence_score=confidence_score,
            compatibility_score=compatibility_score,
            join_potential=join_potential,
            semantic_similarity=semantic_similarity,
            notes=notes
        )

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between column names"""
        if name1.lower() == name2.lower():
            return 1.0

        # Use difflib for sequence matching
        similarity = difflib.SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

        # Boost similarity for common patterns
        common_patterns = [
            ('_id', '_key'), ('date', 'time'), ('addr', 'address'),
            ('lat', 'latitude'), ('lng', 'longitude'), ('desc', 'description')
        ]

        for pattern1, pattern2 in common_patterns:
            if pattern1 in name1.lower() and pattern2 in name2.lower():
                similarity = max(similarity, 0.8)
            elif pattern2 in name1.lower() and pattern1 in name2.lower():
                similarity = max(similarity, 0.8)

        return similarity

    def _calculate_type_compatibility(self, type1: ColumnType, type2: ColumnType) -> float:
        """Calculate type compatibility score"""
        if type1 == type2:
            return 1.0

        # Define compatibility matrix
        compatibility_groups = [
            {ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.NUMERIC},
            {ColumnType.DATE, ColumnType.DATETIME},
            {ColumnType.TEXT, ColumnType.CATEGORICAL},
            {ColumnType.ADDRESS, ColumnType.LOCATION},
            {ColumnType.COORDINATE}
        ]

        for group in compatibility_groups:
            if type1 in group and type2 in group:
                return 0.8

        # Partial compatibility
        if (type1 in [ColumnType.TEXT, ColumnType.CATEGORICAL] and
            type2 in [ColumnType.TEXT, ColumnType.CATEGORICAL]):
            return 0.6

        return 0.1

    def _calculate_semantic_similarity(self, tags1: List[str], tags2: List[str]) -> float:
        """Calculate semantic similarity based on tags"""
        if not tags1 or not tags2:
            return 0.0

        set1, set2 = set(tags1), set(tags2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _calculate_value_compatibility(self, source: ColumnMetadata, target: ColumnMetadata) -> float:
        """Calculate compatibility based on value characteristics"""
        score = 0.0

        # Null percentage similarity
        null_diff = abs(source.null_percentage - target.null_percentage)
        null_score = max(0, 1 - null_diff / 100)
        score += null_score * 0.3

        # Unique count ratio (for categorical/identifier columns)
        if source.total_count > 0 and target.total_count > 0:
            source_unique_ratio = source.unique_count / source.total_count
            target_unique_ratio = target.unique_count / target.total_count
            unique_diff = abs(source_unique_ratio - target_unique_ratio)
            unique_score = max(0, 1 - unique_diff)
            score += unique_score * 0.4

        # Pattern similarity
        if source.common_patterns and target.common_patterns:
            pattern_overlap = len(set(source.common_patterns).intersection(set(target.common_patterns)))
            max_patterns = max(len(source.common_patterns), len(target.common_patterns))
            pattern_score = pattern_overlap / max_patterns if max_patterns > 0 else 0
            score += pattern_score * 0.3

        return min(1.0, score)

    def _determine_relationship_type(self, source: ColumnMetadata, target: ColumnMetadata,
                                   name_sim: float, type_compat: float, semantic_sim: float) -> RelationshipType:
        """Determine the type of relationship between columns"""

        # Exact match
        if name_sim > 0.95 and type_compat > 0.95:
            return RelationshipType.EXACT_MATCH

        # Semantic match
        if semantic_sim > 0.7 and type_compat > 0.7:
            return RelationshipType.SEMANTIC_MATCH

        # Hierarchical relationship
        hierarchical_indicators = [
            ('borough', 'address'), ('category', 'subcategory'),
            ('state', 'city'), ('year', 'date')
        ]

        for parent, child in hierarchical_indicators:
            if ((parent in source.name.lower() and child in target.name.lower()) or
                (child in source.name.lower() and parent in target.name.lower())):
                return RelationshipType.HIERARCHICAL

        # Geographic relationship
        if ('geographic' in source.semantic_tags and 'geographic' in target.semantic_tags):
            return RelationshipType.GEOGRAPHIC

        # Temporal relationship
        if ('temporal' in source.semantic_tags and 'temporal' in target.semantic_tags):
            return RelationshipType.TEMPORAL

        # Reference relationship (ID-like columns)
        if ('identifier' in source.semantic_tags and 'identifier' in target.semantic_tags):
            return RelationshipType.REFERENCE

        # Type compatible
        if type_compat > 0.7:
            return RelationshipType.TYPE_COMPATIBLE

        return RelationshipType.COMPLEMENTARY

    def _calculate_join_potential(self, source: ColumnMetadata, target: ColumnMetadata) -> float:
        """Calculate potential for using these columns in joins"""

        # High join potential for identifiers
        if ('identifier' in source.semantic_tags and 'identifier' in target.semantic_tags):
            return 0.9

        # Medium join potential for categorical data
        if (source.data_type == ColumnType.CATEGORICAL and
            target.data_type == ColumnType.CATEGORICAL):
            return 0.7

        # Geographic join potential
        if ('geographic' in source.semantic_tags and 'geographic' in target.semantic_tags):
            return 0.8

        # Temporal join potential
        if ('temporal' in source.semantic_tags and 'temporal' in target.semantic_tags):
            return 0.6

        # Lower potential for descriptive fields
        return 0.3

    def _generate_relationship_notes(self, source: ColumnMetadata, target: ColumnMetadata,
                                   rel_type: RelationshipType) -> str:
        """Generate human-readable notes about the relationship"""

        notes = []

        if rel_type == RelationshipType.EXACT_MATCH:
            notes.append("Identical column structure - perfect for joins/unions")
        elif rel_type == RelationshipType.SEMANTIC_MATCH:
            notes.append("Similar meaning with compatible data types")
        elif rel_type == RelationshipType.HIERARCHICAL:
            notes.append("Hierarchical relationship - useful for grouping/aggregation")
        elif rel_type == RelationshipType.GEOGRAPHIC:
            notes.append("Geographic relationship - spatial analysis potential")
        elif rel_type == RelationshipType.TEMPORAL:
            notes.append("Time-based relationship - temporal analysis potential")
        elif rel_type == RelationshipType.REFERENCE:
            notes.append("Reference relationship - potential foreign key")

        # Add data quality notes
        if source.null_percentage > 50 or target.null_percentage > 50:
            notes.append("High null percentage - consider data quality")

        if source.unique_count == source.total_count or target.unique_count == target.total_count:
            notes.append("Unique identifier detected")

        return "; ".join(notes)