"""
Data Quality Assessment Module

Implements comprehensive data quality assessment based on Scout methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .exceptions import DataQualityError, ValidationError


class DataQualityAssessor:
    """
    Comprehensive data quality assessment class following Scout's methodology.

    Evaluates datasets across multiple dimensions:
    - Completeness: Missing data assessment
    - Consistency: Data type and format consistency
    - Accuracy: Outlier detection and validation
    - Timeliness: Update frequency and freshness
    - Usability: Structure and accessibility
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.assessment_cache = {}

    def assess_dataset_quality(self, dataset_id: str, df: pd.DataFrame,
                             metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment on a dataset.

        Args:
            dataset_id: Unique identifier for the dataset
            df: DataFrame containing the dataset
            metadata: Optional metadata about the dataset

        Returns:
            Dictionary containing quality assessment results

        Raises:
            DataQualityError: If assessment fails
            ValidationError: If input data is invalid
        """
        try:
            if df.empty:
                raise ValidationError(f"Dataset {dataset_id} is empty")

            self.logger.info(f"Starting quality assessment for dataset {dataset_id}")

            assessment = {
                'dataset_id': dataset_id,
                'assessment_timestamp': datetime.now().isoformat(),
                'basic_stats': self._get_basic_statistics(df),
                'completeness': self._assess_completeness(df),
                'consistency': self._assess_consistency(df),
                'accuracy': self._assess_accuracy(df),
                'timeliness': self._assess_timeliness(df, metadata),
                'usability': self._assess_usability(df),
                'overall_scores': {}
            }

            # Calculate overall quality scores
            assessment['overall_scores'] = self._calculate_overall_scores(assessment)

            # Cache the assessment
            self.assessment_cache[dataset_id] = assessment

            self.logger.info(f"Quality assessment completed for {dataset_id}. "
                           f"Overall score: {assessment['overall_scores']['total_score']:.2f}")

            return assessment

        except Exception as e:
            error_msg = f"Quality assessment failed for dataset {dataset_id}: {str(e)}"
            self.logger.error(error_msg)
            raise DataQualityError(error_msg) from e

    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics"""
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'dtypes_distribution': df.dtypes.value_counts().to_dict(),
            'column_names': list(df.columns),
            'size_category': self._categorize_dataset_size(len(df), len(df.columns))
        }

    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess dataset completeness"""
        missing_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        total_missing = missing_counts.sum()

        # Identify columns by missing data severity
        complete_columns = missing_counts[missing_counts == 0].index.tolist()
        partial_missing = missing_counts[(missing_counts > 0) & (missing_counts < len(df))].to_dict()
        empty_columns = missing_counts[missing_counts == len(df)].index.tolist()

        # Calculate completeness score (0-100)
        completeness_score = max(0, 100 - (total_missing / total_cells * 100))

        return {
            'total_missing_cells': int(total_missing),
            'missing_percentage': (total_missing / total_cells) * 100,
            'complete_columns': complete_columns,
            'partially_missing_columns': partial_missing,
            'empty_columns': empty_columns,
            'completeness_score': completeness_score,
            'columns_with_high_completeness': [
                col for col in df.columns
                if (df[col].isnull().sum() / len(df)) < 0.05
            ]
        }

    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data type and format consistency"""
        consistency_issues = {}
        potential_improvements = {}

        for col in df.columns:
            col_issues = []
            col_improvements = []

            # Check for mixed data types in object columns
            if df[col].dtype == 'object' and not df[col].empty:
                sample_values = df[col].dropna().head(100)

                # Check for potential numeric columns stored as text
                numeric_pattern = sample_values.astype(str).str.match(r'^-?\d+\.?\d*$')
                if numeric_pattern.sum() > len(sample_values) * 0.8:
                    col_improvements.append('Consider converting to numeric type')

                # Check for potential date columns
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',
                    r'\d{2}/\d{2}/\d{4}',
                    r'\d{2}-\d{2}-\d{4}'
                ]
                for pattern in date_patterns:
                    if sample_values.astype(str).str.match(pattern).any():
                        col_improvements.append('Consider converting to datetime type')
                        break

                # Check for inconsistent formatting
                unique_formats = sample_values.astype(str).str.len().value_counts()
                if len(unique_formats) > len(sample_values) * 0.3:
                    col_issues.append('Inconsistent string formatting detected')

            if col_issues:
                consistency_issues[col] = col_issues
            if col_improvements:
                potential_improvements[col] = col_improvements

        # Calculate consistency score
        total_columns = len(df.columns)
        problematic_columns = len(consistency_issues)
        consistency_score = max(0, 100 - (problematic_columns / total_columns * 50))

        return {
            'consistency_score': consistency_score,
            'consistency_issues': consistency_issues,
            'potential_improvements': potential_improvements,
            'dtype_distribution': df.dtypes.value_counts().to_dict(),
            'columns_needing_attention': list(consistency_issues.keys())
        }

    def _assess_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data accuracy through outlier detection"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        accuracy_flags = []

        for col in numeric_cols:
            if not df[col].empty and df[col].notna().sum() > 10:  # Need sufficient data
                # IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_percentage = len(outliers) / len(df[col].dropna()) * 100

                # Statistical summary
                stats = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': outlier_percentage,
                    'min_value': float(df[col].min()),
                    'max_value': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                }

                outlier_analysis[col] = stats

                # Flag potential accuracy issues
                if outlier_percentage > 10:
                    accuracy_flags.append(f"High outlier percentage in {col}: {outlier_percentage:.1f}%")

                if df[col].min() < 0 and col.lower() in ['age', 'count', 'amount', 'price']:
                    accuracy_flags.append(f"Potentially invalid negative values in {col}")

        # Calculate accuracy score
        avg_outlier_percentage = np.mean([
            analysis['outlier_percentage']
            for analysis in outlier_analysis.values()
        ]) if outlier_analysis else 0

        accuracy_score = max(0, 100 - avg_outlier_percentage)

        return {
            'accuracy_score': accuracy_score,
            'outlier_analysis': outlier_analysis,
            'accuracy_flags': accuracy_flags,
            'numeric_columns_analyzed': len(numeric_cols),
            'average_outlier_percentage': avg_outlier_percentage
        }

    def _assess_timeliness(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Assess data timeliness and freshness"""
        timeliness_info = {
            'timeliness_score': 75,  # Default score when no metadata available
            'last_updated': None,
            'update_frequency': None,
            'freshness_assessment': 'Unknown - no metadata provided'
        }

        if metadata:
            # Extract update information from metadata
            if 'updatedAt' in metadata:
                try:
                    last_updated = pd.to_datetime(metadata['updatedAt'])
                    days_since_update = (datetime.now() - last_updated).days

                    timeliness_info.update({
                        'last_updated': last_updated.isoformat(),
                        'days_since_update': days_since_update,
                    })

                    # Score based on recency (assuming monthly updates are good)
                    if days_since_update <= 7:
                        timeliness_score = 100
                        freshness = 'Very Fresh'
                    elif days_since_update <= 30:
                        timeliness_score = 90
                        freshness = 'Fresh'
                    elif days_since_update <= 90:
                        timeliness_score = 70
                        freshness = 'Moderately Fresh'
                    elif days_since_update <= 365:
                        timeliness_score = 50
                        freshness = 'Stale'
                    else:
                        timeliness_score = 25
                        freshness = 'Very Stale'

                    timeliness_info.update({
                        'timeliness_score': timeliness_score,
                        'freshness_assessment': freshness
                    })

                except Exception:
                    pass

        # Look for date columns to assess temporal coverage
        date_columns = []
        for col in df.columns:
            if df[col].dtype.name.startswith('datetime'):
                date_columns.append(col)
            elif df[col].dtype == 'object':
                # Try to identify date-like strings
                sample = df[col].dropna().head(10).astype(str)
                if sample.str.match(r'\d{4}-\d{2}-\d{2}').sum() > 5:
                    date_columns.append(col)

        timeliness_info['date_columns_found'] = date_columns

        return timeliness_info

    def _assess_usability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess dataset usability"""
        usability_factors = {
            'column_naming': self._assess_column_naming(df),
            'structure_clarity': self._assess_structure_clarity(df),
            'accessibility': self._assess_accessibility(df)
        }

        # Calculate overall usability score
        naming_score = usability_factors['column_naming']['score']
        structure_score = usability_factors['structure_clarity']['score']
        accessibility_score = usability_factors['accessibility']['score']

        usability_score = (naming_score + structure_score + accessibility_score) / 3

        return {
            'usability_score': usability_score,
            'factors': usability_factors,
            'recommendations': self._generate_usability_recommendations(usability_factors)
        }

    def _assess_column_naming(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess quality of column naming"""
        total_columns = len(df.columns)
        issues = []
        good_practices = 0

        for col in df.columns:
            col_str = str(col)

            # Check for good practices
            if col_str.islower() or col_str.replace('_', '').islower():
                good_practices += 1
            elif ' ' not in col_str and col_str.replace('_', '').replace('-', '').isalnum():
                good_practices += 1

            # Check for issues
            if ' ' in col_str:
                issues.append(f"Column '{col}' contains spaces")
            if col_str.startswith(('_', '-')) or col_str.endswith(('_', '-')):
                issues.append(f"Column '{col}' has leading/trailing separators")
            if len(col_str) > 50:
                issues.append(f"Column '{col}' has very long name")
            if not col_str.replace('_', '').replace('-', '').replace(' ', '').isalnum():
                issues.append(f"Column '{col}' contains special characters")

        score = (good_practices / total_columns) * 100 if total_columns > 0 else 0

        return {
            'score': score,
            'total_columns': total_columns,
            'good_naming_count': good_practices,
            'issues': issues[:10]  # Limit to first 10 issues
        }

    def _assess_structure_clarity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess structural clarity of the dataset"""
        score = 100
        issues = []

        # Check for duplicate columns
        if df.columns.duplicated().any():
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            issues.append(f"Duplicate column names found: {duplicate_cols}")
            score -= 20

        # Check for excessive width (too many columns)
        if len(df.columns) > 100:
            issues.append(f"Very wide dataset with {len(df.columns)} columns")
            score -= 10

        # Check for mixed data types in columns (already covered in consistency)
        # This is a structural usability issue

        return {
            'score': max(0, score),
            'issues': issues,
            'column_count': len(df.columns),
            'row_count': len(df)
        }

    def _assess_accessibility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess how accessible/usable the data structure is"""
        score = 100
        accessibility_features = []

        # Check if dataset size is manageable
        if len(df) < 1000000 and len(df.columns) < 50:
            accessibility_features.append("Manageable size for analysis")
        elif len(df) > 10000000:
            score -= 15

        # Check for reasonable data types
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        if 0.2 <= numeric_ratio <= 0.8:
            accessibility_features.append("Good mix of data types")

        return {
            'score': score,
            'features': accessibility_features,
            'size_assessment': self._categorize_dataset_size(len(df), len(df.columns))
        }

    def _categorize_dataset_size(self, rows: int, columns: int) -> str:
        """Categorize dataset size for usability assessment"""
        if rows < 1000 and columns < 10:
            return "Small"
        elif rows < 100000 and columns < 50:
            return "Medium"
        elif rows < 1000000 and columns < 100:
            return "Large"
        else:
            return "Very Large"

    def _generate_usability_recommendations(self, factors: Dict) -> List[str]:
        """Generate recommendations for improving usability"""
        recommendations = []

        # Column naming recommendations
        naming = factors['column_naming']
        if naming['score'] < 70:
            recommendations.append("Consider standardizing column names (lowercase, underscores)")

        # Structure recommendations
        structure = factors['structure_clarity']
        if structure['issues']:
            recommendations.append("Address structural issues: " + "; ".join(structure['issues'][:3]))

        return recommendations

    def _calculate_overall_scores(self, assessment: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality scores"""
        # Extract individual scores
        completeness = assessment['completeness']['completeness_score']
        consistency = assessment['consistency']['consistency_score']
        accuracy = assessment['accuracy']['accuracy_score']
        timeliness = assessment['timeliness']['timeliness_score']
        usability = assessment['usability']['usability_score']

        # Weighted average (adjust weights as needed)
        weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'accuracy': 0.20,
            'timeliness': 0.15,
            'usability': 0.20
        }

        total_score = (
            completeness * weights['completeness'] +
            consistency * weights['consistency'] +
            accuracy * weights['accuracy'] +
            timeliness * weights['timeliness'] +
            usability * weights['usability']
        )

        # Quality grade
        if total_score >= 90:
            grade = 'A'
        elif total_score >= 80:
            grade = 'B'
        elif total_score >= 70:
            grade = 'C'
        elif total_score >= 60:
            grade = 'D'
        else:
            grade = 'F'

        return {
            'completeness_score': completeness,
            'consistency_score': consistency,
            'accuracy_score': accuracy,
            'timeliness_score': timeliness,
            'usability_score': usability,
            'total_score': total_score,
            'grade': grade,
            'weights_used': weights
        }

    def generate_quality_report(self, assessments: Dict[str, Dict]) -> pd.DataFrame:
        """Generate a summary quality report from multiple assessments"""
        try:
            reports = []

            for dataset_id, assessment in assessments.items():
                if 'error' in assessment or 'overall_scores' not in assessment:
                    continue

                scores = assessment['overall_scores']
                basic_stats = assessment['basic_stats']

                report = {
                    'dataset_id': dataset_id,
                    'total_score': round(scores['total_score'], 2),
                    'grade': scores['grade'],
                    'completeness': round(scores['completeness_score'], 2),
                    'consistency': round(scores['consistency_score'], 2),
                    'accuracy': round(scores['accuracy_score'], 2),
                    'timeliness': round(scores['timeliness_score'], 2),
                    'usability': round(scores['usability_score'], 2),
                    'row_count': basic_stats['row_count'],
                    'column_count': basic_stats['column_count'],
                    'size_category': basic_stats['size_category']
                }
                reports.append(report)

            if not reports:
                return pd.DataFrame()

            df = pd.DataFrame(reports)
            return df.sort_values('total_score', ascending=False)

        except Exception as e:
            self.logger.error(f"Failed to generate quality report: {str(e)}")
            raise DataQualityError(f"Report generation failed: {str(e)}") from e