"""
Data Validation utilities for Scout Data Discovery integration
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Data validation and quality assessment utilities
    """
    
    def __init__(self):
        """Initialize the data validator"""
        self.validation_rules = {
            'completeness': self._check_completeness,
            'consistency': self._check_consistency,
            'accuracy': self._check_accuracy,
            'validity': self._check_validity,
            'uniqueness': self._check_uniqueness
        }
    
    def validate_dataset(self, df: pd.DataFrame, 
                        rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive dataset validation
        
        Args:
            df: DataFrame to validate
            rules: List of validation rules to apply (default: all)
            
        Returns:
            Validation results dictionary
        """
        if rules is None:
            rules = list(self.validation_rules.keys())
        
        results = {
            'overall_score': 0.0,
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_shape': df.shape,
            'rule_results': {},
            'issues': [],
            'recommendations': []
        }
        
        scores = []
        
        for rule in rules:
            if rule in self.validation_rules:
                try:
                    rule_result = self.validation_rules[rule](df)
                    results['rule_results'][rule] = rule_result
                    scores.append(rule_result.get('score', 0.0))
                    
                    # Collect issues and recommendations
                    if 'issues' in rule_result:
                        results['issues'].extend(rule_result['issues'])
                    if 'recommendations' in rule_result:
                        results['recommendations'].extend(rule_result['recommendations'])
                        
                except Exception as e:
                    logger.error(f"Error validating rule {rule}: {e}")
                    results['rule_results'][rule] = {
                        'score': 0.0,
                        'error': str(e)
                    }
                    scores.append(0.0)
        
        # Calculate overall score
        if scores:
            results['overall_score'] = np.mean(scores)
        
        # Add quality level
        results['quality_level'] = self._get_quality_level(results['overall_score'])
        
        return results
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness (missing values)"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells)
        
        column_completeness = {}
        issues = []
        recommendations = []
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_ratio = missing_count / len(df)
            column_completeness[column] = {
                'missing_count': int(missing_count),
                'missing_ratio': float(missing_ratio),
                'completeness': float(1 - missing_ratio)
            }
            
            if missing_ratio > 0.5:
                issues.append(f"Column '{column}' has {missing_ratio:.1%} missing values")
                recommendations.append(f"Consider dropping column '{column}' or improving data collection")
            elif missing_ratio > 0.1:
                recommendations.append(f"Consider data imputation for column '{column}'")
        
        return {
            'score': float(completeness_ratio),
            'overall_completeness': float(completeness_ratio),
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells),
            'column_completeness': column_completeness,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency (format, type consistency)"""
        consistency_scores = []
        issues = []
        recommendations = []
        column_consistency = {}
        
        for column in df.columns:
            column_score = 1.0
            column_issues = []
            
            # Check data type consistency
            if df[column].dtype == 'object':
                # Check for mixed types in object columns
                sample = df[column].dropna().head(100)
                if len(sample) > 0:
                    type_consistency = self._check_type_consistency(sample)
                    if type_consistency < 0.9:
                        column_score *= type_consistency
                        column_issues.append("Mixed data types detected")
                
                # Check format consistency for strings
                format_consistency = self._check_format_consistency(df[column])
                column_score *= format_consistency
                if format_consistency < 0.9:
                    column_issues.append("Inconsistent formats detected")
            
            # Check for obvious inconsistencies
            if self._has_obvious_inconsistencies(df[column]):
                column_score *= 0.7
                column_issues.append("Obvious data inconsistencies found")
            
            column_consistency[column] = {
                'score': float(column_score),
                'issues': column_issues
            }
            
            consistency_scores.append(column_score)
            
            if column_issues:
                issues.extend([f"Column '{column}': {issue}" for issue in column_issues])
                recommendations.append(f"Review and standardize data in column '{column}'")
        
        overall_score = np.mean(consistency_scores) if consistency_scores else 1.0
        
        return {
            'score': float(overall_score),
            'column_consistency': column_consistency,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data accuracy (outliers, unrealistic values)"""
        accuracy_scores = []
        issues = []
        recommendations = []
        column_accuracy = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            column_score = 1.0
            column_issues = []
            
            # Check for outliers using IQR method
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_ratio = len(outliers) / len(df)
            
            if outlier_ratio > 0.1:
                column_score *= (1 - outlier_ratio * 0.5)
                column_issues.append(f"{outlier_ratio:.1%} outliers detected")
            
            # Check for unrealistic values (negative values where they shouldn't be)
            if column.lower() in ['age', 'price', 'count', 'quantity', 'amount']:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    column_score *= 0.8
                    column_issues.append(f"{negative_count} negative values in '{column}'")
            
            column_accuracy[column] = {
                'score': float(column_score),
                'outlier_ratio': float(outlier_ratio),
                'issues': column_issues
            }
            
            accuracy_scores.append(column_score)
            
            if column_issues:
                issues.extend([f"Column '{column}': {issue}" for issue in column_issues])
                recommendations.append(f"Review outliers and extreme values in column '{column}'")
        
        # Check categorical columns for suspicious values
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            suspicious_patterns = self._find_suspicious_categorical_values(df[column])
            if suspicious_patterns:
                issues.extend([f"Column '{column}': {pattern}" for pattern in suspicious_patterns])
                recommendations.append(f"Review categorical values in column '{column}'")
        
        overall_score = np.mean(accuracy_scores) if accuracy_scores else 1.0
        
        return {
            'score': float(overall_score),
            'column_accuracy': column_accuracy,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity (format compliance, business rules)"""
        validity_scores = []
        issues = []
        recommendations = []
        column_validity = {}
        
        for column in df.columns:
            column_score = 1.0
            column_issues = []
            
            # Email validation
            if 'email' in column.lower():
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_emails = df[column].apply(
                    lambda x: not bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
                ).sum()
                if invalid_emails > 0:
                    column_score *= (1 - invalid_emails / len(df))
                    column_issues.append(f"{invalid_emails} invalid email formats")
            
            # Phone number validation
            if 'phone' in column.lower():
                phone_pattern = r'^[\+]?[1-9][\d]{7,14}$'
                invalid_phones = df[column].apply(
                    lambda x: not bool(re.match(phone_pattern, str(x).replace('-', '').replace(' ', ''))) 
                    if pd.notna(x) else False
                ).sum()
                if invalid_phones > 0:
                    column_score *= (1 - invalid_phones / len(df))
                    column_issues.append(f"{invalid_phones} invalid phone formats")
            
            # Date validation
            if 'date' in column.lower() or df[column].dtype == 'datetime64[ns]':
                try:
                    pd.to_datetime(df[column], errors='coerce')
                    invalid_dates = df[column].isnull().sum()
                    if invalid_dates > 0:
                        column_score *= (1 - invalid_dates / len(df))
                        column_issues.append(f"{invalid_dates} invalid date formats")
                except:
                    column_score *= 0.5
                    column_issues.append("Date parsing issues detected")
            
            column_validity[column] = {
                'score': float(column_score),
                'issues': column_issues
            }
            
            validity_scores.append(column_score)
            
            if column_issues:
                issues.extend([f"Column '{column}': {issue}" for issue in column_issues])
                recommendations.append(f"Validate and fix format issues in column '{column}'")
        
        overall_score = np.mean(validity_scores) if validity_scores else 1.0
        
        return {
            'score': float(overall_score),
            'column_validity': column_validity,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data uniqueness (duplicates, key violations)"""
        duplicate_count = df.duplicated().sum()
        duplicate_ratio = duplicate_count / len(df)
        uniqueness_score = 1 - duplicate_ratio
        
        issues = []
        recommendations = []
        column_uniqueness = {}
        
        # Check for duplicate rows
        if duplicate_count > 0:
            issues.append(f"{duplicate_count} duplicate rows found ({duplicate_ratio:.1%})")
            recommendations.append("Remove duplicate rows to improve data quality")
        
        # Check uniqueness of potential key columns
        for column in df.columns:
            unique_ratio = df[column].nunique() / len(df)
            column_uniqueness[column] = {
                'unique_count': int(df[column].nunique()),
                'unique_ratio': float(unique_ratio),
                'potential_key': unique_ratio > 0.95
            }
            
            # Check for suspicious duplicate patterns in key-like columns
            if ('id' in column.lower() or 'key' in column.lower()) and unique_ratio < 0.95:
                issues.append(f"Potential key column '{column}' has duplicates")
                recommendations.append(f"Review uniqueness constraints for column '{column}'")
        
        return {
            'score': float(uniqueness_score),
            'duplicate_count': int(duplicate_count),
            'duplicate_ratio': float(duplicate_ratio),
            'column_uniqueness': column_uniqueness,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_type_consistency(self, series: pd.Series) -> float:
        """Check if all values in a series have consistent types"""
        if len(series) == 0:
            return 1.0
        
        # Sample values and check their types
        type_counts = {}
        for value in series.head(50):
            try:
                if pd.isna(value):
                    continue
                
                # Try to determine the actual type
                str_value = str(value).strip()
                
                if str_value.isdigit():
                    type_counts['integer'] = type_counts.get('integer', 0) + 1
                elif re.match(r'^-?\d+\.\d+$', str_value):
                    type_counts['float'] = type_counts.get('float', 0) + 1
                elif str_value.lower() in ['true', 'false', 'yes', 'no']:
                    type_counts['boolean'] = type_counts.get('boolean', 0) + 1
                else:
                    type_counts['string'] = type_counts.get('string', 0) + 1
                    
            except:
                type_counts['unknown'] = type_counts.get('unknown', 0) + 1
        
        if not type_counts:
            return 1.0
        
        # Calculate consistency as ratio of most common type
        max_count = max(type_counts.values())
        total_count = sum(type_counts.values())
        
        return max_count / total_count
    
    def _check_format_consistency(self, series: pd.Series) -> float:
        """Check format consistency for string columns"""
        if series.dtype != 'object' or len(series) == 0:
            return 1.0
        
        # Sample non-null string values
        sample = series.dropna().astype(str).head(100)
        if len(sample) == 0:
            return 1.0
        
        # Check for common format patterns
        patterns = {
            'length': [],
            'case': [],
            'special_chars': []
        }
        
        for value in sample:
            patterns['length'].append(len(value))
            patterns['case'].append(value.islower())
            patterns['special_chars'].append(bool(re.search(r'[^a-zA-Z0-9\s]', value)))
        
        # Calculate consistency scores for each pattern
        scores = []
        
        # Length consistency
        length_std = np.std(patterns['length'])
        length_mean = np.mean(patterns['length'])
        length_cv = length_std / length_mean if length_mean > 0 else 0
        scores.append(max(0, 1 - length_cv))
        
        # Case consistency
        case_consistency = np.mean(patterns['case'])
        scores.append(max(case_consistency, 1 - case_consistency))
        
        return np.mean(scores)
    
    def _has_obvious_inconsistencies(self, series: pd.Series) -> bool:
        """Check for obvious data inconsistencies"""
        if len(series) == 0:
            return False
        
        # Check for mixed case issues in categorical data
        if series.dtype == 'object':
            sample = series.dropna().astype(str).head(50)
            if len(sample) > 10:
                # Check for same values with different cases
                lower_values = sample.str.lower()
                if len(lower_values.unique()) < len(sample.unique()) * 0.8:
                    return True
        
        return False
    
    def _find_suspicious_categorical_values(self, series: pd.Series) -> List[str]:
        """Find suspicious patterns in categorical data"""
        suspicious = []
        
        if series.dtype not in ['object', 'category'] or len(series) == 0:
            return suspicious
        
        value_counts = series.value_counts()
        
        # Check for single-character values that might be codes
        single_chars = value_counts[value_counts.index.astype(str).str.len() == 1]
        if len(single_chars) > 3:
            suspicious.append(f"{len(single_chars)} single-character categories (possible encoding issue)")
        
        # Check for many rare categories
        rare_categories = value_counts[value_counts == 1]
        if len(rare_categories) > len(value_counts) * 0.5:
            suspicious.append(f"{len(rare_categories)} categories with only 1 occurrence")
        
        return suspicious
    
    def _get_quality_level(self, score: float) -> str:
        """Convert numeric score to quality level"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    def quick_quality_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform a quick quality assessment"""
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'duplicate_ratio': df.duplicated().sum() / len(df),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }