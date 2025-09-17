"""
Data Processing utilities for Scout Data Discovery integration
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processing and transformation utilities for dashboard
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.numeric_types = ['int64', 'float64', 'int32', 'float32']
        self.datetime_types = ['datetime64[ns]', 'object']
        self.categorical_types = ['object', 'category']
    
    def infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer appropriate data types for columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to inferred types
        """
        type_mapping = {}
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'int32']:
                type_mapping[column] = 'integer'
            elif df[column].dtype in ['float64', 'float32']:
                type_mapping[column] = 'numeric'
            elif df[column].dtype == 'bool':
                type_mapping[column] = 'boolean'
            elif df[column].dtype == 'datetime64[ns]':
                type_mapping[column] = 'datetime'
            else:
                # Check if it's a date string
                if self._is_date_column(df[column]):
                    type_mapping[column] = 'datetime'
                # Check if it's numeric stored as string
                elif self._is_numeric_string(df[column]):
                    type_mapping[column] = 'numeric'
                # Check if it's categorical
                elif self._is_categorical(df[column]):
                    type_mapping[column] = 'categorical'
                else:
                    type_mapping[column] = 'text'
        
        return type_mapping
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a series contains date strings"""
        if series.dtype != 'object':
            return False
        
        # Sample a few non-null values
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        matches = 0
        for value in sample:
            if any(re.match(pattern, str(value)) for pattern in date_patterns):
                matches += 1
        
        return matches / len(sample) > 0.5
    
    def _is_numeric_string(self, series: pd.Series) -> bool:
        """Check if a series contains numeric values stored as strings"""
        if series.dtype != 'object':
            return False
        
        sample = series.dropna().head(20)
        if len(sample) == 0:
            return False
        
        numeric_count = 0
        for value in sample:
            try:
                float(str(value).replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
            except (ValueError, TypeError):
                continue
        
        return numeric_count / len(sample) > 0.7
    
    def _is_categorical(self, series: pd.Series) -> bool:
        """Check if a series should be treated as categorical"""
        if len(series) == 0:
            return False
        
        unique_ratio = series.nunique() / len(series)
        return unique_ratio < 0.1 or series.nunique() < 20
    
    def clean_data(self, df: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   handle_missing: str = 'auto') -> pd.DataFrame:
        """
        Clean and prepare data for visualization
        
        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: How to handle missing values ('auto', 'drop', 'fill')
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates if requested
        if remove_duplicates:
            original_len = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed = original_len - len(df_clean)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")
        
        # Handle missing values
        if handle_missing == 'auto':
            df_clean = self._auto_handle_missing(df_clean)
        elif handle_missing == 'drop':
            df_clean = df_clean.dropna()
        elif handle_missing == 'fill':
            df_clean = self._fill_missing_values(df_clean)
        
        return df_clean
    
    def _auto_handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically handle missing values based on data types"""
        df_clean = df.copy()
        
        for column in df_clean.columns:
            missing_ratio = df_clean[column].isnull().sum() / len(df_clean)
            
            if missing_ratio > 0.5:
                # Drop columns with more than 50% missing values
                logger.warning(f"Dropping column {column} (>{missing_ratio:.1%} missing)")
                df_clean = df_clean.drop(columns=[column])
            elif missing_ratio > 0:
                # Fill missing values based on data type
                if df_clean[column].dtype in self.numeric_types:
                    df_clean[column].fillna(df_clean[column].median(), inplace=True)
                else:
                    df_clean[column].fillna('Unknown', inplace=True)
        
        return df_clean
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults"""
        df_filled = df.copy()
        
        for column in df_filled.columns:
            if df_filled[column].dtype in self.numeric_types:
                df_filled[column].fillna(df_filled[column].median(), inplace=True)
            elif df_filled[column].dtype == 'bool':
                df_filled[column].fillna(False, inplace=True)
            else:
                df_filled[column].fillna('Unknown', inplace=True)
        
        return df_filled
    
    def convert_data_types(self, df: pd.DataFrame, 
                          type_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Convert DataFrame columns to appropriate data types
        
        Args:
            df: Input DataFrame
            type_mapping: Optional explicit type mapping
            
        Returns:
            DataFrame with converted types
        """
        if type_mapping is None:
            type_mapping = self.infer_data_types(df)
        
        df_converted = df.copy()
        
        for column, target_type in type_mapping.items():
            if column not in df_converted.columns:
                continue
            
            try:
                if target_type == 'integer':
                    df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce').astype('Int64')
                elif target_type == 'numeric':
                    # Remove common formatting characters
                    if df_converted[column].dtype == 'object':
                        df_converted[column] = df_converted[column].astype(str).str.replace('[,$%]', '', regex=True)
                    df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                elif target_type == 'datetime':
                    df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
                elif target_type == 'categorical':
                    df_converted[column] = df_converted[column].astype('category')
                elif target_type == 'boolean':
                    df_converted[column] = df_converted[column].astype('bool')
                
                logger.info(f"Converted column {column} to {target_type}")
                
            except Exception as e:
                logger.warning(f"Failed to convert column {column} to {target_type}: {e}")
        
        return df_converted
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data summary statistics
        """
        summary = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'columns': {},
            'missing_values': {},
            'data_types': {},
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Column-level analysis
        for column in df.columns:
            col_data = df[column]
            
            # Basic info
            summary['columns'][column] = {
                'dtype': str(col_data.dtype),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(df)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(df)) * 100
            }
            
            # Type-specific analysis
            if col_data.dtype in self.numeric_types:
                summary['numeric_summary'][column] = {
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': col_data.quantile(0.25),
                    'q75': col_data.quantile(0.75)
                }
            elif col_data.nunique() < 20:  # Categorical-like
                value_counts = col_data.value_counts().head(10)
                summary['categorical_summary'][column] = {
                    'top_values': value_counts.to_dict(),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'frequency': value_counts.iloc[0] if len(value_counts) > 0 else None
                }
        
        return summary
    
    def prepare_for_visualization(self, df: pd.DataFrame, 
                                 chart_type: str,
                                 x_column: Optional[str] = None,
                                 y_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare data specifically for visualization requirements
        
        Args:
            df: Input DataFrame
            chart_type: Type of chart ('bar', 'line', 'scatter', 'map', etc.)
            x_column: Column for x-axis
            y_column: Column for y-axis
            
        Returns:
            Dictionary with prepared data and metadata
        """
        result = {
            'data': df.copy(),
            'x_column': x_column,
            'y_column': y_column,
            'chart_type': chart_type,
            'metadata': {}
        }
        
        if chart_type == 'bar' and x_column:
            # For bar charts, aggregate data if needed
            if df[x_column].nunique() > 50:
                # Too many categories, group small ones
                result['data'] = self._group_small_categories(df, x_column, y_column)
                result['metadata']['grouped_small_categories'] = True
        
        elif chart_type == 'line' and x_column:
            # For line charts, ensure proper ordering
            if df[x_column].dtype == 'datetime64[ns]':
                result['data'] = df.sort_values(x_column)
                result['metadata']['sorted_by_date'] = True
        
        elif chart_type == 'scatter':
            # For scatter plots, handle large datasets
            if len(df) > 10000:
                result['data'] = df.sample(n=10000, random_state=42)
                result['metadata']['sampled'] = True
        
        return result
    
    def _group_small_categories(self, df: pd.DataFrame, 
                               category_column: str,
                               value_column: Optional[str] = None,
                               min_frequency: int = 10) -> pd.DataFrame:
        """Group small categories into 'Others' category"""
        df_grouped = df.copy()
        
        if value_column:
            # Group by category and sum values
            grouped = df_grouped.groupby(category_column)[value_column].sum()
        else:
            # Count occurrences
            grouped = df_grouped[category_column].value_counts()
        
        # Identify small categories
        small_categories = grouped[grouped < min_frequency].index
        
        # Replace small categories with 'Others'
        mask = df_grouped[category_column].isin(small_categories)
        df_grouped.loc[mask, category_column] = 'Others'
        
        return df_grouped
    
    def detect_outliers(self, df: pd.DataFrame, 
                       column: str, 
                       method: str = 'iqr') -> pd.Series:
        """
        Detect outliers in a numeric column
        
        Args:
            df: Input DataFrame
            column: Column to analyze
            method: Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            Boolean series indicating outliers
        """
        if column not in df.columns or df[column].dtype not in self.numeric_types:
            return pd.Series([False] * len(df), index=df.index)
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            return z_scores > 3
        
        return pd.Series([False] * len(df), index=df.index)