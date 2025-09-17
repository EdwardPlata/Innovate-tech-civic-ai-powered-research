"""
Filter Panel Widget for dynamic data filtering
"""
import dash
from dash import dcc, html, callback, Input, Output, State, ctx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FilterPanel:
    """
    Dynamic filter panel that adapts to dataset structure
    """
    
    def __init__(self):
        """Initialize the filter panel"""
        self.filters = {}
        self.active_filters = {}
    
    def create_layout(self, component_id: str = "filter-panel") -> html.Div:
        """
        Create the filter panel layout
        
        Args:
            component_id: Base ID for components
            
        Returns:
            Dash HTML Div containing the filter interface
        """
        return html.Div([
            html.Div([
                html.H4("Data Filters", className="mb-3"),
                html.Button(
                    "Clear All Filters",
                    id=f"{component_id}-clear-btn",
                    className="btn btn-outline-secondary btn-sm mb-3",
                    n_clicks=0
                )
            ], className="filter-header"),
            
            html.Div(
                id=f"{component_id}-filters-container",
                children="No dataset loaded.",
                className="filters-container"
            ),
            
            html.Div([
                html.H6("Active Filters Summary", className="mt-3 mb-2"),
                html.Div(
                    id=f"{component_id}-active-summary",
                    children="No filters applied.",
                    className="active-filters-summary"
                )
            ])
            
        ], className="filter-panel", id=component_id)
    
    def register_callbacks(self, app: dash.Dash, component_id: str = "filter-panel"):
        """
        Register callbacks for the filter panel
        
        Args:
            app: Dash application instance
            component_id: Base ID for components
        """
        
        # Generate filters when dataset changes
        @app.callback(
            Output(f"{component_id}-filters-container", "children"),
            Input("dataset-data-store", "data")
        )
        def generate_filters(dataset_data):
            if not dataset_data:
                return "No dataset loaded."
            
            try:
                # Convert data back to DataFrame
                df = pd.DataFrame(dataset_data["data"])
                
                # Generate filter components for each column
                filter_components = []
                
                for column in df.columns:
                    filter_component = self._create_column_filter(df, column, component_id)
                    if filter_component:
                        filter_components.append(filter_component)
                
                return html.Div(filter_components)
                
            except Exception as e:
                logger.error(f"Error generating filters: {e}")
                return f"Error generating filters: {str(e)}"
        
        # Handle filter changes
        @app.callback(
            [Output("filtered-data-store", "data"),
             Output(f"{component_id}-active-summary", "children")],
            [Input({"type": "filter-input", "column": dash.dependencies.ALL}, "value"),
             Input(f"{component_id}-clear-btn", "n_clicks")],
            [State("dataset-data-store", "data")]
        )
        def apply_filters(filter_values, clear_clicks, dataset_data):
            if not dataset_data:
                return None, "No dataset loaded."
            
            try:
                df = pd.DataFrame(dataset_data["data"])
                
                # Check if clear button was clicked
                if ctx.triggered and f"{component_id}-clear-btn" in ctx.triggered[0]["prop_id"]:
                    return df.to_dict('records'), "No filters applied."
                
                # Apply filters
                filtered_df = df.copy()
                active_filters_info = []
                
                # Get filter inputs
                if ctx.triggered:
                    for i, filter_value in enumerate(filter_values):
                        if filter_value and i < len(df.columns):
                            column = df.columns[i]
                            
                            try:
                                filtered_df, filter_info = self._apply_column_filter(
                                    filtered_df, column, filter_value
                                )
                                if filter_info:
                                    active_filters_info.append(filter_info)
                            except Exception as e:
                                logger.warning(f"Error applying filter for {column}: {e}")
                
                # Create summary
                if active_filters_info:
                    summary = html.Div([
                        html.P(f"Showing {len(filtered_df):,} of {len(df):,} rows"),
                        html.Ul([html.Li(info) for info in active_filters_info])
                    ])
                else:
                    summary = f"Showing all {len(df):,} rows (no filters applied)"
                
                return filtered_df.to_dict('records'), summary
                
            except Exception as e:
                logger.error(f"Error applying filters: {e}")
                return dataset_data["data"], f"Error applying filters: {str(e)}"
    
    def _create_column_filter(self, df: pd.DataFrame, column: str, component_id: str) -> Optional[html.Div]:
        """Create appropriate filter component for a column"""
        try:
            col_data = df[column]
            
            # Skip columns with too many nulls
            if col_data.isnull().sum() / len(df) > 0.9:
                return None
            
            filter_component = None
            
            # Numeric columns
            if col_data.dtype in ['int64', 'int32', 'float64', 'float32']:
                filter_component = self._create_numeric_filter(col_data, column, component_id)
            
            # Categorical columns (including object types with limited unique values)
            elif col_data.dtype in ['object', 'category'] or col_data.nunique() < 20:
                filter_component = self._create_categorical_filter(col_data, column, component_id)
            
            # DateTime columns
            elif col_data.dtype == 'datetime64[ns]':
                filter_component = self._create_datetime_filter(col_data, column, component_id)
            
            # Boolean columns
            elif col_data.dtype == 'bool':
                filter_component = self._create_boolean_filter(col_data, column, component_id)
            
            # Text columns (object type with many unique values)
            else:
                filter_component = self._create_text_filter(col_data, column, component_id)
            
            if filter_component:
                return html.Div([
                    html.Label(f"{column}:", className="form-label fw-bold"),
                    filter_component
                ], className="mb-3")
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating filter for column {column}: {e}")
            return None
    
    def _create_numeric_filter(self, col_data: pd.Series, column: str, component_id: str) -> html.Div:
        """Create numeric range filter"""
        clean_data = col_data.dropna()
        if len(clean_data) == 0:
            return html.Div("No data available")
        
        min_val = float(clean_data.min())
        max_val = float(clean_data.max())
        
        return html.Div([
            dcc.RangeSlider(
                id={"type": "filter-input", "column": column},
                min=min_val,
                max=max_val,
                value=[min_val, max_val],
                marks={
                    min_val: f"{min_val:.1f}",
                    max_val: f"{max_val:.1f}"
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Small(f"Range: {min_val:.2f} to {max_val:.2f}", className="text-muted")
        ])
    
    def _create_categorical_filter(self, col_data: pd.Series, column: str, component_id: str) -> html.Div:
        """Create categorical multi-select filter"""
        unique_values = col_data.dropna().unique()
        
        if len(unique_values) > 50:
            # Too many categories, create searchable dropdown
            options = [{"label": str(val), "value": str(val)} for val in sorted(unique_values)[:50]]
            return dcc.Dropdown(
                id={"type": "filter-input", "column": column},
                options=options,
                multi=True,
                placeholder=f"Select {column} values...",
                className="mb-1"
            )
        else:
            # Create checklist for smaller number of categories
            options = [{"label": str(val), "value": str(val)} for val in sorted(unique_values)]
            return dcc.Checklist(
                id={"type": "filter-input", "column": column},
                options=options,
                value=[],
                className="form-check-input-group",
                style={"max-height": "150px", "overflow-y": "auto"}
            )
    
    def _create_datetime_filter(self, col_data: pd.Series, column: str, component_id: str) -> html.Div:
        """Create datetime range filter"""
        clean_data = col_data.dropna()
        if len(clean_data) == 0:
            return html.Div("No data available")
        
        min_date = clean_data.min().date()
        max_date = clean_data.max().date()
        
        return html.Div([
            dcc.DatePickerRange(
                id={"type": "filter-input", "column": column},
                start_date=min_date,
                end_date=max_date,
                display_format="YYYY-MM-DD"
            ),
            html.Small(f"Range: {min_date} to {max_date}", className="text-muted d-block")
        ])
    
    def _create_boolean_filter(self, col_data: pd.Series, column: str, component_id: str) -> html.Div:
        """Create boolean filter"""
        return dcc.RadioItems(
            id={"type": "filter-input", "column": column},
            options=[
                {"label": "All", "value": "all"},
                {"label": "True", "value": "true"},
                {"label": "False", "value": "false"}
            ],
            value="all",
            className="form-check-input-group"
        )
    
    def _create_text_filter(self, col_data: pd.Series, column: str, component_id: str) -> html.Div:
        """Create text search filter"""
        return html.Div([
            dcc.Input(
                id={"type": "filter-input", "column": column},
                type="text",
                placeholder=f"Search in {column}...",
                className="form-control"
            ),
            html.Small("Enter text to search (case-insensitive)", className="text-muted")
        ])
    
    def _apply_column_filter(self, df: pd.DataFrame, column: str, filter_value: Any) -> tuple:
        """Apply filter to a specific column"""
        if not filter_value or column not in df.columns:
            return df, None
        
        col_data = df[column]
        filter_info = None
        
        try:
            # Numeric range filter
            if isinstance(filter_value, list) and len(filter_value) == 2:
                if col_data.dtype in ['int64', 'int32', 'float64', 'float32']:
                    min_val, max_val = filter_value
                    mask = (col_data >= min_val) & (col_data <= max_val)
                    df_filtered = df[mask]
                    filter_info = f"{column}: {min_val:.2f} to {max_val:.2f}"
                    return df_filtered, filter_info
                
                # DateTime range filter
                elif col_data.dtype == 'datetime64[ns]':
                    start_date, end_date = filter_value
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    mask = (col_data >= start_date) & (col_data <= end_date)
                    df_filtered = df[mask]
                    filter_info = f"{column}: {start_date.date()} to {end_date.date()}"
                    return df_filtered, filter_info
            
            # Categorical multi-select filter
            elif isinstance(filter_value, list) and filter_value:
                mask = col_data.astype(str).isin([str(val) for val in filter_value])
                df_filtered = df[mask]
                filter_info = f"{column}: {len(filter_value)} values selected"
                return df_filtered, filter_info
            
            # Boolean filter
            elif isinstance(filter_value, str):
                if filter_value == "true":
                    mask = col_data == True
                    df_filtered = df[mask]
                    filter_info = f"{column}: True only"
                    return df_filtered, filter_info
                elif filter_value == "false":
                    mask = col_data == False
                    df_filtered = df[mask]
                    filter_info = f"{column}: False only"
                    return df_filtered, filter_info
                elif filter_value != "all" and filter_value:
                    # Text search filter
                    mask = col_data.astype(str).str.contains(filter_value, case=False, na=False)
                    df_filtered = df[mask]
                    filter_info = f"{column}: contains '{filter_value}'"
                    return df_filtered, filter_info
            
        except Exception as e:
            logger.error(f"Error applying filter for {column}: {e}")
        
        return df, None
    
    def get_filter_summary(self, df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of applied filters"""
        return {
            "total_rows": len(original_df),
            "filtered_rows": len(df),
            "filter_ratio": len(df) / len(original_df) if len(original_df) > 0 else 0,
            "rows_filtered_out": len(original_df) - len(df)
        }