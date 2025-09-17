"""
Chart Factory for dynamic chart generation based on data characteristics
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ChartFactory:
    """
    Factory class for creating appropriate chart types based on data characteristics
    """
    
    def __init__(self):
        """Initialize the chart factory"""
        self.chart_mappings = {
            ('categorical', 'numeric'): ['bar', 'violin', 'box'],
            ('categorical', 'categorical'): ['bar', 'heatmap', 'sankey'],
            ('numeric', 'numeric'): ['scatter', 'line', 'density'],
            ('datetime', 'numeric'): ['line', 'area', 'bar'],
            ('datetime', 'categorical'): ['line', 'bar'],
            ('geographic', 'numeric'): ['choropleth', 'scatter_map'],
            ('geographic', 'categorical'): ['scatter_map'],
            ('single_numeric',): ['histogram', 'box', 'violin'],
            ('single_categorical',): ['bar', 'pie', 'treemap']
        }
        
        self.color_schemes = {
            'categorical': px.colors.qualitative.Set3,
            'sequential': px.colors.sequential.Viridis,
            'diverging': px.colors.diverging.RdBu
        }
    
    def suggest_chart_types(self, df: pd.DataFrame, 
                           x_column: Optional[str] = None,
                           y_column: Optional[str] = None) -> List[str]:
        """
        Suggest appropriate chart types based on data characteristics
        
        Args:
            df: Input DataFrame
            x_column: Column for x-axis
            y_column: Column for y-axis
            
        Returns:
            List of suggested chart types
        """
        if x_column is None and y_column is None:
            # Single column analysis
            if len(df.columns) == 1:
                column = df.columns[0]
                data_type = self._get_data_type(df[column])
                return self.chart_mappings.get((f'single_{data_type}',), ['bar'])
            else:
                # Multiple columns - suggest overview charts
                return ['correlation', 'pairplot', 'distribution_grid']
        
        elif y_column is None:
            # Single variable analysis
            data_type = self._get_data_type(df[x_column])
            return self.chart_mappings.get((f'single_{data_type}',), ['bar'])
        
        else:
            # Two variable analysis
            x_type = self._get_data_type(df[x_column])
            y_type = self._get_data_type(df[y_column])
            
            return self.chart_mappings.get((x_type, y_type), ['scatter'])
    
    def create_chart(self, df: pd.DataFrame,
                    chart_type: str,
                    x_column: Optional[str] = None,
                    y_column: Optional[str] = None,
                    color_column: Optional[str] = None,
                    size_column: Optional[str] = None,
                    **kwargs) -> go.Figure:
        """
        Create a chart of the specified type
        
        Args:
            df: Input DataFrame
            chart_type: Type of chart to create
            x_column: Column for x-axis
            y_column: Column for y-axis
            color_column: Column for color coding
            size_column: Column for size coding
            **kwargs: Additional chart-specific parameters
            
        Returns:
            Plotly Figure object
        """
        try:
            # Prepare data
            chart_data = self._prepare_chart_data(df, chart_type, x_column, y_column)
            
            # Create chart based on type
            if chart_type == 'bar':
                return self._create_bar_chart(chart_data, x_column, y_column, color_column, **kwargs)
            elif chart_type == 'line':
                return self._create_line_chart(chart_data, x_column, y_column, color_column, **kwargs)
            elif chart_type == 'scatter':
                return self._create_scatter_chart(chart_data, x_column, y_column, color_column, size_column, **kwargs)
            elif chart_type == 'histogram':
                return self._create_histogram(chart_data, x_column, **kwargs)
            elif chart_type == 'box':
                return self._create_box_chart(chart_data, x_column, y_column, color_column, **kwargs)
            elif chart_type == 'heatmap':
                return self._create_heatmap(chart_data, x_column, y_column, **kwargs)
            elif chart_type == 'pie':
                return self._create_pie_chart(chart_data, x_column, y_column, **kwargs)
            elif chart_type == 'area':
                return self._create_area_chart(chart_data, x_column, y_column, color_column, **kwargs)
            elif chart_type == 'violin':
                return self._create_violin_chart(chart_data, x_column, y_column, color_column, **kwargs)
            elif chart_type == 'density':
                return self._create_density_chart(chart_data, x_column, y_column, **kwargs)
            elif chart_type == 'correlation':
                return self._create_correlation_matrix(chart_data, **kwargs)
            else:
                logger.warning(f"Chart type '{chart_type}' not implemented, creating scatter plot")
                return self._create_scatter_chart(chart_data, x_column, y_column, color_column, size_column, **kwargs)
                
        except Exception as e:
            logger.error(f"Error creating {chart_type} chart: {e}")
            return self._create_error_chart(str(e))
    
    def _get_data_type(self, series: pd.Series) -> str:
        """Determine the data type category for chart selection"""
        if series.dtype in ['int64', 'int32', 'float64', 'float32']:
            return 'numeric'
        elif series.dtype == 'datetime64[ns]':
            return 'datetime'
        elif series.dtype in ['object', 'category']:
            # Check if it's geographic data
            if self._is_geographic_column(series):
                return 'geographic'
            # Check unique values ratio
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1 or series.nunique() < 20:
                return 'categorical'
            else:
                return 'text'
        else:
            return 'categorical'
    
    def _is_geographic_column(self, series: pd.Series) -> bool:
        """Check if a column contains geographic data"""
        if series.dtype != 'object':
            return False
        
        geographic_keywords = ['state', 'country', 'city', 'zip', 'postal', 'address', 'location']
        column_name = series.name.lower() if series.name else ''
        
        return any(keyword in column_name for keyword in geographic_keywords)
    
    def _prepare_chart_data(self, df: pd.DataFrame, chart_type: str,
                           x_column: Optional[str], y_column: Optional[str]) -> pd.DataFrame:
        """Prepare data for specific chart type"""
        chart_data = df.copy()
        
        # Handle aggregation for bar charts with many categories
        if chart_type == 'bar' and x_column:
            if chart_data[x_column].nunique() > 50:
                # Group small categories
                value_counts = chart_data[x_column].value_counts()
                small_categories = value_counts[value_counts < 5].index
                mask = chart_data[x_column].isin(small_categories)
                chart_data.loc[mask, x_column] = 'Others'
        
        # Sample data for performance if too large
        if len(chart_data) > 10000 and chart_type in ['scatter', 'density']:
            chart_data = chart_data.sample(n=10000, random_state=42)
            logger.info(f"Sampled data to 10,000 rows for {chart_type} chart")
        
        return chart_data
    
    def _create_bar_chart(self, df: pd.DataFrame, x_column: str, y_column: Optional[str],
                         color_column: Optional[str], **kwargs) -> go.Figure:
        """Create a bar chart"""
        if y_column is None:
            # Count chart
            data = df[x_column].value_counts().reset_index()
            data.columns = [x_column, 'count']
            fig = px.bar(data, x=x_column, y='count',
                        title=f"Count of {x_column}",
                        color_discrete_sequence=self.color_schemes['categorical'])
        else:
            # Aggregated bar chart
            if color_column:
                fig = px.bar(df, x=x_column, y=y_column, color=color_column,
                            title=f"{y_column} by {x_column}",
                            color_discrete_sequence=self.color_schemes['categorical'])
            else:
                fig = px.bar(df, x=x_column, y=y_column,
                            title=f"{y_column} by {x_column}",
                            color_discrete_sequence=self.color_schemes['categorical'])
        
        # Customize layout
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column or 'Count',
            showlegend=color_column is not None,
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, x_column: str, y_column: str,
                          color_column: Optional[str], **kwargs) -> go.Figure:
        """Create a line chart"""
        if color_column:
            fig = px.line(df, x=x_column, y=y_column, color=color_column,
                         title=f"{y_column} over {x_column}",
                         color_discrete_sequence=self.color_schemes['categorical'])
        else:
            fig = px.line(df, x=x_column, y=y_column,
                         title=f"{y_column} over {x_column}")
        
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_scatter_chart(self, df: pd.DataFrame, x_column: str, y_column: str,
                             color_column: Optional[str], size_column: Optional[str],
                             **kwargs) -> go.Figure:
        """Create a scatter plot"""
        fig = px.scatter(df, x=x_column, y=y_column,
                        color=color_column, size=size_column,
                        title=f"{y_column} vs {x_column}",
                        color_continuous_scale=self.color_schemes['sequential'])
        
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, x_column: str, **kwargs) -> go.Figure:
        """Create a histogram"""
        fig = px.histogram(df, x=x_column,
                          title=f"Distribution of {x_column}",
                          nbins=kwargs.get('bins', 30))
        
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title='Frequency',
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_box_chart(self, df: pd.DataFrame, x_column: Optional[str], y_column: str,
                         color_column: Optional[str], **kwargs) -> go.Figure:
        """Create a box plot"""
        if x_column:
            fig = px.box(df, x=x_column, y=y_column, color=color_column,
                        title=f"{y_column} distribution by {x_column}")
        else:
            fig = px.box(df, y=y_column,
                        title=f"{y_column} distribution")
        
        fig.update_layout(
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, x_column: str, y_column: str,
                       **kwargs) -> go.Figure:
        """Create a heatmap"""
        # Create pivot table for heatmap
        pivot_data = df.groupby([x_column, y_column]).size().unstack(fill_value=0)
        
        fig = px.imshow(pivot_data,
                       title=f"Heatmap of {x_column} vs {y_column}",
                       color_continuous_scale=self.color_schemes['sequential'],
                       aspect='auto')
        
        fig.update_layout(
            xaxis_title=y_column,
            yaxis_title=x_column,
            font=dict(size=12)
        )
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, x_column: str, y_column: Optional[str],
                         **kwargs) -> go.Figure:
        """Create a pie chart"""
        if y_column is None:
            # Count chart
            data = df[x_column].value_counts().reset_index()
            data.columns = [x_column, 'count']
            fig = px.pie(data, names=x_column, values='count',
                        title=f"Distribution of {x_column}")
        else:
            fig = px.pie(df, names=x_column, values=y_column,
                        title=f"{y_column} by {x_column}")
        
        fig.update_layout(font=dict(size=12))
        return fig
    
    def _create_area_chart(self, df: pd.DataFrame, x_column: str, y_column: str,
                          color_column: Optional[str], **kwargs) -> go.Figure:
        """Create an area chart"""
        if color_column:
            fig = px.area(df, x=x_column, y=y_column, color=color_column,
                         title=f"{y_column} over {x_column}")
        else:
            fig = px.area(df, x=x_column, y=y_column,
                         title=f"{y_column} over {x_column}")
        
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_violin_chart(self, df: pd.DataFrame, x_column: Optional[str], y_column: str,
                            color_column: Optional[str], **kwargs) -> go.Figure:
        """Create a violin plot"""
        if x_column:
            fig = px.violin(df, x=x_column, y=y_column, color=color_column,
                           title=f"{y_column} distribution by {x_column}")
        else:
            fig = px.violin(df, y=y_column,
                           title=f"{y_column} distribution")
        
        fig.update_layout(
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_density_chart(self, df: pd.DataFrame, x_column: str, y_column: str,
                             **kwargs) -> go.Figure:
        """Create a density plot"""
        fig = px.density_contour(df, x=x_column, y=y_column,
                                title=f"Density plot of {y_column} vs {x_column}")
        
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_correlation_matrix(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create a correlation matrix heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self._create_error_chart("No numeric columns found for correlation matrix")
        
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix,
                       title="Correlation Matrix",
                       color_continuous_scale=self.color_schemes['diverging'],
                       aspect='auto')
        
        fig.update_layout(font=dict(size=12))
        return fig
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create an error chart when chart creation fails"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart:<br>{error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Chart Creation Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        return fig
    
    def get_chart_options(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get available chart options based on data characteristics"""
        options = {
            'x_columns': list(df.columns),
            'y_columns': list(df.select_dtypes(include=[np.number]).columns),
            'color_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'size_columns': list(df.select_dtypes(include=[np.number]).columns)
        }
        
        # Add suggested chart types for each column combination
        options['suggested_charts'] = {}
        for x_col in options['x_columns']:
            for y_col in options['y_columns']:
                if x_col != y_col:
                    suggestions = self.suggest_chart_types(df, x_col, y_col)
                    options['suggested_charts'][f"{x_col}_vs_{y_col}"] = suggestions
        
        return options