"""
Optimized visualization utilities for Streamlit app
Performance improvements for charts, graphs, and tables
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st


@st.cache_data(ttl=600)
def create_optimized_quality_gauge(score: float, title: str = "Quality Score") -> go.Figure:
    """
    Create an optimized gauge chart for quality scores with caching
    
    Args:
        score: Quality score value (0-100)
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 80], 'color': "#fff4cc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Optimize layout
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


@st.cache_data(ttl=600)
def create_optimized_network_visualization(
    network_data: Dict[str, Any],
    height: int = 500,
    layout_algorithm: str = 'force_directed'
) -> go.Figure:
    """
    Create optimized network visualization using Plotly with advanced layout algorithms
    
    Args:
        network_data: Dictionary with 'nodes' and 'edges' keys
        height: Chart height in pixels
        layout_algorithm: Layout algorithm to use ('force_directed', 'circular', 'hierarchical', 'kamada_kawai')
        
    Returns:
        Plotly figure object
    """
    if not network_data or not network_data.get('nodes'):
        return go.Figure()
    
    nodes = network_data['nodes']
    edges = network_data.get('edges', [])
    
    n = len(nodes)
    if n == 0:
        return go.Figure()
    
    # Use NetworkX for better layout algorithms
    try:
        import networkx as nx
        
        # Build NetworkX graph
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'], **node)
        
        for edge in edges:
            weight = edge.get('weight', 1.0)
            G.add_edge(edge['source'], edge['target'], weight=weight)
        
        # Apply selected layout algorithm
        if layout_algorithm == 'force_directed' or layout_algorithm == 'spring':
            # Force-directed layout (Spring) - better for showing relationships
            positions = nx.spring_layout(G, k=1.5/np.sqrt(n), iterations=50, seed=42)
        elif layout_algorithm == 'kamada_kawai':
            # Kamada-Kawai layout - good for showing distances
            positions = nx.kamada_kawai_layout(G)
        elif layout_algorithm == 'hierarchical':
            # Hierarchical layout - good for tree-like structures
            try:
                positions = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                # Fallback to spring layout if graphviz not available
                positions = nx.spring_layout(G, k=1.5/np.sqrt(n), iterations=50, seed=42)
        else:  # circular (default fallback)
            positions = nx.circular_layout(G)
        
        # Convert node IDs to positions for all nodes
        node_positions = {node['id']: positions.get(node['id'], (0, 0)) for node in nodes}
        
    except ImportError:
        # Fallback to circular layout if networkx not available
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        node_positions = {
            node['id']: (np.cos(angle), np.sin(angle))
            for node, angle in zip(nodes, angles)
        }
    
    # Create enhanced edge traces with varying thickness based on weight
    edge_traces = []
    edge_weights = [edge.get('weight', 0.5) for edge in edges]
    max_weight = max(edge_weights) if edge_weights else 1.0
    
    for edge in edges:
        if edge['source'] in node_positions and edge['target'] in node_positions:
            x0, y0 = node_positions[edge['source']]
            x1, y1 = node_positions[edge['target']]
            weight = edge.get('weight', 0.5)
            
            # Normalize edge width based on weight
            edge_width = 0.5 + (weight / max_weight) * 4
            edge_opacity = 0.3 + (weight / max_weight) * 0.5
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=edge_width,
                    color=f'rgba(136, 136, 136, {edge_opacity})'
                ),
                hoverinfo='text',
                hovertext=f"Connection strength: {weight:.2f}",
                showlegend=False
            )
            edge_traces.append(edge_trace)
    
    # Create enhanced node trace with better hover information
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_colors = []
    node_sizes = []
    
    for node in nodes:
        if node['id'] in node_positions:
            x, y = node_positions[node['id']]
            node_x.append(x)
            node_y.append(y)
            
            # Truncate long names for display
            display_name = node.get('name', '')[:30] + '...' if len(node.get('name', '')) > 30 else node.get('name', '')
            node_text.append(display_name)
            
            # Enhanced hover info
            hover_info = f"<b>{node.get('name', 'Unknown')}</b><br>"
            hover_info += f"Category: {node.get('category', 'N/A')}<br>"
            hover_info += f"Connections: {node.get('connections', 0)}"
            if 'description' in node:
                desc = node['description'][:100] + '...' if len(node.get('description', '')) > 100 else node.get('description', '')
                hover_info += f"<br>Description: {desc}"
            node_hover.append(hover_info)
            
            # Dynamic node sizing based on connections/importance
            base_size = node.get('size', 15)
            connection_count = node.get('connections', 0)
            adjusted_size = base_size + min(connection_count * 2, 20)
            node_sizes.append(adjusted_size)
            
            node_colors.append(node.get('color', '#1f77b4'))
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_hover,
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color='#333'),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            opacity=0.9,
            symbol='circle'
        ),
        showlegend=False
    )
    
    # Create optimized figure with all traces
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title={
                'text': f'Dataset Relationship Network ({layout_algorithm.replace("_", " ").title()} Layout)',
                'font': {'size': 16, 'color': '#333'},
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[min(node_x) - 0.2, max(node_x) + 0.2] if node_x else [-1, 1]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[min(node_y) - 0.2, max(node_y) + 0.2] if node_y else [-1, 1]
            ),
            height=height,
            paper_bgcolor='rgba(250, 250, 250, 1)',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            # Add dragmode for better interaction
            dragmode='pan'
        )
    )
    
    # Add configuration for interactivity
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.1,
                y=1.15,
                buttons=[
                    dict(label="Reset", method="relayout", args=["xaxis.autorange", True]),
                ]
            )
        ]
    )
    
    return fig


@st.cache_data(ttl=600)
def create_optimized_bar_chart(
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Create optimized bar chart with caching
    
    Args:
        data_df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        color_col: Optional column for color mapping
        
    Returns:
        Plotly figure object
    """
    fig = px.bar(
        data_df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col if color_col else None,
        color_continuous_scale='RdYlGn' if color_col else None
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=400
    )
    
    return fig


@st.cache_data(ttl=600)
def create_optimized_pie_chart(
    values: List[float],
    names: List[str],
    title: str
) -> go.Figure:
    """
    Create optimized pie chart with caching
    
    Args:
        values: List of values
        names: List of category names
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = px.pie(
        values=values,
        names=names,
        title=title,
        hole=0.3  # Make it a donut chart for better readability
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        margin=dict(l=10, r=120, t=40, b=10),
        height=400
    )
    
    return fig


def optimize_dataframe_display(
    df: pd.DataFrame,
    max_rows: int = 100,
    max_cols: int = 20
) -> pd.DataFrame:
    """
    Optimize DataFrame for display by limiting rows and columns
    
    Args:
        df: Input DataFrame
        max_rows: Maximum number of rows to display
        max_cols: Maximum number of columns to display
        
    Returns:
        Optimized DataFrame
    """
    # Limit rows
    if len(df) > max_rows:
        df = df.head(max_rows)
    
    # Limit columns
    if len(df.columns) > max_cols:
        df = df.iloc[:, :max_cols]
    
    return df


def render_optimized_dataframe(
    df: pd.DataFrame,
    title: Optional[str] = None,
    max_rows: int = 100,
    use_container_width: bool = True,
    hide_index: bool = True
) -> None:
    """
    Render DataFrame with optimizations for large datasets
    
    Args:
        df: DataFrame to display
        title: Optional title to display above the table
        max_rows: Maximum rows to display
        use_container_width: Whether to use full container width
        hide_index: Whether to hide the index column
    """
    if title:
        st.subheader(title)
    
    # Optimize for display
    optimized_df = optimize_dataframe_display(df, max_rows=max_rows)
    
    # Display with info if truncated
    if len(df) > max_rows:
        st.info(f"Showing first {max_rows} of {len(df):,} rows")
    
    st.dataframe(
        optimized_df,
        use_container_width=use_container_width,
        hide_index=hide_index
    )


@st.cache_data(ttl=600)
def prepare_table_data(
    datasets: List[Dict[str, Any]],
    columns: List[str],
    format_funcs: Optional[Dict[str, callable]] = None
) -> pd.DataFrame:
    """
    Prepare and cache table data from datasets
    
    Args:
        datasets: List of dataset dictionaries
        columns: Columns to extract
        format_funcs: Optional dictionary of column name to formatting function
        
    Returns:
        Formatted DataFrame
    """
    df = pd.DataFrame(datasets)
    
    if df.empty:
        return df
    
    # Select and rename columns
    if columns:
        df = df[columns].copy()
    
    # Apply formatting functions
    if format_funcs:
        for col, func in format_funcs.items():
            if col in df.columns:
                df[col] = df[col].apply(func)
    
    return df


def format_large_number(num: int) -> str:
    """
    Format large numbers with K, M, B suffixes
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)
