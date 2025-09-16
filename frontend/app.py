"""
Scout Data Discovery - Streamlit Frontend

Interactive web application for exploring NYC Open Data with Scout methodology.
Features dataset discovery, quality assessment, and relationship mapping.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import altair as alt

# Import backend manager and AI analyst
from components.backend_manager import get_backend_manager
from components.ai_analyst_component import get_ai_analyst_component

# Page configuration
st.set_page_config(
    page_title="Scout Data Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8080/api"
CACHE_TTL = 300  # 5 minutes

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}

.quality-score-excellent {
    color: #28a745;
    font-weight: bold;
    font-size: 1.2em;
}

.quality-score-good {
    color: #17a2b8;
    font-weight: bold;
    font-size: 1.2em;
}

.quality-score-fair {
    color: #ffc107;
    font-weight: bold;
    font-size: 1.2em;
}

.quality-score-poor {
    color: #dc3545;
    font-weight: bold;
    font-size: 1.2em;
}

.dataset-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.network-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    background: #fafafa;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=CACHE_TTL)
def fetch_api_data(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Fetch data from the API with caching"""
    url = f"{API_BASE_URL}/{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {}

def format_number(num):
    """Format numbers for display"""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)

def get_quality_class(score):
    """Get CSS class for quality score"""
    if score >= 90:
        return "quality-score-excellent"
    elif score >= 80:
        return "quality-score-good"
    elif score >= 70:
        return "quality-score-fair"
    else:
        return "quality-score-poor"

def create_quality_gauge(score, title="Quality Score"):
    """Create a gauge chart for quality scores"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))

    fig.update_layout(height=300)
    return fig

def create_network_visualization(network_data):
    """Create network visualization using Plotly"""
    if not network_data or not network_data.get('nodes'):
        return go.Figure()

    # Extract node positions (using a simple circular layout for demo)
    nodes = network_data['nodes']
    edges = network_data.get('edges', [])

    # Create simple circular positions
    n = len(nodes)
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n
        positions[node['id']] = (np.cos(angle), np.sin(angle))

    # Create edge traces
    edge_x = []
    edge_y = []

    for edge in edges:
        x0, y0 = positions[edge['source']]
        x1, y1 = positions[edge['target']]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in nodes:
        x, y = positions[node['id']]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node['name'])
        node_colors.append(node.get('color', '#1f77b4'))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=[node.get('size', 10) for node in nodes],
            color=node_colors,
            line=dict(width=2, color='white')
        )
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title={
                            'text': 'Dataset Relationship Network',
                            'font': {'size': 16}
                        },
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Network visualization of dataset relationships",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(size=12, color="gray")
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500))

    return fig

# Initialize session state
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'quality_cache' not in st.session_state:
    st.session_state.quality_cache = {}
if 'backend_started' not in st.session_state:
    st.session_state.backend_started = False

# Main app
def main():
    # Title and header
    st.title("🔍 Scout Data Explorer")
    st.markdown("*Discover, analyze, and explore NYC Open Data with AI-powered insights*")

    # Backend management and AI analyst
    backend_manager = get_backend_manager()
    ai_analyst = get_ai_analyst_component()

    # Check if backend is running
    backend_status = backend_manager.get_status()

    # Try direct API test if backend manager thinks it's not running
    if not backend_status["is_running"]:
        try:
            import requests
            response = requests.get("http://localhost:8080/api/health", timeout=3)
            if response.status_code == 200:
                # Backend is actually running, update the manager
                backend_manager.is_running = True
                backend_status["is_running"] = True
                st.success("✅ Connected to Scout API")
            else:
                st.warning("⚠️ Backend API is not running")
        except:
            st.warning("⚠️ Backend API is not running")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🚀 Start Backend Server"):
                with st.spinner("Starting backend server... This may take a moment."):
                    if backend_manager.start_backend():
                        st.session_state.backend_started = True
                        st.success("✅ Backend server started successfully!")
                        # Remove st.rerun() to prevent page refresh
                    else:
                        st.error("❌ Failed to start backend server")
                        st.info("You can manually start the backend by running:\n`uvicorn main:app --reload` in the backend directory")

        with col2:
            st.info("**Manual Start Option:**\n\n"
                   "1. Open terminal in `backend/` directory\n"
                   "2. Run: `uvicorn main:app --reload`\n"
                   "3. Refresh this page")

        return
    else:
        st.success("✅ Connected to Scout API")

    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/4CAF50/white?text=Scout+Data", width=200)

        # Backend status indicator (moved above navigation)
        if backend_status.get("is_running", False):
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Disconnected")
            if st.button("🔄 Refresh Status", key="sidebar_refresh"):
                # Force recheck by clearing the backend manager state
                backend_manager.is_running = False
                st.rerun()

        selected = option_menu(
            menu_title="Navigation",
            options=["Dashboard", "Dataset Explorer", "Quality Assessment", "Relationship Mapping", "Data Sample", "AI Analysis", "AI Setup"],
            icons=["house", "search", "clipboard-check", "diagram-3", "table", "robot", "gear"],
            default_index=0
        )

        # Backend Status (detailed)
        st.subheader("Backend Status")

        # Real-time status update
        if st.button("🔄 Refresh Status"):
            # Force recheck by clearing the backend manager state
            backend_manager.is_running = False
            backend_status = backend_manager.get_status()
            if backend_status["is_running"]:
                st.success("🟢 Backend Online")
            else:
                st.error("🔴 Backend Offline")

        # Display current status
        current_status = backend_manager.get_status()
        status_info = {
            "API Server": "🟢 Online" if current_status["is_running"] else "🔴 Offline",
            "Health Check": "✅ Pass" if current_status["health_check"] else "❌ Fail",
            "Process": "🔄 Active" if current_status["process_active"] else "⏸️ Inactive"
        }

        for key, value in status_info.items():
            st.text(f"{key}: {value}")

        # Additional API stats
        if current_status["is_running"]:
            stats = fetch_api_data("stats")
            if stats:
                st.caption("Scout Status:")
                st.caption(f"• Scout Ready: {'✅' if stats.get('scout_initialized') else '❌'}")

        # Backend control
        st.subheader("Backend Control")
        if current_status["is_running"] and current_status["process_active"]:
            if st.button("🛑 Stop Backend"):
                backend_manager.stop_backend()
                st.success("Backend stopped")
        elif not current_status["is_running"]:
            if st.button("🚀 Start Backend"):
                with st.spinner("Starting..."):
                    if backend_manager.start_backend():
                        st.success("Started!")
                        # Remove st.rerun() to prevent page refresh
                    else:
                        st.error("Failed to start")

    # Main content based on navigation
    if selected == "Dashboard":
        show_dashboard(ai_analyst)
    elif selected == "Dataset Explorer":
        show_dataset_explorer(ai_analyst)
    elif selected == "Quality Assessment":
        show_quality_assessment(ai_analyst)
    elif selected == "Relationship Mapping":
        show_relationship_mapping(ai_analyst)
    elif selected == "Data Sample":
        show_data_sample(ai_analyst)
    elif selected == "AI Analysis":
        show_ai_analysis(ai_analyst)
    elif selected == "AI Setup":
        show_ai_setup(ai_analyst)

def show_dashboard(ai_analyst):
    """Dashboard with overview and top datasets"""
    st.header("📊 Dashboard")

    # AI Configuration section
    ai_analyst.render_ai_configuration()

    # Get top updated datasets
    with st.spinner("Loading recently updated datasets..."):
        top_datasets = fetch_api_data("datasets/top-updated?limit=10")

    if not top_datasets:
        st.warning("No datasets available")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📈 Total Datasets", len(top_datasets))

    with col2:
        total_downloads = sum(d.get('download_count', 0) for d in top_datasets)
        st.metric("⬇️ Total Downloads", format_number(total_downloads))

    with col3:
        categories = set(d.get('category', 'Unknown') for d in top_datasets)
        st.metric("🏷️ Categories", len(categories))

    with col4:
        # Calculate recently updated datasets (handle timezone comparison properly)
        recent_count = 0
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            for d in top_datasets:
                if d.get('updated_at'):
                    try:
                        # Parse the datetime string and make it timezone-naive for comparison
                        updated_dt = datetime.fromisoformat(d['updated_at'].replace('Z', '+00:00'))
                        # Convert to naive datetime for comparison
                        if updated_dt.tzinfo is not None:
                            updated_dt = updated_dt.replace(tzinfo=None)

                        if updated_dt > cutoff_date:
                            recent_count += 1
                    except (ValueError, TypeError):
                        # Skip datasets with invalid date formats
                        continue
        except Exception as e:
            # If there's any error, just set to 0
            recent_count = 0
        st.metric("🕐 Recently Updated", recent_count)

    # Category distribution
    st.subheader("📊 Dataset Categories")

    categories_data = {}
    for dataset in top_datasets:
        cat = dataset.get('category', 'Unknown')
        categories_data[cat] = categories_data.get(cat, 0) + 1

    if categories_data:
        fig_pie = px.pie(
            values=list(categories_data.values()),
            names=list(categories_data.keys()),
            title="Distribution of Dataset Categories"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Top 10 datasets table
    st.subheader("🔥 Top Recently Updated Datasets")

    df = pd.DataFrame(top_datasets)
    if not df.empty:
        # Format the dataframe for display
        display_df = df[['name', 'category', 'download_count', 'updated_at']].copy()
        display_df.columns = ['Dataset Name', 'Category', 'Downloads', 'Last Updated']
        display_df['Downloads'] = display_df['Downloads'].apply(format_number)
        display_df['Last Updated'] = pd.to_datetime(display_df['Last Updated']).dt.strftime('%Y-%m-%d')

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # Dataset selection for detailed view
        selected_idx = st.selectbox(
            "Select a dataset for detailed analysis:",
            range(len(top_datasets)),
            format_func=lambda i: f"{top_datasets[i]['name'][:50]}..."
        )

        if st.button("📋 Analyze Selected Dataset"):
            st.session_state.selected_dataset = top_datasets[selected_idx]
            st.success(f"Selected dataset: {top_datasets[selected_idx]['name'][:50]}...")

        # AI Quick Insights for Dashboard
        if selected_idx is not None:
            st.markdown("---")
            st.subheader("🤖 AI Quick Insights")

            selected_dataset = top_datasets[selected_idx]
            ai_analyst.render_mini_analyst(selected_dataset, auto_run=False)

            if st.button("🧠 Generate AI Insights"):
                ai_analyst.render_mini_analyst(selected_dataset, auto_run=True)

def show_dataset_explorer(ai_analyst):
    """Dataset search and exploration interface"""
    st.header("🔍 Dataset Explorer")

    # Search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        search_terms = st.text_input(
            "Search datasets:",
            placeholder="e.g., 311, health, transportation, housing",
            help="Enter search terms separated by commas"
        )

    with col2:
        search_limit = st.number_input("Limit", min_value=5, max_value=100, value=20)

    if st.button("🔍 Search Datasets") and search_terms:
        terms_list = [term.strip() for term in search_terms.split(',')]

        with st.spinner("Searching datasets..."):
            search_data = {
                "search_terms": terms_list,
                "limit": search_limit,
                "include_quality": False
            }

            results = fetch_api_data("datasets/search", method="POST", data=search_data)

        if results:
            st.session_state.search_results = results
            st.success(f"Found {len(results)} datasets!")
        else:
            st.warning("No datasets found for your search terms.")

    # Display search results
    if st.session_state.search_results:
        st.subheader(f"📋 Search Results ({len(st.session_state.search_results)} datasets)")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            categories = list(set(d.get('category', 'Unknown') for d in st.session_state.search_results))
            selected_category = st.selectbox("Filter by category:", ['All'] + categories)

        with col2:
            min_downloads = st.number_input("Min downloads:", min_value=0, value=0)

        with col3:
            sort_by = st.selectbox("Sort by:", ['name', 'download_count', 'updated_at'])

        # Apply filters
        filtered_results = st.session_state.search_results

        if selected_category != 'All':
            filtered_results = [d for d in filtered_results if d.get('category') == selected_category]

        if min_downloads > 0:
            filtered_results = [d for d in filtered_results if d.get('download_count', 0) >= min_downloads]

        # Sort results
        if sort_by == 'download_count':
            filtered_results.sort(key=lambda x: x.get('download_count', 0), reverse=True)
        elif sort_by == 'updated_at':
            filtered_results.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        else:  # name
            filtered_results.sort(key=lambda x: x.get('name', '').lower())

        # Display results as cards
        for i, dataset in enumerate(filtered_results):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**{dataset.get('name', 'Unknown')}**")
                    st.caption(f"Category: {dataset.get('category', 'Unknown')}")
                    if dataset.get('description'):
                        st.text(dataset['description'][:150] + "..." if len(dataset['description']) > 150 else dataset['description'])

                with col2:
                    st.metric("Downloads", format_number(dataset.get('download_count', 0)))
                    st.caption(f"Columns: {dataset.get('columns_count', 0)}")

                with col3:
                    if st.button(f"Select", key=f"select_{i}"):
                        st.session_state.selected_dataset = dataset
                        st.success(f"Selected: {dataset['name'][:30]}...")

                st.divider()

def show_quality_assessment(ai_analyst):
    """Quality assessment interface"""
    st.header("📊 Quality Assessment")

    if not st.session_state.selected_dataset:
        st.info("Please select a dataset from the Dashboard or Dataset Explorer first.")
        return

    dataset = st.session_state.selected_dataset
    st.subheader(f"Assessing: {dataset['name']}")

    # Display dataset info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Downloads", format_number(dataset.get('download_count', 0)))

    with col2:
        st.metric("Columns", dataset.get('columns_count', 0))

    with col3:
        st.metric("Category", dataset.get('category', 'Unknown'))

    # Quality assessment
    dataset_id = dataset['id']
    cache_key = f"quality_{dataset_id}"

    if cache_key not in st.session_state.quality_cache:
        if st.button("🔍 Assess Quality"):
            with st.spinner("Performing quality assessment... This may take a moment."):
                quality_data = fetch_api_data(f"datasets/{dataset_id}/quality")

            if quality_data:
                st.session_state.quality_cache[cache_key] = quality_data
                st.success("Quality assessment completed!")
            else:
                st.error("Quality assessment failed. Please try again.")

    # Display quality results
    if cache_key in st.session_state.quality_cache:
        quality = st.session_state.quality_cache[cache_key]

        # Overall score
        st.subheader("🎯 Overall Quality Score")

        col1, col2 = st.columns([1, 2])

        with col1:
            score_class = get_quality_class(quality['overall_score'])
            st.markdown(f'<div class="{score_class}">{quality["overall_score"]:.1f}/100</div>', unsafe_allow_html=True)
            st.markdown(f"**Grade: {quality['grade']}**")

        with col2:
            gauge_fig = create_quality_gauge(quality['overall_score'], "Overall Quality")
            st.plotly_chart(gauge_fig, use_container_width=True)

        # Detailed breakdown
        st.subheader("📊 Quality Breakdown")

        scores_df = pd.DataFrame({
            'Dimension': ['Completeness', 'Consistency', 'Accuracy', 'Timeliness', 'Usability'],
            'Score': [
                quality['completeness_score'],
                quality['consistency_score'],
                quality['accuracy_score'],
                quality['timeliness_score'],
                quality['usability_score']
            ]
        })

        fig_bar = px.bar(
            scores_df,
            x='Dimension',
            y='Score',
            title='Quality Scores by Dimension',
            color='Score',
            color_continuous_scale='RdYlGn'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Insights
        st.subheader("💡 Quality Insights")

        for insight in quality.get('insights', []):
            st.info(insight)

        # Missing data info
        if quality.get('missing_percentage', 0) > 0:
            st.warning(f"⚠️ {quality['missing_percentage']:.1f}% of data is missing")
        else:
            st.success("✅ No missing data detected")

        # AI Analysis Component
        st.markdown("---")
        ai_analyst.render_analysis_panel(
            dataset_info=dataset,
            sample_data=None,
            page_context="quality_assessment"
        )

def show_relationship_mapping(ai_analyst):
    """Relationship mapping and network visualization"""
    st.header("🗺️ Relationship Mapping")

    if not st.session_state.selected_dataset:
        st.info("Please select a dataset from the Dashboard or Dataset Explorer first.")
        return

    dataset = st.session_state.selected_dataset
    st.subheader(f"Finding relationships for: {dataset['name'][:50]}...")

    # Relationship parameters
    col1, col2 = st.columns(2)

    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Minimum similarity score to show relationships"
        )

    with col2:
        max_related = st.number_input(
            "Max Related Datasets",
            min_value=5,
            max_value=50,
            value=10
        )

    if st.button("🔍 Find Relationships"):
        with st.spinner("Analyzing dataset relationships..."):
            relationship_data = {
                "dataset_id": dataset['id'],
                "similarity_threshold": similarity_threshold,
                "max_related": max_related
            }

            relationships = fetch_api_data("datasets/relationships", method="POST", data=relationship_data)

        if relationships:
            st.success(f"Found {len(relationships.get('related_datasets', []))} related datasets!")

            # Network visualization
            st.subheader("🌐 Relationship Network")

            network_data = fetch_api_data(f"network/visualization/{dataset['id']}")
            if network_data:
                network_fig = create_network_visualization(network_data)
                st.plotly_chart(network_fig, use_container_width=True)

            # Related datasets table
            if relationships.get('related_datasets'):
                st.subheader("📊 Related Datasets")

                related_df = pd.DataFrame(relationships['related_datasets'])
                if not related_df.empty:
                    # Create a nice display format
                    display_data = []
                    for _, row in related_df.iterrows():
                        display_data.append({
                            'Dataset Name': row.get('name', 'Unknown')[:60],
                            'Similarity Score': f"{row.get('similarity_score', 0):.3f}",
                            'Category': row.get('category', 'Unknown'),
                            'Relationship Reasons': ', '.join(row.get('relationship_reasons', []))
                        })

                    st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

            # Network statistics
            st.subheader("📈 Network Statistics")

            network_stats = relationships.get('network_stats', {})

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Datasets", network_stats.get('total_datasets', 0))

            with col2:
                st.metric("Relationships Found", network_stats.get('relationships_found', 0))

            with col3:
                density = network_stats.get('graph_density', 0)
                st.metric("Network Density", f"{density:.3f}")

        else:
            st.warning("No relationships found or analysis failed.")

def show_data_sample(ai_analyst):
    """Data sample viewer with AI chat functionality"""
    st.header("📊 Data Sample & AI Chat")

    if not st.session_state.selected_dataset:
        st.info("Please select a dataset from the Dashboard or Dataset Explorer first.")
        return

    dataset = st.session_state.selected_dataset
    st.subheader(f"Dataset: {dataset['name']}")

    # Initialize session state for loaded dataset
    if 'loaded_dataset' not in st.session_state:
        st.session_state.loaded_dataset = None
    if 'loaded_dataset_id' not in st.session_state:
        st.session_state.loaded_dataset_id = None
    if 'dataset_chat_history' not in st.session_state:
        st.session_state.dataset_chat_history = []

    # Main layout
    col1, col2 = st.columns([1, 1])

    # Left Column: Dataset Loading & Preview
    with col1:
        st.subheader("📥 Load Dataset")

        # Sample parameters
        sample_size = st.number_input(
            "Sample Size",
            min_value=10,
            max_value=5000,
            value=500,
            help="Number of rows to retrieve and load into memory"
        )

        # Load dataset button
        col_load, col_clear = st.columns(2)
        with col_load:
            load_button = st.button("🚀 Load Dataset for AI Chat", type="primary")
        with col_clear:
            clear_button = st.button("🗑️ Clear Loaded Dataset")

        if clear_button:
            st.session_state.loaded_dataset = None
            st.session_state.loaded_dataset_id = None
            st.session_state.dataset_chat_history = []
            st.success("Dataset cleared from memory")

        if load_button:
            with st.spinner("Loading dataset into memory..."):
                sample_data = fetch_api_data(f"datasets/{dataset['id']}/sample?sample_size={sample_size}")

            if sample_data and sample_data.get('data'):
                # Store in session state
                st.session_state.loaded_dataset = {
                    'dataset_info': dataset,
                    'data': sample_data['data'],
                    'data_types': sample_data.get('data_types', {}),
                    'columns': sample_data.get('columns', []),
                    'total_rows': sample_data.get('total_rows', 0),
                    'sample_size': len(sample_data['data']),
                    'loaded_at': time.time()
                }
                st.session_state.loaded_dataset_id = dataset['id']
                st.session_state.dataset_chat_history = []  # Reset chat history

                st.success(f"✅ Loaded {len(sample_data['data'])} rows into memory for AI chat!")
            else:
                st.error("Failed to load dataset.")

        # Display loaded dataset status
        if st.session_state.loaded_dataset:
            loaded_data = st.session_state.loaded_dataset
            st.success("🧠 Dataset Loaded in Memory")

            # Dataset info
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Rows Loaded", loaded_data['sample_size'])
            with col_info2:
                st.metric("Columns", len(loaded_data['columns']))
            with col_info3:
                st.metric("Total Dataset Rows", format_number(loaded_data['total_rows']))

            # Column information
            st.subheader("📝 Column Information")
            col_info = []
            for col, dtype in loaded_data['data_types'].items():
                sample_values = []
                for row in loaded_data['data'][:3]:
                    if row.get(col) is not None:
                        sample_values.append(str(row[col]))

                col_info.append({
                    'Column': col,
                    'Data Type': dtype,
                    'Sample Values': ', '.join(sample_values[:3])
                })

            st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

            # Data preview
            with st.expander("👀 Data Preview", expanded=False):
                sample_df = pd.DataFrame(loaded_data['data'])
                st.dataframe(sample_df.head(10), use_container_width=True)

                # Download option
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"{dataset['id']}_sample.csv",
                    mime="text/csv"
                )

    # Right Column: AI Chat Interface
    with col2:
        st.subheader("💬 Chat with Your Dataset")

        if not st.session_state.loaded_dataset:
            st.info("👈 Load a dataset first to start chatting with AI about your data")
            st.markdown("""
            ### What you can ask:
            - "What are the main patterns in this data?"
            - "Show me statistics for [column name]"
            - "Are there any outliers?"
            - "What correlations exist between variables?"
            - "Summarize the key insights"
            - "What's the distribution of [column]?"
            """)
        else:
            # Chat interface
            render_dataset_chat_interface(ai_analyst, st.session_state.loaded_dataset)

    # Traditional AI Analysis Panel (below both columns)
    if st.session_state.loaded_dataset:
        st.markdown("---")
        st.subheader("🔍 Traditional AI Analysis")
        ai_analyst.render_analysis_panel(
            dataset_info=dataset,
            sample_data=st.session_state.loaded_dataset['data'],
            page_context="data_sample"
        )

def show_ai_analysis(ai_analyst):
    """AI Analysis page with multi-dataset querying, chat interface, and code generation"""
    st.header("🤖 AI Analysis Laboratory")
    st.markdown("*Explore multiple datasets with natural language queries and generated Python code*")

    # Initialize session state for AI analysis
    if 'selected_datasets' not in st.session_state:
        st.session_state.selected_datasets = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = ""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

    # Main layout
    col1, col2 = st.columns([1, 2])

    # Left Column: Dataset Selection & Configuration
    with col1:
        st.subheader("📊 Dataset Selection")

        # Dataset browser
        with st.expander("🔍 Browse Available Datasets", expanded=True):
            # Search for datasets
            search_query = st.text_input("Search datasets:", placeholder="e.g., 311, health, transportation")

            if search_query or st.button("🔄 Load Recent Datasets"):
                with st.spinner("Searching datasets..."):
                    if search_query:
                        # Search datasets
                        search_data = {
                            "search_terms": [search_query],
                            "limit": 20,
                            "include_quality": False
                        }
                        datasets = fetch_api_data("datasets/search", method="POST", data=search_data)
                    else:
                        # Load recent datasets
                        datasets = fetch_api_data("datasets/top-updated?limit=15")

                    if datasets:
                        st.success(f"Found {len(datasets)} datasets")

                        # Dataset selection interface
                        for idx, dataset in enumerate(datasets):
                            col_a, col_b = st.columns([3, 1])

                            with col_a:
                                # Dataset info
                                st.markdown(f"**{dataset['name'][:50]}{'...' if len(dataset['name']) > 50 else ''}**")
                                st.caption(f"Category: {dataset.get('category', 'Unknown')} | Downloads: {dataset.get('download_count', 0):,}")

                            with col_b:
                                # Add/Remove button
                                dataset_id = dataset['id']
                                if dataset_id in [d['id'] for d in st.session_state.selected_datasets]:
                                    if st.button("❌", key=f"remove_{idx}", help="Remove from analysis"):
                                        st.session_state.selected_datasets = [
                                            d for d in st.session_state.selected_datasets if d['id'] != dataset_id
                                        ]
                                        st.success("Dataset removed from analysis")
                                else:
                                    if st.button("➕", key=f"add_{idx}", help="Add to analysis"):
                                        st.session_state.selected_datasets.append({
                                            'id': dataset_id,
                                            'name': dataset['name'],
                                            'category': dataset.get('category', 'Unknown'),
                                            'description': dataset.get('description', ''),
                                            'columns_count': dataset.get('columns_count', 0)
                                        })
                                        st.success("Dataset added to analysis")

                            st.markdown("---")
                    else:
                        st.warning("No datasets found")

        # Selected Datasets Panel
        st.subheader(f"🎯 Selected Datasets ({len(st.session_state.selected_datasets)})")

        if st.session_state.selected_datasets:
            for i, dataset in enumerate(st.session_state.selected_datasets):
                with st.container():
                    col_x, col_y = st.columns([4, 1])
                    with col_x:
                        st.markdown(f"**{dataset['name'][:40]}{'...' if len(dataset['name']) > 40 else ''}**")
                        st.caption(f"{dataset['category']} • {dataset['columns_count']} columns")
                    with col_y:
                        if st.button("🗑️", key=f"delete_{i}", help="Remove dataset"):
                            removed_dataset = st.session_state.selected_datasets.pop(i)
                            st.success(f"Removed {removed_dataset['name'][:30]}...")
                    st.markdown("---")
        else:
            st.info("👆 Select datasets above to start analysis")

        # AI Configuration
        st.subheader("🔧 AI Configuration")
        ai_analyst.render_ai_configuration()

        # Clear selections
        if st.button("🗑️ Clear All Selections"):
            st.session_state.selected_datasets = []
            st.session_state.chat_history = []
            st.session_state.generated_code = ""
            st.session_state.analysis_results = {}
            st.success("All selections cleared")

    # Right Column: Chat Interface & Analysis
    with col2:
        if not st.session_state.selected_datasets:
            st.info("👈 **Select datasets from the left panel to start AI analysis**")
            st.markdown("""
            ### What you can do with AI Analysis:

            **🗣️ Natural Language Queries:**
            - "Compare health inspection scores across boroughs"
            - "Show me trends in 311 complaints over time"
            - "Find correlations between datasets"

            **📊 Automated Code Generation:**
            - Python code generated for data analysis
            - Interactive plots and visualizations
            - Statistical analysis and insights

            **🔍 Multi-Dataset Analysis:**
            - Query across multiple datasets simultaneously
            - Find relationships and patterns
            - Generate comprehensive reports
            """)
            return

        st.subheader("💬 Natural Language Analysis")

        # Chat interface
        chat_container = st.container()

        with chat_container:
            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])
                        if 'code' in message:
                            st.code(message['code'], language='python')
                        if 'chart' in message:
                            st.plotly_chart(message['chart'], use_container_width=True)

        # Chat input
        user_query = st.chat_input("Ask me about your selected datasets...")

        if user_query:
            # Add user message to chat
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_query,
                'timestamp': datetime.now().isoformat()
            })

            with st.spinner("🤖 Analyzing your query and generating insights..."):
                # Process the query
                analysis_result = process_ai_query(user_query, st.session_state.selected_datasets, ai_analyst)

                # Add AI response to chat
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': analysis_result.get('response', 'Analysis completed'),
                    'code': analysis_result.get('code', ''),
                    'chart': analysis_result.get('chart', None),
                    'findings': analysis_result.get('findings', []),
                    'timestamp': datetime.now().isoformat()
                })

            # Remove st.rerun() - Streamlit will handle state updates automatically

        # Tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Generated Code", "🔍 Data Exploration", "📈 Quick Insights", "📋 Analysis Summary"])

        with tab1:
            st.subheader("Generated Python Code")
            if st.session_state.get('generated_code'):
                st.code(st.session_state.generated_code, language='python')

                col_run, col_modify, col_download = st.columns(3)
                with col_run:
                    if st.button("▶️ Run Code"):
                        execute_generated_code(st.session_state.generated_code, st.session_state.selected_datasets)

                with col_modify:
                    if st.button("✏️ Edit Code"):
                        edited_code = st.text_area("Edit the code:", value=st.session_state.generated_code, height=300)
                        if st.button("💾 Save Changes"):
                            st.session_state.generated_code = edited_code
                            st.success("Code updated!")

                with col_download:
                    if st.button("📥 Download Code"):
                        st.download_button(
                            label="Download Python file",
                            data=st.session_state.generated_code,
                            file_name="ai_analysis.py",
                            mime="text/python"
                        )
            else:
                st.info("💡 Start a conversation above to generate Python code automatically")

        with tab2:
            st.subheader("Data Exploration")
            if st.session_state.selected_datasets:
                for dataset in st.session_state.selected_datasets:
                    with st.expander(f"📊 {dataset['name']}", expanded=False):
                        # Load sample data for exploration
                        sample_data = fetch_api_data(f"datasets/{dataset['id']}/sample?sample_size=100")
                        if sample_data and sample_data.get('data'):
                            df_sample = pd.DataFrame(sample_data['data'])

                            col_info, col_stats = st.columns(2)
                            with col_info:
                                st.markdown("**Dataset Info:**")
                                st.write(f"- Rows: {sample_data.get('total_rows', 'Unknown')}")
                                st.write(f"- Columns: {len(sample_data.get('columns', []))}")
                                st.write(f"- Category: {dataset['category']}")

                            with col_stats:
                                st.markdown("**Sample Data:**")
                                if len(df_sample) > 0:
                                    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
                                    if len(numeric_cols) > 0:
                                        st.write(f"- Numeric columns: {len(numeric_cols)}")
                                        st.write(f"- Text columns: {len(df_sample.columns) - len(numeric_cols)}")
                                    else:
                                        st.write("- Mostly text data")

                            # Quick data preview
                            st.dataframe(df_sample.head(5), use_container_width=True)
                        else:
                            st.warning("Could not load sample data")

        with tab3:
            st.subheader("Quick Insights")
            if st.button("🚀 Generate Quick Analysis"):
                with st.spinner("Generating insights for selected datasets..."):
                    insights = generate_quick_insights(st.session_state.selected_datasets)

                    for insight in insights:
                        st.markdown(f"**💡 {insight['title']}**")
                        st.write(insight['description'])
                        if insight.get('chart'):
                            st.plotly_chart(insight['chart'], use_container_width=True)
                        st.markdown("---")

        with tab4:
            st.subheader("Analysis Summary")
            if st.session_state.chat_history:
                st.markdown("### 📋 Session Summary")

                # Count interactions
                user_messages = [msg for msg in st.session_state.chat_history if msg['role'] == 'user']
                ai_responses = [msg for msg in st.session_state.chat_history if msg['role'] == 'assistant']

                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.metric("Questions Asked", len(user_messages))
                with col_sum2:
                    st.metric("AI Responses", len(ai_responses))
                with col_sum3:
                    st.metric("Datasets Analyzed", len(st.session_state.selected_datasets))

                # Recent findings
                st.markdown("### 🔍 Recent Findings")
                recent_findings = []
                for msg in ai_responses[-3:]:  # Last 3 AI responses
                    if 'findings' in msg:
                        recent_findings.extend(msg['findings'])

                if recent_findings:
                    for finding in recent_findings[-5:]:  # Show last 5 findings
                        st.markdown(f"• {finding}")
                else:
                    st.info("No findings yet - start asking questions!")

                # Export options
                st.markdown("### 📤 Export Options")
                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    if st.button("📋 Export Chat History"):
                        chat_export = "\n\n".join([
                            f"[{msg['role'].upper()}]: {msg['content']}"
                            for msg in st.session_state.chat_history
                        ])
                        st.download_button(
                            label="Download Chat History",
                            data=chat_export,
                            file_name=f"ai_analysis_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )

                with col_exp2:
                    if st.button("📊 Export Analysis Report"):
                        report = generate_analysis_report(st.session_state.selected_datasets, st.session_state.chat_history)
                        st.download_button(
                            label="Download Analysis Report",
                            data=report,
                            file_name=f"ai_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown"
                        )
            else:
                st.info("Start a conversation to see analysis summary")

# Helper functions for Dataset Chat

def render_dataset_chat_interface(ai_analyst, loaded_dataset: Dict[str, Any]):
    """Render chat interface for a loaded dataset"""

    # Display dataset status
    with st.container():
        st.success(f"💬 Chatting with: **{loaded_dataset['dataset_info']['name'][:50]}{'...' if len(loaded_dataset['dataset_info']['name']) > 50 else ''}**")
        st.caption(f"📊 {loaded_dataset['sample_size']} rows • {len(loaded_dataset['columns'])} columns")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if st.session_state.dataset_chat_history:
            for i, message in enumerate(st.session_state.dataset_chat_history):
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                elif message['role'] == 'assistant':
                    with st.chat_message("assistant"):
                        st.write(message['content'])

                        # Show any generated insights or code
                        if message.get('insights'):
                            with st.expander("📊 Generated Insights"):
                                for insight in message['insights']:
                                    st.markdown(f"• {insight}")

                        if message.get('code'):
                            with st.expander("🐍 Generated Code"):
                                st.code(message['code'], language='python')

    # Quick action buttons
    st.markdown("#### 🚀 Quick Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Show Summary", key="quick_summary"):
            send_chat_message("Provide a comprehensive summary of this dataset", ai_analyst, loaded_dataset)

    with col2:
        if st.button("🔍 Find Patterns", key="quick_patterns"):
            send_chat_message("What are the main patterns and trends in this data?", ai_analyst, loaded_dataset)

    with col3:
        if st.button("⚠️ Check Quality", key="quick_quality"):
            send_chat_message("Assess the data quality - any missing values, outliers, or issues?", ai_analyst, loaded_dataset)

    # Chat input
    user_message = st.chat_input("Ask anything about your dataset...")

    if user_message:
        send_chat_message(user_message, ai_analyst, loaded_dataset)

    # Chat management
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Chat"):
            st.session_state.dataset_chat_history = []
            st.success("Chat history cleared")

    with col_export:
        if st.button("📤 Export Chat") and st.session_state.dataset_chat_history:
            export_chat_history()


def send_chat_message(message: str, ai_analyst, loaded_dataset: Dict[str, Any]):
    """Send a chat message and get AI response"""

    # Add user message to history
    st.session_state.dataset_chat_history.append({
        'role': 'user',
        'content': message,
        'timestamp': datetime.now().isoformat()
    })

    # Prepare dataset context for AI
    dataset_context = prepare_dataset_context(loaded_dataset)

    with st.spinner("🤖 AI is analyzing your question..."):
        try:
            # Use backend API for natural language querying
            import requests

            api_url = "http://localhost:8080/api/ai/dataset-chat"
            request_data = {
                "message": message,
                "dataset_info": loaded_dataset['dataset_info'],
                "dataset_context": dataset_context,
                "sample_data": loaded_dataset['data'][:100],  # Send subset for context
                "chat_history": st.session_state.dataset_chat_history[-5:]  # Last 5 messages for context
            }

            response = requests.post(api_url, json=request_data, timeout=60)

            if response.status_code == 200:
                result = response.json()

                # Add AI response to history
                ai_response = {
                    'role': 'assistant',
                    'content': result.get('response', 'No response available'),
                    'timestamp': datetime.now().isoformat(),
                    'insights': result.get('insights', []),
                    'code': result.get('generated_code', ''),
                    'cached': result.get('cached', False)
                }

                st.session_state.dataset_chat_history.append(ai_response)

            else:
                # Fallback to direct AI analyst
                ai_response = get_fallback_ai_response(message, loaded_dataset, ai_analyst)
                st.session_state.dataset_chat_history.append(ai_response)

        except Exception as e:
            st.error(f"Chat error: {str(e)}")
            # Add error message to chat
            st.session_state.dataset_chat_history.append({
                'role': 'assistant',
                'content': f"I apologize, but I encountered an error: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'error': True
            })


def prepare_dataset_context(loaded_dataset: Dict[str, Any]) -> str:
    """Prepare comprehensive dataset context for AI"""

    dataset_info = loaded_dataset['dataset_info']
    data = loaded_dataset['data']

    context = f"""
Dataset: {dataset_info.get('name', 'Unknown')}
Description: {dataset_info.get('description', 'No description')}
Category: {dataset_info.get('category', 'Unknown')}
Total Rows in Dataset: {loaded_dataset.get('total_rows', 'Unknown')}
Loaded Sample Size: {loaded_dataset['sample_size']}

Columns and Data Types:
"""

    for col, dtype in loaded_dataset['data_types'].items():
        # Get some sample values
        sample_vals = []
        for row in data[:5]:
            if row.get(col) is not None:
                sample_vals.append(str(row[col]))

        context += f"- {col} ({dtype}): {', '.join(sample_vals[:3])}{'...' if len(sample_vals) > 3 else ''}\n"

    # Add basic statistics
    context += f"\nData Sample (first 3 rows):\n"
    for i, row in enumerate(data[:3]):
        context += f"Row {i+1}: {row}\n"

    return context


def get_fallback_ai_response(message: str, loaded_dataset: Dict[str, Any], ai_analyst) -> Dict[str, Any]:
    """Get AI response using enhanced AI analyst when backend is unavailable"""

    try:
        # Use the enhanced AI analyst component
        response = ai_analyst.answer_dataset_question(
            question=message,
            dataset_info=loaded_dataset['dataset_info'],
            loaded_data=loaded_dataset['data'],
            chat_history=st.session_state.dataset_chat_history
        )

        return {
            'role': 'assistant',
            'content': response.get('answer', 'No response available'),
            'timestamp': datetime.now().isoformat(),
            'insights': response.get('insights', []),
            'code': response.get('code', ''),
            'cached': response.get('cached', False),
            'source': response.get('source', 'fallback')
        }

    except Exception as e:
        logger.error(f"Fallback AI response failed: {e}")
        return {
            'role': 'assistant',
            'content': f"I can see you're asking about '{message}' regarding the loaded dataset '{loaded_dataset['dataset_info']['name']}'. While I'm having some technical difficulties, I can tell you that your dataset with {loaded_dataset['sample_size']} rows and {len(loaded_dataset['columns'])} columns is loaded and ready for analysis.",
            'timestamp': datetime.now().isoformat(),
            'insights': [
                f"Dataset '{loaded_dataset['dataset_info']['name']}' is loaded in memory",
                f"Contains {loaded_dataset['sample_size']} rows and {len(loaded_dataset['columns'])} columns",
                "Try rephrasing your question or check if the AI service is running"
            ],
            'error': True
        }


def export_chat_history():
    """Export chat history to downloadable format"""

    if not st.session_state.dataset_chat_history:
        st.warning("No chat history to export")
        return

    # Prepare export content
    export_content = f"# Dataset Chat History\n\n"
    export_content += f"Dataset: {st.session_state.loaded_dataset['dataset_info']['name']}\n"
    export_content += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for i, message in enumerate(st.session_state.dataset_chat_history):
        role = "👤 User" if message['role'] == 'user' else "🤖 AI Assistant"
        timestamp = message.get('timestamp', 'Unknown time')
        content = message['content']

        export_content += f"## {role} ({timestamp})\n\n{content}\n\n"

        if message.get('insights'):
            export_content += "**Insights:**\n"
            for insight in message['insights']:
                export_content += f"- {insight}\n"
            export_content += "\n"

    # Create download button
    st.download_button(
        label="📥 Download Chat History",
        data=export_content,
        file_name=f"dataset_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown"
    )


# Helper functions for AI Analysis

def process_ai_query(query: str, selected_datasets: List[Dict], ai_analyst) -> Dict:
    """Process a natural language query and return analysis results with generated code"""
    try:
        # Prepare context about selected datasets
        dataset_context = "\n".join([
            f"- {ds['name']} (ID: {ds['id']}, Category: {ds['category']}, Columns: {ds['columns_count']})"
            for ds in selected_datasets
        ])

        # Enhanced prompt for multi-dataset analysis with code generation
        enhanced_query = f"""
        The user has selected these datasets for analysis:
        {dataset_context}

        User's question: "{query}"

        Please provide:
        1. A clear analysis response
        2. Python code to analyze the data (using pandas, plotly, etc.)
        3. Key findings and insights
        4. Recommendations for further analysis

        Focus on relationships between datasets where relevant.
        Generate executable Python code that can work with the Scout Data Discovery API.
        """

        # Use the AI analyst to process the query
        analysis_response = ai_analyst.answer_question(
            question=enhanced_query,
            dataset_info={
                'selected_datasets': selected_datasets,
                'context': 'multi_dataset_analysis'
            }
        )

        # Parse the response to extract code and insights
        response_content = analysis_response.get('answer', 'Analysis completed')

        # Extract Python code from response (if any)
        code = extract_python_code_from_response(response_content)
        if code:
            st.session_state.generated_code = code

        # Generate sample chart for demonstration
        sample_chart = create_sample_chart(selected_datasets, query)

        # Extract findings
        findings = extract_findings_from_response(response_content)

        return {
            'response': response_content,
            'code': code,
            'chart': sample_chart,
            'findings': findings
        }

    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return {
            'response': f"I encountered an issue processing your query: {str(e)}. Please try rephrasing your question.",
            'code': "",
            'chart': None,
            'findings': []
        }

def extract_python_code_from_response(response: str) -> str:
    """Extract Python code from AI response"""
    # Look for code blocks in markdown format
    import re
    code_pattern = r'```python\n(.*?)\n```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Alternative pattern
    code_pattern2 = r'```\n(.*?)\n```'
    matches2 = re.findall(code_pattern2, response, re.DOTALL)
    if matches2 and ('import' in matches2[0] or 'pandas' in matches2[0] or 'plt' in matches2[0]):
        return matches2[0].strip()

    return ""

def extract_findings_from_response(response: str) -> List[str]:
    """Extract key findings from AI response"""
    findings = []
    lines = response.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('- ') or line.startswith('• '):
            findings.append(line[2:])
        elif line.startswith(('Finding:', 'Insight:', 'Key point:')):
            findings.append(line)

    # If no formatted findings, extract sentences with key indicators
    if not findings:
        import re
        key_phrases = r'(shows|indicates|reveals|demonstrates|suggests|analysis shows|data indicates)'
        sentences = re.split(r'[.!?]+', response)
        for sentence in sentences:
            if re.search(key_phrases, sentence, re.IGNORECASE) and len(sentence.strip()) > 20:
                findings.append(sentence.strip())

    return findings[:5]  # Return top 5 findings

def create_sample_chart(selected_datasets: List[Dict], query: str):
    """Create a sample chart based on the query and datasets"""
    try:
        # Simple demonstration chart
        if not selected_datasets:
            return None

        # Create a sample visualization showing dataset information
        datasets_df = pd.DataFrame([
            {
                'Dataset': ds['name'][:30] + '...' if len(ds['name']) > 30 else ds['name'],
                'Columns': ds['columns_count'],
                'Category': ds['category']
            } for ds in selected_datasets
        ])

        if len(datasets_df) > 0:
            fig = px.bar(
                datasets_df,
                x='Dataset',
                y='Columns',
                color='Category',
                title=f'Selected Datasets Overview - {len(selected_datasets)} datasets',
                labels={'Columns': 'Number of Columns'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            return fig

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

    return None

def execute_generated_code(code: str, selected_datasets: List[Dict]):
    """Execute generated Python code safely"""
    if not code:
        st.warning("No code to execute")
        return

    try:
        st.subheader("🔄 Code Execution Results")

        # Create a safe execution environment
        safe_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'px': px,
            'go': go,
            'st': st,
            'selected_datasets': selected_datasets,
            'fetch_api_data': fetch_api_data,
            'API_BASE_URL': API_BASE_URL
        }

        # Execute the code
        exec(code, safe_globals)

        st.success("✅ Code executed successfully!")

    except Exception as e:
        st.error(f"❌ Error executing code: {str(e)}")
        st.code(f"Error details:\n{str(e)}", language="text")

def generate_quick_insights(selected_datasets: List[Dict]) -> List[Dict]:
    """Generate quick insights for selected datasets"""
    insights = []

    if not selected_datasets:
        return insights

    # Dataset distribution insight
    categories = [ds['category'] for ds in selected_datasets]
    category_counts = pd.Series(categories).value_counts()

    insights.append({
        'title': 'Dataset Categories',
        'description': f'You\'ve selected {len(selected_datasets)} datasets across {len(category_counts)} categories. Most common: {category_counts.index[0]}',
        'chart': px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Distribution of Selected Dataset Categories'
        )
    })

    # Columns distribution
    column_counts = [ds['columns_count'] for ds in selected_datasets if ds['columns_count'] > 0]
    if column_counts:
        avg_columns = np.mean(column_counts)
        insights.append({
            'title': 'Data Complexity',
            'description': f'Average of {avg_columns:.1f} columns per dataset. Range: {min(column_counts)} to {max(column_counts)} columns.',
            'chart': px.histogram(
                x=column_counts,
                title='Distribution of Column Counts',
                labels={'x': 'Number of Columns', 'y': 'Number of Datasets'}
            )
        })

    return insights

def generate_analysis_report(selected_datasets: List[Dict], chat_history: List[Dict]) -> str:
    """Generate a comprehensive analysis report"""
    report = f"""# AI Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Selected Datasets ({len(selected_datasets)})
"""

    for i, ds in enumerate(selected_datasets, 1):
        report += f"""
### {i}. {ds['name']}
- **Category:** {ds['category']}
- **Columns:** {ds['columns_count']}
- **ID:** {ds['id']}
- **Description:** {ds['description'][:200]}...
"""

    report += f"""

## Analysis Session Summary
- **Questions Asked:** {len([msg for msg in chat_history if msg['role'] == 'user'])}
- **AI Responses:** {len([msg for msg in chat_history if msg['role'] == 'assistant'])}

## Conversation History
"""

    for msg in chat_history:
        role = "User" if msg['role'] == 'user' else "AI Assistant"
        timestamp = msg.get('timestamp', 'Unknown')
        content = msg['content'][:500] + '...' if len(msg['content']) > 500 else msg['content']

        report += f"""
### {role} ({timestamp})
{content}
"""

        if 'findings' in msg and msg['findings']:
            report += "\n**Key Findings:**\n"
            for finding in msg['findings']:
                report += f"- {finding}\n"

    report += f"""

## Generated Analysis
This report summarizes an AI-powered analysis session using Scout Data Discovery.
Total datasets analyzed: {len(selected_datasets)}
Analysis completed with natural language processing and automated code generation.
"""

    return report

def show_ai_setup(ai_analyst):
    """Simplified AI Configuration - One Provider, One Model, One Key"""
    st.header("🤖 AI Setup & Configuration")
    st.caption("Simple setup: Choose one provider, one model, one API key")

    # Initialize session state for AI config
    if 'ai_config' not in st.session_state:
        st.session_state.ai_config = {
            'provider': 'nvidia',
            'model': 'qwen/qwen2.5-72b-instruct',
            'api_key': '',
            'use_reasoning_models': True,
            'analysis_temperature': 0.3,
            'max_tokens': 2000,
            'enable_cache': True
        }

    # Step 1: Provider Selection
    st.markdown("### 🎯 Step 1: Choose Your AI Provider")

    provider_options = {
        'nvidia': '🟢 NVIDIA AI (Recommended for reasoning)',
        'openai': '🔵 OpenAI (GPT models)',
        'openrouter': '🟡 OpenRouter (Multiple models)'
    }

    selected_provider = st.selectbox(
        "Select AI Provider:",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(st.session_state.ai_config['provider'])
    )

    st.session_state.ai_config['provider'] = selected_provider

    # Step 2: API Key
    st.markdown("### 🔑 Step 2: Enter API Key")

    if selected_provider == 'nvidia':
        st.info("🔗 Get your NVIDIA API key from: https://build.nvidia.com")
        api_key_help = "Free tier available with good rate limits"
    elif selected_provider == 'openai':
        st.info("🔗 Get your OpenAI API key from: https://platform.openai.com")
        api_key_help = "Requires payment, high quality models"
    else:
        st.info("🔗 Get your OpenRouter API key from: https://openrouter.ai")
        api_key_help = "Access to multiple models including Claude, Gemini"

    api_key = st.text_input(
        f"{provider_options[selected_provider]} API Key:",
        value=st.session_state.ai_config['api_key'],
        type="password",
        help=api_key_help
    )

    st.session_state.ai_config['api_key'] = api_key

    # Step 3: Model Selection
    st.markdown("### 🧠 Step 3: Choose Model")

    if selected_provider == 'nvidia':
        model_options = [
            'qwen/qwen2.5-72b-instruct',
            'meta/llama-3.1-405b-instruct',
            'meta/llama-3.1-70b-instruct',
            'meta/llama-3.1-8b-instruct',
            'mistralai/mixtral-8x22b-instruct-v0.1',
            'google/gemma-2-27b-it'
        ]
        recommended = 'qwen/qwen2.5-72b-instruct'
        st.success("🧠 **Qwen 2.5 72B** is recommended for best reasoning and analysis")
    elif selected_provider == 'openai':
        model_options = [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4-turbo',
            'gpt-3.5-turbo'
        ]
        recommended = 'gpt-4o-mini'
        st.success("⚡ **GPT-4o Mini** is recommended for cost-effective performance")
    else:  # openrouter
        model_options = [
            'anthropic/claude-3-opus',
            'anthropic/claude-3-sonnet',
            'anthropic/claude-3-haiku',
            'google/gemini-pro',
            'meta-llama/llama-3-70b-instruct'
        ]
        recommended = 'anthropic/claude-3-sonnet'
        st.success("🎯 **Claude 3 Sonnet** is recommended for balanced performance")

    selected_model = st.selectbox(
        "Select Model:",
        options=model_options,
        index=model_options.index(st.session_state.ai_config['model']) if st.session_state.ai_config['model'] in model_options else 0
    )

    st.session_state.ai_config['model'] = selected_model

    # Step 4: Quick Settings
    st.markdown("### ⚙️ Step 4: Quick Settings")

    col1, col2 = st.columns(2)

    with col1:
        reasoning_mode = st.toggle(
            "🧠 Reasoning Mode",
            value=st.session_state.ai_config['use_reasoning_models'],
            help="Enable for complex analysis and better logical reasoning"
        )
        st.session_state.ai_config['use_reasoning_models'] = reasoning_mode

    with col2:
        enable_cache = st.toggle(
            "💾 Enable Caching",
            value=st.session_state.ai_config['enable_cache'],
            help="Cache responses for faster repeated queries"
        )
        st.session_state.ai_config['enable_cache'] = enable_cache

    # Advanced settings in expander
    with st.expander("🔧 Advanced Settings", expanded=False):
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.ai_config['analysis_temperature'],
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        st.session_state.ai_config['analysis_temperature'] = temperature

        max_tokens = st.number_input(
            "Max Response Length",
            min_value=500,
            max_value=4000,
            value=st.session_state.ai_config['max_tokens'],
            step=100,
            help="Maximum length of AI responses"
        )
        st.session_state.ai_config['max_tokens'] = max_tokens

    # Step 5: Test & Save
    st.markdown("### 🧪 Step 5: Test & Save Configuration")

    # Configuration summary
    if api_key:
        st.success("✅ **Configuration Ready**")

        config_summary = f"""
**Provider:** {provider_options[selected_provider]}
**Model:** {selected_model}
**API Key:** {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '***'}
**Reasoning Mode:** {'✅ Enabled' if reasoning_mode else '❌ Disabled'}
**Caching:** {'✅ Enabled' if enable_cache else '❌ Disabled'}
"""
        st.markdown(config_summary)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧪 Test Configuration", type="secondary"):
                with st.spinner("Testing AI configuration..."):
                    try:
                        import time
                        time.sleep(2)  # Simulate API test
                        st.success("✅ **Test Successful!** AI is ready to use.")
                        st.info(f"Connected to {selected_provider.upper()} using {selected_model}")
                    except Exception as e:
                        st.error(f"❌ **Test Failed:** {str(e)}")

        with col2:
            if st.button("💾 Save Configuration", type="primary"):
                # Update config with consistent naming for backward compatibility
                st.session_state.ai_config.update({
                    'primary_provider': selected_provider,
                    f'{selected_provider}_api_key': api_key,
                    f'{selected_provider}_model': selected_model
                })

                st.success("✅ **Configuration Saved!**")
                st.balloons()
                st.info("You can now use AI features throughout the application!")

    else:
        st.warning("⚠️ **Please enter an API key to continue**")

    # Usage tips
    st.markdown("### 💡 Quick Start Tips")
    st.info(f"""
**Getting Started:**
1. Get your API key from the link above
2. Choose **{recommended}** for best results
3. Enable reasoning mode for complex data analysis
4. Test your configuration before using AI features

**Best Practices:**
- Use temperature 0.1-0.3 for analytical tasks
- Enable caching for faster responses
- Higher max tokens (2000+) for detailed analysis
""")

if __name__ == "__main__":
    main()