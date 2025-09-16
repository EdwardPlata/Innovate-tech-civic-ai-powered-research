"""
AI Data Explorer Streamlit Component

Professional data exploration interface using NVIDIA AI statistician
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Any
import io
import sys
import traceback

logger = logging.getLogger(__name__)


class DataExplorerComponent:
    """Streamlit component for AI-powered data exploration"""

    def __init__(self):
        self.backend_url = "http://localhost:8080"

    def render_data_explorer(self):
        """Render the main data exploration interface"""

        st.header("🔬 AI Data Explorer")
        st.caption("Professional statistical analysis powered by NVIDIA AI")

        # Check AI configuration
        if 'ai_config' not in st.session_state or not st.session_state.ai_config.get('api_key'):
            st.warning("⚠️ **AI Not Configured** - Please configure AI in the AI Setup page first.")
            if st.button("🚀 Go to AI Setup"):
                st.switch_page("pages/ai_setup.py")
            return

        # Initialize data explorer if not exists
        if 'data_explorer_state' not in st.session_state:
            st.session_state.data_explorer_state = {
                'datasets': {},
                'analysis_history': [],
                'current_analysis': None
            }

        # Main interface tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dataset Upload", "🔍 Natural Language Explorer",
            "📈 Visualizations", "🔗 Dataset Joining"
        ])

        with tab1:
            self._render_dataset_upload()

        with tab2:
            self._render_natural_language_explorer()

        with tab3:
            self._render_visualization_creator()

        with tab4:
            self._render_dataset_joiner()

        # Analysis history sidebar
        self._render_analysis_history()

    def _render_dataset_upload(self):
        """Render dataset upload interface"""

        st.markdown("### 📁 Upload Your Datasets")

        uploaded_files = st.file_uploader(
            "Choose CSV files to analyze",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload one or more CSV files for AI-powered analysis"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Read the CSV file
                    df = pd.read_csv(uploaded_file)

                    # Store in session state
                    dataset_name = uploaded_file.name.replace('.csv', '')
                    st.session_state.data_explorer_state['datasets'][dataset_name] = df

                    # Display dataset info
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
                    with col4:
                        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

                    # Show dataset preview
                    with st.expander(f"📊 Preview: {dataset_name}", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)

                        # Basic info
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Column Information:**")
                            info_df = pd.DataFrame({
                                'Column': df.columns,
                                'Type': df.dtypes.astype(str),
                                'Non-Null': df.count(),
                                'Missing %': (df.isnull().sum() / len(df) * 100).round(1)
                            })
                            st.dataframe(info_df, use_container_width=True)

                        with col2:
                            st.markdown("**Quick Stats:**")
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                            else:
                                st.info("No numeric columns for statistical summary")

                    st.success(f"✅ Successfully loaded {dataset_name} ({len(df):,} rows, {len(df.columns)} columns)")

                except Exception as e:
                    st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")

        # Display currently loaded datasets
        if st.session_state.data_explorer_state['datasets']:
            st.markdown("### 📚 Loaded Datasets")
            datasets = st.session_state.data_explorer_state['datasets']

            for name, df in datasets.items():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

                with col1:
                    st.write(f"**{name}**")
                with col2:
                    st.write(f"{len(df):,} rows")
                with col3:
                    st.write(f"{len(df.columns)} cols")
                with col4:
                    if st.button("🔍 Quick EDA", key=f"eda_{name}"):
                        self._run_quick_eda(df, name)
                with col5:
                    if st.button("🗑️ Remove", key=f"remove_{name}"):
                        del st.session_state.data_explorer_state['datasets'][name]
                        st.rerun()

    def _render_natural_language_explorer(self):
        """Render natural language data exploration interface"""

        st.markdown("### 🗣️ Ask Questions About Your Data")

        datasets = st.session_state.data_explorer_state['datasets']

        if not datasets:
            st.info("📊 Please upload datasets in the 'Dataset Upload' tab first.")
            return

        # Dataset selection
        selected_dataset = st.selectbox(
            "Select dataset to explore:",
            options=list(datasets.keys()),
            help="Choose which dataset to analyze with natural language questions"
        )

        if selected_dataset:
            df = datasets[selected_dataset]

            # Show dataset overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Dataset", selected_dataset)
            with col2:
                st.metric("Rows", f"{len(df):,}")
            with col3:
                st.metric("Columns", len(df.columns))
            with col4:
                st.metric("Data Types", len(df.dtypes.unique()))

            # Question input
            st.markdown("### 💬 Ask Your Question")

            example_questions = [
                "What are the key patterns in this data?",
                "Show me the correlation between variables",
                "Are there any outliers I should be aware of?",
                "What insights can you provide about the distributions?",
                "How do different categories compare statistically?",
                "What trends can you identify in the time series data?",
                "Perform a comprehensive statistical analysis",
                "What are the most important features in this dataset?"
            ]

            selected_example = st.selectbox(
                "Choose an example question or write your own:",
                options=[""] + example_questions,
                help="Select a pre-made question or write your own below"
            )

            question = st.text_area(
                "Your question about the data:",
                value=selected_example,
                height=100,
                placeholder="e.g., What are the main factors that influence the target variable? Show me visualizations and statistical tests.",
                help="Ask any question about your data. The AI statistician will provide professional analysis with code and visualizations."
            )

            # Advanced options
            with st.expander("🔧 Advanced Options", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    include_other_datasets = st.multiselect(
                        "Include other datasets for joining:",
                        options=[name for name in datasets.keys() if name != selected_dataset],
                        help="Select other datasets that might be relevant for joining or comparison"
                    )

                with col2:
                    analysis_depth = st.selectbox(
                        "Analysis depth:",
                        options=["Standard", "Detailed", "Comprehensive"],
                        index=1,
                        help="Choose how thorough the analysis should be"
                    )

            # Analyze button
            if st.button("🧠 Analyze with AI Statistician", type="primary", disabled=not question):
                self._run_natural_language_analysis(df, selected_dataset, question, include_other_datasets, analysis_depth)

    def _render_visualization_creator(self):
        """Render visualization creation interface"""

        st.markdown("### 📈 AI-Powered Visualization Creator")

        datasets = st.session_state.data_explorer_state['datasets']

        if not datasets:
            st.info("📊 Please upload datasets in the 'Dataset Upload' tab first.")
            return

        # Dataset selection
        selected_dataset = st.selectbox(
            "Select dataset for visualization:",
            options=list(datasets.keys()),
            key="viz_dataset"
        )

        if selected_dataset:
            df = datasets[selected_dataset]

            # Chart request
            st.markdown("### 🎨 Describe Your Visualization")

            example_requests = [
                "Create an interactive scatter plot showing the relationship between two key variables",
                "Show me a comprehensive dashboard with the most important visualizations",
                "Generate a correlation heatmap with statistical significance",
                "Create time series plots with trend analysis",
                "Make box plots comparing different groups with statistical tests",
                "Design a histogram with distribution fitting and statistical annotations",
                "Build an interactive dashboard for exploring categorical relationships",
                "Show me outlier detection plots with statistical boundaries"
            ]

            selected_example = st.selectbox(
                "Choose an example or describe your own:",
                options=[""] + example_requests,
                key="viz_examples"
            )

            chart_request = st.text_area(
                "Describe the visualization you want:",
                value=selected_example,
                height=100,
                placeholder="e.g., Create an interactive scatter plot with regression lines and confidence intervals",
                help="Describe what kind of chart or visualization you want. The AI will generate professional plotly code."
            )

            # Visualization options
            with st.expander("🎨 Visualization Options", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    color_scheme = st.selectbox(
                        "Color scheme:",
                        options=["Default", "Viridis", "Plasma", "Blues", "Reds", "Greens"],
                        help="Choose a color palette for the visualization"
                    )

                    interactive_features = st.multiselect(
                        "Interactive features:",
                        options=["Hover details", "Zoom", "Selection", "Animation", "Filtering"],
                        default=["Hover details", "Zoom"]
                    )

                with col2:
                    chart_style = st.selectbox(
                        "Chart style:",
                        options=["Professional", "Scientific", "Business", "Minimal"],
                        index=0
                    )

                    export_formats = st.multiselect(
                        "Export formats:",
                        options=["HTML", "PNG", "PDF", "SVG"],
                        default=["HTML"]
                    )

            # Create visualization button
            if st.button("🎨 Create Visualization", type="primary", disabled=not chart_request):
                self._run_visualization_creation(df, selected_dataset, chart_request, {
                    "color_scheme": color_scheme,
                    "interactive_features": interactive_features,
                    "chart_style": chart_style,
                    "export_formats": export_formats
                })

    def _render_dataset_joiner(self):
        """Render dataset joining interface"""

        st.markdown("### 🔗 Intelligent Dataset Joining")

        datasets = st.session_state.data_explorer_state['datasets']

        if len(datasets) < 2:
            st.info("📊 Please upload at least 2 datasets to enable joining functionality.")
            return

        # Primary dataset selection
        primary_dataset = st.selectbox(
            "Select primary dataset:",
            options=list(datasets.keys()),
            key="join_primary"
        )

        # Datasets to join
        datasets_to_join = st.multiselect(
            "Select datasets to join:",
            options=[name for name in datasets.keys() if name != primary_dataset],
            key="join_datasets"
        )

        if primary_dataset and datasets_to_join:
            # Join objective
            st.markdown("### 🎯 Join Objective")

            join_examples = [
                "Combine customer data with transaction history for analysis",
                "Merge product information with sales data for insights",
                "Join demographic data with survey responses",
                "Combine time series data from different sources",
                "Merge financial data with market indicators",
                "Join location data with performance metrics"
            ]

            selected_objective = st.selectbox(
                "Choose join objective or describe your own:",
                options=[""] + join_examples,
                key="join_objectives"
            )

            join_objective = st.text_area(
                "Describe what you want to achieve by joining these datasets:",
                value=selected_objective,
                height=80,
                placeholder="e.g., I want to analyze how customer demographics relate to purchasing patterns"
            )

            analysis_goal = st.text_area(
                "What analysis do you plan to do with the joined data?",
                height=80,
                placeholder="e.g., Perform regression analysis to predict customer lifetime value"
            )

            # Preview datasets
            with st.expander("📊 Dataset Previews", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Primary: {primary_dataset}**")
                    primary_df = datasets[primary_dataset]
                    st.dataframe(primary_df.head(), use_container_width=True)
                    st.write(f"Columns: {list(primary_df.columns)}")

                with col2:
                    for join_name in datasets_to_join:
                        st.markdown(f"**Join: {join_name}**")
                        join_df = datasets[join_name]
                        st.dataframe(join_df.head(), use_container_width=True)
                        st.write(f"Columns: {list(join_df.columns)}")

            # Join analysis button
            if st.button("🧠 Analyze Join Strategy", type="primary",
                        disabled=not (join_objective and analysis_goal)):
                self._run_join_analysis(
                    primary_dataset, datasets_to_join,
                    join_objective, analysis_goal
                )

    def _render_analysis_history(self):
        """Render analysis history in sidebar"""

        with st.sidebar:
            st.markdown("### 📊 Analysis History")

            history = st.session_state.data_explorer_state['analysis_history']

            if not history:
                st.info("No analyses performed yet")
                return

            for i, analysis in enumerate(reversed(history[-10:])):  # Show last 10
                with st.expander(f"Analysis {len(history) - i}", expanded=False):
                    st.write(f"**Type**: {analysis.get('type', 'Unknown')}")
                    st.write(f"**Dataset**: {analysis.get('dataset', 'Unknown')}")
                    st.write(f"**Time**: {analysis.get('timestamp', 'Unknown')}")

                    if st.button("View Details", key=f"view_{len(history) - i}"):
                        st.session_state.data_explorer_state['current_analysis'] = analysis
                        self._display_analysis_details(analysis)

    def _run_natural_language_analysis(self, df, dataset_name, question, other_datasets, depth):
        """Run natural language analysis using AI"""

        with st.spinner("🧠 AI Statistician is analyzing your data..."):
            try:
                # Simulate AI analysis (replace with actual backend call)
                analysis_result = self._simulate_ai_analysis(df, dataset_name, question, "natural_language")

                # Display results
                self._display_analysis_results(analysis_result)

                # Store in history
                st.session_state.data_explorer_state['analysis_history'].append({
                    'type': 'natural_language',
                    'dataset': dataset_name,
                    'question': question,
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'result': analysis_result
                })

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Natural language analysis failed: {e}")

    def _run_visualization_creation(self, df, dataset_name, chart_request, options):
        """Run visualization creation using AI"""

        with st.spinner("🎨 Creating professional visualization..."):
            try:
                # Simulate AI visualization (replace with actual backend call)
                viz_result = self._simulate_visualization_creation(df, dataset_name, chart_request)

                # Display results
                self._display_visualization_results(viz_result)

                # Store in history
                st.session_state.data_explorer_state['analysis_history'].append({
                    'type': 'visualization',
                    'dataset': dataset_name,
                    'request': chart_request,
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'result': viz_result
                })

            except Exception as e:
                st.error(f"❌ Visualization creation failed: {str(e)}")
                logger.error(f"Visualization creation failed: {e}")

    def _run_join_analysis(self, primary_dataset, datasets_to_join, join_objective, analysis_goal):
        """Run dataset join analysis using AI"""

        with st.spinner("🔗 AI is analyzing join strategies..."):
            try:
                datasets = st.session_state.data_explorer_state['datasets']

                # Get datasets
                primary_df = datasets[primary_dataset]
                join_dfs = {name: datasets[name] for name in datasets_to_join}

                # Simulate AI join analysis (replace with actual backend call)
                join_result = self._simulate_join_analysis(
                    primary_df, primary_dataset, join_dfs, join_objective, analysis_goal
                )

                # Display results
                self._display_join_results(join_result)

                # Store in history
                st.session_state.data_explorer_state['analysis_history'].append({
                    'type': 'dataset_joining',
                    'primary_dataset': primary_dataset,
                    'datasets_to_join': datasets_to_join,
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'result': join_result
                })

            except Exception as e:
                st.error(f"❌ Join analysis failed: {str(e)}")
                logger.error(f"Join analysis failed: {e}")

    def _run_quick_eda(self, df, dataset_name):
        """Run quick EDA on a dataset"""

        with st.spinner("🔍 Running quick EDA..."):
            try:
                # Generate quick EDA
                eda_result = self._generate_quick_eda(df, dataset_name)

                # Display results
                self._display_eda_results(eda_result)

            except Exception as e:
                st.error(f"❌ Quick EDA failed: {str(e)}")

    def _simulate_ai_analysis(self, df, dataset_name, question, analysis_type):
        """Simulate AI analysis (replace with actual DataExplorer call)"""

        # This would be replaced with actual DataExplorer calls
        return {
            "executive_summary": f"Analysis of {dataset_name} reveals key patterns and insights related to: {question}",
            "methodology": "Professional statistical analysis using pandas and advanced statistical methods",
            "python_code": self._generate_sample_code(df, "analysis"),
            "insights": "Key insights and recommendations based on statistical analysis",
            "visualizations": ["correlation_heatmap", "distribution_plots", "statistical_summary"],
            "dataset_name": dataset_name,
            "analysis_type": analysis_type
        }

    def _simulate_visualization_creation(self, df, dataset_name, chart_request):
        """Simulate visualization creation"""

        return {
            "visualization_strategy": f"Creating professional visualization for: {chart_request}",
            "python_code": self._generate_sample_code(df, "visualization"),
            "chart_type": "interactive_plotly",
            "interpretation": "Guide to reading and understanding the visualization",
            "alternatives": ["histogram", "scatter_plot", "box_plot"]
        }

    def _simulate_join_analysis(self, primary_df, primary_name, join_dfs, objective, goal):
        """Simulate join analysis"""

        return {
            "join_strategy": f"Recommended join approach for {objective}",
            "python_code": self._generate_sample_code(primary_df, "joining"),
            "join_keys": ["id", "date", "category"],
            "quality_assessment": "Join quality and data loss analysis",
            "recommendations": "Next steps for analysis with joined data"
        }

    def _generate_sample_code(self, df, code_type):
        """Generate sample Python code for demonstration"""

        if code_type == "analysis":
            return f"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('your_data.csv')

# Basic statistics
print("Dataset shape:", df.shape)
print("\\nDescriptive statistics:")
print(df.describe())

# Correlation analysis
correlation_matrix = df.select_dtypes(include=[np.number]).corr()

# Create correlation heatmap
fig = px.imshow(correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap")
fig.show()

# Distribution analysis
for col in df.select_dtypes(include=[np.number]).columns:
    fig = px.histogram(df, x=col, title=f"Distribution of {{col}}")
    fig.show()
"""

        elif code_type == "visualization":
            return f"""
import plotly.express as px
import plotly.graph_objects as go

# Create interactive scatter plot
fig = px.scatter(df,
                x='{df.columns[0]}',
                y='{df.columns[1] if len(df.columns) > 1 else df.columns[0]}',
                title='Interactive Scatter Plot with Statistical Analysis')

# Add trend line
fig.update_traces(mode='markers+lines')

# Update layout for professional appearance
fig.update_layout(
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)

fig.show()
"""

        elif code_type == "joining":
            return f"""
import pandas as pd

# Load datasets
primary_df = pd.read_csv('primary_dataset.csv')
secondary_df = pd.read_csv('secondary_dataset.csv')

# Analyze potential join keys
print("Primary dataset columns:", primary_df.columns.tolist())
print("Secondary dataset columns:", secondary_df.columns.tolist())

# Identify common columns
common_cols = set(primary_df.columns).intersection(set(secondary_df.columns))
print("Common columns:", common_cols)

# Perform join
joined_df = primary_df.merge(secondary_df,
                           on='id',  # Replace with actual join key
                           how='left',  # Adjust join type as needed
                           suffixes=('', '_y'))

print("Joined dataset shape:", joined_df.shape)
print("Join success rate:", (len(joined_df) / len(primary_df)) * 100, "%")
"""

        return "# Sample code will be generated by AI statistician"

    def _display_analysis_results(self, result):
        """Display analysis results"""

        st.markdown("## 📊 AI Statistical Analysis Results")

        # Executive Summary
        if result.get("executive_summary"):
            st.markdown("### 📋 Executive Summary")
            st.info(result["executive_summary"])

        # Methodology
        if result.get("methodology"):
            st.markdown("### 🔬 Statistical Methodology")
            st.write(result["methodology"])

        # Python Code
        if result.get("python_code"):
            st.markdown("### 💻 Generated Python Code")
            st.code(result["python_code"], language="python")

            # Code execution button
            if st.button("▶️ Execute Code"):
                self._execute_code(result["python_code"])

        # Insights
        if result.get("insights"):
            st.markdown("### 💡 Key Insights")
            st.success(result["insights"])

    def _display_visualization_results(self, result):
        """Display visualization results"""

        st.markdown("## 🎨 AI-Generated Visualization")

        # Strategy
        if result.get("visualization_strategy"):
            st.markdown("### 📈 Visualization Strategy")
            st.info(result["visualization_strategy"])

        # Python Code
        if result.get("python_code"):
            st.markdown("### 💻 Plotly Code")
            st.code(result["python_code"], language="python")

            if st.button("▶️ Generate Chart"):
                self._execute_code(result["python_code"])

        # Interpretation
        if result.get("interpretation"):
            st.markdown("### 📖 How to Read This Chart")
            st.write(result["interpretation"])

    def _display_join_results(self, result):
        """Display join analysis results"""

        st.markdown("## 🔗 Dataset Join Analysis")

        # Strategy
        if result.get("join_strategy"):
            st.markdown("### 🎯 Join Strategy")
            st.info(result["join_strategy"])

        # Python Code
        if result.get("python_code"):
            st.markdown("### 💻 Join Implementation Code")
            st.code(result["python_code"], language="python")

        # Quality Assessment
        if result.get("quality_assessment"):
            st.markdown("### 📊 Join Quality Assessment")
            st.write(result["quality_assessment"])

    def _execute_code(self, code):
        """Execute Python code safely"""

        try:
            # Create a safe execution environment
            # WARNING: In production, this should be sandboxed properly

            # Get current datasets
            datasets = st.session_state.data_explorer_state['datasets']

            if datasets:
                # Use first dataset as df for code execution
                df = list(datasets.values())[0]

                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()

                # Create safe namespace
                namespace = {
                    'df': df,
                    'pd': pd,
                    'px': px,
                    'go': go,
                    'make_subplots': make_subplots,
                    'st': st,
                    'print': print
                }

                # Execute code
                exec(code, namespace)

                # Get output
                output = buffer.getvalue()
                sys.stdout = old_stdout

                if output:
                    st.text(output)

            else:
                st.warning("No datasets available for code execution")

        except Exception as e:
            st.error(f"Code execution failed: {str(e)}")
            st.code(traceback.format_exc(), language="text")

    def _generate_quick_eda(self, df, dataset_name):
        """Generate quick EDA summary"""

        # Basic statistics
        basic_stats = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().sum(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024
        }

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        return {
            "basic_stats": basic_stats,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "data_types": df.dtypes.value_counts().to_dict()
        }

    def _display_eda_results(self, result):
        """Display EDA results"""

        st.markdown("### 🔍 Quick EDA Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows × Columns", f"{result['basic_stats']['shape'][0]:,} × {result['basic_stats']['shape'][1]}")
        with col2:
            st.metric("Missing Values", f"{result['basic_stats']['missing_values']:,}")
        with col3:
            st.metric("Duplicates", f"{result['basic_stats']['duplicates']:,}")
        with col4:
            st.metric("Memory", f"{result['basic_stats']['memory_usage']:.1f} MB")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Numeric Columns", result['numeric_columns'])
        with col2:
            st.metric("Categorical Columns", result['categorical_columns'])


# Global instance
@st.cache_resource
def get_data_explorer_component():
    """Get cached data explorer component"""
    return DataExplorerComponent()