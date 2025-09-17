"""
Dataset Selector Widget for Scout Data Discovery integration
"""
import dash
from dash import dcc, html, callback, Input, Output, State, ctx
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from ..data.connector import ScoutDataConnector
from ..data.validator import DataValidator

logger = logging.getLogger(__name__)

class DatasetSelector:
    """
    Widget for searching and selecting datasets from Scout Data Discovery
    """
    
    def __init__(self, connector: ScoutDataConnector):
        """
        Initialize the dataset selector
        
        Args:
            connector: Scout Data Discovery connector instance
        """
        self.connector = connector
        self.validator = DataValidator()
        self.selected_dataset = None
        self.dataset_cache = {}
    
    def create_layout(self, component_id: str = "dataset-selector") -> html.Div:
        """
        Create the dataset selector layout
        
        Args:
            component_id: Base ID for components
            
        Returns:
            Dash HTML Div containing the selector interface
        """
        return html.Div([
            # Search Section
            html.Div([
                html.H4("Dataset Search", className="mb-3"),
                
                # Search input
                dcc.Input(
                    id=f"{component_id}-search-input",
                    type="text",
                    placeholder="Search datasets...",
                    className="form-control mb-2",
                    style={"width": "100%"}
                ),
                
                # Domain filter
                html.Div([
                    html.Label("Domain Filter:", className="form-label"),
                    dcc.Dropdown(
                        id=f"{component_id}-domain-filter",
                        placeholder="Select domain (optional)",
                        className="mb-2"
                    )
                ]),
                
                # Search button
                html.Button(
                    "Search Datasets",
                    id=f"{component_id}-search-btn",
                    className="btn btn-primary mb-3",
                    n_clicks=0
                )
            ], className="search-section mb-4"),
            
            # Results Section
            html.Div([
                html.H5("Search Results", className="mb-3"),
                html.Div(
                    id=f"{component_id}-results-container",
                    children="No search performed yet.",
                    className="results-container"
                )
            ], className="results-section mb-4"),
            
            # Selected Dataset Info
            html.Div([
                html.H5("Selected Dataset", className="mb-3"),
                html.Div(
                    id=f"{component_id}-selected-info",
                    children="No dataset selected.",
                    className="selected-info"
                )
            ], className="selected-section")
            
        ], className="dataset-selector", id=component_id)
    
    def register_callbacks(self, app: dash.Dash, component_id: str = "dataset-selector"):
        """
        Register callbacks for the dataset selector
        
        Args:
            app: Dash application instance
            component_id: Base ID for components
        """
        
        # Load available domains on app start
        @app.callback(
            Output(f"{component_id}-domain-filter", "options"),
            Input(f"{component_id}-domain-filter", "id")
        )
        def load_domains(_):
            try:
                domains = self.connector.get_available_domains()
                return [{"label": domain, "value": domain} for domain in domains]
            except Exception as e:
                logger.error(f"Error loading domains: {e}")
                return []
        
        # Search datasets
        @app.callback(
            Output(f"{component_id}-results-container", "children"),
            [Input(f"{component_id}-search-btn", "n_clicks")],
            [State(f"{component_id}-search-input", "value"),
             State(f"{component_id}-domain-filter", "value")]
        )
        def search_datasets(n_clicks, search_query, domain):
            if n_clicks == 0:
                return "No search performed yet."
            
            try:
                # Perform search
                results = self.connector.search_datasets(
                    query=search_query or "",
                    domain=domain,
                    limit=20
                )
                
                if not results:
                    return html.Div("No datasets found.", className="text-muted")
                
                # Create result cards
                result_cards = []
                for i, dataset in enumerate(results):
                    card = self._create_dataset_card(dataset, f"{component_id}-dataset-{i}")
                    result_cards.append(card)
                
                return html.Div(result_cards, className="dataset-results")
                
            except Exception as e:
                logger.error(f"Error searching datasets: {e}")
                return html.Div(
                    f"Error searching datasets: {str(e)}",
                    className="text-danger"
                )
        
        # Select dataset
        @app.callback(
            [Output(f"{component_id}-selected-info", "children"),
             Output("dataset-data-store", "data")],
            [Input({"type": "dataset-select-btn", "index": dash.dependencies.ALL}, "n_clicks")],
            [State(f"{component_id}-results-container", "children")],
            prevent_initial_call=True
        )
        def select_dataset(n_clicks_list, results_container):
            if not any(n_clicks_list):
                return "No dataset selected.", None
            
            # Find which button was clicked
            ctx_triggered = ctx.triggered[0]
            if not ctx_triggered["value"]:
                return "No dataset selected.", None
            
            try:
                # Extract dataset index from button ID
                button_info = ctx_triggered["prop_id"].split(".")[0]
                dataset_index = eval(button_info)["index"]
                
                # Get dataset info from search results
                # Note: In a real implementation, you'd store dataset info in a more robust way
                dataset_id = f"dataset_{dataset_index}"  # Simplified for demo
                
                # Load dataset sample and metadata
                sample_data = self.connector.download_dataset_sample(dataset_id, sample_size=100)
                metadata = self.connector.get_dataset_metadata(dataset_id)
                quality_info = self.connector.assess_dataset_quality(dataset_id)
                
                if sample_data is not None:
                    self.selected_dataset = {
                        "id": dataset_id,
                        "data": sample_data,
                        "metadata": metadata,
                        "quality": quality_info
                    }
                    
                    selected_info = self._create_selected_dataset_info(self.selected_dataset)
                    
                    # Store data for other components
                    dataset_store = {
                        "dataset_id": dataset_id,
                        "data": sample_data.to_dict('records'),
                        "columns": sample_data.columns.tolist(),
                        "metadata": metadata,
                        "quality": quality_info
                    }
                    
                    return selected_info, dataset_store
                else:
                    return html.Div("Error loading dataset.", className="text-danger"), None
                    
            except Exception as e:
                logger.error(f"Error selecting dataset: {e}")
                return html.Div(f"Error: {str(e)}", className="text-danger"), None
    
    def _create_dataset_card(self, dataset: Dict[str, Any], dataset_id: str) -> html.Div:
        """Create a card for a dataset in search results"""
        return html.Div([
            html.Div([
                html.H6(dataset.get("title", "Unnamed Dataset"), className="card-title"),
                html.P(
                    dataset.get("description", "No description available")[:200] + "...",
                    className="card-text text-muted"
                ),
                html.Div([
                    html.Span(f"Domain: {dataset.get('domain', 'Unknown')}", className="badge bg-secondary me-2"),
                    html.Span(f"Rows: {dataset.get('row_count', 'Unknown')}", className="badge bg-info me-2"),
                    html.Span(f"Columns: {dataset.get('column_count', 'Unknown')}", className="badge bg-info")
                ], className="mb-2"),
                html.Button(
                    "Select Dataset",
                    id={"type": "dataset-select-btn", "index": dataset_id},
                    className="btn btn-outline-primary btn-sm",
                    n_clicks=0
                )
            ], className="card-body")
        ], className="card mb-2")
    
    def _create_selected_dataset_info(self, dataset_info: Dict[str, Any]) -> html.Div:
        """Create display for selected dataset information"""
        data = dataset_info["data"]
        metadata = dataset_info.get("metadata", {})
        quality = dataset_info.get("quality", {})
        
        # Basic info
        basic_info = html.Div([
            html.H6("Dataset Overview"),
            html.P(f"Dataset ID: {dataset_info['id']}"),
            html.P(f"Rows: {len(data):,}"),
            html.P(f"Columns: {len(data.columns)}"),
            html.P(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        ])
        
        # Column info
        columns_info = html.Div([
            html.H6("Columns"),
            html.Ul([
                html.Li(f"{col} ({str(data[col].dtype)})") 
                for col in data.columns[:10]  # Show first 10 columns
            ])
        ])
        
        # Quality info
        quality_score = quality.get("overall_score", 0)
        quality_level = quality.get("quality_level", "Unknown")
        
        quality_info = html.Div([
            html.H6("Data Quality"),
            html.P(f"Overall Score: {quality_score:.2f}"),
            html.P(f"Quality Level: {quality_level}"),
            html.Div([
                html.Span(
                    quality_level,
                    className=f"badge bg-{'success' if quality_score > 0.8 else 'warning' if quality_score > 0.6 else 'danger'}"
                )
            ])
        ])
        
        # Sample data preview
        data_preview = html.Div([
            html.H6("Data Preview"),
            html.Div([
                html.Table([
                    html.Thead([
                        html.Tr([html.Th(col) for col in data.columns[:5]])  # First 5 columns
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(str(data.iloc[i, j]) if j < len(data.columns) else "")
                            for j in range(min(5, len(data.columns)))
                        ]) for i in range(min(5, len(data)))  # First 5 rows
                    ])
                ], className="table table-sm table-striped")
            ], style={"max-height": "200px", "overflow": "auto"})
        ])
        
        return html.Div([
            basic_info,
            html.Hr(),
            columns_info,
            html.Hr(),
            quality_info,
            html.Hr(),
            data_preview
        ], className="selected-dataset-info")
    
    def get_selected_dataset(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected dataset"""
        return self.selected_dataset
    
    def clear_selection(self):
        """Clear the current dataset selection"""
        self.selected_dataset = None