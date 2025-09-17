"""
Interactive Dashboard Main Application
Professional Plotly Dash Dashboard for Dynamic Data Visualization

This is the main entry point for the interactive dashboard application.
It integrates with the Scout Data Discovery backend for comprehensive data analysis.
"""

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging
from datetime import datetime
import json
import traceback

# Local imports
from config import config, ensure_directories, CHART_TYPES
from layouts.main_layout import create_main_layout
from layouts.sidebar import create_sidebar
from layouts.header import create_header
from components.data_ingestion import DataIngestionComponent
from components.chart_factory import ChartFactory
from components.filter_panel import FilterPanel
from utils.api_client import ScoutAPIClient
from utils.cache_manager import DashboardCacheManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
ensure_directories()

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title=config.app_name
)

# Initialize components
data_ingestion = DataIngestionComponent()
chart_factory = ChartFactory()
filter_panel = FilterPanel()
api_client = ScoutAPIClient(config.scout_api_base_url)
cache_manager = DashboardCacheManager()

# Auto-fetch datasets from Scout API on startup
def initialize_datasets():
    """Fetch initial datasets from Scout API"""
    try:
        logger.info("🔄 Auto-fetching datasets from Scout API...")
        result = data_ingestion.fetch_datasets_from_api(limit=5)  # Fetch 5 datasets initially
        
        if 'datasets' in result and result['datasets']:
            logger.info(f"✅ Auto-fetched {len(result['datasets'])} datasets from Scout API")
            return result['datasets']
        else:
            logger.warning("⚠️ No datasets auto-fetched from Scout API")
            return {}
    except Exception as e:
        logger.error(f"❌ Error auto-fetching datasets: {str(e)}")
        return {}

# Initialize with API data
initial_datasets = initialize_datasets()

# Define the app layout
app.layout = dbc.Container([
    # Header
    create_header(),
    
    # Main content area
    dbc.Row([
        # Sidebar
        dbc.Col([
            create_sidebar()
        ], width=3),
        
        # Main dashboard area
        dbc.Col([
            create_main_layout()
        ], width=9)
    ]),
    
    # Toast notifications
    dbc.Toast(
        id="notification-toast",
        header="Notification",
        is_open=False,
        dismissable=True,
        duration=4000,
        style={"position": "fixed", "top": 80, "right": 10, "width": 350, "z-index": 9999}
    ),
    
    # Loading overlay
    dcc.Loading(
        id="loading-overlay",
        type="circle",
        fullscreen=True,
        children=html.Div(id="loading-output")
    ),
    
    # Store components for state management
    dcc.Store(id="dashboard-state", data={}),
    dcc.Store(id="uploaded-datasets", data={}),
    dcc.Store(id="current-chart-config", data={}),
    dcc.Store(id="filter-state", data={}),
    
    # Interval component for auto-refresh
    dcc.Interval(
        id="auto-refresh-interval",
        interval=300000,  # 5 minutes
        n_intervals=0,
        disabled=True
    )
], fluid=True, className="dashboard-container")


# =============================================================================
# CALLBACK FUNCTIONS
# =============================================================================

@app.callback(
    [Output("uploaded-datasets", "data"),
     Output("dataset-upload-status", "children"),
     Output("notification-toast", "is_open"),
     Output("notification-toast", "children")],
    [Input("upload-data", "contents"),
     Input("upload-data", "filename"),
     Input("refresh-api-data", "n_clicks")],
    [State("uploaded-datasets", "data")]
)
def handle_data_upload_and_refresh(contents, filenames, refresh_clicks, current_datasets):
    """Handle file uploads and API data refresh."""
    ctx = callback_context
    
    try:
        if not ctx.triggered:
            # Return initial datasets fetched on startup
            return initial_datasets, html.Div(), False, ""
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "upload-data" and contents:
            # Handle file upload
            new_datasets = data_ingestion.process_uploaded_files(contents, filenames)
            
            # Merge with existing datasets
            all_datasets = {**current_datasets, **new_datasets}
            
            status_message = dbc.Alert(
                f"Successfully uploaded {len(new_datasets)} dataset(s)",
                color="success",
                dismissable=True
            )
            
            notification = f"Uploaded {len(new_datasets)} new dataset(s)"
            
            return all_datasets, status_message, True, notification
            
        elif trigger_id == "refresh-api-data":
            # Handle API data refresh
            logger.info("🔄 Refreshing data from Scout API...")
            result = data_ingestion.fetch_datasets_from_api(limit=5)
            
            if 'datasets' in result and result['datasets']:
                # Merge with existing datasets
                all_datasets = {**current_datasets, **result['datasets']}
                
                status_message = dbc.Alert(
                    f"Refreshed {len(result['datasets'])} dataset(s) from Scout API",
                    color="info",
                    dismissable=True
                )
                
                notification = f"Refreshed {len(result['datasets'])} datasets from Scout API"
                
                return all_datasets, status_message, True, notification
            else:
                status_message = dbc.Alert(
                    "No new datasets found from Scout API",
                    color="warning",
                    dismissable=True
                )
                
                return current_datasets, status_message, True, "No new datasets found"
    
    except Exception as e:
        logger.error(f"Error in data handling: {str(e)}")
        error_message = dbc.Alert(
            f"Error: {str(e)}",
            color="danger",
            dismissable=True
        )
        return current_datasets, error_message, True, f"Error: {str(e)}"
    
    return current_datasets, html.Div(), False, ""


@app.callback(
    Output("main-chart-display", "figure"),
    [Input("chart-type-selector", "value"),
     Input("dataset-selector", "value"),
     Input("x-axis-selector", "value"),
     Input("y-axis-selector", "value"),
     Input("color-selector", "value"),
     Input("chart-config-apply", "n_clicks")],
    [State("uploaded-datasets", "data"),
     State("filter-state", "data")]
)
def update_main_chart(chart_type, dataset_id, x_col, y_col, color_col, apply_clicks, datasets, filter_state):
    """Update the main chart based on user selections."""
    try:
        if not dataset_id or dataset_id not in datasets:
            return px.scatter(title="Select a dataset to begin visualization")
        
        # Get dataset
        dataset_info = datasets[dataset_id]
        
        if dataset_info.get("source") == "scout":
            # Fetch data from Scout API
            df = api_client.fetch_dataset(dataset_info["id"])
        else:
            # Load uploaded data
            df = data_ingestion.get_dataset(dataset_id)
        
        if df is None or df.empty:
            return px.scatter(title="No data available for selected dataset")
        
        # Apply filters
        if filter_state:
            df = filter_panel.apply_filters(df, filter_state)
        
        # Generate chart
        chart_config = {
            "chart_type": chart_type or "bar",
            "x_column": x_col,
            "y_column": y_col,
            "color_column": color_col,
            "title": f"{chart_type.title()} Chart - {dataset_info.get('name', dataset_id)}"
        }
        
        figure = chart_factory.create_chart(df, chart_config)
        
        return figure
    
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        return px.scatter(title=f"Error creating chart: {str(e)}")


@app.callback(
    [Output("x-axis-selector", "options"),
     Output("y-axis-selector", "options"),
     Output("color-selector", "options")],
    [Input("dataset-selector", "value")],
    [State("uploaded-datasets", "data")]
)
def update_column_selectors(dataset_id, datasets):
    """Update column selector options based on selected dataset."""
    if not dataset_id or dataset_id not in datasets:
        return [], [], []
    
    try:
        dataset_info = datasets[dataset_id]
        
        if dataset_info.get("source") == "scout":
            df = api_client.fetch_dataset_sample(dataset_info["id"])
        else:
            df = data_ingestion.get_dataset(dataset_id)
        
        if df is None or df.empty:
            return [], [], []
        
        # Create options for each column
        options = [{"label": col, "value": col} for col in df.columns]
        
        return options, options, options
    
    except Exception as e:
        logger.error(f"Error updating column selectors: {str(e)}")
        return [], [], []


@app.callback(
    Output("dataset-info-display", "children"),
    [Input("dataset-selector", "value")],
    [State("uploaded-datasets", "data")]
)
def display_dataset_info(dataset_id, datasets):
    """Display information about the selected dataset."""
    if not dataset_id or dataset_id not in datasets:
        return html.Div("Select a dataset to view information")
    
    try:
        dataset_info = datasets[dataset_id]
        
        # Create info cards
        info_cards = [
            dbc.Card([
                dbc.CardBody([
                    html.H5(dataset_info.get("name", dataset_id), className="card-title"),
                    html.P(f"Source: {dataset_info.get('source', 'Upload').title()}", className="card-text"),
                    html.P(f"Shape: {dataset_info.get('shape', [0, 0])[0]} rows × {dataset_info.get('shape', [0, 0])[1]} columns", 
                           className="card-text"),
                    html.P(f"Category: {dataset_info.get('category', 'Unknown')}", className="card-text"),
                ])
            ], className="mb-3")
        ]
        
        return info_cards
    
    except Exception as e:
        logger.error(f"Error displaying dataset info: {str(e)}")
        return html.Div(f"Error loading dataset info: {str(e)}")


@app.callback(
    Output("export-download", "data"),
    [Input("export-button", "n_clicks")],
    [State("main-chart-display", "figure"),
     State("export-format-selector", "value")]
)
def export_chart(n_clicks, figure, export_format):
    """Export chart in selected format."""
    if not n_clicks or not figure:
        return dash.no_update
    
    try:
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_chart_{timestamp}"
        
        if export_format == "png":
            import plotly.io as pio
            img_bytes = pio.to_image(figure, format="png")
            return dcc.send_bytes(img_bytes, f"{filename}.png")
        
        elif export_format == "html":
            import plotly.offline as pyo
            html_str = pyo.plot(figure, output_type='div', include_plotlyjs=True)
            return dcc.send_string(html_str, f"{filename}.html")
        
        elif export_format == "json":
            json_str = json.dumps(figure, indent=2)
            return dcc.send_string(json_str, f"{filename}.json")
    
    except Exception as e:
        logger.error(f"Error exporting chart: {str(e)}")
        return dash.no_update


@app.callback(
    [Output("newest-dataset-info", "children"),
     Output("newest-dataset-table", "children")],
    [Input("uploaded-datasets", "data"),
     Input("auto-refresh-interval", "n_intervals")]
)
def update_newest_dataset_display(datasets, n_intervals):
    """Update the display of the newest dataset in tabular format."""
    try:
        # Get the newest dataset
        newest_dataset = data_ingestion.get_newest_dataset()

        if newest_dataset is None:
            info_div = dbc.Alert(
                "No datasets available. The dashboard will automatically fetch datasets from the Scout API when it starts.",
                color="info"
            )
            table_div = html.Div()
            return info_div, table_div

        dataset_id, df, metadata = newest_dataset

        # Create info display
        source_badge = "API" if metadata.get('source') == 'scout_api' else "Upload"
        category = metadata.get('category', 'Uncategorized')
        
        info_div = html.Div([
            html.Strong("Dataset ID: "), f"{dataset_id}", 
            html.Span(f" ({source_badge})", className="badge bg-primary ms-2"), html.Br(),
            html.Strong("Name: "), f"{metadata.get('filename', 'Unknown')}", html.Br(),
            html.Strong("Category: "), f"{category}", html.Br(),
            html.Strong("Shape: "), f"{df.shape[0]} rows × {df.shape[1]} columns", html.Br(),
            html.Strong("Size: "), f"{metadata.get('size_mb', 0):.2f} MB", html.Br(),
            html.Strong("Last Updated: "), f"{metadata.get('updated_at', 'Unknown')}", html.Br(),
            html.Strong("Columns: "), ", ".join(df.columns.tolist())
        ])

        # Create table display (show first 100 rows)
        display_df = df.head(100)

        # Convert DataFrame to dash table format
        table_columns = [{"name": col, "id": col} for col in display_df.columns]
        table_data = display_df.to_dict('records')

        table_div = html.Div([
            html.P(f"Showing first {len(display_df)} rows of {len(df)} total rows",
                  className="text-muted mb-2"),
            dash_table.DataTable(
                id='newest-dataset-datatable',
                columns=table_columns,
                data=table_data,
                style_table={
                    'overflowX': 'auto',
                    'maxHeight': '400px',
                    'overflowY': 'auto'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'minWidth': '100px',
                    'maxWidth': '200px',
                    'whiteSpace': 'normal'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                page_size=20,
                sort_action='native',
                filter_action='native'
            )
        ])

        return info_div, table_div

    except Exception as e:
        logger.error(f"Error updating newest dataset display: {str(e)}")
        error_div = dbc.Alert(
            f"Error displaying newest dataset: {str(e)}",
            color="danger"
        )
        return error_div, html.Div()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Starting {config.app_name} v{config.app_version}")
    logger.info(f"Debug mode: {config.debug}")
    logger.info(f"Scout API: {config.scout_api_base_url}")
    
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug
    )