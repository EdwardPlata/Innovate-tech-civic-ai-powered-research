"""
Main Layout Module
Creates the main dashboard layout component
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_main_layout():
    """
    Create the main dashboard layout component.

    Returns:
        dbc.Container: Main dashboard container with tabs and content areas
    """
    return dbc.Container([
        # Main content header
        dbc.Row([
            dbc.Col([
                html.H2("Interactive Data Dashboard", className="text-primary mb-4"),
                html.P("Explore and analyze your data with interactive visualizations",
                      className="text-muted")
            ])
        ], className="mb-4"),

        # Main content tabs
        dbc.Tabs([
            # Data Overview Tab
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Data Overview", className="mb-3"),
                            html.P("Datasets are automatically fetched from the Scout API. Upload additional files or refresh to get more data.",
                                  className="text-muted"),
                            # Data upload area will be populated by callbacks
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=True
                            ),
                            html.Div(id='upload-status'),
                            # Add refresh button for API data
                            dbc.Button([
                                html.I(className="fas fa-sync me-2"),
                                "Refresh from API"
                            ], 
                            id="refresh-api-data", 
                            color="primary", 
                            className="mt-2",
                            n_clicks=0)
                        ], className="border rounded p-3 mb-3")
                    ])
                ]),

                # Newest Dataset Table
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Newest Dataset", className="mb-3"),
                            html.Div(id="newest-dataset-info", className="mb-2"),
                            html.Div(id="newest-dataset-table")
                        ], className="border rounded p-3 mb-3")
                    ])
                ]),

                # Charts area
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Visualizations", className="mb-3"),
                            html.Div(id="charts-container")
                        ], className="border rounded p-3")
                    ])
                ], className="mt-3")

            ], label="Data Overview", tab_id="data-overview"),

            # Analysis Tab
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Data Analysis", className="mb-3"),
                            html.P("Perform advanced analysis on your data."),
                            html.Div(id="analysis-container")
                        ], className="border rounded p-3")
                    ])
                ])
            ], label="Analysis", tab_id="analysis"),

            # Reports Tab
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Reports & Exports", className="mb-3"),
                            html.P("Generate and export reports from your analysis."),
                            html.Div(id="reports-container")
                        ], className="border rounded p-3")
                    ])
                ])
            ], label="Reports", tab_id="reports")

        ], id="main-tabs", active_tab="data-overview", className="mb-4")

    ], fluid=True, className="p-4")