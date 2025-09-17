"""
Sidebar Layout Module
Creates the sidebar navigation component
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output


def create_sidebar():
    """
    Create the sidebar navigation component.

    Returns:
        html.Div: Sidebar with navigation menu and controls
    """
    return html.Div([
        # Sidebar header
        html.Div([
            html.H4("Navigation", className="text-white mb-4"),
            html.Hr(className="bg-light")
        ], className="p-3"),

        # Navigation menu
        dbc.Nav([
            dbc.NavItem(
                dbc.NavLink([
                    html.I(className="fas fa-tachometer-alt me-2"),
                    "Dashboard"
                ], href="#", active=True, id="nav-dashboard")
            ),
            dbc.NavItem(
                dbc.NavLink([
                    html.I(className="fas fa-upload me-2"),
                    "Data Upload"
                ], href="#", id="nav-upload")
            ),
            dbc.NavItem(
                dbc.NavLink([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Charts"
                ], href="#", id="nav-charts")
            ),
            dbc.NavItem(
                dbc.NavLink([
                    html.I(className="fas fa-filter me-2"),
                    "Filters"
                ], href="#", id="nav-filters")
            ),
            dbc.NavItem(
                dbc.NavLink([
                    html.I(className="fas fa-cog me-2"),
                    "Settings"
                ], href="#", id="nav-settings")
            )
        ], vertical=True, pills=True, className="px-3"),

        html.Hr(className="bg-light my-4"),

        # Quick actions
        html.Div([
            html.H5("Quick Actions", className="text-white mb-3"),
            dbc.Button([
                html.I(className="fas fa-plus me-2"),
                "New Analysis"
            ], color="success", size="sm", className="w-100 mb-2", id="btn-new-analysis"),
            dbc.Button([
                html.I(className="fas fa-save me-2"),
                "Save Dashboard"
            ], color="info", size="sm", className="w-100 mb-2", id="btn-save-dashboard"),
            dbc.Button([
                html.I(className="fas fa-share me-2"),
                "Share"
            ], color="warning", size="sm", className="w-100", id="btn-share")
        ], className="px-3"),

        html.Hr(className="bg-light my-4"),

        # Data status
        html.Div([
            html.H5("Data Status", className="text-white mb-3"),
            html.Div([
                html.Small("Datasets loaded: ", className="text-light"),
                html.Span("0", id="datasets-count", className="text-warning fw-bold")
            ], className="mb-2"),
            html.Div([
                html.Small("Active filters: ", className="text-light"),
                html.Span("0", id="filters-count", className="text-warning fw-bold")
            ], className="mb-2"),
            html.Div([
                html.Small("Charts rendered: ", className="text-light"),
                html.Span("0", id="charts-count", className="text-warning fw-bold")
            ])
        ], className="px-3")

    ], className="bg-primary text-white", style={
        "minHeight": "100vh",
        "position": "sticky",
        "top": "0"
    })