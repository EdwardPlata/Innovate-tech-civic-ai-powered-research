"""
Header Layout Module
Creates the application header component
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
from datetime import datetime


def create_header():
    """
    Create the application header component.

    Returns:
        dbc.Navbar: Application header with branding and controls
    """
    return dbc.Navbar([
        dbc.Container([
            dbc.NavbarBrand([
                html.I(className="fas fa-chart-line me-2"),
                "Interactive Data Dashboard"
            ], className="fw-bold"),

            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),

            dbc.Collapse([
                dbc.Nav([
                    dbc.NavItem(
                        dbc.NavLink([
                            html.I(className="fas fa-home me-1"),
                            "Home"
                        ], href="/", active=True)
                    ),
                    dbc.NavItem(
                        dbc.NavLink([
                            html.I(className="fas fa-info-circle me-1"),
                            "About"
                        ], href="/about")
                    ),
                    dbc.NavItem(
                        dbc.NavLink([
                            html.I(className="fas fa-question-circle me-1"),
                            "Help"
                        ], href="/help")
                    )
                ], navbar=True, className="ms-auto"),

                # User controls
                dbc.Nav([
                    dbc.NavItem([
                        html.Div([
                            html.I(className="fas fa-clock me-1"),
                            html.Span(
                                datetime.now().strftime("%H:%M:%S"),
                                id="current-time",
                                className="small"
                            )
                        ], className="text-muted")
                    ]),
                    dbc.NavItem(
                        dbc.Button([
                            html.I(className="fas fa-bell me-1"),
                            dbc.Badge("0", color="danger", className="ms-1", id="notification-badge")
                        ], color="link", size="sm", id="btn-notifications")
                    ),
                    dbc.NavItem(
                        dbc.Button([
                            html.I(className="fas fa-user me-1"),
                            "Profile"
                        ], color="link", size="sm", id="btn-profile")
                    )
                ], navbar=True)

            ], id="navbar-collapse", navbar=True, is_open=False)

        ], fluid=True)
    ],
    color="dark",
    dark=True,
    className="mb-4",
    style={"zIndex": "1030"})