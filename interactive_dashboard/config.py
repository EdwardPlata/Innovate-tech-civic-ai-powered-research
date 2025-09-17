"""
Interactive Dashboard Configuration
Professional Plotly Dash Dashboard Configuration Management
"""

import os
from typing import Optional
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from pathlib import Path


class DashboardConfig(BaseSettings):
    """Dashboard configuration with environment variable support."""
    
    # Application Settings
    app_name: str = Field(default="Interactive Data Dashboard", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8050, description="Server port")
    
    # Scout API Integration
    scout_api_base_url: str = Field(default="http://localhost:8080/api", description="Scout API base URL")
    scout_api_timeout: int = Field(default=30, description="Scout API timeout in seconds")
    
    # Data Configuration
    max_file_size_mb: int = Field(default=100, description="Maximum file upload size in MB")
    max_rows_display: int = Field(default=10000, description="Maximum rows to display in tables")
    cache_timeout: int = Field(default=3600, description="Cache timeout in seconds")
    
    # Chart Configuration
    default_chart_height: int = Field(default=500, description="Default chart height in pixels")
    default_chart_width: Optional[int] = Field(default=None, description="Default chart width in pixels")
    animation_duration: int = Field(default=750, description="Chart animation duration in milliseconds")
    
    # Performance Settings
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_compression: bool = Field(default=True, description="Enable response compression")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent API requests")
    
    # Redis Configuration (for caching)
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    
    # File Storage
    upload_folder: str = Field(default="data/uploads", description="Upload folder path")
    cache_folder: str = Field(default="data/cache", description="Cache folder path")
    export_folder: str = Field(default="data/exports", description="Export folder path")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", description="Secret key for sessions")
    allowed_file_extensions: list = Field(
        default=["csv", "xlsx", "xls", "json", "parquet"], 
        description="Allowed file extensions for upload"
    )
    
    # UI Configuration
    theme: str = Field(default="bootstrap", description="Dashboard theme")
    sidebar_collapsed: bool = Field(default=False, description="Sidebar collapsed by default")
    show_data_preview: bool = Field(default=True, description="Show data preview by default")
    
    # Chart Defaults
    default_color_palette: str = Field(default="plotly", description="Default color palette")
    chart_templates: list = Field(
        default=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
        description="Available chart templates"
    )
    
    model_config = ConfigDict(
        env_file=".env",
        env_prefix="DASHBOARD_",
        case_sensitive=False
    )


# Global configuration instance
config = DashboardConfig()

# Ensure required directories exist
def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        config.upload_folder,
        config.cache_folder,
        config.export_folder,
        "assets",
        "data/samples"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Chart configuration constants
CHART_TYPES = {
    "bar": {
        "name": "Bar Chart",
        "description": "Compare categories with rectangular bars",
        "supported_orientations": ["vertical", "horizontal"],
        "supports_grouping": True,
        "supports_stacking": True
    },
    "line": {
        "name": "Line Chart",
        "description": "Show trends and changes over time",
        "supports_multiple_series": True,
        "supports_markers": True,
        "supports_fill": True
    },
    "scatter": {
        "name": "Scatter Plot",
        "description": "Show relationships between two variables",
        "supports_sizing": True,
        "supports_coloring": True,
        "supports_regression": True
    },
    "map": {
        "name": "Geographic Map",
        "description": "Visualize geographic data",
        "supported_types": ["choropleth", "scatter_map", "heatmap"],
        "requires_geo_data": True
    }
}

# Color palettes
COLOR_PALETTES = {
    "plotly": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"],
    "viridis": ["#440154", "#31688e", "#35b779", "#fde725"],
    "plasma": ["#0d0887", "#7e03a8", "#cc4778", "#f89441", "#f0f921"],
    "blues": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6"],
    "reds": ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a"],
    "greens": ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476"],
    "custom_civic": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
}

# Widget configurations
WIDGET_CONFIGS = {
    "data_selector": {
        "multi_select": True,
        "searchable": True,
        "clearable": True
    },
    "date_picker": {
        "display_format": "YYYY-MM-DD",
        "calendar_orientation": "vertical"
    },
    "numeric_slider": {
        "step": 1,
        "marks_every": 10,
        "tooltip_always_visible": False
    }
}

# Export formats
EXPORT_FORMATS = {
    "png": {"extension": "png", "mime_type": "image/png"},
    "jpeg": {"extension": "jpg", "mime_type": "image/jpeg"},
    "svg": {"extension": "svg", "mime_type": "image/svg+xml"},
    "pdf": {"extension": "pdf", "mime_type": "application/pdf"},
    "html": {"extension": "html", "mime_type": "text/html"},
    "json": {"extension": "json", "mime_type": "application/json"}
}