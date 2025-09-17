"""
Configuration for Scout Integration Dashboard Components
"""
import os
from typing import Dict, List, Optional

# Scout API Configuration
SCOUT_API_URL = os.getenv("SCOUT_API_URL", "http://localhost:8000")
SCOUT_API_TIMEOUT = int(os.getenv("SCOUT_API_TIMEOUT", "30"))

# Dashboard Configuration
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DEBUG_MODE = os.getenv("DEBUG", "True").lower() == "true"

# Cache Configuration
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# Data Processing Configuration
MAX_ROWS_DISPLAY = int(os.getenv("MAX_ROWS_DISPLAY", "10000"))
SAMPLE_SIZE_DEFAULT = int(os.getenv("SAMPLE_SIZE_DEFAULT", "1000"))
MAX_CATEGORIES_FILTER = int(os.getenv("MAX_CATEGORIES_FILTER", "50"))

# Chart Configuration
DEFAULT_CHART_HEIGHT = int(os.getenv("DEFAULT_CHART_HEIGHT", "500"))
DEFAULT_COLOR_SCHEME = os.getenv("DEFAULT_COLOR_SCHEME", "viridis")

# Chart type mappings
CHART_TYPE_MAPPING = {
    'bar': 'Bar Chart',
    'line': 'Line Chart',
    'scatter': 'Scatter Plot',
    'histogram': 'Histogram',
    'box': 'Box Plot',
    'violin': 'Violin Plot',
    'heatmap': 'Heatmap',
    'pie': 'Pie Chart',
    'area': 'Area Chart',
    'density': 'Density Plot',
    'correlation': 'Correlation Matrix'
}

# Color schemes for different data types
COLOR_SCHEMES = {
    'categorical': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ],
    'sequential': 'viridis',
    'diverging': 'RdBu'
}

# Default Scout API endpoints
SCOUT_ENDPOINTS = {
    'search_datasets': '/search_datasets',
    'assess_quality': '/assess_quality',
    'download_sample': '/download_sample',
    'get_metadata': '/datasets/{dataset_id}/metadata',
    'advanced_search': '/advanced_search',
    'get_domains': '/domains'
}

def get_scout_config() -> Dict[str, str]:
    """Get Scout API configuration"""
    return {
        'base_url': SCOUT_API_URL,
        'timeout': SCOUT_API_TIMEOUT,
        'endpoints': SCOUT_ENDPOINTS
    }

def get_dashboard_config() -> Dict[str, any]:
    """Get dashboard configuration"""
    return {
        'port': DASHBOARD_PORT,
        'host': DASHBOARD_HOST,
        'debug': DEBUG_MODE,
        'cache_ttl': CACHE_TTL,
        'max_rows_display': MAX_ROWS_DISPLAY,
        'sample_size_default': SAMPLE_SIZE_DEFAULT,
        'chart_height': DEFAULT_CHART_HEIGHT,
        'color_schemes': COLOR_SCHEMES
    }