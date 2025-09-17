"""
API Configuration for Scout Data Discovery Backend

Centralized configuration for API endpoints, timeouts, retries, and error handling
to improve reliability and reduce timeout errors.
"""

from typing import Dict, Any, List
import os
from pathlib import Path

class APIConfig:
    """Configuration settings for API endpoints and error handling"""

    # Timeout configurations (in seconds)
    REQUEST_TIMEOUT = 90
    SEARCH_TIMEOUT = 120  # Longer for search operations
    SAMPLE_TIMEOUT = 180  # Longest for data downloads
    QUALITY_TIMEOUT = 150  # For quality assessments

    # Retry configurations
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2
    RATE_LIMIT_DELAY = 1.0

    # Cache configurations
    CACHE_DURATIONS = {
        'dashboard': 900,      # 15 minutes for dashboard data
        'api_responses': 1800, # 30 minutes for API responses
        'dataset_samples': 7200, # 2 hours for dataset samples
        'quality_assessments': 3600, # 1 hour for quality assessments
        'categories': 1800,    # 30 minutes for categories
        'memory_cache': 300    # 5 minutes for memory cache
    }

    # ThreadPool configuration
    MAX_WORKERS = 8

    # API endpoint specific settings
    ENDPOINT_CONFIG = {
        '/api/datasets/top-updated': {
            'timeout': SEARCH_TIMEOUT,
            'cache_duration': CACHE_DURATIONS['dashboard'],
            'max_retries': MAX_RETRIES,
            'enable_fallback': True
        },
        '/api/datasets/search': {
            'timeout': SEARCH_TIMEOUT,
            'cache_duration': CACHE_DURATIONS['api_responses'],
            'max_retries': MAX_RETRIES,
            'enable_fallback': True
        },
        '/api/datasets/{dataset_id}/sample': {
            'timeout': SAMPLE_TIMEOUT,
            'cache_duration': CACHE_DURATIONS['dataset_samples'],
            'max_retries': 2,  # Fewer retries for large downloads
            'enable_fallback': False
        },
        '/api/datasets/{dataset_id}/quality': {
            'timeout': QUALITY_TIMEOUT,
            'cache_duration': CACHE_DURATIONS['quality_assessments'],
            'max_retries': MAX_RETRIES,
            'enable_fallback': True
        },
        '/api/categories': {
            'timeout': REQUEST_TIMEOUT,
            'cache_duration': CACHE_DURATIONS['categories'],
            'max_retries': MAX_RETRIES,
            'enable_fallback': True,
            'static_fallback': [
                {"name": "City Government", "count": 45, "color": "#ff6b6b"},
                {"name": "Public Safety", "count": 38, "color": "#4ecdc4"},
                {"name": "Transportation", "count": 32, "color": "#45b7d1"},
                {"name": "Health", "count": 28, "color": "#96ceb4"}
            ]
        },
        '/api/datasets/relationships': {
            'timeout': 60,  # Shorter timeout to prevent hanging
            'cache_duration': CACHE_DURATIONS['api_responses'],
            'max_retries': 2,  # Fewer retries for complex operations
            'enable_fallback': True
        },
        '/api/network/visualization': {
            'timeout': 45,  # Even shorter for network viz
            'cache_duration': CACHE_DURATIONS['api_responses'],
            'max_retries': 2,
            'enable_fallback': True
        },
        '/api/chat/memory-config': {
            'timeout': REQUEST_TIMEOUT,
            'cache_duration': 0,  # No caching for memory config
            'max_retries': MAX_RETRIES,
            'enable_fallback': False
        },
        '/api/chat/ask': {
            'timeout': 90,  # Longer timeout for AI chat responses
            'cache_duration': 300,  # Short-term caching (5 minutes)
            'max_retries': 2,
            'enable_fallback': False
        }
    }

    # Fallback datasets for when API fails
    FALLBACK_DATASETS = [
        {
            "id": "fallback-311",
            "name": "NYC 311 Service Requests (Cached)",
            "description": "Cached sample of 311 service requests data",
            "download_count": 50000,
            "updated_at": "2024-01-01T00:00:00",
            "category": "City Government",
            "tags": ["311", "services", "government"],
            "columns_count": 15
        },
        {
            "id": "fallback-health",
            "name": "NYC Health Data (Cached)",
            "description": "Cached sample of NYC health inspection data",
            "download_count": 25000,
            "updated_at": "2024-01-01T00:00:00",
            "category": "Health",
            "tags": ["health", "inspections", "restaurants"],
            "columns_count": 12
        }
    ]

    @classmethod
    def get_endpoint_config(cls, endpoint: str) -> Dict[str, Any]:
        """Get configuration for a specific endpoint"""
        # Handle parameterized endpoints
        for pattern, config in cls.ENDPOINT_CONFIG.items():
            if '{' in pattern:
                # Simple pattern matching for parameterized endpoints
                pattern_base = pattern.split('{')[0]
                if endpoint.startswith(pattern_base):
                    return config
            elif endpoint == pattern:
                return config

        # Default configuration
        return {
            'timeout': cls.REQUEST_TIMEOUT,
            'cache_duration': cls.CACHE_DURATIONS['api_responses'],
            'max_retries': cls.MAX_RETRIES,
            'enable_fallback': False
        }

    @classmethod
    def get_fallback_data(cls, data_type: str) -> List[Dict[str, Any]]:
        """Get fallback data when API is unavailable"""
        if data_type == 'datasets':
            return cls.FALLBACK_DATASETS
        elif data_type == 'categories':
            return cls.ENDPOINT_CONFIG['/api/categories']['static_fallback']
        return []

    @classmethod
    def should_use_fallback(cls, error_type: str, attempt_count: int) -> bool:
        """Determine if fallback should be used based on error type and attempts"""
        timeout_errors = ['timeout', 'connectionerror', 'httperror_504']

        if error_type.lower() in timeout_errors and attempt_count >= cls.MAX_RETRIES:
            return True
        return False

# Environment-based overrides
if os.getenv('ENVIRONMENT') == 'development':
    # Shorter timeouts for development
    APIConfig.REQUEST_TIMEOUT = 30
    APIConfig.SEARCH_TIMEOUT = 45
    APIConfig.SAMPLE_TIMEOUT = 60

elif os.getenv('ENVIRONMENT') == 'production':
    # Longer timeouts for production
    APIConfig.REQUEST_TIMEOUT = 120
    APIConfig.SEARCH_TIMEOUT = 180
    APIConfig.SAMPLE_TIMEOUT = 300