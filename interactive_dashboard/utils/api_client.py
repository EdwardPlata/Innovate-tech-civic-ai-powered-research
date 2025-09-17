"""
API Client for Scout Data Discovery Backend
Handles communication with the Scout API
"""

import requests
import logging
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ScoutAPIClient:
    """Client for interacting with the Scout Data Discovery API."""

    def __init__(self, base_url: str = "http://localhost:8080/api", timeout: int = 30):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the Scout API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            Response data as dict, or None if failed
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            kwargs.setdefault('timeout', self.timeout)
            response = self.session.request(method, url, **kwargs)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return None

    def get_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of available datasets.

        Returns:
            List of dataset metadata
        """
        response = self._make_request('GET', '/datasets')
        return response.get('datasets', []) if response else []

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a dataset.

        Args:
            dataset_id: The dataset identifier

        Returns:
            Dataset metadata dict, or None if not found
        """
        return self._make_request('GET', f'/datasets/{dataset_id}')

    def query_dataset(self, dataset_id: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Query a dataset with filters and aggregations.

        Args:
            dataset_id: The dataset identifier
            query: Query parameters

        Returns:
            Query results, or None if failed
        """
        return self._make_request('POST', f'/datasets/{dataset_id}/query',
                                json=query)

    def get_data_preview(self, dataset_id: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Get a preview of dataset contents.

        Args:
            dataset_id: The dataset identifier
            limit: Maximum number of rows to return

        Returns:
            Preview data, or None if failed
        """
        return self._make_request('GET', f'/datasets/{dataset_id}/preview',
                                params={'limit': limit})

    def get_dataset_stats(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistical information about a dataset.

        Args:
            dataset_id: The dataset identifier

        Returns:
            Dataset statistics, or None if failed
        """
        return self._make_request('GET', f'/datasets/{dataset_id}/stats')

    def search_datasets(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for datasets by name or description.

        Args:
            query: Search query string

        Returns:
            List of matching datasets
        """
        response = self._make_request('GET', '/datasets/search',
                                    params={'q': query})
        return response.get('results', []) if response else []

    def get_api_status(self) -> Dict[str, Any]:
        """
        Get the status of the Scout API.

        Returns:
            API status information
        """
        response = self._make_request('GET', '/status')
        if response:
            return response
        return {'status': 'unavailable', 'timestamp': datetime.now().isoformat()}