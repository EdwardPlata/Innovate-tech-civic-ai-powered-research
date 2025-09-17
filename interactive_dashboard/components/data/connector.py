"""
Data Connector for Scout Data Discovery API Integration
"""
import asyncio
import pandas as pd
import requests
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoutDataConnector:
    """
    Connector for Scout Data Discovery API integration
    Provides methods to search, assess, and download datasets
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", cache_ttl: int = 3600):
        """
        Initialize the connector
        
        Args:
            base_url: Base URL for Scout Data Discovery API
            cache_ttl: Cache time-to-live in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.cache_ttl = cache_ttl
        self.session = requests.Session()
        self._cache = {}
        
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key from endpoint and parameters"""
        return f"{endpoint}_{hash(str(sorted(params.items())))}"
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid"""
        return datetime.now() - timestamp < timedelta(seconds=self.cache_ttl)
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve from cache if valid"""
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                logger.info(f"Cache hit for {cache_key}")
                return data
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache"""
        self._cache[cache_key] = (data, datetime.now())
        logger.info(f"Cached data for {cache_key}")
    
    def search_datasets(self, 
                       query: str = "", 
                       domain: Optional[str] = None,
                       limit: int = 50,
                       include_metadata: bool = True) -> List[Dict]:
        """
        Search for datasets using Scout Data Discovery
        
        Args:
            query: Search query string
            domain: Optional domain filter
            limit: Maximum number of results
            include_metadata: Whether to include dataset metadata
            
        Returns:
            List of dataset information dictionaries
        """
        params = {
            "query": query,
            "limit": limit,
            "include_metadata": include_metadata
        }
        if domain:
            params["domain"] = domain
            
        cache_key = self._get_cache_key("search_datasets", params)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            response = self.session.get(
                f"{self.base_url}/search_datasets",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self._set_cache(cache_key, result)
            
            logger.info(f"Found {len(result)} datasets for query: '{query}'")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching datasets: {e}")
            return []
    
    def assess_dataset_quality(self, dataset_id: str) -> Dict:
        """
        Assess quality of a specific dataset
        
        Args:
            dataset_id: Unique identifier for the dataset
            
        Returns:
            Dictionary containing quality assessment results
        """
        cache_key = self._get_cache_key("assess_quality", {"dataset_id": dataset_id})
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            response = self.session.get(
                f"{self.base_url}/assess_quality/{dataset_id}",
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            self._set_cache(cache_key, result)
            
            logger.info(f"Quality assessment completed for dataset: {dataset_id}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error assessing dataset quality: {e}")
            return {"error": str(e)}
    
    def download_dataset_sample(self, 
                               dataset_id: str, 
                               sample_size: int = 1000,
                               random_sample: bool = True) -> Optional[pd.DataFrame]:
        """
        Download a sample of the dataset
        
        Args:
            dataset_id: Unique identifier for the dataset
            sample_size: Number of rows to sample
            random_sample: Whether to use random sampling
            
        Returns:
            Pandas DataFrame with sample data or None if error
        """
        params = {
            "sample_size": sample_size,
            "random_sample": random_sample
        }
        
        cache_key = self._get_cache_key("download_sample", 
                                       {"dataset_id": dataset_id, **params})
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            response = self.session.get(
                f"{self.base_url}/download_sample/{dataset_id}",
                params=params,
                timeout=120
            )
            response.raise_for_status()
            
            # Handle different response formats
            content_type = response.headers.get('content-type', '')
            
            if 'application/json' in content_type:
                data = response.json()
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame(data)
            elif 'text/csv' in content_type:
                df = pd.read_csv(response.content)
            else:
                # Try to parse as JSON first, then CSV
                try:
                    data = response.json()
                    df = pd.DataFrame(data)
                except:
                    df = pd.read_csv(response.content)
            
            self._set_cache(cache_key, df)
            logger.info(f"Downloaded sample of {len(df)} rows for dataset: {dataset_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading dataset sample: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing dataset sample: {e}")
            return None
    
    def get_dataset_metadata(self, dataset_id: str) -> Dict:
        """
        Get comprehensive metadata for a dataset
        
        Args:
            dataset_id: Unique identifier for the dataset
            
        Returns:
            Dictionary containing dataset metadata
        """
        cache_key = self._get_cache_key("get_metadata", {"dataset_id": dataset_id})
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            response = self.session.get(
                f"{self.base_url}/datasets/{dataset_id}/metadata",
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self._set_cache(cache_key, result)
            
            logger.info(f"Retrieved metadata for dataset: {dataset_id}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting dataset metadata: {e}")
            return {"error": str(e)}
    
    def advanced_search(self, 
                       search_criteria: Dict,
                       sort_by: str = "relevance",
                       ascending: bool = False) -> List[Dict]:
        """
        Perform advanced search with multiple criteria
        
        Args:
            search_criteria: Dictionary of search parameters
            sort_by: Field to sort results by
            ascending: Sort order
            
        Returns:
            List of dataset information dictionaries
        """
        params = {
            "sort_by": sort_by,
            "ascending": ascending,
            **search_criteria
        }
        
        cache_key = self._get_cache_key("advanced_search", params)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            response = self.session.post(
                f"{self.base_url}/advanced_search",
                json=params,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            self._set_cache(cache_key, result)
            
            logger.info(f"Advanced search returned {len(result)} results")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in advanced search: {e}")
            return []
    
    def get_available_domains(self) -> List[str]:
        """
        Get list of available data domains
        
        Returns:
            List of domain names
        """
        cache_key = self._get_cache_key("get_domains", {})
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            response = self.session.get(
                f"{self.base_url}/domains",
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self._set_cache(cache_key, result)
            
            logger.info(f"Retrieved {len(result)} available domains")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting domains: {e}")
            return []
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_entries = len(self._cache)
        valid_entries = sum(1 for _, (_, timestamp) in self._cache.items() 
                           if self._is_cache_valid(timestamp))
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries,
            "cache_ttl": self.cache_ttl
        }