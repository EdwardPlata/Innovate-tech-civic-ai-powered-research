"""
Enhanced NYC Open Data API Client

Advanced API wrapper that integrates with Scout data discovery system,
supporting both pandas and Polars DataFrames with comprehensive SoQL capabilities.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from urllib.parse import urlencode, quote
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from .exceptions import APIError, DataDownloadError, ConfigurationError


class SoQLQueryBuilder:
    """
    Advanced SoQL (Socrata Query Language) query builder for complex data filtering
    """

    def __init__(self):
        self.query_params = {}
        self.filters = []
        self.select_fields = []
        self.order_fields = []
        self.group_fields = []

    def select(self, *fields: str) -> 'SoQLQueryBuilder':
        """Select specific fields"""
        self.select_fields.extend(fields)
        return self

    def where(self, condition: str) -> 'SoQLQueryBuilder':
        """Add WHERE condition"""
        self.filters.append(condition)
        return self

    def where_date_range(self, field: str, start_date: Union[str, datetime],
                        end_date: Union[str, datetime] = None) -> 'SoQLQueryBuilder':
        """Add date range filter"""
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()

        if end_date is None:
            condition = f"{field} >= '{start_date}'"
        else:
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
            condition = f"{field} between '{start_date}' and '{end_date}'"

        self.filters.append(condition)
        return self

    def where_in(self, field: str, values: List[Any]) -> 'SoQLQueryBuilder':
        """Add IN condition"""
        quoted_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in values]
        condition = f"{field} in ({', '.join(quoted_values)})"
        self.filters.append(condition)
        return self

    def where_like(self, field: str, pattern: str) -> 'SoQLQueryBuilder':
        """Add LIKE condition for text search"""
        condition = f"upper({field}) like upper('%{pattern}%')"
        self.filters.append(condition)
        return self

    def where_not_null(self, field: str) -> 'SoQLQueryBuilder':
        """Filter out null values"""
        condition = f"{field} is not null"
        self.filters.append(condition)
        return self

    def where_numeric_range(self, field: str, min_val: float = None,
                           max_val: float = None) -> 'SoQLQueryBuilder':
        """Add numeric range filter"""
        if min_val is not None and max_val is not None:
            condition = f"{field} between {min_val} and {max_val}"
        elif min_val is not None:
            condition = f"{field} >= {min_val}"
        elif max_val is not None:
            condition = f"{field} <= {max_val}"
        else:
            return self

        self.filters.append(condition)
        return self

    def order_by(self, field: str, ascending: bool = True) -> 'SoQLQueryBuilder':
        """Add ORDER BY clause"""
        direction = "ASC" if ascending else "DESC"
        self.order_fields.append(f"{field} {direction}")
        return self

    def group_by(self, *fields: str) -> 'SoQLQueryBuilder':
        """Add GROUP BY clause"""
        self.group_fields.extend(fields)
        return self

    def limit(self, count: int) -> 'SoQLQueryBuilder':
        """Set result limit"""
        self.query_params['$limit'] = count
        return self

    def offset(self, count: int) -> 'SoQLQueryBuilder':
        """Set result offset"""
        self.query_params['$offset'] = count
        return self

    def full_text_search(self, query: str) -> 'SoQLQueryBuilder':
        """Add full-text search"""
        self.query_params['$q'] = query
        return self

    def build(self) -> Dict[str, str]:
        """Build the final query parameters"""
        params = self.query_params.copy()

        if self.select_fields:
            params['$select'] = ', '.join(self.select_fields)

        if self.filters:
            params['$where'] = ' AND '.join(self.filters)

        if self.order_fields:
            params['$order'] = ', '.join(self.order_fields)

        if self.group_fields:
            params['$group'] = ', '.join(self.group_fields)

        return params


class EnhancedNYCDataClient:
    """
    Enhanced NYC Open Data API client with Scout integration,
    advanced SoQL support, and dual pandas/Polars backend
    """

    def __init__(self,
                 app_token: Optional[str] = None,
                 default_backend: str = "pandas",
                 rate_limit_delay: float = 1.0,
                 max_retries: int = 3,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the enhanced API client

        Args:
            app_token: Socrata app token for higher rate limits
            default_backend: Default data backend ('pandas' or 'polars')
            rate_limit_delay: Seconds between requests
            max_retries: Maximum retry attempts
            logger: Optional logger instance
        """
        self.base_url = "https://data.cityofnewyork.us/resource"
        self.discovery_url = "http://api.us.socrata.com/api/catalog/v1"
        self.app_token = app_token
        self.default_backend = default_backend
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

        # Validate backend
        if default_backend == "polars" and not POLARS_AVAILABLE:
            raise ConfigurationError("Polars not available. Install with: pip install polars")

        # Setup logging
        self.logger = logger or logging.getLogger(__name__)

        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Enhanced-NYC-Data-Client/1.0 (Scout Integration)',
            'Accept': 'application/json'
        })

        if self.app_token:
            self.session.headers.update({'X-App-Token': self.app_token})
            self.logger.info("Initialized with app token for higher rate limits")

        # Rate limiting
        self.last_request_time = 0

        # Caching
        self.response_cache = {}
        self.cache_duration = 300  # 5 minutes

        # Statistics
        self.request_count = 0
        self.cache_hits = 0

        self.logger.info(f"Enhanced NYC Data Client initialized with {default_backend} backend")

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make HTTP request with retries and caching"""
        # Create cache key
        cache_key = f"{url}?{urlencode(params or {})}"

        # Check cache
        if cache_key in self.response_cache:
            cached_time, cached_data = self.response_cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                self.cache_hits += 1
                return cached_data

        # Rate limiting
        self._rate_limit()

        # Make request with retries
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                self.request_count += 1
                response = self.session.get(url, params=params, timeout=90)  # Increased timeout
                response.raise_for_status()

                data = response.json()

                # Cache successful response
                self.response_cache[cache_key] = (time.time(), data)

                self.logger.debug(f"Request successful: {url} (attempt {attempt + 1})")
                return data

            except requests.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Request failed after {self.max_retries} attempts: {str(e)}")

        raise APIError(f"Request failed: {str(last_exception)}",
                      getattr(last_exception.response, 'status_code', None) if hasattr(last_exception, 'response') else None)

    def query(self) -> SoQLQueryBuilder:
        """Create a new SoQL query builder"""
        return SoQLQueryBuilder()

    def get_dataset(self,
                   dataset_id: str,
                   query_builder: SoQLQueryBuilder = None,
                   backend: Optional[str] = None,
                   **params) -> Union[pd.DataFrame, 'pl.DataFrame']:
        """
        Fetch dataset with advanced querying capabilities

        Args:
            dataset_id: Dataset identifier
            query_builder: SoQL query builder instance
            backend: Data backend ('pandas' or 'polars')
            **params: Additional query parameters

        Returns:
            DataFrame in requested backend format
        """
        try:
            backend = backend or self.default_backend
            url = f"{self.base_url}/{dataset_id}.json"

            # Build query parameters
            if query_builder:
                query_params = query_builder.build()
                query_params.update(params)
            else:
                query_params = params

            self.logger.info(f"Fetching dataset {dataset_id} with {len(query_params)} parameters")

            # Make request
            data = self._make_request(url, query_params)

            if not data:
                self.logger.warning(f"Dataset {dataset_id} returned empty data")
                return pd.DataFrame() if backend == "pandas" else (pl.DataFrame() if POLARS_AVAILABLE else pd.DataFrame())

            # Convert to requested backend
            if backend == "pandas":
                df = pd.DataFrame(data)
                self.logger.info(f"Retrieved {len(df)} rows, {len(df.columns)} columns as pandas DataFrame")
                return df
            elif backend == "polars" and POLARS_AVAILABLE:
                df = pl.DataFrame(data)
                self.logger.info(f"Retrieved {df.shape[0]} rows, {df.shape[1]} columns as Polars DataFrame")
                return df
            else:
                # Fallback to pandas
                df = pd.DataFrame(data)
                self.logger.warning(f"Backend '{backend}' not available, using pandas instead")
                return df

        except Exception as e:
            error_msg = f"Failed to fetch dataset {dataset_id}: {str(e)}"
            self.logger.error(error_msg)
            raise DataDownloadError(error_msg, dataset_id) from e

    def get_dataset_streaming(self,
                            dataset_id: str,
                            chunk_size: int = 10000,
                            query_builder: SoQLQueryBuilder = None,
                            backend: Optional[str] = None) -> Iterator[Union[pd.DataFrame, 'pl.DataFrame']]:
        """
        Stream large datasets in chunks

        Args:
            dataset_id: Dataset identifier
            chunk_size: Size of each chunk
            query_builder: SoQL query builder
            backend: Data backend

        Yields:
            DataFrame chunks
        """
        backend = backend or self.default_backend
        offset = 0

        while True:
            # Create query with offset and limit
            chunk_query = query_builder or SoQLQueryBuilder()
            chunk_query = chunk_query.limit(chunk_size).offset(offset)

            try:
                chunk_df = self.get_dataset(dataset_id, chunk_query, backend)

                if chunk_df.empty:
                    break

                yield chunk_df
                offset += chunk_size

                # Check if we got fewer rows than requested (end of data)
                if len(chunk_df) < chunk_size:
                    break

            except Exception as e:
                self.logger.error(f"Streaming failed at offset {offset}: {str(e)}")
                break

    def search_datasets(self,
                       query: str = None,
                       category: str = None,
                       tags: List[str] = None,
                       limit: int = 100) -> pd.DataFrame:
        """
        Search for datasets using the Discovery API

        Args:
            query: Search query text
            category: Dataset category filter
            tags: Tag filters
            limit: Maximum results

        Returns:
            DataFrame with dataset metadata
        """
        try:
            params = {
                'domains': 'data.cityofnewyork.us',
                'search_context': 'data.cityofnewyork.us',
                'limit': limit
            }

            if query:
                params['q'] = query

            if category:
                params['categories'] = category

            if tags:
                params['tags'] = ','.join(tags)

            self.logger.info(f"Searching datasets with query: {query}, category: {category}, tags: {tags}")

            data = self._make_request(self.discovery_url, params)

            # Process results
            datasets = []
            for item in data.get('results', []):
                resource = item.get('resource', {})
                classification = item.get('classification', {})

                dataset_info = {
                    'id': resource.get('id'),
                    'name': resource.get('name', ''),
                    'description': resource.get('description', ''),
                    'attribution': resource.get('attribution', ''),
                    'type': resource.get('type'),
                    'updated_at': resource.get('updatedAt'),
                    'created_at': resource.get('createdAt'),
                    'download_count': resource.get('download_count', 0),
                    'page_views': resource.get('page_views', {}).get('page_views_total', 0),
                    'columns_count': len(resource.get('columns_name', [])),
                    'columns': resource.get('columns_name', []),
                    'category': classification.get('domain_category', ''),
                    'tags': classification.get('tags', []),
                    'domain_tags': classification.get('domain_tags', [])
                }
                datasets.append(dataset_info)

            result_df = pd.DataFrame(datasets)
            self.logger.info(f"Found {len(result_df)} datasets")

            return result_df

        except Exception as e:
            error_msg = f"Dataset search failed: {str(e)}"
            self.logger.error(error_msg)
            raise APIError(error_msg) from e

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a dataset

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dictionary with detailed metadata
        """
        try:
            url = f"https://data.cityofnewyork.us/api/views/{dataset_id}.json"

            self.logger.info(f"Fetching metadata for dataset {dataset_id}")

            data = self._make_request(url)

            # Extract key metadata
            metadata = {
                'id': data.get('id'),
                'name': data.get('name'),
                'description': data.get('description'),
                'attribution': data.get('attribution'),
                'category': data.get('category'),
                'tags': data.get('tags', []),
                'created_at': data.get('createdAt'),
                'updated_at': data.get('rowsUpdatedAt'),
                'row_count': data.get('rowsUpdatedAt'),  # This is actually the last update time
                'column_count': len(data.get('columns', [])),
                'download_count': data.get('downloadCount', 0),
                'view_count': data.get('viewCount', 0),
                'columns': []
            }

            # Process column information
            for col in data.get('columns', []):
                col_info = {
                    'id': col.get('id'),
                    'name': col.get('name'),
                    'field_name': col.get('fieldName'),
                    'data_type': col.get('dataTypeName'),
                    'description': col.get('description'),
                    'render_type': col.get('renderTypeName'),
                    'width': col.get('width'),
                    'cached_contents': col.get('cachedContents', {})
                }
                metadata['columns'].append(col_info)

            return metadata

        except Exception as e:
            error_msg = f"Failed to get metadata for dataset {dataset_id}: {str(e)}"
            self.logger.error(error_msg)
            raise APIError(error_msg) from e

    def batch_download(self,
                      dataset_configs: List[Dict[str, Any]],
                      max_workers: int = 5,
                      backend: Optional[str] = None) -> Dict[str, Union[pd.DataFrame, 'pl.DataFrame']]:
        """
        Download multiple datasets in parallel

        Args:
            dataset_configs: List of dataset configurations
                Each config should have: {'id': str, 'query': SoQLQueryBuilder, 'name': str}
            max_workers: Maximum parallel workers
            backend: Data backend

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        backend = backend or self.default_backend
        results = {}

        def download_single(config):
            try:
                dataset_id = config['id']
                query_builder = config.get('query')
                name = config.get('name', dataset_id)

                df = self.get_dataset(dataset_id, query_builder, backend)
                return name, df

            except Exception as e:
                self.logger.error(f"Failed to download {config.get('name', config['id'])}: {str(e)}")
                return config.get('name', config['id']), None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {executor.submit(download_single, config): config for config in dataset_configs}

            for future in as_completed(future_to_config):
                name, df = future.result()
                results[name] = df

                if df is not None:
                    self.logger.info(f"Successfully downloaded {name}: {len(df)} rows")
                else:
                    self.logger.error(f"Failed to download {name}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get client usage statistics"""
        return {
            'total_requests': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(self.request_count, 1) * 100,
            'cached_responses': len(self.response_cache),
            'default_backend': self.default_backend,
            'rate_limit_delay': self.rate_limit_delay
        }

    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        self.logger.info("Response cache cleared")

    def export_query_template(self, query_builder: SoQLQueryBuilder,
                            filename: str = None) -> str:
        """
        Export query as reusable template

        Args:
            query_builder: SoQL query builder
            filename: Optional filename to save

        Returns:
            Query template as JSON string
        """
        template = {
            'query_params': query_builder.build(),
            'created_at': datetime.now().isoformat(),
            'description': 'SoQL query template generated by Enhanced NYC Data Client'
        }

        json_str = json.dumps(template, indent=2)

        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
            self.logger.info(f"Query template saved to {filename}")

        return json_str

    def load_query_template(self, filename: str) -> SoQLQueryBuilder:
        """
        Load query from template file

        Args:
            filename: Template file path

        Returns:
            Configured SoQL query builder
        """
        with open(filename, 'r') as f:
            template = json.load(f)

        # Reconstruct query builder
        builder = SoQLQueryBuilder()
        builder.query_params = template['query_params']

        return builder


class ScoutIntegratedClient(EnhancedNYCDataClient):
    """
    NYC Data Client specifically integrated with Scout Data Discovery system
    """

    def __init__(self, scout_instance=None, **kwargs):
        """
        Initialize Scout-integrated client

        Args:
            scout_instance: ScoutDataDiscovery instance for seamless integration
            **kwargs: Arguments passed to EnhancedNYCDataClient
        """
        super().__init__(**kwargs)
        self.scout = scout_instance

    def discover_and_assess(self,
                           search_terms: Union[str, List[str]],
                           quality_threshold: float = 70,
                           max_datasets: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Discover datasets and assess their quality in one step

        Args:
            search_terms: Search terms for dataset discovery
            quality_threshold: Minimum quality score
            max_datasets: Maximum datasets to assess

        Returns:
            Tuple of (datasets_df, quality_assessments)
        """
        if not self.scout:
            raise ConfigurationError("Scout instance required for integrated discovery")

        # Search for datasets
        if isinstance(search_terms, str):
            search_terms = [search_terms]

        all_datasets = []
        for term in search_terms:
            datasets = self.search_datasets(query=term, limit=max_datasets)
            if not datasets.empty:
                all_datasets.append(datasets)

        if not all_datasets:
            return pd.DataFrame(), {}

        combined_datasets = pd.concat(all_datasets, ignore_index=True).drop_duplicates(subset=['id'])

        # Assess quality using Scout
        quality_assessments = {}
        high_quality_datasets = []

        for _, dataset in combined_datasets.head(max_datasets).iterrows():
            try:
                # Download sample using this client
                sample_df = self.get_dataset(dataset['id'], self.query().limit(1000))

                if not sample_df.empty:
                    # Use Scout for quality assessment
                    assessment = self.scout.assess_dataset_quality(
                        dataset['id'],
                        sample_df,
                        metadata=dataset.to_dict()
                    )

                    quality_assessments[dataset['id']] = assessment

                    # Filter by quality threshold
                    if assessment.get('overall_scores', {}).get('total_score', 0) >= quality_threshold:
                        high_quality_datasets.append(dataset)

            except Exception as e:
                self.logger.error(f"Failed to assess dataset {dataset['id']}: {str(e)}")

        high_quality_df = pd.DataFrame(high_quality_datasets) if high_quality_datasets else pd.DataFrame()

        return high_quality_df, quality_assessments

    def get_recommended_datasets(self,
                               reference_dataset_id: str,
                               catalog_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get dataset recommendations using Scout's recommendation engine

        Args:
            reference_dataset_id: Reference dataset for recommendations
            catalog_df: Optional catalog DataFrame (will search if not provided)

        Returns:
            DataFrame with recommended datasets
        """
        if not self.scout:
            raise ConfigurationError("Scout instance required for recommendations")

        if catalog_df is None:
            # Search for related datasets
            reference_metadata = self.get_dataset_metadata(reference_dataset_id)
            search_terms = reference_metadata.get('tags', [])[:3]  # Use first 3 tags

            if search_terms:
                catalog_df = self.search_datasets(' '.join(search_terms), limit=50)
            else:
                catalog_df = pd.DataFrame()

        if catalog_df.empty:
            return pd.DataFrame()

        # Use Scout's recommendation engine
        recommendations = self.scout.generate_recommendations(
            reference_dataset_id,
            catalog_df
        )

        return recommendations