"""
Scout Data Discovery - Main Workflow Class

Comprehensive data discovery and quality assessment workflow based on Scout methodology.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

from .data_quality import DataQualityAssessor
from .enhanced_api_client import EnhancedNYCDataClient, ScoutIntegratedClient, SoQLQueryBuilder
from .exceptions import (
    ScoutDiscoveryError, APIError, DataDownloadError,
    SearchError, ConfigurationError, ValidationError
)


class ScoutDataDiscovery:
    """
    Comprehensive data discovery workflow class implementing Scout methodology.

    This class provides end-to-end data discovery capabilities including:
    - Dataset search and metadata extraction
    - Automated data quality assessment
    - Dataset recommendations
    - Batch processing and caching
    - Export functionality

    Based on Scout (https://scout.tsdataclinic.com/) principles and the Socrata Discovery API.
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 cache_dir: Optional[str] = None,
                 log_level: str = "INFO",
                 max_workers: int = 5,
                 app_token: Optional[str] = None,
                 use_enhanced_client: bool = True,
                 default_backend: str = "pandas"):
        """
        Initialize Scout Data Discovery workflow.

        Args:
            config: Configuration dictionary
            cache_dir: Directory for caching results
            log_level: Logging level
            max_workers: Maximum number of concurrent workers
            app_token: Socrata app token for enhanced API access
            use_enhanced_client: Whether to use the enhanced API client
            default_backend: Default data backend ('pandas' or 'polars')
        """
        # Setup logging
        self.logger = self._setup_logging(log_level)

        # Configuration
        self.config = self._load_config(config)

        # Initialize components
        self.quality_assessor = DataQualityAssessor(logger=self.logger)

        # Always create session first (needed for both modes)
        try:
            self.session = self._create_session()
            self.logger.debug(f"Session created successfully: {type(self.session)}")
        except Exception as e:
            self.logger.error(f"Failed to create session: {str(e)}")
            raise

        # Initialize API clients
        if use_enhanced_client:
            try:
                self.api_client = ScoutIntegratedClient(
                    scout_instance=self,
                    app_token=app_token,
                    default_backend=default_backend,
                    rate_limit_delay=self.config.get('rate_limit_delay', 0.5),
                    max_retries=self.config.get('retry_attempts', 3),
                    logger=self.logger
                )
                self.logger.info("Initialized with enhanced API client")
            except Exception as e:
                self.logger.warning(f"Failed to initialize enhanced client: {str(e)}, falling back to basic mode")
                self.api_client = None
        else:
            # Fallback to basic session
            self.api_client = None

        # Threading
        self.max_workers = max_workers

        # Caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Results storage
        self.results = {
            'search_results': {},
            'quality_assessments': {},
            'recommendations': {},
            'metadata_cache': {},
            'pipeline_stats': {}
        }

        # API endpoints (for backward compatibility)
        self.socrata_discovery_url = "http://api.us.socrata.com/api/catalog/v1"
        self.socrata_resource_base = "https://data.cityofnewyork.us/resource"

        self.logger.info("ScoutDataDiscovery initialized successfully")

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_config(self, config: Optional[Dict]) -> Dict:
        """Load and validate configuration"""
        default_config = {
            'rate_limit_delay': 0.5,
            'request_timeout': 30,
            'retry_attempts': 3,
            'cache_duration_hours': 24,
            'default_sample_size': 1000,
            'quality_threshold': 70,
            'max_search_results': 100,
            'supported_domains': [
                'data.cityofnewyork.us',
                'data.cityofchicago.org',
                'data.seattle.gov',
                'data.sfgov.org'
            ]
        }

        if config:
            # Handle nested config structure
            if 'api' in config:
                api_config = config['api']
                if 'rate_limit_delay' in api_config:
                    default_config['rate_limit_delay'] = api_config['rate_limit_delay']
                if 'request_timeout' in api_config:
                    default_config['request_timeout'] = api_config['request_timeout']
                if 'retry_attempts' in api_config:
                    default_config['retry_attempts'] = api_config['retry_attempts']
            
            if 'data' in config:
                data_config = config['data']
                if 'quality_threshold' in data_config:
                    default_config['quality_threshold'] = data_config['quality_threshold']
                if 'default_sample_size' in data_config:
                    default_config['default_sample_size'] = data_config['default_sample_size']
            
            if 'cache' in config:
                cache_config = config['cache']
                if 'duration_hours' in cache_config:
                    default_config['cache_duration_hours'] = cache_config['duration_hours']
            
            # Also handle flat config structure (backward compatibility)
            flat_keys = ['rate_limit_delay', 'request_timeout', 'retry_attempts', 
                        'cache_duration_hours', 'default_sample_size', 'quality_threshold', 
                        'max_search_results', 'supported_domains']
            for key in flat_keys:
                if key in config:
                    default_config[key] = config[key]

        return default_config

    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'ScoutDataDiscovery/1.0 (Research Tool)',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        return session

    def search_datasets(self,
                       search_terms: Union[str, List[str]],
                       domains: Optional[List[str]] = None,
                       limit: int = None) -> pd.DataFrame:
        """
        Search for datasets using specified terms.

        Args:
            search_terms: Single term or list of search terms
            domains: List of domains to search (uses default if None)
            limit: Maximum number of results per term

        Returns:
            DataFrame containing dataset metadata

        Raises:
            SearchError: If search fails
        """
        try:
            if isinstance(search_terms, str):
                search_terms = [search_terms]

            domains = domains or self.config['supported_domains'][:1]  # Default to NYC
            limit = limit or self.config['max_search_results']

            self.logger.info(f"Searching for datasets: {search_terms} in domains: {domains}")

            all_results = []

            for domain in domains:
                for term in search_terms:
                    self.logger.debug(f"Searching '{term}' in {domain}")

                    results = self._search_single_term(term, domain, limit)

                    if results:
                        # Add search metadata
                        for result in results:
                            result['search_term'] = term
                            result['search_domain'] = domain
                            result['search_timestamp'] = datetime.now().isoformat()

                        all_results.extend(results)
                        self.results['search_results'][f"{term}_{domain}"] = results

                    # Rate limiting
                    time.sleep(self.config['rate_limit_delay'])

            if not all_results:
                self.logger.warning("No datasets found for search terms")
                return pd.DataFrame()

            # Convert to DataFrame and deduplicate
            df = pd.DataFrame(all_results)
            initial_count = len(df)
            df = df.drop_duplicates(subset=['id']).reset_index(drop=True)

            self.logger.info(f"Found {len(df)} unique datasets ({initial_count} total before deduplication)")

            return df

        except Exception as e:
            error_msg = f"Dataset search failed: {str(e)}"
            self.logger.error(error_msg)
            raise SearchError(error_msg) from e

    def _search_single_term(self, term: str, domain: str, limit: int) -> List[Dict]:
        """Search for a single term in a specific domain"""
        params = {
            'domains': domain,
            'search_context': domain,
            'q': term,
            'limit': limit
        }

        for attempt in range(self.config['retry_attempts']):
            try:
                response = self.session.get(
                    self.socrata_discovery_url,
                    params=params,
                    timeout=self.config['request_timeout']
                )
                response.raise_for_status()

                data = response.json()
                results = []

                for dataset in data.get('results', []):
                    metadata = self._extract_dataset_metadata(dataset)
                    results.append(metadata)

                return results

            except requests.RequestException as e:
                self.logger.warning(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config['retry_attempts'] - 1:
                    raise APIError(f"Search failed after {self.config['retry_attempts']} attempts",
                                 getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None)
                time.sleep(2 ** attempt)  # Exponential backoff

    def _extract_dataset_metadata(self, dataset: Dict) -> Dict[str, Any]:
        """Extract and normalize dataset metadata"""
        resource = dataset.get('resource', {})
        classification = dataset.get('classification', {})

        return {
            'id': resource.get('id'),
            'name': (resource.get('name') or '').strip(),
            'description': (resource.get('description') or '').strip(),
            'attribution': (resource.get('attribution') or '').strip(),
            'type': resource.get('type'),
            'updatedAt': resource.get('updatedAt'),
            'createdAt': resource.get('createdAt'),
            'download_count': resource.get('download_count', 0),
            'page_views_total': resource.get('page_views', {}).get('page_views_total', 0),
            'columns_count': len(resource.get('columns_name', [])),
            'columns_names': resource.get('columns_name', []),
            'columns_field_names': resource.get('columns_field_name', []),
            'columns_datatypes': resource.get('columns_datatype', []),
            'domain_category': classification.get('domain_category', ''),
            'domain_tags': classification.get('domain_tags', []),
            'categories': classification.get('categories', []),
            'tags': classification.get('tags', []),
            'metadata_quality': self._assess_metadata_quality(resource, classification)
        }

    def _assess_metadata_quality(self, resource: Dict, classification: Dict) -> Dict[str, Any]:
        """Quick assessment of metadata completeness"""
        score = 0
        max_score = 10

        # Basic fields present
        if resource.get('name'): score += 1
        if resource.get('description'): score += 2
        if resource.get('attribution'): score += 1
        if resource.get('columns_name'): score += 2

        # Classification present
        if classification.get('domain_category'): score += 1
        if classification.get('tags'): score += 1
        if classification.get('domain_tags'): score += 1

        # Usage metrics
        if resource.get('download_count', 0) > 0: score += 1

        return {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100,
            'quality_level': 'High' if score >= 8 else 'Medium' if score >= 5 else 'Low'
        }

    def download_dataset_sample(self,
                              dataset_id: str,
                              sample_size: Optional[int] = None,
                              force_refresh: bool = False,
                              query_builder: Optional[SoQLQueryBuilder] = None,
                              backend: Optional[str] = None) -> Union[pd.DataFrame, Any]:
        """
        Download a sample of the specified dataset.

        Args:
            dataset_id: Unique dataset identifier
            sample_size: Number of rows to download
            force_refresh: Force download even if cached
            query_builder: Optional SoQL query builder for advanced filtering
            backend: Data backend ('pandas' or 'polars')

        Returns:
            DataFrame containing dataset sample

        Raises:
            DataDownloadError: If download fails
        """
        try:
            sample_size = sample_size or self.config['default_sample_size']

            # Use enhanced client if available
            if self.api_client:
                # Build query if not provided
                if query_builder is None:
                    query_builder = self.api_client.query().limit(sample_size)
                elif query_builder and not any('limit' in str(param).lower() for param in query_builder.query_params):
                    query_builder = query_builder.limit(sample_size)

                df = self.api_client.get_dataset(dataset_id, query_builder, backend)
                self.logger.info(f"Downloaded {len(df)} rows using enhanced client")
                return df

            # Fallback to original implementation
            # Check cache first
            cache_file = self.cache_dir / f"{dataset_id}_sample.parquet"
            if not force_refresh and cache_file.exists():
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(hours=self.config['cache_duration_hours']):
                    self.logger.debug(f"Loading cached sample for {dataset_id}")
                    return pd.read_parquet(cache_file)

            self.logger.info(f"Downloading sample of {sample_size} rows from {dataset_id}")

            # Download from Socrata
            url = f"{self.socrata_resource_base}/{dataset_id}.json"
            params = {'$limit': sample_size}

            for attempt in range(self.config['retry_attempts']):
                try:
                    response = self.session.get(
                        url,
                        params=params,
                        timeout=self.config['request_timeout']
                    )
                    response.raise_for_status()

                    data = response.json()

                    if not data:
                        self.logger.warning(f"Dataset {dataset_id} returned empty data")
                        return pd.DataFrame()

                    df = pd.DataFrame(data)

                    # Cache the result
                    df.to_parquet(cache_file, index=False)

                    self.logger.info(f"Downloaded {len(df)} rows, {len(df.columns)} columns from {dataset_id}")
                    return df

                except requests.RequestException as e:
                    self.logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.config['retry_attempts'] - 1:
                        raise DataDownloadError(
                            f"Failed to download {dataset_id} after {self.config['retry_attempts']} attempts",
                            dataset_id
                        ) from e
                    time.sleep(2 ** attempt)

        except Exception as e:
            error_msg = f"Dataset download failed for {dataset_id}: {str(e)}"
            self.logger.error(error_msg)
            raise DataDownloadError(error_msg, dataset_id) from e

    def assess_dataset_quality(self,
                             dataset_id: str,
                             df: Optional[pd.DataFrame] = None,
                             metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Assess quality of a dataset.

        Args:
            dataset_id: Dataset identifier
            df: DataFrame to assess (will download if not provided)
            metadata: Additional metadata for assessment

        Returns:
            Quality assessment results
        """
        try:
            if df is None:
                df = self.download_dataset_sample(dataset_id)

            if df.empty:
                self.logger.warning(f"Cannot assess quality: dataset {dataset_id} is empty")
                return {'error': 'Dataset is empty', 'dataset_id': dataset_id}

            self.logger.info(f"Assessing quality for dataset {dataset_id}")

            # Use cached metadata if available
            if not metadata and dataset_id in self.results['metadata_cache']:
                metadata = self.results['metadata_cache'][dataset_id]

            assessment = self.quality_assessor.assess_dataset_quality(dataset_id, df, metadata)

            # Cache the assessment
            self.results['quality_assessments'][dataset_id] = assessment

            return assessment

        except Exception as e:
            error_msg = f"Quality assessment failed for {dataset_id}: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'dataset_id': dataset_id}

    def generate_recommendations(self,
                               dataset_id: str,
                               catalog_df: pd.DataFrame,
                               top_n: int = 5) -> pd.DataFrame:
        """
        Generate dataset recommendations based on similarity.

        Args:
            dataset_id: Target dataset ID
            catalog_df: DataFrame containing catalog of datasets
            top_n: Number of recommendations to return

        Returns:
            DataFrame with recommended datasets
        """
        try:
            target_dataset = catalog_df[catalog_df['id'] == dataset_id]

            if target_dataset.empty:
                self.logger.warning(f"Dataset {dataset_id} not found in catalog")
                return pd.DataFrame()

            target_row = target_dataset.iloc[0]
            recommendations = []

            for _, candidate_row in catalog_df.iterrows():
                if candidate_row['id'] == dataset_id:
                    continue

                similarity_score = self._calculate_similarity(target_row, candidate_row)

                recommendations.append({
                    'id': candidate_row['id'],
                    'name': candidate_row['name'],
                    'similarity_score': similarity_score,
                    'domain_category': candidate_row.get('domain_category', ''),
                    'download_count': candidate_row.get('download_count', 0),
                    'recommendation_reason': self._generate_recommendation_reason(target_row, candidate_row)
                })

            # Sort by similarity score and return top N
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df = recommendations_df.sort_values('similarity_score', ascending=False).head(top_n)

            # Cache recommendations
            self.results['recommendations'][dataset_id] = recommendations_df

            self.logger.info(f"Generated {len(recommendations_df)} recommendations for {dataset_id}")

            return recommendations_df

        except Exception as e:
            error_msg = f"Recommendation generation failed for {dataset_id}: {str(e)}"
            self.logger.error(error_msg)
            return pd.DataFrame()

    def _calculate_similarity(self, target: pd.Series, candidate: pd.Series) -> float:
        """Calculate similarity score between two datasets"""
        score = 0.0

        # Category similarity (40% weight)
        if target.get('domain_category') == candidate.get('domain_category'):
            score += 0.4

        # Tag similarity (30% weight)
        target_tags = set(target.get('domain_tags', []) + target.get('tags', []))
        candidate_tags = set(candidate.get('domain_tags', []) + candidate.get('tags', []))

        if target_tags and candidate_tags:
            tag_similarity = len(target_tags.intersection(candidate_tags)) / len(target_tags.union(candidate_tags))
            score += tag_similarity * 0.3

        # Column name similarity (20% weight)
        target_columns = set(target.get('columns_names', []))
        candidate_columns = set(candidate.get('columns_names', []))

        if target_columns and candidate_columns:
            column_similarity = len(target_columns.intersection(candidate_columns)) / len(target_columns.union(candidate_columns))
            score += column_similarity * 0.2

        # Description similarity (10% weight) - simple word overlap
        target_desc = set(str(target.get('description', '')).lower().split())
        candidate_desc = set(str(candidate.get('description', '')).lower().split())

        if target_desc and candidate_desc:
            desc_similarity = len(target_desc.intersection(candidate_desc)) / len(target_desc.union(candidate_desc))
            score += desc_similarity * 0.1

        return min(1.0, score)

    def _generate_recommendation_reason(self, target: pd.Series, candidate: pd.Series) -> str:
        """Generate human-readable reason for recommendation"""
        reasons = []

        if target.get('domain_category') == candidate.get('domain_category'):
            reasons.append("Same category")

        target_tags = set(target.get('domain_tags', []))
        candidate_tags = set(candidate.get('domain_tags', []))
        common_tags = target_tags.intersection(candidate_tags)
        if common_tags:
            reasons.append(f"Shared tags: {', '.join(list(common_tags)[:3])}")

        target_columns = set(target.get('columns_names', []))
        candidate_columns = set(candidate.get('columns_names', []))
        common_columns = target_columns.intersection(candidate_columns)
        if len(common_columns) > 2:
            reasons.append(f"Similar data structure ({len(common_columns)} common columns)")

        return "; ".join(reasons) if reasons else "General similarity"

    def run_discovery_pipeline(self,
                             search_terms: Union[str, List[str]],
                             domains: Optional[List[str]] = None,
                             quality_threshold: Optional[float] = None,
                             max_assessments: int = 20,
                             include_recommendations: bool = True,
                             export_results: bool = False) -> Dict[str, Any]:
        """
        Run complete data discovery pipeline.

        Args:
            search_terms: Terms to search for
            domains: Domains to search in
            quality_threshold: Minimum quality score for inclusion
            max_assessments: Maximum number of datasets to assess
            include_recommendations: Whether to generate recommendations
            export_results: Whether to export results to files

        Returns:
            Complete pipeline results
        """
        try:
            start_time = datetime.now()
            self.logger.info("🚀 Starting Scout Data Discovery Pipeline")

            # Clear previous results
            self.results = {
                'search_results': {},
                'quality_assessments': {},
                'recommendations': {},
                'metadata_cache': {},
                'pipeline_stats': {}
            }

            # Step 1: Search for datasets
            self.logger.info("📋 Step 1: Searching for datasets")
            datasets_df = self.search_datasets(search_terms, domains)

            if datasets_df.empty:
                self.logger.warning("No datasets found. Pipeline terminated.")
                return self._generate_pipeline_results(start_time, datasets_df)

            # Cache metadata
            for _, row in datasets_df.iterrows():
                self.results['metadata_cache'][row['id']] = row.to_dict()

            # Step 2: Quality assessment
            self.logger.info(f"🔍 Step 2: Quality Assessment (up to {max_assessments} datasets)")

            # Select datasets for assessment (prioritize by popularity)
            assessment_candidates = self._select_assessment_candidates(datasets_df, max_assessments)

            # Parallel quality assessment
            quality_results = self._parallel_quality_assessment(assessment_candidates)

            # Filter by quality threshold
            quality_threshold = quality_threshold or self.config['quality_threshold']
            high_quality_datasets = {
                dataset_id: assessment
                for dataset_id, assessment in quality_results.items()
                if assessment.get('overall_scores', {}).get('total_score', 0) >= quality_threshold
            }

            self.logger.info(f"Found {len(high_quality_datasets)} datasets above quality threshold {quality_threshold}")

            # Step 3: Generate recommendations
            recommendations = {}
            if include_recommendations and len(quality_results) > 1:
                self.logger.info("🎯 Step 3: Generating recommendations")

                for dataset_id in list(high_quality_datasets.keys())[:5]:  # Top 5 for recommendations
                    recs = self.generate_recommendations(dataset_id, datasets_df)
                    if not recs.empty:
                        recommendations[dataset_id] = recs

            # Step 4: Generate pipeline results
            pipeline_results = self._generate_pipeline_results(
                start_time, datasets_df, quality_results, recommendations
            )

            # Step 5: Export if requested
            if export_results:
                self.logger.info("📁 Step 5: Exporting results")
                self.export_results(pipeline_results)

            self.logger.info(f"✅ Pipeline completed successfully in {pipeline_results['execution_time_seconds']:.1f} seconds")

            return pipeline_results

        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error(error_msg)
            raise ScoutDiscoveryError(error_msg) from e

    def _select_assessment_candidates(self, df: pd.DataFrame, max_count: int) -> pd.DataFrame:
        """Select most promising datasets for quality assessment"""
        if len(df) <= max_count:
            return df

        # Score datasets by metadata quality and popularity
        df_scored = df.copy()
        df_scored['assessment_priority'] = (
            df_scored['download_count'].fillna(0) * 0.4 +
            df_scored['page_views_total'].fillna(0) * 0.3 +
            df_scored['metadata_quality'].apply(lambda x: x.get('score', 0) if isinstance(x, dict) else 0) * 0.3
        )

        return df_scored.nlargest(max_count, 'assessment_priority')

    def _parallel_quality_assessment(self, candidates_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform quality assessment in parallel"""
        quality_results = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(candidates_df))) as executor:
            # Submit assessment tasks
            future_to_dataset = {}

            for _, row in candidates_df.iterrows():
                dataset_id = row['id']
                future = executor.submit(self._assess_single_dataset, dataset_id, row.to_dict())
                future_to_dataset[future] = dataset_id

            # Collect results
            for future in as_completed(future_to_dataset):
                dataset_id = future_to_dataset[future]
                try:
                    assessment = future.result()
                    quality_results[dataset_id] = assessment

                    if 'error' not in assessment:
                        score = assessment.get('overall_scores', {}).get('total_score', 0)
                        self.logger.info(f"✓ {dataset_id}: Quality score {score:.1f}/100")
                    else:
                        self.logger.warning(f"✗ {dataset_id}: Assessment failed")

                except Exception as e:
                    self.logger.error(f"Assessment failed for {dataset_id}: {str(e)}")
                    quality_results[dataset_id] = {'error': str(e), 'dataset_id': dataset_id}

        return quality_results

    def _assess_single_dataset(self, dataset_id: str, metadata: Dict) -> Dict[str, Any]:
        """Assess a single dataset (for parallel execution)"""
        try:
            # Download sample
            df = self.download_dataset_sample(dataset_id)

            if df.empty:
                return {'error': 'Dataset is empty', 'dataset_id': dataset_id}

            # Assess quality
            return self.quality_assessor.assess_dataset_quality(dataset_id, df, metadata)

        except Exception as e:
            return {'error': str(e), 'dataset_id': dataset_id}

    def _generate_pipeline_results(self,
                                 start_time: datetime,
                                 datasets_df: pd.DataFrame,
                                 quality_results: Optional[Dict] = None,
                                 recommendations: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive pipeline results"""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        quality_results = quality_results or {}
        recommendations = recommendations or {}

        # Calculate quality statistics
        valid_assessments = [
            assessment for assessment in quality_results.values()
            if 'error' not in assessment and 'overall_scores' in assessment
        ]

        quality_stats = {}
        if valid_assessments:
            scores = [a['overall_scores']['total_score'] for a in valid_assessments]
            quality_stats = {
                'mean_score': np.mean(scores),
                'median_score': np.median(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'std_score': np.std(scores)
            }

        # Pipeline statistics
        pipeline_stats = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'execution_time_seconds': execution_time,
            'datasets_found': len(datasets_df),
            'datasets_assessed': len(valid_assessments),
            'assessment_success_rate': len(valid_assessments) / len(quality_results) * 100 if quality_results else 0,
            'recommendations_generated': len(recommendations),
            'quality_statistics': quality_stats
        }

        return {
            'datasets_metadata': datasets_df,
            'quality_assessments': quality_results,
            'recommendations': recommendations,
            'pipeline_stats': pipeline_stats,
            'search_results': self.results['search_results'],
            'metadata_cache': self.results['metadata_cache']
        }

    def export_results(self,
                      results: Dict[str, Any],
                      output_dir: Optional[str] = None,
                      filename_prefix: Optional[str] = None) -> List[str]:
        """
        Export pipeline results to files.

        Args:
            results: Pipeline results to export
            output_dir: Output directory
            filename_prefix: Prefix for output files

        Returns:
            List of exported file paths
        """
        try:
            output_dir = Path(output_dir) if output_dir else Path("exports")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = filename_prefix or f"scout_discovery_{timestamp}"

            exported_files = []

            # Export datasets metadata
            if not results['datasets_metadata'].empty:
                catalog_file = output_dir / f"{prefix}_catalog.csv"
                results['datasets_metadata'].to_csv(catalog_file, index=False)
                exported_files.append(str(catalog_file))
                self.logger.info(f"Exported dataset catalog to {catalog_file}")

            # Export quality assessments
            if results['quality_assessments']:
                quality_file = output_dir / f"{prefix}_quality_assessments.json"
                with open(quality_file, 'w') as f:
                    json.dump(results['quality_assessments'], f, indent=2, default=str)
                exported_files.append(str(quality_file))

                # Also export quality summary
                quality_df = self.quality_assessor.generate_quality_report(results['quality_assessments'])
                if not quality_df.empty:
                    quality_summary_file = output_dir / f"{prefix}_quality_summary.csv"
                    quality_df.to_csv(quality_summary_file, index=False)
                    exported_files.append(str(quality_summary_file))

            # Export recommendations
            if results['recommendations']:
                rec_file = output_dir / f"{prefix}_recommendations.json"
                recommendations_serializable = {}
                for dataset_id, rec_df in results['recommendations'].items():
                    recommendations_serializable[dataset_id] = rec_df.to_dict('records')

                with open(rec_file, 'w') as f:
                    json.dump(recommendations_serializable, f, indent=2, default=str)
                exported_files.append(str(rec_file))

            # Export pipeline statistics
            stats_file = output_dir / f"{prefix}_pipeline_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(results['pipeline_stats'], f, indent=2, default=str)
            exported_files.append(str(stats_file))

            # Export summary report
            summary_file = output_dir / f"{prefix}_summary.txt"
            self._export_text_summary(results, summary_file)
            exported_files.append(str(summary_file))

            self.logger.info(f"Exported {len(exported_files)} files to {output_dir}")

            return exported_files

        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self.logger.error(error_msg)
            raise ScoutDiscoveryError(error_msg) from e

    def _export_text_summary(self, results: Dict[str, Any], filepath: Path):
        """Export human-readable summary report"""
        with open(filepath, 'w') as f:
            f.write("SCOUT DATA DISCOVERY PIPELINE SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            stats = results['pipeline_stats']

            # Pipeline overview
            f.write(f"Execution Time: {stats['execution_time_seconds']:.1f} seconds\n")
            f.write(f"Datasets Found: {stats['datasets_found']}\n")
            f.write(f"Datasets Assessed: {stats['datasets_assessed']}\n")
            f.write(f"Assessment Success Rate: {stats['assessment_success_rate']:.1f}%\n")
            f.write(f"Recommendations Generated: {stats['recommendations_generated']}\n\n")

            # Quality statistics
            quality_stats = stats.get('quality_statistics', {})
            if quality_stats:
                f.write("QUALITY STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Mean Quality Score: {quality_stats['mean_score']:.1f}/100\n")
                f.write(f"Median Quality Score: {quality_stats['median_score']:.1f}/100\n")
                f.write(f"Score Range: {quality_stats['min_score']:.1f} - {quality_stats['max_score']:.1f}\n\n")

            # Top datasets
            if not results['datasets_metadata'].empty:
                f.write("TOP DATASETS BY POPULARITY\n")
                f.write("-" * 30 + "\n")
                top_datasets = results['datasets_metadata'].nlargest(5, 'download_count')
                for i, (_, dataset) in enumerate(top_datasets.iterrows(), 1):
                    f.write(f"{i}. {dataset['name']} (Downloads: {dataset['download_count']})\n")

    def get_cached_results(self) -> Dict[str, Any]:
        """Get cached results from current session"""
        return self.results.copy()

    def clear_cache(self):
        """Clear all cached data"""
        try:
            # Clear memory cache
            self.results = {
                'search_results': {},
                'quality_assessments': {},
                'recommendations': {},
                'metadata_cache': {},
                'pipeline_stats': {}
            }

            # Clear file cache
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.parquet"):
                    cache_file.unlink()

            self.logger.info("Cache cleared successfully")

        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a specific dataset"""
        try:
            info = {
                'dataset_id': dataset_id,
                'metadata': self.results['metadata_cache'].get(dataset_id),
                'quality_assessment': self.results['quality_assessments'].get(dataset_id),
                'recommendations': None
            }

            # Check if recommendations exist for this dataset
            for rec_dataset_id, recommendations in self.results['recommendations'].items():
                if rec_dataset_id == dataset_id:
                    info['recommendations'] = recommendations.to_dict('records') if hasattr(recommendations, 'to_dict') else recommendations
                    break

            return info if info['metadata'] else None

        except Exception as e:
            self.logger.error(f"Failed to get dataset info for {dataset_id}: {str(e)}")
            return None

    # Enhanced methods leveraging the new API client

    def create_query_builder(self) -> Optional[SoQLQueryBuilder]:
        """
        Create a SoQL query builder for advanced dataset filtering.

        Returns:
            SoQLQueryBuilder instance if enhanced client is available
        """
        if self.api_client:
            return self.api_client.query()
        else:
            self.logger.warning("Enhanced API client not available. Initialize with use_enhanced_client=True")
            return None

    def search_datasets_advanced(self,
                                query: str = None,
                                category: str = None,
                                tags: List[str] = None,
                                date_range: Tuple[datetime, datetime] = None,
                                min_downloads: int = None,
                                limit: int = 100) -> pd.DataFrame:
        """
        Advanced dataset search with multiple filter options.

        Args:
            query: Text search query
            category: Dataset category filter
            tags: Tag filters
            date_range: Tuple of (start_date, end_date) for filtering by update time
            min_downloads: Minimum download count filter
            limit: Maximum results

        Returns:
            DataFrame with filtered datasets
        """
        if not self.api_client:
            self.logger.warning("Enhanced client not available, falling back to basic search")
            return self.search_datasets(query or "", limit=limit)

        try:
            # Use enhanced client for advanced search
            datasets_df = self.api_client.search_datasets(
                query=query,
                category=category,
                tags=tags,
                limit=limit
            )

            if datasets_df.empty:
                return datasets_df

            # Apply additional filters
            if date_range:
                start_date, end_date = date_range
                datasets_df['updated_at'] = pd.to_datetime(datasets_df['updated_at'], errors='coerce')
                datasets_df = datasets_df[
                    (datasets_df['updated_at'] >= start_date) &
                    (datasets_df['updated_at'] <= end_date)
                ]

            if min_downloads:
                datasets_df = datasets_df[datasets_df['download_count'] >= min_downloads]

            self.logger.info(f"Advanced search returned {len(datasets_df)} datasets")
            return datasets_df

        except Exception as e:
            error_msg = f"Advanced search failed: {str(e)}"
            self.logger.error(error_msg)
            raise SearchError(error_msg) from e

    def download_dataset_streaming(self,
                                  dataset_id: str,
                                  chunk_size: int = 10000,
                                  query_builder: Optional[SoQLQueryBuilder] = None,
                                  backend: Optional[str] = None):
        """
        Stream large datasets in chunks for memory-efficient processing.

        Args:
            dataset_id: Dataset identifier
            chunk_size: Size of each chunk
            query_builder: Optional SoQL query for filtering
            backend: Data backend ('pandas' or 'polars')

        Yields:
            DataFrame chunks
        """
        if not self.api_client:
            raise ConfigurationError("Enhanced client required for streaming. Initialize with use_enhanced_client=True")

        try:
            self.logger.info(f"Starting streaming download of {dataset_id} with chunk size {chunk_size}")

            chunk_count = 0
            for chunk_df in self.api_client.get_dataset_streaming(dataset_id, chunk_size, query_builder, backend):
                chunk_count += 1
                self.logger.debug(f"Yielding chunk {chunk_count} with {len(chunk_df)} rows")
                yield chunk_df

            self.logger.info(f"Streaming completed: {chunk_count} chunks processed")

        except Exception as e:
            error_msg = f"Streaming download failed for {dataset_id}: {str(e)}"
            self.logger.error(error_msg)
            raise DataDownloadError(error_msg, dataset_id) from e

    def batch_download_datasets(self,
                               dataset_configs: List[Dict[str, Any]],
                               max_workers: Optional[int] = None,
                               backend: Optional[str] = None) -> Dict[str, Any]:
        """
        Download multiple datasets in parallel with enhanced error handling.

        Args:
            dataset_configs: List of dataset configurations
                Each config: {'id': str, 'query': SoQLQueryBuilder, 'name': str}
            max_workers: Maximum parallel workers
            backend: Data backend

        Returns:
            Dictionary with results and statistics
        """
        if not self.api_client:
            raise ConfigurationError("Enhanced client required for batch download")

        try:
            max_workers = max_workers or self.max_workers

            self.logger.info(f"Starting batch download of {len(dataset_configs)} datasets")

            results = self.api_client.batch_download(dataset_configs, max_workers, backend)

            # Compile statistics
            successful_downloads = sum(1 for df in results.values() if df is not None)
            failed_downloads = len(results) - successful_downloads

            stats = {
                'total_datasets': len(dataset_configs),
                'successful_downloads': successful_downloads,
                'failed_downloads': failed_downloads,
                'success_rate': successful_downloads / len(dataset_configs) * 100
            }

            self.logger.info(f"Batch download completed: {successful_downloads}/{len(dataset_configs)} successful")

            return {
                'datasets': results,
                'statistics': stats
            }

        except Exception as e:
            error_msg = f"Batch download failed: {str(e)}"
            self.logger.error(error_msg)
            raise DataDownloadError(error_msg) from e

    def discover_and_assess_integrated(self,
                                     search_terms: Union[str, List[str]],
                                     quality_threshold: float = None,
                                     max_datasets: int = 20) -> Dict[str, Any]:
        """
        Integrated discovery and assessment using enhanced client.

        Args:
            search_terms: Search terms for discovery
            quality_threshold: Minimum quality threshold
            max_datasets: Maximum datasets to process

        Returns:
            Comprehensive results with datasets and assessments
        """
        if not self.api_client:
            self.logger.warning("Enhanced client not available, using standard pipeline")
            return self.run_discovery_pipeline(search_terms, max_assessments=max_datasets)

        try:
            quality_threshold = quality_threshold or self.config.get('data', {}).get('quality_threshold', 70)

            self.logger.info(f"Starting integrated discovery and assessment")

            # Use integrated client method
            high_quality_datasets, quality_assessments = self.api_client.discover_and_assess(
                search_terms=search_terms,
                quality_threshold=quality_threshold,
                max_datasets=max_datasets
            )

            # Store results
            for dataset_id, assessment in quality_assessments.items():
                self.results['quality_assessments'][dataset_id] = assessment

            # Generate recommendations for top datasets
            recommendations = {}
            if not high_quality_datasets.empty:
                for _, dataset in high_quality_datasets.head(5).iterrows():
                    try:
                        recs = self.api_client.get_recommended_datasets(
                            dataset['id'],
                            high_quality_datasets
                        )
                        if not recs.empty:
                            recommendations[dataset['id']] = recs
                    except Exception as e:
                        self.logger.warning(f"Failed to generate recommendations for {dataset['id']}: {str(e)}")

            # Compile results
            results = {
                'high_quality_datasets': high_quality_datasets,
                'quality_assessments': quality_assessments,
                'recommendations': recommendations,
                'statistics': {
                    'datasets_found': len(high_quality_datasets),
                    'datasets_assessed': len(quality_assessments),
                    'quality_threshold': quality_threshold,
                    'recommendations_generated': len(recommendations)
                }
            }

            self.logger.info(f"Integrated discovery completed: {len(high_quality_datasets)} high-quality datasets found")

            return results

        except Exception as e:
            error_msg = f"Integrated discovery and assessment failed: {str(e)}"
            self.logger.error(error_msg)
            raise ScoutDiscoveryError(error_msg) from e

    def get_api_statistics(self) -> Dict[str, Any]:
        """
        Get API usage statistics from enhanced client.

        Returns:
            Dictionary with API usage stats
        """
        if self.api_client:
            api_stats = self.api_client.get_statistics()
            return {
                'enhanced_client_stats': api_stats,
                'scout_stats': {
                    'cached_search_results': len(self.results['search_results']),
                    'cached_quality_assessments': len(self.results['quality_assessments']),
                    'cached_recommendations': len(self.results['recommendations'])
                }
            }
        else:
            return {
                'enhanced_client_available': False,
                'scout_stats': {
                    'cached_search_results': len(self.results['search_results']),
                    'cached_quality_assessments': len(self.results['quality_assessments']),
                    'cached_recommendations': len(self.results['recommendations'])
                }
            }