"""
Data Ingestion Component
Handles data upload, processing, and management for the dashboard
"""

import pandas as pd
import io
import base64
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)


"""
Data Ingestion Component
Handles data upload, processing, and management for the dashboard
"""

import pandas as pd
import io
import base64
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import requests
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add scout_data_discovery to path
scout_path = Path(__file__).parent.parent.parent / "scout_data_discovery"
sys.path.append(str(scout_path))

# Import Scout components
try:
    from src.scout_discovery import ScoutDataDiscovery
    SCOUT_AVAILABLE = True
    logger.info("✅ Scout Data Discovery loaded successfully")
except ImportError as e:
    SCOUT_AVAILABLE = False
    logger.warning(f"❌ Scout Data Discovery not available: {e}")
    ScoutDataDiscovery = None


class DataIngestionComponent:
    """Component for handling data ingestion and management."""

    def __init__(self, api_base_url: str = "http://localhost:8080/api"):
        """Initialize the data ingestion component."""
        self.datasets = {}
        self.dataset_metadata = {}
        self.api_base_url = api_base_url
        self.scout_instance = None
        
        # Initialize Scout instance if available
        if SCOUT_AVAILABLE:
            try:
                self.scout_instance = ScoutDataDiscovery()
                logger.info("✅ Scout instance initialized")
            except Exception as e:
                logger.warning(f"❌ Failed to initialize Scout instance: {e}")

    def fetch_datasets_from_api(self, limit: int = 10) -> Dict[str, Any]:
        """
        Fetch datasets from the Scout API the same way the frontend does.
        
        Args:
            limit: Maximum number of datasets to fetch
            
        Returns:
            Dict containing fetched datasets info
        """
        try:
            if not self.scout_instance:
                logger.warning("Scout instance not available, cannot fetch from API")
                return {"error": "Scout instance not available"}
            
            logger.info(f"Fetching top {limit} updated datasets from Scout API...")
            
            # Use the same search terms as the frontend
            search_terms = ["311", "health", "transportation", "housing", "business", "education"]
            
            # Search for datasets using Scout
            datasets_df = self.scout_instance.search_datasets(search_terms, limit=limit*2)
            
            if datasets_df.empty:
                logger.warning("No datasets found from Scout API")
                return {"datasets": [], "message": "No datasets available"}
            
            # Sort by updated date (same as frontend)
            datasets_df['updatedAt'] = pd.to_datetime(datasets_df['updatedAt'], errors='coerce')
            datasets_df = datasets_df.sort_values('updatedAt', ascending=False).head(limit)
            
            fetched_datasets = {}
            
            for idx, (_, row) in enumerate(datasets_df.iterrows()):
                dataset_id = row['id']
                
                try:
                    # Download sample data for each dataset (same as frontend)
                    logger.info(f"Downloading sample for dataset: {dataset_id}")
                    sample_df = self.scout_instance.download_dataset_sample(dataset_id, sample_size=100)
                    
                    if not sample_df.empty:
                        # Store the dataset
                        internal_id = f"scout_dataset_{idx}"
                        self.datasets[internal_id] = sample_df
                        
                        # Store metadata
                        self.dataset_metadata[internal_id] = {
                            'filename': row['name'] or f"Dataset {dataset_id}",
                            'rows': len(sample_df),
                            'columns': len(sample_df.columns),
                            'column_names': list(sample_df.columns),
                            'dtypes': sample_df.dtypes.to_dict(),
                            'size_mb': sample_df.memory_usage(deep=True).sum() / 1024 / 1024,
                            'source': 'scout_api',
                            'dataset_id': dataset_id,
                            'updated_at': row['updatedAt'].isoformat() if pd.notna(row['updatedAt']) else None,
                            'category': row.get('domain_category') or 'Uncategorized',
                            'description': row['description'] or 'No description available'
                        }
                        
                        fetched_datasets[internal_id] = {
                            'filename': row['name'] or f"Dataset {dataset_id}",
                            'shape': sample_df.shape,
                            'columns': list(sample_df.columns),
                            'source': 'scout_api',
                            'category': row.get('domain_category') or 'Uncategorized'
                        }
                        
                        logger.info(f"✅ Processed Scout dataset {internal_id}: {row['name']}")
                    else:
                        logger.warning(f"Empty sample data for dataset {dataset_id}")
                        
                except Exception as e:
                    logger.error(f"Error downloading sample for {dataset_id}: {str(e)}")
                    continue
            
            logger.info(f"✅ Fetched {len(fetched_datasets)} datasets from Scout API")
            return {"datasets": fetched_datasets, "total_found": len(datasets_df)}
            
        except Exception as e:
            logger.error(f"Error fetching datasets from API: {str(e)}")
            return {"error": str(e), "datasets": {}}
        """
        Process uploaded files and convert to DataFrames.

        Args:
            contents: List of file contents (base64 encoded)
            filenames: List of filenames

        Returns:
            Dict containing processed datasets info
        """
        new_datasets = {}

        for content, filename in zip(contents, filenames):
            try:
                # Decode base64 content
                content_string = content.split(',')[1]
                decoded = base64.b64decode(content_string)

                # Determine file type and read accordingly
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(io.BytesIO(decoded))
                elif filename.endswith('.json'):
                    df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
                else:
                    logger.warning(f"Unsupported file type: {filename}")
                    continue

                # Generate dataset ID
                dataset_id = f"dataset_{len(self.datasets)}"

                # Store dataset
                self.datasets[dataset_id] = df

                # Store metadata
                self.dataset_metadata[dataset_id] = {
                    'filename': filename,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                }

                new_datasets[dataset_id] = {
                    'filename': filename,
                    'shape': df.shape,
                    'columns': list(df.columns)
                }

                logger.info(f"Processed dataset {dataset_id}: {filename}")

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue

        return new_datasets

    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Get a dataset by ID.

        Args:
            dataset_id: The dataset identifier

        Returns:
            DataFrame if found, None otherwise
        """
        return self.datasets.get(dataset_id)

    def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a dataset.

        Args:
            dataset_id: The dataset identifier

        Returns:
            Metadata dict if found, None otherwise
        """
        return self.dataset_metadata.get(dataset_id)

    def list_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available datasets with their metadata.

        Returns:
            Dict of dataset_id -> metadata
        """
        return self.dataset_metadata.copy()

    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset.

        Args:
            dataset_id: The dataset identifier

        Returns:
            True if deleted, False if not found
        """
        if dataset_id in self.datasets:
            del self.datasets[dataset_id]
            del self.dataset_metadata[dataset_id]
            logger.info(f"Deleted dataset {dataset_id}")
            return True
        return False

    def get_column_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed column information for a dataset.

        Args:
            dataset_id: The dataset identifier

        Returns:
            Column information dict if found, None otherwise
        """
        df = self.get_dataset(dataset_id)
        if df is None:
            return None

        column_info = {}
        for col in df.columns:
            column_info[col] = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'sample_values': df[col].dropna().head(3).tolist()
            }

        return column_info

    def get_newest_dataset(self) -> Optional[tuple]:
        """
        Get the newest (most recently added) dataset.

        Returns:
            Tuple of (dataset_id, DataFrame, metadata) for the newest dataset,
            or None if no datasets exist
        """
        if not self.datasets:
            return None

        # Find the dataset with the highest ID number (newest)
        dataset_ids = list(self.datasets.keys())
        if not dataset_ids:
            return None

        # Sort by the numeric part of the dataset ID
        def get_dataset_number(dataset_id):
            try:
                return int(dataset_id.split('_')[1])
            except (ValueError, IndexError):
                return 0

        newest_id = max(dataset_ids, key=get_dataset_number)
        df = self.datasets[newest_id]
        metadata = self.dataset_metadata.get(newest_id, {})

        return (newest_id, df, metadata)