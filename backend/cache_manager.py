"""
Cache Manager for Scout Data Discovery Backend

Implements intelligent caching for API responses, dataset samples,
and dashboard data to improve performance and reduce timeout errors.
"""

import json
import hashlib
import pickle
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd


class CacheManager:
    """
    Manages caching for the Scout Data Discovery backend with multiple cache types
    """

    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Create subdirectories for different cache types
        (self.cache_dir / "api_responses").mkdir(exist_ok=True)
        (self.cache_dir / "datasets").mkdir(exist_ok=True)
        (self.cache_dir / "dashboard").mkdir(exist_ok=True)
        (self.cache_dir / "quality").mkdir(exist_ok=True)

        self.default_ttl = default_ttl
        self.logger = logging.getLogger(__name__)

        # Memory cache for frequently accessed data
        self.memory_cache = {}
        self.memory_cache_timestamps = {}
        self.memory_cache_max_size = 100
        self.memory_cache_ttl = 300  # 5 minutes

        self.logger.info(f"Cache manager initialized with directory: {self.cache_dir}")

    def _generate_cache_key(self, data: Any) -> str:
        """Generate a unique cache key from data"""
        if isinstance(data, dict):
            # Sort dict for consistent hashing
            sorted_data = json.dumps(data, sort_keys=True)
        else:
            sorted_data = str(data)

        return hashlib.md5(sorted_data.encode()).hexdigest()

    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """Get the full path for a cache file"""
        return self.cache_dir / cache_type / f"{key}.cache"

    def _is_cache_valid(self, cache_path: Path, ttl: Optional[int] = None) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False

        ttl = ttl or self.default_ttl
        file_age = time.time() - cache_path.stat().st_mtime
        return file_age < ttl

    def set_memory_cache(self, key: str, data: Any, ttl: Optional[int] = None):
        """Set item in memory cache"""
        ttl = ttl or self.memory_cache_ttl

        # Clean up old entries if cache is full
        if len(self.memory_cache) >= self.memory_cache_max_size:
            self._cleanup_memory_cache()

        self.memory_cache[key] = data
        self.memory_cache_timestamps[key] = time.time() + ttl

    def get_memory_cache(self, key: str) -> Optional[Any]:
        """Get item from memory cache"""
        if key not in self.memory_cache:
            return None

        if time.time() > self.memory_cache_timestamps[key]:
            # Cache expired
            del self.memory_cache[key]
            del self.memory_cache_timestamps[key]
            return None

        return self.memory_cache[key]

    def _cleanup_memory_cache(self):
        """Remove expired entries from memory cache"""
        current_time = time.time()
        expired_keys = [
            key for key, expire_time in self.memory_cache_timestamps.items()
            if current_time > expire_time
        ]

        for key in expired_keys:
            del self.memory_cache[key]
            del self.memory_cache_timestamps[key]

    def cache_api_response(self, endpoint: str, params: Dict[str, Any],
                          data: Any, ttl: Optional[int] = None) -> str:
        """Cache an API response"""
        cache_key = self._generate_cache_key({"endpoint": endpoint, "params": params})
        cache_path = self._get_cache_path("api_responses", cache_key)
        ttl = ttl or 1800  # 30 minutes for API responses

        try:
            cache_data = {
                "timestamp": time.time(),
                "ttl": ttl,
                "endpoint": endpoint,
                "params": params,
                "data": data
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, default=str, indent=2)

            # Also store in memory cache for fast access
            self.set_memory_cache(cache_key, data, ttl=300)  # 5 min memory cache

            self.logger.debug(f"Cached API response: {endpoint} -> {cache_key}")
            return cache_key

        except Exception as e:
            self.logger.error(f"Failed to cache API response: {e}")
            return ""

    def get_cached_api_response(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Retrieve cached API response"""
        cache_key = self._generate_cache_key({"endpoint": endpoint, "params": params})

        # Try memory cache first
        memory_data = self.get_memory_cache(cache_key)
        if memory_data is not None:
            self.logger.debug(f"Memory cache hit: {endpoint}")
            return memory_data

        cache_path = self._get_cache_path("api_responses", cache_key)

        if not self._is_cache_valid(cache_path, ttl=1800):
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Verify TTL
            if time.time() - cache_data["timestamp"] > cache_data["ttl"]:
                return None

            data = cache_data["data"]

            # Store in memory cache for next time
            self.set_memory_cache(cache_key, data, ttl=300)

            self.logger.debug(f"Disk cache hit: {endpoint}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to retrieve cached API response: {e}")
            return None

    def cache_dataset_sample(self, dataset_id: str, sample_df: pd.DataFrame,
                           sample_size: int, ttl: Optional[int] = None) -> str:
        """Cache a dataset sample"""
        cache_key = self._generate_cache_key({
            "dataset_id": dataset_id,
            "sample_size": sample_size,
            "type": "sample"
        })

        cache_path = self._get_cache_path("datasets", cache_key)
        ttl = ttl or 7200  # 2 hours for dataset samples

        try:
            cache_data = {
                "timestamp": time.time(),
                "ttl": ttl,
                "dataset_id": dataset_id,
                "sample_size": sample_size,
                "shape": sample_df.shape,
                "columns": list(sample_df.columns)
            }

            # Save metadata as JSON
            with open(cache_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, default=str, indent=2)

            # Save DataFrame as pickle for faster loading
            with open(cache_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(sample_df, f)

            self.logger.debug(f"Cached dataset sample: {dataset_id} -> {cache_key}")
            return cache_key

        except Exception as e:
            self.logger.error(f"Failed to cache dataset sample: {e}")
            return ""

    def get_cached_dataset_sample(self, dataset_id: str, sample_size: int) -> Optional[pd.DataFrame]:
        """Retrieve cached dataset sample"""
        cache_key = self._generate_cache_key({
            "dataset_id": dataset_id,
            "sample_size": sample_size,
            "type": "sample"
        })

        cache_path = self._get_cache_path("datasets", cache_key)

        if not self._is_cache_valid(cache_path.with_suffix('.json'), ttl=7200):
            return None

        try:
            # Load metadata
            with open(cache_path.with_suffix('.json'), 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Verify TTL
            if time.time() - cache_data["timestamp"] > cache_data["ttl"]:
                return None

            # Load DataFrame
            with open(cache_path.with_suffix('.pkl'), 'rb') as f:
                sample_df = pickle.load(f)

            self.logger.debug(f"Cache hit for dataset sample: {dataset_id}")
            return sample_df

        except Exception as e:
            self.logger.error(f"Failed to retrieve cached dataset sample: {e}")
            return None

    def cache_dashboard_data(self, data_type: str, data: Any, ttl: Optional[int] = None) -> str:
        """Cache dashboard data (top datasets, categories, etc.)"""
        cache_key = self._generate_cache_key({"type": data_type, "dashboard": True})
        cache_path = self._get_cache_path("dashboard", cache_key)
        ttl = ttl or 900  # 15 minutes for dashboard data

        try:
            cache_data = {
                "timestamp": time.time(),
                "ttl": ttl,
                "data_type": data_type,
                "data": data
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, default=str, indent=2)

            # Store in memory cache for immediate access
            self.set_memory_cache(f"dashboard_{data_type}", data, ttl=300)

            self.logger.debug(f"Cached dashboard data: {data_type} -> {cache_key}")
            return cache_key

        except Exception as e:
            self.logger.error(f"Failed to cache dashboard data: {e}")
            return ""

    def get_cached_dashboard_data(self, data_type: str) -> Optional[Any]:
        """Retrieve cached dashboard data"""
        # Try memory cache first
        memory_data = self.get_memory_cache(f"dashboard_{data_type}")
        if memory_data is not None:
            self.logger.debug(f"Memory cache hit for dashboard: {data_type}")
            return memory_data

        cache_key = self._generate_cache_key({"type": data_type, "dashboard": True})
        cache_path = self._get_cache_path("dashboard", cache_key)

        if not self._is_cache_valid(cache_path, ttl=900):
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Verify TTL
            if time.time() - cache_data["timestamp"] > cache_data["ttl"]:
                return None

            data = cache_data["data"]

            # Store in memory cache
            self.set_memory_cache(f"dashboard_{data_type}", data, ttl=300)

            self.logger.debug(f"Disk cache hit for dashboard: {data_type}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to retrieve cached dashboard data: {e}")
            return None

    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear all or specific cache type"""
        if cache_type:
            cache_path = self.cache_dir / cache_type
            if cache_path.exists():
                for file_path in cache_path.glob("*"):
                    file_path.unlink()
                self.logger.info(f"Cleared {cache_type} cache")
        else:
            for cache_path in self.cache_dir.glob("*/*"):
                if cache_path.is_file():
                    cache_path.unlink()
            self.memory_cache.clear()
            self.memory_cache_timestamps.clear()
            self.logger.info("Cleared all caches")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "cache_types": {}
        }

        for cache_type in ["api_responses", "datasets", "dashboard", "quality"]:
            cache_path = self.cache_dir / cache_type
            if cache_path.exists():
                files = list(cache_path.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                stats["cache_types"][cache_type] = {
                    "file_count": len(files),
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }

        return stats