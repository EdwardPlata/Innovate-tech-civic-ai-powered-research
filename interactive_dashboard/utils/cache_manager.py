"""
Cache Manager for Dashboard Data and Computations
Provides caching functionality for improved performance
"""

import json
import logging
from typing import Any, Dict, Optional
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class DashboardCacheManager:
    """Cache manager for dashboard data and computations."""

    def __init__(self, cache_dir: str = "data/cache", max_age_hours: int = 24):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)

    def _get_cache_key(self, key: str) -> str:
        """Generate a cache key from the input key."""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.cache"

    def _is_expired(self, cache_path: Path) -> bool:
        """Check if a cache entry has expired."""
        if not cache_path.exists():
            return True

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime > self.max_age

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found or expired
        """
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)

        if self._is_expired(cache_path):
            if cache_path.exists():
                cache_path.unlink()  # Remove expired cache file
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_path}: {str(e)}")
            return None

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Error writing cache file {cache_path}: {str(e)}")

    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.

        Args:
            key: Cache key
        """
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            cache_path.unlink()

    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_file}: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())

        expired_count = sum(1 for f in cache_files if self._is_expired(f))

        return {
            'total_entries': len(cache_files),
            'expired_entries': expired_count,
            'valid_entries': len(cache_files) - expired_count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / 1024 / 1024,
            'cache_dir': str(self.cache_dir)
        }

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        removed_count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            if self._is_expired(cache_file):
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Error deleting expired cache file {cache_file}: {str(e)}")

        logger.info(f"Cleaned up {removed_count} expired cache entries")
        return removed_count