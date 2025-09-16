"""
Advanced Caching Manager for AI Functionality

Implements multiple caching strategies:
1. Prompt Caching - Exact prompt matches
2. Semantic Caching - Similar meaning detection
3. Result Caching - Response caching with TTL
4. Context Caching - Dataset-specific insights
"""

import hashlib
import json
import time
import pickle
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import logging
from dataclasses import asdict

try:
    import diskcache as dc
except ImportError:
    dc = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from scipy.spatial.distance import cosine
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    # Create dummy np for type hints when not available
    class DummyNumpy:
        ndarray = type(None)
    np = DummyNumpy()

from .base_provider import AIRequest, AIResponse


logger = logging.getLogger(__name__)


class CacheManager:
    """Advanced caching manager with multiple strategies"""

    def __init__(
        self,
        cache_dir: str = "./ai_cache",
        enable_semantic: bool = True,
        semantic_threshold: float = 0.85,
        default_ttl: int = 3600,  # 1 hour
        max_cache_size: int = 1000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.semantic_threshold = semantic_threshold
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size

        # Initialize disk cache
        if dc:
            self.prompt_cache = dc.Cache(str(self.cache_dir / "prompts"))
            self.response_cache = dc.Cache(str(self.cache_dir / "responses"))
            self.context_cache = dc.Cache(str(self.cache_dir / "contexts"))
        else:
            logger.warning("diskcache not available, using memory cache")
            self.prompt_cache = {}
            self.response_cache = {}
            self.context_cache = {}

        # Initialize semantic similarity if available
        self.semantic_enabled = enable_semantic and SEMANTIC_AVAILABLE
        self.embedding_model = None
        self.semantic_cache = {}

        if self.semantic_enabled:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic caching enabled with sentence-transformers")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.semantic_enabled = False

    def _get_prompt_hash(self, request: AIRequest) -> str:
        """Generate hash for exact prompt matching"""
        prompt_data = {
            'prompt': request.prompt,
            'system_prompt': request.system_prompt,
            'temperature': request.temperature,
            'model': request.model
        }
        prompt_str = json.dumps(prompt_data, sort_keys=True)
        return hashlib.sha256(prompt_str.encode()).hexdigest()

    def _get_semantic_embedding(self, text: str) -> np.ndarray:
        """Get semantic embedding for text"""
        if not self.semantic_enabled:
            return None

        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None

    def _find_semantic_match(self, prompt: str) -> Optional[Tuple[str, float]]:
        """Find semantically similar cached prompt"""
        if not self.semantic_enabled or not self.semantic_cache:
            return None

        query_embedding = self._get_semantic_embedding(prompt)
        if query_embedding is None:
            return None

        best_match = None
        best_similarity = 0

        for cached_prompt, cached_embedding in self.semantic_cache.items():
            try:
                similarity = 1 - cosine(query_embedding, cached_embedding)
                if similarity > best_similarity and similarity >= self.semantic_threshold:
                    best_similarity = similarity
                    best_match = cached_prompt
            except Exception as e:
                logger.error(f"Error calculating similarity: {e}")
                continue

        return (best_match, best_similarity) if best_match else None

    async def get_cached_response(self, request: AIRequest) -> Optional[AIResponse]:
        """Get cached response using multiple strategies"""

        # 1. Try exact prompt match first
        prompt_hash = self._get_prompt_hash(request)

        if dc and isinstance(self.prompt_cache, dc.Cache):
            cached = self.prompt_cache.get(prompt_hash)
        else:
            cached = self.prompt_cache.get(prompt_hash)

        if cached:
            cached['cached'] = True
            logger.info(f"Cache HIT (exact): {prompt_hash[:8]}")
            return AIResponse(**cached)

        # 2. Try semantic similarity match
        if self.semantic_enabled:
            semantic_match = self._find_semantic_match(request.prompt)
            if semantic_match:
                similar_prompt, similarity = semantic_match
                similar_hash = self._get_prompt_hash(AIRequest(
                    prompt=similar_prompt,
                    system_prompt=request.system_prompt,
                    temperature=request.temperature,
                    model=request.model
                ))

                if dc and isinstance(self.prompt_cache, dc.Cache):
                    cached = self.prompt_cache.get(similar_hash)
                else:
                    cached = self.prompt_cache.get(similar_hash)

                if cached:
                    cached['cached'] = True
                    cached['metadata'] = cached.get('metadata', {})
                    cached['metadata']['semantic_similarity'] = similarity
                    logger.info(f"Cache HIT (semantic): {similarity:.3f} similarity")
                    return AIResponse(**cached)

        logger.info(f"Cache MISS: {prompt_hash[:8]}")
        return None

    async def cache_response(self, request: AIRequest, response: AIResponse):
        """Cache response with multiple strategies"""
        prompt_hash = self._get_prompt_hash(request)

        # Cache the exact response
        cache_data = asdict(response)
        cache_data['timestamp'] = time.time()

        if dc and isinstance(self.prompt_cache, dc.Cache):
            self.prompt_cache.set(prompt_hash, cache_data, expire=self.default_ttl)
        else:
            self.prompt_cache[prompt_hash] = cache_data

        # Cache semantic embedding if enabled
        if self.semantic_enabled:
            embedding = self._get_semantic_embedding(request.prompt)
            if embedding is not None:
                self.semantic_cache[request.prompt] = embedding

                # Limit semantic cache size
                if len(self.semantic_cache) > self.max_cache_size:
                    # Remove oldest entries (simple FIFO)
                    oldest_keys = list(self.semantic_cache.keys())[:len(self.semantic_cache) - self.max_cache_size + 1]
                    for key in oldest_keys:
                        del self.semantic_cache[key]

        logger.info(f"Cached response: {prompt_hash[:8]}")

    def cache_context_analysis(self, dataset_id: str, analysis: Dict[str, Any], ttl: Optional[int] = None):
        """Cache dataset-specific context analysis"""
        if ttl is None:
            ttl = self.default_ttl * 24  # Longer TTL for context

        cache_data = {
            'analysis': analysis,
            'timestamp': time.time()
        }

        if dc and isinstance(self.context_cache, dc.Cache):
            self.context_cache.set(f"context_{dataset_id}", cache_data, expire=ttl)
        else:
            self.context_cache[f"context_{dataset_id}"] = cache_data

    def get_cached_context(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get cached context analysis for dataset"""
        if dc and isinstance(self.context_cache, dc.Cache):
            cached = self.context_cache.get(f"context_{dataset_id}")
        else:
            cached = self.context_cache.get(f"context_{dataset_id}")

        if cached:
            return cached['analysis']
        return None

    def clear_cache(self, cache_type: str = "all"):
        """Clear specified cache type"""
        if cache_type in ["all", "prompts"]:
            if dc and isinstance(self.prompt_cache, dc.Cache):
                self.prompt_cache.clear()
            else:
                self.prompt_cache.clear()

        if cache_type in ["all", "semantic"]:
            self.semantic_cache.clear()

        if cache_type in ["all", "context"]:
            if dc and isinstance(self.context_cache, dc.Cache):
                self.context_cache.clear()
            else:
                self.context_cache.clear()

        logger.info(f"Cleared {cache_type} cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'semantic_enabled': self.semantic_enabled,
            'semantic_threshold': self.semantic_threshold,
            'semantic_cache_size': len(self.semantic_cache)
        }

        if dc and isinstance(self.prompt_cache, dc.Cache):
            stats.update({
                'prompt_cache_size': len(self.prompt_cache),
                'context_cache_size': len(self.context_cache),
                'disk_cache_enabled': True
            })
        else:
            stats.update({
                'prompt_cache_size': len(self.prompt_cache),
                'context_cache_size': len(self.context_cache),
                'disk_cache_enabled': False
            })

        return stats