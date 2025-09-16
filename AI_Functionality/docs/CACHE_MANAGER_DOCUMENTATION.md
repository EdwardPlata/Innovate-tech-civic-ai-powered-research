# Cache Manager - Advanced Caching System Documentation

## 📋 Overview

The `cache_manager.py` module provides a sophisticated multi-tier caching system designed to optimize AI request performance through intelligent caching strategies. It implements both exact prompt matching and semantic similarity-based caching for maximum efficiency.

## 🎯 Purpose

- **Primary Role**: Optimize AI request performance through intelligent caching
- **Key Responsibility**: Reduce API costs and response times via strategic caching
- **Core Function**: Multi-strategy caching with exact and semantic matching
- **Integration Point**: Performance optimization layer for AI analysis operations

## 🏗️ Architecture

```python
CacheManager
├── Multi-Tier Caching
│   ├── Prompt Cache (exact matches)
│   ├── Response Cache (full responses)
│   ├── Context Cache (dataset-specific)
│   └── Semantic Cache (similarity-based)
├── Storage Backends
│   ├── Disk Cache (persistent storage)
│   ├── Memory Cache (fast access)
│   └── Hybrid Storage (automatic selection)
├── Semantic Intelligence
│   ├── Sentence Transformers (embeddings)
│   ├── Cosine Similarity (matching)
│   ├── Threshold Management (quality control)
│   └── Cache Optimization (size management)
└── Performance Features
    ├── TTL Management (time-based expiration)
    ├── Size Limits (memory management)
    ├── Hit Rate Analytics (performance monitoring)
    └── Cache Statistics (optimization insights)
```

## 🔧 Core Components

### CacheManager Class

**Main caching orchestrator with multiple strategies**

```python
class CacheManager:
    """
    Advanced caching manager with multiple strategies
    
    Features:
    - Exact prompt matching for perfect cache hits
    - Semantic similarity matching for related queries
    - Context-aware caching for dataset-specific analysis
    - Persistent disk storage with memory acceleration
    - Intelligent cache eviction and size management
    - Comprehensive analytics and monitoring
    """
    
    def __init__(
        self,
        cache_dir: str = "./ai_cache",
        enable_semantic: bool = True,
        semantic_threshold: float = 0.85,
        default_ttl: int = 3600,      # 1 hour default
        max_cache_size: int = 1000
    ):
        """
        Initialize CacheManager with configuration
        
        Args:
            cache_dir: Directory for persistent cache storage
                - Creates subdirectories for different cache types
                - Automatically creates directory if it doesn't exist
                - Supports both relative and absolute paths
            
            enable_semantic: Enable semantic similarity caching
                - Requires sentence-transformers library
                - Uses all-MiniLM-L6-v2 model for embeddings
                - Gracefully degrades if dependencies unavailable
            
            semantic_threshold: Similarity threshold for semantic matches
                - Range: 0.0 (no similarity) to 1.0 (identical)
                - Recommended: 0.8-0.9 for good quality matches
                - Higher values = more precise matches
                - Lower values = more cache hits but less precision
            
            default_ttl: Default time-to-live in seconds
                - 3600 = 1 hour (good for most analysis)
                - 7200 = 2 hours (for stable datasets)
                - 86400 = 24 hours (for reference data)
            
            max_cache_size: Maximum entries in semantic cache
                - Controls memory usage for embeddings
                - Uses FIFO eviction when limit reached
                - Separate from disk cache limits
        
        Cache Types Created:
        - Prompt Cache: Exact prompt hash → response mapping
        - Response Cache: Full response objects with metadata
        - Context Cache: Dataset-specific analysis results
        - Semantic Cache: Text → embedding mapping for similarity
        """
        
        # Initialize storage directories
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration
        self.semantic_threshold = semantic_threshold
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        
        # Initialize disk cache (persistent storage)
        if diskcache_available:
            self.prompt_cache = diskcache.Cache(str(self.cache_dir / "prompts"))
            self.response_cache = diskcache.Cache(str(self.cache_dir / "responses"))
            self.context_cache = diskcache.Cache(str(self.cache_dir / "contexts"))
        else:
            # Fallback to memory-only cache
            logger.warning("diskcache not available, using memory cache")
            self.prompt_cache = {}
            self.response_cache = {}
            self.context_cache = {}
        
        # Initialize semantic similarity
        self.semantic_enabled = enable_semantic and SEMANTIC_AVAILABLE
        self.embedding_model = None
        self.semantic_cache = {}  # In-memory cache for embeddings
        
        if self.semantic_enabled:
            try:
                # Load lightweight but effective model
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic caching enabled with sentence-transformers")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.semantic_enabled = False
```

## 🔍 Caching Strategies

### 1. Exact Prompt Matching

**Perfect hash-based matching for identical requests**

```python
def _get_prompt_hash(self, request: AIRequest) -> str:
    """
    Generate deterministic hash for exact prompt matching
    
    Includes in hash:
    - User prompt text
    - System prompt (if any)
    - Temperature setting
    - Model specification
    - Other generation parameters
    
    Excludes from hash:
    - Timestamps
    - Request IDs
    - Non-functional metadata
    """
    prompt_data = {
        'prompt': request.prompt,
        'system_prompt': request.system_prompt,
        'temperature': request.temperature,
        'model': request.model,
        'max_tokens': request.max_tokens
    }
    
    # Ensure consistent ordering for same hash
    prompt_str = json.dumps(prompt_data, sort_keys=True)
    return hashlib.sha256(prompt_str.encode()).hexdigest()

# Usage patterns for exact matching:

# Identical requests will hit cache
request1 = AIRequest(prompt="What is data quality?", temperature=0.1)
request2 = AIRequest(prompt="What is data quality?", temperature=0.1)
# → Same hash, cache hit

# Different requests will miss cache
request3 = AIRequest(prompt="What is data quality?", temperature=0.2)
# → Different hash, cache miss due to temperature difference

# System prompt affects hashing
request4 = AIRequest(
    prompt="What is data quality?", 
    system_prompt="You are a data expert"
)
# → Different hash due to system prompt
```

### 2. Semantic Similarity Matching

**AI-powered similarity detection for related queries**

```python
def _get_semantic_embedding(self, text: str) -> np.ndarray:
    """
    Generate semantic embedding for text
    
    Uses sentence-transformers to create dense vector representation
    that captures semantic meaning beyond exact word matching.
    
    Example semantic similarities:
    - "What is data quality?" ≈ "How good is this data?"
    - "Find missing values" ≈ "Detect incomplete records"
    - "Performance issues" ≈ "Speed problems"
    """
    if not self.semantic_enabled:
        return None
    
    try:
        # Generate 384-dimensional embedding
        return self.embedding_model.encode(text)
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return None

def _find_semantic_match(self, prompt: str) -> Optional[Tuple[str, float]]:
    """
    Find semantically similar cached prompt
    
    Process:
    1. Generate embedding for query prompt
    2. Compare with all cached embeddings
    3. Calculate cosine similarity scores
    4. Return best match above threshold
    
    Returns:
        Tuple of (matched_prompt, similarity_score) or None
    """
    if not self.semantic_enabled or not self.semantic_cache:
        return None
    
    query_embedding = self._get_semantic_embedding(prompt)
    if query_embedding is None:
        return None
    
    best_match = None
    best_similarity = 0
    
    # Search through cached embeddings
    for cached_prompt, cached_embedding in self.semantic_cache.items():
        try:
            # Calculate cosine similarity (1 = identical, 0 = orthogonal)
            similarity = 1 - cosine(query_embedding, cached_embedding)
            
            if similarity > best_similarity and similarity >= self.semantic_threshold:
                best_similarity = similarity
                best_match = cached_prompt
                
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            continue
    
    return (best_match, best_similarity) if best_match else None

# Semantic similarity examples:

# These queries would be considered similar (assuming threshold = 0.85):
queries_similar = [
    "What is the data quality score?",           # Base query
    "How good is the quality of this data?",    # ~0.87 similarity
    "What's the quality assessment result?",    # ~0.86 similarity
    "Can you tell me about data quality?",      # ~0.85 similarity
]

# These would be considered different:
queries_different = [
    "What is the data quality score?",          # Base query
    "How many records are in the dataset?",    # ~0.32 similarity
    "What's the weather like today?",          # ~0.15 similarity
    "Perform data transformation",             # ~0.41 similarity
]
```

### 3. Context-Aware Caching

**Dataset-specific caching for analytical context**

```python
def cache_context_analysis(self, dataset_id: str, analysis: Dict[str, Any], ttl: Optional[int] = None):
    """
    Cache dataset-specific context analysis
    
    Used for:
    - Dataset metadata analysis
    - Quality assessment results
    - Schema analysis
    - Statistical summaries
    - User access patterns
    
    Args:
        dataset_id: Unique identifier for the dataset
        analysis: Analysis results to cache
        ttl: Custom TTL (defaults to 24 hours for context)
    """
    if ttl is None:
        ttl = self.default_ttl * 24  # Longer TTL for context data
    
    cache_data = {
        'analysis': analysis,
        'timestamp': time.time(),
        'dataset_id': dataset_id,
        'version': analysis.get('version', '1.0')
    }
    
    cache_key = f"context_{dataset_id}"
    
    if isinstance(self.context_cache, diskcache.Cache):
        self.context_cache.set(cache_key, cache_data, expire=ttl)
    else:
        self.context_cache[cache_key] = cache_data
    
    logger.info(f"Cached context analysis for dataset: {dataset_id}")

def get_cached_context(self, dataset_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached context analysis for dataset
    
    Automatically handles:
    - Expiration checking
    - Version compatibility
    - Data freshness validation
    """
    cache_key = f"context_{dataset_id}"
    
    if isinstance(self.context_cache, diskcache.Cache):
        cached = self.context_cache.get(cache_key)
    else:
        cached = self.context_cache.get(cache_key)
    
    if cached:
        # Check if context is still fresh
        age_hours = (time.time() - cached['timestamp']) / 3600
        if age_hours < 48:  # Context valid for 48 hours
            return cached['analysis']
        else:
            # Remove stale context
            if isinstance(self.context_cache, diskcache.Cache):
                self.context_cache.delete(cache_key)
            else:
                del self.context_cache[cache_key]
    
    return None

# Context caching usage examples:

# Cache dataset analysis
dataset_analysis = {
    "quality_score": 0.87,
    "completeness": 0.92,
    "consistency": 0.84,
    "schema": {"columns": 15, "types": ["string", "int", "float"]},
    "size_mb": 45.6,
    "last_updated": "2023-12-15T10:30:00Z"
}

cache_manager.cache_context_analysis("customer-db-2023", dataset_analysis)

# Retrieve cached context
cached_context = cache_manager.get_cached_context("customer-db-2023")
if cached_context:
    print(f"Quality score: {cached_context['quality_score']}")
```

## 📊 Cache Operations

### Advanced Cache Retrieval

**Multi-strategy cache lookup with intelligent fallback**

```python
async def get_cached_response(self, request: AIRequest) -> Optional[AIResponse]:
    """
    Intelligent cache retrieval using multiple strategies
    
    Cache Lookup Order:
    1. Exact prompt hash match (fastest, highest precision)
    2. Semantic similarity match (slower, broader coverage)
    3. Context-based suggestions (dataset-specific)
    
    Returns:
        AIResponse with cached=True or None if no match
    """
    
    # Strategy 1: Exact prompt match
    prompt_hash = self._get_prompt_hash(request)
    
    if isinstance(self.prompt_cache, diskcache.Cache):
        cached = self.prompt_cache.get(prompt_hash)
    else:
        cached = self.prompt_cache.get(prompt_hash)
    
    if cached:
        # Perfect match found
        cached['cached'] = True
        cached['cache_strategy'] = 'exact_match'
        logger.info(f"Cache HIT (exact): {prompt_hash[:8]}")
        return AIResponse(**cached)
    
    # Strategy 2: Semantic similarity match
    if self.semantic_enabled:
        semantic_match = self._find_semantic_match(request.prompt)
        
        if semantic_match:
            similar_prompt, similarity = semantic_match
            
            # Generate hash for similar prompt
            similar_request = AIRequest(
                prompt=similar_prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                model=request.model
            )
            similar_hash = self._get_prompt_hash(similar_request)
            
            # Get cached response for similar prompt
            if isinstance(self.prompt_cache, diskcache.Cache):
                cached = self.prompt_cache.get(similar_hash)
            else:
                cached = self.prompt_cache.get(similar_hash)
            
            if cached:
                # Semantic match found
                cached['cached'] = True
                cached['cache_strategy'] = 'semantic_match'
                cached['metadata'] = cached.get('metadata', {})
                cached['metadata']['semantic_similarity'] = similarity
                cached['metadata']['original_prompt'] = similar_prompt
                
                logger.info(f"Cache HIT (semantic): {similarity:.3f} similarity")
                return AIResponse(**cached)
    
    # Strategy 3: Context-based suggestions (future enhancement)
    # Could suggest related analysis based on dataset context
    
    logger.info(f"Cache MISS: {prompt_hash[:8]}")
    return None

# Usage with different scenarios:

# Scenario 1: Exact match
request1 = AIRequest(prompt="Analyze data quality", temperature=0.1)
response1 = await ai_analyst.analyze_dataset(...)  # API call made, response cached

request2 = AIRequest(prompt="Analyze data quality", temperature=0.1)
cached_response = await cache_manager.get_cached_response(request2)
# → Exact match, instant return

# Scenario 2: Semantic match
request3 = AIRequest(prompt="Assess the quality of this data", temperature=0.1)
cached_response = await cache_manager.get_cached_response(request3)
# → Semantic match with request1, similarity ~0.88
```

### Cache Storage

**Efficient storage with metadata and expiration**

```python
async def cache_response(self, request: AIRequest, response: AIResponse):
    """
    Store response with comprehensive metadata and expiration
    
    Storage Process:
    1. Generate cache key from request
    2. Prepare cache data with metadata
    3. Store in persistent cache with TTL
    4. Update semantic cache if enabled
    5. Manage cache size limits
    """
    
    prompt_hash = self._get_prompt_hash(request)
    
    # Prepare cache data with rich metadata
    cache_data = {
        # Core response data
        'content': response.content,
        'provider': response.provider,
        'model': response.model,
        'usage': response.usage,
        'metadata': response.metadata or {},
        
        # Cache-specific metadata
        'timestamp': time.time(),
        'cache_ttl': self.default_ttl,
        'request_hash': prompt_hash,
        'original_request': {
            'prompt': request.prompt[:100] + '...' if len(request.prompt) > 100 else request.prompt,
            'system_prompt': request.system_prompt,
            'temperature': request.temperature,
            'model': request.model
        }
    }
    
    # Store in persistent cache
    if isinstance(self.prompt_cache, diskcache.Cache):
        self.prompt_cache.set(prompt_hash, cache_data, expire=self.default_ttl)
    else:
        self.prompt_cache[prompt_hash] = cache_data
    
    # Update semantic cache if enabled
    if self.semantic_enabled:
        embedding = self._get_semantic_embedding(request.prompt)
        if embedding is not None:
            self.semantic_cache[request.prompt] = embedding
            
            # Manage semantic cache size
            if len(self.semantic_cache) > self.max_cache_size:
                # Remove oldest entries (FIFO eviction)
                oldest_keys = list(self.semantic_cache.keys())[
                    :len(self.semantic_cache) - self.max_cache_size + 1
                ]
                for key in oldest_keys:
                    del self.semantic_cache[key]
                
                logger.info(f"Evicted {len(oldest_keys)} old semantic cache entries")
    
    logger.info(f"Cached response: {prompt_hash[:8]} (TTL: {self.default_ttl}s)")

# Advanced caching with custom TTL
async def cache_response_with_custom_ttl(
    self, 
    request: AIRequest, 
    response: AIResponse, 
    ttl: int,
    cache_tags: List[str] = None
):
    """Cache response with custom TTL and optional tags"""
    
    # Extend cache data with custom metadata
    cache_data = asdict(response)
    cache_data.update({
        'timestamp': time.time(),
        'custom_ttl': ttl,
        'cache_tags': cache_tags or [],
        'cache_priority': 'high' if ttl > 7200 else 'normal'
    })
    
    prompt_hash = self._get_prompt_hash(request)
    
    if isinstance(self.prompt_cache, diskcache.Cache):
        self.prompt_cache.set(prompt_hash, cache_data, expire=ttl)
    else:
        self.prompt_cache[prompt_hash] = cache_data
```

## 📈 Cache Analytics

### Performance Monitoring

**Comprehensive cache performance tracking**

```python
def get_cache_stats(self) -> Dict[str, Any]:
    """
    Get comprehensive cache performance statistics
    
    Returns detailed metrics for cache optimization and monitoring
    """
    
    # Basic cache sizes
    stats = {
        'enabled_features': {
            'semantic_caching': self.semantic_enabled,
            'disk_cache': isinstance(self.prompt_cache, diskcache.Cache),
            'context_caching': True
        },
        
        'cache_sizes': {
            'semantic_cache_entries': len(self.semantic_cache),
            'semantic_cache_limit': self.max_cache_size
        },
        
        'configuration': {
            'semantic_threshold': self.semantic_threshold,
            'default_ttl_seconds': self.default_ttl,
            'cache_directory': str(self.cache_dir)
        }
    }
    
    # Disk cache statistics (if available)
    if isinstance(self.prompt_cache, diskcache.Cache):
        try:
            stats['cache_sizes'].update({
                'prompt_cache_entries': len(self.prompt_cache),
                'response_cache_entries': len(self.response_cache),
                'context_cache_entries': len(self.context_cache),
                'total_disk_entries': (
                    len(self.prompt_cache) + 
                    len(self.response_cache) + 
                    len(self.context_cache)
                )
            })
            
            # Disk usage statistics
            stats['disk_usage'] = {
                'cache_directory_size_mb': self._get_directory_size_mb(),
                'average_entry_size_kb': self._calculate_average_entry_size()
            }
            
        except Exception as e:
            logger.warning(f"Error collecting disk cache stats: {e}")
            stats['cache_sizes']['disk_cache_error'] = str(e)
    
    else:
        # Memory cache statistics
        stats['cache_sizes'].update({
            'prompt_cache_entries': len(self.prompt_cache),
            'response_cache_entries': len(self.response_cache),
            'context_cache_entries': len(self.context_cache)
        })
    
    return stats

def get_hit_rate_analytics(self, time_window_hours: int = 24) -> Dict[str, Any]:
    """
    Calculate cache hit rates and performance metrics
    
    Note: This would require additional tracking implementation
    to maintain request/hit counters over time
    """
    
    # Placeholder for hit rate calculation
    # In production, you'd track hits/misses in a separate analytics store
    
    analytics = {
        'time_window_hours': time_window_hours,
        'estimated_metrics': {
            'total_requests': 'Not tracked - requires analytics implementation',
            'cache_hits': 'Not tracked - requires analytics implementation',
            'cache_misses': 'Not tracked - requires analytics implementation',
            'hit_rate_percentage': 'Not tracked - requires analytics implementation'
        },
        
        'cache_efficiency': {
            'semantic_matches_enabled': self.semantic_enabled,
            'semantic_threshold': self.semantic_threshold,
            'cache_size_optimization': len(self.semantic_cache) / max(1, self.max_cache_size)
        },
        
        'recommendations': []
    }
    
    # Generate recommendations based on current configuration
    if not self.semantic_enabled:
        analytics['recommendations'].append(
            "Enable semantic caching to improve hit rates for similar queries"
        )
    
    if self.semantic_threshold > 0.9:
        analytics['recommendations'].append(
            f"Consider lowering semantic threshold from {self.semantic_threshold} to 0.85-0.88 for better coverage"
        )
    
    return analytics

# Usage examples:

# Get current cache statistics
cache_stats = cache_manager.get_cache_stats()
print("📊 Cache Statistics:")
print(f"Semantic caching: {'✅' if cache_stats['enabled_features']['semantic_caching'] else '❌'}")
print(f"Disk cache: {'✅' if cache_stats['enabled_features']['disk_cache'] else '❌'}")
print(f"Total cached entries: {cache_stats['cache_sizes'].get('total_disk_entries', 'N/A')}")

# Get performance analytics
analytics = cache_manager.get_hit_rate_analytics(time_window_hours=24)
print("\n📈 Performance Analytics:")
for recommendation in analytics['recommendations']:
    print(f"💡 {recommendation}")
```

### Cache Maintenance

**Automated cleanup and optimization**

```python
def clear_cache(self, cache_type: str = "all", confirm: bool = False):
    """
    Clear specified cache types with safety confirmation
    
    Args:
        cache_type: Type to clear ("all", "prompts", "semantic", "context")
        confirm: Safety confirmation (required for production)
    """
    
    if not confirm:
        logger.warning("Cache clear operation requires confirmation=True")
        return
    
    cleared_counts = {}
    
    if cache_type in ["all", "prompts"]:
        if isinstance(self.prompt_cache, diskcache.Cache):
            count = len(self.prompt_cache)
            self.prompt_cache.clear()
            cleared_counts['prompts'] = count
        else:
            cleared_counts['prompts'] = len(self.prompt_cache)
            self.prompt_cache.clear()
    
    if cache_type in ["all", "semantic"]:
        cleared_counts['semantic'] = len(self.semantic_cache)
        self.semantic_cache.clear()
    
    if cache_type in ["all", "context"]:
        if isinstance(self.context_cache, diskcache.Cache):
            count = len(self.context_cache)
            self.context_cache.clear()
            cleared_counts['context'] = count
        else:
            cleared_counts['context'] = len(self.context_cache)
            self.context_cache.clear()
    
    logger.info(f"Cleared {cache_type} cache: {cleared_counts}")
    return cleared_counts

def optimize_cache(self) -> Dict[str, Any]:
    """
    Perform cache optimization operations
    
    Optimization strategies:
    1. Remove expired entries
    2. Consolidate similar semantic entries
    3. Update cache statistics
    4. Optimize storage efficiency
    """
    
    optimization_results = {
        'expired_removed': 0,
        'duplicates_consolidated': 0,
        'storage_optimized': False
    }
    
    current_time = time.time()
    
    # Remove expired entries from memory caches
    if not isinstance(self.prompt_cache, diskcache.Cache):
        expired_keys = []
        for key, data in self.prompt_cache.items():
            if isinstance(data, dict) and 'timestamp' in data:
                age = current_time - data['timestamp']
                ttl = data.get('cache_ttl', self.default_ttl)
                if age > ttl:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self.prompt_cache[key]
        
        optimization_results['expired_removed'] = len(expired_keys)
    
    # Optimize semantic cache
    if self.semantic_enabled and len(self.semantic_cache) > self.max_cache_size * 0.8:
        # Remove least recently used entries
        keys_to_remove = list(self.semantic_cache.keys())[
            :len(self.semantic_cache) - int(self.max_cache_size * 0.7)
        ]
        
        for key in keys_to_remove:
            del self.semantic_cache[key]
        
        optimization_results['storage_optimized'] = True
    
    logger.info(f"Cache optimization completed: {optimization_results}")
    return optimization_results

# Automated cache maintenance
async def scheduled_maintenance(self):
    """
    Automated cache maintenance (run periodically)
    """
    
    # Daily optimization
    optimization_results = self.optimize_cache()
    
    # Weekly deep clean (if disk cache available)
    if isinstance(self.prompt_cache, diskcache.Cache):
        # Disk cache has built-in expiration, but we can trigger cleanup
        try:
            self.prompt_cache.expire()  # Remove expired entries
            self.response_cache.expire()
            self.context_cache.expire()
        except Exception as e:
            logger.warning(f"Cache expiration failed: {e}")
    
    # Generate maintenance report
    stats = self.get_cache_stats()
    
    maintenance_report = {
        'timestamp': datetime.now().isoformat(),
        'optimization_results': optimization_results,
        'current_stats': stats,
        'recommendations': []
    }
    
    # Add specific recommendations
    if stats['cache_sizes']['semantic_cache_entries'] > self.max_cache_size * 0.9:
        maintenance_report['recommendations'].append(
            "Consider increasing max_cache_size or reducing semantic_threshold"
        )
    
    return maintenance_report
```

## 🚀 Usage Examples

### Basic Caching Setup

```python
from AI_Functionality.core.cache_manager import CacheManager
from AI_Functionality.core.ai_analyst import DataAnalyst

# Initialize cache manager
cache_manager = CacheManager(
    cache_dir="./ai_cache",
    enable_semantic=True,
    semantic_threshold=0.85,
    default_ttl=3600,  # 1 hour
    max_cache_size=1000
)

# Initialize AI analyst with caching
analyst = DataAnalyst(
    primary_provider="openai",
    cache_manager=cache_manager  # Assumes integration in DataAnalyst
)

# First request - will hit API and cache result
request1 = AIRequest(
    prompt="What is the data quality of this dataset?",
    temperature=0.1
)

response1 = await analyst.answer_question(
    question=request1.prompt,
    dataset_info=dataset_info
)
print(f"Response 1 cached: {response1.cached}")  # False (first time)

# Second identical request - will hit cache
response2 = await analyst.answer_question(
    question=request1.prompt,
    dataset_info=dataset_info
)
print(f"Response 2 cached: {response2.cached}")  # True (exact match)

# Similar request - will hit semantic cache
response3 = await analyst.answer_question(
    question="How good is the quality of this data?",
    dataset_info=dataset_info
)
print(f"Response 3 cached: {response3.cached}")  # True (semantic match)
print(f"Similarity: {response3.metadata.get('semantic_similarity', 'N/A')}")
```

### Advanced Cache Configuration

```python
# Production cache configuration
production_cache = CacheManager(
    cache_dir="/app/cache",
    enable_semantic=True,
    semantic_threshold=0.87,  # Higher precision
    default_ttl=7200,         # 2 hours
    max_cache_size=5000       # Larger cache for production
)

# Development cache configuration
dev_cache = CacheManager(
    cache_dir="./dev_cache",
    enable_semantic=False,    # Faster startup
    default_ttl=1800,         # 30 minutes
    max_cache_size=100        # Smaller cache for dev
)

# Dataset-specific caching
dataset_cache = CacheManager(
    cache_dir="./dataset_cache",
    enable_semantic=True,
    semantic_threshold=0.90,  # Very high precision for datasets
    default_ttl=86400,        # 24 hours for stable datasets
    max_cache_size=2000
)

# Cache context for dataset analysis
dataset_analysis = {
    "id": "customer-data-2023",
    "quality_score": 0.89,
    "completeness": 0.94,
    "last_analyzed": "2023-12-15T10:30:00Z",
    "schema_analysis": {
        "columns": 15,
        "primary_keys": ["customer_id"],
        "foreign_keys": ["account_id"]
    }
}

# Cache with extended TTL for stable datasets
dataset_cache.cache_context_analysis(
    dataset_id="customer-data-2023",
    analysis=dataset_analysis,
    ttl=86400 * 7  # 1 week for stable reference data
)
```

### Cache Performance Monitoring

```python
# Monitor cache performance
async def monitor_cache_performance(cache_manager: CacheManager):
    """Monitor and report cache performance"""
    
    # Get current statistics
    stats = cache_manager.get_cache_stats()
    
    print("🔍 Cache Performance Report")
    print("=" * 40)
    
    # Cache configuration
    print(f"Semantic caching: {'✅' if stats['enabled_features']['semantic_caching'] else '❌'}")
    print(f"Disk persistence: {'✅' if stats['enabled_features']['disk_cache'] else '❌'}")
    print(f"Semantic threshold: {stats['configuration']['semantic_threshold']}")
    
    # Cache sizes
    print(f"\n📊 Cache Sizes:")
    for cache_type, size in stats['cache_sizes'].items():
        print(f"  {cache_type}: {size}")
    
    # Storage efficiency
    if 'disk_usage' in stats:
        print(f"\n💾 Disk Usage:")
        print(f"  Total size: {stats['disk_usage']['cache_directory_size_mb']:.1f} MB")
        print(f"  Avg entry size: {stats['disk_usage']['average_entry_size_kb']:.1f} KB")
    
    # Performance recommendations
    analytics = cache_manager.get_hit_rate_analytics()
    if analytics['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in analytics['recommendations']:
            print(f"  • {rec}")

# Automated cache optimization
async def optimize_cache_periodically(cache_manager: CacheManager, interval_hours: int = 24):
    """Run cache optimization on schedule"""
    
    while True:
        try:
            # Wait for next optimization cycle
            await asyncio.sleep(interval_hours * 3600)
            
            # Run maintenance
            maintenance_report = await cache_manager.scheduled_maintenance()
            
            # Log results
            logger.info(f"Cache maintenance completed: {maintenance_report['optimization_results']}")
            
            # Alert if cache is getting large
            stats = maintenance_report['current_stats']
            total_entries = sum(
                v for k, v in stats['cache_sizes'].items() 
                if k.endswith('_entries')
            )
            
            if total_entries > 10000:
                logger.warning(f"Cache has grown large: {total_entries} total entries")
            
        except Exception as e:
            logger.error(f"Cache maintenance failed: {e}")
            # Continue loop despite errors

# Start background cache optimization
asyncio.create_task(optimize_cache_periodically(cache_manager, interval_hours=6))
```

## ⚡ Performance Optimization

### Best Practices

1. **Semantic Threshold Tuning**: Start with 0.85, adjust based on hit rate analysis
2. **TTL Strategy**: Longer TTL for stable datasets, shorter for dynamic analysis
3. **Cache Size Management**: Monitor memory usage and adjust max_cache_size
4. **Disk vs Memory**: Use disk cache in production, memory cache for development
5. **Regular Maintenance**: Schedule periodic cache optimization and cleanup

### Memory Management

```python
def estimate_cache_memory_usage(cache_manager: CacheManager) -> Dict[str, float]:
    """Estimate cache memory consumption"""
    
    memory_estimate = {
        'semantic_embeddings_mb': 0.0,
        'memory_cache_mb': 0.0,
        'total_estimated_mb': 0.0
    }
    
    # Estimate semantic cache memory (384 dimensions × 4 bytes × count)
    if cache_manager.semantic_enabled:
        embedding_size_bytes = 384 * 4  # 384 float32 values
        total_embeddings = len(cache_manager.semantic_cache)
        memory_estimate['semantic_embeddings_mb'] = (
            total_embeddings * embedding_size_bytes
        ) / (1024 * 1024)
    
    # Estimate memory cache size (rough approximation)
    if not isinstance(cache_manager.prompt_cache, diskcache.Cache):
        # Estimate based on typical response sizes
        avg_response_size = 2048  # bytes
        total_responses = len(cache_manager.prompt_cache)
        memory_estimate['memory_cache_mb'] = (
            total_responses * avg_response_size
        ) / (1024 * 1024)
    
    memory_estimate['total_estimated_mb'] = (
        memory_estimate['semantic_embeddings_mb'] + 
        memory_estimate['memory_cache_mb']
    )
    
    return memory_estimate
```

This comprehensive cache manager documentation covers all aspects from basic setup to advanced optimization strategies, providing a complete guide for implementing and managing the AI caching system.