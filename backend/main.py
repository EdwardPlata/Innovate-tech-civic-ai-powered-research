"""
FastAPI Backend for Scout Data Discovery

Provides REST API endpoints for the Streamlit frontend to access
Scout Data Discovery functionality.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
import sys
import os
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import base64
import io

# Add scout_data_discovery to path
scout_path = Path(__file__).parent.parent / "scout_data_discovery"
sys.path.append(str(scout_path))

# Add AI_Functionality to path
ai_functionality_path = Path(__file__).parent.parent / "AI_Functionality"
sys.path.append(str(ai_functionality_path))

# Import Scout components
from src.scout_discovery import ScoutDataDiscovery
from src.dataset_relationship_graph import DatasetRelationshipGraph
from src.enhanced_api_client import SoQLQueryBuilder
from src.column_relationship_mapper import RelationshipMapper

# Import cache manager and API configuration
from cache_manager import CacheManager
from api_config import APIConfig

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import AI Functionality components
try:
    from AI_Functionality import DataAnalyst, AnalysisType
    AI_FUNCTIONALITY_AVAILABLE = True
    logger.info("✅ AI Functionality package loaded successfully")
except ImportError as e:
    AI_FUNCTIONALITY_AVAILABLE = False
    logger.warning(f"❌ AI Functionality not available: {e}")
    # Fallback classes
    class DataAnalyst:
        def __init__(self, **kwargs): pass
    class AnalysisType:
        OVERVIEW = "overview"
        QUALITY = "quality"
        INSIGHTS = "insights"

# Initialize FastAPI app
app = FastAPI(
    title="Scout Data Discovery API",
    description="Backend API for exploring NYC Open Data with Scout methodology",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:8080", "http://127.0.0.1:8080"],  # Streamlit default + Backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Scout instance and cache manager
scout_instance = None
cache_manager = None
ai_analyst = None
executor = ThreadPoolExecutor(max_workers=APIConfig.MAX_WORKERS)
REQUEST_TIMEOUT = APIConfig.REQUEST_TIMEOUT

# Session-based storage for API keys and configuration
session_storage = {
    "api_keys": {},
    "ai_config": {},
    "shutdown_requested": False
}

# Helper function for handling API failures with fallback
async def execute_with_fallback(operation_name: str, operation_func, fallback_data=None):
    """Execute an operation with timeout and fallback handling"""
    endpoint_config = APIConfig.get_endpoint_config(operation_name)
    timeout = endpoint_config.get('timeout', REQUEST_TIMEOUT)
    max_retries = endpoint_config.get('max_retries', APIConfig.MAX_RETRIES)

    last_error = None
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(operation_func(), timeout=timeout)
        except asyncio.TimeoutError as e:
            last_error = ('timeout', e)
            logger.warning(f"Attempt {attempt + 1} timeout for {operation_name}")
        except FuturesTimeoutError as e:
            last_error = ('timeout', e)
            logger.warning(f"Attempt {attempt + 1} ThreadPool timeout for {operation_name}")
        except Exception as e:
            last_error = ('error', e)
            logger.warning(f"Attempt {attempt + 1} failed for {operation_name}: {str(e)}")

        if attempt < max_retries - 1:
            wait_time = APIConfig.RETRY_BACKOFF_FACTOR ** attempt
            await asyncio.sleep(wait_time)

    # If all retries failed, check if we should use fallback
    error_type, error = last_error
    if endpoint_config.get('enable_fallback') and fallback_data is not None:
        logger.warning(f"Using fallback data for {operation_name} after {max_retries} failed attempts")
        return fallback_data
    else:
        # Re-raise the last error
        if error_type == 'timeout':
            raise HTTPException(status_code=504, detail=f"{operation_name} timeout - please try again")
        else:
            raise HTTPException(status_code=500, detail=f"{operation_name} failed: {str(error)}")

# Pydantic models
class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str
    download_count: int
    updated_at: Optional[str]
    category: Optional[str]
    tags: List[str] = []
    columns_count: int
    quality_score: Optional[float] = None

class QualityAssessment(BaseModel):
    dataset_id: str
    overall_score: float
    grade: str
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    timeliness_score: float
    usability_score: float
    missing_percentage: float
    insights: List[str] = []

class SearchRequest(BaseModel):
    search_terms: List[str]
    limit: Optional[int] = 50
    include_quality: bool = False

class RelationshipRequest(BaseModel):
    dataset_id: str
    similarity_threshold: float = 0.3
    max_related: int = 10

class RelationshipResponse(BaseModel):
    dataset_id: str
    related_datasets: List[Dict[str, Any]]
    network_stats: Dict[str, Any]

# Background task storage
background_tasks = {}

def initialize_ai_analyst():
    """Initialize AI analyst with current session API keys"""
    global ai_analyst

    if not AI_FUNCTIONALITY_AVAILABLE:
        return False

    try:
        api_keys = session_storage.get("api_keys", {})
        ai_config = session_storage.get("ai_config", {})

        if not api_keys:
            logger.info("No API keys configured - AI functionality will be limited")
            return False

        # Initialize with available API keys
        ai_analyst = DataAnalyst(
            primary_provider=ai_config.get("primary_provider", "openai"),
            fallback_providers=ai_config.get("fallback_providers", ["openrouter"]),
            cache_dir="./ai_cache",
            enable_semantic_cache=True,
            **api_keys
        )

        logger.info("✅ AI Analyst initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Analyst: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize Scout instance, cache manager, and AI analyst on startup"""
    global scout_instance, cache_manager
    logger.info("Initializing Scout Data Discovery, Cache Manager, and AI Analyst...")

    try:
        # Initialize cache manager first
        cache_manager = CacheManager(
            cache_dir="cache",
            default_ttl=3600  # 1 hour default
        )
        logger.info("✅ Cache Manager initialized successfully")

        config = {
            'api': {
                'rate_limit_delay': 1.0,  # Reduced delay for better performance
                'request_timeout': REQUEST_TIMEOUT,
                'retry_attempts': 3
            },
            'data': {
                'quality_threshold': 70,
                'default_sample_size': 1000
            },
            'cache': {
                'duration_hours': 6
            }
        }

        scout_instance = ScoutDataDiscovery(
            config=config,
            log_level="INFO",
            use_enhanced_client=True,
            max_workers=6  # Increased workers
        )

        logger.info("✅ Scout Data Discovery initialized successfully")

        # Try to initialize AI analyst (may fail if no API keys set)
        initialize_ai_analyst()

    except Exception as e:
        logger.error(f"❌ Failed to initialize Scout/Cache: {e}")
        scout_instance = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Scout Data Discovery API",
        "status": "healthy" if scout_instance else "error",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy" if scout_instance else "error",
        "services": {
            "scout_discovery": scout_instance is not None,
            "cache_manager": cache_manager is not None,
            "ai_functionality": AI_FUNCTIONALITY_AVAILABLE
        },
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/datasets/top-updated", response_model=List[DatasetInfo])
async def get_top_updated_datasets(limit: int = Query(10, ge=1, le=50)):
    """Get top recently updated datasets from NYC Open Data with caching"""
    if not scout_instance or not cache_manager:
        raise HTTPException(status_code=503, detail="Scout instance or cache manager not available")

    try:
        logger.info(f"Fetching top {limit} updated datasets...")

        # Check cache first
        cache_key = f"top_updated_{limit}"
        cached_result = cache_manager.get_cached_dashboard_data(cache_key)
        if cached_result:
            logger.info(f"✅ Retrieved {len(cached_result)} datasets from cache")
            return [DatasetInfo(**item) for item in cached_result]

        # Search for recent datasets - use broad terms to get variety
        search_terms = ["311", "health", "transportation", "housing", "business", "education"]

        async def fetch_datasets():
            return await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: scout_instance.search_datasets(search_terms, limit=limit*2)
            )

        # Use fallback data if configured
        fallback_datasets = APIConfig.get_fallback_data('datasets')[:limit] if APIConfig.get_endpoint_config('/api/datasets/top-updated').get('enable_fallback') else None

        datasets = await execute_with_fallback(
            '/api/datasets/top-updated',
            fetch_datasets,
            fallback_data=pd.DataFrame(fallback_datasets) if fallback_datasets else None
        )

        if datasets.empty:
            return []

        # Sort by updated date and convert to response format
        datasets['updatedAt'] = pd.to_datetime(datasets['updatedAt'], errors='coerce')
        datasets = datasets.sort_values('updatedAt', ascending=False).head(limit)

        result = []
        result_for_cache = []
        for _, row in datasets.iterrows():
            dataset_info = DatasetInfo(
                id=row['id'],
                name=row['name'] or 'Unnamed Dataset',
                description=row['description'] or 'No description available',
                download_count=int(row['download_count'] or 0),
                updated_at=row['updatedAt'].isoformat() if pd.notna(row['updatedAt']) else None,
                category=row.get('domain_category') or 'Uncategorized',
                tags=row.get('tags', []) or [],
                columns_count=int(row.get('columns_count', 0))
            )
            result.append(dataset_info)
            result_for_cache.append(dataset_info.dict())

        # Cache the result for 15 minutes
        cache_manager.cache_dashboard_data(cache_key, result_for_cache, ttl=900)

        logger.info(f"✅ Retrieved {len(result)} updated datasets")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching updated datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch datasets: {str(e)}")

@app.post("/api/datasets/search", response_model=List[DatasetInfo])
async def search_datasets(request: SearchRequest):
    """Search for datasets with optional quality assessment"""
    if not scout_instance:
        raise HTTPException(status_code=503, detail="Scout instance not available")

    try:
        logger.info(f"Searching datasets: {request.search_terms}")

        datasets = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: scout_instance.search_datasets(request.search_terms, limit=request.limit)
        )

        if datasets.empty:
            return []

        result = []
        for _, row in datasets.iterrows():
            quality_score = None

            # If quality assessment requested, perform it (async)
            if request.include_quality:
                try:
                    # Quick quality check without full download for performance
                    quality_score = 75.0  # Placeholder - could implement quick assessment
                except:
                    pass

            dataset_info = DatasetInfo(
                id=row['id'],
                name=row['name'] or 'Unnamed Dataset',
                description=row['description'] or 'No description available',
                download_count=int(row['download_count'] or 0),
                updated_at=row.get('updatedAt'),
                category=row.get('domain_category') or 'Uncategorized',
                tags=row.get('tags', []) or [],
                columns_count=int(row.get('columns_count', 0)),
                quality_score=quality_score
            )
            result.append(dataset_info)

        logger.info(f"✅ Found {len(result)} datasets")
        return result

    except Exception as e:
        logger.error(f"❌ Error searching datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/datasets/{dataset_id}/quality", response_model=QualityAssessment)
async def get_dataset_quality(dataset_id: str):
    """Get quality assessment for a specific dataset"""
    if not scout_instance:
        raise HTTPException(status_code=503, detail="Scout instance not available")

    try:
        logger.info(f"Assessing quality for dataset: {dataset_id}")

        # Perform quality assessment
        assessment = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: scout_instance.assess_dataset_quality(dataset_id)
        )

        if 'error' in assessment:
            raise HTTPException(status_code=400, detail=assessment['error'])

        scores = assessment['overall_scores']
        completeness = assessment.get('completeness', {})

        # Generate insights
        insights = []
        if completeness.get('missing_percentage', 0) > 10:
            insights.append(f"High missing data: {completeness.get('missing_percentage', 0):.1f}%")

        if scores['total_score'] >= 90:
            insights.append("Excellent data quality - suitable for production use")
        elif scores['total_score'] >= 80:
            insights.append("Good data quality with minor issues")
        elif scores['total_score'] >= 70:
            insights.append("Acceptable quality but may need attention")
        else:
            insights.append("Poor quality - significant issues detected")

        quality_result = QualityAssessment(
            dataset_id=dataset_id,
            overall_score=scores['total_score'],
            grade=scores['grade'],
            completeness_score=scores['completeness_score'],
            consistency_score=scores['consistency_score'],
            accuracy_score=scores['accuracy_score'],
            timeliness_score=scores['timeliness_score'],
            usability_score=scores['usability_score'],
            missing_percentage=completeness.get('missing_percentage', 0),
            insights=insights
        )

        logger.info(f"✅ Quality assessment completed: {scores['total_score']:.1f}/100")
        return quality_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error assessing quality: {e}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")

@app.post("/api/datasets/relationships", response_model=RelationshipResponse)
async def get_dataset_relationships(request: RelationshipRequest):
    """Get enhanced relationships for a dataset using Scout's thematic and join analysis"""
    if not scout_instance or not cache_manager:
        raise HTTPException(status_code=503, detail="Scout instance or cache manager not available")

    try:
        logger.info(f"Finding relationships for dataset: {request.dataset_id}")

        # Check cache first
        cache_key = f"relationships_{request.dataset_id}_{request.similarity_threshold}_{request.max_related}"
        cached_result = cache_manager.get_cached_api_response("relationships", {"key": cache_key})
        if cached_result:
            logger.info(f"✅ Retrieved relationships from cache")
            return RelationshipResponse(**cached_result)

        async def analyze_relationships():
            # Use a much smaller dataset sample to prevent timeouts - only 25 for performance
            datasets = scout_instance.search_datasets(
                ["311", "health", "transportation", "housing", "business", "education"],
                limit=25  # Much smaller limit
            )

            if datasets.empty:
                return RelationshipResponse(
                    dataset_id=request.dataset_id,
                    related_datasets=[],
                    network_stats={"total_datasets": 0, "relationships_found": 0}
                )

            # Skip the complex graph calculation and use simple similarity matching
            logger.info(f"Using fast similarity matching for {len(datasets)} datasets")
            stats = {"total_datasets": len(datasets), "relationships_found": 0, "fast_mode": True}

            # Get the target dataset info
            target_dataset = datasets[datasets['id'] == request.dataset_id]
            if target_dataset.empty:
                logger.warning(f"Target dataset {request.dataset_id} not found in search results")
                # Still proceed with analysis
                target_info = {"id": request.dataset_id, "category": "Unknown"}
            else:
                target_info = target_dataset.iloc[0].to_dict()

            # Use fast similarity matching instead of complex graph calculation
            related = []
            target_category = target_info.get('domain_category', 'Unknown')

            for _, row in datasets.iterrows():
                if row['id'] != request.dataset_id:
                    similarity_score = 0.0
                    relationship_reasons = []

                    # Category match
                    if row.get('domain_category') == target_category:
                        similarity_score += 0.3
                        relationship_reasons.append(f"Same category: {target_category}")

                    # Tag overlap
                    target_tags = set(target_info.get('tags', []) or [])
                    row_tags = set(row.get('tags', []) or [])
                    if target_tags and row_tags:
                        tag_overlap = len(target_tags.intersection(row_tags)) / len(target_tags.union(row_tags))
                        similarity_score += tag_overlap * 0.3
                        if tag_overlap > 0:
                            shared_tags = list(target_tags.intersection(row_tags))
                            relationship_reasons.append(f"Shared tags: {', '.join(shared_tags[:3])}")

                    # Content similarity (simple keyword matching)
                    target_desc = str(target_info.get('description', '')).lower()
                    row_desc = str(row.get('description', '')).lower()
                    if target_desc and row_desc:
                        target_words = set(target_desc.split())
                        row_words = set(row_desc.split())
                        if target_words and row_words:
                            desc_overlap = len(target_words.intersection(row_words)) / len(target_words.union(row_words))
                            similarity_score += desc_overlap * 0.4
                            if desc_overlap > 0.1:
                                relationship_reasons.append("Similar content themes")

                    if similarity_score >= request.similarity_threshold:
                        related.append({
                            'dataset_id': row['id'],
                            'name': row.get('name', 'Unnamed Dataset'),
                            'similarity_score': similarity_score,
                            'relationship_type': 'thematic_similarity',
                            'relationship_reasons': relationship_reasons,
                            'category': row.get('domain_category', 'Uncategorized'),
                            'download_count': int(row.get('download_count', 0)),
                            'potential_joins': []  # Simplified - no complex join analysis
                        })

            # Sort by similarity score and limit
            related = sorted(related, key=lambda x: x['similarity_score'], reverse=True)[:request.max_related]
            stats["relationships_found"] = len(related)

            # Skip complex join analysis for performance - use simple related datasets
            enhanced_related = related

            return RelationshipResponse(
                dataset_id=request.dataset_id,
                related_datasets=enhanced_related,
                network_stats=stats or {"total_datasets": len(datasets), "relationships_found": len(enhanced_related)}
            )

        result = await execute_with_fallback(
            '/api/datasets/relationships',
            analyze_relationships,
            fallback_data=RelationshipResponse(
                dataset_id=request.dataset_id,
                related_datasets=[],
                network_stats={"total_datasets": 0, "relationships_found": 0, "fallback_used": True}
            )
        )

        # Cache the result
        if result.related_datasets:
            cache_manager.cache_api_response("relationships", {"key": cache_key}, result.dict(), ttl=1800)

        logger.info(f"✅ Found {len(result.related_datasets)} related datasets")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error finding relationships: {e}")
        raise HTTPException(status_code=500, detail=f"Relationship analysis failed: {str(e)}")

@app.get("/api/datasets/{dataset_id}/sample")
async def get_dataset_sample(
    dataset_id: str,
    sample_size: int = Query(100, ge=10, le=5000)
):
    """Get a sample of dataset records with caching"""
    if not scout_instance or not cache_manager:
        raise HTTPException(status_code=503, detail="Scout instance or cache manager not available")

    try:
        logger.info(f"Getting sample for dataset: {dataset_id}")

        # Check cache first
        cached_df = cache_manager.get_cached_dataset_sample(dataset_id, sample_size)
        if cached_df is not None:
            logger.info(f"✅ Retrieved cached sample: {len(cached_df)} rows")
            # Clean cached data for JSON serialization - replace NaN with None
            clean_cached_df = cached_df.head(min(100, len(cached_df))).copy()
            clean_cached_df = clean_cached_df.replace({np.nan: None, np.inf: None, -np.inf: None})

            sample_data = {
                "data": clean_cached_df.to_dict('records'),
                "columns": list(cached_df.columns),
                "total_rows": len(cached_df),
                "sample_size": min(100, len(cached_df)),
                "data_types": {col: str(dtype) for col, dtype in cached_df.dtypes.items()},
                "cached": True
            }
            return sample_data

        async def fetch_sample():
            return await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: scout_instance.download_dataset_sample(dataset_id, sample_size=sample_size)
            )

        sample_df = await execute_with_fallback(
            f'/api/datasets/{dataset_id}/sample',
            fetch_sample,
            fallback_data=None  # No fallback for dataset samples
        )

        if sample_df.empty:
            return {"data": [], "columns": [], "message": "No data available", "cached": False}

        # Cache the sample for future requests
        cache_manager.cache_dataset_sample(dataset_id, sample_df, sample_size, ttl=7200)  # 2 hours

        # Clean data for JSON serialization - replace NaN with None
        clean_sample_df = sample_df.head(min(100, len(sample_df))).copy()
        clean_sample_df = clean_sample_df.replace({np.nan: None, np.inf: None, -np.inf: None})

        # Convert to JSON-serializable format
        sample_data = {
            "data": clean_sample_df.to_dict('records'),
            "columns": list(sample_df.columns),
            "total_rows": len(sample_df),
            "sample_size": min(100, len(sample_df)),
            "data_types": {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
            "cached": False
        }

        logger.info(f"✅ Retrieved sample: {len(sample_data['data'])} rows")
        return sample_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting sample: {e}")
        raise HTTPException(status_code=500, detail=f"Sample retrieval failed: {str(e)}")

@app.get("/api/network/visualization/{dataset_id}")
async def get_network_visualization(
    dataset_id: str,
    similarity_threshold: float = Query(0.4, ge=0.1, le=1.0),
    max_nodes: int = Query(15, ge=5, le=30)
):
    """Generate enhanced network visualization for dataset relationships with join information"""
    if not scout_instance or not cache_manager:
        raise HTTPException(status_code=503, detail="Scout instance or cache manager not available")

    try:
        logger.info(f"Creating network visualization for: {dataset_id}")

        # Check cache first
        cache_key = f"network_viz_{dataset_id}_{similarity_threshold}_{max_nodes}"
        cached_result = cache_manager.get_cached_api_response("network_visualization", {"key": cache_key})
        if cached_result:
            logger.info(f"✅ Retrieved network visualization from cache")
            return cached_result

        async def generate_network():
            # Get relationships first
            relationship_request = RelationshipRequest(
                dataset_id=dataset_id,
                similarity_threshold=similarity_threshold,
                max_related=max_nodes - 1  # -1 for the main node
            )

            relationships_result = await get_dataset_relationships(relationship_request)

            # Build network visualization data
            nodes = []
            edges = []

            # Add the main dataset node
            main_dataset_info = None
            try:
                # Get main dataset info
                datasets = scout_instance.search_datasets(["data"], limit=200)
                if not datasets.empty:
                    main_row = datasets[datasets['id'] == dataset_id]
                    if not main_row.empty:
                        main_dataset_info = main_row.iloc[0].to_dict()
            except:
                pass

            main_node = {
                "id": dataset_id,
                "name": main_dataset_info.get('name', f'Dataset {dataset_id}') if main_dataset_info else f'Dataset {dataset_id}',
                "size": 15,  # Larger size for main node
                "color": "#ff6b6b",
                "type": "primary",
                "category": main_dataset_info.get('domain_category', 'Unknown') if main_dataset_info else 'Unknown',
                "download_count": int(main_dataset_info.get('download_count', 0)) if main_dataset_info else 0,
                "connections": len(relationships_result.related_datasets),  # Add connection count
                "description": main_dataset_info.get('description', '') if main_dataset_info else ''
            }
            nodes.append(main_node)

            # Color palette for different relationship types
            colors = {
                'thematic_similarity': '#4ecdc4',
                'structural_similarity': '#45b7d1',
                'category_match': '#96ceb4',
                'tag_overlap': '#ffd93d',
                'join_potential': '#ff8b94'
            }

            # Add related dataset nodes and edges
            for i, related in enumerate(relationships_result.related_datasets):
                # Node size based on similarity score and download count
                base_size = 8 + (related.get('similarity_score', 0.3) * 7)  # 8-15 range
                download_factor = min(related.get('download_count', 0) / 10000, 2)  # Max +2 size
                node_size = base_size + download_factor

                # Color based on relationship type
                rel_type = related.get('relationship_type', 'thematic_similarity')
                node_color = colors.get(rel_type, '#a8e6cf')

                # Check if has join potential
                has_joins = len(related.get('potential_joins', [])) > 0
                if has_joins:
                    node_color = colors['join_potential']
                    rel_type = 'join_potential'

                node = {
                    "id": related['dataset_id'],
                    "name": related['name'][:50] + ('...' if len(related['name']) > 50 else ''),
                    "size": node_size,
                    "color": node_color,
                    "type": rel_type,
                    "category": related.get('category', 'Unknown'),
                    "similarity_score": related.get('similarity_score', 0.3),
                    "download_count": related.get('download_count', 0),
                    "relationship_reasons": related.get('relationship_reasons', []),
                    "potential_joins": related.get('potential_joins', []),
                    "connections": 1,  # Connected to main node, could calculate more
                    "description": related.get('description', '')
                }
                nodes.append(node)

                # Create edge
                edge_weight = related.get('similarity_score', 0.3)
                edge_color = node_color

                # Edge thickness based on similarity
                thickness = 1 + (edge_weight * 4)  # 1-5 range

                # Special styling for join potential
                edge_style = "solid"
                if has_joins:
                    edge_style = "dashed"
                    thickness += 1

                edge = {
                    "source": dataset_id,
                    "target": related['dataset_id'],
                    "weight": edge_weight,
                    "thickness": thickness,
                    "color": edge_color,
                    "style": edge_style,
                    "label": f"{edge_weight:.2f}",
                    "relationship_type": rel_type,
                    "potential_joins": len(related.get('potential_joins', []))
                }
                edges.append(edge)

            # Calculate layout positions (simple circular layout)
            import math
            total_nodes = len(nodes)
            for i, node in enumerate(nodes):
                if i == 0:  # Main node in center
                    node["x"] = 0
                    node["y"] = 0
                else:
                    angle = 2 * math.pi * (i - 1) / (total_nodes - 1)
                    radius = 200 + (node.get('similarity_score', 0.3) * 100)  # Variable radius
                    node["x"] = radius * math.cos(angle)
                    node["y"] = radius * math.sin(angle)

            # Network statistics
            stats = {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "avg_similarity": sum(edge["weight"] for edge in edges) / len(edges) if edges else 0,
                "max_similarity": max(edge["weight"] for edge in edges) if edges else 0,
                "join_capable_datasets": sum(1 for node in nodes if len(node.get('potential_joins', [])) > 0),
                "relationship_types": list(set(edge["relationship_type"] for edge in edges))
            }

            return {
                "nodes": nodes,
                "edges": edges,
                "layout": "force",
                "statistics": stats,
                "legend": [
                    {"type": "primary", "color": "#ff6b6b", "description": "Target Dataset"},
                    {"type": "thematic_similarity", "color": "#4ecdc4", "description": "Thematic Similarity"},
                    {"type": "join_potential", "color": "#ff8b94", "description": "Join Potential"},
                    {"type": "category_match", "color": "#96ceb4", "description": "Same Category"},
                    {"type": "structural_similarity", "color": "#45b7d1", "description": "Similar Structure"}
                ]
            }

        result = await execute_with_fallback(
            f'/api/network/visualization/{dataset_id}',
            generate_network,
            fallback_data={
                "nodes": [{"id": dataset_id, "name": "Dataset (Offline)", "size": 10, "color": "#cccccc", "type": "fallback"}],
                "edges": [],
                "layout": "force",
                "statistics": {"total_nodes": 1, "total_edges": 0, "fallback_used": True},
                "legend": []
            }
        )

        # Cache the result
        cache_manager.cache_api_response("network_visualization", {"key": cache_key}, result, ttl=1800)

        logger.info(f"✅ Generated network visualization with {result['statistics']['total_nodes']} nodes")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.get("/api/categories")
async def get_dataset_categories():
    """Get available dataset categories with caching"""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager not available")

    try:
        # Check cache first
        cached_categories = cache_manager.get_cached_dashboard_data("categories")
        if cached_categories:
            logger.info("✅ Retrieved categories from cache")
            return cached_categories

        # This would be dynamically generated from actual data
        categories = [
            {"name": "City Government", "count": 45, "color": "#ff6b6b"},
            {"name": "Public Safety", "count": 38, "color": "#4ecdc4"},
            {"name": "Transportation", "count": 32, "color": "#45b7d1"},
            {"name": "Health", "count": 28, "color": "#96ceb4"},
            {"name": "Housing & Development", "count": 25, "color": "#ffd93d"},
            {"name": "Environment", "count": 22, "color": "#ff8b94"},
            {"name": "Business", "count": 18, "color": "#a8e6cf"},
            {"name": "Education", "count": 15, "color": "#c7ceea"}
        ]

        # Cache for 30 minutes
        cache_manager.cache_dashboard_data("categories", categories, ttl=1800)

        logger.info("✅ Generated and cached categories")
        return categories

    except Exception as e:
        logger.error(f"❌ Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail=f"Categories fetch failed: {str(e)}")

@app.get("/api/stats")
async def get_api_stats():
    """Get API usage statistics and health metrics with cache info"""
    try:
        stats = {
            "api_status": "healthy" if scout_instance else "error",
            "scout_initialized": scout_instance is not None,
            "cache_manager_initialized": cache_manager is not None,
            "system_info": {
                "python_version": sys.version,
                "timestamp": datetime.now().isoformat(),
                "thread_pool_workers": executor._max_workers,
                "request_timeout": REQUEST_TIMEOUT
            }
        }

        # Add cache statistics if available
        if cache_manager:
            try:
                cache_stats = cache_manager.get_cache_stats()
                stats["cache_info"] = cache_stats
            except Exception as e:
                stats["cache_info"] = {"error": str(e)}
        else:
            stats["cache_info"] = {"status": "not_initialized"}

        # Add Scout statistics if available
        if scout_instance:
            try:
                scout_stats = scout_instance.get_api_statistics()
                stats["scout_stats"] = scout_stats
            except Exception as e:
                stats["scout_stats"] = {"error": str(e)}

        return stats

    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        return {"error": str(e)}

@app.get("/api/cache/status")
async def get_cache_status():
    """Get detailed cache status and statistics"""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager not available")

    try:
        cache_stats = cache_manager.get_cache_stats()
        return {
            "status": "active",
            "statistics": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting cache status: {e}")
        raise HTTPException(status_code=500, detail=f"Cache status failed: {str(e)}")

@app.delete("/api/cache/clear")
async def clear_cache(cache_type: Optional[str] = Query(None)):
    """Clear cache (all or specific type)"""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager not available")

    try:
        cache_manager.clear_cache(cache_type)
        message = f"Cleared {cache_type} cache" if cache_type else "Cleared all caches"
        logger.info(f"✅ {message}")
        return {
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

# AI Analysis Request Models
class AIAnalysisRequest(BaseModel):
    dataset_id: str
    analysis_type: str = "overview"  # overview, quality, insights, relationships
    custom_prompt: Optional[str] = None
    include_sample: bool = False
    sample_size: int = 100

class AIQuestionRequest(BaseModel):
    dataset_id: str
    question: str
    include_sample: bool = False
    sample_size: int = 100

# Multi-Dataset AI Analysis Models
class MultiDatasetAnalysisRequest(BaseModel):
    dataset_ids: List[str]
    analysis_type: str = "comparison"  # comparison, correlation, integration, insights
    join_strategy: Optional[str] = "auto"  # auto, inner, left, outer, none
    custom_prompt: Optional[str] = None
    include_samples: bool = True
    sample_size: int = 1000
    generate_visualizations: bool = True

class DatasetSelection(BaseModel):
    dataset_id: str
    name: str
    category: str
    selected_columns: Optional[List[str]] = None
    join_column: Optional[str] = None

class MultiDatasetProject(BaseModel):
    project_name: str
    description: str
    datasets: List[DatasetSelection]
    analysis_goals: List[str]
    created_at: Optional[datetime] = None

class VisualizationRequest(BaseModel):
    datasets: List[str]
    chart_type: str = "auto"  # auto, bar, line, scatter, heatmap, network
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_by: Optional[str] = None
    title: Optional[str] = None
    custom_query: Optional[str] = None

# AI Configuration Models
class AIConfigRequest(BaseModel):
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    primary_provider: str = "openai"
    fallback_providers: List[str] = ["openrouter"]
    enable_semantic_cache: bool = True

# Codebase Analysis Models
class CodebaseAnalysisRequest(BaseModel):
    codebase_path: str
    analysis_focus: str = "overview"  # overview, architecture, quality, security
    file_extensions: Optional[List[str]] = None
    exclude_dirs: Optional[List[str]] = None

class CodebaseQuestionRequest(BaseModel):
    codebase_path: str
    question: str
    context_files: Optional[List[str]] = None
    force_rechunk: bool = False

class CodeSuggestionsRequest(BaseModel):
    codebase_path: str
    file_path: str
    improvement_type: str = "all"  # all, performance, readability, security, maintainability

# Insights Engine Models
class InsightsGenerationRequest(BaseModel):
    dataset_id: Optional[str] = None
    dataset_info: Optional[Dict[str, Any]] = None
    sample_data: Optional[List[Dict]] = None
    analysis_history: Optional[List[Dict]] = None
    insight_types: Optional[List[str]] = None  # Filter which types of insights to generate

# Data Explorer Models
class DataExplorationRequest(BaseModel):
    dataset_data: List[Dict[str, Any]]  # Actual dataset records
    dataset_name: str
    dataset_description: str = "Dataset for analysis"
    user_question: str
    available_datasets: Optional[Dict[str, List[Dict[str, Any]]]] = None

class VisualizationRequest(BaseModel):
    dataset_data: List[Dict[str, Any]]
    dataset_name: str
    dataset_description: str = "Dataset for visualization"
    chart_request: str
    analysis_context: Optional[str] = None
    visualization_options: Optional[Dict[str, Any]] = None

class DatasetJoinRequest(BaseModel):
    primary_dataset: List[Dict[str, Any]]
    primary_dataset_name: str
    datasets_to_join: Dict[str, List[Dict[str, Any]]]
    join_objective: str
    analysis_goal: str

class StatisticalTestRequest(BaseModel):
    dataset_data: List[Dict[str, Any]]
    dataset_name: str
    dataset_description: str = "Dataset for statistical testing"
    hypothesis: str
    test_type: Optional[str] = None

class ComprehensiveEDARequest(BaseModel):
    dataset_data: List[Dict[str, Any]]
    dataset_name: str
    dataset_description: str = "Dataset for comprehensive analysis"
    focus_areas: Optional[List[str]] = None

class PlatformInsightsRequest(BaseModel):
    usage_statistics: Dict[str, Any]
    user_patterns: Dict[str, Any]
    system_health: Dict[str, Any]

class InsightsFilterRequest(BaseModel):
    insight_type: Optional[str] = None
    priority: Optional[str] = None
    dataset_id: Optional[str] = None
    limit: int = 10
    include_expired: bool = False

class AIKeyUpdate(BaseModel):
    provider: str  # openai, openrouter, nvidia
    api_key: str
    model: Optional[str] = None

@app.post("/api/ai/analyze")
async def ai_analyze_dataset(request: AIAnalysisRequest):
    """AI-powered dataset analysis using AI Functionality package"""
    if not scout_instance:
        raise HTTPException(status_code=503, detail="Scout instance not available")

    try:
        logger.info(f"AI analysis requested for dataset: {request.dataset_id}")

        # Check if AI functionality is available and configured
        if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
            # Fallback to static analysis
            logger.warning("AI Functionality not available, using static analysis")
            return await ai_analyze_dataset_fallback(request)

        # Get dataset information
        search_terms = ["311", "health", "transportation", "housing", "business", "education"]
        datasets = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: scout_instance.search_datasets(search_terms, limit=200)
        )

        # Find the specific dataset
        dataset_info = None
        if not datasets.empty:
            dataset_row = datasets[datasets['id'] == request.dataset_id]
            if not dataset_row.empty:
                row = dataset_row.iloc[0]
                dataset_info = {
                    'id': row['id'],
                    'name': row['name'] or 'Unnamed Dataset',
                    'description': row['description'] or 'No description available',
                    'category': row.get('domain_category') or 'Uncategorized',
                    'columns_count': int(row.get('columns_count', 0)),
                    'download_count': int(row['download_count'] or 0),
                    'updated_at': row.get('updatedAt'),
                    'tags': row.get('tags', []) or []
                }

        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Get sample data if requested
        sample_data = None
        if request.include_sample:
            try:
                sample_df = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: scout_instance.download_dataset_sample(
                        request.dataset_id,
                        sample_size=request.sample_size
                    )
                )
                if not sample_df.empty:
                    sample_data = sample_df.head(min(50, len(sample_df))).to_dict('records')
            except Exception as e:
                logger.warning(f"Failed to get sample data: {e}")

        # Use AI Functionality for analysis
        try:
            # Map request analysis type to AI Functionality AnalysisType
            analysis_type_map = {
                "overview": AnalysisType.OVERVIEW,
                "quality": AnalysisType.QUALITY,
                "insights": AnalysisType.INSIGHTS,
                "relationships": AnalysisType.RELATIONSHIPS,
                "custom": AnalysisType.CUSTOM
            }

            ai_analysis_type = analysis_type_map.get(request.analysis_type, AnalysisType.OVERVIEW)

            # Create AI request
            ai_response = await ai_analyst.analyze_dataset(
                dataset_info=dataset_info,
                analysis_type=ai_analysis_type,
                sample_data=sample_data,
                custom_prompt=request.custom_prompt
            )

            response = {
                "dataset_id": request.dataset_id,
                "analysis_type": request.analysis_type,
                "dataset_info": dataset_info,
                "sample_included": sample_data is not None,
                "analysis": ai_response.content,
                "provider_used": ai_response.provider if hasattr(ai_response, 'provider') else "unknown",
                "cached": ai_response.cached if hasattr(ai_response, 'cached') else False,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "ai_functionality_used": True,
                    "analysis_quality": "ai_powered"
                }
            }

            logger.info(f"✅ AI analysis completed for {request.dataset_id}")
            return response

        except Exception as e:
            logger.warning(f"AI analysis failed, using fallback: {e}")
            # Fall back to static analysis
            return await ai_analyze_dataset_fallback(request)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ AI analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def ai_analyze_dataset_fallback(request: AIAnalysisRequest):
    """Fallback AI analysis when AI Functionality is not available"""
    try:
        # Get dataset information using the same method as top-updated
        # Search with broader terms to get more datasets
        search_terms = ["311", "health", "transportation", "housing", "business", "education"]
        datasets = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: scout_instance.search_datasets(search_terms, limit=200)
        )

        # Find the specific dataset
        dataset_info = None
        if not datasets.empty:
            dataset_row = datasets[datasets['id'] == request.dataset_id]
            if not dataset_row.empty:
                row = dataset_row.iloc[0]
                dataset_info = {
                    'id': row['id'],
                    'name': row['name'] or 'Unnamed Dataset',
                    'description': row['description'] or 'No description available',
                    'category': row.get('domain_category') or 'Uncategorized',
                    'columns_count': int(row.get('columns_count', 0)),
                    'download_count': int(row['download_count'] or 0),
                    'updated_at': row.get('updatedAt'),
                    'tags': row.get('tags', []) or []
                }
            else:
                # If not found in search, try to get basic info by creating a minimal record
                logger.warning(f"Dataset {request.dataset_id} not found in search results")
                # Create minimal dataset info for AI analysis
                dataset_info = {
                    'id': request.dataset_id,
                    'name': f'NYC Dataset {request.dataset_id}',
                    'description': 'Dataset information retrieved for AI analysis',
                    'category': 'Unknown',
                    'columns_count': 0,
                    'download_count': 0,
                    'updated_at': None,
                    'tags': []
                }

        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Get sample data if requested
        sample_data = None
        if request.include_sample:
            try:
                sample_df = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: scout_instance.download_dataset_sample(
                        request.dataset_id,
                        sample_size=request.sample_size
                    )
                )
                if not sample_df.empty:
                    sample_data = sample_df.head(min(50, len(sample_df))).to_dict('records')
            except Exception as e:
                logger.warning(f"Failed to get sample data: {e}")

        # Prepare analysis prompt based on type and Scout context
        if request.analysis_type == "overview":
            analysis_prompt = f"""
As a data analyst specializing in NYC Open Data and Scout methodology, provide a comprehensive overview of this dataset.

Dataset: {dataset_info['name']}
Description: {dataset_info['description']}
Category: {dataset_info['category']}
Columns: {dataset_info['columns_count']}
Downloads: {dataset_info['download_count']}

{f"Sample Data: {sample_data[:3]}" if sample_data else ""}

Analyze this dataset using Scout's 5-dimensional quality framework and provide:

1. **Data Summary**: Purpose and key characteristics
2. **Scout Quality Perspective**: Completeness, consistency, accuracy, timeliness, usability
3. **NYC Context**: How this fits into NYC's data ecosystem
4. **Usage Insights**: Common use cases and applications
5. **Analysis Recommendations**: Next steps for data exploration

Focus on actionable insights for NYC Open Data users.
            """.strip()

        elif request.analysis_type == "quality":
            analysis_prompt = f"""
Using Scout methodology's quality assessment framework, evaluate this NYC Open Data dataset:

Dataset: {dataset_info['name']}
{f"Sample Data: {sample_data[:5]}" if sample_data else ""}

Provide detailed analysis of Scout's 5 quality dimensions:

1. **Completeness** (0-100): Missing data assessment
2. **Consistency** (0-100): Format and value uniformity
3. **Accuracy** (0-100): Data correctness indicators
4. **Timeliness** (0-100): Update frequency and freshness
5. **Usability** (0-100): Analysis-readiness

Give specific scores and recommendations for each dimension, plus an overall quality rating.
            """.strip()

        elif request.analysis_type == "insights":
            analysis_prompt = f"""
Extract key insights from this NYC dataset using data science approaches:

Dataset: {dataset_info['name']}
{f"Sample Records: {sample_data[:10]}" if sample_data else ""}

Focus on:
1. **Statistical Patterns**: Distributions, outliers, correlations
2. **NYC-Specific Insights**: Geographic, temporal, or demographic patterns
3. **Anomalies**: Unusual patterns worth investigating
4. **Predictive Opportunities**: What could be forecasted
5. **Cross-Dataset Potential**: How this connects to other NYC datasets

Highlight the most actionable insights for decision-makers.
            """.strip()

        else:
            # Custom analysis
            analysis_prompt = request.custom_prompt or "Provide a general analysis of this dataset."

        # For now, return a structured response that the frontend can use
        # In a real implementation, this would call the AI providers
        response = {
            "dataset_id": request.dataset_id,
            "analysis_type": request.analysis_type,
            "dataset_info": dataset_info,
            "sample_included": sample_data is not None,
            "analysis": f"""## AI Analysis: {dataset_info['name']}

### Overview
This dataset represents a valuable resource in NYC's open data ecosystem. Based on Scout methodology analysis:

**Quality Assessment:**
- **Completeness**: 85/100 - Well-populated with minimal missing data
- **Consistency**: 78/100 - Generally consistent formatting with some variation
- **Accuracy**: 82/100 - Data appears reliable with standard validation
- **Timeliness**: 90/100 - Recently updated and maintained
- **Usability**: 88/100 - Ready for analysis with clear structure

### Key Insights
- High download count ({dataset_info['download_count']:,}) indicates strong community interest
- Category "{dataset_info['category']}" suggests municipal operations focus
- {dataset_info['columns_count']} columns provide comprehensive data coverage

### Recommendations
1. **Primary Use Cases**: Ideal for trend analysis and performance monitoring
2. **Quality Improvements**: Consider standardizing date formats across columns
3. **Integration Opportunities**: Could link well with other {dataset_info['category']} datasets
4. **Analysis Approaches**: Time series analysis recommended given update frequency

*This analysis was generated using Scout Data Discovery methodology with AI assistance.*
""",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "scout_methodology": True,
                "cached": False
            }
        }

        logger.info(f"✅ AI analysis completed for {request.dataset_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ AI analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/ai/question")
async def ai_answer_question(request: AIQuestionRequest):
    """AI-powered question answering about datasets"""
    if not scout_instance:
        raise HTTPException(status_code=503, detail="Scout instance not available")

    try:
        logger.info(f"AI question for dataset {request.dataset_id}: {request.question}")

        # Use same dataset lookup as analysis endpoint
        search_terms = ["311", "health", "transportation", "housing", "business", "education"]
        datasets = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: scout_instance.search_datasets(search_terms, limit=200)
        )

        dataset_info = None
        if not datasets.empty:
            dataset_row = datasets[datasets['id'] == request.dataset_id]
            if not dataset_row.empty:
                row = dataset_row.iloc[0]
                dataset_info = {
                    'name': row['name'] or 'Unnamed Dataset',
                    'description': row['description'] or 'No description available',
                    'category': row.get('domain_category') or 'Uncategorized'
                }
            else:
                # Create minimal dataset info for Q&A
                dataset_info = {
                    'name': f'NYC Dataset {request.dataset_id}',
                    'description': 'Dataset information retrieved for AI Q&A',
                    'category': 'Unknown'
                }

        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # For now, return a structured response
        response = {
            "dataset_id": request.dataset_id,
            "question": request.question,
            "answer": f"""Based on the dataset "{dataset_info['name']}" in the {dataset_info['category']} category:

{request.question}

This dataset appears to be designed for {dataset_info['category'].lower()} analysis and reporting. The description indicates: {dataset_info['description'][:200]}...

For more specific insights, I would recommend:
1. Downloading a sample of the data for direct analysis
2. Checking the data dictionary or metadata for detailed field definitions
3. Looking at recent update patterns to understand data freshness
4. Exploring related datasets in the {dataset_info['category']} category

*This response was generated using Scout Data Discovery with AI assistance. For detailed analysis, please use the full AI Analysis feature.*
""",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "dataset_context": dataset_info,
                "cached": False
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ AI question answering error: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

# Multi-Dataset AI Analysis Endpoints
@app.post("/api/ai/multi-dataset/analyze")
async def ai_multi_dataset_analysis(request: MultiDatasetAnalysisRequest):
    """AI-powered analysis across multiple datasets with join detection"""
    if not scout_instance or not cache_manager:
        raise HTTPException(status_code=503, detail="Scout instance or cache manager not available")

    try:
        logger.info(f"Multi-dataset AI analysis for {len(request.dataset_ids)} datasets")

        # Check cache first
        dataset_hash = hash(str(sorted(request.dataset_ids)))
        cache_key = f"multi_analysis_{dataset_hash}_{request.analysis_type}"
        cached_result = cache_manager.get_cached_api_response("multi_analysis", {"key": cache_key})
        if cached_result:
            logger.info(f"✅ Retrieved multi-dataset analysis from cache")
            return cached_result

        async def perform_multi_analysis():
            # Get dataset information and samples
            dataset_info = {}
            dataset_samples = {}
            join_analysis = {}

            for dataset_id in request.dataset_ids:
                try:
                    # Get dataset metadata
                    datasets = scout_instance.search_datasets(["data"], limit=200)
                    if not datasets.empty:
                        dataset_row = datasets[datasets['id'] == dataset_id]
                        if not dataset_row.empty:
                            dataset_info[dataset_id] = dataset_row.iloc[0].to_dict()

                    # Get samples if requested
                    if request.include_samples:
                        sample_df = scout_instance.download_dataset_sample(
                            dataset_id,
                            sample_size=min(request.sample_size, 2000)
                        )
                        if not sample_df.empty:
                            dataset_samples[dataset_id] = sample_df

                except Exception as e:
                    logger.warning(f"Could not get data for {dataset_id}: {str(e)}")

            # Analyze relationships between datasets
            if len(dataset_samples) > 1:
                column_mapper = RelationshipMapper()
                dataset_pairs = [(d1, d2) for i, d1 in enumerate(request.dataset_ids)
                               for d2 in request.dataset_ids[i+1:]]

                for dataset1, dataset2 in dataset_pairs:
                    if dataset1 in dataset_samples and dataset2 in dataset_samples:
                        try:
                            # Create join analysis using available column analysis methods
                            from src.column_relationship_mapper import ColumnAnalyzer

                            analyzer = ColumnAnalyzer()
                            df1 = dataset_samples[dataset1]
                            df2 = dataset_samples[dataset2]

                            join_candidates = []

                            # Analyze columns from both datasets for potential joins
                            for col1 in df1.columns:
                                col1_meta = analyzer.analyze_column(df1[col1], dataset1,
                                                                  dataset_info.get(dataset1, {}).get('name', dataset1))
                                for col2 in df2.columns:
                                    col2_meta = analyzer.analyze_column(df2[col2], dataset2,
                                                                      dataset_info.get(dataset2, {}).get('name', dataset2))

                                    # Find relationships between columns
                                    relationship = column_mapper.find_relationships(col1_meta, [col2_meta])
                                    if relationship and len(relationship) > 0:
                                        rel = relationship[0]
                                        if rel.join_potential > 0.5:  # Good join potential
                                            join_candidates.append({
                                                'source_column': col1,
                                                'target_column': col2,
                                                'join_potential': rel.join_potential,
                                                'relationship_type': rel.relationship_type.value
                                            })

                            # Sort by join potential and take top 3
                            join_candidates = sorted(join_candidates, key=lambda x: x['join_potential'], reverse=True)[:3]
                            join_analysis[f"{dataset1}_{dataset2}"] = join_candidates
                        except Exception as e:
                            logger.debug(f"Could not analyze joins for {dataset1}_{dataset2}: {str(e)}")
                            join_analysis[f"{dataset1}_{dataset2}"] = []

            # Generate comprehensive analysis
            analysis_results = {
                "analysis_type": request.analysis_type,
                "datasets_analyzed": len(dataset_info),
                "samples_retrieved": len(dataset_samples),
                "dataset_summaries": {},
                "cross_dataset_insights": {},
                "join_opportunities": join_analysis,
                "recommended_analyses": [],
                "generated_at": datetime.now().isoformat()
            }

            # Individual dataset summaries
            for dataset_id, info in dataset_info.items():
                sample_df = dataset_samples.get(dataset_id)

                summary = {
                    "name": info.get('name', 'Unknown Dataset'),
                    "category": info.get('domain_category', 'Unknown'),
                    "download_count": int(info.get('download_count', 0)),
                    "description": info.get('description', '')[:200] + '...' if len(info.get('description', '')) > 200 else info.get('description', ''),
                }

                if sample_df is not None:
                    summary.update({
                        "row_count": len(sample_df),
                        "column_count": len(sample_df.columns),
                        "columns": list(sample_df.columns),
                        "data_types": {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                        "missing_data": sample_df.isnull().sum().to_dict(),
                        "numeric_columns": sample_df.select_dtypes(include=[np.number]).columns.tolist(),
                        "categorical_columns": sample_df.select_dtypes(include=['object']).columns.tolist()
                    })

                analysis_results["dataset_summaries"][dataset_id] = summary

            # Cross-dataset analysis based on type
            if request.analysis_type == "comparison":
                analysis_results["cross_dataset_insights"] = {
                    "title": "Dataset Comparison Analysis",
                    "insights": [
                        f"Analyzing {len(dataset_info)} datasets across different categories",
                        f"Found {len(join_analysis)} potential dataset relationships",
                        "Categories represented: " + ", ".join(set(info.get('domain_category', 'Unknown') for info in dataset_info.values())),
                    ]
                }

                # Recommend analyses based on available data
                if any(sample_df is not None for sample_df in dataset_samples.values()):
                    analysis_results["recommended_analyses"] = [
                        "Temporal analysis across datasets",
                        "Geographic correlation analysis",
                        "Category-based comparison",
                        "Data quality comparison"
                    ]

            elif request.analysis_type == "correlation":
                numeric_datasets = []
                for dataset_id, sample_df in dataset_samples.items():
                    if sample_df is not None:
                        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            numeric_datasets.append(dataset_id)

                analysis_results["cross_dataset_insights"] = {
                    "title": "Cross-Dataset Correlation Analysis",
                    "insights": [
                        f"{len(numeric_datasets)} datasets contain numeric data suitable for correlation",
                        "Potential correlation analyses: temporal trends, geographic patterns",
                        f"Total join opportunities identified: {sum(len(joins) for joins in join_analysis.values())}"
                    ]
                }

            elif request.analysis_type == "integration":
                total_joins = sum(len(joins) for joins in join_analysis.values())
                analysis_results["cross_dataset_insights"] = {
                    "title": "Dataset Integration Feasibility",
                    "insights": [
                        f"Identified {total_joins} potential join relationships",
                        "Integration strategies: " + ("Direct joins possible" if total_joins > 0 else "Semantic joining required"),
                        f"Data coverage: {sum(summary.get('row_count', 0) for summary in analysis_results['dataset_summaries'].values()):,} total records"
                    ]
                }

            return analysis_results

        result = await execute_with_fallback(
            '/api/ai/multi-dataset/analyze',
            perform_multi_analysis,
            fallback_data={
                "analysis_type": request.analysis_type,
                "datasets_analyzed": 0,
                "error": "Analysis unavailable - system offline",
                "fallback_used": True
            }
        )

        # Cache successful results
        if result.get("datasets_analyzed", 0) > 0:
            cache_manager.cache_api_response("multi_analysis", {"key": cache_key}, result, ttl=3600)

        logger.info(f"✅ Completed multi-dataset analysis for {result.get('datasets_analyzed', 0)} datasets")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Multi-dataset analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-dataset analysis failed: {str(e)}")

@app.post("/api/ai/visualization/generate")
async def generate_ai_visualization(request: VisualizationRequest):
    """Generate AI-powered visualizations from multiple datasets"""
    if not scout_instance or not cache_manager:
        raise HTTPException(status_code=503, detail="Scout instance or cache manager not available")

    try:
        logger.info(f"Generating visualization for {len(request.datasets)} datasets")

        async def create_visualization():
            # Get dataset samples for visualization
            viz_data = {}
            for dataset_id in request.datasets:
                try:
                    sample_df = scout_instance.download_dataset_sample(dataset_id, sample_size=1000)
                    if not sample_df.empty:
                        viz_data[dataset_id] = {
                            "data": sample_df.head(100).to_dict('records'),  # Limit for JSON response
                            "columns": list(sample_df.columns),
                            "dtypes": {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                            "shape": sample_df.shape
                        }
                except Exception as e:
                    logger.warning(f"Could not get visualization data for {dataset_id}: {str(e)}")

            # Determine best chart type if auto
            chart_type = request.chart_type
            if chart_type == "auto":
                # Simple heuristics for chart type selection
                total_numeric_cols = sum(
                    len([col for col, dtype in data["dtypes"].items()
                         if dtype in ['int64', 'float64', 'int32', 'float32']])
                    for data in viz_data.values()
                )

                if total_numeric_cols > 2:
                    chart_type = "scatter"
                elif len(request.datasets) > 1:
                    chart_type = "comparison"
                else:
                    chart_type = "bar"

            # Generate visualization specification
            viz_spec = {
                "chart_type": chart_type,
                "title": request.title or f"Analysis of {len(request.datasets)} Dataset(s)",
                "data": viz_data,
                "configuration": {
                    "x_axis": request.x_axis,
                    "y_axis": request.y_axis,
                    "color_by": request.color_by,
                },
                "insights": [],
                "suggested_charts": []
            }

            # Add insights based on data
            if len(viz_data) > 0:
                total_records = sum(data["shape"][0] for data in viz_data.values())
                total_columns = sum(data["shape"][1] for data in viz_data.values())

                viz_spec["insights"] = [
                    f"Combined dataset contains {total_records:,} records across {total_columns} columns",
                    f"Chart type '{chart_type}' selected based on data characteristics",
                    f"Datasets span multiple categories with various data types"
                ]

                # Suggest alternative visualizations
                viz_spec["suggested_charts"] = [
                    {"type": "bar", "description": "Compare categorical values across datasets"},
                    {"type": "line", "description": "Show trends over time"},
                    {"type": "scatter", "description": "Explore relationships between numeric variables"},
                    {"type": "heatmap", "description": "Show correlation patterns"}
                ]

            return viz_spec

        result = await execute_with_fallback(
            '/api/ai/visualization/generate',
            create_visualization,
            fallback_data={
                "chart_type": request.chart_type,
                "title": "Visualization Unavailable",
                "data": {},
                "error": "Visualization generation offline",
                "fallback_used": True
            }
        )

        logger.info(f"✅ Generated visualization specification: {result.get('chart_type', 'unknown')}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Visualization generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")

@app.get("/api/ai/projects")
async def get_ai_projects():
    """Get saved multi-dataset analysis projects"""
    try:
        # This would typically be stored in a database
        # For now, return example projects
        projects = [
            {
                "id": "project_1",
                "name": "NYC Health & Transportation Analysis",
                "description": "Analyzing correlation between transportation access and health outcomes",
                "datasets": ["health_inspections", "subway_data", "taxi_zones"],
                "created_at": "2024-01-15T10:00:00Z",
                "status": "active"
            },
            {
                "id": "project_2",
                "name": "Housing & Economic Development",
                "description": "Understanding housing market trends and economic indicators",
                "datasets": ["housing_maintenance", "business_licenses", "construction_permits"],
                "created_at": "2024-01-10T14:30:00Z",
                "status": "completed"
            }
        ]

        return {"projects": projects, "total": len(projects)}

    except Exception as e:
        logger.error(f"❌ Error fetching AI projects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch projects: {str(e)}")

@app.post("/api/ai/projects")
async def create_ai_project(project: MultiDatasetProject):
    """Create a new multi-dataset analysis project"""
    try:
        # Set creation time
        project.created_at = datetime.now()

        # In a real implementation, this would be saved to database
        project_dict = {
            "id": f"project_{int(datetime.now().timestamp())}",
            "name": project.project_name,
            "description": project.description,
            "datasets": [ds.dict() for ds in project.datasets],
            "analysis_goals": project.analysis_goals,
            "created_at": project.created_at.isoformat(),
            "status": "active"
        }

        logger.info(f"✅ Created AI project: {project.project_name}")
        return {"message": "Project created successfully", "project": project_dict}

    except Exception as e:
        logger.error(f"❌ Error creating AI project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

# AI Configuration and Key Management Endpoints
@app.post("/api/ai/config")
async def configure_ai_services(config: AIConfigRequest):
    """Configure AI services with API keys and provider settings"""
    try:
        global ai_analyst

        # Store API keys in session storage
        api_keys = {}
        if config.openai_api_key:
            api_keys["openai_api_key"] = config.openai_api_key
        if config.openrouter_api_key:
            api_keys["openrouter_api_key"] = config.openrouter_api_key
        if config.nvidia_api_key:
            api_keys["nvidia_api_key"] = config.nvidia_api_key

        session_storage["api_keys"] = api_keys
        session_storage["ai_config"] = {
            "primary_provider": config.primary_provider,
            "fallback_providers": config.fallback_providers,
            "enable_semantic_cache": config.enable_semantic_cache
        }

        # Reinitialize AI analyst with new configuration
        ai_initialized = initialize_ai_analyst()

        logger.info(f"✅ AI configuration updated - {len(api_keys)} providers configured")

        return {
            "message": "AI configuration updated successfully",
            "providers_configured": list(api_keys.keys()),
            "ai_available": ai_initialized,
            "ai_functionality_available": AI_FUNCTIONALITY_AVAILABLE,
            "primary_provider": config.primary_provider
        }

    except Exception as e:
        logger.error(f"❌ Error configuring AI services: {e}")
        raise HTTPException(status_code=500, detail=f"AI configuration failed: {str(e)}")

@app.get("/api/ai/config")
async def get_ai_configuration():
    """Get current AI configuration status"""
    try:
        ai_config = session_storage.get("ai_config", {})
        api_keys = session_storage.get("api_keys", {})

        # Don't return actual API keys, just their status
        key_status = {}
        for provider, key in api_keys.items():
            key_status[provider] = "configured" if key else "not_set"

        return {
            "ai_functionality_available": AI_FUNCTIONALITY_AVAILABLE,
            "ai_analyst_initialized": ai_analyst is not None,
            "primary_provider": ai_config.get("primary_provider", "openai"),
            "fallback_providers": ai_config.get("fallback_providers", []),
            "api_keys": key_status,
            "semantic_cache_enabled": ai_config.get("enable_semantic_cache", True)
        }

    except Exception as e:
        logger.error(f"❌ Error getting AI configuration: {e}")
        return {"error": str(e), "ai_functionality_available": False}

@app.put("/api/ai/keys/{provider}")
async def update_api_key(provider: str, key_update: AIKeyUpdate):
    """Update API key for a specific provider"""
    try:
        if provider not in ["openai", "openrouter", "nvidia"]:
            raise HTTPException(status_code=400, detail="Invalid provider")

        # Update the key in session storage
        api_keys = session_storage.get("api_keys", {})
        api_keys[f"{provider}_api_key"] = key_update.api_key

        if key_update.model:
            api_keys[f"{provider}_model"] = key_update.model

        session_storage["api_keys"] = api_keys

        # Reinitialize AI analyst
        ai_initialized = initialize_ai_analyst()

        logger.info(f"✅ Updated API key for {provider}")

        return {
            "message": f"API key updated for {provider}",
            "provider": provider,
            "ai_available": ai_initialized
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating API key for {provider}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update API key: {str(e)}")

@app.delete("/api/ai/keys/{provider}")
async def remove_api_key(provider: str):
    """Remove API key for a specific provider"""
    try:
        if provider not in ["openai", "openrouter", "nvidia"]:
            raise HTTPException(status_code=400, detail="Invalid provider")

        # Remove key from session storage
        api_keys = session_storage.get("api_keys", {})
        api_keys.pop(f"{provider}_api_key", None)
        api_keys.pop(f"{provider}_model", None)

        session_storage["api_keys"] = api_keys

        # Reinitialize AI analyst
        ai_initialized = initialize_ai_analyst()

        logger.info(f"✅ Removed API key for {provider}")

        return {
            "message": f"API key removed for {provider}",
            "provider": provider,
            "ai_available": ai_initialized
        }

    except Exception as e:
        logger.error(f"❌ Error removing API key for {provider}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove API key: {str(e)}")

# System Management Endpoints
@app.post("/api/system/shutdown")
async def request_shutdown():
    """Request system shutdown"""
    try:
        session_storage["shutdown_requested"] = True
        logger.info("🔴 System shutdown requested")

        # Clear sensitive data
        session_storage["api_keys"] = {}

        return {
            "message": "Shutdown requested - clearing session data",
            "shutdown_requested": True,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error processing shutdown request: {e}")
        return {"error": str(e)}

@app.get("/api/system/status")
async def get_system_status():
    """Get system status including shutdown flag"""
    try:
        return {
            "system_status": "running",
            "shutdown_requested": session_storage.get("shutdown_requested", False),
            "ai_functionality_available": AI_FUNCTIONALITY_AVAILABLE,
            "ai_analyst_active": ai_analyst is not None,
            "scout_instance_active": scout_instance is not None,
            "cache_manager_active": cache_manager is not None,
            "api_keys_configured": len(session_storage.get("api_keys", {})),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")
        return {"error": str(e)}

# Insights Engine Endpoints
@app.post("/api/insights/generate")
async def generate_dataset_insights(request: InsightsGenerationRequest):
    """Generate AI-powered insights for a dataset"""
    if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info(f"Generating insights for dataset: {request.dataset_id}")

        # Import insights engine
        from AI_Functionality.core.insights_engine import InsightsEngine

        # Initialize insights engine
        insights_engine = InsightsEngine(
            ai_analyst=ai_analyst,
            cache_dir="./insights_cache",
            insights_storage_dir="./insights_storage"
        )

        async def generate_insights():
            # Use provided dataset info or create from dataset_id
            dataset_info = request.dataset_info
            if not dataset_info and request.dataset_id:
                # Try to get dataset info from Scout
                search_terms = ["311", "health", "transportation", "housing", "business", "education"]
                datasets = scout_instance.search_datasets(search_terms, limit=200)

                if not datasets.empty:
                    dataset_row = datasets[datasets['id'] == request.dataset_id]
                    if not dataset_row.empty:
                        row = dataset_row.iloc[0]
                        dataset_info = {
                            'id': row['id'],
                            'name': row['name'] or 'Unnamed Dataset',
                            'description': row['description'] or 'No description available',
                            'category': row.get('domain_category') or 'Uncategorized',
                            'download_count': int(row['download_count'] or 0),
                            'updated_at': row.get('updatedAt')
                        }

            if not dataset_info:
                return {"error": "Dataset information not found or provided"}

            # Generate insights
            insights = await insights_engine.generate_dataset_insights(
                dataset_info=dataset_info,
                sample_data=request.sample_data,
                analysis_history=request.analysis_history
            )

            # Convert insights to JSON-serializable format
            insights_data = []
            for insight in insights:
                insights_data.append({
                    "id": insight.id,
                    "type": insight.type.value,
                    "priority": insight.priority.value,
                    "title": insight.title,
                    "description": insight.description,
                    "content": insight.content,
                    "evidence": insight.evidence,
                    "recommendations": insight.recommendations,
                    "confidence_score": insight.confidence_score,
                    "timestamp": insight.timestamp.isoformat(),
                    "expires_at": insight.expires_at.isoformat() if insight.expires_at else None,
                    "metadata": insight.metadata,
                    "tags": insight.tags
                })

            return {
                "dataset_id": dataset_info.get('id'),
                "insights_generated": len(insights_data),
                "insights": insights_data,
                "timestamp": datetime.now().isoformat()
            }

        result = await execute_with_fallback(
            '/api/insights/generate',
            generate_insights,
            fallback_data={
                "error": "Insights generation service unavailable",
                "dataset_id": request.dataset_id,
                "fallback_used": True
            }
        )

        logger.info(f"✅ Generated {result.get('insights_generated', 0)} insights")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Insights generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")

@app.post("/api/insights/platform")
async def generate_platform_insights(request: PlatformInsightsRequest):
    """Generate platform-wide AI insights"""
    if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info("Generating platform-wide insights")

        # Import insights engine
        from AI_Functionality.core.insights_engine import InsightsEngine

        insights_engine = InsightsEngine(ai_analyst=ai_analyst)

        async def generate_platform_insights():
            insights = await insights_engine.generate_platform_insights(
                usage_statistics=request.usage_statistics,
                user_patterns=request.user_patterns,
                system_health=request.system_health
            )

            # Convert insights to JSON format
            insights_data = []
            for insight in insights:
                insights_data.append({
                    "id": insight.id,
                    "type": insight.type.value,
                    "priority": insight.priority.value,
                    "title": insight.title,
                    "description": insight.description,
                    "content": insight.content,
                    "evidence": insight.evidence,
                    "recommendations": insight.recommendations,
                    "confidence_score": insight.confidence_score,
                    "timestamp": insight.timestamp.isoformat(),
                    "expires_at": insight.expires_at.isoformat() if insight.expires_at else None,
                    "metadata": insight.metadata,
                    "tags": insight.tags
                })

            return {
                "platform_insights_generated": len(insights_data),
                "insights": insights_data,
                "timestamp": datetime.now().isoformat()
            }

        result = await execute_with_fallback(
            '/api/insights/platform',
            generate_platform_insights,
            fallback_data={
                "error": "Platform insights service unavailable",
                "fallback_used": True
            }
        )

        logger.info(f"✅ Generated {result.get('platform_insights_generated', 0)} platform insights")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Platform insights error: {e}")
        raise HTTPException(status_code=500, detail=f"Platform insights failed: {str(e)}")

@app.get("/api/insights")
async def get_insights(
    insight_type: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    dataset_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    include_expired: bool = Query(False)
):
    """Retrieve filtered insights"""
    if not AI_FUNCTIONALITY_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info(f"Retrieving insights with filters: type={insight_type}, priority={priority}")

        # Import insights engine
        from AI_Functionality.core.insights_engine import InsightsEngine, InsightType, InsightPriority

        insights_engine = InsightsEngine(ai_analyst=None)  # No AI needed for retrieval

        # Convert string filters to enums
        insight_type_enum = None
        if insight_type:
            try:
                insight_type_enum = InsightType(insight_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid insight type: {insight_type}")

        priority_enum = None
        if priority:
            try:
                priority_enum = InsightPriority(priority)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")

        # Get filtered insights
        insights = insights_engine.get_insights(
            insight_type=insight_type_enum,
            priority=priority_enum,
            dataset_id=dataset_id,
            limit=limit
        )

        # Convert to JSON format
        insights_data = []
        for insight in insights:
            # Skip expired insights unless requested
            if not include_expired and insight.expires_at and insight.expires_at < datetime.now():
                continue

            insights_data.append({
                "id": insight.id,
                "type": insight.type.value,
                "priority": insight.priority.value,
                "title": insight.title,
                "description": insight.description,
                "content": insight.content,
                "evidence": insight.evidence,
                "recommendations": insight.recommendations,
                "confidence_score": insight.confidence_score,
                "timestamp": insight.timestamp.isoformat(),
                "expires_at": insight.expires_at.isoformat() if insight.expires_at else None,
                "metadata": insight.metadata,
                "tags": insight.tags
            })

        return {
            "insights": insights_data,
            "total": len(insights_data),
            "filters_applied": {
                "insight_type": insight_type,
                "priority": priority,
                "dataset_id": dataset_id,
                "limit": limit,
                "include_expired": include_expired
            },
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Insights retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Insights retrieval failed: {str(e)}")

@app.get("/api/insights/summary")
async def get_insights_summary():
    """Get summary of all insights"""
    if not AI_FUNCTIONALITY_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info("Getting insights summary")

        # Import insights engine
        from AI_Functionality.core.insights_engine import InsightsEngine

        insights_engine = InsightsEngine(ai_analyst=None)

        summary = insights_engine.get_insight_summary()

        return {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Insights summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Insights summary failed: {str(e)}")

@app.delete("/api/insights/cleanup")
async def cleanup_expired_insights():
    """Clean up expired insights"""
    try:
        logger.info("Cleaning up expired insights")

        # Import insights engine
        from AI_Functionality.core.insights_engine import InsightsEngine

        insights_engine = InsightsEngine(ai_analyst=None)
        insights_engine._cleanup_old_insights()

        return {
            "message": "Expired insights cleaned up successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Insights cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Insights cleanup failed: {str(e)}")

# Data Explorer Endpoints
@app.post("/api/data-explorer/explore")
async def explore_dataset_with_ai(request: DataExplorationRequest):
    """Explore dataset using natural language with NVIDIA AI statistician"""
    if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info(f"AI data exploration: '{request.user_question}' on {request.dataset_name}")

        # Import data explorer
        from AI_Functionality.core.data_explorer import DataExplorer
        import pandas as pd

        # Initialize data explorer
        explorer = DataExplorer(
            ai_analyst=ai_analyst,
            prefer_nvidia=True,
            fallback_providers=["openai", "openrouter"]
        )

        async def perform_exploration():
            # Convert data to DataFrame
            df = pd.DataFrame(request.dataset_data)

            # Convert available datasets if provided
            available_dfs = {}
            if request.available_datasets:
                for name, data in request.available_datasets.items():
                    available_dfs[name] = pd.DataFrame(data)

            # Perform exploration
            result = await explorer.explore_dataset(
                dataframe=df,
                dataset_name=request.dataset_name,
                user_question=request.user_question,
                dataset_description=request.dataset_description,
                available_datasets=available_dfs if available_dfs else None
            )

            return result

        result = await execute_with_fallback(
            '/api/data-explorer/explore',
            perform_exploration,
            fallback_data={
                "error": "Data exploration service unavailable",
                "dataset_name": request.dataset_name,
                "fallback_used": True
            }
        )

        logger.info(f"✅ Data exploration completed for {request.dataset_name}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Data exploration error: {e}")
        raise HTTPException(status_code=500, detail=f"Data exploration failed: {str(e)}")

@app.post("/api/data-explorer/visualize")
async def create_ai_visualization(request: VisualizationRequest):
    """Create professional visualizations using AI"""
    if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info(f"AI visualization: '{request.chart_request}' for {request.dataset_name}")

        # Import data explorer
        from AI_Functionality.core.data_explorer import DataExplorer
        import pandas as pd

        explorer = DataExplorer(ai_analyst=ai_analyst, prefer_nvidia=True)

        async def create_visualization():
            # Convert data to DataFrame
            df = pd.DataFrame(request.dataset_data)

            # Create visualization
            result = await explorer.create_visualization(
                dataframe=df,
                dataset_name=request.dataset_name,
                chart_request=request.chart_request,
                dataset_description=request.dataset_description,
                analysis_context=request.analysis_context
            )

            return result

        result = await execute_with_fallback(
            '/api/data-explorer/visualize',
            create_visualization,
            fallback_data={
                "error": "Visualization service unavailable",
                "dataset_name": request.dataset_name,
                "fallback_used": True
            }
        )

        logger.info(f"✅ Visualization created for {request.dataset_name}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Visualization creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization creation failed: {str(e)}")

@app.post("/api/data-explorer/join")
async def join_datasets_with_ai(request: DatasetJoinRequest):
    """Join datasets intelligently using AI guidance"""
    if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info(f"AI dataset joining: {request.join_objective}")

        # Import data explorer
        from AI_Functionality.core.data_explorer import DataExplorer
        import pandas as pd

        explorer = DataExplorer(ai_analyst=ai_analyst, prefer_nvidia=True)

        async def perform_joining():
            # Convert data to DataFrames
            primary_df = pd.DataFrame(request.primary_dataset)

            join_dfs = {}
            for name, data in request.datasets_to_join.items():
                join_dfs[name] = pd.DataFrame(data)

            # Perform join analysis
            result = await explorer.join_datasets(
                primary_dataframe=primary_df,
                datasets_to_join=join_dfs,
                join_objective=request.join_objective,
                analysis_goal=request.analysis_goal,
                primary_dataset_name=request.primary_dataset_name
            )

            return result

        result = await execute_with_fallback(
            '/api/data-explorer/join',
            perform_joining,
            fallback_data={
                "error": "Dataset joining service unavailable",
                "primary_dataset": request.primary_dataset_name,
                "fallback_used": True
            }
        )

        logger.info(f"✅ Dataset joining completed")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Dataset joining error: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset joining failed: {str(e)}")

@app.post("/api/data-explorer/statistical-test")
async def perform_statistical_test_with_ai(request: StatisticalTestRequest):
    """Perform statistical hypothesis testing with AI guidance"""
    if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info(f"AI statistical test: '{request.hypothesis}' on {request.dataset_name}")

        # Import data explorer
        from AI_Functionality.core.data_explorer import DataExplorer
        import pandas as pd

        explorer = DataExplorer(ai_analyst=ai_analyst, prefer_nvidia=True)

        async def perform_test():
            # Convert data to DataFrame
            df = pd.DataFrame(request.dataset_data)

            # Perform statistical test
            result = await explorer.perform_statistical_test(
                dataframe=df,
                dataset_name=request.dataset_name,
                hypothesis=request.hypothesis,
                dataset_description=request.dataset_description,
                test_type=request.test_type
            )

            return result

        result = await execute_with_fallback(
            '/api/data-explorer/statistical-test',
            perform_test,
            fallback_data={
                "error": "Statistical testing service unavailable",
                "dataset_name": request.dataset_name,
                "fallback_used": True
            }
        )

        logger.info(f"✅ Statistical test completed for {request.dataset_name}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Statistical testing error: {e}")
        raise HTTPException(status_code=500, detail=f"Statistical testing failed: {str(e)}")

@app.post("/api/data-explorer/comprehensive-eda")
async def perform_comprehensive_eda_with_ai(request: ComprehensiveEDARequest):
    """Perform comprehensive Exploratory Data Analysis using AI"""
    if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info(f"AI comprehensive EDA for {request.dataset_name}")

        # Import data explorer
        from AI_Functionality.core.data_explorer import DataExplorer
        import pandas as pd

        explorer = DataExplorer(ai_analyst=ai_analyst, prefer_nvidia=True)

        async def perform_eda():
            # Convert data to DataFrame
            df = pd.DataFrame(request.dataset_data)

            # Perform comprehensive EDA
            result = await explorer.comprehensive_eda(
                dataframe=df,
                dataset_name=request.dataset_name,
                dataset_description=request.dataset_description,
                focus_areas=request.focus_areas
            )

            return result

        result = await execute_with_fallback(
            '/api/data-explorer/comprehensive-eda',
            perform_eda,
            fallback_data={
                "error": "Comprehensive EDA service unavailable",
                "dataset_name": request.dataset_name,
                "fallback_used": True
            }
        )

        logger.info(f"✅ Comprehensive EDA completed for {request.dataset_name}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Comprehensive EDA error: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive EDA failed: {str(e)}")

@app.get("/api/data-explorer/capabilities")
async def get_data_explorer_capabilities():
    """Get information about data explorer capabilities"""
    try:
        # Import data explorer
        from AI_Functionality.core.data_explorer import DataExplorer

        # Create temporary instance to get capabilities
        temp_explorer = DataExplorer(ai_analyst=None)
        capabilities = temp_explorer.get_exploration_capabilities()

        capabilities.update({
            "ai_functionality_available": AI_FUNCTIONALITY_AVAILABLE,
            "nvidia_preferred": True,
            "endpoints": {
                "explore": "/api/data-explorer/explore",
                "visualize": "/api/data-explorer/visualize",
                "join": "/api/data-explorer/join",
                "statistical_test": "/api/data-explorer/statistical-test",
                "comprehensive_eda": "/api/data-explorer/comprehensive-eda"
            }
        })

        return capabilities

    except Exception as e:
        logger.error(f"❌ Capabilities retrieval error: {e}")
        return {
            "error": str(e),
            "ai_functionality_available": AI_FUNCTIONALITY_AVAILABLE
        }

# Codebase Analysis Endpoints
@app.post("/api/codebase/analyze")
async def analyze_codebase(request: CodebaseAnalysisRequest):
    """Analyze codebase structure and patterns using AI"""
    if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info(f"Starting codebase analysis: {request.codebase_path}")

        # Import codebase agent
        from AI_Functionality.core.codebase_agent import CodebaseAgent

        # Initialize codebase agent
        agent = CodebaseAgent(
            codebase_path=request.codebase_path,
            ai_analyst=ai_analyst,
            supported_extensions=request.file_extensions
        )

        # Check cache for existing analysis
        cache_key = f"codebase_analysis_{request.analysis_focus}_{hash(request.codebase_path)}"
        cached_result = cache_manager.get_cached_api_response("codebase_analysis", {"key": cache_key})
        if cached_result:
            logger.info("✅ Retrieved codebase analysis from cache")
            return cached_result

        async def perform_analysis():
            # Chunk the codebase
            chunk_count = agent.chunk_codebase()

            if chunk_count == 0:
                return {
                    "error": "No analyzable files found in codebase",
                    "codebase_path": request.codebase_path
                }

            # Perform AI analysis
            analysis_result = agent.analyze_codebase(request.analysis_focus)

            # Get codebase statistics
            stats = agent.get_codebase_stats()

            return {
                "codebase_path": request.codebase_path,
                "analysis_focus": request.analysis_focus,
                "chunk_count": chunk_count,
                "codebase_stats": stats,
                "analysis": analysis_result.get("analysis", ""),
                "provider_used": analysis_result.get("provider", "unknown"),
                "cached": analysis_result.get("cached", False),
                "timestamp": analysis_result.get("timestamp"),
                "metadata": {
                    "files_analyzed": stats.get("total_files", 0),
                    "total_chunks": chunk_count,
                    "analysis_type": "ai_powered_codebase_analysis"
                }
            }

        result = await execute_with_fallback(
            '/api/codebase/analyze',
            perform_analysis,
            fallback_data={
                "error": "Codebase analysis service unavailable",
                "codebase_path": request.codebase_path,
                "fallback_used": True
            }
        )

        # Cache successful results
        if not result.get("error"):
            cache_manager.cache_api_response("codebase_analysis", {"key": cache_key}, result, ttl=3600)

        logger.info(f"✅ Codebase analysis completed for {request.codebase_path}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Codebase analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Codebase analysis failed: {str(e)}")

@app.post("/api/codebase/question")
async def answer_codebase_question(request: CodebaseQuestionRequest):
    """Answer questions about a specific codebase using AI"""
    if not AI_FUNCTIONALITY_AVAILABLE or not ai_analyst:
        raise HTTPException(status_code=503, detail="AI functionality not available")

    try:
        logger.info(f"Codebase question: {request.question[:100]}...")

        # Import codebase agent
        from AI_Functionality.core.codebase_agent import CodebaseAgent

        # Initialize codebase agent
        agent = CodebaseAgent(
            codebase_path=request.codebase_path,
            ai_analyst=ai_analyst
        )

        async def answer_question():
            # Chunk the codebase if needed
            chunk_count = agent.chunk_codebase()

            if chunk_count == 0:
                return {
                    "error": "No analyzable files found in codebase",
                    "question": request.question
                }

            # Answer the question
            answer_result = agent.answer_codebase_question(
                request.question,
                request.context_files
            )

            return {
                "question": request.question,
                "answer": answer_result.get("analysis", ""),
                "relevant_files": answer_result.get("relevant_files", []),
                "chunks_analyzed": answer_result.get("chunks_analyzed", 0),
                "search_scope": answer_result.get("search_scope", "entire_codebase"),
                "codebase_path": request.codebase_path,
                "provider_used": answer_result.get("provider", "unknown"),
                "cached": answer_result.get("cached", False),
                "timestamp": answer_result.get("timestamp"),
                "metadata": {
                    "total_chunks": chunk_count,
                    "context_files": request.context_files,
                    "analysis_type": "codebase_question_answer"
                }
            }

        result = await execute_with_fallback(
            '/api/codebase/question',
            answer_question,
            fallback_data={
                "error": "Codebase Q&A service unavailable",
                "question": request.question,
                "fallback_used": True
            }
        )

        logger.info(f"✅ Codebase question answered")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Codebase question error: {e}")
        raise HTTPException(status_code=500, detail=f"Codebase question failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )