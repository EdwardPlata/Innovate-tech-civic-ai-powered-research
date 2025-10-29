# Backend Code Organization

This document explains the reorganized backend structure for better maintainability and scalability.

## Overview

The backend has been restructured to follow best practices for API development, with clear separation of concerns and modular organization.

## Directory Structure

```
backend/
├── main.py                 # FastAPI app initialization and main endpoints
├── api_config.py          # Configuration settings and constants
├── cache_manager.py       # Caching functionality
├── run_server.py          # Server startup script
├── requirements.txt       # Python dependencies
├── README.md              # Backend documentation
└── api/                   # Organized API modules
    ├── __init__.py
    ├── models.py          # All Pydantic models
    ├── routes/            # Route handlers (future)
    │   └── __init__.py
    ├── services/          # Business logic (future)
    │   └── __init__.py
    └── utils/             # Helper functions (future)
        └── __init__.py
```

## Module Descriptions

### `main.py`
**Purpose**: FastAPI application initialization and core endpoints

**Contents:**
- FastAPI app setup
- CORS middleware configuration
- Startup/shutdown event handlers
- All API route definitions
- Scout instance initialization

**Future**: Can be split into smaller route files as the API grows

### `api/models.py`
**Purpose**: Centralized data models for the API

**Contents:**
All Pydantic models for:
- Dataset operations (`DatasetInfo`, `QualityAssessment`, etc.)
- Search and filtering (`SearchRequest`, `RelationshipRequest`)
- AI analysis (`AIAnalysisRequest`, `AIQuestionRequest`, etc.)
- Configuration (`AIConfigRequest`, `AIKeyUpdate`)
- Data exploration (`DataExplorationRequest`, etc.)

**Benefits:**
- Single source of truth for data structures
- Easy to import across the application
- Type safety and validation
- Clear API contracts

**Usage:**
```python
from api.models import DatasetInfo, SearchRequest

# In a route handler
@app.post("/api/search")
async def search(request: SearchRequest):
    # request is validated automatically
    ...
```

### `api/routes/` (Future)
**Purpose**: Organized route handlers by feature

**Planned Structure:**
```
api/routes/
├── __init__.py
├── datasets.py      # Dataset-related routes
├── quality.py       # Quality assessment routes
├── relationships.py # Relationship mapping routes
├── ai_analysis.py   # AI analysis routes
└── system.py        # System/health routes
```

**Example** (`datasets.py`):
```python
from fastapi import APIRouter
from api.models import DatasetInfo, SearchRequest

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

@router.get("/top-updated")
async def get_top_updated(limit: int = 10):
    ...

@router.post("/search")
async def search_datasets(request: SearchRequest):
    ...
```

### `api/services/` (Future)
**Purpose**: Business logic and data processing

**Planned Structure:**
```
api/services/
├── __init__.py
├── dataset_service.py    # Dataset operations
├── quality_service.py    # Quality assessment logic
├── ai_service.py         # AI integration logic
└── cache_service.py      # Caching operations
```

**Benefits:**
- Separates business logic from route handlers
- Reusable across multiple endpoints
- Easier to test
- Clear responsibility boundaries

**Example** (`dataset_service.py`):
```python
class DatasetService:
    def __init__(self, scout_instance, cache_manager):
        self.scout = scout_instance
        self.cache = cache_manager
    
    async def get_top_updated(self, limit: int):
        # Business logic here
        ...
    
    async def search_datasets(self, search_terms, limit):
        # Search logic here
        ...
```

### `api/utils/` (Future)
**Purpose**: Helper functions and utilities

**Planned Structure:**
```
api/utils/
├── __init__.py
├── formatters.py    # Data formatting utilities
├── validators.py    # Custom validation logic
└── helpers.py       # General helper functions
```

## Migration Path

### Phase 1: Models (✅ Complete)
- Extracted all Pydantic models to `api/models.py`
- Created package structure with `__init__.py` files

### Phase 2: Routes (Future)
When `main.py` grows beyond 3000 lines:
1. Create route files in `api/routes/`
2. Move related endpoints to appropriate files
3. Use FastAPI's `APIRouter` for modular routing
4. Import and include routers in `main.py`

```python
# main.py
from api.routes import datasets, quality, ai_analysis

app.include_router(datasets.router)
app.include_router(quality.router)
app.include_router(ai_analysis.router)
```

### Phase 3: Services (Future)
When business logic becomes complex:
1. Create service classes in `api/services/`
2. Extract logic from route handlers
3. Inject services via dependency injection
4. Makes testing easier

### Phase 4: Utilities (Future)
As helper functions accumulate:
1. Move to `api/utils/`
2. Organize by functionality
3. Keep utilities pure and stateless when possible

## Best Practices

### 1. Keep Models Updated
When adding new endpoints, add models to `api/models.py`:

```python
class NewFeatureRequest(BaseModel):
    field1: str
    field2: int
    optional_field: Optional[str] = None
```

### 2. Use Type Hints
Always use proper type hints:

```python
from typing import List, Dict, Optional
from api.models import DatasetInfo

async def process_datasets(datasets: List[DatasetInfo]) -> Dict[str, Any]:
    ...
```

### 3. Document Your Models
Add docstrings and field descriptions:

```python
class SearchRequest(BaseModel):
    """Request model for dataset search"""
    search_terms: List[str]  # List of terms to search for
    limit: Optional[int] = 50  # Maximum results to return
```

### 4. Follow RESTful Conventions
- Use appropriate HTTP methods (GET, POST, PUT, DELETE)
- Use noun-based URLs: `/api/datasets` not `/api/get_datasets`
- Use HTTP status codes correctly
- Return consistent response formats

### 5. Error Handling
Use FastAPI's exception handling:

```python
from fastapi import HTTPException

if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")
```

## Current API Structure

### Dataset Endpoints
- `GET /api/datasets/top-updated` - Get recently updated datasets
- `POST /api/datasets/search` - Search datasets
- `GET /api/datasets/{id}/quality` - Get quality assessment
- `GET /api/datasets/{id}/sample` - Get data sample

### Relationship Endpoints
- `POST /api/datasets/relationships` - Find related datasets
- `GET /api/network/visualization/{id}` - Get network graph

### AI Analysis Endpoints
- `POST /api/ai/analyze` - Analyze a dataset
- `POST /api/ai/question` - Ask questions about a dataset
- `POST /api/ai/multi-dataset/analyze` - Analyze multiple datasets
- `POST /api/ai/visualization/generate` - Generate visualizations

### Configuration Endpoints
- `POST /api/ai/config` - Configure AI settings
- `PUT /api/ai/keys/{provider}` - Update API keys
- `DELETE /api/ai/keys/{provider}` - Remove API keys

### System Endpoints
- `GET /api/health` - Health check
- `GET /api/stats` - API statistics
- `POST /api/system/shutdown` - Shutdown server

## Benefits of This Organization

1. **Maintainability**: Easy to find and update code
2. **Scalability**: Ready for growth without major refactoring
3. **Testability**: Clear boundaries make testing easier
4. **Team Development**: Multiple developers can work on different modules
5. **Documentation**: Organized structure is self-documenting
6. **Code Reuse**: Services can be reused across routes
7. **Type Safety**: Centralized models ensure consistency

## Related Documentation

- [Performance Optimization Guide](./PERFORMANCE_OPTIMIZATION_GUIDE.md)
- [Backend README](../backend/README.md)
- [API Documentation](http://localhost:8080/docs) (when server is running)

## Questions?

For questions about backend architecture, refer to the main project documentation or check the implementation examples in `main.py`.
