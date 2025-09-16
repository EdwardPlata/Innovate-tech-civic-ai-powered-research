# Scout Data Discovery Backend API

A FastAPI-based REST API server that provides web-accessible endpoints for the Scout Data Discovery engine. This backend serves as the bridge between the Streamlit frontend and the core Scout data discovery and quality assessment functionality.

## 🏗️ Architecture Overview

```
┌─────────────────────┐    HTTP/REST    ┌─────────────────────┐    Python API    ┌─────────────────────┐
│                     │    Requests     │                     │    Calls          │                     │
│   Streamlit         │◄───────────────►│   FastAPI           │◄─────────────────►│   Scout Data        │
│   Frontend          │                 │   Backend           │                   │   Discovery         │
│   (Port 8501)       │                 │   (Port 8080)       │                   │   Engine            │
│                     │                 │                     │                   │                     │
└─────────────────────┘                 └─────────────────────┘                   └─────────────────────┘
│                                       │                                         │
│ • Interactive UI                      │ • REST API Endpoints                   │ • Core Discovery Logic
│ • Data Visualization                  │ • Request Validation                   │ • Quality Assessment  
│ • User Experience                     │ • Error Handling                       │ • Relationship Analysis
│ • Dashboard                           │ • Async Processing                     │ • Data Sampling
│                                       │ • CORS Configuration                   │ • Caching & Performance
                                        │ • Background Tasks                     │
                                        │                                         │
                                        └─────────────────────────────────────────┘
                                        │                     │
                                        ▼                     ▼
                                ┌─────────────────┐   ┌─────────────────┐
                                │                 │   │                 │
                                │   NYC Open      │   │   Scout Core    │
                                │   Data API      │   │   Components    │
                                │   (Socrata)     │   │                 │
                                │                 │   │                 │
                                └─────────────────┘   └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Scout Data Discovery package in parent directory

### Installation & Setup

1. **Navigate to the backend directory:**
   ```bash
   cd /path/to/QLT_Workshop/backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server:**
   ```bash
   python run_server.py
   ```

4. **Access the API:**
   - **API Server**: http://localhost:8080
   - **Interactive Documentation**: http://localhost:8080/docs
   - **Health Check**: http://localhost:8080

## 📋 API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Scout Component Used |
|----------|--------|-------------|---------------------|
| `/` | GET | Health check and status | `scout_instance` status |
| `/api/stats` | GET | API usage statistics | `scout_instance.get_api_statistics()` |

### Dataset Discovery

| Endpoint | Method | Description | Scout Component Used |
|----------|--------|-------------|---------------------|
| `/api/datasets/top-updated` | GET | Get recently updated datasets | `scout_instance.search_datasets()` |
| `/api/datasets/search` | POST | Search datasets by terms | `scout_instance.search_datasets()` |
| `/api/datasets/{id}/sample` | GET | Get dataset sample data | `scout_instance.download_dataset_sample()` |

### Quality Assessment

| Endpoint | Method | Description | Scout Component Used |
|----------|--------|-------------|---------------------|
| `/api/datasets/{id}/quality` | GET | Comprehensive quality assessment | `scout_instance.assess_dataset_quality()` |

### Relationship Analysis

| Endpoint | Method | Description | Scout Component Used |
|----------|--------|-------------|---------------------|
| `/api/datasets/relationships` | POST | Find related datasets | `DatasetRelationshipGraph` |
| `/api/network/visualization/{id}` | GET | Network visualization data | `DatasetRelationshipGraph` |

### Metadata & Categories

| Endpoint | Method | Description | Scout Component Used |
|----------|--------|-------------|---------------------|
| `/api/categories` | GET | Available dataset categories | Static data + future Scout integration |

## 🔗 Scout Data Discovery Integration

### Core Scout Components Used

#### 1. **ScoutDataDiscovery** (`scout_discovery.py`)
- **Purpose**: Main orchestration class for data discovery workflows
- **Backend Usage**: Initialized globally as `scout_instance` during startup
- **Key Methods Used**:
  - `search_datasets()` - Powers dataset search endpoints
  - `assess_dataset_quality()` - Provides quality scores and insights
  - `download_dataset_sample()` - Retrieves data samples for preview

```python
# Backend initialization
scout_instance = ScoutDataDiscovery(
    config={
        'api': {'rate_limit_delay': 1.5, 'request_timeout': 60},
        'data': {'quality_threshold': 70, 'default_sample_size': 1000},
        'cache': {'duration_hours': 6}
    },
    use_enhanced_client=True,
    max_workers=3
)
```

#### 2. **DatasetRelationshipGraph** (`dataset_relationship_graph.py`)
- **Purpose**: Analyzes relationships between datasets using similarity metrics
- **Backend Usage**: Created on-demand for relationship analysis
- **Key Methods Used**:
  - `add_datasets()` - Adds datasets to the analysis graph
  - `calculate_relationships()` - Computes similarity scores
  - `get_related_datasets()` - Returns most similar datasets

#### 3. **Enhanced API Client** (`enhanced_api_client.py`)
- **Purpose**: Advanced Socrata API integration with caching and optimization
- **Backend Usage**: Integrated within ScoutDataDiscovery instance
- **Features Leveraged**:
  - Rate limiting and retry logic
  - SoQL query building
  - Parallel data fetching

#### 4. **Data Quality Assessor** (`data_quality.py`)
- **Purpose**: Multi-dimensional quality assessment framework
- **Backend Usage**: Integrated within ScoutDataDiscovery for quality endpoints
- **Assessment Dimensions**:
  - Completeness (missing data analysis)
  - Consistency (type validation, format checking)
  - Accuracy (outlier detection, range validation)
  - Timeliness (freshness analysis)
  - Usability (structure and documentation quality)

### Data Flow Through Scout Components

```
API Request → FastAPI Backend → Scout Components → NYC Open Data → Response
     ↑                              ↓
     └─── JSON Response ←─── Data Processing ←─── Raw Data ←─────────┘
```

1. **Frontend Request**: Streamlit sends HTTP request to backend
2. **Request Validation**: FastAPI validates request using Pydantic models
3. **Scout Invocation**: Backend calls appropriate Scout component method
4. **Data Processing**: Scout components interact with NYC Open Data API
5. **Quality Assessment**: If requested, data quality is analyzed
6. **Response Formation**: Results are formatted and returned as JSON

## 🏛️ Core Architecture Components

### 1. **FastAPI Application** (`main.py`)

```python
app = FastAPI(
    title="Scout Data Discovery API",
    description="Backend API for exploring NYC Open Data with Scout methodology",
    version="1.0.0"
)
```

**Key Features:**
- Automatic API documentation generation
- Request/response validation with Pydantic
- Async/await support for better performance
- Built-in error handling and logging

### 2. **Pydantic Models**

Data validation and serialization models that ensure type safety:

```python
class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str
    download_count: int
    category: Optional[str]
    quality_score: Optional[float]

class QualityAssessment(BaseModel):
    overall_score: float
    grade: str
    completeness_score: float
    consistency_score: float
    # ... other quality dimensions
```

### 3. **Async Processing**

Background task execution for heavy operations:

```python
# Run Scout operations in thread pool to avoid blocking
datasets = await asyncio.get_event_loop().run_in_executor(
    executor,
    lambda: scout_instance.search_datasets(search_terms, limit=limit)
)
```

### 4. **CORS Configuration**

Cross-Origin Resource Sharing setup for frontend communication:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🛡️ Error Handling & Logging

### Error Response Format

```json
{
    "status_code": 500,
    "detail": "Quality assessment failed: Dataset not found",
    "type": "DataDownloadError"
}
```

### Logging Configuration

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Log Sources:**
- FastAPI request/response logs
- Scout Data Discovery operation logs
- Error tracking and debugging information

## ⚡ Performance Optimizations

### 1. **Async Processing**
- Non-blocking API endpoints using `asyncio`
- Background task execution with `ThreadPoolExecutor`
- Concurrent Scout operations where possible

### 2. **Caching Strategy**
- Scout-level caching for dataset metadata and samples
- Configurable cache duration (default: 6 hours)
- In-memory result caching for frequently accessed data

### 3. **Rate Limiting**
- Respectful API usage with configurable delays
- Retry logic for failed requests
- Connection pooling for improved performance

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host address | `0.0.0.0` |
| `PORT` | Server port | `8080` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `SCOUT_CACHE_DIR` | Cache directory for Scout | `./cache` |

### Scout Configuration

```python
config = {
    'api': {
        'rate_limit_delay': 1.5,     # Delay between API calls
        'request_timeout': 60,        # Request timeout in seconds
        'retry_attempts': 3           # Number of retry attempts
    },
    'data': {
        'quality_threshold': 70,      # Minimum quality score
        'default_sample_size': 1000   # Default sample size
    },
    'cache': {
        'duration_hours': 6           # Cache validity period
    }
}
```

## 🧪 Testing

### Manual Testing

1. **Health Check:**
   ```bash
   curl http://localhost:8080/
   ```

2. **Dataset Search:**
   ```bash
   curl -X POST "http://localhost:8080/api/datasets/search" \
        -H "Content-Type: application/json" \
        -d '{"search_terms": ["311", "complaints"], "limit": 10}'
   ```

3. **Quality Assessment:**
   ```bash
   curl http://localhost:8080/api/datasets/{dataset_id}/quality
   ```

### API Documentation

Interactive API testing available at: http://localhost:8080/docs

## 🚧 Development

### Project Structure

```
backend/
├── main.py              # FastAPI application and endpoints
├── run_server.py        # Server startup script
├── requirements.txt     # Python dependencies
├── README.md           # This documentation
└── __pycache__/        # Python cache files
```

### Adding New Endpoints

1. **Define Pydantic models** for request/response validation
2. **Implement endpoint function** with proper error handling
3. **Add Scout component integration** as needed
4. **Update documentation** and tests

### Development Server

```bash
# Run with auto-reload for development
python run_server.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

## 🔮 Future Enhancements

### Planned Features

1. **Authentication & Authorization**
   - User management and API keys
   - Rate limiting per user/organization
   - Role-based access control

2. **Enhanced Caching**
   - Redis integration for distributed caching
   - Cache invalidation strategies
   - Performance metrics

3. **Background Jobs**
   - Async quality assessment jobs
   - Batch processing capabilities
   - Job status tracking

4. **Monitoring & Analytics**
   - API usage analytics
   - Performance monitoring
   - Health checks and alerts

### Scout Integration Improvements

1. **Real-time Updates**
   - WebSocket support for live data updates
   - Streaming quality assessments
   - Real-time relationship graph updates

2. **Advanced Analytics**
   - Machine learning-powered recommendations
   - Trend analysis and forecasting
   - Automated insight generation

## 📞 Support & Contributing

### Common Issues

1. **Scout Instance Initialization Failed**
   - Check that `scout_data_discovery` directory exists
   - Verify all dependencies are installed
   - Check network connectivity for API access

2. **Port Already in Use**
   - Change port in `run_server.py` and update frontend configuration
   - Check for other running services

3. **CORS Errors**
   - Verify frontend URL in CORS middleware configuration
   - Check browser console for specific CORS issues

### Development Guidelines

- Follow FastAPI best practices for endpoint design
- Maintain compatibility with Scout Data Discovery API changes
- Add comprehensive error handling for all Scout operations
- Update documentation when adding new features

---

**Related Documentation:**
- [Scout Data Discovery README](../scout_data_discovery/README.md)
- [Frontend Documentation](../frontend/README.md)
- [API Documentation](http://localhost:8080/docs) (when server is running)