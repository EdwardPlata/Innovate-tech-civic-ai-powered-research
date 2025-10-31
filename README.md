# Scout Data Discovery Platform

A comprehensive platform for exploring NYC Open Data using AI-powered data discovery and relationship analysis.

## 🎯 First Time Here?

Run the automated setup script:
```bash
./setup.sh
```

This will check your system, install dependencies, and get you started in minutes!

## 🚀 Quick Start

### Deployment Options

Choose your preferred deployment method:

#### Option 1: Simple Run Script (Recommended)
```bash
# Quick start with automated setup
./run.sh

# With optimization
./run.sh --optimize
```

#### Option 2: Standard Startup Script
```bash
# Direct startup
./start_scout.sh
```

#### Option 3: Docker Deployment
```bash
# Using Docker Compose
docker-compose up -d

# Access the application
# Frontend: http://localhost:8501
# Backend: http://localhost:8080
```

See [Docker Deployment Guide](DOCKER_DEPLOYMENT.md) for detailed Docker instructions.

### Access the Application
- **Frontend (Main UI)**: http://localhost:8501
- **Backend API**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs

## 📋 Deployment Scripts

### Main Scripts

**`run.sh`** - Simplified deployment script
```bash
./run.sh              # Start services
./run.sh --optimize   # Start with optimization
```

**`start_scout.sh`** - Full-featured startup script
```bash
./start_scout.sh           # Start services
./start_scout.sh --status  # Check status
./start_scout.sh --stop    # Stop services
./start_scout.sh --help    # Show help
```

### Optimization Scripts

Located in `scripts/` directory:

**`optimize-deps.sh`** - Manage dependencies
```bash
./scripts/optimize-deps.sh                  # Check/install dependencies
./scripts/optimize-deps.sh --check-outdated # Show outdated packages
./scripts/optimize-deps.sh --clean-cache    # Clean pip cache
```

**`optimize-cache.sh`** - Manage cache
```bash
./scripts/optimize-cache.sh --clean-old    # Clean old files
./scripts/optimize-cache.sh --clean-all    # Clean all cache
./scripts/optimize-cache.sh --stats        # Show statistics
```

**`health-check.sh`** - Verify services
```bash
./scripts/health-check.sh              # Basic health check
./scripts/health-check.sh --verbose    # Detailed output
./scripts/health-check.sh --check-deps # Check dependencies
```

See [Scripts Documentation](scripts/README.md) for detailed usage.

## 🏗️ Architecture

```
Innovate-tech-civic-ai-powered-research/
├── scout_data_discovery/     # Core Scout library
│   ├── src/                  # Scout modules
│   ├── examples/             # Usage examples & notebooks
│   ├── tests/                # Unit tests
│   └── README.md
├── backend/                  # FastAPI backend service
│   ├── main.py              # API endpoints
│   ├── api/                 # Organized API modules
│   │   ├── models.py        # Pydantic models
│   │   ├── routes/          # Route handlers
│   │   ├── services/        # Business logic
│   │   └── utils/           # Helper functions
│   ├── cache_manager.py     # Caching system
│   ├── api_config.py        # Configuration
│   ├── run_server.py        # Server runner
│   └── requirements.txt
├── frontend/                 # Streamlit web application
│   ├── app.py               # Main Streamlit app
│   ├── components/          # UI components
│   │   ├── backend_manager.py
│   │   ├── ai_analyst_component.py
│   │   └── visualization_utils.py  # Optimized visualizations
│   ├── run_app.py           # App runner
│   └── requirements.txt
├── AI_Functionality/         # AI analysis modules
│   ├── core/                # Core AI components
│   ├── providers/           # AI provider integrations
│   └── docs/                # AI documentation
├── scripts/                 # Deployment & optimization scripts
│   ├── optimize-deps.sh    # Dependency management
│   ├── optimize-cache.sh   # Cache optimization
│   ├── health-check.sh     # Service health checks
│   └── README.md           # Scripts documentation
├── docs/                    # General documentation
│   ├── QUICK_START_GUIDE.md
│   ├── USAGE.md
│   ├── PERFORMANCE_OPTIMIZATION_GUIDE.md
│   └── BACKEND_ORGANIZATION.md
├── transcripts/             # Implementation history
│   └── [various implementation summaries]
├── subrepos/                # External integrations
│   └── n8n-workflows/
├── run.sh                   # Simplified deployment script
├── start_scout.sh           # Full-featured startup script
├── docker-compose.yml       # Docker orchestration
├── Dockerfile.backend       # Backend container image
├── Dockerfile.frontend      # Frontend container image
├── DOCKER_DEPLOYMENT.md     # Docker deployment guide
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda for package management

### 1. Install Dependencies

```bash
# Install Scout Data Discovery core
cd scout_data_discovery
pip install -r requirements.txt

# Install backend dependencies
cd ../backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
pip install -r requirements.txt
```

### 2. Launch the Platform

```bash
# Start from the frontend directory
cd frontend
python run_app.py
```

The web app will be available at: http://localhost:8501

### 🎯 Integrated Backend Management

The backend API server is now **managed directly from the frontend**:

- ✅ **Automatic Start**: Click "🚀 Start Backend Server" in the web interface
- ✅ **Real-time Status**: Backend status shown in the sidebar
- ✅ **Easy Control**: Start/stop backend from the UI
- ✅ **Health Monitoring**: Automatic health checks and status updates

No need to manually start the backend - everything is controlled from the web interface!

## ✨ Features

### 🔍 Scout Data Discovery Library
- **Dataset Discovery**: Search across NYC Open Data with intelligent filtering
- **Quality Assessment**: 5-dimensional quality scoring (completeness, consistency, accuracy, timeliness, usability)
- **Relationship Analysis**: AI-powered discovery of dataset connections
- **Network Visualization**: Interactive graphs showing dataset relationships
- **Export Capabilities**: Multiple formats for further analysis

### 🌐 Web Interface
- **Interactive Dashboard**: Overview of recently updated datasets
- **Advanced Search**: Multi-term search with filtering and sorting
- **Quality Assessment**: Visual quality scoring with detailed breakdowns
- **Relationship Mapping**: Network graphs showing dataset connections
- **Data Sample Viewer**: Preview dataset contents before download
- **Export Options**: Download samples and analysis results

### 🔌 API Backend
- **RESTful API**: Clean endpoints for all Scout functionality
- **Async Processing**: Background tasks for long-running operations
- **Caching**: Intelligent caching for improved performance
- **CORS Support**: Cross-origin requests from frontend
- **Auto Documentation**: Swagger/OpenAPI docs at `/docs`

## 📊 Use Cases

### For Data Analysts
- **Discover Related Datasets**: Find complementary data sources
- **Assess Data Quality**: Make informed decisions about data usage
- **Integration Planning**: Identify datasets suitable for joining

### For Researchers
- **Literature Discovery**: Find relevant datasets for research
- **Data Landscape Analysis**: Understand available data domains
- **Quality Benchmarking**: Compare data quality across sources

### For City Officials
- **Data Inventory**: Comprehensive view of available open data
- **Usage Insights**: Understand popular datasets and trends
- **Quality Monitoring**: Track data quality across departments

## 🛠️ API Endpoints

### Dataset Operations
- `GET /api/datasets/top-updated` - Recently updated datasets
- `POST /api/datasets/search` - Search datasets
- `GET /api/datasets/{id}/quality` - Quality assessment
- `GET /api/datasets/{id}/sample` - Data sample

### Relationship Analysis
- `POST /api/datasets/relationships` - Find related datasets
- `GET /api/network/visualization/{id}` - Network graph data

### System
- `GET /api/stats` - API usage statistics
- `GET /api/categories` - Available dataset categories

## 🔧 Configuration

### Backend Configuration
Edit `backend/main.py` for API settings:
- Rate limiting
- Cache duration
- Sample sizes
- Worker threads

### Frontend Configuration
Edit `frontend/app.py` for UI settings:
- API endpoints
- Cache TTL
- Display options
- Visualization parameters

## 📈 Performance

### Recent Optimizations
- **Visualization Caching**: Charts cached for 10 minutes (60% faster)
- **Optimized Table Rendering**: 80% improvement for large datasets
- **Efficient Network Graphs**: 75% faster with 50% less memory
- **Smart Data Limiting**: Prevents browser slowdown
- See [Performance Optimization Guide](docs/PERFORMANCE_OPTIMIZATION_GUIDE.md) for details

### Core Features
- **Intelligent Caching**: Results cached with TTL management
- **Background Processing**: Long operations run asynchronously
- **Progressive Loading**: Large datasets loaded in chunks
- **Connection Pooling**: Efficient API connections

### Scalability
- **Stateless Design**: Easy horizontal scaling
- **Organized Backend**: Modular structure ready for growth
- **Database Ready**: Can integrate with PostgreSQL/Redis
- **Container Ready**: Full Docker support with docker-compose orchestration
- **Deployment Scripts**: Automated setup and optimization scripts

## 🧪 Development

### Running Tests
```bash
# Scout library tests
cd scout_data_discovery
python -m pytest tests/ -v

# Backend tests (if implemented)
cd backend
python -m pytest tests/ -v
```

### Development Mode
Both frontend and backend support hot reload:

```bash
# Backend (auto-reload on code changes)
cd backend
uvicorn main:app --reload

# Frontend (auto-reload on code changes)
cd frontend
streamlit run app.py
```

## 📚 Examples

### Using the Scout Library
```python
from src.scout_discovery import ScoutDataDiscovery

# Initialize Scout
scout = ScoutDataDiscovery(log_level="INFO")

# Search for datasets
datasets = scout.search_datasets(["311", "health"])

# Assess quality
quality = scout.assess_dataset_quality(dataset_id)

# Find relationships
from src.dataset_relationship_graph import DatasetRelationshipGraph
graph = DatasetRelationshipGraph()
graph.add_datasets(datasets)
relationships = graph.calculate_relationships()
```

### Using the API
```python
import requests

# Search datasets
response = requests.post("http://localhost:8000/api/datasets/search",
                        json={"search_terms": ["311"], "limit": 10})
datasets = response.json()

# Get quality assessment
quality = requests.get(f"http://localhost:8000/api/datasets/{dataset_id}/quality")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Test thoroughly
5. Submit a pull request

## 📚 Documentation

Comprehensive documentation is available in the `/docs` folder:

- **[Quick Start Guide](docs/QUICK_START_GUIDE.md)** - Get up and running quickly
- **[Usage Guide](docs/USAGE.md)** - Detailed usage instructions
- **[Performance Optimization Guide](docs/PERFORMANCE_OPTIMIZATION_GUIDE.md)** - Optimization strategies
- **[Backend Organization](docs/BACKEND_ORGANIZATION.md)** - Backend code structure
- **[Scout Technical Analysis](docs/Scout_Technical_Analysis.md)** - Deep dive into Scout methodology

Implementation history and change logs are in `/transcripts`.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on [Scout](https://scout.tsdataclinic.com/) methodology
- Powered by [NYC Open Data](https://opendata.cityofnewyork.us/)
- Uses [Socrata Discovery API](https://dev.socrata.com/)

---

For detailed usage examples, see the `scout_data_discovery/examples/` directory.