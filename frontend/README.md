# Scout Data Discovery Frontend

A modern, interactive Streamlit web application that provides an intuitive interface for exploring NYC Open Data using the Scout Data Discovery methodology. This frontend connects seamlessly with the Scout Data Discovery backend API to deliver AI-powered data exploration, quality assessment, and relationship mapping capabilities.

## 🎨 User Interface Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Scout Data Explorer                                   │
│                    🔍 Discover, analyze, and explore NYC Open Data             │
├─────────────────────┬───────────────────────────────────────────────────────────┤
│                     │                                                         │
│    SIDEBAR          │                    MAIN CONTENT AREA                     │
│                     │                                                         │
│  🏠 Dashboard       │  ┌─────────────────────────────────────────────────────┐ │
│  🔍 Dataset Explorer│  │                                                     │ │
│  📊 Quality Assess. │  │              Dynamic Content                        │ │
│  🗺️  Relationship Map│  │           Based on Selected Page                   │ │
│  📋 Data Sample     │  │                                                     │ │
│                     │  │  • Dashboard: Overview & Top Datasets              │ │
│  Backend Status:    │  │  • Explorer: Search & Filter Interface             │ │
│  🟢 Backend Online  │  │  • Quality: Multi-dimensional Assessment           │ │
│  ✅ Health Pass     │  │  • Mapping: Network Visualization                  │ │
│  🔄 Process Active  │  │  • Sample: Data Preview & Analysis                 │ │
│                     │  │                                                     │ │
│  🚀 Start Backend   │  └─────────────────────────────────────────────────────┘ │
│  🛑 Stop Backend    │                                                         │
│                     │              🤖 AI Analyst Panel                       │
│  🤖 AI Configuration│  ┌─────────────────────────────────────────────────────┐ │
│                     │  │  AI-powered insights and Q&A for each page         │ │
└─────────────────────┴──┴─────────────────────────────────────────────────────┴─┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Backend API running (managed automatically via UI)

### Installation & Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd /path/to/QLT_Workshop/frontend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application:**
   ```bash
   python run_app.py
   ```

4. **Access the application:**
   - **Frontend UI**: http://localhost:8501
   - **Backend API** (auto-managed): http://localhost:8080

## 🏗️ Application Architecture

### Core Components

#### 1. **Main Application** (`app.py`)
- **Streamlit Framework**: Modern web UI with responsive design
- **Page Navigation**: Multi-page application with sidebar navigation
- **State Management**: Session-based state persistence across pages
- **API Integration**: RESTful communication with Scout backend
- **Real-time Updates**: Dynamic content refresh and status monitoring

#### 2. **Backend Manager** (`components/backend_manager.py`)
- **Process Management**: Automatic backend server lifecycle management
- **Health Monitoring**: Real-time backend status and health checks
- **Start/Stop Controls**: One-click backend server control from UI
- **Connection Validation**: Automatic reconnection and error handling

#### 3. **AI Analyst Component** (`components/ai_analyst_component.py`)
- **AI Integration**: Multi-provider AI analysis capabilities
- **Context-Aware Analysis**: Page-specific AI insights and recommendations
- **Interactive Q&A**: Natural language querying of dataset information
- **Configurable Providers**: Support for OpenAI, OpenRouter, and NVIDIA APIs

## 📱 Page Features & Functionality

### 🏠 **Dashboard Page**
The central hub providing overview and quick access to key datasets.

**Features:**
- **Key Metrics Display**: Total datasets, downloads, categories, recent updates
- **Category Distribution**: Interactive pie charts showing dataset categories
- **Top Datasets Table**: Recently updated datasets with sorting/filtering
- **Quick Analysis**: One-click dataset selection for detailed exploration
- **AI Quick Insights**: Context-aware AI analysis of selected datasets

**Scout Integration:**
- `GET /api/datasets/top-updated` - Retrieves recently updated datasets
- Real-time metrics calculation and visualization
- Automatic data refresh with configurable caching (5 minutes TTL)

### 🔍 **Dataset Explorer**
Advanced search and discovery interface for finding relevant datasets.

**Features:**
- **Intelligent Search**: Multi-term search with relevance ranking
- **Dynamic Filtering**: Category, download count, and date filters
- **Flexible Sorting**: Sort by name, popularity, or recency
- **Dataset Cards**: Rich preview cards with key information
- **Batch Operations**: Select multiple datasets for comparison

**Scout Integration:**
- `POST /api/datasets/search` - Powered by Scout's enhanced search algorithms
- Leverages Scout's relationship mapping for search result ranking
- Integrated with Scout's metadata enrichment

**Search Parameters:**
```python
search_data = {
    "search_terms": ["311", "complaints", "housing"],
    "limit": 20,
    "include_quality": False  # Performance optimization
}
```

### 📊 **Quality Assessment**
Comprehensive multi-dimensional quality analysis using Scout methodology.

**Features:**
- **Overall Quality Score**: 0-100 scale with letter grades (A-F)
- **Dimensional Breakdown**: 5-axis quality assessment
  - **Completeness**: Missing data analysis
  - **Consistency**: Data type and format validation
  - **Accuracy**: Outlier detection and range validation
  - **Timeliness**: Freshness and update frequency analysis
  - **Usability**: Structure and documentation quality
- **Interactive Gauges**: Real-time quality score visualizations
- **Actionable Insights**: Specific recommendations for data improvement
- **Quality Trends**: Historical quality tracking

**Scout Integration:**
- `GET /api/datasets/{id}/quality` - Scout's advanced quality assessment engine
- Multi-threaded analysis for performance optimization
- Comprehensive quality metrics with detailed explanations

**Quality Visualization:**
```python
def create_quality_gauge(score, title="Quality Score"):
    """Interactive gauge showing quality score with color-coded ranges"""
    # Green: 80-100, Yellow: 50-80, Red: 0-50
```

### 🗺️ **Relationship Mapping**
Network analysis and visualization of dataset relationships.

**Features:**
- **Similarity Analysis**: Multi-factor similarity scoring
- **Network Visualization**: Interactive relationship graphs using Plotly
- **Relationship Reasons**: Detailed explanations for each connection
- **Configurable Thresholds**: Adjustable similarity sensitivity
- **Network Statistics**: Graph density, clustering coefficients
- **Related Dataset Discovery**: Find datasets similar to your selection

**Scout Integration:**
- `POST /api/datasets/relationships` - Scout's relationship analysis engine
- `GET /api/network/visualization/{id}` - Network graph data
- Powered by Scout's `DatasetRelationshipGraph` component

**Network Configuration:**
```python
relationship_data = {
    "dataset_id": dataset['id'],
    "similarity_threshold": 0.3,  # Adjustable via UI slider
    "max_related": 10            # User-configurable limit
}
```

### 📋 **Data Sample**
Interactive data preview and exploration interface.

**Features:**
- **Configurable Sampling**: User-defined sample sizes (10-5000 rows)
- **Column Information**: Data types, sample values, and statistics
- **Interactive Data Grid**: Sortable, filterable data table
- **CSV Export**: Download samples for external analysis
- **Data Profiling**: Basic statistical summaries
- **Missing Data Visualization**: Visual representation of data completeness

**Scout Integration:**
- `GET /api/datasets/{id}/sample` - Scout's intelligent sampling engine
- Optimized data retrieval with automatic type inference
- Statistical profiling and metadata extraction

## 🎛️ Advanced Features

### 🔧 **Backend Management System**

**Automatic Backend Control:**
- **Health Monitoring**: Continuous backend status checking
- **Auto-Start Capability**: One-click backend server deployment
- **Process Management**: Graceful start/stop with proper cleanup
- **Connection Recovery**: Automatic reconnection on network issues
- **Status Dashboard**: Real-time process and health indicators

**Status Indicators:**
```
🟢 Backend Online    - API server responding
✅ Health Check Pass - All systems operational  
🔄 Process Active    - Background processes running
🔴 Backend Offline   - Service unavailable
```

### 🤖 **AI Analyst Integration**

**Multi-Provider Support:**
- **OpenAI GPT Models**: GPT-4, GPT-3.5-turbo
- **OpenRouter**: Access to multiple model providers
- **NVIDIA NIM**: Enterprise-grade AI inference

**Context-Aware Analysis:**
- **Page-Specific Insights**: Tailored analysis based on current view
- **Dataset Intelligence**: Deep understanding of data structure and content
- **Interactive Q&A**: Natural language queries about datasets
- **Automated Insights**: Proactive recommendations and findings

**AI Configuration:**
```python
# Configurable via Streamlit secrets or session state
config = {
    'openai_api_key': 'your_openai_key',
    'openrouter_api_key': 'your_openrouter_key', 
    'nvidia_api_key': 'your_nvidia_key'
}
```

### 📊 **Advanced Visualizations**

**Interactive Charts:**
- **Plotly Integration**: Dynamic, responsive charts and graphs
- **Quality Gauges**: Real-time quality score indicators
- **Network Graphs**: Interactive relationship visualizations
- **Distribution Charts**: Category and metric distributions
- **Time Series**: Temporal data analysis and trending

**Visualization Examples:**
```python
# Quality gauge with color-coded ranges
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=quality_score,
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "darkblue"},
           'steps': [{'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}]}
))
```

## 🔗 Scout Backend Integration

### API Communication Pattern

```python
# Standardized API call with error handling and caching
@st.cache_data(ttl=CACHE_TTL)
def fetch_api_data(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Fetch data from Scout API with automatic error handling"""
    url = f"{API_BASE_URL}/{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {}
```

### Scout Service Integration Points

| Frontend Feature | Backend Endpoint | Scout Component |
|-------------------|------------------|-----------------|
| Dashboard metrics | `/api/datasets/top-updated` | `ScoutDataDiscovery.search_datasets()` |
| Dataset search | `/api/datasets/search` | `ScoutDataDiscovery.search_datasets()` |
| Quality assessment | `/api/datasets/{id}/quality` | `ScoutDataDiscovery.assess_dataset_quality()` |
| Relationship mapping | `/api/datasets/relationships` | `DatasetRelationshipGraph` |
| Data sampling | `/api/datasets/{id}/sample` | `ScoutDataDiscovery.download_dataset_sample()` |
| Network visualization | `/api/network/visualization/{id}` | `DatasetRelationshipGraph` |
| API statistics | `/api/stats` | `ScoutDataDiscovery.get_api_statistics()` |

### Data Flow Architecture

```
User Interaction → Streamlit Frontend → FastAPI Backend → Scout Engine → NYC Open Data
     ↑                    ↓                   ↓              ↓             ↓
UI Updates    ←    JSON Response    ←    API Response  ←  Processed Data  ←  Raw Data
```

## 🎨 User Experience Features

### 📱 **Responsive Design**
- **Wide Layout**: Optimized for desktop and large screens
- **Flexible Columns**: Adaptive column layouts for different content
- **Mobile-Friendly**: Responsive components that work on smaller screens
- **Sidebar Navigation**: Persistent navigation with collapsible sections

### 🎯 **Performance Optimizations**
- **Smart Caching**: 5-minute TTL on API calls with `@st.cache_data`
- **Lazy Loading**: On-demand data fetching for improved performance
- **Async Processing**: Non-blocking API calls where possible
- **Progress Indicators**: Real-time feedback during long operations

### 🔄 **State Management**
```python
# Session state for cross-page data persistence
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'quality_cache' not in st.session_state:
    st.session_state.quality_cache = {}
```

### 🎨 **Custom Styling**
```css
/* Quality score color coding */
.quality-score-excellent { color: #28a745; }  /* Green: 90+ */
.quality-score-good { color: #17a2b8; }       /* Blue: 80-89 */
.quality-score-fair { color: #ffc107; }       /* Yellow: 70-79 */
.quality-score-poor { color: #dc3545; }       /* Red: <70 */

/* Dataset cards with subtle shadows */
.dataset-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
```

## ⚙️ Configuration & Setup

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `STREAMLIT_SERVER_PORT` | Frontend port | `8501` | No |
| `BACKEND_URL` | Backend API URL | `http://localhost:8080` | No |
| `CACHE_TTL` | Cache duration (seconds) | `300` | No |
| `OPENAI_API_KEY` | OpenAI API key for AI features | None | Optional |
| `OPENROUTER_API_KEY` | OpenRouter API key | None | Optional |
| `NVIDIA_API_KEY` | NVIDIA NIM API key | None | Optional |

### Streamlit Configuration

The app uses the following Streamlit configuration:

```python
st.set_page_config(
    page_title="Scout Data Explorer",
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Dependencies Management

**Core Dependencies:**
- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation and analysis
- `plotly>=5.15.0` - Interactive visualizations
- `requests>=2.31.0` - HTTP client for API communication

**Visualization Stack:**
- `matplotlib>=3.6.0` - Static plotting
- `seaborn>=0.12.0` - Statistical visualizations
- `altair>=5.0.0` - Declarative visualizations
- `networkx>=3.0` - Network analysis and visualization

**UI Components:**
- `streamlit-option-menu>=0.3.6` - Enhanced navigation menus
- `streamlit-folium>=0.15.0` - Interactive maps (future feature)

## 🧪 Development & Testing

### Development Server

```bash
# Standard development mode
streamlit run app.py --server.port=8501

# Or use the run script
python run_app.py

# Development with auto-reload
streamlit run app.py --server.runOnSave=true
```

### Adding New Pages

1. **Create page function** in `app.py`:
   ```python
   def show_new_page(ai_analyst):
       """New page implementation"""
       st.header("🆕 New Page")
       # Page content here
   ```

2. **Add to navigation** in sidebar:
   ```python
   selected = option_menu(
       menu_title="Navigation",
       options=["Dashboard", "Dataset Explorer", "New Page"],
       icons=["house", "search", "star"],
       default_index=0
   )
   ```

3. **Add routing** in main function:
   ```python
   elif selected == "New Page":
       show_new_page(ai_analyst)
   ```

### Component Development

**Backend Manager Extension:**
```python
class BackendManager:
    def add_new_feature(self):
        """Extend backend management capabilities"""
        pass
```

**AI Analyst Enhancement:**
```python
class AIAnalystComponent:
    def add_analysis_type(self, analysis_type):
        """Add new AI analysis capabilities"""
        pass
```

## 🔧 Troubleshooting

### Common Issues

#### 1. **Backend Connection Failed**
```
❌ Error: Connection refused to http://localhost:8080
```
**Solutions:**
- Click "🚀 Start Backend" in the sidebar
- Manually start backend: `cd backend && python run_server.py`
- Check port conflicts and firewall settings

#### 2. **Module Import Errors**
```
ModuleNotFoundError: No module named 'streamlit_option_menu'
```
**Solutions:**
- Reinstall requirements: `pip install -r requirements.txt`
- Check virtual environment activation
- Verify Python version compatibility (3.8+)

#### 3. **AI Features Unavailable**
```
AI Functionality not available: No module named 'AI_Functionality'
```
**Solutions:**
- Ensure AI_Functionality directory exists in parent directory
- Install AI dependencies if available
- AI features are optional and app works without them

#### 4. **Slow Loading Performance**
**Optimizations:**
- Reduce cache TTL for development: `CACHE_TTL = 60`
- Decrease default sample sizes
- Use smaller dataset limits for testing

### Performance Monitoring

**Built-in Metrics:**
- API response times displayed in UI
- Cache hit/miss ratios in browser console
- Backend health check frequency

**Development Tools:**
```python
# Enable debug mode
if st.checkbox("Debug Mode"):
    st.write("Session State:", st.session_state)
    st.write("Cache Info:", st.cache_data.clear())
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup:**
   ```bash
   # Production requirements
   pip install -r requirements.txt
   
   # Set production environment variables
   export STREAMLIT_SERVER_PORT=8501
   export BACKEND_URL=https://your-backend-domain.com
   ```

2. **Streamlit Configuration** (`.streamlit/config.toml`):
   ```toml
   [server]
   port = 8501
   address = "0.0.0.0"
   
   [browser]
   serverAddress = "your-domain.com"
   serverPort = 8501
   ```

3. **SSL/TLS Setup:**
   ```bash
   # Use reverse proxy (nginx/traefik) for SSL termination
   # Configure HTTPS redirects and security headers
   ```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔮 Future Enhancements

### Planned Features

1. **Enhanced Visualizations**
   - Interactive geospatial maps using Folium
   - Advanced time series analysis
   - Custom dashboard creation

2. **User Management**
   - Authentication and user profiles
   - Saved searches and favorites
   - Collaborative analysis features

3. **Advanced AI Features**
   - Multi-language support for AI analysis
   - Custom AI model integration
   - Automated report generation

4. **Performance Improvements**
   - WebSocket connections for real-time updates
   - Progressive data loading
   - Advanced caching strategies

### API Enhancements

1. **Streaming Updates**
   - Real-time data updates via WebSockets
   - Live quality assessment progress
   - Dynamic relationship graph updates

2. **Bulk Operations**
   - Multi-dataset quality assessment
   - Batch relationship analysis
   - Parallel data processing

## 📞 Support & Contributing

### Development Guidelines

- Follow Streamlit best practices for component design
- Maintain compatibility with Scout backend API changes
- Add comprehensive error handling for all user interactions
- Update documentation when adding new features

### Code Style

```python
# Function documentation
def show_page_function(ai_analyst):
    """
    Page function following standard pattern
    
    Args:
        ai_analyst: AI analysis component for the page
    """
    st.header("📊 Page Title")
    # Implementation here
```

### Testing

```python
# Manual testing checklist
def test_page_functionality():
    """Test all page features manually"""
    # 1. Navigation works
    # 2. API calls succeed
    # 3. Error handling works
    # 4. UI responds correctly
    pass
```

---

**Related Documentation:**
- [Backend API Documentation](../backend/README.md)
- [Scout Data Discovery README](../scout_data_discovery/README.md)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Component Documentation](./components/README.md) (if available)

**Quick Links:**
- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/