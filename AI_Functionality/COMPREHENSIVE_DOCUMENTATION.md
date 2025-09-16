# AI Functionality - Comprehensive Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Providers](#providers)
5. [Usage Examples](#usage-examples)
6. [Configuration](#configuration)
7. [Advanced Features](#advanced-features)
8. [Integration Guide](#integration-guide)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## 🎯 Overview

The **AI_Functionality** module provides a comprehensive, multi-provider AI analysis system for the Scout Data Discovery platform. It enables intelligent data analysis, automated insights generation, and AI-powered Q&A capabilities across multiple AI providers.

### Key Features

- **Multi-Provider Support**: OpenAI GPT models, OpenRouter, NVIDIA NIM
- **Advanced Caching**: Two-tier caching system (prompt-based and semantic)
- **Automated Insights**: AI-powered insight generation engine
- **Codebase Analysis**: Intelligent code analysis and documentation
- **Unified Management**: Single interface for all AI operations
- **Performance Optimization**: Timeout handling, retry mechanisms, background processing

### Use Cases

- **Data Analysis**: Automated quality assessment and trend analysis
- **Insight Generation**: Periodic analysis and recommendation generation
- **Interactive Q&A**: Natural language queries about datasets
- **Code Documentation**: Automated codebase analysis and documentation
- **Platform Monitoring**: AI-powered system health analysis

---

## 🏗️ Architecture

```
AI_Functionality/
├── core/                           # Core AI functionality
│   ├── ai_analyst.py              # Main AI analyst orchestrator
│   ├── base_provider.py           # Provider interface and contracts
│   ├── cache_manager.py           # Advanced caching system
│   ├── codebase_agent.py          # Code analysis and documentation
│   ├── insights_engine.py         # Automated insight generation
│   └── unified_ai_manager.py      # Unified AI operations manager
├── providers/                      # AI provider implementations
│   ├── openai_provider.py         # OpenAI GPT integration
│   ├── openrouter_provider.py     # OpenRouter multi-model access
│   └── nvidia_provider.py         # NVIDIA NIM integration
├── utils/                          # Utility functions (empty)
├── cache/                          # Cache storage directory
├── requirements.txt                # Python dependencies
├── README.md                       # Basic documentation
└── __init__.py                     # Module initialization
```

### Design Principles

1. **Provider Abstraction**: Unified interface for different AI providers
2. **Extensibility**: Easy to add new providers and capabilities
3. **Performance First**: Advanced caching and optimization
4. **Fault Tolerance**: Graceful fallbacks and error handling
5. **Configuration Driven**: Flexible configuration management

---

## 🧠 Core Components

### 1. AI Analyst (`core/ai_analyst.py`)

**Purpose**: Main orchestrator for AI-powered data analysis

**Key Features**:
- Multi-provider management with fallbacks
- Analysis type routing (overview, quality, insights, relationships)
- Interactive Q&A capabilities
- Advanced caching integration
- Performance monitoring

**Core Classes**:
```python
class DataAnalyst:
    """Main AI analyst for data analysis and Q&A"""
    
    def __init__(self, primary_provider, fallback_providers, **config):
        """Initialize with provider configuration"""
    
    async def analyze_dataset(self, dataset_info, sample_data, analysis_type):
        """Perform comprehensive dataset analysis"""
    
    async def answer_question(self, question, dataset_info, context):
        """Answer natural language questions about data"""
    
    async def generate_insights(self, data_context, insight_type):
        """Generate AI-powered insights"""
```

**Analysis Types**:
- `OVERVIEW`: General dataset analysis and summary
- `QUALITY`: Data quality assessment and recommendations
- `INSIGHTS`: Deep insights and pattern detection
- `RELATIONSHIPS`: Inter-dataset relationship analysis
- `CUSTOM`: User-defined analysis types

### 2. Base Provider (`core/base_provider.py`)

**Purpose**: Abstract interface for AI providers

**Key Components**:
```python
@dataclass
class AIRequest:
    """Standardized AI request format"""
    prompt: str
    system_message: Optional[str]
    temperature: float
    max_tokens: int
    model: Optional[str]
    metadata: Dict[str, Any]

@dataclass  
class AIResponse:
    """Standardized AI response format"""
    content: str
    model: str
    provider: str
    tokens_used: int
    cost_estimate: float
    processing_time: float
    cached: bool

class BaseAIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate AI response from request"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
```

### 3. Cache Manager (`core/cache_manager.py`)

**Purpose**: Advanced two-tier caching system

**Caching Strategies**:
1. **Prompt-based Caching**: Exact prompt matching with TTL
2. **Semantic Caching**: Similar content detection using embeddings
3. **Response Optimization**: Intelligent cache eviction

**Configuration**:
```python
class CacheManager:
    def __init__(self, 
                 cache_dir: str = "./cache",
                 enable_semantic: bool = True,
                 default_ttl: int = 3600,
                 max_cache_size: int = 1000):
```

**Cache Types**:
- **Memory Cache**: Fast in-memory storage for recent queries
- **Disk Cache**: Persistent storage for long-term caching
- **Semantic Cache**: Embedding-based similarity matching

### 4. Insights Engine (`core/insights_engine.py`)

**Purpose**: Automated AI-powered insight generation

**Insight Types**:
```python
class InsightType(Enum):
    TREND_ANALYSIS = "trend_analysis"
    USAGE_PATTERNS = "usage_patterns"
    DATA_QUALITY_SHIFTS = "data_quality_shifts"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE = "predictive"
    COMPARATIVE = "comparative"
```

**Priority Levels**:
```python
class InsightPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

**Core Methods**:
```python
async def generate_dataset_insights(self, dataset_info, sample_data, analysis_history)
async def generate_platform_insights(self, usage_statistics, user_patterns, system_health)
def get_insights(self, insight_type=None, priority=None, dataset_id=None, limit=10)
def get_insight_summary(self)
```

### 5. Codebase Agent (`core/codebase_agent.py`)

**Purpose**: Intelligent code analysis and documentation

**Capabilities**:
- Code structure analysis
- Documentation generation
- Code quality assessment
- Architecture understanding
- Automated code reviews

### 6. Unified AI Manager (`core/unified_ai_manager.py`)

**Purpose**: Single interface for all AI operations

**Features**:
- Provider lifecycle management
- Request routing and load balancing
- Health monitoring
- Configuration management
- Performance metrics

---

## 🔌 Providers

### 1. OpenAI Provider (`providers/openai_provider.py`)

**Supported Models**:
- `gpt-4o`: Latest GPT-4 Omni model
- `gpt-4o-mini`: Lightweight GPT-4 variant
- `gpt-4-turbo`: High-performance GPT-4
- `gpt-3.5-turbo`: Cost-effective option

**Configuration**:
```python
{
    "api_key": "your_openai_api_key",
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 2000,
    "timeout": 30
}
```

**Features**:
- Function calling support
- Streaming responses
- Token usage tracking
- Cost estimation

### 2. OpenRouter Provider (`providers/openrouter_provider.py`)

**Supported Models**:
- `anthropic/claude-3.5-sonnet`
- `google/gemini-pro-1.5`
- `meta-llama/llama-3.1-405b-instruct`
- `mistralai/mixtral-8x22b-instruct`
- And 100+ other models

**Configuration**:
```python
{
    "api_key": "your_openrouter_api_key", 
    "model": "anthropic/claude-3.5-sonnet",
    "site_url": "https://your-app.com",
    "app_name": "Scout Data Discovery"
}
```

**Benefits**:
- Access to multiple AI providers
- Competitive pricing
- Model diversity
- Failover capabilities

### 3. NVIDIA Provider (`providers/nvidia_provider.py`)

**Supported Models**:
- `qwen/qwen2.5-72b-instruct`: Recommended for reasoning
- `meta/llama-3.1-405b-instruct`: Largest model
- `meta/llama-3.1-70b-instruct`: Balanced performance
- `mistralai/mixtral-8x22b-instruct`: Mixture of experts

**Configuration**:
```python
{
    "api_key": "your_nvidia_api_key",
    "model": "qwen/qwen2.5-72b-instruct",
    "base_url": "https://integrate.api.nvidia.com/v1"
}
```

**Features**:
- Enterprise-grade performance
- Local deployment options
- Specialized models
- High throughput

---

## 🚀 Usage Examples

### Basic AI Analysis

```python
from AI_Functionality.core.ai_analyst import DataAnalyst, AnalysisType

# Initialize analyst
analyst = DataAnalyst(
    primary_provider="openai",
    fallback_providers=["openrouter", "nvidia"],
    openai_api_key="your_key",
    openrouter_api_key="your_key",
    nvidia_api_key="your_key"
)

# Analyze a dataset
dataset_info = {
    "id": "nyc-311-data",
    "name": "NYC 311 Service Requests",
    "description": "All 311 service requests from 2010 to present"
}

sample_data = [
    {"complaint_type": "Noise", "borough": "Manhattan", "status": "Open"},
    {"complaint_type": "Heat", "borough": "Brooklyn", "status": "Closed"}
]

# Perform analysis
response = await analyst.analyze_dataset(
    dataset_info=dataset_info,
    sample_data=sample_data,
    analysis_type=AnalysisType.OVERVIEW
)

print(f"Analysis: {response.content}")
print(f"Model used: {response.model}")
print(f"Cached: {response.cached}")
```

### Interactive Q&A

```python
# Ask questions about the dataset
question = "What are the most common complaint types in this dataset?"

response = await analyst.answer_question(
    question=question,
    dataset_info=dataset_info,
    context={
        "sample_data": sample_data,
        "analysis_type": "interactive"
    }
)

print(f"Answer: {response.content}")
```

### Insights Generation

```python
from AI_Functionality.core.insights_engine import InsightsEngine

# Initialize insights engine
insights_engine = InsightsEngine(
    ai_analyst=analyst,
    cache_dir="./insights_cache",
    insights_storage_dir="./insights_storage"
)

# Generate dataset insights
insights = await insights_engine.generate_dataset_insights(
    dataset_info=dataset_info,
    sample_data=sample_data,
    analysis_history=[]
)

for insight in insights:
    print(f"Insight: {insight.title}")
    print(f"Priority: {insight.priority.value}")
    print(f"Content: {insight.content}")
    print("---")
```

### Multi-Provider Configuration

```python
# Configure multiple providers with fallbacks
config = {
    "primary_provider": "openai",
    "fallback_providers": ["openrouter", "nvidia"],
    "provider_configs": {
        "openai": {
            "api_key": "sk-...",
            "model": "gpt-4o",
            "temperature": 0.1
        },
        "openrouter": {
            "api_key": "sk-or-v1-...",
            "model": "anthropic/claude-3.5-sonnet",
            "site_url": "https://scout-data-discovery.com"
        },
        "nvidia": {
            "api_key": "nvapi-...",
            "model": "qwen/qwen2.5-72b-instruct"
        }
    }
}

analyst = DataAnalyst(**config)
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Provider API Keys
OPENAI_API_KEY=sk-your-openai-key
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key
NVIDIA_API_KEY=nvapi-your-nvidia-key

# Cache Configuration
AI_CACHE_DIR=./ai_cache
AI_CACHE_TTL=3600
AI_CACHE_MAX_SIZE=1000

# Performance Settings
AI_REQUEST_TIMEOUT=30
AI_MAX_RETRIES=3
AI_RETRY_DELAY=1.0
```

### Configuration File

```yaml
# config/ai_config.yaml
ai:
  primary_provider: "openai"
  fallback_providers: ["openrouter", "nvidia"]
  
  providers:
    openai:
      model: "gpt-4o"
      temperature: 0.1
      max_tokens: 2000
      timeout: 30
    
    openrouter:
      model: "anthropic/claude-3.5-sonnet"
      temperature: 0.2
      max_tokens: 4000
      site_url: "https://scout-data-discovery.com"
      app_name: "Scout Data Discovery"
    
    nvidia:
      model: "qwen/qwen2.5-72b-instruct"
      temperature: 0.1
      max_tokens: 2000
      base_url: "https://integrate.api.nvidia.com/v1"

  cache:
    enabled: true
    semantic_enabled: true
    ttl: 3600
    max_size: 1000
    directory: "./cache"

  insights:
    auto_generate: true
    generation_interval: 3600  # 1 hour
    max_insights_per_dataset: 10
    priority_threshold: "medium"
```

### Runtime Configuration

```python
from AI_Functionality.core.unified_ai_manager import UnifiedAIManager

# Load configuration
manager = UnifiedAIManager.from_config_file("config/ai_config.yaml")

# Override specific settings
manager.update_config({
    "providers.openai.temperature": 0.05,
    "cache.ttl": 7200,
    "insights.auto_generate": False
})
```

---

## 🔬 Advanced Features

### 1. Semantic Caching

Intelligent caching based on content similarity rather than exact matches:

```python
from AI_Functionality.core.cache_manager import CacheManager

cache_manager = CacheManager(
    enable_semantic=True,
    semantic_similarity_threshold=0.85,
    embedding_model="text-embedding-ada-002"
)

# Similar queries will hit the cache
query1 = "What is the data quality of this dataset?"
query2 = "How good is the quality of this data?"
# These would be considered semantically similar
```

### 2. Background Processing

Long-running analysis tasks executed asynchronously:

```python
import asyncio
from AI_Functionality.core.ai_analyst import DataAnalyst

async def background_analysis():
    """Run comprehensive analysis in background"""
    
    analyst = DataAnalyst(...)
    
    # Start multiple analysis tasks
    tasks = [
        analyst.analyze_dataset(dataset1, None, AnalysisType.QUALITY),
        analyst.analyze_dataset(dataset2, None, AnalysisType.INSIGHTS),
        analyst.analyze_dataset(dataset3, None, AnalysisType.RELATIONSHIPS)
    ]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results
```

### 3. Custom Analysis Types

Extend the system with custom analysis capabilities:

```python
from AI_Functionality.core.ai_analyst import AnalysisType

# Define custom analysis type
class CustomAnalysisType(AnalysisType):
    SECURITY_AUDIT = "security_audit"
    COMPLIANCE_CHECK = "compliance_check"
    PERFORMANCE_ANALYSIS = "performance_analysis"

# Implement custom analysis logic
async def custom_security_audit(dataset_info, sample_data):
    """Custom security-focused analysis"""
    
    prompt = f"""
    Perform a security audit of this dataset:
    
    Dataset: {dataset_info['name']}
    Sample: {sample_data[:5]}
    
    Analyze for:
    1. PII data exposure risks
    2. Data anonymization requirements
    3. Access control recommendations
    4. Compliance considerations
    """
    
    return await analyst.generate_response(prompt)
```

### 4. Provider Load Balancing

Distribute requests across multiple providers:

```python
from AI_Functionality.core.unified_ai_manager import LoadBalancingStrategy

manager = UnifiedAIManager(
    providers=["openai", "openrouter", "nvidia"],
    load_balancing=LoadBalancingStrategy.ROUND_ROBIN,
    health_check_interval=60
)

# Requests automatically distributed across healthy providers
```

### 5. Cost Optimization

Monitor and optimize AI usage costs:

```python
from AI_Functionality.core.ai_analyst import CostTracker

cost_tracker = CostTracker()

# Track costs per request
response = await analyst.analyze_dataset(...)
cost_tracker.record_usage(response)

# Get cost summary
daily_costs = cost_tracker.get_daily_costs()
provider_costs = cost_tracker.get_costs_by_provider()
model_costs = cost_tracker.get_costs_by_model()

print(f"Daily AI costs: ${daily_costs:.2f}")
print(f"Most expensive provider: {max(provider_costs, key=provider_costs.get)}")
```

---

## 🔗 Integration Guide

### FastAPI Backend Integration

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from AI_Functionality.core.ai_analyst import DataAnalyst

app = FastAPI()

# Initialize AI analyst
analyst = DataAnalyst(
    primary_provider="openai",
    fallback_providers=["openrouter"],
    **config
)

@app.post("/api/ai/analyze")
async def analyze_dataset(request: AnalysisRequest):
    """Analyze dataset with AI"""
    
    try:
        response = await analyst.analyze_dataset(
            dataset_info=request.dataset_info,
            sample_data=request.sample_data,
            analysis_type=request.analysis_type
        )
        
        return {
            "analysis": response.content,
            "model": response.model,
            "cached": response.cached,
            "processing_time": response.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/insights/generate")
async def generate_insights(
    request: InsightRequest,
    background_tasks: BackgroundTasks
):
    """Generate insights for dataset"""
    
    # Add to background task queue
    background_tasks.add_task(
        generate_dataset_insights,
        request.dataset_id,
        request.dataset_info
    )
    
    return {"status": "insights_generation_started"}
```

### Streamlit Frontend Integration

```python
import streamlit as st
from AI_Functionality.core.ai_analyst import DataAnalyst

class AIAnalystComponent:
    """Streamlit component for AI analysis"""
    
    def __init__(self):
        self.analyst = self._initialize_analyst()
    
    def render_analysis_panel(self, dataset_info, sample_data):
        """Render AI analysis interface"""
        
        st.subheader("🤖 AI Analysis")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Analysis Type",
            ["overview", "quality", "insights", "relationships"]
        )
        
        if st.button("🧠 Analyze with AI"):
            with st.spinner("Analyzing..."):
                try:
                    response = await self.analyst.analyze_dataset(
                        dataset_info=dataset_info,
                        sample_data=sample_data,
                        analysis_type=analysis_type
                    )
                    
                    st.success("✅ Analysis Complete")
                    st.markdown(response.content)
                    
                    # Show metadata
                    with st.expander("📊 Analysis Details"):
                        st.write(f"Model: {response.model}")
                        st.write(f"Provider: {response.provider}")
                        st.write(f"Cached: {response.cached}")
                        st.write(f"Processing time: {response.processing_time:.2f}s")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
```

### Scout Data Discovery Integration

```python
from scout_data_discovery.src.scout_discovery import ScoutDataDiscovery
from AI_Functionality.core.ai_analyst import DataAnalyst

class EnhancedScoutDiscovery(ScoutDataDiscovery):
    """Scout Discovery enhanced with AI capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize AI analyst
        self.ai_analyst = DataAnalyst(
            primary_provider="openai",
            fallback_providers=["openrouter", "nvidia"],
            **ai_config
        )
    
    async def assess_dataset_quality_with_ai(self, dataset_id):
        """Enhanced quality assessment with AI insights"""
        
        # Standard Scout quality assessment
        quality_data = self.assess_dataset_quality(dataset_id)
        
        # AI-powered analysis
        ai_response = await self.ai_analyst.analyze_dataset(
            dataset_info=quality_data['dataset_info'],
            sample_data=quality_data['sample_data'],
            analysis_type=AnalysisType.QUALITY
        )
        
        # Combine results
        enhanced_quality = {
            **quality_data,
            "ai_insights": ai_response.content,
            "ai_recommendations": self._extract_recommendations(ai_response.content),
            "ai_analysis_metadata": {
                "model": ai_response.model,
                "provider": ai_response.provider,
                "cached": ai_response.cached
            }
        }
        
        return enhanced_quality
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Provider API Key Issues

**Problem**: `AuthenticationError: Invalid API key`

**Solutions**:
```python
# Check API key format
assert api_key.startswith("sk-"), "OpenAI keys should start with 'sk-'"
assert api_key.startswith("sk-or-v1-"), "OpenRouter keys should start with 'sk-or-v1-'"
assert api_key.startswith("nvapi-"), "NVIDIA keys should start with 'nvapi-'"

# Test provider availability
provider = OpenAIProvider(api_key=api_key)
assert provider.is_available(), "Provider not available"
```

#### 2. Cache Issues

**Problem**: Cache not working or stale data

**Solutions**:
```python
# Clear cache
cache_manager.clear_cache()

# Check cache configuration
cache_manager.get_cache_stats()

# Disable cache temporarily
cache_manager.disable_cache()
```

#### 3. Timeout Errors

**Problem**: `TimeoutError: Request timed out`

**Solutions**:
```python
# Increase timeout
analyst = DataAnalyst(
    request_timeout=60,  # Increase from default 30s
    max_retries=5       # Increase retry attempts
)

# Use background processing for large tasks
await asyncio.wait_for(
    analyst.analyze_dataset(...),
    timeout=120  # 2 minutes
)
```

#### 4. Memory Issues

**Problem**: High memory usage with large datasets

**Solutions**:
```python
# Limit sample data size
sample_data = sample_data[:100]  # First 100 records only

# Use data chunking
chunks = [sample_data[i:i+50] for i in range(0, len(sample_data), 50)]
results = []
for chunk in chunks:
    result = await analyst.analyze_dataset(dataset_info, chunk, analysis_type)
    results.append(result)
```

### Debugging Tools

#### Enable Debug Logging

```python
import logging

# Set debug level
logging.basicConfig(level=logging.DEBUG)

# AI-specific logger
ai_logger = logging.getLogger("AI_Functionality")
ai_logger.setLevel(logging.DEBUG)
```

#### Performance Monitoring

```python
from AI_Functionality.core.ai_analyst import PerformanceMonitor

monitor = PerformanceMonitor()

# Monitor request performance
with monitor.track_request("dataset_analysis"):
    response = await analyst.analyze_dataset(...)

# Get performance stats
stats = monitor.get_stats()
print(f"Average response time: {stats['avg_response_time']:.2f}s")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

#### Health Checks

```python
async def health_check():
    """Comprehensive health check"""
    
    health_status = {
        "providers": {},
        "cache": cache_manager.is_healthy(),
        "insights_engine": insights_engine.is_healthy()
    }
    
    # Check each provider
    for provider_name in ["openai", "openrouter", "nvidia"]:
        provider = get_provider(provider_name)
        health_status["providers"][provider_name] = {
            "available": provider.is_available(),
            "response_time": await provider.test_connection()
        }
    
    return health_status
```

---

## 📚 API Reference

### Core Classes

#### DataAnalyst

```python
class DataAnalyst:
    """Main AI analyst for data analysis and Q&A"""
    
    def __init__(self, 
                 primary_provider: str,
                 fallback_providers: List[str] = None,
                 request_timeout: int = 30,
                 max_retries: int = 3,
                 enable_cache: bool = True,
                 **provider_configs):
        """
        Initialize DataAnalyst
        
        Args:
            primary_provider: Primary AI provider ("openai", "openrouter", "nvidia")
            fallback_providers: List of fallback providers
            request_timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            enable_cache: Enable response caching
            **provider_configs: Provider-specific configurations
        """
    
    async def analyze_dataset(self,
                            dataset_info: Dict[str, Any],
                            sample_data: Optional[List[Dict]] = None,
                            analysis_type: AnalysisType = AnalysisType.OVERVIEW,
                            custom_prompt: Optional[str] = None) -> AIResponse:
        """
        Analyze dataset with AI
        
        Args:
            dataset_info: Dataset metadata
            sample_data: Sample records from dataset
            analysis_type: Type of analysis to perform
            custom_prompt: Optional custom analysis prompt
            
        Returns:
            AIResponse: Analysis results
        """
    
    async def answer_question(self,
                            question: str,
                            dataset_info: Dict[str, Any],
                            context: Optional[Dict] = None,
                            use_cache: bool = True) -> AIResponse:
        """
        Answer question about dataset
        
        Args:
            question: Natural language question
            dataset_info: Dataset metadata
            context: Additional context information
            use_cache: Whether to use cached responses
            
        Returns:
            AIResponse: Answer to the question
        """
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all configured providers"""
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics and costs"""
```

#### InsightsEngine

```python
class InsightsEngine:
    """AI-powered insights generation and management"""
    
    def __init__(self,
                 ai_analyst: DataAnalyst,
                 cache_dir: str = "./insights_cache",
                 insights_storage_dir: str = "./insights_storage"):
        """
        Initialize InsightsEngine
        
        Args:
            ai_analyst: DataAnalyst instance
            cache_dir: Cache directory
            insights_storage_dir: Persistent storage directory
        """
    
    async def generate_dataset_insights(self,
                                      dataset_info: Dict[str, Any],
                                      sample_data: Optional[List[Dict]] = None,
                                      analysis_history: Optional[List[Dict]] = None) -> List[Insight]:
        """
        Generate comprehensive insights for dataset
        
        Args:
            dataset_info: Dataset metadata
            sample_data: Sample records
            analysis_history: Historical analysis results
            
        Returns:
            List[Insight]: Generated insights
        """
    
    async def generate_platform_insights(self,
                                       usage_statistics: Dict[str, Any],
                                       user_patterns: Dict[str, Any],
                                       system_health: Dict[str, Any]) -> List[Insight]:
        """
        Generate platform-wide insights
        
        Args:
            usage_statistics: Platform usage data
            user_patterns: User behavior patterns
            system_health: System performance metrics
            
        Returns:
            List[Insight]: Platform insights
        """
    
    def get_insights(self,
                    insight_type: Optional[InsightType] = None,
                    priority: Optional[InsightPriority] = None,
                    dataset_id: Optional[str] = None,
                    limit: int = 10) -> List[Insight]:
        """
        Retrieve filtered insights
        
        Args:
            insight_type: Filter by insight type
            priority: Filter by priority
            dataset_id: Filter by dataset ID
            limit: Maximum number of insights
            
        Returns:
            List[Insight]: Filtered insights
        """
    
    def get_insight_summary(self) -> Dict[str, Any]:
        """Get summary of all insights"""
```

#### CacheManager

```python
class CacheManager:
    """Advanced caching system with semantic capabilities"""
    
    def __init__(self,
                 cache_dir: str = "./cache",
                 enable_semantic: bool = True,
                 default_ttl: int = 3600,
                 max_cache_size: int = 1000,
                 semantic_similarity_threshold: float = 0.85):
        """
        Initialize CacheManager
        
        Args:
            cache_dir: Cache storage directory
            enable_semantic: Enable semantic caching
            default_ttl: Default time-to-live in seconds
            max_cache_size: Maximum cache entries
            semantic_similarity_threshold: Similarity threshold for semantic cache
        """
    
    async def get(self, key: str, use_semantic: bool = True) -> Optional[Any]:
        """
        Get cached value
        
        Args:
            key: Cache key
            use_semantic: Use semantic similarity matching
            
        Returns:
            Cached value or None
        """
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set cached value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live override
        """
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache entries matching pattern"""
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
```

### Provider Interfaces

#### BaseAIProvider

```python
class BaseAIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate AI response"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check provider availability"""
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
    
    @abstractmethod
    def estimate_cost(self, request: AIRequest) -> float:
        """Estimate request cost"""
```

### Data Models

#### AIRequest

```python
@dataclass
class AIRequest:
    """Standardized AI request format"""
    prompt: str
    system_message: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    model: Optional[str] = None
    timeout: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### AIResponse

```python
@dataclass
class AIResponse:
    """Standardized AI response format"""
    content: str
    model: str
    provider: str
    tokens_used: int
    cost_estimate: float
    processing_time: float
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Insight

```python
@dataclass
class Insight:
    """AI-generated insight"""
    id: str
    type: InsightType
    priority: InsightPriority
    title: str
    description: str
    content: str
    evidence: List[str]
    recommendations: List[str]
    confidence_score: float
    timestamp: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    tags: List[str] = None
```

---

## 📈 Performance Guidelines

### Optimization Strategies

1. **Use Caching Effectively**:
   - Enable semantic caching for similar queries
   - Set appropriate TTL values
   - Monitor cache hit rates

2. **Provider Selection**:
   - Use faster models for simple tasks
   - Reserve powerful models for complex analysis
   - Implement provider fallbacks

3. **Request Optimization**:
   - Limit sample data size
   - Use appropriate temperature settings
   - Set reasonable token limits

4. **Background Processing**:
   - Use async/await for concurrent operations
   - Implement background task queues
   - Monitor system resources

### Performance Monitoring

```python
# Track request metrics
metrics = {
    "total_requests": 1250,
    "avg_response_time": 2.3,
    "cache_hit_rate": 0.78,
    "provider_distribution": {
        "openai": 0.65,
        "openrouter": 0.25,
        "nvidia": 0.10
    },
    "error_rate": 0.02
}
```

---

This comprehensive documentation covers every aspect of the AI_Functionality module. Each component is detailed with purpose, usage examples, configuration options, and integration guidelines.