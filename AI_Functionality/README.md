# 🧠 AI Functionality - Advanced AI Integration System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive AI-powered analysis system that provides multi-provider AI integration, intelligent caching, dataset analysis, codebase analytics, and automated insights generation for data discovery applications.

## 🚀 Features

### Core Capabilities
- **🔄 Multi-Provider Support**: OpenAI, OpenRouter, NVIDIA AI with automatic fallback
- **⚡ Advanced Caching**: Prompt-based and semantic caching for optimal performance
- **📊 Dataset Analysis**: AI-powered data quality assessment and insights generation
- **🧪 Codebase Analytics**: Intelligent code analysis, Q&A, and improvement suggestions
- **💡 Insights Engine**: Automated generation of actionable insights and recommendations
- **🛡️ Error Resilience**: Comprehensive error handling with retry mechanisms
- **⏱️ Timeout Management**: Configurable timeouts with progress tracking

### Analysis Types
- **Overview Analysis**: Comprehensive dataset summaries and key characteristics
- **Quality Assessment**: 5-dimensional quality scoring (Scout methodology)
- **Trend Analysis**: Historical pattern detection and future projections
- **Predictive Analytics**: Time-series forecasting opportunities
- **Usage Analytics**: Download patterns and user adoption insights
- **Code Analysis**: Structure, quality, security, and maintainability assessment

## 📁 Architecture Overview

```
AI_Functionality/
├── core/                           # Core AI system components
│   ├── __init__.py                # Package initialization
│   ├── ai_analyst.py              # Main AI orchestrator
│   ├── base_provider.py           # Provider interface & models
│   ├── cache_manager.py           # Multi-layer caching system
│   ├── codebase_agent.py          # Code analysis & Q&A agent
│   └── insights_engine.py         # Automated insights generation
│
├── providers/                      # AI provider implementations
│   ├── __init__.py                # Provider package init
│   ├── openai_provider.py         # OpenAI GPT models
│   ├── openrouter_provider.py     # Multi-model access via OpenRouter
│   └── nvidia_provider.py         # NVIDIA AI Foundation models
│
├── examples/                       # Usage examples and tutorials
├── tests/                          # Unit and integration tests
└── README.md                       # This documentation
```

## 🔧 Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required dependencies
pip install asyncio aiohttp pandas numpy streamlit fastapi
pip install openai httpx tiktoken  # For AI providers
```

### Setup

1. **Clone or copy the AI_Functionality directory** to your project
2. **Install dependencies** (requirements.txt coming soon)
3. **Configure API keys** in your environment or application settings
4. **Import and initialize** the DataAnalyst in your application

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from AI_Functionality import DataAnalyst, AnalysisType

# Initialize with API keys
analyst = DataAnalyst(
    openai_api_key="sk-your-openai-key",
    nvidia_api_key="nvapi-your-nvidia-key",
    primary_provider="openai",
    fallback_providers=["nvidia"],
    cache_dir="./ai_cache"
)

# Analyze a dataset
dataset_info = {
    "id": "nyc-311-data",
    "name": "NYC 311 Service Requests",
    "description": "Citizen service requests and complaints",
    "category": "Public Services"
}

sample_data = [
    {"complaint_type": "Noise", "borough": "Manhattan", "status": "Open"},
    {"complaint_type": "Heat/Hot Water", "borough": "Brooklyn", "status": "Closed"}
]

async def run_analysis():
    # Dataset overview analysis
    response = await analyst.analyze_dataset(
        dataset_info=dataset_info,
        sample_data=sample_data,
        analysis_type=AnalysisType.OVERVIEW
    )

    print(f"Analysis: {response.content}")
    print(f"Provider: {response.provider}")
    print(f"Cached: {response.cached}")

# Run the analysis
asyncio.run(run_analysis())
```

## 🔌 Provider Configuration

### OpenAI Configuration

```python
analyst = DataAnalyst(
    openai_api_key="sk-your-openai-api-key",
    primary_provider="openai"
)

# Available models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
```

**Get API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Cost**: Pay-per-use pricing
- **Benefits**: High-quality responses, fast processing
- **Best for**: Production applications, detailed analysis

### NVIDIA AI Configuration

```python
analyst = DataAnalyst(
    nvidia_api_key="nvapi-your-nvidia-api-key",
    primary_provider="nvidia"
)

# Available models: qwen/qwen2.5-72b-instruct, meta/llama-3.1-405b-instruct
```

**Get API Key**: [NVIDIA AI](https://build.nvidia.com)
- **Cost**: Free tier available with good rate limits
- **Benefits**: Powerful reasoning models, cost-effective
- **Best for**: Development, complex reasoning tasks

### OpenRouter Configuration

```python
analyst = DataAnalyst(
    openrouter_api_key="sk-or-your-openrouter-key",
    primary_provider="openrouter"
)

# Available models: anthropic/claude-3-sonnet, google/gemini-pro, meta-llama/llama-3-70b
```

**Get API Key**: [OpenRouter](https://openrouter.ai/keys)
- **Cost**: Varies by model, competitive pricing
- **Benefits**: Access to multiple AI providers, model diversity
- **Best for**: Experimentation, accessing specialized models

### Multi-Provider Setup (Recommended)

```python
# Configure multiple providers with fallback
analyst = DataAnalyst(
    openai_api_key="sk-your-openai-key",
    nvidia_api_key="nvapi-your-nvidia-key",
    openrouter_api_key="sk-or-your-openrouter-key",
    primary_provider="nvidia",           # Use NVIDIA first (free tier)
    fallback_providers=["openai", "openrouter"]  # Fallback order
)
```

## ⚡ Performance & Optimization

### Caching System

The AI Functionality includes a sophisticated multi-layer caching system:

```python
# Configure cache settings
cache_manager = CacheManager(
    cache_dir="./ai_cache",
    enable_semantic=True,        # Semantic similarity caching
    default_ttl=3600,           # 1 hour cache lifetime
    max_cache_size_mb=500       # Maximum cache size
)

# Cache statistics
stats = cache_manager.get_cache_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit ratio: {stats['hit_ratio']:.2%}")
```

## 🛠️ API Integration

The system provides ready-to-use API endpoints:

- `POST /api/ai/analyze` - Dataset analysis
- `POST /api/ai/question` - Dataset Q&A
- `POST /api/insights/generate` - Generate insights
- `GET /api/insights` - Retrieve insights
- `POST /api/codebase/analyze` - Codebase analysis
- `POST /api/codebase/question` - Code Q&A

## 📝 License

This project is licensed under the MIT License.

---

**Built with ❤️ for intelligent data discovery and analysis**