# AI Functionality Setup Guide

## Quick Setup for Development

If you want to enable the full AI capabilities, install the AI Functionality package:

### 1. Install Dependencies
```bash
# Install AI Functionality requirements
pip install -r ../AI_Functionality/requirements.txt

# Optional for better performance
pip install sentence-transformers diskcache
```

### 2. Test AI Package Installation
```bash
python -c "
from AI_Functionality import DataAnalyst, AnalysisType
print('✅ AI Functionality installed successfully')
"
```

### 3. Configure API Keys via Backend
Once the server is running, configure your AI providers:

```bash
# Configure AI services
curl -X POST http://localhost:8080/api/ai/config \
  -H "Content-Type: application/json" \
  -d '{
    "openai_api_key": "sk-your-openai-key",
    "primary_provider": "openai",
    "enable_semantic_cache": true
  }'

# Check configuration
curl http://localhost:8080/api/ai/config
```

### 4. Test AI Analysis
```bash
curl -X POST http://localhost:8080/api/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "abc123",
    "analysis_type": "overview",
    "include_sample": true
  }'
```

## Without AI Package (Current State)
The backend works perfectly without the AI Functionality package:
- ✅ All Scout features work normally
- ✅ AI endpoints return static analysis
- ✅ API key management endpoints available
- ✅ Shutdown functionality works
- ✅ Fallback analysis provides meaningful insights

## System Status
Check what's available:
```bash
curl http://localhost:8080/api/system/status
```

Response shows:
- `ai_functionality_available`: false (without package) / true (with package)
- `ai_analyst_active`: false (no keys) / true (configured)
- `api_keys_configured`: Number of configured providers