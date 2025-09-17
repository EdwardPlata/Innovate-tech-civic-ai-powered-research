# AI Testing Suite

This directory contains comprehensive test scripts to validate AI functionality across all components of the Scout Data Discovery system.

## Test Scripts

### 1. `test_api_keys.py`
**Direct API Key Testing**
- Tests OpenAI and NVIDIA APIs directly with HTTP requests
- Validates API keys work independently of the application
- Tries multiple NVIDIA models to find working endpoints
- Helps identify quota/billing issues vs code issues

```bash
python test_api_keys.py
```

### 2. `test_ai_functionality.py`
**AI Functionality Package Testing**
- Tests the AI_Functionality package imports and class creation
- Validates DataAnalyst and AnalysisType functionality
- Tests simple data analysis with mock data
- Isolates package-level issues from API issues

```bash
python test_ai_functionality.py
```

### 3. `test_backend_integration.py`
**Backend API Integration Testing**
- Tests FastAPI backend endpoints for AI functionality
- Validates /api/ai/config and /api/ai/analyze endpoints
- Tests dataset listing and AI configuration workflow
- Requires backend server to be running

```bash
# Start backend first
cd ../backend && python run_server.py

# Then run test
python test_backend_integration.py
```

### 4. `run_all_tests.py`
**Comprehensive Test Suite**
- Runs all tests in sequence
- Provides final summary and next steps
- Captures output from all test scripts

```bash
python run_all_tests.py
```

## Quick Start

1. **Run comprehensive tests:**
   ```bash
   cd /workspaces/Innovate-tech-civic-ai-powered-research/test
   python run_all_tests.py
   ```

2. **Test specific component:**
   ```bash
   # Just API keys
   python test_api_keys.py
   
   # Just AI package
   python test_ai_functionality.py
   
   # Just backend (requires server running)
   python test_backend_integration.py
   ```

## API Keys Used

- **OpenAI**: `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`
- **NVIDIA**: `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`

## Common Issues & Solutions

### ❌ OpenAI Quota Exceeded
- **Issue**: "You exceeded your current quota" (Error 429)
- **Solution**: Check OpenAI billing dashboard, add payment method, or use NVIDIA

### ❌ NVIDIA 404 Errors
- **Issue**: "404 Not Found" for model endpoints
- **Solution**: Test different models, check NVIDIA API documentation for available models

### ❌ Backend Connection Failed
- **Issue**: "Connection refused" to localhost:8080
- **Solution**: Start backend server: `cd backend && python run_server.py`

### ❌ Import Errors
- **Issue**: Cannot import AI_Functionality modules
- **Solution**: Install dependencies: `pip install tiktoken httpx openai aiohttp`

## Expected Output

### ✅ All Working
```
📊 FINAL TEST RESULTS
Direct API Key Testing: ✅ PASSED
AI Functionality Package Testing: ✅ PASSED  
Backend API Integration Testing: ✅ PASSED

🎉 All tests passed! AI functionality is working correctly.
```

### ⚠️ Partial Working
```
📊 FINAL TEST RESULTS
Direct API Key Testing: ❌ FAILED
AI Functionality Package Testing: ✅ PASSED
Backend API Integration Testing: ✅ PASSED

⚠️ Some tests passed. AI functionality is partially working.
```

## Test Flow

1. **API Keys** → Test raw API access
2. **AI Package** → Test application-level AI classes
3. **Backend** → Test full integration through FastAPI
4. **Summary** → Provide actionable next steps

This systematic approach isolates issues at each layer of the AI functionality stack.