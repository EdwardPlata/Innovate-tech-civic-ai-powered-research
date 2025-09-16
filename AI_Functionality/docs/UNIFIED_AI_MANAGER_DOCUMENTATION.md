# Unified AI Manager - Intelligent Multi-Provider AI Orchestration Documentation

## 📋 Overview

The `unified_ai_manager.py` module provides an intelligent orchestration layer for managing multiple AI providers with automatic failover, load balancing, performance optimization, and enhanced error handling. It acts as a unified interface that abstracts away the complexity of managing multiple AI services.

## 🎯 Purpose

- **Primary Role**: Single point of access for all AI operations across multiple providers
- **Key Responsibility**: Intelligent provider selection, failover, and performance optimization
- **Core Function**: Seamless AI service orchestration with automatic error recovery
- **Integration Point**: Central hub for all AI operations in the platform

## 🏗️ Architecture

```python
UnifiedAIManager
├── Provider Management
│   ├── Multi-Provider Support (OpenAI, OpenRouter, NVIDIA)
│   ├── Dynamic Provider Registration/Removal
│   ├── Intelligent Provider Selection
│   └── Automatic Health Monitoring
├── Request Orchestration
│   ├── Automatic Failover (Provider → Provider)
│   ├── Load Balancing (Performance-Based)
│   ├── Retry Logic (Exponential Backoff)
│   └── Request Context Management
├── Performance Intelligence
│   ├── Real-Time Performance Tracking
│   ├── Success Rate Monitoring
│   ├── Response Time Analytics
│   └── Provider Ranking System
├── Caching Integration
│   ├── Semantic Response Caching
│   ├── Request Deduplication
│   ├── Cache Performance Analytics
│   └── Multi-Tier Cache Management
└── Error Handling & Recovery
    ├── Provider Status Management
    ├── Rate Limit Detection & Handling
    ├── Graceful Degradation
    └── Comprehensive Error Reporting
```

## 📊 Data Models

### ProviderStatus Enumeration

**Provider availability and health states**

```python
class ProviderStatus(Enum):
    """
    Comprehensive provider status tracking for intelligent routing
    """
    
    AVAILABLE = "available"
    """
    Provider fully operational and responding normally
    - All requests completing successfully
    - Response times within acceptable ranges
    - No rate limiting or errors detected
    - Ready for new requests
    """
    
    UNAVAILABLE = "unavailable" 
    """
    Provider currently not accessible or responsive
    - Network connectivity issues
    - Service temporarily down
    - Provider initialization failed
    - Should be skipped for new requests
    """
    
    ERROR = "error"
    """
    Provider experiencing errors or failures
    - Recent request failures detected
    - Service returning error responses
    - API authentication issues
    - Temporary cooldown period applied
    """
    
    RATE_LIMITED = "rate_limited"
    """
    Provider has hit rate limits or quota restrictions
    - API quota exceeded
    - Rate limiting responses detected
    - Temporary throttling in effect
    - Will retry after cooldown period
    """

# Usage Example:
provider_status = ProviderStatus.AVAILABLE
if provider_status == ProviderStatus.AVAILABLE:
    # Provider ready for requests
    pass
elif provider_status == ProviderStatus.RATE_LIMITED:
    # Apply rate limit backoff
    await asyncio.sleep(60)
```

### Provider Statistics Model

**Comprehensive performance tracking for each provider**

```python
# Provider Statistics Structure (automatically maintained)
provider_stats = {
    'provider_name': {
        'requests': int,              # Total requests sent to provider
        'successes': int,             # Successfully completed requests
        'failures': int,              # Failed requests (all types)
        'avg_response_time': float,   # Average response time in seconds
        'last_used': float,           # Timestamp of last request
        'status': ProviderStatus      # Current operational status
    }
}

# Example provider statistics:
openai_stats = {
    'requests': 847,
    'successes': 832,
    'failures': 15,
    'avg_response_time': 2.3,
    'last_used': 1703123456.789,
    'status': ProviderStatus.AVAILABLE
}

# Success rate calculation:
success_rate = openai_stats['successes'] / openai_stats['requests']  # 0.982 (98.2%)

# Provider scoring for intelligent routing:
def score_provider(stats):
    """Calculate provider preference score for routing decisions"""
    if stats['requests'] == 0:
        return 0.5  # Neutral score for untested providers
    
    success_rate = stats['successes'] / stats['requests']
    response_score = 1 / (1 + stats['avg_response_time'])  # Lower time = higher score
    
    return success_rate * 0.7 + response_score * 0.3  # Weighted combination
```

## 🧠 Core Classes

### UnifiedAIManager

**Central orchestrator for intelligent multi-provider AI operations**

```python
class UnifiedAIManager:
    """
    Unified AI Manager with intelligent provider selection and failover
    
    Features:
    - Multi-provider support with automatic initialization
    - Intelligent request routing based on performance
    - Automatic failover and retry logic
    - Real-time performance monitoring and analytics
    - Semantic caching with deduplication
    - Dynamic provider management (add/remove at runtime)
    - Comprehensive error handling and recovery
    - Rate limit detection and backoff strategies
    """
    
    def __init__(
        self,
        cache_dir: str = "./ai_cache",
        enable_semantic_cache: bool = True,
        **provider_configs
    ):
        """
        Initialize Unified AI Manager with comprehensive provider support
        
        Args:
            cache_dir: Directory for response caching and analytics
                - Stores cached responses for fast retrieval
                - Includes performance metrics and provider analytics
                - Automatically managed with cleanup policies
            
            enable_semantic_cache: Enable semantic similarity caching
                - Uses embedding-based similarity for cache hits
                - Reduces redundant requests for similar queries
                - Improves response time and reduces costs
            
            **provider_configs: Provider-specific configuration
                - API keys, models, and custom settings
                - Supports all available providers dynamically
                - Can be updated at runtime
        
        Provider Configuration Examples:
        
        # Basic configuration with API keys
        manager = UnifiedAIManager(
            openai_api_key="sk-...",
            openrouter_api_key="sk-or-v1-...",
            nvidia_api_key="nvapi-...",
            
            # Model preferences
            openai_model="gpt-4",
            openrouter_model="anthropic/claude-3-opus",
            nvidia_model="meta/llama2-70b-chat",
            
            # Caching configuration
            cache_dir="./production_ai_cache",
            enable_semantic_cache=True
        )
        
        # Advanced configuration with custom settings
        manager = UnifiedAIManager(
            # Provider API Keys (can also use environment variables)
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
            nvidia_api_key=os.getenv('NVIDIA_API_KEY'),
            
            # Performance tuning
            cache_dir="./cache",
            enable_semantic_cache=True,
            
            # Provider-specific model overrides
            openai_model="gpt-4-turbo-preview",
            openrouter_model="openai/gpt-4-32k",
            nvidia_model="nvidia/nemotron-4-340b-instruct"
        )
        """
        
        self.provider_configs = provider_configs
        
        # Initialize performance tracking for all providers
        self.provider_stats = defaultdict(lambda: {
            'requests': 0,           # Total requests attempted
            'successes': 0,          # Successful completions
            'failures': 0,           # Failed requests
            'avg_response_time': 0.0, # Moving average response time
            'last_used': None,       # Timestamp of last usage
            'status': ProviderStatus.UNAVAILABLE  # Current status
        })
        
        # Initialize advanced caching
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            enable_semantic=enable_semantic_cache
        )
        
        # Provider registry and priority management
        self.providers: Dict[str, BaseAIProvider] = {}
        self.provider_priorities = []  # Dynamic priority based on performance
        
        # Initialize all available providers
        self._initialize_all_providers()
        
        logger.info(f"🚀 UnifiedAIManager initialized with {len(self.providers)} providers")
        
        # Log provider initialization summary
        for provider_name in self.providers:
            status = self.provider_stats[provider_name]['status']
            model = self.providers[provider_name].default_model
            logger.info(f"  ✅ {provider_name}: {model} ({status.value})")
```

## 🔧 Core Methods

### generate_response()

**Intelligent AI response generation with automatic failover**

```python
async def generate_response(
    self,
    request: AIRequest,
    use_cache: bool = True,
    max_retries: int = 2
) -> AIResponse:
    """
    Generate AI response with intelligent provider selection and comprehensive failover
    
    Intelligent Routing Process:
    1. Cache Check: Look for cached responses based on request similarity
    2. Provider Selection: Choose optimal provider based on performance metrics
    3. Request Execution: Send request to selected provider with monitoring
    4. Error Handling: Automatic failover to next best provider on failure
    5. Performance Update: Update provider statistics and rankings
    6. Response Caching: Cache successful responses for future use
    
    Provider Selection Algorithm:
    - Success Rate (70%): Historical success/failure ratio
    - Response Time (30%): Average response latency
    - Status Filtering: Only consider AVAILABLE/RATE_LIMITED providers
    - Cooldown Handling: Skip providers in error state with recent failures
    
    Args:
        request: AIRequest object containing prompt and parameters
            - prompt: The main text to process
            - system_prompt: Optional system context
            - temperature: Response creativity (0.0-1.0)
            - max_tokens: Maximum response length
            - model: Optional model override
        
        use_cache: Whether to use response caching
            - True: Check cache first, store successful responses
            - False: Always make fresh requests, don't cache
            
        max_retries: Maximum retry attempts per provider
            - 0: No retries, fail immediately on error
            - 1-3: Recommended range for balance of reliability/speed
            - 4+: High reliability but potentially slow
    
    Returns:
        AIResponse with comprehensive metadata:
        - content: Generated response text
        - provider: Which provider generated the response
        - model: Specific model used
        - usage: Token usage statistics
        - cached: Whether response came from cache
        - response_time: How long the request took
        - metadata: Additional provider-specific information
    
    Failover Behavior:
    - Provider failures trigger immediate fallback to next provider
    - Rate limits cause temporary provider cooldown
    - Exponential backoff between retry attempts
    - All provider failures result in comprehensive error
    """
    
    logger.info(f"🧠 Generating AI response (cache: {use_cache}, retries: {max_retries})")
    
    # Step 1: Check cache for existing response
    if use_cache:
        logger.debug("🔍 Checking cache for similar requests...")
        cached_response = await self.cache_manager.get_cached_response(request)
        if cached_response:
            logger.info("♻️ Cache hit! Returning cached response")
            cached_response.cached = True
            return cached_response
    
    # Step 2: Get optimal provider ordering
    provider_order = self._get_provider_order()
    logger.debug(f"📊 Provider order: {provider_order}")
    
    if not provider_order:
        raise Exception("❌ No available AI providers found")
    
    # Step 3: Try providers in priority order with failover
    last_exception = None
    
    for provider_name in provider_order:
        if provider_name not in self.providers:
            logger.warning(f"⚠️ Provider {provider_name} not found in registry")
            continue
        
        provider = self.providers[provider_name]
        
        # Skip providers in cooldown or error state
        if self._should_skip_provider(provider_name):
            status = self.provider_stats[provider_name]['status'].value
            logger.debug(f"⏭️ Skipping {provider_name} (status: {status})")
            continue
        
        # Step 4: Attempt request with retries
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Update request statistics
                self.provider_stats[provider_name]['requests'] += 1
                
                logger.info(f"🔄 Attempting {provider_name} (attempt {attempt + 1}/{max_retries + 1})")
                
                # Execute the request
                response = await provider.generate_response(request)
                
                # Step 5: Update success metrics
                response_time = time.time() - start_time
                self._update_provider_success(provider_name, response_time)
                
                # Add provider metadata to response
                response.provider = provider_name
                response.response_time = response_time
                response.cached = False
                
                logger.info(f"✅ {provider_name} succeeded in {response_time:.2f}s")
                
                # Step 6: Cache successful response
                if use_cache:
                    try:
                        await self.cache_manager.cache_response(request, response)
                        logger.debug("💾 Response cached for future use")
                    except Exception as cache_error:
                        logger.warning(f"⚠️ Cache storage failed: {cache_error}")
                
                return response
                
            except Exception as e:
                last_exception = e
                self._update_provider_failure(provider_name, str(e))
                
                error_type = type(e).__name__
                logger.warning(f"❌ {provider_name} attempt {attempt + 1} failed: {error_type}: {e}")
                
                # Apply exponential backoff between retries
                if attempt < max_retries:
                    backoff_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s, 4s...
                    logger.debug(f"⏱️ Backing off {backoff_time}s before retry")
                    await asyncio.sleep(backoff_time)
    
    # Step 7: All providers failed
    provider_statuses = {name: stats['status'].value for name, stats in self.provider_stats.items()}
    logger.error(f"💥 All providers failed. Provider statuses: {provider_statuses}")
    
    if last_exception:
        raise Exception(f"All AI providers failed. Last error: {last_exception}")
    else:
        raise Exception("No available AI providers")

# Usage Examples:

# Basic usage with automatic provider selection
request = AIRequest(
    prompt="Explain quantum computing in simple terms",
    temperature=0.7,
    max_tokens=500
)

response = await ai_manager.generate_response(request)
print(f"Response from {response.provider}: {response.content}")

# High reliability request with maximum retries
critical_request = AIRequest(
    prompt="Generate critical analysis report",
    system_prompt="You are an expert analyst",
    temperature=0.3,
    max_tokens=2000
)

try:
    response = await ai_manager.generate_response(
        critical_request,
        use_cache=True,
        max_retries=3  # Maximum reliability
    )
    print(f"✅ Critical analysis completed by {response.provider}")
except Exception as e:
    logger.error(f"❌ Critical request failed: {e}")

# Performance monitoring example
for i in range(10):
    response = await ai_manager.generate_response(
        AIRequest(prompt=f"Test request {i}")
    )
    print(f"Request {i}: {response.provider} ({response.response_time:.2f}s)")

# Check provider performance after testing
status = ai_manager.get_provider_status()
for provider, stats in status.items():
    print(f"{provider}: {stats['success_rate']:.1%} success, {stats['avg_response_time']:.2f}s avg")
```

### answer_question()

**High-level question answering with intelligent context management**

```python
async def answer_question(
    self,
    question: str,
    dataset_info: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Answer questions with intelligent context assembly and response formatting
    
    Context Assembly Process:
    1. Dataset Context: Incorporate dataset metadata and descriptions
    2. Additional Context: Include user-provided context information
    3. Question Analysis: Analyze question type and requirements
    4. Prompt Construction: Build optimized prompt for AI processing
    5. Response Generation: Generate comprehensive answer
    6. Response Formatting: Structure response with metadata
    
    Question Types Supported:
    
    Data Analysis Questions:
    - "What does this dataset tell us about X?"
    - "How can I analyze Y using this data?"
    - "What patterns might exist in Z?"
    
    Technical Implementation Questions:
    - "How do I query this dataset for X?"
    - "What tools work best with this data?"
    - "How should I preprocess this dataset?"
    
    Insight and Discovery Questions:
    - "What insights can be derived from this data?"
    - "What are the limitations of this dataset?"
    - "How reliable is this data for X analysis?"
    
    Contextual Domain Questions:
    - "What does X mean in the context of Y?"
    - "How does this relate to Z domain knowledge?"
    - "What are the implications of these findings?"
    
    Args:
        question: User's natural language question
            - Can be simple or complex queries
            - Supports follow-up and contextual questions
            - Handles technical and domain-specific terminology
        
        dataset_info: Optional dataset metadata for context
            Structure: {
                'name': str,           # Dataset name/identifier
                'description': str,    # Dataset description
                'category': str,       # Data category/domain
                'columns': List[str],  # Column names (optional)
                'size': int,          # Number of records (optional)
                'source': str,        # Data source (optional)
                'last_updated': str,  # Update timestamp (optional)
                'quality_score': float, # Quality assessment (optional)
                'tags': List[str]     # Dataset tags (optional)
            }
        
        context: Additional context information
            - Domain knowledge or background information
            - Previous conversation context
            - Specific constraints or requirements
            - Related analysis results
        
        use_cache: Whether to cache responses
            - Improves response time for similar questions
            - Reduces API costs for repeated queries
    
    Returns:
        Comprehensive answer dictionary:
        {
            'answer': str,              # Main answer text
            'provider': str,            # AI provider used
            'model': str,              # Specific model used
            'cached': bool,            # Whether response was cached
            'metadata': {
                'question': str,        # Original question
                'context_provided': bool, # Whether context was available
                'dataset_info': bool,   # Whether dataset info was provided
                'response_time': float, # Time to generate response
                'confidence': float,    # AI confidence (if available)
                'sources': List[str]    # Information sources used
            }
        }
    """
    
    logger.info(f"❓ Answering question: {question[:100]}...")
    
    # Step 1: Build comprehensive context
    full_context = ""
    context_components = []
    
    if dataset_info:
        logger.debug("📊 Adding dataset context")
        dataset_context = f"""Dataset Information:
Name: {dataset_info.get('name', 'Unknown')}
Description: {dataset_info.get('description', 'No description available')}
Category: {dataset_info.get('category', 'Unknown')}"""
        
        # Add optional dataset details
        if 'columns' in dataset_info:
            dataset_context += f"\nColumns: {', '.join(dataset_info['columns'][:10])}"
            if len(dataset_info['columns']) > 10:
                dataset_context += f" (and {len(dataset_info['columns']) - 10} more)"
        
        if 'size' in dataset_info:
            dataset_context += f"\nRecords: {dataset_info['size']:,}"
        
        if 'source' in dataset_info:
            dataset_context += f"\nSource: {dataset_info['source']}"
        
        if 'quality_score' in dataset_info:
            quality = dataset_info['quality_score']
            dataset_context += f"\nQuality Score: {quality:.1f}/10"
        
        full_context += dataset_context + "\n\n"
        context_components.append("dataset_info")
    
    if context:
        logger.debug("📝 Adding additional context")
        full_context += f"Additional Context:\n{context}\n\n"
        context_components.append("additional_context")
    
    # Step 2: Construct optimized prompt
    prompt = f"""Based on the following information, please answer this question: "{question}"

{full_context}

Please provide a comprehensive and helpful answer that:

🎯 **Directly addresses the question** with specific, actionable information
📊 **Uses the provided data context** to give relevant insights
💡 **Provides practical guidance** for next steps or implementation
⚠️ **Identifies limitations** if the available information is insufficient
🔗 **Suggests related areas** to explore for deeper understanding

If the question cannot be fully answered from the provided information, clearly explain what additional data or context would be needed to provide a complete answer."""

    # Step 3: Create optimized AI request
    request = AIRequest(
        prompt=prompt,
        system_prompt="""You are an expert data analyst and consultant. Your role is to:

1. Provide clear, accurate, and actionable answers based on available data
2. Identify insights and patterns that may not be immediately obvious
3. Suggest practical approaches for data analysis and implementation
4. Clearly communicate limitations and recommend additional data when needed
5. Use domain expertise to provide contextual understanding

Focus on being helpful, specific, and practical in your responses.""",
        temperature=0.4,  # Balanced creativity with factual accuracy
        max_tokens=1500   # Sufficient length for comprehensive answers
    )
    
    try:
        # Step 4: Generate AI response
        start_time = time.time()
        response = await self.generate_response(request, use_cache=use_cache)
        response_time = time.time() - start_time
        
        logger.info(f"✅ Question answered by {response.provider} in {response_time:.2f}s")
        
        # Step 5: Structure comprehensive response
        structured_response = {
            'answer': response.content,
            'provider': response.provider,
            'model': response.model,
            'cached': response.cached,
            'metadata': {
                'question': question,
                'context_provided': bool(full_context.strip()),
                'dataset_info': bool(dataset_info),
                'additional_context': bool(context),
                'context_components': context_components,
                'response_time': response_time,
                'timestamp': time.time(),
                'cached': response.cached,
                'token_usage': getattr(response, 'usage', None)
            }
        }
        
        # Add confidence assessment if available
        if hasattr(response, 'confidence'):
            structured_response['metadata']['confidence'] = response.confidence
        
        return structured_response
        
    except Exception as e:
        logger.error(f"❌ Question answering failed: {e}")
        
        # Return graceful error response
        return {
            'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question.",
            'provider': 'error',
            'model': 'none',
            'cached': False,
            'metadata': {
                'error': str(e),
                'error_type': type(e).__name__,
                'question': question,
                'context_provided': bool(full_context.strip()),
                'dataset_info': bool(dataset_info),
                'timestamp': time.time()
            }
        }

# Usage Examples:

# Basic question answering
response = await ai_manager.answer_question(
    "What are the key trends in this data?"
)
print(f"Answer: {response['answer']}")

# Question with dataset context
dataset_info = {
    'name': 'NYC Traffic Accidents',
    'description': 'Motor vehicle collision data from NYC Open Data',
    'category': 'Transportation',
    'columns': ['date', 'borough', 'injuries', 'fatalities', 'vehicle_type'],
    'size': 1500000,
    'quality_score': 8.5
}

response = await ai_manager.answer_question(
    "What patterns exist in traffic accidents across NYC boroughs?",
    dataset_info=dataset_info
)
print(f"Dataset-aware answer: {response['answer']}")

# Question with additional context
response = await ai_manager.answer_question(
    "How should I approach analyzing seasonal patterns?",
    dataset_info=dataset_info,
    context="I'm particularly interested in winter vs summer accident patterns and want to account for weather conditions."
)
print(f"Contextual answer: {response['answer']}")
print(f"Response time: {response['metadata']['response_time']:.2f}s")
print(f"Cached: {response['cached']}")
```

### health_check_all()

**Comprehensive health monitoring for all providers**

```python
async def health_check_all(self) -> Dict[str, bool]:
    """
    Perform comprehensive health check on all registered providers
    
    Health Check Process:
    1. Concurrent Testing: Test all providers simultaneously for speed
    2. Basic Connectivity: Verify API endpoint accessibility
    3. Authentication: Confirm API keys and credentials are valid
    4. Response Quality: Send test prompt and validate response
    5. Performance Baseline: Measure response times under normal conditions
    6. Status Update: Update provider status based on results
    
    Health Check Criteria:
    - API Connectivity: Can reach provider endpoints
    - Authentication: Valid API credentials
    - Response Generation: Can generate valid responses
    - Response Time: Within acceptable performance thresholds
    - Error Rate: Low error rate for test requests
    
    Returns:
        Dict mapping provider names to health status:
        {
            'provider_name': bool,  # True if healthy, False if unhealthy
            ...
        }
        
    Side Effects:
        - Updates provider_stats['status'] for each provider
        - Logs detailed health check results
        - May trigger provider cooldowns for failed checks
    """
    
    logger.info("🏥 Starting comprehensive health check for all providers...")
    
    # Prepare health check tasks for concurrent execution
    health_tasks = {}
    
    for name, provider in self.providers.items():
        logger.debug(f"🔍 Preparing health check for {name}")
        health_tasks[name] = provider.health_check()
    
    # Execute all health checks concurrently
    results = {}
    start_time = time.time()
    
    # Use asyncio.gather to run health checks in parallel
    try:
        task_results = await asyncio.gather(
            *health_tasks.values(),
            return_exceptions=True
        )
        
        # Process results and update provider status
        for (name, _), result in zip(health_tasks.items(), task_results):
            if isinstance(result, Exception):
                logger.error(f"❌ Health check failed for {name}: {result}")
                results[name] = False
                self.provider_stats[name]['status'] = ProviderStatus.ERROR
            else:
                results[name] = result
                
                if result:
                    logger.info(f"✅ {name} is healthy")
                    self.provider_stats[name]['status'] = ProviderStatus.AVAILABLE
                else:
                    logger.warning(f"⚠️ {name} failed health check")
                    self.provider_stats[name]['status'] = ProviderStatus.ERROR
    
    except Exception as e:
        logger.error(f"💥 Health check execution failed: {e}")
        # Mark all providers as unknown status
        for name in self.providers:
            results[name] = False
            self.provider_stats[name]['status'] = ProviderStatus.ERROR
    
    total_time = time.time() - start_time
    healthy_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"🏥 Health check completed: {healthy_count}/{total_count} providers healthy ({total_time:.2f}s)")
    
    # Log detailed status summary
    for name, is_healthy in results.items():
        status_emoji = "✅" if is_healthy else "❌"
        logger.info(f"  {status_emoji} {name}: {'healthy' if is_healthy else 'unhealthy'}")
    
    return results

# Usage Examples:

# Basic health check
health_results = await ai_manager.health_check_all()
print("Provider Health Status:")
for provider, healthy in health_results.items():
    status = "✅ Healthy" if healthy else "❌ Unhealthy"
    print(f"  {provider}: {status}")

# Monitor health over time
import asyncio

async def monitor_provider_health():
    """Monitor provider health every 5 minutes"""
    while True:
        try:
            health = await ai_manager.health_check_all()
            unhealthy = [name for name, status in health.items() if not status]
            
            if unhealthy:
                logger.warning(f"⚠️ Unhealthy providers detected: {unhealthy}")
            else:
                logger.info("✅ All providers healthy")
                
        except Exception as e:
            logger.error(f"❌ Health monitoring failed: {e}")
        
        await asyncio.sleep(300)  # 5 minutes

# Start health monitoring
# asyncio.create_task(monitor_provider_health())

# Conditional request routing based on health
health = await ai_manager.health_check_all()
healthy_providers = [name for name, status in health.items() if status]

if not healthy_providers:
    logger.error("❌ No healthy providers available")
else:
    logger.info(f"✅ {len(healthy_providers)} healthy providers: {healthy_providers}")
```

### get_provider_status()

**Comprehensive provider performance and status analytics**

```python
def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive status and performance analytics for all providers
    
    Analytics Collected:
    - Availability Status: Current operational state
    - Performance Metrics: Success rates, response times, reliability
    - Usage Statistics: Request counts, error patterns
    - Configuration Details: Models, provider settings
    - Historical Trends: Performance over time
    
    Returns:
        Comprehensive provider analytics:
        {
            'provider_name': {
                'available': bool,           # Currently available for requests
                'model': str,               # Default model being used
                'provider_class': str,      # Implementation class name
                'status': str,              # Current status (available/error/rate_limited)
                'requests': int,            # Total requests attempted
                'successes': int,           # Successful completions
                'failures': int,            # Failed requests
                'success_rate': float,      # Success percentage (0.0-1.0)
                'avg_response_time': float, # Average response time in seconds
                'last_used': Optional[float], # Timestamp of last usage
                'performance_score': float  # Overall performance rating
            },
            ...
        }
    """
    
    logger.debug("📊 Collecting comprehensive provider status analytics")
    
    status_report = {}
    current_time = time.time()
    
    for name in self.providers:
        provider = self.providers[name]
        stats = self.provider_stats[name]
        
        # Calculate derived metrics
        total_requests = max(stats['requests'], 1)  # Avoid division by zero
        success_rate = stats['successes'] / total_requests
        
        # Calculate performance score (weighted combination of metrics)
        response_score = 1 / (1 + stats['avg_response_time']) if stats['avg_response_time'] > 0 else 0.5
        availability_score = 1.0 if stats['status'] == ProviderStatus.AVAILABLE else 0.0
        performance_score = (success_rate * 0.5) + (response_score * 0.3) + (availability_score * 0.2)
        
        # Time since last use
        time_since_last_use = None
        if stats['last_used']:
            time_since_last_use = current_time - stats['last_used']
        
        # Build comprehensive status
        provider_status = {
            # Availability
            'available': stats['status'] == ProviderStatus.AVAILABLE,
            'status': stats['status'].value,
            
            # Configuration
            'model': provider.default_model,
            'provider_class': provider.__class__.__name__,
            'api_endpoint': getattr(provider, 'api_endpoint', 'unknown'),
            
            # Usage Statistics
            'requests': stats['requests'],
            'successes': stats['successes'],
            'failures': stats['failures'],
            
            # Performance Metrics
            'success_rate': success_rate,
            'avg_response_time': stats['avg_response_time'],
            'performance_score': performance_score,
            
            # Temporal Information
            'last_used': stats['last_used'],
            'time_since_last_use': time_since_last_use,
            
            # Status Details
            'is_healthy': stats['status'] in [ProviderStatus.AVAILABLE, ProviderStatus.RATE_LIMITED],
            'is_rate_limited': stats['status'] == ProviderStatus.RATE_LIMITED,
            'has_errors': stats['status'] == ProviderStatus.ERROR,
            
            # Additional Metrics
            'failure_rate': stats['failures'] / total_requests,
            'reliability_score': success_rate if stats['requests'] >= 10 else None,  # Reliable only with sufficient data
        }
        
        # Add provider-specific metadata if available
        if hasattr(provider, 'get_provider_info'):
            try:
                provider_info = provider.get_provider_info()
                provider_status.update(provider_info)
            except Exception as e:
                logger.debug(f"Could not get provider info for {name}: {e}")
        
        status_report[name] = provider_status
    
    # Add summary statistics
    total_providers = len(status_report)
    available_providers = sum(1 for status in status_report.values() if status['available'])
    
    status_report['_summary'] = {
        'total_providers': total_providers,
        'available_providers': available_providers,
        'availability_percentage': available_providers / total_providers if total_providers > 0 else 0,
        'best_provider': self.get_best_provider(),
        'timestamp': current_time
    }
    
    logger.debug(f"📊 Status report generated: {available_providers}/{total_providers} providers available")
    
    return status_report

# Usage Examples:

# Get comprehensive status report
status = ai_manager.get_provider_status()

print("🔍 Provider Status Report:")
print(f"Available: {status['_summary']['available_providers']}/{status['_summary']['total_providers']}")
print(f"Best Provider: {status['_summary']['best_provider']}")

for provider_name, details in status.items():
    if provider_name.startswith('_'):  # Skip summary
        continue
        
    print(f"\n📊 {provider_name} ({details['provider_class']}):")
    print(f"  Status: {details['status']}")
    print(f"  Model: {details['model']}")
    print(f"  Requests: {details['requests']}")
    print(f"  Success Rate: {details['success_rate']:.1%}")
    print(f"  Avg Response Time: {details['avg_response_time']:.2f}s")
    print(f"  Performance Score: {details['performance_score']:.2f}")
    
    if details['last_used']:
        last_used_minutes = details['time_since_last_use'] / 60
        print(f"  Last Used: {last_used_minutes:.1f} minutes ago")

# Monitor performance trends
def analyze_provider_performance(status_report):
    """Analyze provider performance and identify issues"""
    
    issues = []
    recommendations = []
    
    for name, details in status_report.items():
        if name.startswith('_'):
            continue
        
        # Check for performance issues
        if details['success_rate'] < 0.9 and details['requests'] > 10:
            issues.append(f"❌ {name}: Low success rate ({details['success_rate']:.1%})")
            recommendations.append(f"💡 Check {name} API key and service status")
        
        if details['avg_response_time'] > 5.0:
            issues.append(f"⏱️ {name}: Slow response times ({details['avg_response_time']:.1f}s)")
            recommendations.append(f"💡 Consider alternative models for {name} or check network")
        
        if not details['available']:
            issues.append(f"🚫 {name}: Currently unavailable")
            recommendations.append(f"💡 Run health check for {name}")
    
    return {
        'issues': issues,
        'recommendations': recommendations,
        'overall_health': len(issues) == 0
    }

# Analyze current performance
analysis = analyze_provider_performance(status)
if analysis['issues']:
    print("\n⚠️ Performance Issues Detected:")
    for issue in analysis['issues']:
        print(f"  {issue}")
    
    print("\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")
else:
    print("\n✅ All providers performing well!")
```

## 🚀 Usage Examples

### Basic Multi-Provider Setup

```python
import asyncio
from AI_Functionality.core.unified_ai_manager import UnifiedAIManager
from AI_Functionality.core.base_provider import AIRequest

async def basic_setup_example():
    """Basic setup with multiple providers and intelligent routing"""
    
    # Initialize with multiple providers
    ai_manager = UnifiedAIManager(
        openai_api_key="sk-...",
        openrouter_api_key="sk-or-v1-...",
        nvidia_api_key="nvapi-...",
        
        # Performance tuning
        cache_dir="./ai_cache",
        enable_semantic_cache=True
    )
    
    # Check initial health
    health = await ai_manager.health_check_all()
    print("🏥 Provider Health:")
    for provider, is_healthy in health.items():
        print(f"  {provider}: {'✅' if is_healthy else '❌'}")
    
    # Make requests with automatic failover
    requests = [
        "Explain machine learning in simple terms",
        "What are the benefits of cloud computing?",
        "How does blockchain technology work?",
        "Describe the future of artificial intelligence"
    ]
    
    print("\n🧠 Testing automatic provider selection:")
    for i, prompt in enumerate(requests):
        request = AIRequest(prompt=prompt, temperature=0.7)
        response = await ai_manager.generate_response(request)
        
        print(f"{i+1}. Provider: {response.provider}")
        print(f"   Response: {response.content[:100]}...")
        print(f"   Time: {response.response_time:.2f}s")
        print()

# Run the example
# asyncio.run(basic_setup_example())
```

### Advanced Question Answering with Context

```python
async def advanced_qa_example():
    """Advanced question answering with rich context"""
    
    ai_manager = UnifiedAIManager(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        enable_semantic_cache=True
    )
    
    # Rich dataset context
    nyc_dataset = {
        'name': 'NYC Motor Vehicle Collisions',
        'description': 'Comprehensive traffic accident data from NYC Open Data portal',
        'category': 'Transportation Safety',
        'columns': ['crash_date', 'borough', 'zip_code', 'latitude', 'longitude', 
                   'persons_injured', 'persons_killed', 'vehicle_type_1', 'contributing_factor'],
        'size': 1800000,
        'source': 'NYC Department of Transportation',
        'quality_score': 8.7,
        'last_updated': '2024-01-15'
    }
    
    # Series of related questions with context
    questions = [
        {
            'question': 'What are the main patterns in NYC traffic accidents?',
            'context': 'Focus on temporal patterns, geographic distribution, and severity factors.'
        },
        {
            'question': 'How can I identify accident hotspots using this data?',
            'context': 'I want to create visualizations for city planning purposes.'
        },
        {
            'question': 'What machine learning approaches work best for accident prediction?',
            'context': 'Consider both traditional ML and deep learning methods for time series prediction.'
        }
    ]
    
    print("🎯 Advanced Q&A with Rich Context:")
    for i, q in enumerate(questions):
        print(f"\n{i+1}. {q['question']}")
        
        response = await ai_manager.answer_question(
            question=q['question'],
            dataset_info=nyc_dataset,
            context=q['context'],
            use_cache=True
        )
        
        print(f"Provider: {response['provider']}")
        print(f"Cached: {response['cached']}")
        print(f"Answer: {response['answer'][:300]}...")
        
        # Show metadata
        metadata = response['metadata']
        print(f"Context Used: {metadata['context_provided']}")
        print(f"Response Time: {metadata['response_time']:.2f}s")

# asyncio.run(advanced_qa_example())
```

### Performance Monitoring and Optimization

```python
async def performance_monitoring_example():
    """Monitor and optimize provider performance"""
    
    ai_manager = UnifiedAIManager(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
        cache_dir="./performance_cache"
    )
    
    # Generate load test requests
    test_requests = [
        AIRequest(prompt=f"Analyze this scenario: {i}", temperature=0.5)
        for i in range(20)
    ]
    
    print("🔄 Running performance test...")
    start_time = time.time()
    
    # Execute requests and track performance
    results = []
    for i, request in enumerate(test_requests):
        try:
            response = await ai_manager.generate_response(request)
            results.append({
                'request_id': i,
                'provider': response.provider,
                'response_time': response.response_time,
                'cached': response.cached,
                'success': True
            })
            print(f"✅ {i+1}/20: {response.provider} ({response.response_time:.2f}s)")
        except Exception as e:
            results.append({
                'request_id': i,
                'error': str(e),
                'success': False
            })
            print(f"❌ {i+1}/20: Failed - {e}")
    
    total_time = time.time() - start_time
    
    # Analyze performance
    successful_requests = [r for r in results if r['success']]
    provider_usage = {}
    
    for result in successful_requests:
        provider = result['provider']
        if provider not in provider_usage:
            provider_usage[provider] = []
        provider_usage[provider].append(result['response_time'])
    
    print(f"\n📊 Performance Analysis ({total_time:.1f}s total):")
    print(f"Success Rate: {len(successful_requests)}/{len(test_requests)} ({len(successful_requests)/len(test_requests):.1%})")
    
    for provider, times in provider_usage.items():
        avg_time = sum(times) / len(times)
        print(f"  {provider}: {len(times)} requests, {avg_time:.2f}s avg")
    
    # Get detailed provider status
    status = ai_manager.get_provider_status()
    print(f"\n🔍 Current Provider Status:")
    for provider, details in status.items():
        if provider.startswith('_'):
            continue
        print(f"  {provider}: {details['success_rate']:.1%} success, {details['performance_score']:.2f} score")

# asyncio.run(performance_monitoring_example())
```

### Production Deployment Example

```python
class ProductionAIManager:
    """Production-ready AI manager with monitoring and error handling"""
    
    def __init__(self):
        self.ai_manager = UnifiedAIManager(
            # Use environment variables for security
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
            nvidia_api_key=os.getenv('NVIDIA_API_KEY'),
            
            # Production optimizations
            cache_dir="/var/cache/ai_responses",
            enable_semantic_cache=True,
            
            # Model preferences for production
            openai_model="gpt-4-turbo-preview",
            openrouter_model="anthropic/claude-3-opus",
        )
        
        # Start monitoring
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._performance_monitor())
    
    async def _health_monitor(self):
        """Continuous health monitoring"""
        while True:
            try:
                health = await self.ai_manager.health_check_all()
                unhealthy = [name for name, status in health.items() if not status]
                
                if unhealthy:
                    logger.warning(f"🚨 Unhealthy providers: {unhealthy}")
                    # Could trigger alerts here
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _performance_monitor(self):
        """Performance analytics and optimization"""
        while True:
            try:
                status = self.ai_manager.get_provider_status()
                
                # Log performance metrics
                for provider, details in status.items():
                    if provider.startswith('_'):
                        continue
                    
                    logger.info(f"📊 {provider}: {details['success_rate']:.1%} success, "
                              f"{details['avg_response_time']:.2f}s avg")
                
                # Check for performance degradation
                for provider, details in status.items():
                    if provider.startswith('_'):
                        continue
                    
                    if details['success_rate'] < 0.9 and details['requests'] > 100:
                        logger.warning(f"⚠️ {provider} performance degraded: {details['success_rate']:.1%}")
                        
                    if details['avg_response_time'] > 10.0:
                        logger.warning(f"⚠️ {provider} slow responses: {details['avg_response_time']:.1f}s")
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
            
            await asyncio.sleep(3600)  # Check every hour
    
    async def process_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Production request processing with full error handling"""
        try:
            request = AIRequest(prompt=prompt, **kwargs)
            response = await self.ai_manager.generate_response(request)
            
            return {
                'success': True,
                'content': response.content,
                'provider': response.provider,
                'model': response.model,
                'response_time': response.response_time,
                'cached': response.cached
            }
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_response': "I apologize, but I'm unable to process your request at the moment. Please try again later."
            }

# Production usage
production_ai = ProductionAIManager()

async def handle_user_request(user_prompt: str):
    """Handle user requests in production"""
    result = await production_ai.process_request(user_prompt)
    
    if result['success']:
        return result['content']
    else:
        # Log error and return graceful fallback
        logger.error(f"User request failed: {result['error']}")
        return result['fallback_response']
```

This comprehensive documentation covers all aspects of the UnifiedAIManager, from basic setup to production deployment with intelligent provider management, performance optimization, and comprehensive error handling.