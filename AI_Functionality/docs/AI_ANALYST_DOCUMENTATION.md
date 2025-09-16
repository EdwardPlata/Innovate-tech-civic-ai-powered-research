# AI Analyst - Core Orchestrator Documentation

## 📋 Overview

The `ai_analyst.py` module provides the main `DataAnalyst` class, which serves as the primary orchestrator for all AI-powered data analysis operations in the Scout Data Discovery platform.

## 🎯 Purpose

- **Primary Role**: Unified interface for AI-powered data analysis
- **Key Responsibility**: Coordinate between multiple AI providers for robust analysis
- **Core Function**: Transform data analysis requests into AI provider calls
- **Integration Point**: Bridge between Scout Discovery and AI providers

## 🏗️ Architecture

```python
DataAnalyst
├── Provider Management
│   ├── Primary Provider (main AI service)
│   ├── Fallback Providers (backup AI services)
│   └── Provider Health Monitoring
├── Analysis Engine
│   ├── Dataset Analysis (quality, overview, insights)
│   ├── Interactive Q&A (natural language queries)
│   └── Custom Analysis Types
├── Caching Integration
│   ├── Prompt-based Caching
│   ├── Semantic Caching
│   └── Response Optimization
└── Performance Monitoring
    ├── Request Tracking
    ├── Cost Monitoring
    └── Usage Analytics
```

## 📚 Core Classes

### DataAnalyst

**Main orchestrator class for AI-powered data analysis**

```python
class DataAnalyst:
    """
    AI-powered data analyst with multi-provider support
    
    Features:
    - Multi-provider AI integration with automatic failover
    - Advanced caching for performance optimization
    - Comprehensive analysis types for different use cases
    - Interactive Q&A capabilities
    - Cost tracking and usage monitoring
    """
    
    def __init__(self, 
                 primary_provider: str,
                 fallback_providers: List[str] = None,
                 request_timeout: int = 30,
                 max_retries: int = 3,
                 enable_cache: bool = True,
                 cache_ttl: int = 3600,
                 **provider_configs):
        """
        Initialize DataAnalyst with provider configuration
        
        Args:
            primary_provider: Primary AI provider ("openai", "openrouter", "nvidia")
            fallback_providers: List of fallback providers for redundancy
            request_timeout: Maximum time to wait for AI response (seconds)
            max_retries: Number of retry attempts on failure
            enable_cache: Enable response caching for performance
            cache_ttl: Cache time-to-live in seconds
            **provider_configs: Provider-specific configuration dictionaries
        
        Example:
            analyst = DataAnalyst(
                primary_provider="openai",
                fallback_providers=["openrouter", "nvidia"],
                request_timeout=45,
                max_retries=5,
                openai_api_key="sk-...",
                openai_model="gpt-4o",
                openrouter_api_key="sk-or-v1-...",
                nvidia_api_key="nvapi-..."
            )
        """
```

### AnalysisType Enumeration

**Defines the types of analysis that can be performed**

```python
class AnalysisType(Enum):
    """
    Available analysis types for dataset examination
    """
    
    OVERVIEW = "overview"
    """
    General dataset analysis and summary
    - Dataset structure and schema
    - Basic statistics and patterns
    - Data type distribution
    - Initial insights and observations
    """
    
    QUALITY = "quality" 
    """
    Comprehensive data quality assessment
    - Missing value analysis
    - Data consistency checks
    - Outlier detection
    - Quality score calculation
    - Improvement recommendations
    """
    
    INSIGHTS = "insights"
    """
    Deep insights and pattern detection
    - Trend analysis and patterns
    - Correlation discovery
    - Anomaly identification
    - Predictive indicators
    - Business insights
    """
    
    RELATIONSHIPS = "relationships"
    """
    Inter-dataset relationship analysis
    - Common fields identification
    - Join possibility assessment
    - Data lineage mapping
    - Dependency analysis
    - Integration recommendations
    """
    
    CUSTOM = "custom"
    """
    User-defined custom analysis
    - Custom prompt-based analysis
    - Specialized domain analysis
    - Ad-hoc investigation
    - Targeted deep-dives
    """
```

## 🔧 Core Methods

### analyze_dataset()

**Primary method for comprehensive dataset analysis**

```python
async def analyze_dataset(self,
                        dataset_info: Dict[str, Any],
                        sample_data: Optional[List[Dict]] = None,
                        analysis_type: AnalysisType = AnalysisType.OVERVIEW,
                        custom_prompt: Optional[str] = None,
                        include_recommendations: bool = True,
                        max_sample_size: int = 100) -> AIResponse:
    """
    Perform comprehensive AI-powered dataset analysis
    
    Args:
        dataset_info: Dataset metadata including:
            - id: Unique dataset identifier
            - name: Human-readable dataset name
            - description: Dataset description
            - source: Data source information
            - schema: Column information (optional)
            - size: Dataset size metrics (optional)
        
        sample_data: Sample records from the dataset
            - Limited to max_sample_size records for performance
            - Should be representative of the full dataset
            - Can be None for metadata-only analysis
        
        analysis_type: Type of analysis to perform
            - OVERVIEW: General analysis and summary
            - QUALITY: Data quality assessment
            - INSIGHTS: Deep pattern analysis
            - RELATIONSHIPS: Inter-dataset analysis
            - CUSTOM: Custom prompt-based analysis
        
        custom_prompt: Custom analysis instructions (for CUSTOM type)
            - Specific analysis requirements
            - Domain-specific instructions
            - Custom output format requests
        
        include_recommendations: Include actionable recommendations
        
        max_sample_size: Maximum number of sample records to analyze
    
    Returns:
        AIResponse: Comprehensive analysis results including:
            - content: Detailed analysis text
            - model: AI model used for analysis
            - provider: AI provider used
            - tokens_used: Token consumption count
            - cost_estimate: Estimated cost of the request
            - processing_time: Time taken for analysis
            - cached: Whether result was from cache
    
    Raises:
        ProviderError: When all AI providers fail
        ValidationError: When input data is invalid
        TimeoutError: When analysis exceeds timeout
    
    Example:
        dataset_info = {
            "id": "nyc-311-2023",
            "name": "NYC 311 Service Requests 2023",
            "description": "All 311 service requests for NYC in 2023",
            "source": "NYC Open Data Portal"
        }
        
        sample_data = [
            {"complaint_type": "Noise", "borough": "Manhattan", "status": "Open"},
            {"complaint_type": "Heat", "borough": "Brooklyn", "status": "Closed"}
        ]
        
        response = await analyst.analyze_dataset(
            dataset_info=dataset_info,
            sample_data=sample_data,
            analysis_type=AnalysisType.QUALITY,
            include_recommendations=True
        )
        
        print(f"Analysis: {response.content}")
        print(f"Quality Score: {response.metadata.get('quality_score')}")
    """
```

### answer_question()

**Interactive Q&A about datasets using natural language**

```python
async def answer_question(self,
                        question: str,
                        dataset_info: Dict[str, Any],
                        context: Optional[Dict] = None,
                        conversation_history: Optional[List[Dict]] = None,
                        use_cache: bool = True) -> AIResponse:
    """
    Answer natural language questions about datasets
    
    Args:
        question: Natural language question about the dataset
            Examples:
            - "What are the most common complaint types?"
            - "How has data quality changed over time?"
            - "What patterns can you identify in this data?"
            - "What recommendations do you have for data improvement?"
        
        dataset_info: Dataset metadata (same format as analyze_dataset)
        
        context: Additional context for answering
            - sample_data: Relevant data samples
            - previous_analysis: Prior analysis results
            - user_preferences: User-specific context
            - domain_knowledge: Domain-specific information
        
        conversation_history: Previous questions and answers
            - Enables context-aware conversations
            - Maintains conversation flow
            - References previous insights
        
        use_cache: Whether to use cached responses for performance
    
    Returns:
        AIResponse: Natural language answer with supporting details
    
    Example:
        # Basic question
        response = await analyst.answer_question(
            question="What is the data quality of this dataset?",
            dataset_info=dataset_info,
            context={"sample_data": sample_data}
        )
        
        # Follow-up question with conversation history
        follow_up = await analyst.answer_question(
            question="How can we improve the quality issues you mentioned?",
            dataset_info=dataset_info,
            conversation_history=[
                {"question": "What is the data quality?", "answer": response.content}
            ]
        )
    """
```

### generate_insights()

**Generate AI-powered insights for datasets**

```python
async def generate_insights(self,
                          data_context: Dict[str, Any],
                          insight_type: str = "comprehensive",
                          focus_areas: Optional[List[str]] = None,
                          priority_level: str = "medium") -> List[Dict[str, Any]]:
    """
    Generate structured insights about data patterns and opportunities
    
    Args:
        data_context: Complete data context including:
            - dataset_info: Dataset metadata
            - sample_data: Representative data samples
            - analysis_history: Previous analysis results
            - usage_patterns: How the data is being used
            - quality_metrics: Data quality assessments
        
        insight_type: Type of insights to generate
            - "comprehensive": Full analysis across all areas
            - "quality": Focus on data quality insights
            - "usage": Focus on usage patterns and optimization
            - "trends": Focus on trend analysis and patterns
            - "opportunities": Focus on improvement opportunities
        
        focus_areas: Specific areas to emphasize
            - ["completeness", "accuracy", "consistency"]
            - ["performance", "accessibility", "integration"]
            - ["trends", "patterns", "anomalies"]
        
        priority_level: Minimum priority level for insights
            - "low": Include all insights
            - "medium": Medium and high priority only
            - "high": High priority insights only
    
    Returns:
        List[Dict]: Structured insights with:
            - title: Insight headline
            - description: Detailed insight description
            - priority: Priority level (low/medium/high)
            - type: Insight category
            - evidence: Supporting data points
            - recommendations: Actionable recommendations
            - confidence: Confidence score (0-1)
    
    Example:
        insights = await analyst.generate_insights(
            data_context={
                "dataset_info": dataset_info,
                "sample_data": sample_data,
                "quality_metrics": {"completeness": 0.85, "accuracy": 0.92}
            },
            insight_type="comprehensive",
            focus_areas=["quality", "patterns"],
            priority_level="medium"
        )
        
        for insight in insights:
            print(f"📊 {insight['title']} ({insight['priority']})")
            print(f"   {insight['description']}")
            print(f"   Confidence: {insight['confidence']:.1%}")
    """
```

## 🎛️ Provider Management

### Multi-Provider Architecture

**Robust provider management with automatic failover**

```python
class ProviderManager:
    """
    Manages multiple AI providers with health monitoring and failover
    """
    
    def __init__(self, primary_provider: str, fallback_providers: List[str]):
        self.primary = primary_provider
        self.fallbacks = fallback_providers
        self.provider_health = {}
        self.provider_stats = {}
    
    async def execute_request(self, request: AIRequest) -> AIResponse:
        """
        Execute AI request with automatic provider failover
        
        Process:
        1. Try primary provider first
        2. If primary fails, try fallback providers in order
        3. Track provider health and performance
        4. Implement circuit breaker pattern for failed providers
        5. Return response from first successful provider
        """
        
        providers_to_try = [self.primary] + self.fallbacks
        
        for provider_name in providers_to_try:
            if not self._is_provider_healthy(provider_name):
                continue
                
            try:
                provider = self._get_provider(provider_name)
                response = await provider.generate_response(request)
                
                # Track successful usage
                self._update_provider_stats(provider_name, success=True)
                return response
                
            except Exception as e:
                # Track failure and try next provider
                self._update_provider_stats(provider_name, success=False, error=e)
                continue
        
        raise ProviderError("All AI providers failed")
```

### Provider Health Monitoring

**Continuous monitoring of provider availability and performance**

```python
async def monitor_provider_health(self):
    """
    Continuously monitor AI provider health and performance
    
    Metrics Tracked:
    - Response time (average, p95, p99)
    - Success rate (last 100 requests)
    - Error types and frequency
    - Cost per request
    - Model performance
    
    Health Scoring:
    - Green (0.8-1.0): Fully operational
    - Yellow (0.5-0.8): Degraded performance
    - Red (0.0-0.5): Unhealthy, exclude from rotation
    """
    
    for provider_name, provider in self.providers.items():
        try:
            # Test basic connectivity
            start_time = time.time()
            test_response = await provider.test_connection()
            response_time = time.time() - start_time
            
            # Calculate health score
            health_score = self._calculate_health_score(
                provider_name, 
                response_time, 
                test_response
            )
            
            self.provider_health[provider_name] = {
                "score": health_score,
                "last_check": datetime.now(),
                "response_time": response_time,
                "status": "healthy" if health_score > 0.8 else "degraded",
                "available": test_response is not None
            }
            
        except Exception as e:
            self.provider_health[provider_name] = {
                "score": 0.0,
                "last_check": datetime.now(),
                "status": "unhealthy",
                "error": str(e),
                "available": False
            }
```

## 💾 Caching Integration

### Intelligent Caching Strategy

**Multi-tier caching for optimal performance**

```python
class CacheIntegration:
    """
    Advanced caching integration for AI analysis requests
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.cache_strategies = {
            AnalysisType.OVERVIEW: {"ttl": 3600, "semantic": True},
            AnalysisType.QUALITY: {"ttl": 1800, "semantic": True},
            AnalysisType.INSIGHTS: {"ttl": 7200, "semantic": False},
            AnalysisType.RELATIONSHIPS: {"ttl": 14400, "semantic": True}
        }
    
    async def get_cached_analysis(self, 
                                request_key: str, 
                                analysis_type: AnalysisType) -> Optional[AIResponse]:
        """
        Retrieve cached analysis with type-specific strategies
        
        Cache Strategy by Analysis Type:
        - OVERVIEW: Long TTL, semantic matching enabled
        - QUALITY: Medium TTL, semantic matching for similar datasets
        - INSIGHTS: Extended TTL, exact matching only
        - RELATIONSHIPS: Very long TTL, semantic matching enabled
        """
        
        strategy = self.cache_strategies[analysis_type]
        
        # Try exact match first
        cached_response = await self.cache_manager.get(
            key=request_key,
            use_semantic=False
        )
        
        if cached_response:
            return cached_response
        
        # Try semantic match if enabled
        if strategy["semantic"]:
            cached_response = await self.cache_manager.get(
                key=request_key,
                use_semantic=True
            )
        
        return cached_response
    
    async def cache_analysis(self, 
                           request_key: str, 
                           response: AIResponse, 
                           analysis_type: AnalysisType):
        """
        Cache analysis response with type-specific TTL
        """
        strategy = self.cache_strategies[analysis_type]
        
        await self.cache_manager.set(
            key=request_key,
            value=response,
            ttl=strategy["ttl"]
        )
```

## 📊 Performance Monitoring

### Usage Analytics

**Comprehensive tracking of AI usage and performance**

```python
class UsageAnalytics:
    """
    Track and analyze AI analyst usage patterns and performance
    """
    
    def __init__(self):
        self.usage_stats = {
            "total_requests": 0,
            "requests_by_type": defaultdict(int),
            "requests_by_provider": defaultdict(int),
            "cache_hit_rate": 0.0,
            "average_response_time": 0.0,
            "total_cost": 0.0,
            "error_rate": 0.0
        }
        self.request_history = []
    
    def track_request(self, 
                     request_type: str, 
                     provider: str, 
                     response_time: float,
                     tokens_used: int,
                     cost: float,
                     cached: bool,
                     success: bool):
        """
        Track individual request metrics
        """
        
        self.usage_stats["total_requests"] += 1
        self.usage_stats["requests_by_type"][request_type] += 1
        self.usage_stats["requests_by_provider"][provider] += 1
        self.usage_stats["total_cost"] += cost
        
        # Update rolling averages
        self._update_rolling_average("response_time", response_time)
        self._update_cache_hit_rate(cached)
        self._update_error_rate(success)
        
        # Store detailed history
        self.request_history.append({
            "timestamp": datetime.now(),
            "type": request_type,
            "provider": provider,
            "response_time": response_time,
            "tokens_used": tokens_used,
            "cost": cost,
            "cached": cached,
            "success": success
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        """
        
        return {
            "summary": self.usage_stats,
            "top_analysis_types": self._get_top_analysis_types(),
            "provider_performance": self._get_provider_performance(),
            "cost_breakdown": self._get_cost_breakdown(),
            "performance_trends": self._get_performance_trends(),
            "recommendations": self._get_optimization_recommendations()
        }
```

## 🔍 Error Handling

### Comprehensive Error Management

**Robust error handling with detailed diagnostics**

```python
class AIAnalystError(Exception):
    """Base exception for AI analyst errors"""
    pass

class ProviderError(AIAnalystError):
    """Error related to AI provider failures"""
    
    def __init__(self, message: str, provider: str, error_code: str = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.timestamp = datetime.now()

class ValidationError(AIAnalystError):
    """Error related to input validation"""
    
    def __init__(self, message: str, field: str, value: Any):
        super().__init__(message)
        self.field = field
        self.value = value

class TimeoutError(AIAnalystError):
    """Error related to request timeouts"""
    
    def __init__(self, message: str, timeout_duration: int):
        super().__init__(message)
        self.timeout_duration = timeout_duration

async def handle_analysis_error(self, error: Exception, context: Dict[str, Any]) -> AIResponse:
    """
    Centralized error handling with recovery strategies
    
    Recovery Strategies:
    1. Provider failover for provider errors
    2. Request simplification for complexity errors
    3. Retry with exponential backoff for temporary errors
    4. Graceful degradation for timeout errors
    """
    
    if isinstance(error, ProviderError):
        # Try fallback providers
        return await self._try_fallback_providers(context)
    
    elif isinstance(error, TimeoutError):
        # Simplify request and retry
        simplified_context = self._simplify_request(context)
        return await self._retry_with_simplified_request(simplified_context)
    
    elif isinstance(error, ValidationError):
        # Fix validation issues and retry
        fixed_context = self._fix_validation_issues(context, error)
        return await self.analyze_dataset(**fixed_context)
    
    else:
        # Log error and return graceful failure response
        logger.error(f"Unhandled error in AI analysis: {error}")
        return self._create_error_response(error, context)
```

## 🚀 Usage Examples

### Basic Analysis

```python
from AI_Functionality.core.ai_analyst import DataAnalyst, AnalysisType

# Initialize analyst
analyst = DataAnalyst(
    primary_provider="openai",
    fallback_providers=["openrouter", "nvidia"],
    openai_api_key="sk-...",
    openrouter_api_key="sk-or-v1-...",
    nvidia_api_key="nvapi-..."
)

# Dataset information
dataset_info = {
    "id": "nyc-taxi-2023",
    "name": "NYC Taxi Trip Data 2023",
    "description": "Yellow taxi trip records for NYC in 2023",
    "source": "NYC TLC",
    "schema": ["pickup_datetime", "dropoff_datetime", "pickup_location", "dropoff_location", "fare_amount"]
}

# Sample data
sample_data = [
    {"pickup_datetime": "2023-01-01 08:30:00", "pickup_location": "Manhattan", "fare_amount": 12.50},
    {"pickup_datetime": "2023-01-01 09:15:00", "pickup_location": "Brooklyn", "fare_amount": 8.75}
]

# Perform overview analysis
response = await analyst.analyze_dataset(
    dataset_info=dataset_info,
    sample_data=sample_data,
    analysis_type=AnalysisType.OVERVIEW
)

print(f"Analysis: {response.content}")
print(f"Model: {response.model}")
print(f"Provider: {response.provider}")
print(f"Cached: {response.cached}")
```

### Quality Assessment

```python
# Comprehensive data quality analysis
quality_response = await analyst.analyze_dataset(
    dataset_info=dataset_info,
    sample_data=sample_data,
    analysis_type=AnalysisType.QUALITY,
    include_recommendations=True
)

print("🔍 Data Quality Assessment:")
print(quality_response.content)

# Extract quality metrics if available
if "quality_score" in quality_response.metadata:
    score = quality_response.metadata["quality_score"]
    print(f"📊 Overall Quality Score: {score}/100")
```

### Interactive Q&A

```python
# Ask specific questions about the dataset
questions = [
    "What are the peak hours for taxi trips?",
    "What is the average fare amount?",
    "Are there any data quality issues I should be aware of?",
    "What patterns do you see in pickup locations?"
]

conversation_history = []

for question in questions:
    response = await analyst.answer_question(
        question=question,
        dataset_info=dataset_info,
        context={"sample_data": sample_data},
        conversation_history=conversation_history
    )
    
    print(f"❓ {question}")
    print(f"🤖 {response.content}")
    print("---")
    
    # Add to conversation history
    conversation_history.append({
        "question": question,
        "answer": response.content,
        "timestamp": datetime.now()
    })
```

### Advanced Insights Generation

```python
# Generate comprehensive insights
insights = await analyst.generate_insights(
    data_context={
        "dataset_info": dataset_info,
        "sample_data": sample_data,
        "quality_metrics": {"completeness": 0.95, "accuracy": 0.88},
        "usage_patterns": {"daily_queries": 150, "popular_fields": ["pickup_location", "fare_amount"]}
    },
    insight_type="comprehensive",
    focus_areas=["trends", "quality", "opportunities"],
    priority_level="medium"
)

print("💡 Generated Insights:")
for insight in insights:
    print(f"🔍 {insight['title']} ({insight['priority']})")
    print(f"   {insight['description']}")
    print(f"   Confidence: {insight['confidence']:.1%}")
    if insight['recommendations']:
        print("   📋 Recommendations:")
        for rec in insight['recommendations']:
            print(f"      • {rec}")
    print()
```

### Performance Monitoring

```python
# Get performance statistics
stats = analyst.get_usage_stats()
print("📈 AI Analyst Performance:")
print(f"Total Requests: {stats['total_requests']}")
print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
print(f"Average Response Time: {stats['average_response_time']:.2f}s")
print(f"Total Cost: ${stats['total_cost']:.2f}")

# Get provider status
provider_status = analyst.get_provider_status()
print("\n🔧 Provider Status:")
for provider, status in provider_status.items():
    print(f"{provider}: {status['status']} (Score: {status['health_score']:.2f})")
```

## ⚡ Performance Optimization Tips

1. **Efficient Sampling**: Limit sample data to 50-100 records for optimal performance
2. **Cache Usage**: Enable caching for repeated similar queries
3. **Provider Selection**: Use appropriate models for task complexity
4. **Batch Processing**: Process multiple datasets concurrently when possible
5. **Timeout Management**: Set appropriate timeouts based on analysis complexity

This completes the comprehensive documentation for the AI Analyst core component.