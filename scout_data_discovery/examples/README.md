# Examples Documentation

## Overview

The `examples/` directory provides comprehensive, real-world demonstrations of the Scout Data Discovery package capabilities. These examples serve as both learning resources and integration templates for different use cases and complexity levels.

## Example Files Structure

```
examples/
├── README.md                           # This documentation
├── basic_usage.py                     # Core Scout functionality examples
├── enhanced_api_examples.py           # Advanced API client demonstrations
└── multi_dataset_workflow_examples.py # Complete multi-dataset workflows
```

---

## Example Files Details

### 1. basic_usage.py
**Getting Started with Scout Data Discovery**

**Purpose**: Demonstrates fundamental Scout functionality with clear, easy-to-follow examples for newcomers.

**Key Examples**:

**Basic Dataset Search**
```python
def example_basic_search():
    scout = ScoutDataDiscovery(log_level="INFO")
    search_terms = ["transportation", "traffic"]
    datasets_df = scout.search_datasets(search_terms, limit=10)
```

**Quality Assessment**
```python
def example_quality_assessment():
    # Comprehensive 5-dimensional quality analysis
    quality_report = scout.assess_dataset_quality("dataset-id")
    # Returns: completeness, consistency, accuracy, timeliness, usability scores
```

**Configuration Management**
```python
def example_custom_configuration():
    # Custom configuration with user-defined settings
    config = ConfigManager('custom_config.yaml')
    scout = ScoutDataDiscovery(config=config.to_dict())
```

**Advanced Search with Filters**
```python
def example_advanced_search():
    # Multi-criteria search with quality filtering
    datasets_df = scout.search_datasets_advanced(
        keywords=["environment", "air quality"],
        min_quality_score=70,
        include_quality_assessment=True
    )
```

**Batch Quality Assessment**
```python
def example_batch_assessment():
    # Parallel quality assessment of multiple datasets
    dataset_ids = ["abc-123", "def-456", "ghi-789"]
    results = scout.batch_assess_quality(dataset_ids, max_workers=3)
```

**Features Demonstrated**:
- Basic Scout initialization and configuration
- Dataset discovery and ranking
- Quality assessment workflow
- Error handling and logging
- Configuration customization
- Results export and interpretation

**Target Audience**: Beginners, data analysts starting with Scout methodology

---

### 2. enhanced_api_examples.py
**Advanced API Integration and SoQL Querying**

**Purpose**: Comprehensive demonstrations of the Enhanced NYC Open Data API Client with advanced SoQL capabilities and performance optimizations.

**Key Examples**:

**Enhanced Client Initialization**
```python
def example_basic_enhanced_client():
    client = EnhancedNYCDataClient(
        app_token="your_token",
        default_backend="pandas",  # or "polars" for performance
        rate_limit_delay=0.5
    )
```

**Advanced SoQL Query Building**
```python
def example_advanced_soql_queries():
    query = (client.query()
             .select("date", "complaint_type", "borough", "latitude", "longitude")
             .where_date_range("created_date", start_date, end_date)
             .where_in("borough", ["MANHATTAN", "BROOKLYN"])
             .where_like("complaint_type", "noise")
             .where_numeric_range("latitude", min_val=40.5, max_val=40.9)
             .order_by("created_date", ascending=False)
             .limit(10000))

    data = client.get_dataset("erm2-nwe9", query)
```

**High-Performance Processing**
```python
def example_polars_backend():
    # Using Polars for high-performance data processing
    client = EnhancedNYCDataClient(default_backend="polars")
    data = client.get_dataset("large-dataset", limit=100000)
    # Returns Polars DataFrame for fast operations
```

**Streaming for Large Datasets**
```python
def example_streaming_processing():
    total_processed = 0
    for chunk in client.get_dataset_streaming("very-large-dataset", chunk_size=5000):
        # Process each chunk individually
        processed_chunk = process_data_chunk(chunk)
        total_processed += len(chunk)
```

**Batch Dataset Processing**
```python
def example_batch_processing():
    dataset_ids = ["dataset1", "dataset2", "dataset3"]
    results = client.batch_download_datasets(
        dataset_ids,
        query_configs={"dataset1": query1, "dataset2": query2},
        max_workers=3
    )
```

**Scout Integration**
```python
def example_scout_integrated_client():
    # Client with built-in Scout quality assessment
    integrated_client = ScoutIntegratedClient(app_token="your_token")

    data_with_quality = integrated_client.get_dataset_with_quality("dataset-id")
    recommendations = integrated_client.get_recommendations("source-dataset")
```

**Intelligent Caching**
```python
def example_caching_strategies():
    # Response caching with TTL
    cached_data = client.get_dataset_cached("dataset-id", ttl_hours=24)

    # Cache invalidation and management
    client.clear_cache("dataset-id")
    client.clear_all_cache()
```

**Features Demonstrated**:
- Dual backend support (pandas/Polars)
- Complex SoQL query construction
- Streaming and batch processing
- Intelligent caching strategies
- Rate limiting and error recovery
- Scout methodology integration
- Performance optimization techniques

**Target Audience**: Advanced users, data engineers, performance-conscious applications

---

### 3. multi_dataset_workflow_examples.py
**Complete Multi-Dataset Discovery and Integration**

**Purpose**: End-to-end workflows demonstrating the complete Scout methodology for multi-dataset discovery, relationship mapping, and unified querying.

**Key Examples**:

**Complete Multi-Dataset Workflow**
```python
def example_complete_workflow():
    # Step 1: Initialize workflow orchestrator
    orchestrator = MultiDatasetOrchestrator(
        scout_client=scout,
        enhanced_client=enhanced_client
    )

    # Step 2: Execute complete workflow
    results = orchestrator.execute_complete_workflow(
        source_dataset_id="erm2-nwe9",
        max_related_datasets=5,
        date_range=("2024-01-01", "2024-12-31")
    )
```

**Column Relationship Analysis**
```python
def example_column_analysis():
    # Analyze column relationships across datasets
    analyzer = ColumnAnalyzer()
    mapper = RelationshipMapper()

    # Get comprehensive column metadata
    column_metadata = analyzer.analyze_column(column_data, "dataset-id", "dataset_name")

    # Find relationships between datasets
    relationships = mapper.find_relationships(source_columns, target_columns)
```

**Unified Query Generation**
```python
def example_unified_query_generation():
    # Generate JSON-serializable multi-dataset queries
    workflow = MultiDatasetWorkflow(scout, enhanced_client)

    unified_query = workflow.generate_unified_query(
        source_dataset_id="primary-dataset",
        related_datasets=["related1", "related2"],
        selected_columns={
            "primary-dataset": ["col1", "col2"],
            "related1": ["col3", "col4"],
            "related2": ["col5", "col6"]
        }
    )

    # Save query for reuse
    query_json = unified_query.to_json()
```

**Query Execution and Optimization**
```python
def example_query_execution():
    # Execute unified queries with optimization
    executor = UnifiedQueryExecutor(enhanced_client)

    # Load and execute saved query
    unified_query = UnifiedQuery.from_json(query_json)
    results = executor.execute(unified_query)

    print(f"Integrated dataset: {len(results.data)} rows")
    print(f"Execution time: {results.execution_time:.2f}s")
    print(f"Join success rate: {results.join_success_rate:.1%}")
```

**Workflow Monitoring and Callbacks**
```python
def example_workflow_monitoring():
    def progress_callback(step_name, progress, message):
        print(f"[{step_name}] {progress:.1%}: {message}")

    # Execute with real-time monitoring
    results = orchestrator.execute_complete_workflow(
        source_dataset_id="source-id",
        progress_callback=progress_callback,
        export_results=True,
        export_formats=["csv", "json", "parquet"]
    )
```

**Relationship Graph Analysis**
```python
def example_relationship_graph():
    # Build and analyze dataset relationship networks
    graph = RelationshipGraph()

    # Add datasets and relationships
    for dataset in related_datasets:
        graph.add_dataset(dataset)

    for relationship in column_relationships:
        graph.add_relationship(relationship)

    # Analyze graph structure
    strongest_relationships = graph.get_strongest_relationships(top_k=10)
    relationship_clusters = graph.find_clusters()
```

**Advanced Workflow Configuration**
```python
def example_advanced_configuration():
    # Comprehensive workflow configuration
    config = WorkflowConfig(
        max_related_datasets=10,
        min_relationship_confidence=0.6,
        enable_parallel_processing=True,
        quality_threshold=75,
        export_intermediate_results=True,
        custom_filters={
            "borough": ["MANHATTAN", "BROOKLYN"],
            "date_range": ["2024-01-01", "2024-12-31"]
        }
    )

    results = orchestrator.execute_workflow(config)
```

**Features Demonstrated**:
- Complete end-to-end multi-dataset workflows
- Intelligent column relationship mapping (15+ data types, 9 relationship types)
- JSON-serializable unified query objects
- Query optimization and execution planning
- Workflow orchestration with monitoring
- Relationship graph analysis and visualization
- Advanced configuration and customization
- Result export and archiving

**Target Audience**: Data scientists, analysts working with complex multi-dataset scenarios, enterprise users

---

## Running the Examples

### Prerequisites
```bash
# Basic requirements
pip install pandas numpy requests pyyaml

# Optional high-performance backend
pip install polars

# Development and testing
pip install pytest jupyter
```

### Environment Setup
```bash
# Set API token (optional but recommended)
export NYC_OPEN_DATA_APP_TOKEN="your_app_token_here"

# Set configuration
export SCOUT_API_RATE_LIMIT_DELAY=0.5
export SCOUT_DATA_QUALITY_THRESHOLD=70
```

### Execution Examples
```bash
# Basic usage examples
python examples/basic_usage.py

# Enhanced API examples
python examples/enhanced_api_examples.py

# Complete multi-dataset workflows
python examples/multi_dataset_workflow_examples.py
```

### Interactive Usage
```python
# In Jupyter notebook or Python REPL
import sys
sys.path.append('/path/to/scout_data_discovery')

from examples.basic_usage import example_basic_search
results = example_basic_search()
```

---

## Example Use Cases

### 1. **Data Discovery & Exploration**
- Find relevant datasets using Scout methodology
- Assess data quality before analysis
- Explore dataset relationships and compatibility

### 2. **Multi-Dataset Analysis**
- Combine related datasets for comprehensive analysis
- Perform cross-dataset joins and aggregations
- Generate unified queries for complex data needs

### 3. **Production Workflows**
- Automate data discovery and quality assessment
- Build data pipelines with Scout intelligence
- Monitor and optimize multi-dataset operations

### 4. **Performance Optimization**
- Leverage Polars backend for large-scale processing
- Use streaming for memory-efficient operations
- Implement intelligent caching strategies

---

## Best Practices Demonstrated

### Configuration Management
- Environment-specific configurations
- Runtime parameter overrides
- Validation and error handling

### Error Handling
- Graceful degradation strategies
- Comprehensive error logging
- Recovery and retry mechanisms

### Performance Optimization
- Backend selection based on use case
- Memory-efficient processing patterns
- Parallel and batch operations

### Code Organization
- Modular example structure
- Clear documentation and comments
- Reusable code patterns

These examples provide comprehensive coverage of all Scout Data Discovery capabilities, from basic functionality to advanced multi-dataset workflows, serving as both learning resources and production implementation templates.