# Scout Data Discovery - Source Code Documentation

## Overview

The `src/` directory contains the core implementation of the Scout Data Discovery system. This directory houses all the primary engines, processors, and orchestrators that make up the intelligence of the system.

## Module Architecture

### Core Workflow Engine
- **scout_discovery.py** - Main orchestrator integrating all components
- **workflow_orchestrator.py** - End-to-end workflow coordination and monitoring

### Data Processing Engines
- **data_quality.py** - Multi-dimensional quality assessment
- **enhanced_api_client.py** - Advanced Socrata API client with SoQL
- **column_relationship_mapper.py** - Column analysis and relationship detection

### Query & Execution System
- **multi_dataset_workflow.py** - Multi-dataset discovery and query generation
- **unified_query_executor.py** - Optimized query execution engine

### Foundation
- **exceptions.py** - Custom exception hierarchy for structured error handling

---

## Module Details

### 1. scout_discovery.py
**Primary Scout Data Discovery Orchestrator**

**Purpose**: Main entry point that integrates all system components into a cohesive workflow.

**Key Classes**:
- `ScoutDataDiscovery`: Main workflow orchestrator with enhanced API integration

**Key Features**:
- Dataset search with Scout methodology
- Quality assessment integration
- Enhanced API client support with pandas/Polars backends
- Advanced search with multiple filters
- Streaming support for large datasets
- Batch processing capabilities
- Comprehensive result export

**Usage Pattern**:
```python
scout = ScoutDataDiscovery(
    use_enhanced_client=True,
    app_token="your_token",
    default_backend="pandas"
)

# Basic workflow
datasets = scout.search_datasets(["transportation"])
assessment = scout.assess_dataset_quality("dataset-id")

# Advanced workflow with streaming
for chunk in scout.download_dataset_streaming("large-dataset"):
    process_chunk(chunk)
```

**Integration Points**:
- Enhanced API Client for data access
- Data Quality Assessor for quality metrics
- Configuration Manager for settings
- All other modules for complete functionality

---

### 2. data_quality.py
**Multi-Dimensional Data Quality Assessment Engine**

**Purpose**: Comprehensive quality assessment following Scout methodology with 5-dimensional scoring.

**Key Classes**:
- `DataQualityAssessor`: Main assessment engine with pluggable metrics

**Quality Dimensions**:
1. **Completeness (25%)**: Missing data assessment and coverage analysis
2. **Consistency (20%)**: Data type consistency and format standardization
3. **Accuracy (20%)**: Outlier detection and validation using statistical methods
4. **Timeliness (15%)**: Data freshness and update frequency analysis
5. **Usability (20%)**: Dataset accessibility, structure clarity, and naming conventions

**Key Features**:
- Automated outlier detection using IQR method
- Missing data pattern analysis with recommendations
- Data type optimization suggestions
- Column naming quality assessment
- Statistical validation and range checking
- Comprehensive scoring with letter grades (A-F)

**Assessment Output**:
```python
{
    'overall_scores': {
        'total_score': 85.5,
        'grade': 'B',
        'completeness_score': 90,
        'consistency_score': 80,
        # ... other dimensions
    },
    'completeness': {
        'missing_percentage': 5.2,
        'complete_columns': ['id', 'date'],
        'empty_columns': []
    },
    # ... detailed metrics for each dimension
}
```

**Advanced Capabilities**:
- Batch assessment of multiple datasets
- Quality trend analysis over time
- Custom scoring weights for domain-specific requirements
- Integration with workflow systems for automated quality gates

---

### 3. enhanced_api_client.py
**Advanced Socrata API Client with SoQL Support**

**Purpose**: Production-ready API client with advanced querying, caching, and dual backend support.

**Key Classes**:
- `EnhancedNYCDataClient`: Core API client with advanced features
- `ScoutIntegratedClient`: Scout-integrated client with quality assessment
- `SoQLQueryBuilder`: Fluent API for complex Socrata queries

**Advanced Features**:
- **Dual Backend Support**: pandas (default) and Polars (high-performance)
- **Advanced SoQL Queries**: Complex filtering, joins, aggregations, and date ranges
- **Intelligent Caching**: Response caching with TTL and invalidation
- **Streaming Support**: Memory-efficient processing of large datasets
- **Batch Processing**: Parallel downloads of multiple datasets
- **Rate Limiting**: Configurable delays with exponential backoff
- **Error Recovery**: Retry logic with graceful degradation

**SoQL Query Builder**:
```python
query = (client.query()
         .select("date", "type", "borough", "count")
         .where_date_range("date", start_date, end_date)
         .where_in("borough", ["MANHATTAN", "BROOKLYN"])
         .where_like("description", "noise")
         .where_numeric_range("count", min_val=1, max_val=100)
         .order_by("date", ascending=False)
         .limit(5000))

data = client.get_dataset("dataset-id", query)
```

**Performance Optimizations**:
- Connection pooling and session reuse
- Compressed response handling
- Parallel request processing
- Memory-mapped file caching
- Query result optimization

**Integration Features**:
- Scout quality assessment integration
- Automatic dataset recommendations
- Cross-dataset relationship detection
- Workflow orchestration support

---

### 4. column_relationship_mapper.py
**Intelligent Column Analysis and Relationship Detection**

**Purpose**: Advanced column analysis and cross-dataset relationship mapping using semantic understanding.

**Key Classes**:
- `ColumnAnalyzer`: Deep column analysis with 15+ data type detection
- `RelationshipMapper`: Cross-dataset relationship discovery and scoring
- `ColumnMetadata`: Comprehensive column representation
- `ColumnRelationship`: Detailed relationship information

**Column Type Detection**:
- **15+ Data Types**: text, numeric, date, datetime, categorical, identifier, location, coordinate, address, phone, email, URL, JSON, boolean
- **Pattern Recognition**: Automatic pattern detection for formats and structures
- **Semantic Tagging**: Contextual understanding of column meanings
- **Quality Metrics**: Completeness, uniqueness, and distribution analysis

**Relationship Types**:
1. **Exact Match**: Identical column names and types
2. **Semantic Match**: Similar meaning with different names (address/location)
3. **Type Compatible**: Same data types with different semantics
4. **Hierarchical**: Parent-child relationships (borough/address, category/subcategory)
5. **Temporal**: Time-based relationships (year/date, created/updated)
6. **Geographic**: Location-based relationships (lat/lng, zip/borough)
7. **Reference**: Foreign key potential (ID relationships)
8. **Derived**: One column can be calculated from another
9. **Complementary**: Columns that together provide complete information

**Scoring System**:
- **Confidence Score**: Overall relationship strength (0.0-1.0)
- **Compatibility Score**: Data type and semantic compatibility
- **Join Potential**: Likelihood of successful joins
- **Semantic Similarity**: Meaning-based similarity using NLP techniques

**Analysis Output**:
```python
{
    'source_column': 'borough',
    'target_column': 'borough_name',
    'relationship_type': 'semantic_match',
    'confidence_score': 0.87,
    'join_potential': 0.92,
    'notes': 'Semantic match with high join potential'
}
```

---

### 5. multi_dataset_workflow.py
**Multi-Dataset Discovery and Query Generation**

**Purpose**: Orchestrates the discovery of related datasets and generates unified query objects for multi-table operations.

**Key Classes**:
- `MultiDatasetWorkflow`: Main workflow engine for multi-dataset operations
- `DatasetSchema`: Complete dataset schema representation
- `RelationshipGraph`: Network graph of dataset relationships
- `UnifiedQuery`: JSON-serializable multi-dataset query object

**Workflow Process**:
1. **Schema Analysis**: Deep analysis of source dataset structure
2. **Related Dataset Discovery**: Scout-powered search for compatible datasets
3. **Relationship Mapping**: Column-level relationship analysis across datasets
4. **Query Generation**: Intelligent join strategy selection and query building
5. **Optimization**: Query optimization based on dataset characteristics

**Relationship Graph**:
- Network representation of dataset relationships
- Confidence scoring for each relationship
- Join potential assessment
- Relationship type categorization
- Graph traversal for complex multi-dataset queries

**Unified Query Object**:
```json
{
  "query_id": "uuid-string",
  "primary_dataset": "source-dataset-id",
  "datasets": ["dataset1", "dataset2", "dataset3"],
  "joins": [
    {
      "left_dataset": "dataset1",
      "right_dataset": "dataset2",
      "left_column": "borough",
      "right_column": "borough",
      "join_type": "inner",
      "confidence": 0.85
    }
  ],
  "selected_columns": {
    "dataset1": ["col1", "col2"],
    "dataset2": ["col3", "col4"]
  },
  "filters": {
    "dataset1": {
      "conditions": ["status = 'active'"],
      "date_column": "created_date"
    }
  },
  "date_range": ["2024-01-01", "2024-12-31"],
  "limit": 10000
}
```

**Advanced Features**:
- Smart column selection based on relationship strength
- Automatic filter generation based on data patterns
- Join strategy optimization (inner, left, outer)
- Query serialization for reusability and sharing
- Integration with workflow orchestration systems

---

### 6. unified_query_executor.py
**Optimized Multi-Dataset Query Execution Engine**

**Purpose**: Executes unified queries with intelligent optimization, join handling, and result merging.

**Key Classes**:
- `UnifiedQueryExecutor`: Main execution engine with optimization
- `QueryOptimizer`: Intelligent execution planning and cost estimation
- `ExecutionPlan`: Optimized execution strategy
- `ExecutionResults`: Comprehensive execution results with metadata

**Execution Optimization**:
- **Cost-Based Optimization**: Dataset size estimation and join ordering
- **Parallel Execution**: Concurrent dataset fetching and processing
- **Memory Management**: Streaming and chunking for large datasets
- **Join Optimization**: Smart join strategy selection based on data characteristics
- **Error Recovery**: Graceful handling of failed operations with fallback strategies

**Execution Process**:
1. **Query Analysis**: Parse and validate unified query
2. **Execution Planning**: Generate optimized execution plan with cost estimation
3. **Data Fetching**: Parallel retrieval of individual datasets with filtering
4. **Join Execution**: Intelligent join processing with error handling
5. **Result Merging**: Combine results into final integrated dataset
6. **Quality Assessment**: Validate join success and data integrity

**Performance Features**:
- Execution plan caching for repeated queries
- Join result caching with TTL
- Memory usage monitoring and optimization
- Parallel processing with configurable worker pools
- Progress tracking and cancellation support

**Monitoring & Analytics**:
- Detailed execution statistics and timing
- Join success rate tracking
- Memory usage profiling
- Performance bottleneck identification
- Historical execution analytics

---

### 7. workflow_orchestrator.py
**Complete End-to-End Workflow Coordination**

**Purpose**: Orchestrates complete multi-dataset workflows from discovery to final results with monitoring and callbacks.

**Key Classes**:
- `MultiDatasetOrchestrator`: Main workflow coordinator
- `WorkflowConfig`: Comprehensive workflow configuration
- `WorkflowResults`: Complete workflow execution results
- `WorkflowStep`: Individual step tracking and monitoring

**Workflow Capabilities**:
- **Complete Automation**: End-to-end workflow execution with minimal configuration
- **Step-by-Step Control**: Granular control over each workflow phase
- **Progress Monitoring**: Real-time progress tracking with callback system
- **Error Recovery**: Intelligent error handling with recovery strategies
- **Result Management**: Automatic export and archiving of results

**Workflow Steps**:
1. **Discovery**: Related dataset discovery using Scout intelligence
2. **Analysis**: Relationship graph generation and analysis
3. **Query Generation**: Unified query creation with optimization
4. **Execution**: Multi-dataset query execution with monitoring
5. **Export**: Result export and archiving in multiple formats

**Monitoring System**:
- Step-level progress tracking
- Execution time monitoring
- Success/failure rate tracking
- Callback system for custom monitoring
- Comprehensive execution statistics

**Configuration Management**:
- Environment-specific configurations
- Runtime parameter overrides
- Performance tuning options
- Integration with external monitoring systems

---

### 8. exceptions.py
**Structured Error Handling System**

**Purpose**: Comprehensive exception hierarchy for structured error handling across all system components.

**Exception Hierarchy**:
```
ScoutDiscoveryError (base)
├── APIError (API-related failures)
│   ├── status_code
│   └── response_text
├── DataQualityError (quality assessment failures)
├── DataDownloadError (dataset download failures)
│   └── dataset_id
├── SearchError (search operation failures)
├── ConfigurationError (configuration issues)
└── ValidationError (data validation failures)
```

**Error Handling Features**:
- Structured error information with context
- Error recovery suggestions
- Logging integration with appropriate levels
- Error aggregation for batch operations
- User-friendly error messages with technical details

**Integration**:
- Used consistently across all modules
- Integrated with logging systems
- Supports error reporting and monitoring
- Enables graceful degradation strategies

---

## Module Dependencies

```
scout_discovery.py
├── enhanced_api_client.py
├── data_quality.py
├── workflow_orchestrator.py
└── exceptions.py

workflow_orchestrator.py
├── multi_dataset_workflow.py
├── unified_query_executor.py
└── column_relationship_mapper.py

multi_dataset_workflow.py
├── column_relationship_mapper.py
├── enhanced_api_client.py
└── scout_discovery.py

unified_query_executor.py
├── multi_dataset_workflow.py
├── enhanced_api_client.py
└── exceptions.py

enhanced_api_client.py
└── exceptions.py

data_quality.py
└── exceptions.py

column_relationship_mapper.py
└── exceptions.py
```

## Performance Characteristics

### Memory Usage
- **Streaming Support**: Handles datasets larger than available memory
- **Configurable Sampling**: Adjustable sample sizes for analysis
- **Efficient Caching**: LRU caching with TTL and size limits

### Execution Speed
- **Parallel Processing**: Multi-threaded operations where applicable
- **Optimized Joins**: Intelligent join algorithm selection
- **Query Optimization**: Cost-based execution planning

### Scalability
- **Horizontal Scaling**: Supports distributed processing architectures
- **Batch Processing**: Efficient handling of multiple datasets
- **Resource Management**: Configurable resource limits and monitoring

This source code architecture provides a robust, maintainable, and scalable foundation for advanced data discovery and integration workflows.