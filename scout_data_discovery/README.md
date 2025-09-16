# Scout Data Discovery

A comprehensive Python package for automated data discovery, quality assessment, and curation based on Scout (https://scout.tsdataclinic.com/) methodology.

## Features

🔍 **Intelligent Dataset Discovery**
- Search across multiple Socrata-powered open data portals
- Semantic search with filtering and categorization
- Automated metadata extraction and normalization

📊 **Comprehensive Quality Assessment**
- Multi-dimensional quality scoring (completeness, consistency, accuracy, timeliness, usability)
- Automated outlier detection and data validation
- Detailed quality reports with actionable insights

🎯 **Smart Recommendations**
- ML-powered dataset similarity matching
- Content-based filtering using tags, categories, and metadata
- Cross-dataset relationship discovery

🌐 **Dataset Relationship Graphs** (NEW)
- Visual network analysis of dataset relationships
- Interactive graph visualizations with Plotly
- Cluster analysis and similarity metrics
- Cross-domain relationship discovery

⚡ **High-Performance Processing**
- Parallel processing for large-scale assessments
- Intelligent caching for improved performance
- Configurable rate limiting and retry logic

📁 **Flexible Export Options**
- Multiple export formats (CSV, JSON, Parquet)
- Comprehensive reporting with visualizations
- Integration-ready API responses

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd scout_data_discovery

# Install dependencies
pip install -r requirements.txt

# For interactive visualizations (recommended):
pip install plotly

# For advanced graph analysis (optional):
pip install python-igraph
```

### Basic Usage

```python
from scout_data_discovery import ScoutDataDiscovery

# Initialize the discovery engine
scout = ScoutDataDiscovery(log_level="INFO")

# Search for datasets
datasets = scout.search_datasets(["transportation", "traffic"])
print(f"Found {len(datasets)} datasets")

# Assess quality of a specific dataset
dataset_id = datasets.iloc[0]['id']
assessment = scout.assess_dataset_quality(dataset_id)
print(f"Quality score: {assessment['overall_scores']['total_score']:.1f}/100")

# Generate recommendations
recommendations = scout.generate_recommendations(dataset_id, datasets)
print(f"Generated {len(recommendations)} recommendations")

# NEW: Create relationship graph
from src.dataset_relationship_graph import DatasetRelationshipGraph

relationship_graph = DatasetRelationshipGraph()
relationship_graph.add_datasets(datasets)
stats = relationship_graph.calculate_relationships()
print(f"Found {stats['relationships_found']} dataset relationships")
```

### Complete Pipeline

```python
# Run comprehensive discovery pipeline
results = scout.run_discovery_pipeline(
    search_terms=["housing", "health", "education"],
    max_assessments=20,
    include_recommendations=True,
    export_results=True
)

# Access results
print(f"Pipeline found {results['pipeline_stats']['datasets_found']} datasets")
print(f"Average quality score: {results['pipeline_stats']['quality_statistics']['mean_score']:.1f}")
```

## Architecture

### Core Components

```
scout_data_discovery/
├── src/
│   ├── scout_discovery.py              # Main workflow class
│   ├── data_quality.py                 # Quality assessment engine
│   ├── enhanced_api_client.py          # Enhanced NYC Data API client
│   ├── dataset_relationship_graph.py   # NEW: Relationship graph analysis
│   ├── multi_dataset_workflow.py       # Multi-dataset operations
│   ├── workflow_orchestrator.py        # Workflow orchestration
│   └── exceptions.py                   # Custom exceptions
├── config/
│   ├── default_config.yaml             # Default configuration
│   └── config_manager.py               # Configuration management
├── examples/
│   ├── basic_usage.py                  # Basic usage examples
│   ├── multi_dataset_workflow_examples.py # Multi-dataset workflows
│   └── relationship_graph_examples.py  # NEW: Relationship graph examples
└── tests/
    └── test_scout_discovery.py         # Test suite
```

### Key Classes

- **`ScoutDataDiscovery`**: Main workflow orchestrator
- **`DataQualityAssessor`**: Comprehensive quality assessment
- **`DatasetRelationshipGraph`**: NEW: Network analysis and visualization of dataset relationships
- **`EnhancedNYCDataClient`**: Advanced NYC Open Data API client with SoQL support
- **`ConfigManager`**: Configuration management with YAML/JSON support

## Configuration

### Default Configuration

The package comes with sensible defaults that can be customized:

```yaml
# API Settings
api:
  rate_limit_delay: 0.5
  request_timeout: 30
  retry_attempts: 3

# Quality Assessment
quality:
  weights:
    completeness: 0.25
    consistency: 0.20
    accuracy: 0.20
    timeliness: 0.15
    usability: 0.20
```

### Custom Configuration

```python
from scout_data_discovery.config import ConfigManager

# Load custom configuration
config = ConfigManager('my_config.yaml')

# Initialize with custom config
scout = ScoutDataDiscovery(config=config.to_dict())
```

### Environment Variables

Override configuration using environment variables:

```bash
export SCOUT_API_RATE_LIMIT_DELAY=1.0
export SCOUT_DATA_QUALITY_THRESHOLD=80
export SCOUT_CACHE_DURATION_HOURS=48
```

## Quality Assessment

### Quality Dimensions

1. **Completeness (25%)**: Missing data assessment
   - Missing value percentages
   - Empty columns identification
   - Data coverage analysis

2. **Consistency (20%)**: Data type and format consistency
   - Type validation and recommendations
   - Format standardization assessment
   - Column naming consistency

3. **Accuracy (20%)**: Data accuracy and outlier detection
   - Statistical outlier identification
   - Range validation
   - Logical consistency checks

4. **Timeliness (15%)**: Data freshness and update frequency
   - Last update analysis
   - Update frequency assessment
   - Data staleness scoring

5. **Usability (20%)**: Dataset accessibility and structure
   - Column naming quality
   - Dataset size appropriateness
   - Structure clarity assessment

### Quality Scores

- **Grade A (90-100)**: Excellent quality, ready for analysis
- **Grade B (80-89)**: Good quality, minor issues
- **Grade C (70-79)**: Acceptable quality, some attention needed
- **Grade D (60-69)**: Poor quality, significant issues
- **Grade F (<60)**: Very poor quality, major problems

## Dataset Relationship Graphs (NEW)

The Dataset Relationship Graph module provides powerful network analysis capabilities to discover and visualize relationships between datasets based on content similarity, structural patterns, and metadata connections.

### Key Features

- **Multi-dimensional Similarity Analysis**: Analyzes content, structure, metadata, tags, and categories
- **Network Visualization**: Create static and interactive network graphs
- **Cluster Detection**: Automatically identify groups of related datasets
- **Cross-domain Analysis**: Discover unexpected relationships across different data domains
- **Comprehensive Reporting**: Generate detailed relationship analysis reports

### Basic Usage

```python
from src.dataset_relationship_graph import DatasetRelationshipGraph

# Initialize the relationship graph
relationship_graph = DatasetRelationshipGraph()

# Add datasets from a discovery search
datasets_df = scout.search_datasets(["transportation", "housing", "health"])
relationship_graph.add_datasets(datasets_df)

# Calculate relationships with custom weights
stats = relationship_graph.calculate_relationships(
    content_weight=0.3,      # Text similarity weight
    structural_weight=0.25,  # Column similarity weight
    metadata_weight=0.2,     # Usage patterns weight
    tag_weight=0.15,         # Tag overlap weight
    category_weight=0.1,     # Category matching weight
    similarity_threshold=0.3 # Minimum similarity for connections
)

print(f"Found {stats['relationships_found']} relationships")
print(f"Graph density: {stats['graph_density']:.3f}")
```

### Find Related Datasets

```python
# Get datasets most similar to a target dataset
related = relationship_graph.get_related_datasets(
    dataset_id="target-dataset-id",
    top_n=5,
    min_similarity=0.4
)

for dataset in related:
    print(f"{dataset['name']}: {dataset['similarity_score']:.3f}")
    print(f"Reasons: {', '.join(dataset['relationship_reasons'])}")
```

### Create Visualizations

```python
# Static network visualization
relationship_graph.visualize_graph(
    output_path="network_graph.png",
    layout="spring",
    node_size_by="download_count",
    show_labels=True
)

# Interactive visualization (requires plotly)
relationship_graph.create_interactive_graph(
    output_path="interactive_network.html",
    height=800
)
```

### Generate Analysis Reports

```python
# Comprehensive relationship report
report = relationship_graph.generate_relationship_report(
    output_path="relationship_report.txt"
)

# Export graph data for further analysis
relationship_graph.export_graph_data(
    "graph_data.json",
    format="json"
)
```

## Advanced Features

### Parallel Processing

```python
# Configure parallel processing
scout = ScoutDataDiscovery(
    max_workers=10,  # Concurrent workers
    config={
        'performance': {
            'enable_parallel_processing': True,
            'chunk_size_for_large_datasets': 50000
        }
    }
)
```

### Caching

```python
# Configure caching
scout = ScoutDataDiscovery(
    cache_dir='./my_cache',
    config={
        'cache': {
            'duration_hours': 48,
            'enable_file_cache': True
        }
    }
)

# Clear cache when needed
scout.clear_cache()
```

### Custom Quality Weights

```python
# Customize quality assessment weights
custom_config = {
    'quality': {
        'weights': {
            'completeness': 0.4,  # Prioritize completeness
            'consistency': 0.3,
            'accuracy': 0.2,
            'timeliness': 0.05,
            'usability': 0.05
        }
    }
}

scout = ScoutDataDiscovery(config=custom_config)
```

## API Reference

### ScoutDataDiscovery

#### Methods

- `search_datasets(terms, domains=None, limit=None)`: Search for datasets
- `download_dataset_sample(dataset_id, sample_size=None)`: Download dataset sample
- `assess_dataset_quality(dataset_id, df=None)`: Assess dataset quality
- `generate_recommendations(dataset_id, catalog_df)`: Generate recommendations
- `run_discovery_pipeline(...)`: Execute complete workflow
- `export_results(results, output_dir=None)`: Export results to files

### DataQualityAssessor

#### Methods

- `assess_dataset_quality(dataset_id, df, metadata=None)`: Comprehensive assessment
- `generate_quality_report(assessments)`: Generate summary report

### ConfigManager

#### Methods

- `get(key, default=None)`: Get configuration value
- `set(key, value)`: Set configuration value
- `get_section(section)`: Get configuration section
- `save_user_config(filepath)`: Save configuration to file

## Error Handling

The package includes comprehensive error handling with custom exceptions:

```python
from scout_data_discovery.src.exceptions import (
    ScoutDiscoveryError,  # Base exception
    APIError,             # API-related errors
    DataDownloadError,    # Dataset download errors
    SearchError,          # Search-related errors
    ConfigurationError    # Configuration errors
)

try:
    results = scout.search_datasets("test")
except SearchError as e:
    print(f"Search failed: {e}")
except APIError as e:
    print(f"API error (status {e.status_code}): {e}")
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test class
python -m pytest tests/test_scout_discovery.py::TestScoutDataDiscovery -v
```

## Examples

### Example 1: Basic Dataset Search

```python
from scout_data_discovery import ScoutDataDiscovery

scout = ScoutDataDiscovery()
datasets = scout.search_datasets("housing")

print(f"Found {len(datasets)} housing datasets")
for _, dataset in datasets.head().iterrows():
    print(f"- {dataset['name']} (Downloads: {dataset['download_count']})")
```

### Example 2: Quality Assessment Pipeline

```python
# Assess multiple datasets
dataset_ids = ['h9gi-nx95', 'erm2-nwe9']  # NYC datasets
assessments = {}

for dataset_id in dataset_ids:
    try:
        assessment = scout.assess_dataset_quality(dataset_id)
        assessments[dataset_id] = assessment
        score = assessment['overall_scores']['total_score']
        print(f"{dataset_id}: Quality score {score:.1f}/100")
    except Exception as e:
        print(f"{dataset_id}: Assessment failed - {e}")
```

### Example 3: Custom Configuration

```python
# Create performance-optimized configuration
performance_config = {
    'api': {
        'rate_limit_delay': 0.1,  # Faster requests
        'max_concurrent_requests': 20
    },
    'cache': {
        'duration_hours': 72,  # Longer caching
        'enable_file_cache': True
    },
    'data': {
        'quality_threshold': 85,  # Higher quality threshold
        'default_sample_size': 5000  # Larger samples
    }
}

scout = ScoutDataDiscovery(config=performance_config)
```

### Example 4: Dataset Relationship Analysis

```python
from src.dataset_relationship_graph import DatasetRelationshipGraph

# Search for datasets across multiple domains
scout = ScoutDataDiscovery()
datasets = scout.search_datasets(["housing", "transportation", "health"], limit=20)

# Create relationship graph
graph = DatasetRelationshipGraph()
graph.add_datasets(datasets)

# Calculate relationships
stats = graph.calculate_relationships(similarity_threshold=0.25)
print(f"Found {stats['relationships_found']} relationships")

# Find datasets related to the first one
target_id = datasets.iloc[0]['id']
related = graph.get_related_datasets(target_id, top_n=5)

for rel in related:
    print(f"Related: {rel['name']}")
    print(f"Similarity: {rel['similarity_score']:.3f}")
    print(f"Reasons: {', '.join(rel['relationship_reasons'])}")

# Create visualizations
graph.visualize_graph("network.png", show_labels=True)
graph.create_interactive_graph("network.html")

# Generate comprehensive report
report = graph.generate_relationship_report("analysis_report.txt")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
python -m pytest tests/ -v

# Format code
black src/ tests/ examples/
```

## Performance Considerations

### Memory Usage

- Use sampling for large datasets (`default_sample_size`)
- Enable file caching for repeated access
- Configure memory limits in performance settings

### Rate Limiting

- Default: 0.5 seconds between requests
- Configurable per API endpoint
- Exponential backoff for retries

### Parallel Processing

- Automatic parallel quality assessment
- Configurable worker pool size
- Memory-aware processing for large datasets

## Troubleshooting

### Common Issues

1. **API Rate Limiting**
   ```python
   # Increase delay between requests
   config = {'api': {'rate_limit_delay': 2.0}}
   ```

2. **Memory Issues**
   ```python
   # Reduce sample sizes
   config = {'data': {'default_sample_size': 500}}
   ```

3. **Slow Performance**
   ```python
   # Enable caching and parallel processing
   config = {
       'cache': {'enable_file_cache': True},
       'performance': {'enable_parallel_processing': True}
   }
   ```

### Logging

Enable detailed logging for debugging:

```python
scout = ScoutDataDiscovery(log_level="DEBUG")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on the methodology pioneered by [Scout](https://scout.tsdataclinic.com/)
- Utilizes the [Socrata Discovery API](https://dev.socrata.com/foundry/api.us.socrata.com/catalog)
- Inspired by The Data Clinic's work on automated data discovery

---

For more information, examples, and advanced usage patterns, see the `/examples` directory.